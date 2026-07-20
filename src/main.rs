//! Lore CLI: standalone AI agent infrastructure — library + service + terminal.
//!
//! Subcommands share a persistent data directory (SQLite memory + persona files);
//! so an agent created from the terminal is also reachable via `serve` over the network.

use clap::{Parser, Subcommand};
use lore::{
    Agent, AgentId, AnthropicAuth, AnthropicModel, AppState, CalcTool, CodexModel, Credential,
    FileReadTool, HashingEmbedder, InMemoryStore, KeywordRouter, Memory, MemoryGraph, MemoryStore,
    MessageKind, MockModel, Model, OpenAiModel, Orchestrator, Party, Persona, PersonaPatch, Query,
    RefreshingToken, Scope, SemanticCat, SqliteStore, TimeTool, TokenStore, ToolContext,
    ToolRegistry, WebFetchTool,
};
use std::sync::Arc;

/// Rate-limit fixed window (seconds) — `LORE_RATE_LIMIT` is requests per minute.
const RATE_WINDOW_SECS: u64 = 60;

#[derive(Parser)]
#[command(
    name = "lore",
    version,
    about = "Standalone AI agent infrastructure (identity + memory + orchestration)"
)]
struct Cli {
    /// Persistent data directory (identities + memory).
    #[arg(long, global = true, env = "LORE_DATA", default_value = "lore-data")]
    data: String,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP service.
    Serve {
        /// Address to listen on.
        #[arg(long, env = "LORE_SERVE", default_value = "127.0.0.1:3777")]
        addr: String,
    },
    /// Create a new agent.
    NewAgent {
        #[arg(long)]
        name: String,
        #[arg(long)]
        role: String,
        /// Comma-separated character traits.
        #[arg(long, value_delimiter = ',')]
        traits: Vec<String>,
    },
    /// List agents.
    List,
    /// Ask an agent.
    Ask {
        id: String,
        message: String,
        /// Session name (meaningful in service mode; same session remembers history).
        #[arg(long)]
        session: Option<String>,
    },
    /// Interactive chat with agent (multi-turn — conversation history enabled).
    Chat { id: String },
    /// Make an agent do something (runs if a tool matches, otherwise replies).
    Act { id: String, input: String },
    /// Multi-step tool loop (ReAct): the model chains tools to solve the task.
    Solve {
        id: String,
        input: String,
        /// Step limit (default 5, cap 10).
        #[arg(long, default_value_t = 5)]
        steps: usize,
    },
    /// Inter-agent message (ask waits for reply, tell provides information).
    Message {
        to: String,
        content: String,
        #[arg(long)]
        from: Option<String>,
        #[arg(long, default_value = "tell")]
        kind: String,
    },
    /// Add an episodic memory to the agent.
    Remember {
        id: String,
        #[arg(long)]
        title: String,
        #[arg(long)]
        body: String,
    },
    /// Reflection: distill frequently recalled episodic memories into the semantic tier.
    Reflect {
        /// Agent identity.
        id: String,
    },
    /// Reinforce a memory record (decay/Wilson feedback: accessed|success|failure).
    Reinforce {
        /// Agent identity.
        id: String,
        /// Memory record identity (the id from recall output).
        memory: String,
        /// Outcome: accessed | success | failure.
        outcome: String,
    },
    /// Recall from the agent's memory.
    Recall {
        id: String,
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Semantic recall (captures morphology/synonyms).
        #[arg(long)]
        semantic: bool,
    },
    /// Update persona (version increments — identity evolution).
    Update {
        id: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        role: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long, value_delimiter = ',')]
        traits: Option<Vec<String>>,
        #[arg(long)]
        system_prompt: Option<String>,
    },
    /// Delete an agent.
    Delete { id: String },
    /// Collective reasoning: ask the whole team, collect replies (writes to the board).
    Deliberate {
        question: String,
        /// If provided, this agent does not participate in the poll but synthesizes replies (hierarchical team).
        #[arg(long)]
        synthesizer: Option<String>,
    },
    /// Read the shared board.
    Board {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Introductory demo (identity + orchestration + memory).
    Demo,
    /// Log in to a provider with a subscription (OAuth): `anthropic` or `openai`.
    Login {
        /// Provider name (`anthropic` | `openai`).
        provider: String,
        /// Anthropic only: paste-the-code flow instead of the browser loopback
        /// flow (SSH/headless friendly).
        #[arg(long)]
        device: bool,
    },
    /// Remove a stored credential for a provider.
    Logout {
        /// Provider name (`anthropic` | `openai`).
        provider: String,
    },
    /// Show configured provider credentials and their status.
    Auth,
    /// Export memory as JSON (live records; backup/migration).
    Export {
        /// Output file (defaults to stdout).
        #[arg(long)]
        out: Option<String>,
    },
    /// Import memory from a JSON dump (ids are preserved, existing ones are updated).
    Import { file: String },
    /// Trigger memory maintenance manually (near-duplicate merge + decay report).
    Consolidate,
    /// Recompute embeddings for all records with the active embedder
    /// (run after changing embedder — e.g. hashing → neural).
    Reembed,
}

/// Optional response token cap (`LORE_LLM_MAX_TOKENS`).
fn env_max_tokens() -> Option<u32> {
    match std::env::var("LORE_LLM_MAX_TOKENS") {
        Ok(mt) => match mt.parse::<u32>() {
            Ok(n) if n > 0 => Some(n),
            _ => {
                tracing::warn!(value = %mt, "LORE_LLM_MAX_TOKENS invalid, ignored");
                None
            }
        },
        Err(_) => None,
    }
}

/// Optional request timeout in seconds (`LORE_LLM_TIMEOUT`).
fn env_timeout() -> Option<std::time::Duration> {
    match std::env::var("LORE_LLM_TIMEOUT") {
        Ok(to) => match to.parse::<u64>() {
            Ok(n) if n > 0 => Some(std::time::Duration::from_secs(n)),
            _ => {
                tracing::warn!(value = %to, "LORE_LLM_TIMEOUT invalid, ignored");
                None
            }
        },
        Err(_) => None,
    }
}

/// Refresh closure for Anthropic subscription tokens.
fn anthropic_refresh_fn() -> lore::auth::RefreshFn {
    Box::new(|rt: String| Box::pin(async move { lore::auth::refresh_anthropic(&rt).await }))
}

/// Refresh closure for OpenAI (Codex) subscription tokens.
fn openai_refresh_fn() -> lore::auth::RefreshFn {
    Box::new(|rt: String| Box::pin(async move { lore::auth::refresh_openai(&rt).await }))
}

/// Builds an OpenAI provider: subscription (Codex Responses) or metered API key
/// (Chat Completions via `OpenAiModel`).
fn build_openai(data: &str) -> Arc<dyn Model> {
    let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "gpt-5".into());
    let store = TokenStore::new(data);
    let stored = store.load("openai").ok().flatten();
    let mode = std::env::var("LORE_AUTH").ok();
    let want_key = mode.as_deref() == Some("key");
    let want_subs = mode.as_deref() == Some("subs");
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("LORE_LLM_KEY"))
        .ok()
        .filter(|k| !k.trim().is_empty());

    // Subscription (Codex) path.
    if !want_key {
        if let Some(cred @ Credential::OAuth { account_id, .. }) = &stored {
            let account_id = account_id.clone();
            let refreshing =
                RefreshingToken::new(store, "openai", cred.clone(), openai_refresh_fn());
            let mut m = CodexModel::new(name, Arc::new(refreshing), account_id);
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            return Arc::new(m);
        }
        if want_subs {
            tracing::warn!(
                "LORE_AUTH=subs but no OpenAI subscription credential; run `lore login openai`"
            );
        }
    }
    // Metered API-key path (official Chat Completions).
    let key = api_key.or_else(|| match &stored {
        Some(Credential::ApiKey { key }) => Some(key.clone()),
        _ => None,
    });
    match key {
        Some(k) => {
            let mut m = OpenAiModel::new("https://api.openai.com/v1", name).with_api_key(k);
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        None => {
            tracing::warn!(
                "LORE_PROVIDER=openai but no credential found \
                 (run `lore login openai` or set OPENAI_API_KEY); using MockModel"
            );
            Arc::new(MockModel::new())
        }
    }
}

/// Resolves Anthropic auth: `LORE_AUTH=key|subs` (default: subs if a stored
/// OAuth credential exists, else an API key from `ANTHROPIC_API_KEY`/`LORE_LLM_KEY`).
fn resolve_anthropic_auth(data: &str) -> Option<AnthropicAuth> {
    let store = TokenStore::new(data);
    let stored = store.load("anthropic").ok().flatten();
    let mode = std::env::var("LORE_AUTH").ok();
    let want_key = mode.as_deref() == Some("key");
    let want_subs = mode.as_deref() == Some("subs");
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .or_else(|_| std::env::var("LORE_LLM_KEY"))
        .ok()
        .filter(|k| !k.trim().is_empty());

    // Explicit API-key mode, or a stored api-key credential.
    if want_key {
        return api_key.map(AnthropicAuth::ApiKey);
    }
    if let Some(Credential::ApiKey { key }) = &stored {
        if !want_subs {
            return Some(AnthropicAuth::ApiKey(key.clone()));
        }
    }
    // Subscription (OAuth) from the token store.
    if let Some(cred @ Credential::OAuth { .. }) = stored {
        let refreshing = RefreshingToken::new(store, "anthropic", cred, anthropic_refresh_fn());
        return Some(AnthropicAuth::OAuth(Arc::new(refreshing)));
    }
    // Fall back to an API key from the environment.
    api_key.map(AnthropicAuth::ApiKey)
}

/// Builds an Anthropic model (subscription or API key).
fn build_anthropic(data: &str) -> Arc<dyn Model> {
    let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".into());
    match resolve_anthropic_auth(data) {
        Some(auth) => {
            let mut m = AnthropicModel::new(name, auth);
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        None => {
            tracing::warn!(
                "LORE_PROVIDER=anthropic but no credential found \
                 (run `lore login anthropic` or set ANTHROPIC_API_KEY); using MockModel"
            );
            Arc::new(MockModel::new())
        }
    }
}

/// Sets up the model. `LORE_PROVIDER=anthropic` selects the Anthropic provider
/// (subscription/API key); otherwise `LORE_LLM_BASE` uses the OpenAI-compatible
/// path (incl. Ollama); with neither set, `MockModel`.
fn build_model(data: &str) -> Arc<dyn Model> {
    match std::env::var("LORE_PROVIDER").ok().as_deref() {
        Some("anthropic") => return build_anthropic(data),
        Some("openai") => return build_openai(data),
        _ => {}
    }
    match std::env::var("LORE_LLM_BASE") {
        Ok(base) => {
            let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
            let mut m = OpenAiModel::new(base, name);
            if let Ok(key) = std::env::var("LORE_LLM_KEY") {
                m = m.with_api_key(key);
            }
            // Optional response token limit. Low values on reasoning models may
            // spend the budget on thinking — use deliberately.
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            // Optional request timeout (seconds). Slow local models (e.g. 14B+
            // on CPU) may exceed the default 120 s — can be increased.
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        Err(_) => Arc::new(MockModel::new()),
    }
}

/// Sets up the embedder: `LORE_EMBEDDER=neural` + `neural` feature → fastembed;
/// otherwise native `HashingEmbedder` (fully offline).
fn build_embedder() -> Arc<dyn lore::Embedder> {
    #[cfg(feature = "neural")]
    if std::env::var("LORE_EMBEDDER").as_deref() == Ok("neural") {
        match lore::NeuralEmbedder::new() {
            Ok(e) => {
                println!("🧠 Neural embedder active (multilingual-e5-small)");
                return Arc::new(e);
            }
            Err(e) => {
                tracing::warn!(error = %e, "neural embedder failed to initialize; falling back to native")
            }
        }
    }
    Arc::new(HashingEmbedder::new())
}

/// Sets up persistent application state (SQLite memory + persona directory).
/// Validates `LORE_API_KEY`: empty/whitespace keys are REJECTED.
/// Otherwise `LORE_API_KEY=""` would leave auth as an open door — an empty `x-api-key:`
/// header would pass via `ct_eq("", "")` (operator would assume auth is off).
fn parse_api_key(raw: Option<String>) -> Option<String> {
    let key = raw?.trim().to_string();
    if key.is_empty() {
        tracing::warn!(
            "LORE_API_KEY empty — ignoring (auth stays OFF; empty key does not authenticate)"
        );
        None
    } else {
        Some(key)
    }
}

fn build_state(data: &str) -> anyhow::Result<AppState> {
    std::fs::create_dir_all(data)?;
    let store: Arc<dyn MemoryStore> =
        Arc::new(SqliteStore::open(&format!("{data}/lore.db"))?.with_embedder(build_embedder()));
    // Platform tools: every agent can use these in `act`.
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(CalcTool::new()));
    reg.register(Arc::new(TimeTool::new()));
    // SSRF protection is on by default; to reach your internal services,
    // set LORE_WEB_ALLOW_PRIVATE=1 (deliberate opt-in).
    let web =
        WebFetchTool::new().with_private_allowed(std::env::var("LORE_WEB_ALLOW_PRIVATE").is_ok());
    reg.register(Arc::new(web));
    // File tool: sandboxed under the data directory.
    reg.register(Arc::new(FileReadTool::new(format!("{data}/files"))));
    let tools = ToolContext {
        registry: reg,
        router: Arc::new(
            KeywordRouter::new()
                .on("calculate", "calc")
                .on("time", "time")
                .on("date", "time")
                .on("download", "web")
                .on("web", "web")
                .on("read", "file")
                .on("file", "file"),
        ),
    };
    let mut app =
        AppState::persistent(format!("{data}/agents"), store, build_model(data))?.with_tools(tools);
    // Security: if LORE_API_KEY is set, auth is mandatory; LORE_RATE_LIMIT caps requests per minute.
    if let Some(key) = parse_api_key(std::env::var("LORE_API_KEY").ok()) {
        app = app.with_api_key(key);
    }
    // Invalid values must not silently fall back to "unlimited" — warn if the security setting was intentional.
    if let Ok(raw) = std::env::var("LORE_RATE_LIMIT") {
        match raw.parse::<u32>() {
            Ok(max) if max > 0 => app = app.with_rate_limit(max, RATE_WINDOW_SECS),
            _ => tracing::warn!(value = %raw, "LORE_RATE_LIMIT invalid, rate limit NOT APPLIED"),
        }
    }
    // Agent cap (DoS brake): configured via LORE_MAX_AGENTS.
    if let Ok(raw) = std::env::var("LORE_MAX_AGENTS") {
        match raw.parse::<usize>() {
            Ok(max) if max > 0 => app = app.with_max_agents(max),
            _ => tracing::warn!(value = %raw, "LORE_MAX_AGENTS invalid, keeping default"),
        }
    }
    // Federation: LORE_PEERS is a comma-separated list of base URLs (+ optional LORE_PEER_KEY).
    if let Ok(peers) = std::env::var("LORE_PEERS") {
        // Normalization (trim, trailing '/', dedupe) is handled inside with_peers.
        let list: Vec<String> = peers.split(',').map(str::to_string).collect();
        if !list.is_empty() {
            app = app.with_peers(list, std::env::var("LORE_PEER_KEY").ok());
        }
    }
    Ok(app)
}

/// Sets up structured logging. Filter: `LORE_LOG` env (e.g.
/// `LORE_LOG=debug`, `LORE_LOG=lore=trace`); default is `lore=info` — our own
/// events are visible, dependencies are silent. CLI user-facing
/// output (println) is not logging, it bypasses this filter.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_env("LORE_LOG").unwrap_or_else(|_| EnvFilter::new("lore=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    // No subcommand: if LORE_SERVE is set, run service; otherwise demo (backward compat).
    let cmd = cli
        .cmd
        .unwrap_or_else(|| match std::env::var("LORE_SERVE") {
            Ok(addr) => Cmd::Serve { addr },
            Err(_) => Cmd::Demo,
        });

    match cmd {
        Cmd::Serve { addr } => {
            let st = build_state(&cli.data)?;
            println!("🌐 Lore HTTP service: http://{addr}  (data: {}/)", cli.data);
            lore::server::serve(&addr, st).await?;
        }
        Cmd::NewAgent { name, role, traits } => {
            let st = build_state(&cli.data)?;
            let v = st.create_agent(&name, &role, traits).await?;
            println!(
                "✅ created: {}  {}  [{}]",
                v.id,
                v.name,
                v.traits.join(", ")
            );
        }
        Cmd::List => {
            let st = build_state(&cli.data)?;
            let agents = st.list_agents().await;
            if agents.is_empty() {
                println!("(no agents — create one with 'lore new-agent')");
            }
            for a in agents {
                println!(
                    "{}  {:<12} v{}  [{}]  {}",
                    a.id,
                    a.name,
                    a.version,
                    a.role,
                    a.traits.join(", ")
                );
            }
        }
        Cmd::Ask {
            id,
            message,
            session,
        } => {
            let st = build_state(&cli.data)?;
            let reply = st
                .ask_session(&AgentId::from(id), session.as_deref(), &message)
                .await?;
            println!("{reply}");
        }
        Cmd::Chat { id } => {
            let st = build_state(&cli.data)?;
            let aid = AgentId::from(id);
            // Validate agent upfront (retrieve name for the greeting).
            let name = st
                .list_agents()
                .await
                .into_iter()
                .find(|a| a.id == aid.to_string())
                .map(|a| a.name)
                .ok_or_else(|| anyhow::anyhow!("agent not found: {aid}"))?;
            println!("💬 Chat with {name} — type /q or empty line to exit");
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                use std::io::Write;
                print!("you> ");
                std::io::stdout().flush()?;
                line.clear();
                if stdin.read_line(&mut line)? == 0 {
                    break; // EOF (ctrl-d)
                }
                let msg = line.trim();
                if msg.is_empty() || msg == "/q" {
                    break;
                }
                // Real-time streaming: chunks are printed as they arrive.
                match st.ask_stream(&aid, Some("cli-chat"), msg).await {
                    Ok(mut stream) => {
                        use futures::StreamExt;
                        print!("{name}> ");
                        std::io::stdout().flush()?;
                        while let Some(chunk) = stream.next().await {
                            match chunk {
                                Ok(t) => {
                                    print!("{t}");
                                    std::io::stdout().flush()?;
                                }
                                Err(e) => {
                                    eprintln!("\n⚠️  stream interrupted: {e}");
                                    break;
                                }
                            }
                        }
                        println!("\n");
                    }
                    Err(e) => eprintln!("⚠️  {e}"),
                }
            }
            println!("👋 chat ended");
        }
        Cmd::Act { id, input } => {
            let st = build_state(&cli.data)?;
            let result = st.act(&AgentId::from(id), &input).await?;
            println!("{result}");
        }
        Cmd::Solve { id, input, steps } => {
            let st = build_state(&cli.data)?;
            let result = st.solve(&AgentId::from(id), &input, steps).await?;
            println!("{result}");
        }
        Cmd::Message {
            to,
            content,
            from,
            kind,
        } => {
            let st = build_state(&cli.data)?;
            let from_id = from.map(AgentId::from);
            // Validate the freeform string: a typo ("Ask", "tel") must not silently
            // fall back to `tell` — the server side uses an enum, CLI should be strict too.
            let is_ask = match kind.to_lowercase().as_str() {
                "ask" => true,
                "tell" => false,
                other => {
                    anyhow::bail!("invalid --kind: {other} (expected: ask | tell)");
                }
            };
            let reply = st
                .message(&AgentId::from(to), from_id.as_ref(), is_ask, &content)
                .await?;
            if reply.is_empty() {
                println!("✉️  delivered");
            } else {
                println!("{reply}");
            }
        }
        Cmd::Remember { id, title, body } => {
            let st = build_state(&cli.data)?;
            st.experience(&AgentId::from(id), &title, &body).await?;
            println!("✅ memory saved");
        }
        Cmd::Reflect { id } => {
            let st = build_state(&cli.data)?;
            let n = st.reflect(&AgentId::from(id)).await?;
            println!("✅ reflection done: {n} memories promoted to semantic tier");
        }
        Cmd::Reinforce {
            id,
            memory,
            outcome,
        } => {
            let st = build_state(&cli.data)?;
            let outcome = match outcome.to_lowercase().as_str() {
                "accessed" => lore::Outcome::Accessed,
                "success" => lore::Outcome::Success,
                "failure" => lore::Outcome::Failure,
                other => {
                    anyhow::bail!(
                        "invalid outcome: {other} (expected: accessed | success | failure)"
                    );
                }
            };
            st.reinforce(
                &AgentId::from(id),
                &lore::MemoryId::from(memory.clone()),
                outcome,
            )
            .await?;
            println!("✅ reinforced: {memory}");
        }
        Cmd::Recall {
            id,
            query,
            limit,
            semantic,
        } => {
            let st = build_state(&cli.data)?;
            for m in st
                .recall(&AgentId::from(id), &query, limit, semantic)
                .await?
            {
                // Show id so it can be reinforced via `lore reinforce`.
                println!("{}  [{:.3}] {}", m.id, m.score, m.summary);
            }
        }
        Cmd::Update {
            id,
            name,
            role,
            description,
            traits,
            system_prompt,
        } => {
            let st = build_state(&cli.data)?;
            let patch = PersonaPatch {
                name,
                role,
                description,
                traits,
                system_prompt,
            };
            let v = st.update_agent(&AgentId::from(id), patch).await?;
            println!(
                "✅ updated: {}  {}  v{}  [{}]",
                v.id,
                v.name,
                v.version,
                v.traits.join(", ")
            );
        }
        Cmd::Delete { id } => {
            let st = build_state(&cli.data)?;
            st.delete_agent(&AgentId::from(id.clone())).await?;
            println!("🗑️  deleted: {id}");
        }
        Cmd::Deliberate {
            question,
            synthesizer,
        } => {
            let st = build_state(&cli.data)?;
            let (replies, synthesis) = match synthesizer {
                Some(s) => {
                    let (r, syn) = st
                        .deliberate_synth(&question, &AgentId::from(s), false)
                        .await?;
                    (r, Some(syn))
                }
                None => (st.deliberate(&question).await?, None),
            };
            if replies.is_empty() {
                println!("(no agents — first 'lore new-agent')");
            }
            for r in replies {
                match &r.node {
                    Some(n) => println!("{}@{} → {}", r.name, n, r.reply),
                    None => println!("{} → {}", r.name, r.reply),
                }
            }
            if let Some(syn) = synthesis {
                println!("\n🧠 Synthesis → {syn}");
            }
        }
        Cmd::Board { limit } => {
            let st = build_state(&cli.data)?;
            for m in st.read_board(limit).await? {
                println!("[{:.3}] {}", m.score, m.summary);
            }
        }
        Cmd::Demo => run_demo(&cli.data).await?,
        Cmd::Login { provider, device } => login(&cli.data, &provider, device).await?,
        Cmd::Logout { provider } => {
            TokenStore::new(&cli.data).delete(&provider)?;
            println!("🚪 logged out: {provider}");
        }
        Cmd::Auth => show_auth(&cli.data)?,
        Cmd::Export { out } => {
            let store = SqliteStore::open(&format!("{}/lore.db", cli.data))?;
            let mut mems = store.export().await?;
            mems.sort_by_key(|m| m.id.to_string()); // deterministic dump (diffable)
            let json = serde_json::to_string_pretty(&mems)?;
            match out {
                Some(p) => {
                    std::fs::write(&p, json)?;
                    println!("✅ {} records → {p}", mems.len());
                }
                None => println!("{json}"),
            }
        }
        Cmd::Import { file } => {
            let store = SqliteStore::open(&format!("{}/lore.db", cli.data))?
                .with_embedder(build_embedder());
            let mems: Vec<Memory> = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
            let n = mems.len();
            for m in mems {
                store.remember(m).await?;
            }
            println!(
                "✅ {n} records imported (run `lore reembed` if they came from a different embedder)"
            );
        }
        Cmd::Consolidate => {
            let store = SqliteStore::open(&format!("{}/lore.db", cli.data))?;
            let r = store.consolidate().await?;
            println!(
                "🧹 scanned: {} · merged: {} · forgotten: {}",
                r.scanned, r.merged, r.forgotten
            );
        }
        Cmd::Reembed => {
            let store = SqliteStore::open(&format!("{}/lore.db", cli.data))?
                .with_embedder(build_embedder());
            let n = store.reembed().await?;
            println!("✅ {n} records re-embedded with active embedder");
        }
    }
    Ok(())
}

/// Introductory demo showcasing identity + orchestration + memory + tools + graph.
async fn run_demo(data: &str) -> anyhow::Result<()> {
    // Native embedder attached → recall is hybrid (keyword + cosine).
    let store: Arc<dyn MemoryStore> =
        Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));
    let model = build_model(data);
    match std::env::var("LORE_LLM_BASE") {
        Ok(base) => println!("🔌 Real model: {base}"),
        Err(_) => println!("🧪 MockModel (LORE_LLM_BASE not set)"),
    }

    // Shared world knowledge.
    store
        .remember(Memory::semantic(
            Scope::World,
            "Lore is a fully standalone AI core",
            SemanticCat::Convention,
        ))
        .await?;

    // --- Two identities ---
    let aria = Agent::new(
        Persona::new("Aria", "researcher").with_traits(["curious", "meticulous"]),
        store.clone(),
        model.clone(),
    );
    let kai = Agent::new(
        Persona::new("Kai", "engineer").with_traits(["pragmatic", "calm"]),
        store.clone(),
        model.clone(),
    );

    // --- Register with orchestrator (shared blackboard) ---
    let mut orch = Orchestrator::new().with_blackboard(Arc::new(InMemoryStore::new()));
    let aria_id = orch.register(aria);
    let kai_id = orch.register(kai);
    println!(
        "🏛️  Orchestration set up: {} agents registered\n",
        orch.len()
    );

    // Kai knows something.
    orch.agent(&kai_id)
        .unwrap()
        .experience(
            "learned the async model",
            "async tasks are spawned with tokio, messaging via mpsc",
        )
        .await?;

    // --- Aria, Kai'ye sorar ---
    println!("📨 Aria → Kai: \"how does async work\"");
    orch.ask(
        Party::Agent(aria_id.clone()),
        &kai_id,
        "how does async work",
    );
    let transcript = orch.run().await?;

    println!("\n📜 Transcript:");
    for d in &transcript {
        let to = orch.party_name(&Party::Agent(d.to.clone()));
        let from = orch.party_name(&d.from);
        let tag = match d.kind {
            MessageKind::Ask => "ASK ",
            MessageKind::Tell => "TELL",
        };
        println!("   [{tag}] {from} → {to}: \"{}\"", d.content);
        if let Some(r) = &d.reply {
            println!("          ↳ {r}");
        }
    }

    // --- Sistem herkese duyuru yapar ---
    println!("\n📢 System → everyone (broadcast): \"sprint starts tomorrow\"");
    orch.broadcast(Party::System, "sprint starts tomorrow");
    orch.run().await?;

    // --- What's on each agent's mind ---
    for (name, id) in [("Aria", &aria_id), ("Kai", &kai_id)] {
        println!("\n🗂️  {name}'s mind:");
        for r in orch
            .agent(id)
            .unwrap()
            .recall(&Query::new("").limit(4))
            .await?
        {
            println!("   [{:.3}] {}", r.score, r.item.summary());
        }
    }

    // --- Semantic recall: keyword tutmasa da morfolojiyi yakala ---
    orch.agent(&kai_id)
        .unwrap()
        .remember(Memory::semantic(
            Scope::World, // remember() pulls into Kai's scope
            "interest in mathematics",
            SemanticCat::Preference,
        ))
        .await?;
    println!("\n🔎 Kai semantic recall(\"mathematics\") — no keyword match, cosine catches it:");
    for r in orch
        .agent(&kai_id)
        .unwrap()
        .recall(&Query::new("matematik").semantic())
        .await?
    {
        println!("   [{:.3}] {}", r.score, r.item.summary());
    }

    // --- Tool usage: agent calls calculator ---
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(CalcTool::new()));
    let hesapci = Agent::new(
        Persona::new("Calculator", "assistant"),
        store.clone(),
        model.clone(),
    )
    .with_tools(reg, Arc::new(KeywordRouter::new().on("calculate", "calc")));
    println!(
        "\n🔧 Tool: Calculator 'calculate 12 * 3' → {}",
        hesapci.act("calculate 12 * 3").await?
    );

    // --- Collective reasoning: the whole team answers a system question (blackboard) ---
    println!("\n🧩 Deliberate \"summarize\" — entire team responds, replies land on board:");
    for (id, reply) in orch.deliberate("summarize").await? {
        println!("   {} → {}", orch.party_name(&Party::Agent(id)), reply);
    }
    println!(
        "   📋 {} records on board",
        orch.read_board(&Query::new("").limit(100)).await?.len()
    );

    // --- Knowledge graph: relationships in Kai's memory ---
    let kai_ref = orch.agent(&kai_id).unwrap();
    let kg = MemoryGraph::from_store(kai_ref.memory.as_ref(), &kai_ref.scope()).await?;
    println!(
        "\n🕸️  Kai knowledge graph: {} nodes, {} entities",
        kg.node_count(),
        kg.entity_count()
    );

    Ok(())
}

/// Interactive subscription login. Supports Anthropic (Claude Pro/Max) and
/// OpenAI (ChatGPT Plus/Pro, Codex).
async fn login(data: &str, provider: &str, device: bool) -> anyhow::Result<()> {
    let pkce = lore::auth::pkce();
    let state = ulid::Ulid::new().to_string();
    let (outcome, hint) = match provider {
        "anthropic" => {
            // Anthropic's public (Claude Code) client only accepts its console
            // redirect, so a localhost/loopback redirect fails with "Invalid
            // request format". The paste-the-code flow uses the registered
            // redirect and is the reliable path.
            let _ = device;
            let o = login_anthropic_manual(&pkce, &state).await?;
            (
                o,
                "LORE_PROVIDER=anthropic LORE_LLM_MODEL=claude-sonnet-4-5-20250929",
            )
        }
        "openai" => {
            if device {
                anyhow::bail!("openai supports only the browser flow (drop --device)");
            }
            let o = login_openai_loopback(&pkce, &state).await?;
            (o, "LORE_PROVIDER=openai LORE_LLM_MODEL=gpt-5")
        }
        _ => anyhow::bail!("unknown provider '{provider}' (supported: anthropic, openai)"),
    };
    let cred = Credential::OAuth {
        access: outcome.access,
        refresh: outcome.refresh,
        expires_ms: outcome.expires_ms,
        account_id: outcome.account_id,
    };
    TokenStore::new(data).save(provider, &cred)?;
    println!(
        "✅ logged in to {provider} (subscription).\n   Use it: {hint} lore ask <agent> \"...\""
    );
    Ok(())
}

/// OpenAI (Codex) browser loopback login. The redirect must match the client's
/// registered URI (`http://localhost:1455/auth/callback`); both IPv4 and IPv6
/// loopback are bound so `localhost` resolves either way.
async fn login_openai_loopback(
    pkce: &lore::auth::Pkce,
    state: &str,
) -> anyhow::Result<lore::auth::OAuthOutcome> {
    let redirect = "http://localhost:1455/auth/callback";
    let mut listeners = Vec::new();
    match std::net::TcpListener::bind("127.0.0.1:1455") {
        Ok(l) => listeners.push(l),
        Err(e) => tracing::warn!("could not bind 127.0.0.1:1455: {e}"),
    }
    if let Ok(l) = std::net::TcpListener::bind("[::1]:1455") {
        listeners.push(l);
    }
    if listeners.is_empty() {
        anyhow::bail!("could not bind localhost:1455 for the OpenAI callback (port in use?)");
    }
    let url = lore::auth::openai_authorize_url(pkce, redirect, state);
    println!("Opening browser for authorization…\nIf it doesn't open, visit:\n\n{url}\n");
    open_browser(&url);
    println!("Waiting for the redirect on {redirect} … (Ctrl-C to cancel)");
    let (code, got_state) = wait_for_redirect(listeners).await?;
    verify_state(state, &got_state)?;
    Ok(lore::auth::exchange_openai_code(&code, &pkce.verifier, redirect).await?)
}

/// Paste-the-code flow (SSH/headless friendly).
async fn login_anthropic_manual(
    pkce: &lore::auth::Pkce,
    state: &str,
) -> anyhow::Result<lore::auth::OAuthOutcome> {
    let redirect = lore::auth::ANTHROPIC_MANUAL_REDIRECT;
    let url = lore::auth::anthropic_authorize_url(pkce, redirect, state);
    println!("Open this URL, authorize, then paste the code shown:\n\n{url}\n");
    print!("code: ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let (code, pasted_state) = lore::auth::split_manual_code(&line);
    if code.is_empty() {
        anyhow::bail!("no code entered");
    }
    let st = pasted_state.unwrap_or_else(|| state.to_string());
    Ok(lore::auth::exchange_anthropic_code(&code, &st, &pkce.verifier, redirect).await?)
}

/// Rejects a callback whose `state` does not match what we sent (CSRF guard).
fn verify_state(sent: &str, got: &Option<String>) -> anyhow::Result<()> {
    if let Some(g) = got {
        if g != sent {
            anyhow::bail!("OAuth state mismatch (possible CSRF); aborting login");
        }
    }
    Ok(())
}

/// Waits (off the async runtime) for the OAuth redirect across the given
/// loopback listeners.
async fn wait_for_redirect(
    listeners: Vec<std::net::TcpListener>,
) -> anyhow::Result<(String, Option<String>)> {
    tokio::task::spawn_blocking(move || {
        capture_redirect(listeners, std::time::Duration::from_secs(300))
    })
    .await?
}

/// Accepts one connection on any listener (one blocking thread each, so IPv4 and
/// IPv6 localhost both work), replies with a friendly page, and returns the
/// captured `(code, state)`. Times out instead of hanging forever.
fn capture_redirect(
    listeners: Vec<std::net::TcpListener>,
    timeout: std::time::Duration,
) -> anyhow::Result<(String, Option<String>)> {
    use std::io::{Read, Write};
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    for l in listeners {
        let tx = tx.clone();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = l.accept() {
                let mut buf = [0u8; 8192];
                if let Ok(n) = sock.read(&mut buf) {
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let parsed = parse_callback_query(&req);
                    let body = if parsed.is_some() {
                        "<html><body>Lore: login complete. You can close this tab.</body></html>"
                    } else {
                        "<html><body>Lore: could not read the authorization code.</body></html>"
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes());
                    let _ = tx.send(parsed);
                }
            }
        });
    }
    drop(tx);
    match rx.recv_timeout(timeout) {
        Ok(Some(cs)) => Ok(cs),
        Ok(None) => anyhow::bail!("could not read the authorization code from the redirect"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!(
                "timed out waiting for the OAuth redirect ({}s)",
                timeout.as_secs()
            )
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("OAuth callback listeners closed before a redirect arrived")
        }
    }
}

/// Extracts `code` (and optional `state`) from an HTTP GET request line.
fn parse_callback_query(req: &str) -> Option<(String, Option<String>)> {
    let first = req.lines().next()?;
    let path = first.split_whitespace().nth(1)?;
    let query = path.split_once('?')?.1;
    let mut code = None;
    let mut state = None;
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            let val = urldecode(v);
            match k {
                "code" => code = Some(val),
                "state" => state = Some(val),
                _ => {}
            }
        }
    }
    Some((code?, state))
}

/// Minimal percent-decoding for redirect query values.
fn urldecode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hex = |c: u8| (c as char).to_digit(16);
                match (hex(b[i + 1]), hex(b[i + 2])) {
                    (Some(hi), Some(lo)) => {
                        out.push((hi * 16 + lo) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Best-effort: opens `url` in the platform browser.
fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = "xdg-open";
    let _ = std::process::Command::new(cmd)
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Prints configured provider credentials and their status.
fn show_auth(data: &str) -> anyhow::Result<()> {
    let store = TokenStore::new(data);
    let providers = store.list()?;
    if providers.is_empty() {
        println!("(no credentials — run `lore login anthropic`)");
        return Ok(());
    }
    for p in &providers {
        if let Some(c) = store.load(p)? {
            let (kind, status) = match &c {
                Credential::ApiKey { .. } => ("api-key", "configured".to_string()),
                Credential::OAuth { expires_ms, .. } => {
                    let mins = (*expires_ms - chrono::Utc::now().timestamp_millis()) / 60_000;
                    let s = if c.is_expired(0) {
                        "expired (auto-refresh on use)".to_string()
                    } else {
                        format!("valid (~{mins}m left)")
                    };
                    ("subscription", s)
                }
            };
            println!("{p:<14} {kind:<14} {status}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn callback_query_parsing() {
        let req = "GET /callback?code=abc%23xyz&state=st%20ate HTTP/1.1\r\nHost: x\r\n\r\n";
        let (code, state) = super::parse_callback_query(req).unwrap();
        assert_eq!(code, "abc#xyz");
        assert_eq!(state.as_deref(), Some("st ate"));
        assert!(super::parse_callback_query("GET /callback HTTP/1.1").is_none());
    }

    #[test]
    fn state_mismatch_is_rejected() {
        assert!(super::verify_state("abc", &Some("abc".to_string())).is_ok());
        assert!(super::verify_state("abc", &None).is_ok()); // provider omitted state
        assert!(super::verify_state("abc", &Some("evil".to_string())).is_err());
    }

    #[test]
    fn urldecode_basics() {
        assert_eq!(super::urldecode("a%2Bb"), "a+b");
        assert_eq!(super::urldecode("a+b"), "a b");
        assert_eq!(super::urldecode("plain"), "plain");
        assert_eq!(super::urldecode("trailing%"), "trailing%");
    }

    #[test]
    fn empty_api_key_is_rejected_not_open_door() {
        // L1: LORE_API_KEY="" was making auth an open door — passed with an empty header.
        assert_eq!(super::parse_api_key(Some(String::new())), None);
        assert_eq!(super::parse_api_key(Some("   ".into())), None);
        assert_eq!(super::parse_api_key(None), None);
        assert_eq!(
            super::parse_api_key(Some("  secret  ".into())),
            Some("secret".to_string())
        );
    }
}
