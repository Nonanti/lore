//! Lore CLI: standalone AI agent infrastructure — library + service + terminal.
//!
//! Subcommands share a persistent data directory (SQLite memory + persona files);
//! so an agent created from the terminal is also reachable via `serve` over the network.

use clap::{Parser, Subcommand};
use lore::{
    Agent, AgentId, AppState, CalcTool, FileReadTool, HashingEmbedder, InMemoryStore,
    KeywordRouter, Memory, MemoryGraph, MemoryStore, MessageKind, MockModel, Model, OpenAiModel,
    Orchestrator, Party, Persona, PersonaPatch, Query, Scope, SemanticCat, SqliteStore, TimeTool,
    ToolContext, ToolRegistry, WebFetchTool,
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

/// Sets up the model: if LORE_LLM_BASE is set, uses OpenAI-compatible (Ollama); otherwise MockModel.
fn build_model() -> Arc<dyn Model> {
    match std::env::var("LORE_LLM_BASE") {
        Ok(base) => {
            let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
            let mut m = OpenAiModel::new(base, name);
            if let Ok(key) = std::env::var("LORE_LLM_KEY") {
                m = m.with_api_key(key);
            }
            // Optional response token limit. Low values on reasoning models may
            // spend the budget on thinking — use deliberately.
            if let Ok(mt) = std::env::var("LORE_LLM_MAX_TOKENS") {
                match mt.parse::<u32>() {
                    Ok(n) if n > 0 => m = m.with_max_tokens(n),
                    _ => tracing::warn!(value = %mt, "LORE_LLM_MAX_TOKENS invalid, ignored"),
                }
            }
            // Optional request timeout (seconds). Slow local models (e.g. 14B+
            // on CPU) may exceed the default 120 s — can be increased. Invalid/0 is ignored.
            if let Ok(to) = std::env::var("LORE_LLM_TIMEOUT") {
                match to.parse::<u64>() {
                    Ok(n) if n > 0 => m = m.with_timeout(std::time::Duration::from_secs(n)),
                    _ => tracing::warn!(value = %to, "LORE_LLM_TIMEOUT invalid, ignored"),
                }
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
        AppState::persistent(format!("{data}/agents"), store, build_model())?.with_tools(tools);
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
        Cmd::Demo => run_demo().await?,
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
async fn run_demo() -> anyhow::Result<()> {
    // Native embedder attached → recall is hybrid (keyword + cosine).
    let store: Arc<dyn MemoryStore> =
        Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));
    let model = build_model();
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

#[cfg(test)]
mod tests {
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
