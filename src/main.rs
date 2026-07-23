//! Lore CLI: standalone AI agent infrastructure — library + service + terminal.
//!
//! Subcommands share a persistent data directory (SQLite memory + persona files);
//! so an agent created from the terminal is also reachable via `serve` over the network.

use clap::{Parser, Subcommand};
use lore::{
    build_model, build_model_from_env, preset, Agent, AgentId, AppState, AuthKind, CalcTool,
    Credential, FileReadTool, HashingEmbedder, InMemoryStore, KeywordRouter, Memory, MemoryGraph,
    MemoryStore, MessageKind, Model, ModelConfig, NewTask, Orchestrator, Party, Persona,
    PersonaPatch, ProviderKind, Query, Scope, SemanticCat, SqliteStore, TaskStore, TimeTool,
    TokenStore, ToolContext, ToolRegistry, WebFetchTool,
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

#[derive(Debug, Subcommand)]
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
    /// Run the task queue daemon in the foreground.
    Daemon,
    /// Task management subcommands.
    Task {
        #[command(subcommand)]
        task_cmd: TaskCmd,
    },
    /// Agent management subcommands.
    Agent {
        #[command(subcommand)]
        agent_cmd: AgentCmd,
    },
    /// Show pending approval inbox.
    Inbox,
    /// Approve a pending approval.
    Approve { id: String },
    /// Deny a pending approval.
    Deny { id: String },
}

#[derive(Debug, Subcommand)]
enum TaskCmd {
    /// Enqueue a new task.
    Add {
        /// Agent name (persona file stem).
        agent: String,
        /// Goal description.
        goal: String,
        /// Workspace root path.
        #[arg(long)]
        workspace: Option<String>,
        /// Verification commands (repeatable).
        #[arg(long, short = 'v', value_delimiter = ',')]
        verify: Vec<String>,
        /// Team task: agent forced to 'pm' for decomposition.
        #[arg(long)]
        team: bool,
    },
    /// List tasks (compact table).
    List {
        /// Max tasks to show (default 20).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Show full task record.
    Status { id: String },
    /// Print the task log file.
    Log {
        id: String,
        /// Show last N lines (default all).
        #[arg(long)]
        tail: Option<usize>,
    },
}

#[derive(Debug, Subcommand)]
enum AgentCmd {
    /// Create a new agent with a role preset and optional model config.
    Create {
        /// Agent name (used as persona file stem).
        name: String,
        /// Role preset (backend, frontend, reviewer, pm) or freeform role.
        #[arg(long)]
        role: String,
        /// LLM provider (anthropic, openai, openai-compat, mock).
        /// Omit → env fallback (no model field in JSON).
        #[arg(long)]
        provider: Option<String>,
        /// Model name (e.g. claude-sonnet-4-5-20250929, qwen3:8b).
        #[arg(long)]
        model: Option<String>,
        /// Auth method: key (metered) or subs (subscription).
        #[arg(long)]
        auth: Option<String>,
        /// Base URL for OpenAI-compatible provider.
        #[arg(long)]
        base_url: Option<String>,
    },
    /// List agents (name, role, provider/model or '(env)').
    List,
}
/// Sets up the model from env config (centralized in `lore::model::factory`).
fn build_model_from_env_cli(data: &str) -> Arc<dyn Model> {
    let path = std::path::Path::new(data);
    build_model_from_env(path)
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
    let mut app = AppState::persistent(
        format!("{data}/agents"),
        store,
        build_model_from_env_cli(data),
    )?
    .with_tools(tools);
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
/// Handle agent subcommands: create with role preset + optional model, list.
fn handle_agent(data: &str, cmd: AgentCmd) -> anyhow::Result<()> {
    let agents_dir = format!("{}/agents", data);
    std::fs::create_dir_all(&agents_dir)?;

    match cmd {
        AgentCmd::Create {
            name,
            role,
            provider,
            model,
            auth,
            base_url,
        } => {
            // Resolve role preset or use the role string as-is.
            let role_preset = preset(&role);
            let (role_str, traits, extra_lines) = match &role_preset {
                Some(r) => (
                    r.role.to_string(),
                    r.traits.to_vec(),
                    vec![r.identity_extra.to_string()],
                ),
                None => (role.clone(), Vec::new(), Vec::new()),
            };

            let mut persona = Persona::new(&name, &role_str);
            if !traits.is_empty() {
                persona = persona.with_traits(traits.iter().map(|s| s.to_string()));
            }
            if !extra_lines.is_empty() {
                persona = persona.with_extra(extra_lines);
            }

            // Build ModelConfig only if provider is specified.
            // No provider → no model field → env fallback at runtime.
            let model_config: Option<ModelConfig> = if let Some(p_str) = &provider {
                let provider_kind = match p_str.as_str() {
                    "anthropic" => ProviderKind::Anthropic,
                    "openai" => ProviderKind::OpenAI,
                    "openai-compat" => ProviderKind::OpenAiCompat,
                    "mock" => ProviderKind::Mock,
                    other => anyhow::bail!(
                        "unknown provider '{other}' (expected: anthropic, openai, openai-compat, mock)"
                    ),
                };
                let model_name = model.clone().unwrap_or_else(|| match provider_kind {
                    ProviderKind::Anthropic => "claude-sonnet-4-5".to_string(),
                    ProviderKind::OpenAI => "gpt-5".to_string(),
                    ProviderKind::OpenAiCompat => "llama3.2".to_string(),
                    ProviderKind::Mock => "mock".to_string(),
                });
                let auth_kind = match auth.as_deref() {
                    Some("key") => Some(AuthKind::Key),
                    Some("subs") => Some(AuthKind::Subs),
                    Some(other) => anyhow::bail!("unknown auth '{other}' (expected: key, subs)"),
                    None => None,
                };
                Some(ModelConfig {
                    provider: provider_kind,
                    model: model_name,
                    auth: auth_kind,
                    base_url: base_url.clone(),
                })
            } else {
                None
            };

            // Build model: use per-agent config if present, else env.
            let data_path = std::path::Path::new(data);
            let arc_model: Arc<dyn Model> = match &model_config {
                Some(cfg) => build_model(cfg, data_path)?,
                None => build_model_from_env(data_path),
            };

            let agent = Agent::new(persona, Arc::new(lore::InMemoryStore::new()), arc_model);
            let agent = match model_config {
                Some(cfg) => agent.with_model_config(cfg),
                None => agent,
            };

            // Save persona+model JSON under <data>/agents/<name>.json.
            let path = std::path::PathBuf::from(&agents_dir).join(format!("{name}.json"));
            agent.save_to(&path)?;

            // Summary line.
            let model_label = match agent.model_config() {
                Some(cfg) => format!(
                    "{}/{}",
                    serde_json::to_value(&cfg.provider)?.as_str().unwrap_or("?"),
                    cfg.model
                ),
                None => "(env)".to_string(),
            };
            println!(
                "✅ created: {}  {}  {}  model: {}",
                agent.id, agent.persona.name, agent.persona.role, model_label
            );
        }
        AgentCmd::List => {
            // Scan <data>/agents/*.json, show name, role, provider/model or '(env)'.
            let dir = std::path::Path::new(&agents_dir);
            if !dir.exists() {
                println!("(no agents — create one with 'lore agent create')");
                return Ok(());
            }
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let json = std::fs::read_to_string(&path)?;
                let rec: serde_json::Value = serde_json::from_str(&json)?;
                let name = rec
                    .get("persona")
                    .and_then(|p| p.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("?");
                let role = rec
                    .get("persona")
                    .and_then(|p| p.get("role"))
                    .and_then(|r| r.as_str())
                    .unwrap_or("?");
                let model_label = match rec.get("model") {
                    Some(m) => {
                        let provider = m.get("provider").and_then(|p| p.as_str()).unwrap_or("?");
                        let model = m.get("model").and_then(|m| m.as_str()).unwrap_or("?");
                        format!("{provider}/{model}")
                    }
                    None => "(env)".to_string(),
                };
                entries.push(format!("{name:<12} {role:<20} {model_label}"));
            }
            if entries.is_empty() {
                println!("(no agents — create one with 'lore agent create')");
            } else {
                for line in &entries {
                    println!("{line}");
                }
            }
        }
    }
    Ok(())
}

/// Handle task subcommands.
fn handle_task(data: &str, cmd: TaskCmd) -> anyhow::Result<()> {
    let db_path = format!("{}/tasks.db", data);
    let store = TaskStore::open(std::path::Path::new(&db_path))?;

    match cmd {
        TaskCmd::Add {
            agent,
            goal,
            workspace,
            verify,
            team,
        } => {
            let final_agent = if team { "pm" } else { &agent };
            let ws = workspace.map(std::path::PathBuf::from).unwrap_or_else(|| {
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
            });
            // Canonicalize to an absolute path before storing, so the
            // daemon (which may run from a different CWD, e.g. systemd)
            // resolves the workspace correctly.
            let ws = ws.canonicalize().unwrap_or(ws);
            let new_task = NewTask {
                agent: final_agent.to_string(),
                goal,
                workspace: ws,
                verify,
                parent_id: None,
            };
            let task = store.enqueue(new_task)?;
            if team {
                println!("✅ team task {} queued (agent: pm)", task.id);
            } else {
                println!("✅ task {} queued", task.id);
            }
        }
        TaskCmd::List { limit } => {
            let tasks = store.list(limit)?;
            if tasks.is_empty() {
                println!("(no tasks — add one with 'lore task add')");
            } else {
                // Compact table: id, agent, status, age, goal(60ch).
                for t in &tasks {
                    let mins = chrono::Utc::now()
                        .signed_duration_since(t.created_at)
                        .num_minutes();
                    let goal_short = if t.goal.chars().count() > 60 {
                        let mut s: String = t.goal.chars().take(57).collect();
                        s.push('…');
                        s
                    } else {
                        t.goal.clone()
                    };
                    println!(
                        "{}  {:<10} {:<20} {mins}m  {goal_short}",
                        t.id,
                        t.agent,
                        t.status.as_str()
                    );
                }
            }
        }
        TaskCmd::Status { id } => {
            let task = store.get(&id)?;
            match task {
                Some(t) => {
                    println!("id:          {}", t.id);
                    println!("agent:       {}", t.agent);
                    println!("goal:        {}", t.goal);
                    println!("workspace:   {}", t.workspace.display());
                    println!("status:      {}", t.status.as_str());
                    println!("created:     {}", t.created_at.to_rfc3339());
                    println!("updated:     {}", t.updated_at.to_rfc3339());
                    if let Some(report) = &t.report {
                        let summary: serde_json::Value = serde_json::from_str(report)
                            .unwrap_or_else(|_| serde_json::Value::String(report.clone()));
                        println!("report:      {}", summary);
                    }
                    if let Some(pid) = &t.parent_id {
                        println!("parent:      {}", pid);
                    }
                    // Show children if present.
                    let children = store.children_of(&id)?;
                    if !children.is_empty() {
                        println!("children:");
                        for c in &children {
                            println!(
                                "  {}  {:<10} {:<20}  {}",
                                c.id,
                                c.agent,
                                c.status.as_str(),
                                c.goal
                            );
                        }
                    }
                }
                None => anyhow::bail!("task {id} not found"),
            }
        }
        TaskCmd::Log { id, tail } => {
            // Reject IDs containing path separators or ".." to prevent
            // path traversal (single-operator risk is self-inflicted, but
            // worth validating).
            if id.contains('/') || id.contains('\\') || id.contains("..") {
                anyhow::bail!("invalid task id: {id}");
            }
            let log_path = std::path::PathBuf::from(data)
                .join("logs")
                .join(format!("{id}.log"));
            if !log_path.exists() {
                anyhow::bail!("log file not found for task {id}");
            }
            let content = std::fs::read_to_string(&log_path)?;
            let lines: Vec<&str> = content.lines().collect();
            match tail {
                Some(n) => {
                    for line in lines.iter().rev().take(n).rev() {
                        println!("{line}");
                    }
                }
                None => {
                    for line in &lines {
                        println!("{line}");
                    }
                }
            }
        }
    }
    Ok(())
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
        Cmd::Daemon => {
            let db_path = format!("{}/tasks.db", cli.data);
            lore::run_daemon(
                std::path::Path::new(&cli.data),
                std::path::Path::new(&db_path),
            )
            .await?;
        }
        Cmd::Task { task_cmd } => handle_task(&cli.data, task_cmd)?,
        Cmd::Agent { agent_cmd } => handle_agent(&cli.data, agent_cmd)?,
        Cmd::Inbox => {
            let store = TaskStore::open(std::path::Path::new(&format!("{}/tasks.db", cli.data)))?;
            let pending = store.pending_approvals()?;
            if pending.is_empty() {
                println!("(inbox empty — no pending approvals)");
            } else {
                for a in &pending {
                    let mins = chrono::Utc::now()
                        .signed_duration_since(a.created_at)
                        .num_minutes();
                    println!("{}  task:{}  {}  {mins}m ago", a.id, a.task_id, a.reason);
                }
            }
        }
        Cmd::Approve { id } => {
            let store = TaskStore::open(std::path::Path::new(&format!("{}/tasks.db", cli.data)))?;
            let status = store.approval_status(&id)?;
            match status {
                Some(lore::ApprovalStatus::Pending) => {
                    store.decide_approval(&id, true)?;
                    println!("✅ approved: {id}");
                }
                Some(s) => anyhow::bail!("approval {id} is not pending (status: {s:?})"),
                None => anyhow::bail!("approval {id} not found"),
            }
        }
        Cmd::Deny { id } => {
            let store = TaskStore::open(std::path::Path::new(&format!("{}/tasks.db", cli.data)))?;
            let status = store.approval_status(&id)?;
            match status {
                Some(lore::ApprovalStatus::Pending) => {
                    store.decide_approval(&id, false)?;
                    println!("🚫 denied: {id}");
                }
                Some(s) => anyhow::bail!("approval {id} is not pending (status: {s:?})"),
                None => anyhow::bail!("approval {id} not found"),
            }
        }
    }
    Ok(())
}

/// Introductory demo showcasing identity + orchestration + memory + tools + graph.
async fn run_demo(data: &str) -> anyhow::Result<()> {
    // Native embedder attached → recall is hybrid (keyword + cosine).
    let store: Arc<dyn MemoryStore> =
        Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));
    let model = build_model_from_env_cli(data);
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
        .ok_or_else(|| anyhow::anyhow!("agent not found: {kai_id}"))?
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
            .ok_or_else(|| anyhow::anyhow!("agent not found: {id}"))?
            .recall(&Query::new("").limit(4))
            .await?
        {
            println!("   [{:.3}] {}", r.score, r.item.summary());
        }
    }

    // --- Semantic recall: keyword tutmasa da morfolojiyi yakala ---
    orch.agent(&kai_id)
        .ok_or_else(|| anyhow::anyhow!("agent not found: {kai_id}"))?
        .remember(Memory::semantic(
            Scope::World, // remember() pulls into Kai's scope
            "interest in mathematics",
            SemanticCat::Preference,
        ))
        .await?;
    println!("\n🔎 Kai semantic recall(\"mathematics\") — no keyword match, cosine catches it:");
    for r in orch
        .agent(&kai_id)
        .ok_or_else(|| anyhow::anyhow!("agent not found: {kai_id}"))?
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
    let kai_ref = orch
        .agent(&kai_id)
        .ok_or_else(|| anyhow::anyhow!("agent not found: {kai_id}"))?;
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
            // Anthropic uses a fixed registered loopback redirect (port 53692)
            // and the PKCE verifier as the OAuth state.
            let o = if device {
                login_anthropic_manual(&pkce).await?
            } else {
                login_anthropic_loopback(&pkce).await?
            };
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

/// Anthropic browser loopback login on the registered port 53692 (IPv4+IPv6).
/// Anthropic uses the PKCE verifier as the OAuth `state`.
async fn login_anthropic_loopback(
    pkce: &lore::auth::Pkce,
) -> anyhow::Result<lore::auth::OAuthOutcome> {
    let redirect = lore::auth::ANTHROPIC_REDIRECT;
    let port = lore::auth::ANTHROPIC_CALLBACK_PORT;
    let mut listeners = Vec::new();
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => listeners.push(l),
        Err(e) => tracing::warn!("could not bind 127.0.0.1:{port}: {e}"),
    }
    if let Ok(l) = std::net::TcpListener::bind(("::1", port)) {
        listeners.push(l);
    }
    if listeners.is_empty() {
        anyhow::bail!(
            "could not bind localhost:{port} for the Anthropic callback \
             (port in use? try --device)"
        );
    }
    let url = lore::auth::anthropic_authorize_url(pkce, redirect, &pkce.verifier);
    println!("Opening browser for authorization…\nIf it doesn't open, visit:\n\n{url}\n");
    open_browser(&url);
    println!("Waiting for the redirect on {redirect} … (Ctrl-C to cancel, or use --device)");
    let (code, got_state) = wait_for_redirect(listeners).await?;
    verify_state(&pkce.verifier, &got_state)?;
    Ok(
        lore::auth::exchange_anthropic_code(&code, &pkce.verifier, &pkce.verifier, redirect)
            .await?,
    )
}

/// Paste-the-code flow (SSH/headless): user copies the `code#state` shown.
async fn login_anthropic_manual(
    pkce: &lore::auth::Pkce,
) -> anyhow::Result<lore::auth::OAuthOutcome> {
    let redirect = lore::auth::ANTHROPIC_REDIRECT;
    let url = lore::auth::anthropic_authorize_url(pkce, redirect, &pkce.verifier);
    println!(
        "Open this URL, authorize, then paste the code shown (looks like `code#state`):\n\n{url}\n"
    );
    print!("code: ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let (code, pasted_state) = lore::auth::split_manual_code(&line);
    if code.is_empty() {
        anyhow::bail!("no code entered");
    }
    verify_state(&pkce.verifier, &pasted_state)?;
    Ok(
        lore::auth::exchange_anthropic_code(&code, &pkce.verifier, &pkce.verifier, redirect)
            .await?,
    )
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
    use clap::Parser;

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

    // ── Clap parsing: new subcommands ────────────────────────────────

    #[test]
    fn parse_daemon() {
        let cli = super::Cli::parse_from(["lore", "daemon"]);
        assert!(matches!(cli.cmd, Some(super::Cmd::Daemon)));
    }

    #[test]
    fn parse_task_add() {
        let cli = super::Cli::parse_from(["lore", "task", "add", "myagent", "fix the bug"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::Add {
                    agent,
                    goal,
                    workspace,
                    verify,
                    team,
                } => {
                    assert_eq!(agent, "myagent");
                    assert_eq!(goal, "fix the bug");
                    assert!(workspace.is_none());
                    assert!(verify.is_empty());
                    assert!(!team);
                }
                other => panic!("expected Add, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_add_with_workspace_and_verify() {
        let cli = super::Cli::parse_from([
            "lore",
            "task",
            "add",
            "bot",
            "goal",
            "--workspace",
            "/tmp/ws",
            "--verify",
            "cargo test",
        ]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::Add {
                    agent,
                    goal,
                    workspace,
                    verify,
                    team,
                } => {
                    assert_eq!(agent, "bot");
                    assert_eq!(goal, "goal");
                    assert_eq!(workspace.as_deref(), Some("/tmp/ws"));
                    assert_eq!(verify, vec!["cargo test"]);
                    assert!(!team);
                }
                other => panic!("expected Add, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_list() {
        let cli = super::Cli::parse_from(["lore", "task", "list"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::List { limit } => assert_eq!(limit, 20),
                other => panic!("expected List, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_list_with_limit() {
        let cli = super::Cli::parse_from(["lore", "task", "list", "--limit", "5"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::List { limit } => assert_eq!(limit, 5),
                other => panic!("expected List, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_status() {
        let cli = super::Cli::parse_from(["lore", "task", "status", "01ABC"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::Status { id } => assert_eq!(id, "01ABC"),
                other => panic!("expected Status, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_log() {
        let cli = super::Cli::parse_from(["lore", "task", "log", "01ABC"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::Log { id, tail } => {
                    assert_eq!(id, "01ABC");
                    assert!(tail.is_none());
                }
                other => panic!("expected Log, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_task_log_with_tail() {
        let cli = super::Cli::parse_from(["lore", "task", "log", "01ABC", "--tail", "10"]);
        match cli.cmd {
            Some(super::Cmd::Task { task_cmd }) => match task_cmd {
                super::TaskCmd::Log { id, tail } => {
                    assert_eq!(id, "01ABC");
                    assert_eq!(tail, Some(10));
                }
                other => panic!("expected Log, got {other:?}"),
            },
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_inbox() {
        let cli = super::Cli::parse_from(["lore", "inbox"]);
        assert!(matches!(cli.cmd, Some(super::Cmd::Inbox)));
    }

    #[test]
    fn parse_approve() {
        let cli = super::Cli::parse_from(["lore", "approve", "approval_id"]);
        match cli.cmd {
            Some(super::Cmd::Approve { id }) => assert_eq!(id, "approval_id"),
            other => panic!("expected Approve, got {other:?}"),
        }
    }

    #[test]
    fn parse_deny() {
        let cli = super::Cli::parse_from(["lore", "deny", "approval_id"]);
        match cli.cmd {
            Some(super::Cmd::Deny { id }) => assert_eq!(id, "approval_id"),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[test]
    fn parse_agent_create() {
        let cli =
            super::Cli::parse_from(["lore", "agent", "create", "devbot", "--role", "backend"]);
        match cli.cmd {
            Some(super::Cmd::Agent { agent_cmd }) => match agent_cmd {
                super::AgentCmd::Create {
                    name,
                    role,
                    provider,
                    model,
                    auth,
                    base_url,
                } => {
                    assert_eq!(name, "devbot");
                    assert_eq!(role, "backend");
                    assert!(provider.is_none());
                    assert!(model.is_none());
                    assert!(auth.is_none());
                    assert!(base_url.is_none());
                }
                other => panic!("expected Create, got {other:?}"),
            },
            other => panic!("expected Agent, got {other:?}"),
        }
    }

    #[test]
    fn parse_agent_create_with_all_options() {
        let cli = super::Cli::parse_from([
            "lore",
            "agent",
            "create",
            "devbot",
            "--role",
            "backend",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5-20250929",
            "--auth",
            "subs",
            "--base-url",
            "http://localhost:11434/v1",
        ]);
        match cli.cmd {
            Some(super::Cmd::Agent { agent_cmd }) => match agent_cmd {
                super::AgentCmd::Create {
                    name,
                    role,
                    provider,
                    model,
                    auth,
                    base_url,
                } => {
                    assert_eq!(name, "devbot");
                    assert_eq!(role, "backend");
                    assert_eq!(provider.as_deref(), Some("anthropic"));
                    assert_eq!(model.as_deref(), Some("claude-sonnet-4-5-20250929"));
                    assert_eq!(auth.as_deref(), Some("subs"));
                    assert_eq!(base_url.as_deref(), Some("http://localhost:11434/v1"));
                }
                other => panic!("expected Create, got {other:?}"),
            },
            other => panic!("expected Agent, got {other:?}"),
        }
    }

    #[test]
    fn parse_agent_create_mock_provider() {
        let cli = super::Cli::parse_from([
            "lore",
            "agent",
            "create",
            "testbot",
            "--role",
            "tester",
            "--provider",
            "mock",
        ]);
        match cli.cmd {
            Some(super::Cmd::Agent { agent_cmd }) => match agent_cmd {
                super::AgentCmd::Create { name, provider, .. } => {
                    assert_eq!(name, "testbot");
                    assert_eq!(provider.as_deref(), Some("mock"));
                }
                other => panic!("expected Create, got {other:?}"),
            },
            other => panic!("expected Agent, got {other:?}"),
        }
    }

    #[test]
    fn parse_agent_list() {
        let cli = super::Cli::parse_from(["lore", "agent", "list"]);
        match cli.cmd {
            Some(super::Cmd::Agent { agent_cmd }) => match agent_cmd {
                super::AgentCmd::List => {}
                other => panic!("expected List, got {other:?}"),
            },
            other => panic!("expected Agent, got {other:?}"),
        }
    }
}
