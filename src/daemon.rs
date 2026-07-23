//! Task queue daemon: sequential loop that dequeues tasks, runs agents, and
//! records results.
//!
//! The daemon is the sole process that transitions task state (Queued → Running
//! → Completed/Failed). CLI inserts tasks and answers approvals; WAL mode
//! permits concurrent SQLite access. Per-task execution is extracted into
//! [`run_task`] for testing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::agent::{Agent, WorkReport, WorkSpec};
use crate::error::Result;
use crate::memory::{HashingEmbedder, MemoryStore, SqliteStore};
use crate::model::Model;
use crate::policy::approval::Gate;
use crate::policy::Policy;
use crate::task::approver::QueueApprover;
use crate::task::{TaskStatus, TaskStore};
use crate::tool::{
    FileEditTool, FileReadTool, FileWriteTool, LlmRouter, ShellTool, ToolContext, ToolRegistry,
};

/// Idle sleep between next_queued polls (seconds).
const IDLE_POLL_SECS: u64 = 2;

/// Simple per-task log appender. Writes lines to `<data>/logs/<task_id>.log`.
/// Not behind a new logging dependency — plain `std::fs::OpenOptions`.
struct TaskLog {
    file: std::fs::File,
}

impl TaskLog {
    fn open(data_dir: &Path, task_id: &str) -> Result<Self> {
        let logs_dir = data_dir.join("logs");
        std::fs::create_dir_all(&logs_dir)
            .map_err(|e| crate::error::LoreError::Storage(e.to_string()))?;
        let path = logs_dir.join(format!("{task_id}.log"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| crate::error::LoreError::Storage(e.to_string()))?;
        Ok(Self { file })
    }

    fn write_line(&mut self, line: &str) {
        let _ = self.file.write_all(format!("{line}\n").as_bytes());
    }
}

/// Dependencies needed for [`run_task`]. Extracted so tests can inject stubs.
pub struct TaskDeps {
    /// Path to the LORE_DATA directory.
    pub data_dir: PathBuf,
    /// LLM model (built from env config, same as CLI).
    pub model: Arc<dyn Model>,
    /// Task DB file path.
    pub db_path: PathBuf,
}

/// Runs a single task: loads agent, builds policy/gate/tools, calls
/// `agent.work`, and records the result in the task store.
///
/// Returns the [`WorkReport`] from the agent's work loop. Errors from agent
/// loading or model calls are recorded as task failures (daemon continues).
pub async fn run_task(store: &TaskStore, task_id: &str, deps: &TaskDeps) -> Result<WorkReport> {
    let task = store
        .get(task_id)?
        .ok_or_else(|| crate::error::LoreError::NotFound(format!("task {task_id}")))?;

    let mut log = TaskLog::open(&deps.data_dir, task_id)?;

    log.write_line(&format!(
        "[daemon] task {} started — agent: {}, goal: {}",
        task.id, task.agent, task.goal
    ));

    // Load agent persona from <data>/agents/<name>.json.
    let persona_path = deps
        .data_dir
        .join("agents")
        .join(format!("{}.json", task.agent));
    if !persona_path.exists() {
        let msg = format!(
            "persona file not found for agent '{}': {}",
            task.agent,
            persona_path.display()
        );
        tracing::error!(task_id, agent = %task.agent, "{}", msg);
        log.write_line(&format!("[daemon] FATAL: {msg}"));
        let report_json = serde_json::to_string(&WorkReport {
            success: false,
            iterations: 0,
            answer: msg.clone(),
            verify_log: String::new(),
        })?;
        store.fail(task_id, &report_json)?;
        return Err(crate::error::LoreError::InvalidInput(msg));
    }

    // Scoped memory: <data>/memory/<agent_name>.db (same pattern as AppState).
    let mem_path = deps
        .data_dir
        .join("memory")
        .join(format!("{}.db", task.agent));
    let parent = mem_path
        .parent()
        .ok_or_else(|| crate::error::LoreError::Storage("memory dir has no parent".into()))?;
    std::fs::create_dir_all(parent).map_err(|e| crate::error::LoreError::Storage(e.to_string()))?;
    let mem_store: Arc<dyn MemoryStore> = Arc::new(
        SqliteStore::open(&mem_path.to_string_lossy())
            .map_err(|e| crate::error::LoreError::Storage(e.to_string()))?
            .with_embedder(Arc::new(HashingEmbedder::new())),
    );

    let agent = Agent::load_from(&persona_path, mem_store, deps.model.clone())?;

    // Policy: load <data>/policy.json if present, else default_for(workspace).
    let policy_path = deps.data_dir.join("policy.json");
    let policy = if policy_path.exists() {
        Policy::load(&policy_path)?
    } else {
        Policy::default_for(task.workspace.clone())
    };

    // Gate with QueueApprover (approval requests flow through the task DB).
    let approver = QueueApprover::with_default_poll(&deps.db_path, task_id);
    let gate = Arc::new(Gate::new(policy, Arc::new(approver)));

    // ToolContext: shell + write + edit + file-read, LlmRouter.
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(ShellTool::new(
        gate.clone(),
        task.workspace.clone(),
    )));
    reg.register(Arc::new(FileWriteTool::new(
        gate.clone(),
        task.workspace.clone(),
    )));
    reg.register(Arc::new(FileEditTool::new(
        gate.clone(),
        task.workspace.clone(),
    )));
    reg.register(Arc::new(FileReadTool::new(
        task.workspace.to_string_lossy().to_string(),
    )));
    let router = Arc::new(LlmRouter::new(deps.model.clone()));
    let ctx = ToolContext {
        registry: reg,
        router,
    };

    // Build WorkSpec.
    let spec = if task.verify.is_empty() {
        WorkSpec::for_workspace(&task.goal, task.workspace.clone())?
    } else {
        WorkSpec::new(&task.goal, task.workspace.clone(), task.verify)?
    };

    log.write_line(&format!(
        "[daemon] task {} running — workspace: {}, verify: {}",
        task.id,
        task.workspace.display(),
        if spec.verify.is_empty() {
            "(auto-detect)"
        } else {
            &task.goal
        }
    ));

    // Run the work loop. Errors from agent/model/policy are caught and
    // recorded as task failures (daemon continues to next task).
    let report = match agent.work(&ctx, gate, &spec).await {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("work loop error: {e}");
            tracing::error!(task_id, error = %e, "{msg}");
            log.write_line(&format!("[daemon] FATAL: {msg}"));
            let report_json = serde_json::to_string(&WorkReport {
                success: false,
                iterations: 0,
                answer: msg.clone(),
                verify_log: String::new(),
            })?;
            store.fail(task_id, &report_json)?;
            return Err(e);
        }
    };

    log.write_line(&format!(
        "[daemon] task {} completed — success: {}, iterations: {}",
        task.id, report.success, report.iterations
    ));

    // Record result in the task store.
    let report_json = serde_json::to_string(&report)?;
    if report.success {
        store.complete(task_id, &report_json)?;
    } else {
        store.fail(task_id, &report_json)?;
    }

    Ok(report)
}

/// Daemon entry: sequential loop that dequeues and runs tasks.
///
/// Graceful shutdown on SIGTERM/SIGINT: marks the currently-running task
/// back as Queued so it can be resumed later, then exits.
pub async fn run_daemon(data_dir: &Path, db_path: &Path) -> Result<()> {
    tracing::info!("daemon starting — data: {}", data_dir.display());

    let store = TaskStore::open(db_path)?;
    let model = build_model_from_env(data_dir);

    // Crash recovery: sweep orphaned tasks left from a previous crash
    // or kill. Resets Running/WaitingApproval → Queued and denies
    // stale Pending approvals.
    let recovered = store.recover_orphaned()?;
    if recovered > 0 {
        tracing::info!(
            count = recovered,
            "re-queued orphaned tasks from previous run"
        );
    }

    let deps = TaskDeps {
        data_dir: data_dir.to_path_buf(),
        model,
        db_path: db_path.to_path_buf(),
    };

    // Graceful shutdown: listen for SIGTERM/SIGINT.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let shutdown_task = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("shutdown signal received — finishing current task");
        let _ = shutdown_tx.send(()).await;
    });

    loop {
        // Check for shutdown before polling.
        if shutdown_rx.try_recv().is_ok() {
            tracing::info!("daemon exiting (idle, no mid-run task)");
            break;
        }

        // Poll for next queued task.
        let task = store.next_queued()?;
        if task.is_none() {
            tokio::time::sleep(Duration::from_secs(IDLE_POLL_SECS)).await;
            continue;
        }

        let task = task.expect("checked above");
        tracing::info!(task_id = %task.id, agent = %task.agent, "dequeued task");
        store.set_status(&task.id, TaskStatus::Running)?;

        // Run task, with shutdown-awareness: if SIGINT arrives mid-run,
        // mark the task Queued again for later resumption. NOTE: cancellation
        // is at the next Tokio await point (shell command, LLM call, approval
        // poll) — "finish current verify command" is best-effort, not a
        // guaranteed checkpoint.
        let result = tokio::select! {
            r = run_task(&store, &task.id, &deps) => r,
            _ = shutdown_rx.recv() => {
                tracing::warn!(task_id = %task.id, "shutdown mid-run — re-queuing task + denying stale approvals");
                store.set_status(&task.id, TaskStatus::Queued)?;
                // Deny any Pending approvals for the re-queued task to avoid
                // inbox clutter from a stale partial work cycle.
                store.deny_pending_approvals_for_task(&task.id)?;
                break;
            }
        };

        match result {
            Ok(report) => {
                tracing::info!(
                    task_id = %task.id,
                    success = report.success,
                    iterations = report.iterations,
                    "task finished"
                );
            }
            Err(e) => {
                // run_task already records failures (persona missing, etc.)
                // in the task store. If it failed before recording, do it now.
                if store
                    .get(&task.id)?
                    .is_none_or(|t| t.status != TaskStatus::Failed)
                {
                    let report_json = serde_json::to_string(&WorkReport {
                        success: false,
                        iterations: 0,
                        answer: format!("daemon error: {e}"),
                        verify_log: String::new(),
                    })?;
                    store.fail(&task.id, &report_json)?;
                }
                tracing::error!(task_id = %task.id, error = %e, "task failed — daemon continues");
            }
        }
    }

    shutdown_task.abort();
    tracing::info!("daemon stopped");
    Ok(())
}

/// Build the model from the same env config as CLI (reuse main.rs helpers).
/// This function is standalone to avoid importing the entire main.rs module.
fn build_model_from_env(data_dir: &Path) -> Arc<dyn Model> {
    let data_str = data_dir.to_string_lossy().to_string();

    // Same logic as main.rs build_model() — replicated here because
    // daemon.rs is a separate module and we avoid a circular dependency.
    match std::env::var("LORE_PROVIDER").ok().as_deref() {
        Some("anthropic") => return build_anthropic(&data_str),
        Some("openai") => return build_openai(&data_str),
        _ => {}
    }
    match std::env::var("LORE_LLM_BASE") {
        Ok(base) => {
            let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
            let mut m = crate::model::OpenAiModel::new(base, name);
            if let Ok(key) = std::env::var("LORE_LLM_KEY") {
                m = m.with_api_key(key);
            }
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        Err(_) => {
            tracing::warn!("no LLM provider configured — using MockModel");
            Arc::new(crate::model::MockModel::new())
        }
    }
}

fn build_anthropic(data: &str) -> Arc<dyn Model> {
    let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".into());
    match resolve_anthropic_auth(data) {
        Some(auth) => {
            let mut m = crate::model::AnthropicModel::new(name, auth);
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        None => {
            tracing::warn!("LORE_PROVIDER=anthropic but no credential found; using MockModel");
            Arc::new(crate::model::MockModel::new())
        }
    }
}

fn build_openai(data: &str) -> Arc<dyn Model> {
    let name = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "gpt-5".into());
    let store = crate::auth::TokenStore::new(data);
    let stored = store.load("openai").ok().flatten();

    // Subscription (Codex) path.
    if let Some(cred @ crate::auth::Credential::OAuth { account_id, .. }) = &stored {
        let account_id = account_id.clone();
        let refreshing = crate::auth::RefreshingToken::new(
            store,
            "openai",
            cred.clone(),
            Box::new(|rt: String| Box::pin(async move { crate::auth::refresh_openai(&rt).await })),
        );
        let mut m = crate::model::CodexModel::new(name, Arc::new(refreshing), account_id);
        if let Some(d) = env_timeout() {
            m = m.with_timeout(d);
        }
        return Arc::new(m);
    }

    // Metered API-key path.
    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("LORE_LLM_KEY"))
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| match &stored {
            Some(crate::auth::Credential::ApiKey { key }) => Some(key.clone()),
            _ => None,
        });
    match api_key {
        Some(k) => {
            let mut m =
                crate::model::OpenAiModel::new("https://api.openai.com/v1", name).with_api_key(k);
            if let Some(n) = env_max_tokens() {
                m = m.with_max_tokens(n);
            }
            if let Some(d) = env_timeout() {
                m = m.with_timeout(d);
            }
            Arc::new(m)
        }
        None => {
            tracing::warn!("LORE_PROVIDER=openai but no credential found; using MockModel");
            Arc::new(crate::model::MockModel::new())
        }
    }
}

fn resolve_anthropic_auth(data: &str) -> Option<crate::model::AnthropicAuth> {
    let store = crate::auth::TokenStore::new(data);
    let stored = store.load("anthropic").ok().flatten();
    let mode = std::env::var("LORE_AUTH").ok();
    let want_key = mode.as_deref() == Some("key");
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .or_else(|_| std::env::var("LORE_LLM_KEY"))
        .ok()
        .filter(|k| !k.trim().is_empty());

    if want_key {
        return api_key.map(crate::model::AnthropicAuth::ApiKey);
    }
    if let Some(crate::auth::Credential::ApiKey { key }) = &stored {
        if mode.as_deref() != Some("subs") {
            return Some(crate::model::AnthropicAuth::ApiKey(key.clone()));
        }
    }
    if let Some(cred @ crate::auth::Credential::OAuth { .. }) = stored {
        let refreshing = crate::auth::RefreshingToken::new(
            store,
            "anthropic",
            cred,
            Box::new(|rt: String| {
                Box::pin(async move { crate::auth::refresh_anthropic(&rt).await })
            }),
        );
        return Some(crate::model::AnthropicAuth::OAuth(Arc::new(refreshing)));
    }
    api_key.map(crate::model::AnthropicAuth::ApiKey)
}

fn env_max_tokens() -> Option<u32> {
    std::env::var("LORE_LLM_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|n| *n > 0)
}

fn env_timeout() -> Option<Duration> {
    std::env::var("LORE_LLM_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, Persona};
    use crate::memory::InMemoryStore;
    use crate::model::{Completion, MockModel, Model, Prompt};
    use crate::task::NewTask;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    // ── Helpers ───────────────────────────────────────────────────────

    /// Tempdir-backed DB path (manual cleanup with WAL side-files).
    struct TmpDb {
        dir: PathBuf,
        path: PathBuf,
    }

    impl TmpDb {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "lore-daemon-test-{label}-{pid}-{uid}",
                pid = std::process::id(),
                uid = ulid::Ulid::new().to_string()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("tasks.db");
            Self { dir, path }
        }

        fn path(&self) -> &Path {
            &self.path
        }

        fn data_dir(&self) -> &Path {
            &self.dir
        }
    }

    impl Drop for TmpDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn make_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lore-rt-test-{label}-{pid}-{uid}",
            pid = std::process::id(),
            uid = ulid::Ulid::new().to_string()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &PathBuf) {
        std::fs::remove_dir_all(dir).ok();
    }

    /// Save a persona file for the given agent name using Agent::save_to
    /// (so Agent::load_from can read the AgentRecord format).
    fn save_persona(data_dir: &Path, agent_name: &str) {
        let agents_dir = data_dir.join("agents");
        std::fs::create_dir_all(&agents_dir).unwrap();
        let persona = Persona::new(agent_name, "worker");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(persona, store, Arc::new(MockModel::new()));
        agent
            .save_to(agents_dir.join(format!("{agent_name}.json")))
            .unwrap();
    }

    /// Save a permissive policy.json in the data dir so tests don't
    /// block on approval.
    fn save_permissive_policy(data_dir: &Path) {
        let policy = Policy {
            roots: vec![std::env::temp_dir()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: crate::policy::DefaultExec::Allow,
            ask_on_write: false,
        };
        policy.save(&data_dir.join("policy.json")).unwrap();
    }

    /// Test model that returns scripted replies in sequence.
    struct ScriptedModel {
        replies: Mutex<VecDeque<String>>,
    }

    impl ScriptedModel {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for ScriptedModel {
        async fn complete(&self, _p: &Prompt) -> crate::error::Result<Completion> {
            let text = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "no reply left".into());
            Ok(Completion::new(text))
        }
    }

    /// Error model that always fails.
    struct ErrorModel(String);

    #[async_trait::async_trait]
    impl Model for ErrorModel {
        async fn complete(&self, _p: &Prompt) -> crate::error::Result<Completion> {
            Err(crate::error::LoreError::Model(self.0.clone()))
        }
    }

    // ── run_task: success → task Completed ─────────────────────────────

    #[tokio::test]
    async fn run_task_success_marks_completed() {
        let db = TmpDb::new("rt-success");
        let workspace = make_temp_dir("rt-success-ws");
        let store = TaskStore::open(db.path()).unwrap();

        // Create persona file for agent "testbot".
        save_persona(db.data_dir(), "testbot");
        save_permissive_policy(db.data_dir());

        // Enqueue task.
        let task = store
            .enqueue(NewTask {
                agent: "testbot".to_string(),
                goal: "say hello".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        let model = Arc::new(ScriptedModel::new(&["done"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let report = run_task(&store, &task.id, &deps).await.unwrap();
        assert!(report.success, "task should succeed");

        let loaded = store.get(&task.id).unwrap().unwrap();
        assert_eq!(
            loaded.status,
            TaskStatus::Completed,
            "task should be Completed"
        );

        // Log file exists.
        let log_path = db.data_dir().join("logs").join(format!("{}.log", task.id));
        assert!(log_path.exists(), "task log file should exist");
        let log_content = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            log_content.contains("started"),
            "log should contain start marker"
        );

        cleanup(&workspace);
    }

    // ── run_task: model error → Failed, daemon continues ──────────────

    #[tokio::test]
    async fn run_task_model_error_marks_failed() {
        let db = TmpDb::new("rt-model-err");
        let workspace = make_temp_dir("rt-err-ws");
        let store = TaskStore::open(db.path()).unwrap();

        save_persona(db.data_dir(), "errbot");
        save_permissive_policy(db.data_dir());

        let task = store
            .enqueue(NewTask {
                agent: "errbot".to_string(),
                goal: "fail this task".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        let model = Arc::new(ErrorModel("model error: connection refused".into()));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let result = run_task(&store, &task.id, &deps).await;
        assert!(result.is_err(), "run_task should return Err on model error");

        let loaded = store.get(&task.id).unwrap().unwrap();
        assert_eq!(loaded.status, TaskStatus::Failed, "task should be Failed");

        // Daemon continues: can enqueue another task.
        let task2 = store
            .enqueue(NewTask {
                agent: "errbot".to_string(),
                goal: "another task".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
            })
            .unwrap();
        assert_eq!(task2.status, TaskStatus::Queued, "next task is queued fine");

        cleanup(&workspace);
    }

    // ── run_task: missing persona → fail with clear message ───────────

    #[tokio::test]
    async fn run_task_missing_persona_fails_clearly() {
        let db = TmpDb::new("rt-missing-persona");
        let workspace = make_temp_dir("rt-missing-ws");
        let store = TaskStore::open(db.path()).unwrap();

        // No persona file for "missing_agent".
        let task = store
            .enqueue(NewTask {
                agent: "missing_agent".to_string(),
                goal: "do something".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        let model = Arc::new(ScriptedModel::new(&["done"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let result = run_task(&store, &task.id, &deps).await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("persona file not found"),
            "error should mention persona: {err_msg}"
        );
        assert!(
            err_msg.contains("missing_agent"),
            "error should name the agent: {err_msg}"
        );

        let loaded = store.get(&task.id).unwrap().unwrap();
        assert_eq!(
            loaded.status,
            TaskStatus::Failed,
            "missing persona → Failed"
        );

        cleanup(&workspace);
    }

    // ── run_task: log file created ────────────────────────────────────

    #[tokio::test]
    async fn run_task_creates_log_file() {
        let db = TmpDb::new("rt-log-file");
        let workspace = make_temp_dir("rt-log-ws");
        let store = TaskStore::open(db.path()).unwrap();

        save_persona(db.data_dir(), "logbot");
        save_permissive_policy(db.data_dir());

        let task = store
            .enqueue(NewTask {
                agent: "logbot".to_string(),
                goal: "log this".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        let model = Arc::new(ScriptedModel::new(&["logged"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        run_task(&store, &task.id, &deps).await.unwrap();

        let log_path = db.data_dir().join("logs").join(format!("{}.log", task.id));
        assert!(log_path.exists());
        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("started"), "log: started marker");
        assert!(content.contains("completed"), "log: completed marker");

        cleanup(&workspace);
    }
}
