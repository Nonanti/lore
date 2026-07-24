//! Task queue daemon: sequential loop that dequeues tasks, runs agents, and
//! records results.
//!
//! The daemon is the sole process that transitions task state (Queued → Running
//! → Completed/Failed). CLI inserts tasks and answers approvals; WAL mode
//! permits concurrent SQLite access. Per-task execution is extracted into
//! [`run_task`] for testing.
//!
//! Team tasks (agent == "pm") are decomposed into child subtasks instead of
//! being run directly. After each child completes/fails, `maybe_complete_parent`
//! checks whether the parent can transition to Completed or Failed.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::agent::{Agent, WorkReport, WorkSpec};
use crate::error::{LoreError, Result};
use crate::memory::{HashingEmbedder, MemoryStore, SqliteStore};
use crate::model::{Model, ModelConfig};
use crate::orchestrator::pm::{
    build_roster, collect_child_reports, decompose_with_retry, has_review_child, has_reviewer,
    synthesis_prompt,
};
use crate::policy::approval::Gate;
use crate::policy::Policy;
use crate::task::approver::QueueApprover;
use crate::task::{NewTask, TaskStatus, TaskStore};
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
/// Uses per-agent model config if the agent record has one, else env fallback.
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

    // Build per-task model: agent record's ModelConfig if present, else env fallback.
    let model = build_per_task_model(&deps.data_dir, &task.agent, &deps.model)?;

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

    let agent = Agent::load_from(&persona_path, mem_store.clone(), model.clone())?;

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
    // Reviewers are report-only by role preset — enforce it in the
    // registry (no write/edit tools; shell stays, still policy-gated).
    let read_only = agent.persona.role == "reviewer";
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(ShellTool::new(
        gate.clone(),
        task.workspace.clone(),
    )));
    if !read_only {
        reg.register(Arc::new(FileWriteTool::new(
            gate.clone(),
            task.workspace.clone(),
        )));
        reg.register(Arc::new(FileEditTool::new(
            gate.clone(),
            task.workspace.clone(),
        )));
    }
    reg.register(Arc::new(FileReadTool::new(
        task.workspace.to_string_lossy().to_string(),
    )));
    let router = Arc::new(LlmRouter::new(model.clone()));
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

    // Memory distillation: extract durable facts from the completed task.
    // Failed tasks also distill, but ONLY negative lessons — distill_work
    // forces every item to the Constraint category on failure (no wrong
    // conventions from failed attempts; gotchas are still learned).
    // Agent opt-out: `should_distill()` returns false when distill field is Some(false).
    // Distillation errors are logged and never fail the task.
    if agent.should_distill() {
        let distilled = agent.distill_work(&spec, &report).await;
        match distilled {
            Ok(n) if n > 0 => {
                tracing::info!(task_id, distilled = n, "distillation completed");
                log.write_line(&format!(
                    "[daemon] task {} distilled {} memories",
                    task.id, n
                ));
            }
            Ok(_) => {
                log.write_line(&format!(
                    "[daemon] task {} distillation: nothing durable found",
                    task.id
                ));
            }
            Err(e) => {
                tracing::warn!(task_id, error = %e, "distillation error — task not affected");
                log.write_line(&format!(
                    "[daemon] task {} distillation error: {e}",
                    task.id
                ));
            }
        }
    } else {
        tracing::info!(task_id, "distillation skipped (agent opted out)");
        log.write_line(&format!(
            "[daemon] task {} distillation skipped: agent opted out",
            task.id
        ));
    }

    // Memory consolidation: merge near-duplicates and forget decayed items.
    // Prevents unbounded growth from distillation (each task adds ≤3 semantic
    // items that only dedup at consolidation time). Run per-task — cheap for
    // small stores and ensures the store stays manageable.
    if let Err(e) = mem_store.consolidate().await {
        tracing::warn!(task_id, error = %e, "consolidation failed — store not affected");
    }

    // Record result in the task store.
    let report_json = serde_json::to_string(&report)?;
    if report.success {
        store.complete(task_id, &report_json)?;
    } else {
        store.fail(task_id, &report_json)?;
    }

    Ok(report)
}

/// Tracks in-flight tasks for same-workspace warnings and concurrency monitoring.
/// Shared across all workers via `Arc<InFlightRegistry>`.
struct InFlightRegistry {
    tasks: std::sync::Mutex<Vec<(String, PathBuf)>>,
    peak: std::sync::atomic::AtomicUsize,
}

impl InFlightRegistry {
    fn new() -> Self {
        Self {
            tasks: std::sync::Mutex::new(Vec::new()),
            peak: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Register a task as in-flight. Returns the id of an existing task
    /// sharing the same workspace (for the `tracing::warn`), or `None`.
    fn add(&self, task_id: &str, workspace: PathBuf) -> Option<String> {
        let mut tasks = self.tasks.lock().unwrap();
        let same_ws = tasks.iter().find(|(_, ws)| ws == &workspace);
        let warning_id = same_ws.map(|(id, _)| id.clone());
        tasks.push((task_id.to_string(), workspace));
        let size = tasks.len();
        // Track peak concurrency.
        let mut p = self.peak.load(std::sync::atomic::Ordering::Relaxed);
        while size > p {
            match self.peak.compare_exchange(
                p,
                size,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(old) => p = old,
            }
        }
        warning_id
    }

    /// Remove a task from the in-flight registry.
    fn remove(&self, task_id: &str) {
        self.tasks.lock().unwrap().retain(|(id, _)| id != task_id);
    }

    /// Peak number of tasks that were in-flight simultaneously.
    #[allow(dead_code)] // Used in tests.
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Single worker loop: claim → run → repeat. Each worker owns its own
/// `TaskStore` connection (WAL permits concurrent access).
async fn worker_loop(
    worker_id: usize,
    data_dir: PathBuf,
    db_path: PathBuf,
    model: Arc<dyn Model>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    in_flight: Arc<InFlightRegistry>,
) -> Result<()> {
    let store = TaskStore::open(&db_path)?;
    let deps = TaskDeps {
        data_dir,
        model,
        db_path,
    };

    tracing::info!(worker_id, "worker started");

    loop {
        // Check shutdown before claiming.
        if *shutdown_rx.borrow() {
            tracing::info!(worker_id, "worker exiting (shutdown, idle)");
            return Ok(());
        }

        // Atomic claim: sets status to Running + returns the task.
        let task = store.claim_next_queued()?;
        if task.is_none() {
            // Wait: either a new task appears or shutdown signal.
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(IDLE_POLL_SECS)) => {},
                _ = shutdown_rx.changed() => {
                    tracing::info!(worker_id, "worker exiting (shutdown while idle)");
                    return Ok(());
                }
            }
            continue;
        }

        let task = task.expect("checked above");
        tracing::info!(worker_id, task_id = %task.id, agent = %task.agent, "claimed task");

        // Register in-flight + same-workspace check.
        if let Some(existing_id) = in_flight.add(&task.id, task.workspace.clone()) {
            tracing::warn!(
                worker_id,
                task_id = %task.id,
                workspace = %task.workspace.display(),
                existing_task_id = %existing_id,
                "same workspace as another in-flight task — proceeding with caution"
            );
        }

        // Team task: PM agent decomposes goal into subtasks.
        let is_team = task.agent == "pm";
        if is_team {
            let result = tokio::select! {
                r = decompose_team_task(&store, &task.id, &deps) => r,
                _ = shutdown_rx.changed() => {
                    tracing::warn!(worker_id, task_id = %task.id, "shutdown mid-run (team) — re-queuing");
                    store.set_status(&task.id, TaskStatus::Queued)?;
                    store.deny_pending_approvals_for_task(&task.id)?;
                    in_flight.remove(&task.id);
                    return Ok(());
                }
            };
            if let Err(e) = &result {
                tracing::error!(worker_id, task_id = %task.id, error = %e, "PM decomposition failed");
            }
        } else {
            // Regular task.
            let result = tokio::select! {
                r = run_task(&store, &task.id, &deps) => r,
                _ = shutdown_rx.changed() => {
                    tracing::warn!(worker_id, task_id = %task.id, "shutdown mid-run — re-queuing");
                    store.set_status(&task.id, TaskStatus::Queued)?;
                    store.deny_pending_approvals_for_task(&task.id)?;
                    in_flight.remove(&task.id);
                    return Ok(());
                }
            };

            // Remove from in-flight registry.
            in_flight.remove(&task.id);

            match result {
                Ok(report) => {
                    tracing::info!(
                        worker_id,
                        task_id = %task.id,
                        success = report.success,
                        iterations = report.iterations,
                        "task finished"
                    );
                }
                Err(e) => {
                    // run_task already records failures; catch residual.
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
                    tracing::error!(worker_id, task_id = %task.id, error = %e, "task failed — worker continues");
                }
            }
        }

        // After any task completion/failure, check parent.
        if let Ok(completed) = maybe_complete_parent(&store, &task.id, &deps).await {
            if completed {
                tracing::info!(worker_id, child_task_id = %task.id, "parent task finalized");
            }
        }
    }
}

/// Daemon entry: spawns N worker loops, broadcasts shutdown via
/// `tokio::sync::watch`, and joins all workers on ctrl_c.
///
/// `concurrency` is clamped to 1..=8. Recovery sweeps run once at
/// startup before workers spawn.
pub async fn run_daemon(data_dir: &Path, db_path: &Path, concurrency: usize) -> Result<()> {
    let concurrency = concurrency.clamp(1, 8);
    tracing::info!(
        concurrency,
        "daemon starting — data: {}",
        data_dir.display()
    );

    // Startup: open store for recovery sweeps.
    let store = TaskStore::open(db_path)?;

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

    let model = build_model_from_env(data_dir)?;

    let deps = TaskDeps {
        data_dir: data_dir.to_path_buf(),
        model: model.clone(),
        db_path: db_path.to_path_buf(),
    };

    // C-1 recovery: finalize stuck WaitingSubtasks parents whose children
    // are all terminal (crash left parent waiting with no active children).
    let _stuck_recovered = recover_stuck_parents(&store, &deps).await?;

    // Shutdown channel: ctrl_c sets true, workers check via watch::Receiver.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // In-flight registry for same-workspace warnings.
    let in_flight = Arc::new(InFlightRegistry::new());

    // Spawn workers (each owns its own TaskStore connection).
    let mut worker_threads = Vec::new();
    for i in 0..concurrency {
        let rx = shutdown_rx.clone();
        let inf = in_flight.clone();
        let m = model.clone();
        let dd = data_dir.to_path_buf();
        let dp = db_path.to_path_buf();
        worker_threads.push(std::thread::spawn(move || {
            let worker_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("worker runtime init");
            let local = tokio::task::LocalSet::new();
            local.block_on(&worker_rt, worker_loop(i, dd, dp, m, rx, inf))
        }));
    }

    // Wait for ctrl_c → broadcast shutdown → join all workers.
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutdown signal received — finishing current tasks");
    shutdown_tx.send(true).ok();

    for t in worker_threads {
        match t.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::error!(error = %e, "worker error"),
            Err(_) => tracing::error!("worker thread panicked"),
        }
    }

    tracing::info!("daemon stopped");
    Ok(())
}

/// Build per-task model: if the agent record has a ModelConfig, use it;
/// otherwise fall back to env config.
fn build_per_task_model(
    data_dir: &Path,
    agent_name: &str,
    env_model: &Arc<dyn Model>,
) -> Result<Arc<dyn Model>> {
    let persona_path = data_dir.join("agents").join(format!("{agent_name}.json"));
    if persona_path.exists() {
        let json = std::fs::read_to_string(&persona_path)
            .map_err(|e| LoreError::Storage(e.to_string()))?;
        let rec: serde_json::Value = serde_json::from_str(&json)?;
        if let Some(model_cfg) = rec.get("model") {
            let cfg: ModelConfig = serde_json::from_value(model_cfg.clone())?;
            return crate::model::build_model(&cfg, data_dir);
        }
    }
    // No per-agent model config → env fallback.
    Ok(env_model.clone())
}

/// Build the model from env config (centralized in `lore::model::factory`).
/// Returns `Err` when auth is explicitly configured but credentials are absent
/// (M-1: no silent MockModel fallback).
fn build_model_from_env(data_dir: &Path) -> Result<Arc<dyn Model>> {
    crate::model::build_model_from_env(data_dir)
}

/// Decompose a PM team task: prompt the PM agent model with the goal + roster,
/// enqueue children, set parent to WaitingSubtasks.
pub async fn decompose_team_task(store: &TaskStore, task_id: &str, deps: &TaskDeps) -> Result<()> {
    let task = store
        .get(task_id)?
        .ok_or_else(|| LoreError::NotFound(format!("task {task_id}")))?;

    // Idempotency guard (C-3): if children already exist from a prior
    // (crash-interrupted) decomposition, skip re-decomposing and just
    // transition to WaitingSubtasks.
    if !store.children_of(task_id)?.is_empty() {
        tracing::info!(
            task_id,
            "skipping re-decomposition — children already exist from prior attempt"
        );
        store.set_status(task_id, TaskStatus::WaitingSubtasks)?;
        return Ok(());
    }

    let pm_model = build_per_task_model(&deps.data_dir, &task.agent, &deps.model)?;
    let roster = build_roster(&deps.data_dir)?;

    if roster.is_empty() {
        let msg = "PM decomposition failed: no agents in roster";
        tracing::error!(task_id, "{msg}");
        let report_json = serde_json::to_string(&WorkReport {
            success: false,
            iterations: 0,
            answer: msg.to_string(),
            verify_log: String::new(),
        })?;
        store.fail(task_id, &report_json)?;
        return Err(LoreError::InvalidInput(msg.to_string()));
    }

    let specs = decompose_with_retry(&pm_model, &task.goal, &roster).await;
    match specs {
        Ok(subtasks) => {
            for spec in &subtasks {
                let child = NewTask {
                    agent: spec.agent.clone(),
                    goal: spec.goal.clone(),
                    workspace: task.workspace.clone(),
                    verify: spec.verify.clone(),
                    parent_id: None, // enqueue_child sets this
                };
                store.enqueue_child(task_id, child)?;
                tracing::info!(
                    parent_id = task_id,
                    child_agent = %spec.agent,
                    child_goal = %spec.goal,
                    "enqueued child subtask"
                );
            }
            store.set_status(task_id, TaskStatus::WaitingSubtasks)?;
            tracing::info!(
                parent_id = task_id,
                children_count = subtasks.len(),
                "PM task decomposed, waiting for subtasks"
            );
            Ok(())
        }
        Err(e) => {
            let msg = format!("PM decomposition failed: {e}");
            tracing::error!(task_id, error = %e, "{msg}");
            let report_json = serde_json::to_string(&WorkReport {
                success: false,
                iterations: 0,
                answer: msg.clone(),
                verify_log: String::new(),
            })?;
            store.fail(task_id, &report_json)?;
            Err(LoreError::InvalidInput(msg))
        }
    }
}

/// After a child task completes/fails, check if the parent can transition.
/// Returns true if the parent was finalized (Completed or Failed).
pub async fn maybe_complete_parent(
    store: &TaskStore,
    child_task_id: &str,
    deps: &TaskDeps,
) -> Result<bool> {
    let child = store
        .get(child_task_id)?
        .ok_or_else(|| LoreError::NotFound(format!("task {child_task_id}")))?;

    let parent_id = match child.parent_id {
        Some(pid) => pid,
        None => return Ok(false), // No parent — standalone task.
    };

    finalize_parent_if_ready(store, &parent_id, deps).await
}

/// Check if a `WaitingSubtasks` parent can be finalized (Completed or Failed).
/// Core logic extracted from `maybe_complete_parent` so the startup sweep
/// can call it directly by parent_id (C-1 recovery).
async fn finalize_parent_if_ready(
    store: &TaskStore,
    parent_id: &str,
    deps: &TaskDeps,
) -> Result<bool> {
    // Check parent exists and is still WaitingSubtasks.
    let parent = store.get(parent_id)?;
    if parent.is_none_or(|p| p.status != TaskStatus::WaitingSubtasks) {
        return Ok(false); // Parent already finalized or not waiting.
    }

    // Not all children done yet? Wait.
    if !store.all_children_done(parent_id)? {
        return Ok(false);
    }

    let children = store.children_of(parent_id)?;

    // Edge case: no children at all → fail parent (decomposition should have
    // either created children or already failed the parent).
    if children.is_empty() {
        let msg = "PM task stuck in WaitingSubtasks with no children";
        tracing::error!(parent_id, "{msg}");
        let report_json = serde_json::to_string(&WorkReport {
            success: false,
            iterations: 0,
            answer: msg.to_string(),
            verify_log: String::new(),
        })?;
        store.fail(parent_id, &report_json)?;
        return Ok(true);
    }

    // If any child Failed (other than the reviewer) → parent Failed.
    let non_review_failures = children
        .iter()
        .filter(|c| c.status == TaskStatus::Failed && c.agent != "reviewer")
        .count();
    if non_review_failures > 0 {
        let reports = collect_child_reports(store, parent_id)?;
        let failing_reports: Vec<String> = reports
            .iter()
            .filter(|r| r.status == "Failed" && r.agent != "reviewer")
            .map(|r| format!("Agent {} (goal: {}): {}", r.agent, r.goal, r.report))
            .collect();
        let msg = format!(
            "Team task failed — child subtask(s) failed:\n{}",
            failing_reports.join("\n")
        );
        let report_json = serde_json::to_string(&WorkReport {
            success: false,
            iterations: 0,
            answer: msg,
            verify_log: String::new(),
        })?;
        store.fail(parent_id, &report_json)?;
        tracing::info!(
            parent_id,
            failed_children = non_review_failures,
            "parent task Failed (child failure propagated)"
        );
        return Ok(true);
    }

    // All children succeeded. Check if a reviewer should be enqueued.
    let roster = build_roster(&deps.data_dir)?;
    if has_reviewer(&roster) && !has_review_child(store, parent_id)? {
        // Enqueue ONE review child.
        let children_reports = collect_child_reports(store, parent_id)?;
        let review_goal = format!(
            "Review the completed work:\n{}\nLook for gaps, contradictions, missing verification.",
            synthesis_prompt(&children_reports)
        );
        let review_task = NewTask {
            agent: "reviewer".to_string(),
            goal: review_goal,
            workspace: children
                .first()
                .map(|c| c.workspace.clone())
                .unwrap_or_else(std::env::temp_dir),
            verify: vec![],
            parent_id: None, // enqueue_child sets this
        };
        store.enqueue_child(parent_id, review_task)?;
        tracing::info!(
            parent_id,
            "review child enqueued, parent stays WaitingSubtasks"
        );
        return Ok(false); // Parent still waiting (for the review child).
    }

    // All children done + review done (or no reviewer) → PM synthesis.
    let pm_model = build_per_task_model(&deps.data_dir, "pm", &deps.model)?;
    let children_reports = collect_child_reports(store, parent_id)?;
    let synth_text = synthesis_prompt(&children_reports);

    let prompt = crate::model::Prompt {
        system: "You are a project manager synthesizing subtask results into a combined report."
            .to_string(),
        user: synth_text,
        ..Default::default()
    };

    // C-2: catch PM synthesis model failure → fail parent (not wedge).
    let completion = match pm_model.complete(&prompt).await {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("PM synthesis model call failed: {e}");
            tracing::error!(parent_id, error = %e, "{msg}");
            let report_json = serde_json::to_string(&WorkReport {
                success: false,
                iterations: 0,
                answer: msg.clone(),
                verify_log: String::new(),
            })?;
            store.fail(parent_id, &report_json)?;
            tracing::info!(parent_id, "parent task Failed (PM synthesis error)");
            return Ok(true);
        }
    };
    let combined_report = serde_json::json!({
        "success": true,
        "iterations": children.len(),
        "answer": completion.text,
        "children_count": children.len(),
    });
    let report_json = serde_json::to_string(&combined_report)?;
    store.complete(parent_id, &report_json)?;
    tracing::info!(
        parent_id,
        children_count = children.len(),
        "parent task Completed via PM synthesis"
    );
    Ok(true)
}

/// Startup sweep: find `WaitingSubtasks` parents whose children are all
/// terminal and finalize them. Recovers from the crash scenario where
/// all children finished but `maybe_complete_parent` never ran (C-1).
async fn recover_stuck_parents(store: &TaskStore, deps: &TaskDeps) -> Result<usize> {
    let stuck = store.waiting_subtasks_tasks()?;
    let mut finalized = 0;
    for parent_id in &stuck {
        match finalize_parent_if_ready(store, parent_id, deps).await {
            Ok(true) => {
                tracing::info!(
                    parent_id,
                    "stuck WaitingSubtasks parent finalized on startup"
                );
                finalized += 1;
            }
            Ok(false) => {} // Not ready yet (children still active).
            Err(e) => {
                tracing::error!(
                    parent_id,
                    error = %e,
                    "failed to finalize stuck parent on startup"
                );
            }
        }
    }
    if finalized > 0 {
        tracing::info!(
            count = finalized,
            "finalized stuck WaitingSubtasks parents on startup"
        );
    }
    Ok(finalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, Persona};
    use crate::memory::InMemoryStore;
    use crate::model::{Completion, MockModel, Model, Prompt};
    use crate::policy::SandboxMode;
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
            sandbox_exec: SandboxMode::Off,
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
                parent_id: None,
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
                parent_id: None,
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
                parent_id: None,
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
                parent_id: None,
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
                parent_id: None,
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

    // ── maybe_complete_parent: all-success + no reviewer ──────────────

    #[tokio::test]
    async fn maybe_complete_parent_all_success_no_reviewer() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("mcp-success-no-review");

        // Parent PM task.
        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        // Children: both Completed.
        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();
        let c2 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "frontend".to_string(),
                    goal: "build UI".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();

        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"API done"}))
                    .unwrap(),
            )
            .unwrap();
        store
            .complete(
                &c2.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"UI done"}))
                    .unwrap(),
            )
            .unwrap();

        // No reviewer persona in data dir.
        let model = Arc::new(ScriptedModel::new(&["synthesized: all good"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let completed = maybe_complete_parent(&store, &c2.id, &deps).await.unwrap();
        assert!(completed, "parent should be finalized");

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(parent_loaded.status, TaskStatus::Completed);
    }

    // ── maybe_complete_parent: all-success + reviewer persona → review child enqueued ──

    #[tokio::test]
    async fn maybe_complete_parent_all_success_with_reviewer_enqueues_review() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("mcp-review");

        // Create reviewer persona file.
        let agents_dir = db.data_dir().join("agents");
        std::fs::create_dir_all(&agents_dir).unwrap();
        let persona = Persona::new("reviewer", "code reviewer");
        let mem_store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(persona, mem_store, Arc::new(MockModel::new()));
        agent.save_to(agents_dir.join("reviewer.json")).unwrap();

        // Also create pm persona for the synthesis model.
        let pm_persona = Persona::new("pm", "project manager");
        let pm_mem: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let pm_agent = Agent::new(pm_persona, pm_mem, Arc::new(MockModel::new()));
        pm_agent.save_to(agents_dir.join("pm.json")).unwrap();

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"done"}))
                    .unwrap(),
            )
            .unwrap();

        let model = Arc::new(MockModel::new());
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        // First call: review child enqueued, parent stays WaitingSubtasks.
        let completed = maybe_complete_parent(&store, &c1.id, &deps).await.unwrap();
        assert!(
            !completed,
            "parent not finalized yet — review child enqueued"
        );

        let children = store.children_of(&parent.id).unwrap();
        let review_children = children.iter().filter(|c| c.agent == "reviewer").count();
        assert_eq!(review_children, 1, "exactly one review child enqueued");

        // Second call: review already enqueued — does NOT create another.
        let completed2 = maybe_complete_parent(&store, &c1.id, &deps).await.unwrap();
        assert!(
            !completed2,
            "parent not finalized — review still in progress"
        );

        let children2 = store.children_of(&parent.id).unwrap();
        let review_children2 = children2.iter().filter(|c| c.agent == "reviewer").count();
        assert_eq!(
            review_children2, 1,
            "still exactly one review child — no duplicate"
        );

        // Now complete the review child → parent completes via synthesis.
        let review_child = children2.iter().find(|c| c.agent == "reviewer").unwrap();
        store
            .complete(
                &review_child.id,
                &serde_json::to_string(
                    &serde_json::json!({"success":true,"answer":"review looks good"}),
                )
                .unwrap(),
            )
            .unwrap();

        let synth_model = Arc::new(ScriptedModel::new(&["synthesized: all good with review"]));
        let synth_deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model: synth_model,
            db_path: db.path().to_path_buf(),
        };

        let completed3 = maybe_complete_parent(&store, &review_child.id, &synth_deps)
            .await
            .unwrap();
        assert!(completed3, "parent finalized after review");

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(parent_loaded.status, TaskStatus::Completed);
    }

    // ── maybe_complete_parent: child Failed → parent Failed ──────────

    #[tokio::test]
    async fn maybe_complete_parent_child_failed_propagates() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("mcp-fail");

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();
        let c2 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "frontend".to_string(),
                    goal: "build UI".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();

        // c1 succeeded, c2 failed.
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"done"}))
                    .unwrap(),
            )
            .unwrap();
        store
            .fail(
                &c2.id,
                &serde_json::to_string(&serde_json::json!({"success":false,"answer":"UI broken"}))
                    .unwrap(),
            )
            .unwrap();

        let model = Arc::new(MockModel::new());
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let completed = maybe_complete_parent(&store, &c2.id, &deps).await.unwrap();
        assert!(completed, "parent should be Failed");

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(parent_loaded.status, TaskStatus::Failed);
    }

    // ── Recovery interplay: orphaned child re-queued → parent completes ──

    #[tokio::test]
    async fn recovery_orphaned_child_parent_completes_later() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("mcp-recovery");

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();

        // Simulate crash: child was Running, now orphaned → re-queued.
        store.set_status(&c1.id, TaskStatus::Running).unwrap();
        store.recover_orphaned().unwrap();
        // Child is now Queued again.
        let c1_loaded = store.get(&c1.id).unwrap().unwrap();
        assert_eq!(c1_loaded.status, TaskStatus::Queued);

        // Parent still waiting — child not yet done.
        assert!(!store.all_children_done(&parent.id).unwrap());

        // Child completes later (manual sim).
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"done"}))
                    .unwrap(),
            )
            .unwrap();

        let model = Arc::new(ScriptedModel::new(&["synthesized: done"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let completed = maybe_complete_parent(&store, &c1.id, &deps).await.unwrap();
        assert!(completed, "parent should finalize after child completes");

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(parent_loaded.status, TaskStatus::Completed);
    }

    // ── Review child itself fails → parent still completes ──────────
    // Reviewer failure is NOT treated as a regular child failure (reviewer
    // is excluded from the "non-review failure" filter). After the review
    // child fails, all children are done, reviewer exists but review child
    // already enqueued → no duplicate, and the parent proceeds to synthesis.

    #[tokio::test]
    async fn maybe_complete_parent_review_child_fails_parent_completes() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("mcp-review-fail");

        // Create reviewer persona file so roster has reviewer.
        let agents_dir = db.data_dir().join("agents");
        std::fs::create_dir_all(&agents_dir).unwrap();
        let persona = Persona::new("reviewer", "code reviewer");
        let mem_store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(persona, mem_store, Arc::new(MockModel::new()));
        agent.save_to(agents_dir.join("reviewer.json")).unwrap();

        // Also create pm persona for synthesis.
        let pm_persona = Persona::new("pm", "project manager");
        let pm_mem: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let pm_agent = Agent::new(pm_persona, pm_mem, Arc::new(MockModel::new()));
        pm_agent.save_to(agents_dir.join("pm.json")).unwrap();

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        // Worker child succeeds.
        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"API done"}))
                    .unwrap(),
            )
            .unwrap();

        let model = Arc::new(MockModel::new());
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        // First maybe_complete_parent: reviewer exists → review child enqueued.
        let completed1 = maybe_complete_parent(&store, &c1.id, &deps).await.unwrap();
        assert!(
            !completed1,
            "review child enqueued, parent stays WaitingSubtasks"
        );

        let children = store.children_of(&parent.id).unwrap();
        let review_child = children.iter().find(|c| c.agent == "reviewer").unwrap();

        // Review child itself FAILS — reviewer failure is NOT a regular child failure.
        store
            .fail(
                &review_child.id,
                &serde_json::to_string(
                    &serde_json::json!({"success":false,"answer":"review crashed"}),
                )
                .unwrap(),
            )
            .unwrap();

        // Second maybe_complete_parent: all children done, review child failed
        // but reviewer failures don't propagate to parent. Parent should still
        // complete via synthesis (has_review_child is true, so no duplicate review).
        let synth_model = Arc::new(ScriptedModel::new(&[
            "synthesized: accepted despite review failure",
        ]));
        let synth_deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model: synth_model,
            db_path: db.path().to_path_buf(),
        };

        let completed2 = maybe_complete_parent(&store, &review_child.id, &synth_deps)
            .await
            .unwrap();
        assert!(
            completed2,
            "parent should complete even though review child failed"
        );

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(
            parent_loaded.status,
            TaskStatus::Completed,
            "reviewer failure does not wedge the parent"
        );
    }

    // ── decompose_team_task with empty roster → fail ────────────────────

    #[tokio::test]
    async fn decompose_team_task_empty_roster_fails() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("dt-empty-roster");

        // Parent PM task.
        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store.set_status(&parent.id, TaskStatus::Running).unwrap();

        // No agents dir → empty roster → decompose_team_task fails.
        let model = Arc::new(ScriptedModel::new(&["should not be called"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let result = decompose_team_task(&store, &parent.id, &deps).await;
        assert!(result.is_err(), "empty roster should fail decompose");

        // Parent task is marked Failed.
        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(parent_loaded.status, TaskStatus::Failed);
    }

    // ── C-1: stuck WaitingSubtasks parent recovered on startup ────────

    #[tokio::test]
    async fn recover_stuck_parents_finalizes_on_startup() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("c1-stuck-parent");

        // Create pm persona (no reviewer).
        save_persona(db.data_dir(), "pm");

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();

        // Simulate crash: all children already terminal, parent stuck in WaitingSubtasks.
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"API done"}))
                    .unwrap(),
            )
            .unwrap();

        // Startup sweep: recover stuck parents.
        let model = Arc::new(ScriptedModel::new(&["synthesized: done"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let recovered = recover_stuck_parents(&store, &deps).await.unwrap();
        assert_eq!(recovered, 1, "one stuck parent finalized");

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(
            parent_loaded.status,
            TaskStatus::Completed,
            "stuck parent recovered on startup"
        );
    }

    // ── C-2: PM synthesis error → parent Failed (not wedged) ──────────────

    #[tokio::test]
    async fn finalize_parent_synthesis_error_fails_parent() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("c2-synth-err");

        // Create pm persona (no reviewer).
        save_persona(db.data_dir(), "pm");

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store
            .set_status(&parent.id, TaskStatus::WaitingSubtasks)
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"API done"}))
                    .unwrap(),
            )
            .unwrap();

        // Use ErrorModel so PM synthesis will fail.
        let model = Arc::new(ErrorModel("synthesis failed: network error".into()));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let completed = maybe_complete_parent(&store, &c1.id, &deps).await.unwrap();
        assert!(
            completed,
            "parent should be finalized even on synthesis error"
        );

        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(
            parent_loaded.status,
            TaskStatus::Failed,
            "synthesis error → parent Failed"
        );

        let report: serde_json::Value =
            serde_json::from_str(parent_loaded.report.as_deref().unwrap()).unwrap();
        assert!(
            report["answer"].as_str().unwrap().contains("synthesis"),
            "report should mention synthesis failure"
        );
    }

    // ── C-3: idempotency guard — duplicate children skipped ────────────────

    #[tokio::test]
    async fn decompose_team_task_skips_if_children_exist() {
        let store = TaskStore::in_memory().unwrap();
        let db = TmpDb::new("c3-idempotent");

        // Create pm + backend personas so roster is non-empty.
        save_persona(db.data_dir(), "pm");
        save_persona(db.data_dir(), "backend");

        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        store.set_status(&parent.id, TaskStatus::Running).unwrap();

        // Pre-enqueue a child manually (simulating crash before set_status).
        let existing_child = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "impl API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec![],
                    parent_id: None,
                },
            )
            .unwrap();

        let model = Arc::new(ScriptedModel::new(&["should not be called"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        // decompose_team_task should skip re-decomposing (children already exist).
        let result = decompose_team_task(&store, &parent.id, &deps).await;
        assert!(result.is_ok(), "idempotent decompose should succeed");

        // Parent is now WaitingSubtasks, no duplicate children.
        let parent_loaded = store.get(&parent.id).unwrap().unwrap();
        assert_eq!(
            parent_loaded.status,
            TaskStatus::WaitingSubtasks,
            "parent transitions to WaitingSubtasks"
        );

        let children = store.children_of(&parent.id).unwrap();
        assert_eq!(
            children.len(),
            1,
            "no duplicate children — only the pre-existing one"
        );
        assert_eq!(children[0].id, existing_child.id);
    }

    // ── daemon distill opt-out: agent with distill=false skips distillation ──

    #[tokio::test]
    async fn run_task_distill_opt_out_respected() {
        let db = TmpDb::new("rt-distill-optout");
        let workspace = make_temp_dir("rt-distill-optout-ws");
        let store = TaskStore::open(db.path()).unwrap();

        // Create persona file for agent "nodistill" with distill=false.
        let agents_dir = db.data_dir().join("agents");
        std::fs::create_dir_all(&agents_dir).unwrap();
        let persona = Persona::new("nodistill", "worker");
        let mem: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(persona, mem, Arc::new(MockModel::new())).with_distill(false);
        agent.save_to(agents_dir.join("nodistill.json")).unwrap();
        // Verify the saved JSON includes distill=false.
        let saved_json = std::fs::read_to_string(agents_dir.join("nodistill.json")).unwrap();
        let val: serde_json::Value = serde_json::from_str(&saved_json).unwrap();
        assert_eq!(val["distill"], serde_json::Value::Bool(false));

        save_permissive_policy(db.data_dir());

        let task = store
            .enqueue(NewTask {
                agent: "nodistill".to_string(),
                goal: "simple task".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
                parent_id: None,
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        // Model: only 1 reply needed (work solve; distill should NOT be called).
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let report = run_task(&store, &task.id, &deps).await.unwrap();
        assert!(report.success);

        // Log file should contain "distillation skipped".
        let log_path = db.data_dir().join("logs").join(format!("{}.log", task.id));
        let log_content = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            log_content.contains("distillation skipped: agent opted out"),
            "log should mention distillation skipped: {log_content}"
        );

        cleanup(&workspace);
    }

    // ── daemon distillation: failed tasks skip distillation ──

    #[tokio::test]
    async fn run_task_failed_task_distills_negative_lessons() {
        let db = TmpDb::new("rt-distill-skip-fail");
        let workspace = make_temp_dir("rt-distill-skip-fail-ws");
        let store = TaskStore::open(db.path()).unwrap();

        save_persona(db.data_dir(), "failbot");
        save_permissive_policy(db.data_dir());

        // Verify command always fails (exit 1) → task will exhaust iterations and fail.
        let task = store
            .enqueue(NewTask {
                agent: "failbot".to_string(),
                goal: "impossible task".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 1".to_string()],
                parent_id: None,
            })
            .unwrap();
        store.set_status(&task.id, TaskStatus::Running).unwrap();

        // Model provides 5 replies for 5 iterations (default max_iterations)
        // plus a 6th reply for the distillation call: one negative lesson.
        let model = Arc::new(ScriptedModel::new(&[
            "trying 1",
            "trying 2",
            "trying 3",
            "trying 4",
            "trying 5",
            r#"[{"kind":"constraint","title":"avoid `exit 1` as a verify command","body":"it always fails"}]"#,
        ]));
        let deps = TaskDeps {
            data_dir: db.data_dir().to_path_buf(),
            model,
            db_path: db.path().to_path_buf(),
        };

        let report = run_task(&store, &task.id, &deps).await.unwrap();
        assert!(
            !report.success,
            "task should fail when verify always exits 1"
        );

        // Failed tasks no longer skip distillation — they distill
        // negative lessons (constraints) instead.
        let log_path = db.data_dir().join("logs").join(format!("{}.log", task.id));
        let log_content = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            log_content.contains("distilled 1 memories"),
            "log should record failure-lesson distillation: {log_content}"
        );

        cleanup(&workspace);
    }

    // ── Parallel execution: concurrency=3 overlap ────────────────────

    /// Model that tracks concurrent calls and sleeps briefly to force overlap.
    struct CountingModel {
        concurrent: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        delay: Duration,
    }

    impl CountingModel {
        fn new(
            concurrent: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            delay: Duration,
        ) -> Self {
            Self {
                concurrent,
                peak,
                delay,
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for CountingModel {
        async fn complete(&self, _p: &Prompt) -> crate::error::Result<Completion> {
            let c = self
                .concurrent
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            let mut p = self.peak.load(std::sync::atomic::Ordering::SeqCst);
            while c > p {
                match self.peak.compare_exchange(
                    p,
                    c,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(old) => p = old,
                }
            }
            tokio::time::sleep(self.delay).await;
            self.concurrent
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Completion::new(
                "Task completed successfully. All verification commands passed.",
            ))
        }
    }

    #[tokio::test]
    async fn concurrency_3_processes_3_tasks_with_overlap() {
        let db = TmpDb::new("parallel-3");
        let workspace = make_temp_dir("parallel-3-ws");

        save_persona(db.data_dir(), "parbot");
        save_permissive_policy(db.data_dir());

        let store = TaskStore::open(db.path()).unwrap();

        let concurrent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Delay ensures overlap: 3 workers, 200ms per model call.
        let model = std::sync::Arc::new(CountingModel::new(
            concurrent.clone(),
            peak.clone(),
            Duration::from_millis(200),
        ));

        // Enqueue 3 tasks.
        for i in 0..3 {
            store
                .enqueue(NewTask {
                    agent: "parbot".to_string(),
                    goal: format!("task {i}"),
                    workspace: workspace.clone(),
                    verify: vec!["exit 0".to_string()],
                    parent_id: None,
                })
                .unwrap();
        }

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let in_flight = std::sync::Arc::new(InFlightRegistry::new());

        let mut worker_threads = Vec::new();
        for i in 0..3 {
            let rx = shutdown_rx.clone();
            let inf = in_flight.clone();
            let m = model.clone();
            let dd = db.data_dir().to_path_buf();
            let dp = db.path().to_path_buf();
            worker_threads.push(std::thread::spawn(move || {
                let worker_rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("worker runtime init");
                let local = tokio::task::LocalSet::new();
                local.block_on(&worker_rt, worker_loop(i, dd, dp, m, rx, inf))
            }));
        }

        // Wait for all tasks to complete (generous timeout).
        let start = std::time::Instant::now();
        loop {
            let tasks = store.list(100).unwrap();
            let all_done = tasks
                .iter()
                .all(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Failed);
            if all_done {
                break;
            }
            if start.elapsed() > Duration::from_secs(30) {
                panic!("timeout waiting for tasks to complete");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Verify all completed.
        let tasks = store.list(100).unwrap();
        assert_eq!(tasks.len(), 3);
        for t in &tasks {
            assert_eq!(
                t.status,
                TaskStatus::Completed,
                "task {} should be Completed",
                t.id
            );
        }

        // Verify overlap: peak concurrent model calls >= 2.
        // With 3 workers and 200ms delay, at least 2 overlap is expected.
        let peak_val = peak.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            peak_val >= 2,
            "at least 2 tasks ran concurrently (peak={peak_val})"
        );

        // Shutdown workers.
        shutdown_tx.send(true).unwrap();
        for t in worker_threads {
            t.join().unwrap().unwrap();
        }

        cleanup(&workspace);
    }

    // ── Shutdown: watch channel re-queues in-flight ──────────────────────

    /// Model that blocks until a signal is sent (simulates long-running task).
    struct BlockUntilSignalModel {
        unblock: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait::async_trait]
    impl Model for BlockUntilSignalModel {
        async fn complete(&self, _p: &Prompt) -> crate::error::Result<Completion> {
            while !self.unblock.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Ok(Completion::new("done after signal"))
        }
    }

    #[tokio::test]
    async fn shutdown_watch_requeues_inflight_and_workers_join() {
        let db = TmpDb::new("shutdown-requeue");
        let workspace = make_temp_dir("shutdown-requeue-ws");

        save_persona(db.data_dir(), "blockbot");
        save_permissive_policy(db.data_dir());

        let store = TaskStore::open(db.path()).unwrap();

        let unblock = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let model = std::sync::Arc::new(BlockUntilSignalModel {
            unblock: unblock.clone(),
        });

        // Enqueue one task that blocks until we unblock it.
        let task = store
            .enqueue(NewTask {
                agent: "blockbot".to_string(),
                goal: "blocking task".to_string(),
                workspace: workspace.clone(),
                verify: vec!["exit 0".to_string()],
                parent_id: None,
            })
            .unwrap();

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let in_flight = std::sync::Arc::new(InFlightRegistry::new());

        let rx = shutdown_rx.clone();
        let inf = in_flight.clone();
        let m = model.clone();
        let dd = db.data_dir().to_path_buf();
        let dp = db.path().to_path_buf();
        let worker_thread = std::thread::spawn(move || {
            let worker_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("worker runtime init");
            let local = tokio::task::LocalSet::new();
            local.block_on(&worker_rt, worker_loop(0, dd, dp, m, rx, inf))
        });

        // Wait for the task to be claimed (task status → Running).
        let start = std::time::Instant::now();
        loop {
            let loaded = store.get(&task.id).unwrap().unwrap();
            if loaded.status == TaskStatus::Running {
                break;
            }
            if start.elapsed() > Duration::from_secs(10) {
                panic!("timeout waiting for task to be claimed");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Task is now in-flight and blocked. Send shutdown signal.
        shutdown_tx.send(true).unwrap();

        // Wait briefly for the worker to re-queue the task.
        let start = std::time::Instant::now();
        loop {
            let loaded = store.get(&task.id).unwrap().unwrap();
            if loaded.status == TaskStatus::Queued {
                break;
            }
            if start.elapsed() > Duration::from_secs(10) {
                panic!("timeout waiting for task to be re-queued");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Verify worker thread joined successfully.
        let result = worker_thread.join().unwrap();
        assert!(result.is_ok(), "worker should exit cleanly");

        // Task is now Queued again (re-queued on shutdown).
        let loaded = store.get(&task.id).unwrap().unwrap();
        assert_eq!(
            loaded.status,
            TaskStatus::Queued,
            "task re-queued after shutdown"
        );

        // Unblocking the model is not needed since the worker exited
        // (the tokio::select! dropped the run_task future).
        unblock.store(true, std::sync::atomic::Ordering::SeqCst);

        cleanup(&workspace);
    }

    // ── InFlightRegistry: add / remove / peak / same-workspace ───────────

    #[test]
    fn in_flight_registry_add_remove_peak_same_workspace() {
        let registry = InFlightRegistry::new();
        let ws_a = PathBuf::from("/tmp/ws-a");
        let ws_b = PathBuf::from("/tmp/ws-b");

        // Add first task → no same-workspace warning.
        let warning = registry.add("t1", ws_a.clone());
        assert!(warning.is_none(), "no same-workspace for first task");
        assert_eq!(registry.peak(), 1);

        // Add second task with same workspace → warning fires.
        let warning = registry.add("t2", ws_a.clone());
        assert_eq!(
            warning,
            Some("t1".to_string()),
            "same workspace warning returns first task id"
        );
        assert_eq!(registry.peak(), 2);

        // Add third task with different workspace → no warning.
        let warning = registry.add("t3", ws_b.clone());
        assert!(warning.is_none(), "different workspace, no warning");
        assert_eq!(registry.peak(), 3);

        // Remove t2 → peak stays at 3 (peak never decreases).
        registry.remove("t2");
        assert_eq!(registry.peak(), 3, "peak stays at historical max");

        // Add another task to ws_a → warning about t1 (t2 was removed).
        let warning = registry.add("t4", ws_a.clone());
        assert_eq!(
            warning,
            Some("t1".to_string()),
            "warning still fires for remaining same-ws task"
        );

        // Remove all → peak unchanged.
        registry.remove("t1");
        registry.remove("t3");
        registry.remove("t4");
        assert_eq!(registry.peak(), 3, "peak unchanged after all removed");

        // Re-add after empty → peak updates to 1.
        // Actually peak=3 > 1, so peak stays at 3.
        registry.add("t5", ws_b.clone());
        assert_eq!(
            registry.peak(),
            3,
            "peak stays at historical max even after re-add"
        );
    }

    #[test]
    fn in_flight_registry_remove_nonexistent_is_noop() {
        let registry = InFlightRegistry::new();
        registry.remove("nonexistent");
        // No panic, no error.
        assert_eq!(registry.peak(), 0);
    }

    // ── concurrency=1: sequential FIFO (same as old daemon loop) ────────

    #[tokio::test]
    async fn concurrency_1_processes_tasks_fifo_like_sequential_loop() {
        let db = TmpDb::new("seq-1");
        let workspace = make_temp_dir("seq-1-ws");

        save_persona(db.data_dir(), "seqbot");
        save_permissive_policy(db.data_dir());

        let store = TaskStore::open(db.path()).unwrap();

        // Enqueue 3 tasks.
        let t1 = store
            .enqueue(NewTask {
                agent: "seqbot".to_string(),
                goal: "first task".to_string(),
                workspace: workspace.clone(),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        let t2 = store
            .enqueue(NewTask {
                agent: "seqbot".to_string(),
                goal: "second task".to_string(),
                workspace: workspace.clone(),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();
        let t3 = store
            .enqueue(NewTask {
                agent: "seqbot".to_string(),
                goal: "third task".to_string(),
                workspace: workspace.clone(),
                verify: vec![],
                parent_id: None,
            })
            .unwrap();

        let model = std::sync::Arc::new(ScriptedModel::new(&[
            "first done",
            "second done",
            "third done",
        ]));

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let in_flight = std::sync::Arc::new(InFlightRegistry::new());

        // Single worker (concurrency=1).
        let rx = shutdown_rx.clone();
        let inf = in_flight.clone();
        let m = model.clone();
        let dd = db.data_dir().to_path_buf();
        let dp = db.path().to_path_buf();
        let worker_thread = std::thread::spawn(move || {
            let worker_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("worker runtime init");
            let local = tokio::task::LocalSet::new();
            local.block_on(&worker_rt, worker_loop(0, dd, dp, m, rx, inf))
        });

        // Wait for all 3 tasks to complete.
        let start = std::time::Instant::now();
        loop {
            let all = store.list(100).unwrap();
            let all_done = all
                .iter()
                .all(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Failed);
            if all_done {
                break;
            }
            if start.elapsed() > Duration::from_secs(30) {
                panic!("timeout waiting for sequential tasks");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        // Verify FIFO order: t1 completed before t2 before t3.
        // (Check via updated_at timestamps — t1's updated_at < t2's < t3's.)
        let loaded1 = store.get(&t1.id).unwrap().unwrap();
        let loaded2 = store.get(&t2.id).unwrap().unwrap();
        let loaded3 = store.get(&t3.id).unwrap().unwrap();

        assert_eq!(loaded1.status, TaskStatus::Completed);
        assert_eq!(loaded2.status, TaskStatus::Completed);
        assert_eq!(loaded3.status, TaskStatus::Completed);

        // FIFO: t1 finished before t2 (updated_at monotonic).
        assert!(
            loaded1.updated_at <= loaded2.updated_at,
            "t1 finished before t2 (FIFO)"
        );
        assert!(
            loaded2.updated_at <= loaded3.updated_at,
            "t2 finished before t3 (FIFO)"
        );

        // Peak should be 1 (single worker, never overlapping).
        assert_eq!(in_flight.peak(), 1, "single worker peak is always 1");

        shutdown_tx.send(true).unwrap();
        worker_thread.join().unwrap().unwrap();

        cleanup(&workspace);
    }

    // ── Stress: 20 tasks, concurrency 4 → all Completed exactly once ───────

    #[tokio::test]
    async fn stress_20_tasks_concurrency_4_all_completed_once() {
        let db = TmpDb::new("stress-20");
        let workspace = make_temp_dir("stress-20-ws");

        save_persona(db.data_dir(), "stressbot");
        save_permissive_policy(db.data_dir());

        let store = TaskStore::open(db.path()).unwrap();

        // Enqueue 20 tasks with explicit verify ("exit 0" always passes).
        let task_ids: Vec<String> = (0..20)
            .map(|i| {
                store
                    .enqueue(NewTask {
                        agent: "stressbot".to_string(),
                        goal: format!("stress task {i}"),
                        workspace: workspace.clone(),
                        verify: vec!["exit 0".to_string()],
                        parent_id: None,
                    })
                    .unwrap()
                    .id
            })
            .collect();

        let concurrent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Short delay to force overlap but keep test fast.
        let model = std::sync::Arc::new(CountingModel::new(
            concurrent.clone(),
            peak.clone(),
            Duration::from_millis(100),
        ));

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let in_flight = std::sync::Arc::new(InFlightRegistry::new());

        let mut worker_threads = Vec::new();
        for i in 0..4 {
            let rx = shutdown_rx.clone();
            let inf = in_flight.clone();
            let m = model.clone();
            let dd = db.data_dir().to_path_buf();
            let dp = db.path().to_path_buf();
            worker_threads.push(std::thread::spawn(move || {
                let worker_rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("worker runtime init");
                let local = tokio::task::LocalSet::new();
                local.block_on(&worker_rt, worker_loop(i, dd, dp, m, rx, inf))
            }));
        }

        // Wait for all 20 tasks to complete.
        let start = std::time::Instant::now();
        loop {
            let all = store.list(100).unwrap();
            let all_done = all
                .iter()
                .all(|t| t.status == TaskStatus::Completed || t.status == TaskStatus::Failed);
            if all_done {
                break;
            }
            if start.elapsed() > Duration::from_secs(60) {
                let pending = all
                    .iter()
                    .filter(|t| t.status != TaskStatus::Completed && t.status != TaskStatus::Failed)
                    .collect::<Vec<_>>();
                panic!(
                    "timeout — {}/20 done, pending: {:?}",
                    all.iter()
                        .filter(|t| t.status == TaskStatus::Completed)
                        .count(),
                    pending
                        .iter()
                        .map(|t| format!("{} ({})", t.id, t.status.as_str()))
                        .collect::<Vec<_>>()
                );
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }

        // Verify: all 20 Completed exactly once.
        let all = store.list(100).unwrap();
        assert_eq!(all.len(), 20, "exactly 20 tasks exist");
        let completed = all
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        assert_eq!(completed, 20, "all 20 tasks Completed");

        // Verify each original task_id is present.
        for id in &task_ids {
            let t = store.get(id).unwrap().unwrap();
            assert_eq!(
                t.status,
                TaskStatus::Completed,
                "task {} should be Completed",
                id
            );
        }

        // Verify overlap: peak >= 2 with 4 workers and 50ms delay.
        let peak_val = peak.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            peak_val >= 2,
            "at least 2 tasks overlapped (peak={peak_val})"
        );

        shutdown_tx.send(true).unwrap();
        for t in worker_threads {
            t.join().unwrap().unwrap();
        }

        cleanup(&workspace);
    }
}
