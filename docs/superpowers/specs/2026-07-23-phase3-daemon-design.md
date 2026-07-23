# Phase 3 — Daemon + Task Queue + CLI Client

Roadmap context: Phase 1 (hands) ✔, Phase 2 (work loop) spec'd. This phase
makes agents **always-on coworkers**: you enqueue a task, the daemon works it
end-to-end, approval requests wait in an inbox until you answer.

## Architecture

Single operator, single machine → **everything coordinates through SQLite**
(no IPC server, no sockets): the daemon is the only process that transitions
task state; the CLI only inserts tasks and answers approvals. WAL mode lets
both sides read/write concurrently.

### Task store — `src/task/mod.rs`

New SQLite store at `<LORE_DATA>/tasks.db` (rusqlite, bundled — existing dep).
Follow the codebase's existing sqlite patterns (see `src/memory/sqlite.rs`):
WAL, `CREATE TABLE IF NOT EXISTS`, migration-friendly `user_version`.

```rust
pub struct Task {
    pub id: String,                  // ulid
    pub agent: String,               // agent name (persona file stem)
    pub goal: String,
    pub workspace: PathBuf,
    pub verify: Vec<String>,         // JSON array in DB
    pub status: TaskStatus,          // Queued | Running | WaitingApproval | Completed | Failed
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub report: Option<String>,      // WorkReport JSON when Completed/Failed
}

pub struct ApprovalEntry {
    pub id: String,                  // ulid
    pub task_id: String,
    pub action: String,              // Action JSON (policy::Action)
    pub reason: String,
    pub status: ApprovalStatus,      // Pending | Approved | Denied
    pub created_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
}

pub struct TaskStore { /* rusqlite Connection, WAL */ }
impl TaskStore {
    pub fn open(path: &Path) -> Result<Self>;
    pub fn enqueue(&self, task: NewTask) -> Result<Task>;
    pub fn next_queued(&self) -> Result<Option<Task>>;          // FIFO
    pub fn set_status(&self, id: &str, status: TaskStatus) -> Result<()>;
    pub fn complete(&self, id: &str, report_json: &str) -> Result<()>;
    pub fn fail(&self, id: &str, report_json: &str) -> Result<()>;
    pub fn list(&self, limit: usize) -> Result<Vec<Task>>;
    pub fn get(&self, id: &str) -> Result<Option<Task>>;
    // approvals
    pub fn add_approval(&self, task_id: &str, action_json: &str, reason: &str) -> Result<String>;
    pub fn pending_approvals(&self) -> Result<Vec<ApprovalEntry>>;
    pub fn approval_status(&self, id: &str) -> Result<Option<ApprovalStatus>>;
    pub fn decide_approval(&self, id: &str, approve: bool) -> Result<()>;
}
```

### Queued approver — `src/task/approver.rs`

```rust
pub struct QueueApprover { store: TaskStore (clone/connection), task_id: String, poll: Duration }
#[async_trait] impl Approver for QueueApprover {
    async fn decide(&self, req) -> Result<bool> {
        // insert ApprovalEntry(Pending); mark task WaitingApproval;
        // poll approval_status every `poll` (default 2s) until decided;
        // restore task Running; return decision.
    }
}
```

This plugs into `Gate` unchanged — the Phase 1 seam pays off. CLI answers
write the decision row; the daemon's next poll picks it up.

### Daemon — `src/daemon.rs` + `lore daemon` subcommand

Loop (sequential — one task at a time, deliberately):

1. `next_queued()` → none: sleep 2 s, repeat. Some: mark Running.
2. Load agent: persona from `<LORE_DATA>/agents/<name>.json` via
   `Agent::load_from` (existing), memory = its scoped SqliteStore (existing
   pattern in main.rs), model from env config (existing CLI wiring — reuse
   the same helper; per-agent models are Phase 4).
3. Build per-task `Policy`: load `<LORE_DATA>/policy.json` if present, else
   `Policy::default_for(task.workspace)`. Build `Gate` with
   `QueueApprover { task_id }`. Build ToolContext: shell/write/edit (+ file
   read) registered; `LlmRouter` over the same model.
4. `agent.work(&ctx, gate, &WorkSpec { goal, workspace, verify, .. })`.
5. Completed → `complete(id, report_json)`; Err → `fail(id, error string as
   report)`. Log lines (`tracing`) also to `<LORE_DATA>/logs/<task_id>.log`
   via a per-task file appender (simple `std::fs::OpenOptions` writer behind
   a small struct — no new logging deps).
6. Graceful shutdown on SIGTERM/SIGINT (tokio::signal — feature already on):
   finish current verify command, mark task Queued again if mid-run.

### CLI client — new subcommands in `src/main.rs`

- `lore daemon` — runs the loop in the foreground (systemd/tmux friendly).
- `lore task add <agent> <goal> [--workspace PATH] [--verify CMD]...`
  (no --verify → `WorkSpec::for_workspace` detection).
- `lore task list [--limit N]` — table: id, agent, status, age, goal(60ch).
- `lore task status <id>` — full record incl. report summary.
- `lore task log <id> [--tail N]` — prints the task log file.
- `lore inbox` — pending approvals (id, task, action, reason, age).
- `lore approve <id>` / `lore deny <id>` — write the decision.

## Error handling

- Daemon survives agent/model errors: any task error → Failed + continue loop.
- Task DB open failure at daemon start → fatal with clear message.
- `lore approve` on non-pending id → clear error.

## Testing

- TaskStore: enqueue/next FIFO order, status transitions, approval
  add/decide/pending, list limit, re-open persistence (tempdir).
- QueueApprover: with a background task deciding after a delay (insert row,
  spawn decider via store handle, assert decide() returns the decision and
  polls terminate). Use short poll interval (10 ms) in tests.
- Daemon loop: extract the per-task execution into a testable fn
  (`run_task(store, task, deps) -> Result<WorkReport>`); test with ScriptedModel
  + tempdir + AllowAll/QueueApprover hybrid. Full `lore daemon` process test
  is out of scope (manual verification).
- CLI parsing: clap unit tests for the new subcommands (existing pattern).

## Non-goals

Parallel task execution, priorities, retries, per-agent models (Phase 4),
PM/team tasks (Phase 4), HTTP endpoints for tasks (CLI is the client),
systemd unit files.
