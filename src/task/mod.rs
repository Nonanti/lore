//! Task store: SQLite-backed task queue + approval inbox.
//!
//! [`TaskStore`] manages task lifecycle (Queued → Running → Completed/Failed)
//! and approval entries for human-in-the-loop decisions. Schema v1 is designed
//! so Phase 4's `parent_id` column can be added via `user_version` migration.

pub mod approver;

use crate::error::{LoreError, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn sqlite_err(e: rusqlite::Error) -> LoreError {
    LoreError::Storage(format!("sqlite: {e}"))
}

/// Task status lifecycle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Running,
    WaitingApproval,
    /// Task is waiting for subtasks to complete (Phase 4 seam).
    WaitingSubtasks,
    Completed,
    Failed,
}

impl TaskStatus {
    /// SQLite storage string.
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Queued => "Queued",
            TaskStatus::Running => "Running",
            TaskStatus::WaitingApproval => "WaitingApproval",
            TaskStatus::WaitingSubtasks => "WaitingSubtasks",
            TaskStatus::Completed => "Completed",
            TaskStatus::Failed => "Failed",
        }
    }

    /// Parse from SQLite string.
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "Queued" => Ok(TaskStatus::Queued),
            "Running" => Ok(TaskStatus::Running),
            "WaitingApproval" => Ok(TaskStatus::WaitingApproval),
            "WaitingSubtasks" => Ok(TaskStatus::WaitingSubtasks),
            "Completed" => Ok(TaskStatus::Completed),
            "Failed" => Ok(TaskStatus::Failed),
            other => Err(LoreError::Storage(format!("unknown task status: {other}"))),
        }
    }
}

/// A task in the queue.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    /// ULID identifier.
    pub id: String,
    /// Agent name (persona file stem).
    pub agent: String,
    /// What the agent should achieve.
    pub goal: String,
    /// Sandbox root for this task.
    pub workspace: PathBuf,
    /// Verification commands (stored as JSON array text).
    pub verify: Vec<String>,
    /// Current lifecycle status.
    pub status: TaskStatus,
    /// When the task was created.
    pub created_at: DateTime<Utc>,
    /// Last status change.
    pub updated_at: DateTime<Utc>,
    /// WorkReport JSON (present when Completed or Failed).
    pub report: Option<String>,
}

/// Input for enqueue — fields the caller must provide.
#[derive(Clone, Debug)]
pub struct NewTask {
    /// Agent name.
    pub agent: String,
    /// Goal description.
    pub goal: String,
    /// Workspace root (must exist).
    pub workspace: PathBuf,
    /// Verify commands.
    pub verify: Vec<String>,
}

/// Approval entry: tracks a pending human decision on an agent action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApprovalEntry {
    /// ULID identifier.
    pub id: String,
    /// Which task this approval belongs to.
    pub task_id: String,
    /// The action being considered (JSON-serialized `policy::Action`).
    pub action: String,
    /// Why the action requires approval.
    pub reason: String,
    /// Current decision state.
    pub status: ApprovalStatus,
    /// When the approval was requested.
    pub created_at: DateTime<Utc>,
    /// When the decision was made (None while Pending).
    pub decided_at: Option<DateTime<Utc>>,
}

/// Approval decision state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Denied,
}

impl ApprovalStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ApprovalStatus::Pending => "Pending",
            ApprovalStatus::Approved => "Approved",
            ApprovalStatus::Denied => "Denied",
        }
    }

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "Pending" => Ok(ApprovalStatus::Pending),
            "Approved" => Ok(ApprovalStatus::Approved),
            "Denied" => Ok(ApprovalStatus::Denied),
            other => Err(LoreError::Storage(format!(
                "unknown approval status: {other}"
            ))),
        }
    }
}

/// Schema version for `user_version` pragma (v1).
const SCHEMA_VERSION: u32 = 1;

/// SQLite-backed task store (single connection, WAL mode).
///
/// `Connection` is not `Sync`; each `TaskStore` owns its own connection
/// and is NOT meant to be shared across threads. The daemon and CLI each
/// open their own `TaskStore` against the same DB file (WAL permits
/// concurrent access).
pub struct TaskStore {
    conn: Connection,
}

impl TaskStore {
    /// Opens (or creates) the task database at the given path.
    ///
    /// Enables WAL mode for concurrent read/write (daemon + CLI) and sets
    /// `busy_timeout` to 5s. Runs idempotent schema creation + migration
    /// via `user_version`.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(sqlite_err)?;
        Self::init(conn)
    }

    /// In-memory store (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sqlite_err)?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(sqlite_err)?;

        // Idempotent table creation — safe to re-run on every open.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                agent       TEXT NOT NULL,
                goal        TEXT NOT NULL,
                workspace   TEXT NOT NULL,
                verify      TEXT NOT NULL,
                status      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                report      TEXT
            );
            CREATE TABLE IF NOT EXISTS approvals (
                id          TEXT PRIMARY KEY,
                task_id     TEXT NOT NULL,
                action      TEXT NOT NULL,
                reason      TEXT NOT NULL,
                status      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                decided_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_approval_task ON approvals(task_id);
            CREATE INDEX IF NOT EXISTS idx_approval_status ON approvals(status);",
        )
        .map_err(sqlite_err)?;

        Self::migrate(&conn)?;

        Ok(Self { conn })
    }

    /// Migration via `user_version` pragma. v1 is the initial schema;
    /// Phase 4 will add `parent_id` via v2 migration.
    fn migrate(conn: &Connection) -> Result<()> {
        let ver: u32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .map_err(sqlite_err)?;

        if ver >= SCHEMA_VERSION {
            return Ok(());
        }

        // v0 → v1: tables are already created above (CREATE IF NOT EXISTS).
        // Just stamp the version.
        conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))
            .map_err(sqlite_err)?;

        tracing::info!(version = SCHEMA_VERSION, "task store schema initialized");
        Ok(())
    }

    /// Enqueue a new task. Returns the full [`Task`] with generated id and
    /// timestamps.
    pub fn enqueue(&self, task: NewTask) -> Result<Task> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let verify_json = serde_json::to_string(&task.verify)?;
        let workspace_str = task.workspace.to_string_lossy().to_string();

        self.conn.execute(
            "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id,
                task.agent,
                task.goal,
                workspace_str,
                verify_json,
                TaskStatus::Queued.as_str(),
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .map_err(sqlite_err)?;

        Ok(Task {
            id,
            agent: task.agent,
            goal: task.goal,
            workspace: task.workspace,
            verify: task.verify,
            status: TaskStatus::Queued,
            created_at: now,
            updated_at: now,
            report: None,
        })
    }

    /// Returns the oldest Queued task (FIFO by `created_at`), or `None` if
    /// the queue is empty.
    pub fn next_queued(&self) -> Result<Option<Task>> {
        use rusqlite::OptionalExtension;
        self.conn
            .query_row(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report
                 FROM tasks WHERE status = 'Queued'
                 ORDER BY created_at ASC, id ASC LIMIT 1",
                [],
                |r| self.read_task_row(r),
            )
            .optional()
            .map_err(sqlite_err)
    }

    /// Set task status (updates `updated_at`).
    pub fn set_status(&self, id: &str, status: TaskStatus) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = ?1, updated_at = ?2 WHERE id = ?3",
                params![status.as_str(), now, id],
            )
            .map_err(sqlite_err)?;
        if changed == 0 {
            return Err(LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// Mark task Completed with a WorkReport JSON.
    pub fn complete(&self, id: &str, report_json: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = 'Completed', report = ?1, updated_at = ?2 WHERE id = ?3",
                params![report_json, now, id],
            )
            .map_err(sqlite_err)?;
        if changed == 0 {
            return Err(LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// Mark task Failed with a report JSON.
    pub fn fail(&self, id: &str, report_json: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = 'Failed', report = ?1, updated_at = ?2 WHERE id = ?3",
                params![report_json, now, id],
            )
            .map_err(sqlite_err)?;
        if changed == 0 {
            return Err(LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// List tasks ordered by `created_at` descending, limited to `limit`.
    pub fn list(&self, limit: usize) -> Result<Vec<Task>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report
             FROM tasks ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map(params![limit], |r| self.read_task_row(r))
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sqlite_err)?);
        }
        Ok(out)
    }

    /// Get a single task by id, or `None` if absent.
    pub fn get(&self, id: &str) -> Result<Option<Task>> {
        use rusqlite::OptionalExtension;
        self.conn
            .query_row(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report
                 FROM tasks WHERE id = ?1",
                params![id],
                |r| self.read_task_row(r),
            )
            .optional()
            .map_err(sqlite_err)
    }

    /// Insert an approval entry for a task. Returns the approval id.
    ///
    /// The caller should also set the task status to `WaitingApproval`.
    pub fn add_approval(&self, task_id: &str, action_json: &str, reason: &str) -> Result<String> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now().to_rfc3339();
        self.conn
            .execute(
                "INSERT INTO approvals (id, task_id, action, reason, status, created_at)
             VALUES (?1, ?2, ?3, ?4, 'Pending', ?5)",
                params![id, task_id, action_json, reason, now],
            )
            .map_err(sqlite_err)?;
        Ok(id)
    }

    /// List all Pending approval entries.
    pub fn pending_approvals(&self) -> Result<Vec<ApprovalEntry>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, task_id, action, reason, status, created_at, decided_at
             FROM approvals WHERE status = 'Pending'
             ORDER BY created_at ASC",
            )
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |r| self.read_approval_row(r))
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sqlite_err)?);
        }
        Ok(out)
    }

    /// Get the approval status for a specific approval id.
    pub fn approval_status(&self, id: &str) -> Result<Option<ApprovalStatus>> {
        use rusqlite::OptionalExtension;
        self.conn
            .query_row(
                "SELECT status FROM approvals WHERE id = ?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(sqlite_err)
            .and_then(|opt| opt.map(|s| ApprovalStatus::from_str(&s)).transpose())
    }

    /// Decide on an approval: set Approved or Denied, record `decided_at`.
    ///
    /// Idempotent: only updates rows where `status = 'Pending'`. If the
    /// approval was already decided or the id does not exist, returns
    /// [`LoreError::InvalidInput`] with a descriptive message.
    pub fn decide_approval(&self, id: &str, approve: bool) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let status = if approve {
            ApprovalStatus::Approved
        } else {
            ApprovalStatus::Denied
        };
        let changed = self
            .conn
            .execute(
                "UPDATE approvals SET status = ?1, decided_at = ?2 WHERE id = ?3 AND status = 'Pending'",
                params![status.as_str(), now, id],
            )
            .map_err(sqlite_err)?;
        if changed == 0 {
            // Distinguish: not found vs already decided.
            let existing = self.approval_status(id)?;
            match existing {
                None => return Err(LoreError::NotFound(format!("approval {id}"))),
                Some(s) => {
                    return Err(LoreError::InvalidInput(format!(
                        "approval {id} already decided ({})",
                        s.as_str()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Crash-recovery sweep: reset tasks stuck in `Running` or
    /// `WaitingApproval` back to `Queued`, and mark their stale `Pending`
    /// approvals as `Denied`. Returns the number of orphaned tasks
    /// re-queued.
    ///
    /// Call on daemon startup, before entering the poll loop, so that
    /// crash-orphaned tasks are visible to `next_queued()` again.
    pub fn recover_orphaned(&self) -> Result<usize> {
        let now = Utc::now().to_rfc3339();

        // Mark stale Pending approvals for orphaned tasks as Denied.
        let denied = self
            .conn
            .execute(
                "UPDATE approvals SET status = 'Denied', decided_at = ?1\n                 WHERE status = 'Pending'\n                   AND task_id IN (SELECT id FROM tasks WHERE status IN ('Running', 'WaitingApproval'))",
                params![now],
            )
            .map_err(sqlite_err)?;
        if denied > 0 {
            tracing::info!(
                count = denied,
                "denied stale Pending approvals for orphaned tasks"
            );
        }

        // Re-queue orphaned Running/WaitingApproval tasks.
        let requeued = self
            .conn
            .execute(
                "UPDATE tasks SET status = 'Queued', updated_at = ?1\n                 WHERE status IN ('Running', 'WaitingApproval')",
                params![now],
            )
            .map_err(sqlite_err)?;
        if requeued > 0 {
            tracing::info!(count = requeued, "re-queued orphaned tasks on startup");
        }

        Ok(requeued)
    }

    /// Deny all Pending approvals for a specific task (used on SIGINT
    /// re-queue and crash recovery to clear stale approvals).
    pub fn deny_pending_approvals_for_task(&self, task_id: &str) -> Result<usize> {
        let now = Utc::now().to_rfc3339();
        let denied = self
            .conn
            .execute(
                "UPDATE approvals SET status = 'Denied', decided_at = ?1\n                 WHERE task_id = ?2 AND status = 'Pending'",
                params![now, task_id],
            )
            .map_err(sqlite_err)?;
        Ok(denied)
    }

    // ── Row mappers ────────────────────────────────────────────────────

    fn read_task_row(&self, r: &rusqlite::Row) -> rusqlite::Result<Task> {
        let id: String = r.get(0)?;
        let agent: String = r.get(1)?;
        let goal: String = r.get(2)?;
        let workspace: String = r.get(3)?;
        let verify_json: String = r.get(4)?;
        let status_str: String = r.get(5)?;
        let created_at_str: String = r.get(6)?;
        let updated_at_str: String = r.get(7)?;
        let report: Option<String> = r.get(8)?;

        Ok(Task {
            id,
            agent,
            goal,
            workspace: PathBuf::from(workspace),
            verify: serde_json::from_str(&verify_json).unwrap_or_default(),
            status: TaskStatus::from_str(&status_str).unwrap_or(TaskStatus::Queued),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_default(),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_default(),
            report,
        })
    }

    fn read_approval_row(&self, r: &rusqlite::Row) -> rusqlite::Result<ApprovalEntry> {
        let id: String = r.get(0)?;
        let task_id: String = r.get(1)?;
        let action: String = r.get(2)?;
        let reason: String = r.get(3)?;
        let status_str: String = r.get(4)?;
        let created_at_str: String = r.get(5)?;
        let decided_at: Option<String> = r.get(6)?;

        Ok(ApprovalEntry {
            id,
            task_id,
            action,
            reason,
            status: ApprovalStatus::from_str(&status_str).unwrap_or(ApprovalStatus::Pending),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_default(),
            decided_at: decided_at.as_ref().map(|s| {
                DateTime::parse_from_rfc3339(s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_default()
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Tempdir-backed DB path (manual cleanup with WAL side-files).
    struct TmpDb(String);

    impl TmpDb {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("lore-task-test-{}", ulid::Ulid::new()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("tasks.db").to_string_lossy().to_string();
            Self(path)
        }

        fn path(&self) -> &Path {
            Path::new(&self.0)
        }
    }

    impl Drop for TmpDb {
        fn drop(&mut self) {
            // Remove WAL side-files first, then the DB, then the dir.
            for suffix in ["-wal", "-shm", ""] {
                let _ = std::fs::remove_file(format!("{}{}", self.0, suffix));
            }
            if let Some(parent) = Path::new(&self.0).parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    fn new_task(agent: &str, goal: &str) -> NewTask {
        NewTask {
            agent: agent.to_string(),
            goal: goal.to_string(),
            workspace: PathBuf::from("/tmp"),
            verify: vec!["echo ok".to_string()],
        }
    }

    // ── Enqueue / next_queued FIFO ────────────────────────────────────

    #[test]
    fn enqueue_and_next_queued_fifo() {
        let db = TmpDb::new();
        let store = TaskStore::open(db.path()).unwrap();

        let t1 = store.enqueue(new_task("a1", "first goal")).unwrap();
        let t2 = store.enqueue(new_task("a2", "second goal")).unwrap();

        // FIFO: oldest first.
        let next = store.next_queued().unwrap().unwrap();
        assert_eq!(next.id, t1.id, "first queued task comes out first");

        // After consuming t1, t2 is next.
        store.set_status(&t1.id, TaskStatus::Running).unwrap();
        let next2 = store.next_queued().unwrap().unwrap();
        assert_eq!(next2.id, t2.id, "second task after first is consumed");
    }

    #[test]
    fn next_queued_returns_none_when_empty() {
        let store = TaskStore::in_memory().unwrap();
        assert!(store.next_queued().unwrap().is_none());
    }

    // ── Full status-transition path ───────────────────────────────────

    #[test]
    fn full_status_transition_path() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();
        assert_eq!(t.status, TaskStatus::Queued);

        store.set_status(&t.id, TaskStatus::Running).unwrap();
        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::Running);

        store
            .set_status(&t.id, TaskStatus::WaitingApproval)
            .unwrap();
        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::WaitingApproval);

        store.set_status(&t.id, TaskStatus::Running).unwrap();
        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::Running);

        // WaitingSubtasks (Phase 4 seam).
        store
            .set_status(&t.id, TaskStatus::WaitingSubtasks)
            .unwrap();
        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::WaitingSubtasks);
    }

    // ── Complete / fail with report JSON roundtrip ────────────────────

    #[test]
    fn complete_with_report_json_roundtrip() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();
        store.set_status(&t.id, TaskStatus::Running).unwrap();

        let report = serde_json::json!({
            "success": true,
            "iterations": 3,
            "answer": "done"
        });
        let report_json = serde_json::to_string(&report).unwrap();
        store.complete(&t.id, &report_json).unwrap();

        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::Completed);
        assert!(t.report.is_some());
        let parsed: serde_json::Value = serde_json::from_str(t.report.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["iterations"], 3);
    }

    #[test]
    fn fail_with_report_json_roundtrip() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();
        store.set_status(&t.id, TaskStatus::Running).unwrap();

        let report = serde_json::json!({
            "success": false,
            "iterations": 5,
            "answer": "partial"
        });
        let report_json = serde_json::to_string(&report).unwrap();
        store.fail(&t.id, &report_json).unwrap();

        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::Failed);
        let parsed: serde_json::Value = serde_json::from_str(t.report.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["success"], false);
    }

    // ── List limit + ordering ─────────────────────────────────────────

    #[test]
    fn list_limit_and_ordering() {
        let store = TaskStore::in_memory().unwrap();
        let t1 = store.enqueue(new_task("a1", "first")).unwrap();
        let _t2 = store.enqueue(new_task("a2", "second")).unwrap();
        let t3 = store.enqueue(new_task("a3", "third")).unwrap();

        // Default ordering: created_at DESC (newest first).
        let all = store.list(100).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].id, t3.id, "newest first");
        assert_eq!(all[2].id, t1.id, "oldest last");

        // Limit truncates.
        let limited = store.list(2).unwrap();
        assert_eq!(limited.len(), 2);
    }

    // ── Approval add → pending → decide → status ─────────────────────

    #[test]
    fn approval_add_pending_decide_status() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();

        let action_json = serde_json::to_string(&serde_json::json!({
            "Exec": {"command": "rm -rf /", "cwd": "/tmp"}
        }))
        .unwrap();

        // Add approval.
        let approval_id = store
            .add_approval(&t.id, &action_json, "dangerous command")
            .unwrap();

        // Set task WaitingApproval.
        store
            .set_status(&t.id, TaskStatus::WaitingApproval)
            .unwrap();

        // Pending approvals list.
        let pending = store.pending_approvals().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, approval_id);
        assert_eq!(pending[0].task_id, t.id);
        assert_eq!(pending[0].status, ApprovalStatus::Pending);

        // Approval status query.
        let status = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(status, ApprovalStatus::Pending);

        // Decide: approve.
        store.decide_approval(&approval_id, true).unwrap();

        let status = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(status, ApprovalStatus::Approved);

        // No more pending.
        let pending = store.pending_approvals().unwrap();
        assert_eq!(pending.len(), 0);

        // Task can go back to Running.
        store.set_status(&t.id, TaskStatus::Running).unwrap();
        let t = store.get(&t.id).unwrap().unwrap();
        assert_eq!(t.status, TaskStatus::Running);
    }

    #[test]
    fn approval_denied() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();

        let approval_id = store.add_approval(&t.id, "{}", "test").unwrap();

        store.decide_approval(&approval_id, false).unwrap();
        let status = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(status, ApprovalStatus::Denied);
    }

    #[test]
    fn approval_status_returns_none_for_unknown_id() {
        let store = TaskStore::in_memory().unwrap();
        assert!(store.approval_status("nonexistent").unwrap().is_none());
    }

    #[test]
    fn decide_approval_unknown_id_errors() {
        let store = TaskStore::in_memory().unwrap();
        let result = store.decide_approval("nonexistent", true);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LoreError::NotFound(_)));
    }

    #[test]
    fn set_status_unknown_id_errors() {
        let store = TaskStore::in_memory().unwrap();
        let result = store.set_status("nonexistent", TaskStatus::Running);
        assert!(result.is_err());
    }

    // ── Reopen persistence ────────────────────────────────────────────

    #[test]
    fn reopen_persistence() {
        let db = TmpDb::new();

        {
            let store = TaskStore::open(db.path()).unwrap();
            let t = store
                .enqueue(new_task("persist", "persistent goal"))
                .unwrap();
            store.set_status(&t.id, TaskStatus::Running).unwrap();

            let _approval_id = store.add_approval(&t.id, "{}", "test reason").unwrap();
            store
                .set_status(&t.id, TaskStatus::WaitingApproval)
                .unwrap();
        }

        // Reopen: data survives.
        let store = TaskStore::open(db.path()).unwrap();
        let t = store.next_queued().unwrap(); // No Queued tasks — Running/WaitingApproval.
        assert!(t.is_none(), "no queued tasks after reopen");

        let list = store.list(100).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].status, TaskStatus::WaitingApproval);

        let pending = store.pending_approvals().unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].reason, "test reason");
    }

    // ── Verify Vec<String> JSON array roundtrip ──────────────────────

    #[test]
    fn verify_vec_json_roundtrip() {
        let store = TaskStore::in_memory().unwrap();
        let task = NewTask {
            agent: "a".to_string(),
            goal: "g".to_string(),
            workspace: PathBuf::from("/tmp"),
            verify: vec![
                "cargo test".to_string(),
                "cargo clippy -- -D warnings".to_string(),
            ],
        };
        let t = store.enqueue(task).unwrap();
        let loaded = store.get(&t.id).unwrap().unwrap();
        assert_eq!(
            loaded.verify,
            vec![
                "cargo test".to_string(),
                "cargo clippy -- -D warnings".to_string(),
            ]
        );
    }

    // ── DateTime RFC3339 storage ──────────────────────────────────────

    #[test]
    fn datetime_rfc3339_roundtrip() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();
        let loaded = store.get(&t.id).unwrap().unwrap();
        // Timestamps should be equal (RFC3339 preserves UTC).
        assert_eq!(loaded.created_at, t.created_at);
        assert_eq!(loaded.updated_at, t.updated_at);
    }

    // ── next_queued skips non-Queued tasks ────────────────────────────

    #[test]
    fn next_queued_skips_non_queued() {
        let store = TaskStore::in_memory().unwrap();

        // Enqueue 4 tasks, set statuses to Running, Completed, Failed, WaitingApproval.
        let t1 = store.enqueue(new_task("a1", "running goal")).unwrap();
        store.set_status(&t1.id, TaskStatus::Running).unwrap();

        let t2 = store.enqueue(new_task("a2", "completed goal")).unwrap();
        store.set_status(&t2.id, TaskStatus::Completed).unwrap();

        let t3 = store.enqueue(new_task("a3", "failed goal")).unwrap();
        store.set_status(&t3.id, TaskStatus::Failed).unwrap();

        let t4 = store.enqueue(new_task("a4", "waiting goal")).unwrap();
        store
            .set_status(&t4.id, TaskStatus::WaitingApproval)
            .unwrap();

        // Enqueue one more that stays Queued.
        let t_queued = store.enqueue(new_task("a5", "queued goal")).unwrap();

        let next = store.next_queued().unwrap().unwrap();
        assert_eq!(next.id, t_queued.id, "only the Queued task is returned");
    }

    // ── decide_approval on already-decided entry (current: overwrites) ──

    #[test]
    fn decide_approval_idempotent_rejects_already_decided() {
        let store = TaskStore::in_memory().unwrap();
        let t = store.enqueue(new_task("a", "goal")).unwrap();
        let approval_id = store.add_approval(&t.id, "{}", "test").unwrap();

        // Decide: approve.
        store.decide_approval(&approval_id, true).unwrap();
        let status = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(status, ApprovalStatus::Approved);

        // Re-decide: should fail — already decided.
        let err = store.decide_approval(&approval_id, false).unwrap_err();
        assert!(
            matches!(err, LoreError::InvalidInput(_)),
            "re-deciding should return InvalidInput: {err:?}"
        );

        // Status unchanged — still Approved (not overwritten to Denied).
        let status = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(
            status,
            ApprovalStatus::Approved,
            "status unchanged after rejected re-decide"
        );

        // No pending entries.
        let pending = store.pending_approvals().unwrap();
        assert_eq!(pending.len(), 0, "no pending after decision");
    }

    // ── WAL: two connections on same DB, no deadlock ───────────────────

    #[test]
    fn wal_two_connections_no_deadlock() {
        let db = TmpDb::new();
        let conn1 = TaskStore::open(db.path()).unwrap();
        let conn2 = TaskStore::open(db.path()).unwrap();

        // Enqueue on conn1.
        let t1 = conn1.enqueue(new_task("wal1", "goal1")).unwrap();

        // Read on conn2 — should see the task.
        let next = conn2.next_queued().unwrap().unwrap();
        assert_eq!(next.id, t1.id, "conn2 sees conn1's enqueue");

        // Set status on conn2.
        conn2.set_status(&t1.id, TaskStatus::Running).unwrap();

        // Verify on conn1.
        let loaded = conn1.get(&t1.id).unwrap().unwrap();
        assert_eq!(
            loaded.status,
            TaskStatus::Running,
            "conn1 sees conn2's update"
        );

        // Enqueue on conn2 while conn1 is also writing.
        let _t2 = conn2.enqueue(new_task("wal2", "goal2")).unwrap();
        let list = conn1.list(100).unwrap();
        assert_eq!(list.len(), 2, "conn1 sees both tasks");
    }

    // ── Approval decide visible across connections ──────────────────────

    #[test]
    fn approval_decide_visible_across_connections() {
        let db = TmpDb::new();
        let writer = TaskStore::open(db.path()).unwrap();
        let reader = TaskStore::open(db.path()).unwrap();

        let t = writer.enqueue(new_task("a", "goal")).unwrap();
        let approval_id = writer
            .add_approval(&t.id, "{}", "cross-conn reason")
            .unwrap();

        // Writer decides: approve.
        writer.decide_approval(&approval_id, true).unwrap();

        // Reader sees the decision.
        let status = reader.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(
            status,
            ApprovalStatus::Approved,
            "decision visible across connections"
        );

        // Reader sees no pending approvals.
        let pending = reader.pending_approvals().unwrap();
        assert_eq!(pending.len(), 0, "no pending after cross-connection decide");
    }

    // ── Crash recovery: recover_orphaned ──────────────────────────────

    #[test]
    fn recover_orphaned_requeues_running_and_waiting_approval() {
        let store = TaskStore::in_memory().unwrap();

        // Task stuck in Running (daemon crash).
        let t1 = store.enqueue(new_task("a1", "running goal")).unwrap();
        store.set_status(&t1.id, TaskStatus::Running).unwrap();

        // Task stuck in WaitingApproval with a Pending approval.
        let t2 = store.enqueue(new_task("a2", "waiting goal")).unwrap();
        store
            .set_status(&t2.id, TaskStatus::WaitingApproval)
            .unwrap();
        let approval_id = store.add_approval(&t2.id, "{}", "stale approval").unwrap();

        // A normally queued task stays untouched.
        let t3 = store.enqueue(new_task("a3", "queued goal")).unwrap();

        // Recover.
        let requeued = store.recover_orphaned().unwrap();
        assert_eq!(requeued, 2, "two orphaned tasks re-queued");

        // Both orphaned tasks are now Queued.
        let loaded1 = store.get(&t1.id).unwrap().unwrap();
        assert_eq!(loaded1.status, TaskStatus::Queued, "Running task re-queued");

        let loaded2 = store.get(&t2.id).unwrap().unwrap();
        assert_eq!(
            loaded2.status,
            TaskStatus::Queued,
            "WaitingApproval task re-queued"
        );

        // The stale approval is now Denied.
        let approval = store.approval_status(&approval_id).unwrap().unwrap();
        assert_eq!(
            approval,
            ApprovalStatus::Denied,
            "stale approval marked Denied"
        );

        // t3 remains Queued (was never orphaned).
        let loaded3 = store.get(&t3.id).unwrap().unwrap();
        assert_eq!(loaded3.status, TaskStatus::Queued, "normal task unaffected");

        // next_queued picks t1 first (earliest created_at).
        let next = store.next_queued().unwrap().unwrap();
        assert_eq!(next.id, t1.id, "FIFO order preserved after recovery");
    }

    #[test]
    fn recover_orphaned_no_orphans_is_zero() {
        let store = TaskStore::in_memory().unwrap();
        let requeued = store.recover_orphaned().unwrap();
        assert_eq!(requeued, 0, "no orphans → 0 re-queued");
    }

    // ── FIFO tiebreaker: id ordering ──────────────────────────────────

    #[test]
    fn fifo_tiebreaker_by_id_when_same_created_at() {
        let store = TaskStore::in_memory().unwrap();

        // Insert two tasks with identical created_at by directly writing rows
        // (enqueue generates unique timestamps, so we can't rely on it).
        let now = Utc::now().to_rfc3339();
        let verify_json = serde_json::to_string(&vec!["echo ok"]).unwrap();

        // IDs: smaller ULID first alphabetically.
        let id_a = "01ARZ00000000000000000000"; // smallest ULID-like id
        let id_b = "01ARZ99999999999999999999"; // larger ULID-like id

        for (id, agent) in [(id_a, "first"), (id_b, "second")] {
            store
                .conn
                .execute(
                    "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at)\n                     VALUES (?1, ?2, ?3, ?4, ?5, 'Queued', ?6, ?7)",
                    rusqlite::params![id, agent, "goal", "/tmp", verify_json, now, now],
                )
                .unwrap();
        }

        let next = store.next_queued().unwrap().unwrap();
        assert_eq!(next.id, id_a, "FIFO tiebreaker: smaller id comes first");
    }
}
