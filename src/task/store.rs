//! [`TaskStore`] implementation: SQLite-backed task queue + approval inbox.
//!
//! See [`crate::task`] module-level docs for the overall design. This file
//! contains only the store struct and its methods (schema, CRUD, row mappers).

use super::{sqlite_err, ApprovalEntry, ApprovalStatus, NewTask, Task, TaskStatus};
use crate::error::Result;
use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::Path;

/// Schema version for `user_version` pragma.
/// v1: initial schema (no parent_id).
/// v2: adds `parent_id TEXT` column for team task hierarchy.
const SCHEMA_VERSION: u32 = 2;

/// SQLite-backed task store (single connection, WAL mode).
///
/// `Connection` is not `Sync`; each `TaskStore` owns its own connection
/// and is NOT meant to be shared across threads. The daemon and CLI each
/// open their own `TaskStore` against the same DB file (WAL permits
/// concurrent access).
pub struct TaskStore {
    #[cfg(test)]
    pub(crate) conn: Connection,
    #[cfg(not(test))]
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

    /// Migration via `user_version` pragma.
    /// v0→v1: initial schema (CREATE IF NOT EXISTS).
    /// v1→v2: adds `parent_id TEXT` column for team task hierarchy.
    ///
    /// The v1→v2 step is wrapped in a single transaction (including the
    /// `user_version` bump) so a crash mid-migration cannot leave the DB
    /// with the column added but the version still at 1 (M8 fix).
    fn migrate(conn: &Connection) -> Result<()> {
        let ver: u32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .map_err(sqlite_err)?;

        if ver >= SCHEMA_VERSION {
            return Ok(());
        }

        // v1→v2: add parent_id column (additive, idempotent via column-existence check).
        // Wrapped in a transaction so the schema change and version bump are atomic.
        if ver < 2 {
            let has_parent_id: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'parent_id'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .map_err(sqlite_err)?
                > 0;

            // Use execute_batch for the transaction: BEGIN + ALTER + INDEX + PRAGMA + COMMIT.
            // PRAGMA user_version inside the transaction ensures atomicity.
            if !has_parent_id {
                conn.execute_batch(&format!(
                    "BEGIN;
                     ALTER TABLE tasks ADD COLUMN parent_id TEXT;
                     CREATE INDEX IF NOT EXISTS idx_task_parent ON tasks(parent_id);
                     PRAGMA user_version = {SCHEMA_VERSION};
                     COMMIT;"
                ))
                .map_err(sqlite_err)?;
                tracing::info!(step = "v1→v2", "added parent_id column to tasks");
            } else {
                // Column already exists (prior partial migration) — just bump version.
                conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))
                    .map_err(sqlite_err)?;
            }
        }

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
        let parent_id_str = task.parent_id.as_deref();

        self.conn.execute(
            "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at, parent_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                task.agent,
                task.goal,
                workspace_str,
                verify_json,
                TaskStatus::Queued.as_str(),
                now.to_rfc3339(),
                now.to_rfc3339(),
                parent_id_str,
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
            parent_id: task.parent_id,
        })
    }

    /// Returns the oldest Queued task (FIFO by `created_at`), or `None` if
    /// the queue is empty.
    pub fn next_queued(&self) -> Result<Option<Task>> {
        use rusqlite::OptionalExtension;
        self.conn
            .query_row(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report, parent_id
                 FROM tasks WHERE status = 'Queued'
                 ORDER BY created_at ASC, id ASC LIMIT 1",
                [],
                |r| self.read_task_row(r),
            )
            .optional()
            .map_err(sqlite_err)
    }

    /// Atomically claim the next Queued task: sets status to Running and
    /// returns the full Task in a single SQL statement (RETURNING clause).
    /// Under WAL mode, two concurrent claims on separate connections can
    /// never claim the same task — one gets the task, the other gets `None`.
    pub fn claim_next_queued(&self) -> Result<Option<Task>> {
        let now = Utc::now().to_rfc3339();
        use rusqlite::OptionalExtension;
        self.conn
            .query_row(
                "UPDATE tasks SET status='Running', updated_at=?1
                 WHERE id = (SELECT id FROM tasks WHERE status='Queued'
                             ORDER BY created_at ASC, id ASC LIMIT 1)
                   AND status='Queued'
                 RETURNING id, agent, goal, workspace, verify, status, created_at, updated_at, report, parent_id",
                params![now],
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
            return Err(crate::error::LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// CAS: Set task status only if the current status matches `guard`.
    /// Returns `Ok(true)` if the row was updated, `Ok(false)` if the
    /// guard did not match (task already transitioned).
    pub fn set_status_if(&self, id: &str, status: TaskStatus, guard: TaskStatus) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = ?4",
                params![status.as_str(), now, id, guard.as_str()],
            )
            .map_err(sqlite_err)?;
        Ok(changed > 0)
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
            return Err(crate::error::LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// CAS: Mark task Completed only if its current status matches `guard`.
    /// Returns `Ok(true)` if the row was updated, `Ok(false)` if another
    /// worker already changed the status (compare-and-swap failed).
    pub fn complete_if_status(
        &self,
        id: &str,
        report_json: &str,
        guard: TaskStatus,
    ) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = 'Completed', report = ?1, updated_at = ?2 WHERE id = ?3 AND status = ?4",
                params![report_json, now, id, guard.as_str()],
            )
            .map_err(sqlite_err)?;
        Ok(changed > 0)
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
            return Err(crate::error::LoreError::NotFound(format!("task {id}")));
        }
        Ok(())
    }

    /// CAS: Mark task Failed only if its current status matches `guard`.
    /// Returns `Ok(true)` if the row was updated, `Ok(false)` if another
    /// worker already changed the status (compare-and-swap failed).
    pub fn fail_if_status(&self, id: &str, report_json: &str, guard: TaskStatus) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let changed = self
            .conn
            .execute(
                "UPDATE tasks SET status = 'Failed', report = ?1, updated_at = ?2 WHERE id = ?3 AND status = ?4",
                params![report_json, now, id, guard.as_str()],
            )
            .map_err(sqlite_err)?;
        Ok(changed > 0)
    }

    /// List tasks ordered by `created_at` descending, limited to `limit`.
    pub fn list(&self, limit: usize) -> Result<Vec<Task>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report, parent_id
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
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report, parent_id
                 FROM tasks WHERE id = ?1",
                params![id],
                |r| self.read_task_row(r),
            )
            .optional()
            .map_err(sqlite_err)
    }

    /// Returns children of a parent task (tasks where parent_id = id).
    pub fn children_of(&self, id: &str) -> Result<Vec<Task>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, agent, goal, workspace, verify, status, created_at, updated_at, report, parent_id
                 FROM tasks WHERE parent_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map(params![id], |r| self.read_task_row(r))
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(sqlite_err)?);
        }
        Ok(out)
    }

    /// True when no child of `id` is in an active (unfinished) state.
    /// Active states: Queued, Running, WaitingApproval, WaitingSubtasks.
    /// If there are no children, returns true (vacuously all done).
    pub fn all_children_done(&self, id: &str) -> Result<bool> {
        let active_count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM tasks WHERE parent_id = ?1
                 AND status IN ('Queued', 'Running', 'WaitingApproval', 'WaitingSubtasks')",
                params![id],
                |r| r.get(0),
            )
            .map_err(sqlite_err)?;
        Ok(active_count == 0)
    }

    /// Returns all task IDs currently in `WaitingSubtasks` status.
    /// Used by daemon startup sweep to find stuck parents whose children
    /// are all terminal (crash-recovery for C-1).
    pub fn waiting_subtasks_tasks(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM tasks WHERE status = 'WaitingSubtasks'")
            .map_err(sqlite_err)?;
        let ids: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sqlite_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_err)?;
        Ok(ids)
    }

    /// Enqueue a child task with a parent_id link.
    pub fn enqueue_child(&self, parent_id: &str, task: NewTask) -> Result<Task> {
        let child = NewTask {
            agent: task.agent,
            goal: task.goal,
            workspace: task.workspace,
            verify: task.verify,
            parent_id: Some(parent_id.to_string()),
        };
        self.enqueue(child)
    }

    /// Atomic child enqueue: inserts a child task only if no child with the
    /// same `agent` already exists for this parent. Returns `Ok(Some(Task))`
    /// if inserted, `Ok(None)` if a duplicate was detected (CAS success).
    /// This prevents two concurrent workers from enqueueing duplicate
    /// reviewer children (C-2 fix).
    pub fn enqueue_child_if_agent_absent(
        &self,
        parent_id: &str,
        task: NewTask,
    ) -> Result<Option<Task>> {
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now();
        let verify_json = serde_json::to_string(&task.verify)?;
        let workspace_str = task.workspace.to_string_lossy().to_string();

        let changed = self
            .conn
            .execute(
                "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at, parent_id)
                 SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9
                  WHERE NOT EXISTS (SELECT 1 FROM tasks WHERE parent_id = ?9 AND agent = ?2)",
                params![
                    id,
                    task.agent,
                    task.goal,
                    workspace_str,
                    verify_json,
                    TaskStatus::Queued.as_str(),
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    parent_id,
                ],
            )
            .map_err(sqlite_err)?;

        if changed == 0 {
            // Duplicate agent child exists — CAS blocked.
            return Ok(None);
        }

        Ok(Some(Task {
            id,
            agent: task.agent,
            goal: task.goal,
            workspace: task.workspace,
            verify: task.verify,
            status: TaskStatus::Queued,
            created_at: now,
            updated_at: now,
            report: None,
            parent_id: Some(parent_id.to_string()),
        }))
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
                None => return Err(crate::error::LoreError::NotFound(format!("approval {id}"))),
                Some(s) => {
                    return Err(crate::error::LoreError::InvalidInput(format!(
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
        let parent_id: Option<String> = r.get(9)?;

        // M6: propagate unknown status instead of silently falling back.
        let status = TaskStatus::from_str(&status_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(e))
        })?;

        Ok(Task {
            id,
            agent,
            goal,
            workspace: std::path::PathBuf::from(workspace),
            verify: serde_json::from_str(&verify_json).unwrap_or_default(),
            status,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_default(),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_default(),
            report,
            parent_id,
        })
    }

    pub(crate) fn read_approval_row(&self, r: &rusqlite::Row) -> rusqlite::Result<ApprovalEntry> {
        let id: String = r.get(0)?;
        let task_id: String = r.get(1)?;
        let action: String = r.get(2)?;
        let reason: String = r.get(3)?;
        let status_str: String = r.get(4)?;
        let created_at_str: String = r.get(5)?;
        let decided_at: Option<String> = r.get(6)?;

        // M6: propagate unknown approval status instead of silently falling back.
        let status = ApprovalStatus::from_str(&status_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
        })?;

        Ok(ApprovalEntry {
            id,
            task_id,
            action,
            reason,
            status,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_default(),
            decided_at: decided_at.as_ref().map(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .unwrap_or_default()
            }),
        })
    }
}
