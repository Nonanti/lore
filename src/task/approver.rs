//! Queue-backed approver: polls a [`TaskStore`] approval row for human decisions.
//!
//! [`QueueApprover`] inserts an `ApprovalEntry(Pending)` on Ask verdicts,
//! sets the task `WaitingApproval`, then polls `approval_status` at a
//! configurable interval. When the decision arrives (written by CLI or
//! another process via the same DB), it restores the task to `Running`
//! and returns the decision as a `bool`.

use crate::error::{LoreError, Result};
use crate::policy::approval::{ApprovalRequest, Approver};
use crate::task::{ApprovalStatus, TaskStore};
use async_trait::async_trait;
use std::path::Path;
use std::time::Duration;

/// Queue-backed approver: each `decide()` call opens its own `TaskStore`
/// connection (rusqlite `Connection` is not `Sync`; must not be shared
/// across threads).
///
/// Flow:
/// 1. Insert `ApprovalEntry(Pending)` + set task `WaitingApproval`.
/// 2. Poll `approval_status` every `poll_interval` until decided.
/// 3. Restore task to `Running` + return decision.
pub struct QueueApprover {
    /// Path to the task DB file — each call opens a fresh connection.
    db_path: std::path::PathBuf,
    /// Task id this approver is bound to.
    task_id: String,
    /// How often to poll for the decision (default 2s).
    poll_interval: Duration,
}

impl QueueApprover {
    /// New approver bound to a task id, polling at the given interval.
    pub fn new(db_path: &Path, task_id: &str, poll_interval: Duration) -> Self {
        Self {
            db_path: db_path.to_path_buf(),
            task_id: task_id.to_string(),
            poll_interval,
        }
    }

    /// New approver with default 2s poll interval.
    pub fn with_default_poll(db_path: &Path, task_id: &str) -> Self {
        Self::new(db_path, task_id, Duration::from_secs(2))
    }

    /// Open a fresh connection to the task DB.
    fn open_store(&self) -> Result<TaskStore> {
        TaskStore::open(&self.db_path)
    }
}

#[async_trait]
impl Approver for QueueApprover {
    async fn decide(&self, req: &ApprovalRequest) -> Result<bool> {
        let action_json = serde_json::to_string(&req.action)?;
        let reason = req.reason.clone();
        let task_id = self.task_id.clone();
        let poll_interval = self.poll_interval;

        // Open a single connection for the entire decide flow (minor:
        // avoids re-opening per poll tick).
        let store = self.open_store()?;

        // Phase 1: insert approval entry + mark task WaitingApproval.
        let approval_id = store.add_approval(&task_id, &action_json, &reason)?;
        store.set_status(&task_id, crate::task::TaskStatus::WaitingApproval)?;

        // Phase 2: poll until a decision is written (by CLI or another process).
        loop {
            tokio::time::sleep(poll_interval).await;

            let status = store
                .approval_status(&approval_id)?
                .ok_or_else(|| LoreError::NotFound(format!("approval {approval_id}")))?;

            match status {
                ApprovalStatus::Pending => continue,
                ApprovalStatus::Approved => {
                    store.set_status(&task_id, crate::task::TaskStatus::Running)?;
                    return Ok(true);
                }
                ApprovalStatus::Denied => {
                    store.set_status(&task_id, crate::task::TaskStatus::Running)?;
                    return Ok(false);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Action;
    use crate::task::TaskStore;
    use std::path::PathBuf;

    /// Tempdir-backed DB path (manual cleanup).
    struct TmpDb(String);

    impl TmpDb {
        fn new() -> Self {
            let dir =
                std::env::temp_dir().join(format!("lore-approver-test-{}", ulid::Ulid::new()));
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
            for suffix in ["-wal", "-shm", ""] {
                let _ = std::fs::remove_file(format!("{}{}", self.0, suffix));
            }
            if let Some(parent) = Path::new(&self.0).parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }

    fn new_task_request(agent: &str, goal: &str) -> crate::task::NewTask {
        crate::task::NewTask {
            agent: agent.to_string(),
            goal: goal.to_string(),
            workspace: PathBuf::from("/tmp"),
            verify: vec!["echo ok".to_string()],
            parent_id: None,
        }
    }

    // ── QueueApprover end-to-end: approved ───────────────────────────────

    #[tokio::test]
    async fn queue_approver_approved_e2e() {
        let db = TmpDb::new();
        let db_path = db.path().to_path_buf();

        // Enqueue a task.
        let store = TaskStore::open(&db_path).unwrap();
        let t = store.enqueue(new_task_request("a", "goal")).unwrap();
        store
            .set_status(&t.id, crate::task::TaskStatus::Running)
            .unwrap();
        let task_id = t.id.clone();

        // Create approver with 10ms poll.
        let approver = QueueApprover::new(&db_path, &task_id, Duration::from_millis(10));

        let req = ApprovalRequest {
            action: Action::Exec {
                command: "ls".to_string(),
                cwd: PathBuf::from("/tmp"),
            },
            reason: "needs approval".to_string(),
            agent: Some("a".to_string()),
        };

        // Spawn a decider on a SEPARATE connection that approves after 30ms.
        let db_path_clone = db_path.clone();
        let approval_decider = tokio::spawn(async move {
            // Wait a bit before deciding.
            tokio::time::sleep(Duration::from_millis(30)).await;
            let decider_store = TaskStore::open(&db_path_clone).unwrap();
            let pending = decider_store.pending_approvals().unwrap();
            assert!(!pending.is_empty(), "approval should be pending");
            decider_store.decide_approval(&pending[0].id, true).unwrap();
        });

        // Call decide — should poll until approved.
        let result = approver.decide(&req).await.unwrap();
        assert!(result, "approved decision should be true");

        approval_decider.await.unwrap();

        // Verify task is back to Running.
        let verify_store = TaskStore::open(&db_path).unwrap();
        let task = verify_store.get(&task_id).unwrap().unwrap();
        assert_eq!(task.status, crate::task::TaskStatus::Running);
    }

    // ── QueueApprover end-to-end: denied ─────────────────────────────────

    #[tokio::test]
    async fn queue_approver_denied_e2e() {
        let db = TmpDb::new();
        let db_path = db.path().to_path_buf();

        let store = TaskStore::open(&db_path).unwrap();
        let t = store.enqueue(new_task_request("a", "goal")).unwrap();
        store
            .set_status(&t.id, crate::task::TaskStatus::Running)
            .unwrap();
        let task_id = t.id.clone();

        let approver = QueueApprover::new(&db_path, &task_id, Duration::from_millis(10));

        let req = ApprovalRequest {
            action: Action::Exec {
                command: "rm -rf /".to_string(),
                cwd: PathBuf::from("/tmp"),
            },
            reason: "dangerous".to_string(),
            agent: Some("a".to_string()),
        };

        // Spawn a decider that DENIES after 30ms.
        let db_path_clone = db_path.clone();
        let approval_decider = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let decider_store = TaskStore::open(&db_path_clone).unwrap();
            let pending = decider_store.pending_approvals().unwrap();
            decider_store
                .decide_approval(&pending[0].id, false)
                .unwrap();
        });

        let result = approver.decide(&req).await.unwrap();
        assert!(!result, "denied decision should be false");

        approval_decider.await.unwrap();

        // Task is back to Running (even when denied — the policy gate
        // will convert false → PolicyDenied).
        let verify_store = TaskStore::open(&db_path).unwrap();
        let task = verify_store.get(&task_id).unwrap().unwrap();
        assert_eq!(task.status, crate::task::TaskStatus::Running);
    }
}
