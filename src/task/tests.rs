//! Task module tests: types + store integration.

use super::*;
use std::path::{Path, PathBuf};

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
        parent_id: None,
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
        parent_id: None,
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
    let now = chrono::Utc::now().to_rfc3339();
    let verify_json = serde_json::to_string(&vec!["echo ok"]).unwrap();

    // IDs: smaller ULID first alphabetically.
    let id_a = "01ARZ00000000000000000000"; // smallest ULID-like id
    let id_b = "01ARZ99999999999999999999"; // larger ULID-like id

    for (id, agent) in [(id_a, "first"), (id_b, "second")] {
        store
            .conn
            .execute(
                "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at)\n                 VALUES (?1, ?2, ?3, ?4, ?5, 'Queued', ?6, ?7)",
                rusqlite::params![id, agent, "goal", "/tmp", verify_json, now, now],
            )
            .unwrap();
    }

    let next = store.next_queued().unwrap().unwrap();
    assert_eq!(next.id, id_a, "FIFO tiebreaker: smaller id comes first");
}

// ── v1→v2 migration preserves pre-existing rows ────────────────────

#[test]
fn migration_v1_to_v2_preserves_existing_rows() {
    let db = TmpDb::new();

    // Manually create a v1 schema DB (no parent_id column).
    {
        let conn = rusqlite::Connection::open(db.path()).unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .unwrap();
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
                CREATE INDEX IF NOT EXISTS idx_approval_status ON approvals(status);
                PRAGMA user_version = 1;",
        )
        .unwrap();
        // Insert a row in v1 schema.
        let now = chrono::Utc::now().to_rfc3339();
        let verify_json = serde_json::to_string(&vec!["echo ok"]).unwrap();
        conn.execute(
            "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at)
                 VALUES ('01TESTMIGRATION01', 'migbot', 'migrate me', '/tmp', ?1, 'Completed', ?2, ?3)",
            rusqlite::params![verify_json, now, now],
        )
        .unwrap();
    }

    // Open with TaskStore (triggers v1→v2 migration).
    let store = TaskStore::open(db.path()).unwrap();

    // Pre-existing row is preserved.
    let task = store.get("01TESTMIGRATION01").unwrap().unwrap();
    assert_eq!(task.agent, "migbot");
    assert_eq!(task.goal, "migrate me");
    assert_eq!(task.status, TaskStatus::Completed);
    assert_eq!(task.parent_id, None, "v1 rows have no parent_id");

    // New row with parent_id works.
    let child = store
        .enqueue_child("01TESTMIGRATION01", new_task("child", "sub goal"))
        .unwrap();
    assert_eq!(child.parent_id, Some("01TESTMIGRATION01".to_string()));
}

// ── enqueue_child / children_of / all_children_done ──────────────

#[test]
fn enqueue_child_and_children_of() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "big goal")).unwrap();

    let c1 = store
        .enqueue_child(&parent.id, new_task("backend", "impl feature"))
        .unwrap();
    let c2 = store
        .enqueue_child(&parent.id, new_task("frontend", "build UI"))
        .unwrap();

    assert_eq!(c1.parent_id, Some(parent.id.clone()));
    assert_eq!(c2.parent_id, Some(parent.id.clone()));

    let children = store.children_of(&parent.id).unwrap();
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].id, c1.id);
    assert_eq!(children[1].id, c2.id);

    // Children of a nonexistent id return empty.
    let empty = store.children_of("nonexistent").unwrap();
    assert!(empty.is_empty());
}

#[test]
fn all_children_done_transitions() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "goal")).unwrap();

    let c1 = store
        .enqueue_child(&parent.id, new_task("a", "g1"))
        .unwrap();
    let c2 = store
        .enqueue_child(&parent.id, new_task("b", "g2"))
        .unwrap();

    // Both Queued → not done.
    assert!(!store.all_children_done(&parent.id).unwrap());

    // c1 Running → still not done.
    store.set_status(&c1.id, TaskStatus::Running).unwrap();
    assert!(!store.all_children_done(&parent.id).unwrap());

    // c1 Completed, c2 Queued → still not done.
    store
        .complete(
            &c1.id,
            &serde_json::to_string(&serde_json::json!({"success":true})).unwrap(),
        )
        .unwrap();
    assert!(!store.all_children_done(&parent.id).unwrap());

    // c2 Completed → all done.
    store
        .complete(
            &c2.id,
            &serde_json::to_string(&serde_json::json!({"success":true})).unwrap(),
        )
        .unwrap();
    assert!(store.all_children_done(&parent.id).unwrap());

    // One child Failed, one Completed → all done (both in terminal states).
    let p2 = store.enqueue(new_task("pm", "goal2")).unwrap();
    let f1 = store.enqueue_child(&p2.id, new_task("a", "g")).unwrap();
    let f2 = store.enqueue_child(&p2.id, new_task("b", "g")).unwrap();
    store
        .complete(
            &f1.id,
            &serde_json::to_string(&serde_json::json!({"success":true})).unwrap(),
        )
        .unwrap();
    store
        .fail(
            &f2.id,
            &serde_json::to_string(&serde_json::json!({"success":false})).unwrap(),
        )
        .unwrap();
    assert!(store.all_children_done(&p2.id).unwrap());

    // No children → vacuously all done.
    let solo = store.enqueue(new_task("solo", "goal")).unwrap();
    assert!(store.all_children_done(&solo.id).unwrap());
}

// ── Migration idempotent: re-opening a v2 DB doesn't break ──────────

#[test]
fn migration_v2_idempotent_on_reopen() {
    let db = TmpDb::new();

    // First open: creates v2 schema with parent_id.
    {
        let store = TaskStore::open(db.path()).unwrap();
        let parent = store.enqueue(new_task("pm", "goal")).unwrap();
        let child = store
            .enqueue_child(&parent.id, new_task("backend", "sub"))
            .unwrap();
        assert_eq!(child.parent_id, Some(parent.id.clone()));
    }

    // Second open: migration should be idempotent — no ALTER TABLE error.
    let store = TaskStore::open(db.path()).unwrap();
    let parent_loaded = store.get("01ARZ00000").unwrap(); // nonexistent is fine
    assert!(parent_loaded.is_none(), "sanity check");

    // Parent_id column still works after re-open.
    let new_parent = store.enqueue(new_task("pm2", "goal2")).unwrap();
    let new_child = store
        .enqueue_child(&new_parent.id, new_task("frontend", "sub2"))
        .unwrap();
    assert_eq!(
        new_child.parent_id,
        Some(new_parent.id.clone()),
        "parent_id column works after idempotent migration"
    );

    // Can still query children_of.
    let children = store.children_of(&new_parent.id).unwrap();
    assert_eq!(children.len(), 1);
}

// ── all_children_done with WaitingSubtasks child ───────────────────

#[test]
fn all_children_done_waiting_subtasks_is_not_done() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "goal")).unwrap();
    let c1 = store
        .enqueue_child(&parent.id, new_task("backend", "g"))
        .unwrap();

    // WaitingSubtasks is a non-terminal state → children are NOT done.
    store
        .set_status(&c1.id, TaskStatus::WaitingSubtasks)
        .unwrap();
    assert!(
        !store.all_children_done(&parent.id).unwrap(),
        "WaitingSubtasks child means not all done"
    );
}

// ── all_children_done with WaitingApproval child ───────────────────

#[test]
fn all_children_done_waiting_approval_is_not_done() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "goal")).unwrap();
    let c1 = store
        .enqueue_child(&parent.id, new_task("backend", "g"))
        .unwrap();

    store
        .set_status(&c1.id, TaskStatus::WaitingApproval)
        .unwrap();
    assert!(
        !store.all_children_done(&parent.id).unwrap(),
        "WaitingApproval child means not all done"
    );
}

// ── claim_next_queued ────────────────────────────────────────────

#[test]
fn claim_next_queued_returns_none_when_empty() {
    let store = TaskStore::in_memory().unwrap();
    assert!(store.claim_next_queued().unwrap().is_none());
}

#[test]
fn claim_next_queued_fifo() {
    let store = TaskStore::in_memory().unwrap();
    let t1 = store.enqueue(new_task("a1", "first goal")).unwrap();
    let t2 = store.enqueue(new_task("a2", "second goal")).unwrap();

    let claimed = store.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed.id, t1.id, "FIFO: first task claimed first");
    assert_eq!(claimed.status, TaskStatus::Running, "status set to Running");

    let claimed2 = store.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed2.id, t2.id, "FIFO: second task claimed second");
    assert_eq!(claimed2.status, TaskStatus::Running);

    assert!(
        store.claim_next_queued().unwrap().is_none(),
        "no more tasks"
    );
}

#[test]
fn claim_next_queued_atomic_double_claim_one_task() {
    // Two connections, one task: first claim gets it, second gets None.
    let db = TmpDb::new();
    let conn1 = TaskStore::open(db.path()).unwrap();
    let conn2 = TaskStore::open(db.path()).unwrap();

    let t1 = conn1.enqueue(new_task("a", "only task")).unwrap();

    let claimed = conn1.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed.id, t1.id);

    // conn2 tries to claim the same task — should get None.
    let claimed2 = conn2.claim_next_queued().unwrap();
    assert!(
        claimed2.is_none(),
        "second claim on same task must return None"
    );
}

#[test]
fn claim_next_queued_atomic_two_tasks_two_connections() {
    // Two connections, two tasks: both get different tasks (FIFO preserved).
    let db = TmpDb::new();
    let conn1 = TaskStore::open(db.path()).unwrap();
    let conn2 = TaskStore::open(db.path()).unwrap();

    let t1 = conn1.enqueue(new_task("a1", "first")).unwrap();
    let t2 = conn1.enqueue(new_task("a2", "second")).unwrap();

    let claimed1 = conn1.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed1.id, t1.id, "first connection gets first task");

    let claimed2 = conn2.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed2.id, t2.id, "second connection gets second task");

    // Neither can claim again.
    assert!(conn1.claim_next_queued().unwrap().is_none());
    assert!(conn2.claim_next_queued().unwrap().is_none());
}

#[test]
fn claim_next_queued_skips_non_queued() {
    // claim_next_queued only picks up Queued tasks; Running/Completed/Failed/
    // WaitingApproval/WaitingSubtasks are all skipped.
    let store = TaskStore::in_memory().unwrap();

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

    let t5 = store.enqueue(new_task("a5", "subtasks goal")).unwrap();
    store
        .set_status(&t5.id, TaskStatus::WaitingSubtasks)
        .unwrap();

    // Enqueue one more that stays Queued.
    let t_queued = store.enqueue(new_task("a6", "queued goal")).unwrap();

    let claimed = store.claim_next_queued().unwrap().unwrap();
    assert_eq!(claimed.id, t_queued.id, "only the Queued task is claimed");
    assert_eq!(
        claimed.status,
        TaskStatus::Running,
        "claimed task status set to Running"
    );

    // No more Queued tasks.
    assert!(store.claim_next_queued().unwrap().is_none());
}

// ── complete_if_status / fail_if_status (CAS guards) ──────────────

#[test]
fn complete_if_status_succeeds_when_guard_matches() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();
    store
        .set_status(&t.id, TaskStatus::WaitingSubtasks)
        .unwrap();

    let report = serde_json::json!({"success": true, "answer": "done"});
    let report_json = serde_json::to_string(&report).unwrap();

    // CAS succeeds: status IS WaitingSubtasks.
    let ok = store
        .complete_if_status(&t.id, &report_json, TaskStatus::WaitingSubtasks)
        .unwrap();
    assert!(ok, "CAS should succeed when status matches guard");

    let t = store.get(&t.id).unwrap().unwrap();
    assert_eq!(t.status, TaskStatus::Completed);
}

#[test]
fn complete_if_status_fails_when_guard_wrong() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();

    let report_json = "{\"success\":true}";

    // CAS fails: status is Running, not WaitingSubtasks.
    let ok = store
        .complete_if_status(&t.id, report_json, TaskStatus::WaitingSubtasks)
        .unwrap();
    assert!(!ok, "CAS should fail when status does not match guard");

    // Status unchanged.
    let t = store.get(&t.id).unwrap().unwrap();
    assert_eq!(t.status, TaskStatus::Running);
}

#[test]
fn fail_if_status_succeeds_when_guard_matches() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();
    store
        .set_status(&t.id, TaskStatus::WaitingSubtasks)
        .unwrap();

    let report_json = "{\"success\":false}";
    let ok = store
        .fail_if_status(&t.id, report_json, TaskStatus::WaitingSubtasks)
        .unwrap();
    assert!(ok, "CAS should succeed when status matches guard");

    let t = store.get(&t.id).unwrap().unwrap();
    assert_eq!(t.status, TaskStatus::Failed);
}

#[test]
fn fail_if_status_fails_when_guard_wrong() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();

    let report_json = "{\"success\":false}";
    let ok = store
        .fail_if_status(&t.id, report_json, TaskStatus::WaitingSubtasks)
        .unwrap();
    assert!(!ok, "CAS should fail when status does not match guard");

    let t = store.get(&t.id).unwrap().unwrap();
    assert_eq!(t.status, TaskStatus::Running);
}

// ── enqueue_child_if_agent_absent (atomic reviewer guard) ────────

#[test]
fn enqueue_child_if_agent_absent_inserts_when_no_duplicate() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "big goal")).unwrap();

    let child = store
        .enqueue_child_if_agent_absent(&parent.id, new_task("backend", "impl feature"))
        .unwrap();
    assert!(child.is_some(), "first enqueue should succeed");
    assert_eq!(child.unwrap().agent, "backend");
}

#[test]
fn enqueue_child_if_agent_absent_blocks_duplicate_agent() {
    let store = TaskStore::in_memory().unwrap();
    let parent = store.enqueue(new_task("pm", "big goal")).unwrap();

    // First reviewer child succeeds.
    let first = store
        .enqueue_child_if_agent_absent(&parent.id, new_task("reviewer", "review work"))
        .unwrap();
    assert!(first.is_some(), "first reviewer enqueue succeeds");

    // Second reviewer child is blocked (same agent).
    let second = store
        .enqueue_child_if_agent_absent(&parent.id, new_task("reviewer", "review again"))
        .unwrap();
    assert!(second.is_none(), "duplicate reviewer agent blocked");

    // Different agent still succeeds.
    let other = store
        .enqueue_child_if_agent_absent(&parent.id, new_task("frontend", "build UI"))
        .unwrap();
    assert!(other.is_some(), "different agent still succeeds");
}

// ── set_status_if (CAS for shutdown) ──────────────────────────────

#[test]
fn set_status_if_guard_matches() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();

    let ok = store
        .set_status_if(&t.id, TaskStatus::Queued, TaskStatus::Running)
        .unwrap();
    assert!(ok, "CAS should succeed when guard matches");

    let loaded = store.get(&t.id).unwrap().unwrap();
    assert_eq!(loaded.status, TaskStatus::Queued);
}

#[test]
fn set_status_if_guard_mismatch() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    store.set_status(&t.id, TaskStatus::Running).unwrap();

    // Guard is Queued but status is Running → should fail.
    let ok = store
        .set_status_if(&t.id, TaskStatus::Queued, TaskStatus::Queued)
        .unwrap();
    assert!(!ok, "CAS should fail when guard does not match");

    // Status unchanged.
    let loaded = store.get(&t.id).unwrap().unwrap();
    assert_eq!(loaded.status, TaskStatus::Running);
}

// ── M6: unknown status propagates error ────────────────────────────

#[test]
fn unknown_task_status_propagates_error() {
    let store = TaskStore::in_memory().unwrap();
    // Insert a row with an invalid status directly.
    let now = chrono::Utc::now().to_rfc3339();
    let verify_json = serde_json::to_string(&vec!["echo ok"]).unwrap();
    store
        .conn
        .execute(
            "INSERT INTO tasks (id, agent, goal, workspace, verify, status, created_at, updated_at)
                 VALUES ('bad_status_1', 'a', 'goal', '/tmp', ?1, 'BogusStatus', ?2, ?3)",
            rusqlite::params![verify_json, now, now],
        )
        .unwrap();

    let result = store.get("bad_status_1");
    assert!(
        result.is_err(),
        "unknown status should propagate as error, not silently default"
    );
}

#[test]
fn unknown_approval_status_propagates_error() {
    let store = TaskStore::in_memory().unwrap();
    let t = store.enqueue(new_task("a", "goal")).unwrap();
    let now = chrono::Utc::now().to_rfc3339();
    store
        .conn
        .execute(
            "INSERT INTO approvals (id, task_id, action, reason, status, created_at)
                 VALUES ('bad_appr_1', ?1, '{}', 'test', 'BogusApproval', ?2)",
            rusqlite::params![t.id, now],
        )
        .unwrap();

    let _result = store.pending_approvals();
    // The bogus row has status != 'Pending' so it won't appear in pending_approvals.
    // Instead, test via a direct query that loads all approvals.
    let result = store.conn.query_row(
        "SELECT id, task_id, action, reason, status, created_at, decided_at FROM approvals WHERE id = 'bad_appr_1'",
        [],
        |r| store.read_approval_row(r),
    );
    assert!(
        result.is_err(),
        "unknown approval status should propagate as error"
    );
}

// ── M8: atomic migration v1→v2 with user_version in transaction ──

#[test]
fn migration_v1_to_v2_atomic_includes_user_version() {
    let db = TmpDb::new();

    // Create a v1 schema DB.
    {
        let conn = rusqlite::Connection::open(db.path()).unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .unwrap();
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
                PRAGMA user_version = 1;",
        )
        .unwrap();
    }

    // Open with TaskStore → triggers atomic v1→v2 migration.
    let store = TaskStore::open(db.path()).unwrap();

    // Verify user_version is now 2.
    let ver: u32 = store
        .conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(ver, 2, "user_version should be 2 after migration");

    // Verify parent_id column exists.
    let has_parent_id: bool = store
        .conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'parent_id'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert!(
        has_parent_id,
        "parent_id column should exist after migration"
    );

    // Verify idx_task_parent index exists.
    let has_index: bool = store
        .conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_task_parent'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
        > 0;
    assert!(
        has_index,
        "idx_task_parent index should exist after migration"
    );
}
