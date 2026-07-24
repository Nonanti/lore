//! Task store: SQLite-backed task queue + approval inbox.
//!
//! [`TaskStore`] manages task lifecycle (Queued → Running → Completed/Failed)
//! and approval entries for human-in-the-loop decisions. Schema v1 is designed
//! so Phase 4's `parent_id` column can be added via `user_version` migration.
//!
//! Module layout:
//! - Types: [`TaskStatus`], [`Task`], [`NewTask`], [`ApprovalEntry`], [`ApprovalStatus`]
//! - [`approver`] — queue-based approval (stored in DB until answered)
//! - [`store`] — [`TaskStore`] implementation (SQLite CRUD, migrations, row mappers)

pub mod approver;
pub(crate) mod store;

use crate::error::{LoreError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Map a rusqlite error to a [`LoreError::Storage`] variant.
/// Shared between this module and [`store`].
pub(crate) fn sqlite_err(e: rusqlite::Error) -> LoreError {
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
    pub(crate) fn from_str(s: &str) -> Result<Self> {
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
    /// Parent task id (for team task hierarchy; None for standalone tasks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
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
    /// Parent task id (for team task hierarchy; None for standalone tasks).
    pub parent_id: Option<String>,
}

/// Approval entry: tracks a pending human decision on an agent action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApprovalEntry {
    /// ULID identifier.
    pub id: String,
    /// Which task this approval belongs to.
    pub task_id: String,
    /// The action being considered (JSON-serialized `policy::Action`).
    /// **Warning:** may contain command arguments with environment variable
    /// values — consider redacting before exposing to untrusted clients.
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
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            ApprovalStatus::Pending => "Pending",
            ApprovalStatus::Approved => "Approved",
            ApprovalStatus::Denied => "Denied",
        }
    }

    pub(crate) fn from_str(s: &str) -> Result<Self> {
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

// Re-export store type so that `crate::task::TaskStore` still works.
pub use store::TaskStore;

#[cfg(test)]
mod tests;
