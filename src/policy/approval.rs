//! Approval gate: human-in-the-loop escalation for agent actions.
//!
//! [`Gate`] combines a [`Policy`] with an [`Approver`] trait object.
//! Allow → `Ok(())`, Deny → [`LoreError::PolicyDenied`], Ask →
//! `approver.decide(...)`. The [`Approver`] trait is the seam for
//! future queue-backed approvers (Phase 3).

use std::sync::Arc;

use async_trait::async_trait;

use super::{Action, Verdict};
use crate::error::{LoreError, Result};

/// What the approver needs to decide on.
#[derive(Clone, Debug)]
pub struct ApprovalRequest {
    /// The action being considered.
    pub action: Action,
    /// Why the action wasn't auto-allowed.
    pub reason: String,
    /// Which agent is requesting (optional).
    pub agent: Option<String>,
}

/// Approver trait: decides whether a pending action is approved.
///
/// Implementations: [`DenyAll`] (safe default), [`AllowAll`] (tests/full-auto),
/// [`CliApprover`] (interactive y/N on terminal).
#[async_trait]
pub trait Approver: Send + Sync {
    /// Returns `true` if the action is approved, `false` if denied.
    async fn decide(&self, req: &ApprovalRequest) -> Result<bool>;
}

/// Always denies — safe default for production.
#[derive(Clone, Debug, Default)]
pub struct DenyAll;

#[async_trait]
impl Approver for DenyAll {
    async fn decide(&self, _req: &ApprovalRequest) -> Result<bool> {
        Ok(false)
    }
}

/// Always allows — useful for tests or full-auto mode.
#[derive(Clone, Debug, Default)]
pub struct AllowAll;

#[async_trait]
impl Approver for AllowAll {
    async fn decide(&self, _req: &ApprovalRequest) -> Result<bool> {
        Ok(true)
    }
}

/// Interactive CLI approver: prompts y/N on the terminal.
///
/// Uses `tokio::task::spawn_blocking` to read stdin without blocking
/// the async runtime.
#[derive(Clone, Debug, Default)]
pub struct CliApprover;

#[async_trait]
impl Approver for CliApprover {
    async fn decide(&self, req: &ApprovalRequest) -> Result<bool> {
        let action_desc = match &req.action {
            Action::Exec { command, cwd } => {
                format!("exec \"{command}\" (cwd: {})", cwd.display())
            }
            Action::Write { path } => {
                format!("write \"{}\"", path.display())
            }
        };
        let agent = req.agent.as_deref().unwrap_or("agent");
        let reason = &req.reason;
        let prompt = format!("[{agent}] {action_desc} — reason: {reason}\nApprove? [y/N] ");

        tokio::task::spawn_blocking(move || {
            use std::io::{self, BufRead, Write};
            print!("{prompt}");
            if let Err(e) = io::stdout().flush() {
                tracing::warn!("CLI approver: stdout flush failed: {e}");
            }
            let mut line = String::new();
            if let Err(e) = io::stdin().lock().read_line(&mut line) {
                tracing::warn!("CLI approver: stdin read_line failed: {e}");
            }
            line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes")
        })
        .await
        .map_err(|e| LoreError::Server(format!("cli approver spawn failed: {e}")))
    }
}

/// Policy gate: combines a policy and an approver.
///
/// `check(action)` evaluates the policy; Allow → `Ok(())`, Deny →
/// `PolicyDenied`, Ask → delegates to the approver.
pub struct Gate {
    /// Policy engine (pure evaluation).
    pub(crate) policy: super::Policy,
    /// Approver for Ask verdicts.
    pub approver: Arc<dyn Approver>,
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gate")
            .field("policy", &self.policy)
            .field("approver", &"Arc<dyn Approver>")
            .finish()
    }
}

impl Gate {
    /// New gate with the given policy and approver.
    pub fn new(policy: super::Policy, approver: Arc<dyn Approver>) -> Self {
        Self { policy, approver }
    }

    /// Check an action against the policy + approver.
    ///
    /// - Allow → `Ok(())`
    /// - Deny → `Err(LoreError::PolicyDenied(reason))`
    /// - Ask → `approver.decide(...)`, rejection → `PolicyDenied`
    pub async fn check(&self, action: &Action) -> Result<()> {
        match self.policy.evaluate(action) {
            Verdict::Allow => Ok(()),
            Verdict::Deny { reason } => Err(LoreError::PolicyDenied(reason)),
            Verdict::Ask { reason } => {
                let req = ApprovalRequest {
                    action: action.clone(),
                    reason,
                    agent: None,
                };
                if self.approver.decide(&req).await? {
                    Ok(())
                } else {
                    Err(LoreError::PolicyDenied("human denied approval".into()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Action, DefaultExec, Policy, SandboxMode};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Helper policy with a single root (/tmp) and predictable lists.
    fn test_policy(root: PathBuf) -> Policy {
        Policy {
            roots: vec![root],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
        }
    }

    // ── Gate with AllowAll ───────────────────────────────────────────────

    #[tokio::test]
    async fn gate_allow_all_verdict_allow() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(AllowAll));
        // "ls" is auto-allowed.
        let result = gate
            .check(&Action::Exec {
                command: "ls".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn gate_allow_all_verdict_ask_approved() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(AllowAll));
        // Unknown command → Ask; AllowAll approves.
        let result = gate
            .check(&Action::Exec {
                command: "unknown cmd".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn gate_allow_all_verdict_deny_still_denied() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(AllowAll));
        // "sudo" matches deny-list → Deny (approver not consulted).
        let result = gate
            .check(&Action::Exec {
                command: "sudo something".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoreError::PolicyDenied(_)),
            "deny verdict → PolicyDenied: {err:?}"
        );
    }

    // ── Gate with DenyAll ────────────────────────────────────────────────

    #[tokio::test]
    async fn gate_deny_all_verdict_allow() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(DenyAll));
        // Allow verdict → Ok regardless of approver.
        let result = gate
            .check(&Action::Exec {
                command: "ls".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn gate_deny_all_verdict_ask_rejected() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(DenyAll));
        // Unknown command → Ask; DenyAll rejects.
        let result = gate
            .check(&Action::Exec {
                command: "unknown cmd".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LoreError::PolicyDenied(_)));
    }

    #[tokio::test]
    async fn gate_deny_all_verdict_deny() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(DenyAll));
        // Deny verdict → PolicyDenied.
        let result = gate
            .check(&Action::Exec {
                command: "sudo something".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_err());
    }

    // ── Gate: write actions ──────────────────────────────────────────────

    #[tokio::test]
    async fn gate_write_inside_root_allowed() {
        let root = std::env::temp_dir();
        let p = test_policy(root.clone());
        let gate = Gate::new(p, Arc::new(AllowAll));
        let result = gate
            .check(&Action::Write {
                path: root.join("file.txt"),
            })
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn gate_write_ask_on_write_asks_approver() {
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: true,
            sandbox_exec: SandboxMode::Off,
        };
        let gate = Gate::new(p, Arc::new(AllowAll));
        let result = gate
            .check(&Action::Write {
                path: root.join("file.txt"),
            })
            .await;
        // Ask + AllowAll → approved.
        assert!(result.is_ok());

        // Ask + DenyAll → denied.
        let gate2 = Gate::new(
            Policy {
                roots: vec![root.clone()],
                auto_allow: vec!["ls".into()],
                deny: vec!["sudo".into()],
                default_exec: DefaultExec::Ask,
                ask_on_write: true,
                sandbox_exec: SandboxMode::Off,
            },
            Arc::new(DenyAll),
        );
        let result2 = gate2
            .check(&Action::Write {
                path: root.join("file.txt"),
            })
            .await;
        assert!(result2.is_err());
    }

    // ── Gate: custom Approver returning false ────────────────────────────

    /// An approver that always returns false, providing a reason string.
    #[derive(Debug)]
    struct RejectWithReason;

    #[async_trait]
    impl Approver for RejectWithReason {
        async fn decide(&self, _req: &ApprovalRequest) -> Result<bool> {
            Ok(false)
        }
    }

    #[tokio::test]
    async fn gate_ask_custom_approver_returns_false() {
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
        }; // Everything goes to Ask.
        let gate = Gate::new(p, Arc::new(RejectWithReason));
        let result = gate
            .check(&Action::Exec {
                command: "echo hello".into(),
                cwd: root,
            })
            .await;
        assert!(result.is_err(), "Ask + false approver → denied");
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoreError::PolicyDenied(_)),
            "PolicyDenied: {err:?}"
        );
    }

    #[tokio::test]
    async fn gate_debug_impl_works() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root);
        let gate = Gate::new(p, Arc::new(AllowAll));
        let debug_str = format!("{gate:?}");
        assert!(debug_str.contains("Gate"), "Debug output: {debug_str}");
        assert!(
            debug_str.contains("Arc<dyn Approver>"),
            "Debug placeholder for approver: {debug_str}"
        );
    }
}
