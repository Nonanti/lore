//! Policy engine: pure decision core for agent autonomy.
//!
//! Every dangerous action (shell command, file write) passes through
//! [`Policy::evaluate`], which returns a [`Verdict`] (Allow / Ask / Deny).
//! Deny-list wins over allow-list; path containment uses canonicalization
//! of the nearest existing ancestor to handle not-yet-created files.

pub mod approval;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What the agent wants to do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Run a shell command in the given working directory.
    Exec { command: String, cwd: PathBuf },
    /// Write (create or edit) a file at the given path.
    Write { path: PathBuf },
}

/// Policy verdict: allow, ask for approval, or deny.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Action is allowed without further checks.
    Allow,
    /// Action needs human approval (reason provided).
    Ask { reason: String },
    /// Action is denied (reason provided).
    Deny { reason: String },
}

/// Default verdict for commands matching neither `auto_allow` nor `deny`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DefaultExec {
    /// Ask for human approval (safest default).
    #[default]
    Ask,
    /// Allow anything not on the deny list.
    Allow,
    /// Deny anything not on the auto-allow list.
    Deny,
}

/// Policy configuration. Pure data — no I/O in evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Policy {
    /// Allowed workspace roots (path containment checks).
    pub roots: Vec<PathBuf>,
    /// Command prefixes allowed without approval (e.g. `"cargo test"`).
    pub auto_allow: Vec<String>,
    /// Command substrings always denied (e.g. `"sudo"`, `"rm -rf /"`).
    /// Checked first — deny-list wins.
    pub deny: Vec<String>,
    /// Default verdict for exec commands matching neither list.
    pub default_exec: DefaultExec,
    /// Whether writes inside roots require approval.
    pub ask_on_write: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self::default_for(PathBuf::from("."))
    }
}

impl Policy {
    /// Sensible defaults for personal-use with the given workspace root.
    ///
    /// Deny-list pre-seeded with dangerous commands; `ask_on_write` defaults
    /// to `false` (writes inside roots are allowed); `default_exec` is `Ask`.
    pub fn default_for(root: PathBuf) -> Self {
        Self {
            roots: vec![root],
            auto_allow: vec![
                "cargo test".into(),
                "cargo check".into(),
                "cargo clippy".into(),
                "cargo fmt".into(),
                "cargo build".into(),
                "ls".into(),
                "cat".into(),
                "head".into(),
                "tail".into(),
                "find".into(),
                "rg".into(),
                "git status".into(),
                "git log".into(),
                "git diff".into(),
                "echo".into(),
                "pwd".into(),
                "which".into(),
                "wc".into(),
                "sort".into(),
                "uniq".into(),
            ],
            deny: vec![
                "sudo".into(),
                "rm -rf /".into(),
                "git push --force".into(),
                "shutdown".into(),
                "reboot".into(),
                "systemctl".into(),
                "mkfs".into(),
                "dd".into(),
                "chmod 777".into(),
                "chown".into(),
                "passwd".into(),
                "crontab".into(),
                "curl".into(), // use the web tool instead
                "wget".into(),
            ],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
        }
    }

    /// Evaluate an action against this policy (pure, no I/O).
    ///
    /// Evaluation order:
    /// 1. Deny-list substring match → Deny (always wins).
    /// 2. Exec: auto_allow prefix match → Allow; cwd outside roots → Deny;
    ///    else default_exec.
    /// 3. Write: path inside root → Allow (or Ask if ask_on_write);
    ///    outside → Deny.
    pub fn evaluate(&self, action: &Action) -> Verdict {
        match action {
            Action::Exec { command, cwd } => {
                // Deny-list wins (substring match).
                for d in &self.deny {
                    if command.contains(d) {
                        return Verdict::Deny {
                            reason: format!("command matches deny-list entry: \"{d}\""),
                        };
                    }
                }
                // Cwd must be inside a root.
                if !self.is_inside_root(cwd) {
                    return Verdict::Deny {
                        reason: "cwd is outside allowed roots".into(),
                    };
                }
                // Auto-allow prefix match.
                for a in &self.auto_allow {
                    if command.starts_with(a) {
                        return Verdict::Allow;
                    }
                }
                // Default.
                match self.default_exec {
                    DefaultExec::Allow => Verdict::Allow,
                    DefaultExec::Ask => Verdict::Ask {
                        reason: "command not on auto-allow list".into(),
                    },
                    DefaultExec::Deny => Verdict::Deny {
                        reason: "command not on auto-allow list".into(),
                    },
                }
            }
            Action::Write { path } => {
                // Deny-list doesn't apply to write actions (no command to
                // match), but writes to sensitive paths are blocked later
                // via root containment.
                if self.is_inside_root(path) {
                    if self.ask_on_write {
                        Verdict::Ask {
                            reason: "write action requires approval (ask_on_write)".into(),
                        }
                    } else {
                        Verdict::Allow
                    }
                } else {
                    Verdict::Deny {
                        reason: "write path is outside allowed roots".into(),
                    }
                }
            }
        }
    }

    /// Check whether a path is inside one of the allowed roots.
    ///
    /// Uses canonicalization; for files that don't exist yet,
    /// canonicalizes the nearest existing ancestor. Rejects `..`
    /// traversal and symlink escapes.
    fn is_inside_root(&self, path: &Path) -> bool {
        for root in &self.roots {
            if path_contained(path, root) {
                return true;
            }
        }
        false
    }

    /// Load policy from a JSON file.
    pub fn load(path: &Path) -> crate::error::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| crate::error::LoreError::Storage(e.to_string()))?;
        serde_json::from_str(&text).map_err(crate::error::LoreError::from)
    }

    /// Save policy to a JSON file (creates parent dirs).
    pub fn save(&self, path: &Path) -> crate::error::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| crate::error::LoreError::Storage(e.to_string()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(crate::error::LoreError::from)?;
        std::fs::write(path, text).map_err(|e| crate::error::LoreError::Storage(e.to_string()))
    }
}

/// Check whether `path` is contained within `root`.
///
/// Canonicalizes both sides; for `path` that doesn't yet exist, walks
/// ancestors to find the nearest existing one and canonicalizes that,
/// then appends the remaining non-existing suffix. Rejects `..`
/// traversal and symlink escapes.
fn path_contained(path: &Path, root: &Path) -> bool {
    // Reject traversal components outright.
    for component in path.components() {
        if component == std::path::Component::ParentDir {
            return false;
        }
    }

    // Canonicalize root (must exist).
    let root_canon = match std::fs::canonicalize(root) {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Find the nearest existing ancestor of path and canonicalize it.
    let canon = canonicalize_nearest_existing(path);
    let canon = match canon {
        Some(c) => c,
        None => return false,
    };

    canon.starts_with(&root_canon)
}

/// Canonicalize the nearest existing ancestor of `path`, then append
/// the remaining (non-existing) suffix.
///
/// Returns `None` if no ancestor exists at all (even the root `/`
/// on Unix should exist, but handle the edge case).
fn canonicalize_nearest_existing(path: &Path) -> Option<PathBuf> {
    // Walk ancestors from path upward until we find one that exists.
    for i in 0..path.components().count() {
        let ancestor: PathBuf = path
            .components()
            .take(path.components().count() - i)
            .collect();
        if let Ok(canon) = std::fs::canonicalize(&ancestor) {
            // Append the remaining suffix (non-existing part).
            let suffix: PathBuf = path
                .components()
                .skip(path.components().count() - i)
                .collect();
            if suffix.components().count() == 0 {
                return Some(canon);
            }
            return Some(canon.join(suffix));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Helper: policy with a single root (temp dir) and minimal lists.
    fn test_policy(root: PathBuf) -> Policy {
        Policy {
            roots: vec![root],
            auto_allow: vec!["cargo test".into(), "ls".into()],
            deny: vec!["sudo".into(), "rm -rf /".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
        }
    }

    // ── Deny-list precedence ─────────────────────────────────────────────

    #[test]
    fn deny_list_wins_over_auto_allow() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["sudo".into()], // deliberately allowed
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Exec {
            command: "sudo apt install something".into(),
            cwd: root,
        });
        assert!(
            matches!(v, Verdict::Deny { .. }),
            "deny-list must win: {v:?}"
        );
    }

    #[test]
    fn deny_list_matches_substring() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        // "sudo" is a substring of "sudo rm /var/log".
        let v = p.evaluate(&Action::Exec {
            command: "sudo rm /var/log".into(),
            cwd: root.clone(),
        });
        assert!(matches!(v, Verdict::Deny { .. }));
        // "rm -rf /" is a substring of "rm -rf / --no-preserve-root".
        let v2 = p.evaluate(&Action::Exec {
            command: "rm -rf / --no-preserve-root".into(),
            cwd: root,
        });
        assert!(matches!(v2, Verdict::Deny { .. }));
    }

    // ── Auto-allow prefix matching ───────────────────────────────────────

    #[test]
    fn auto_allow_prefix_match() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        // "cargo test" prefix matches "cargo test -- --test-threads=1".
        let v = p.evaluate(&Action::Exec {
            command: "cargo test -- --test-threads=1".into(),
            cwd: root.clone(),
        });
        assert_eq!(v, Verdict::Allow);
        // "ls" prefix matches "ls -la /tmp".
        let v2 = p.evaluate(&Action::Exec {
            command: "ls -la /tmp".into(),
            cwd: root.clone(),
        });
        assert_eq!(v2, Verdict::Allow);
        // "lsx" does prefix-match "ls" — this IS auto-allowed.
        let v3 = p.evaluate(&Action::Exec {
            command: "lsx -la".into(),
            cwd: root,
        });
        assert_eq!(v3, Verdict::Allow);
        // A command that doesn't start with any auto_allow entry → Ask.
        let v4 = p.evaluate(&Action::Exec {
            command: "some-unknown-command".into(),
            cwd: PathBuf::from("/tmp"),
        });
        assert!(matches!(v4, Verdict::Ask { .. }));
    }

    // ── DefaultExec branches ─────────────────────────────────────────────

    #[test]
    fn default_exec_ask() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Exec {
            command: "some unknown command".into(),
            cwd: root,
        });
        assert!(matches!(v, Verdict::Ask { .. }));
    }

    #[test]
    fn default_exec_allow() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Exec {
            command: "some unknown command".into(),
            cwd: root,
        });
        assert_eq!(v, Verdict::Allow);
    }

    #[test]
    fn default_exec_deny() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Deny,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Exec {
            command: "some unknown command".into(),
            cwd: root,
        });
        assert!(matches!(v, Verdict::Deny { .. }));
    }

    // ── Cwd outside roots → Deny ─────────────────────────────────────────

    #[test]
    fn exec_cwd_outside_roots_denied() {
        let p = Policy {
            roots: vec![PathBuf::from("/tmp")],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Exec {
            command: "ls".into(),
            cwd: PathBuf::from("/etc"),
        });
        assert!(matches!(v, Verdict::Deny { .. }), "cwd outside root → deny");
    }

    // ── Write: inside root → Allow, outside → Deny ──────────────────────

    #[test]
    fn write_inside_root_allowed() {
        let root = PathBuf::from("/tmp");
        let p = test_policy(root.clone());
        let v = p.evaluate(&Action::Write {
            path: root.join("file.txt"),
        });
        // Note: path containment depends on canonicalization. /tmp exists.
        // The exact result depends on whether /tmp/file.txt is contained
        // within /tmp. Since the file doesn't exist yet, we canonicalize
        // /tmp (which exists) and append "file.txt".
        assert!(matches!(v, Verdict::Allow));
    }

    #[test]
    fn write_outside_root_denied() {
        let p = Policy {
            roots: vec![PathBuf::from("/tmp")],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Write {
            path: PathBuf::from("/etc/secret.txt"),
        });
        assert!(matches!(v, Verdict::Deny { .. }));
    }

    #[test]
    fn write_ask_on_write_true() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: true,
        };
        let v = p.evaluate(&Action::Write {
            path: root.join("file.txt"),
        });
        assert!(matches!(v, Verdict::Ask { .. }));
    }

    // ── Path containment: traversal ──────────────────────────────────────

    #[test]
    fn path_traversal_rejected() {
        // Contains ".." → immediately rejected.
        assert!(!path_contained(
            &PathBuf::from("/tmp/../etc/passwd"),
            &PathBuf::from("/tmp"),
        ));
    }

    #[test]
    fn path_traversal_in_exec() {
        let p = Policy {
            roots: vec![PathBuf::from("/tmp")],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        let v = p.evaluate(&Action::Write {
            path: PathBuf::from("/tmp/../etc/passwd"),
        });
        assert!(matches!(v, Verdict::Deny { .. }), "traversal → deny");
    }

    // ── Path containment: not-yet-existing paths ─────────────────────────

    #[test]
    fn not_yet_existing_path_inside_root() {
        let root = std::env::temp_dir();
        // Create a subdirectory so we have a known existing ancestor.
        let subdir = root.join("lore-policy-test-nested");
        std::fs::create_dir_all(&subdir).unwrap();
        let nonexistent = subdir.join("deep/not-yet-created.txt");
        assert!(path_contained(&nonexistent, &root));
        std::fs::remove_dir_all(&subdir).ok();
    }

    #[test]
    fn not_yet_existing_path_outside_root() {
        let root = std::env::temp_dir();
        // A nonexistent path under /etc is outside the temp root.
        let nonexistent = PathBuf::from("/etc/nonexistent/deep/path.txt");
        assert!(!path_contained(&nonexistent, &root));
    }

    // ── Path containment: cwd outside roots ──────────────────────────────

    #[test]
    fn path_contained_with_existing_root() {
        let root = std::env::temp_dir();
        let inside = root.join("some_file.txt");
        // /tmp exists → canonicalizes fine.
        assert!(path_contained(&inside, &root));
    }

    #[test]
    fn path_contained_cwd_outside_roots() {
        let root = std::env::temp_dir();
        let outside = PathBuf::from("/etc/passwd");
        assert!(!path_contained(&outside, &root));
    }

    // ── Canonicalize nearest existing ancestor ───────────────────────────

    #[test]
    fn canonicalize_nearest_existing_works() {
        let root = std::env::temp_dir();
        let subdir = root.join("lore-canon-test");
        std::fs::create_dir_all(&subdir).unwrap();
        // subdir exists → canonicalizes it, appends "new.txt".
        let path = subdir.join("new.txt");
        let canon = canonicalize_nearest_existing(&path);
        assert!(canon.is_some());
        let canon = canon.unwrap();
        assert!(canon.starts_with(&root));
        assert!(canon.ends_with("new.txt"));
        std::fs::remove_dir_all(&subdir).ok();
    }

    #[test]
    fn canonicalize_nearest_existing_fully_nonexistent() {
        // A path where even the top-level parent doesn't exist.
        // On Unix "/" always exists, so this will canonicalize "/" then
        // append the rest — which won't be inside the temp root.
        let root = std::env::temp_dir();
        let path = PathBuf::from("/nonexistent_root/deep/path.txt");
        let result = canonicalize_nearest_existing(&path);
        // "/" exists on Unix, so we get Some(PathBuf("/nonexistent_root/deep/path.txt"))
        // which is NOT inside the temp root.
        if let Some(canon) = result {
            assert!(!canon.starts_with(&root));
        }
    }

    // ── JSON load/save ───────────────────────────────────────────────────

    #[test]
    fn policy_json_roundtrip() {
        let dir = std::env::temp_dir().join("lore-policy-json-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("policy.json");
        let p = Policy::default_for(PathBuf::from("/my/project"));
        p.save(&path).unwrap();
        let loaded = Policy::load(&path).unwrap();
        assert_eq!(loaded.roots, p.roots);
        assert_eq!(loaded.auto_allow, p.auto_allow);
        assert_eq!(loaded.deny, p.deny);
        assert_eq!(loaded.default_exec, p.default_exec);
        assert_eq!(loaded.ask_on_write, p.ask_on_write);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── default_for() sanity ──────────────────────────────────────────────

    #[test]
    fn default_for_has_expected_deny_list() {
        let p = Policy::default_for(PathBuf::from("/my/project"));
        // Core dangerous commands should be in the deny list.
        assert!(p.deny.iter().any(|d| d == "sudo"), "sudo in deny list");
        assert!(
            p.deny.iter().any(|d| d == "rm -rf /"),
            "rm -rf / in deny list"
        );
        assert!(
            p.deny.iter().any(|d| d == "shutdown"),
            "shutdown in deny list"
        );
        // Default exec should be Ask (safest).
        assert_eq!(p.default_exec, DefaultExec::Ask);
        // ask_on_write should be false (writes inside roots are allowed).
        assert!(!p.ask_on_write);
        // Should have at least some auto_allow entries.
        assert!(!p.auto_allow.is_empty(), "auto_allow should not be empty");
    }
}
