//! Shell tool: runs shell commands through the policy gate.
//!
//! [`ShellTool`] executes `sh -c <command>` in the workspace root, captures
//! stdout + stderr + exit status, and truncates large output. Non-zero exit
//! returns `Ok(text)` with the exit code + output — only spawn failure,
//! timeout, and policy denial are `Err`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::{LoreError, Result};
use crate::policy::approval::Gate;
use crate::policy::Action;
use crate::tool::Tool;

/// Default command timeout (seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default max output size (bytes).
const DEFAULT_MAX_OUTPUT: usize = 32 * 1024;

/// Truncation marker appended when output exceeds the cap.
const TRUNCATION_MARKER: &str = "\n[... output truncated]";

/// Shell tool: runs commands via the policy gate.
#[derive(Clone, Debug)]
pub struct ShellTool {
    gate: Arc<Gate>,
    cwd: PathBuf,
    timeout: Duration,
    max_output: usize,
}

impl ShellTool {
    /// New shell tool with the given gate and working directory.
    pub fn new(gate: Arc<Gate>, cwd: PathBuf) -> Self {
        Self {
            gate,
            cwd,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_output: DEFAULT_MAX_OUTPUT,
        }
    }

    /// Builder: set command timeout (default 120 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: set max output size in bytes (default 32 KiB).
    pub fn with_max_output(mut self, max_output: usize) -> Self {
        self.max_output = max_output;
        self
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Runs a shell command and returns stdout, stderr, and exit code"
    }

    fn args_hint(&self) -> &str {
        r#""<command>" — e.g. "cargo test" or "ls -la""#
    }

    async fn run(&self, args: &str) -> Result<String> {
        let command = args.trim();
        if command.is_empty() {
            return Err(LoreError::InvalidInput("shell command required".into()));
        }

        // Policy gate check.
        self.gate
            .check(&Action::Exec {
                command: command.to_string(),
                cwd: self.cwd.clone(),
            })
            .await?;

        // Spawn the child with piped stdout/stderr. We take the pipes out
        // before the timeout so we can kill the process on timeout.
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| LoreError::Server(format!("failed to spawn shell: {e}")))?;

        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();

        // Read stdout and stderr concurrently into buffers.
        let stdout_fut = async {
            let mut buf = String::new();
            if let Some(mut pipe) = stdout_pipe {
                use tokio::io::AsyncReadExt;
                let _ = AsyncReadExt::read_to_string(&mut pipe, &mut buf).await;
            }
            buf
        };
        let stderr_fut = async {
            let mut buf = String::new();
            if let Some(mut pipe) = stderr_pipe {
                use tokio::io::AsyncReadExt;
                let _ = AsyncReadExt::read_to_string(&mut pipe, &mut buf).await;
            }
            buf
        };

        // Race: read output + wait for exit, against the timeout.
        let result = tokio::time::timeout(self.timeout, async {
            let (stdout, stderr, status) = tokio::join!(stdout_fut, stderr_fut, child.wait());
            (stdout, stderr, status)
        })
        .await;

        match result {
            Ok((stdout, stderr, status_result)) => {
                let status = status_result
                    .map_err(|e| LoreError::Server(format!("shell process error: {e}")))?;
                let code = status.code().unwrap_or(-1);

                let mut text = stdout;
                if !stderr.is_empty() {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&stderr);
                }
                text.push_str(&format!("\n[exit code: {code}]"));

                if text.len() > self.max_output {
                    text.truncate(self.max_output);
                    text.push_str(TRUNCATION_MARKER);
                }
                Ok(text)
            }
            Err(_) => {
                // Timeout — kill the child and reap the zombie.
                let _ = child.kill().await;
                let _ = child.wait().await;
                Err(LoreError::Server(format!(
                    "command timed out after {}s",
                    self.timeout.as_secs()
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::approval::{AllowAll, DenyAll};
    use crate::policy::{DefaultExec, Policy};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn allow_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root],
            auto_allow: vec!["echo".into(), "ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        Arc::new(Gate::new(p, Arc::new(AllowAll)))
    }

    fn deny_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root],
            auto_allow: vec!["echo".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Deny,
            ask_on_write: false,
        };
        Arc::new(Gate::new(p, Arc::new(DenyAll)))
    }

    #[tokio::test]
    async fn shell_echo_roundtrip() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root);
        let out = tool.run("echo hello world").await.unwrap();
        assert!(out.contains("hello world"), "output: {out}");
        assert!(out.contains("[exit code: 0]"), "output: {out}");
    }

    #[tokio::test]
    async fn shell_nonzero_exit_in_text() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root);
        let out = tool.run("exit 2").await.unwrap();
        assert!(
            out.contains("[exit code: 2]"),
            "non-zero exit in Ok text: {out}"
        );
    }

    #[tokio::test]
    async fn shell_timeout_kills_child() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root).with_timeout(Duration::from_millis(100));
        let result = tool.run("sleep 10").await;
        assert!(result.is_err(), "timeout must be Err");
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("timed out"), "error message: {msg}");
    }

    #[tokio::test]
    async fn shell_denied_command() {
        let root = std::env::temp_dir();
        let gate = deny_gate(root.clone());
        let tool = ShellTool::new(gate, root);
        // default_exec = Deny + DenyAll approver; "unknown_command" not on
        // auto_allow → denied.
        let result = tool.run("unknown_command").await;
        assert!(result.is_err(), "denied command must be Err");
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("policy denied"), "error: {msg}");
    }

    #[tokio::test]
    async fn shell_truncation() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root).with_max_output(64);
        // Generate output exceeding 64 bytes.
        let out = tool
            .run("python3 -c \"print('A' * 200)\" 2>/dev/null || echo AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_GGGG_HHHH_IIII_JJJJ_KKKK_LLLL")
            .await
            .unwrap();
        assert!(out.contains(TRUNCATION_MARKER), "truncated: {out}");
    }

    #[tokio::test]
    async fn shell_empty_args_error() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root);
        assert!(tool.run("").await.is_err());
        assert!(tool.run("  ").await.is_err());
    }

    // ── Additional edge-case tests ──────────────────────────────────────

    #[tokio::test]
    async fn shell_stderr_is_captured() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root);
        let out = tool
            .run("echo stdout_msg; echo stderr_msg >&2")
            .await
            .unwrap();
        assert!(out.contains("stdout_msg"), "stdout in output: {out}");
        assert!(out.contains("stderr_msg"), "stderr in output: {out}");
    }

    #[tokio::test]
    async fn shell_cwd_outside_roots_denied() {
        // Shell tool with a cwd outside the policy roots → Deny.
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        let gate = Arc::new(Gate::new(p, Arc::new(AllowAll)));
        let tool = ShellTool::new(gate, PathBuf::from("/etc"));
        let result = tool.run("echo hello").await;
        assert!(result.is_err(), "cwd outside roots → policy denied");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("policy denied"), "error: {err}");
    }

    #[tokio::test]
    async fn shell_deny_list_beats_auto_allow() {
        // Both auto_allow and deny contain "sudo" → deny wins.
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["sudo".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        let gate = Arc::new(Gate::new(p, Arc::new(AllowAll)));
        let tool = ShellTool::new(gate, root);
        let result = tool.run("sudo echo hello").await;
        assert!(result.is_err(), "deny-list beats auto_allow");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("policy denied"), "error: {err}");
    }

    #[tokio::test]
    async fn shell_truncation_boundary_exact() {
        let root = std::env::temp_dir();
        // Generate short output that fits within 200 bytes → no truncation.
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root.clone()).with_max_output(200);
        let out = tool.run("echo 1234567890").await.unwrap();
        assert!(
            !out.contains(TRUNCATION_MARKER),
            "should not truncate: {out}"
        );

        // Now set max_output very small → must truncate.
        let gate2 = allow_gate(root.clone());
        let tool2 = ShellTool::new(gate2, root).with_max_output(5);
        let out2 = tool2.run("echo 1234567890").await.unwrap();
        assert!(out2.contains(TRUNCATION_MARKER), "should truncate: {out2}");
    }
}
