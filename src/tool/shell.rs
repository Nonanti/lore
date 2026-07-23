//! Shell tool: runs shell commands through the policy gate.
//!
//! [`ShellTool`] executes `sh -c <command>` in the workspace root, captures
//! stdout + stderr + exit status, and truncates large output. Non-zero exit
//! returns `Ok(text)` with the exit code + output — only spawn failure,
//! timeout, and policy denial are `Err`.
//!
//! **Security boundary:** commands containing shell metacharacters (`;`, `|`,
//! `&&`, `||`, backticks, `$()`, `${}`) are denied unless `default_exec` is
//! `Allow`. This prevents chaining attacks (e.g. `ls; bash -i ...`). The
//! auto-allow list matches the first whitespace-delimited token of the
//! command — word-boundary matching prevents `ls` from allowing `lsof`.

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

/// Round down to the nearest UTF-8 char boundary ≤ `index`.
///
/// `String::truncate(new_len)` panics if `new_len` is not on a char
/// boundary. This helper finds the closest valid byte offset so we
/// can truncate safely even with multi-byte characters near the limit.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

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

        // Reject shell metacharacters to prevent chaining attacks
        // (e.g. `ls; bash -i`, `echo $(python...)`). Commands with
        // default_exec: Allow bypass this check.
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
        // Use read_to_end + from_utf8_lossy to handle binary output
        // gracefully instead of silently discarding InvalidData errors.
        let stdout_fut = async {
            let mut buf = Vec::new();
            if let Some(mut pipe) = stdout_pipe {
                use tokio::io::AsyncReadExt;
                if let Err(e) = AsyncReadExt::read_to_end(&mut pipe, &mut buf).await {
                    buf.extend_from_slice(format!("\n[stdout read error: {e}]").as_bytes());
                }
            }
            String::from_utf8_lossy(&buf).into_owned()
        };
        let stderr_fut = async {
            let mut buf = Vec::new();
            if let Some(mut pipe) = stderr_pipe {
                use tokio::io::AsyncReadExt;
                if let Err(e) = AsyncReadExt::read_to_end(&mut pipe, &mut buf).await {
                    buf.extend_from_slice(format!("\n[stderr read error: {e}]").as_bytes());
                }
            }
            String::from_utf8_lossy(&buf).into_owned()
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

                // Truncate BEFORE appending exit code so the marker is
                // always visible to the LLM (M1 fix). Use char-boundary-
                // safe truncation to avoid panics on multi-byte UTF-8
                // (C1 fix).
                let truncated = text.len() > self.max_output;
                if truncated {
                    let cap = floor_char_boundary(&text, self.max_output);
                    text.truncate(cap);
                    text.push_str(TRUNCATION_MARKER);
                }
                text.push_str(&format!("\n[exit code: {code}]"));
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

    #[test]
    fn floor_char_boundary_ascii() {
        // ASCII: every byte is a char boundary.
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 5), 5);
        assert_eq!(floor_char_boundary("hello", 100), 5); // index >= len → len
    }

    #[test]
    fn floor_char_boundary_multibyte() {
        // CJK: each char is 3 bytes. "你好" = 6 bytes.
        // Byte 4 is mid-char (char 你 = bytes 0-2, char 好 = bytes 3-5).
        assert_eq!(floor_char_boundary("你好", 4), 3);
        assert_eq!(floor_char_boundary("你好", 5), 3);
        assert_eq!(floor_char_boundary("你好", 3), 3);
        assert_eq!(floor_char_boundary("你好", 6), 6);
        // Emoji: 🎉 is 4 bytes (F0 9F 8E 89).
        assert_eq!(floor_char_boundary("🎉hello", 2), 0);
        assert_eq!(floor_char_boundary("🎉hello", 3), 0);
        assert_eq!(floor_char_boundary("🎉hello", 4), 4);
    }

    #[tokio::test]
    async fn shell_truncation_preserves_exit_code() {
        // M1 fix: exit code must always be visible even when output is truncated.
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root).with_max_output(32);
        // Generate enough output to trigger truncation.
        // Use a simple echo command that produces enough chars.
        let out = tool
            .run(r#"python3 -c 'print("A" * 200)' 2>/dev/null || printf 'AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_GGGG_HHHH_IIII_JJJJ'"#)
            .await
            .unwrap();
        // Exit code must appear AFTER truncation marker.
        assert!(
            out.contains("[exit code:"),
            "exit code must be present: {out}"
        );
        // Truncation marker must also appear.
        assert!(out.contains(TRUNCATION_MARKER), "must truncate: {out}");
        // Exit code comes AFTER truncation marker in the string.
        let trunc_idx = out.find(TRUNCATION_MARKER).unwrap();
        let exit_idx = out.find("[exit code:").unwrap();
        assert!(exit_idx > trunc_idx, "exit code after truncation: {out}");
    }

    #[tokio::test]
    async fn shell_shell_metacharacters_denied() {
        // C2 fix: commands with shell metacharacters are denied when
        // default_exec != Allow.
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Ask, // NOT Allow → metachars blocked
            ask_on_write: false,
        };
        let gate = Arc::new(Gate::new(p, Arc::new(DenyAll)));
        let tool = ShellTool::new(gate, root);
        let result = tool.run("ls; echo hello").await;
        assert!(result.is_err(), "semicolon chaining → denied");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("metacharacter"), "error: {err}");
    }
}
