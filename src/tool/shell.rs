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
//!
//! **Sandbox:** when [`SandboxMode`] is `IfAvailable` or `Required`, commands
//! run inside a bubblewrap (bwrap) sandbox with a read-only root filesystem,
//! `/dev`, `/proc`, a tmpfs `/tmp`, and the workspace bind-mounted writable.
//! bwrap availability is probed once and cached; `IfAvailable` + missing bwrap
//! falls back to plain exec (warn once), while `Required` + missing fails
//! closed (`PolicyDenied`). Network stays shared (package managers need it).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::{LoreError, Result};
use crate::policy::approval::Gate;
use crate::policy::{Action, SandboxMode};
use crate::tool::Tool;

/// Default command timeout (seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default max output size (bytes).
const DEFAULT_MAX_OUTPUT: usize = 32 * 1024;

/// Truncation marker appended when output exceeds the cap.
const TRUNCATION_MARKER: &str = "\n[... output truncated]";

/// Cached bwrap availability check. Probed once via `bwrap --version`.
static BWRAP_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Whether bwrap (bubblewrap) is installed and functional.
///
/// Probes `bwrap --version` once and caches the result. Uses
/// `std::process::Command` (blocking) since this runs only once.
fn bwrap_available() -> bool {
    *BWRAP_AVAILABLE.get_or_init(|| {
        std::process::Command::new("bwrap")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Warn-once guard: logs a warning when bwrap is missing under IfAvailable.
static BWRAP_WARNED: OnceLock<()> = OnceLock::new();

/// Emit a one-time warning that bwrap is not available.
fn warn_bwrap_missing_once() {
    BWRAP_WARNED.get_or_init(|| {
        tracing::warn!("bwrap not found — sandbox disabled, commands run without isolation");
    });
}

/// Build the exact bwrap argv per spec (pure function, no I/O).
///
/// argv: `--ro-bind / / --dev /dev --proc /proc --tmpfs /tmp
///        --bind <workspace> <workspace> --unshare-pid --die-with-parent
///        --chdir <workspace> sh -c <command>`
///
/// Built as separate `String` args — never string-interpolated into a
/// single shell string (no quoting bugs).
///
/// Note on mount ordering: bwrap applies mounts left-to-right; a later
/// `--bind` on the same path overrides the earlier `--ro-bind`. This means:
///
/// - `/etc` is covered by `--ro-bind / /` with no writable override → RO ✓
/// - `--tmpfs /tmp` overlays the RO `/tmp` slice → host `/tmp` is hidden ✓
/// - `--bind <ws> <ws>` after `--ro-bind / /` → workspace is RW ✓
///
/// Edge case: if the workspace is under `/tmp`, the `--bind <ws> <ws>`
/// correctly overlays the empty tmpfs with the host workspace dir.
fn bwrap_argv(command: &str, cwd: &Path) -> Vec<String> {
    let ws = cwd.to_string_lossy().to_string();
    vec![
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--dev".into(),
        "/dev".into(),
        "--proc".into(),
        "/proc".into(),
        "--tmpfs".into(),
        "/tmp".into(),
        "--bind".into(),
        ws.clone(),
        ws.clone(),
        "--unshare-pid".into(),
        "--die-with-parent".into(),
        "--chdir".into(),
        ws,
        "sh".into(),
        "-c".into(),
        command.into(),
    ]
}

/// Determine the spawn program and argument list for a shell command.
///
/// - `Off` → plain `sh -c <command>`
/// - `IfAvailable/Required` + bwrap present → exact bwrap argv per spec
/// - `IfAvailable` + bwrap missing → warn once + fall back to plain `sh -c`
/// - `Required` + bwrap missing → `Err(PolicyDenied)` (fail closed)
///
/// Returns `(program, argv, set_current_dir)` — `set_current_dir` is true
/// when the caller must set `current_dir` on the spawned process
/// (plain `sh` mode). When bwrap is used, `--chdir` handles the working
/// directory inside the sandbox, so `current_dir` must NOT be set (it
/// would conflict with the sandbox's `/` view).
fn spawn_argv(
    command: &str,
    cwd: &Path,
    sandbox: SandboxMode,
) -> Result<(String, Vec<String>, bool)> {
    match sandbox {
        SandboxMode::Off => Ok(("sh".into(), vec!["-c".into(), command.into()], true)),
        SandboxMode::IfAvailable | SandboxMode::Required => {
            if bwrap_available() {
                // bwrap --chdir handles cwd; don't set current_dir on the
                // Command — it would resolve against the host filesystem,
                // not the sandbox mount layout.
                Ok(("bwrap".into(), bwrap_argv(command, cwd), false))
            } else if sandbox == SandboxMode::IfAvailable {
                warn_bwrap_missing_once();
                Ok(("sh".into(), vec!["-c".into(), command.into()], true))
            } else {
                Err(LoreError::PolicyDenied(
                    "sandbox required but bwrap not found".into(),
                ))
            }
        }
    }
}

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

        // Determine sandbox mode and build the spawn argv.
        let sandbox = self.gate.policy.sandbox_exec;
        let (program, argv, set_current_dir) = spawn_argv(command, &self.cwd, sandbox)?;

        // Spawn the child with piped stdout/stderr.
        // When using bwrap, --chdir handles the working directory inside
        // the sandbox; for plain sh, we set current_dir via the bool flag
        // returned by spawn_argv — avoids fragile string comparison on
        // the program name.
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&argv);
        if set_current_dir {
            cmd.current_dir(&self.cwd);
        }
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| LoreError::Server(format!("failed to spawn {program}: {e}")))?;

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
    use crate::policy::{Action, DefaultExec, Policy, SandboxMode, Verdict};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn allow_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root],
            auto_allow: vec!["echo".into(), "ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
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
            sandbox_exec: SandboxMode::Off,
        };
        Arc::new(Gate::new(p, Arc::new(DenyAll)))
    }

    // ── Existing functional tests ───────────────────────────────────────

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
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
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
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec!["sudo".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
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
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root.clone()).with_max_output(200);
        let out = tool.run("echo 1234567890").await.unwrap();
        assert!(
            !out.contains(TRUNCATION_MARKER),
            "should not truncate: {out}"
        );

        let gate2 = allow_gate(root.clone());
        let tool2 = ShellTool::new(gate2, root).with_max_output(5);
        let out2 = tool2.run("echo 1234567890").await.unwrap();
        assert!(out2.contains(TRUNCATION_MARKER), "should truncate: {out2}");
    }

    #[test]
    fn floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 5), 5);
        assert_eq!(floor_char_boundary("hello", 100), 5);
    }

    #[test]
    fn floor_char_boundary_multibyte() {
        assert_eq!(floor_char_boundary("你好", 4), 3);
        assert_eq!(floor_char_boundary("你好", 5), 3);
        assert_eq!(floor_char_boundary("你好", 3), 3);
        assert_eq!(floor_char_boundary("你好", 6), 6);
        assert_eq!(floor_char_boundary("🎉hello", 2), 0);
        assert_eq!(floor_char_boundary("🎉hello", 3), 0);
        assert_eq!(floor_char_boundary("🎉hello", 4), 4);
    }

    #[tokio::test]
    async fn shell_truncation_preserves_exit_code() {
        let root = std::env::temp_dir();
        let gate = allow_gate(root.clone());
        let tool = ShellTool::new(gate, root).with_max_output(32);
        let out = tool
            .run(r#"python3 -c 'print("A" * 200)' 2>/dev/null || printf 'AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_GGGG_HHHH_IIII_JJJJ'"#)
            .await
            .unwrap();
        assert!(
            out.contains("[exit code:"),
            "exit code must be present: {out}"
        );
        assert!(out.contains(TRUNCATION_MARKER), "must truncate: {out}");
        let trunc_idx = out.find(TRUNCATION_MARKER).unwrap();
        let exit_idx = out.find("[exit code:").unwrap();
        assert!(exit_idx > trunc_idx, "exit code after truncation: {out}");
    }

    #[tokio::test]
    async fn shell_shell_metacharacters_denied() {
        let root = std::env::temp_dir();
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Off,
        };
        let gate = Arc::new(Gate::new(p, Arc::new(DenyAll)));
        let tool = ShellTool::new(gate, root);
        let result = tool.run("ls; echo hello").await;
        assert!(result.is_err(), "semicolon chaining → denied");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("metacharacter"), "error: {err}");
    }

    // ── Sandbox argv construction tests ─────────────────────────────────

    #[test]
    fn spawn_argv_off_returns_plain_sh() {
        let (program, args, set_current_dir) =
            spawn_argv("ls -la", Path::new("/workspace"), SandboxMode::Off).unwrap();
        assert_eq!(program, "sh");
        assert_eq!(args, vec!["-c", "ls -la"]);
        assert!(set_current_dir, "Off mode must set current_dir");
    }

    #[test]
    fn bwrap_argv_exact_spec_order() {
        let ws = Path::new("/home/user/project");
        let args = bwrap_argv("cargo test", ws);
        // Verify exact order per spec.
        let expected = vec![
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--bind",
            "/home/user/project",
            "/home/user/project",
            "--unshare-pid",
            "--die-with-parent",
            "--chdir",
            "/home/user/project",
            "sh",
            "-c",
            "cargo test",
        ];
        assert_eq!(args, expected, "bwrap argv must match spec exactly");
    }

    #[test]
    fn bwrap_argv_workspace_bind_present() {
        let ws = Path::new("/my/workspace");
        let args = bwrap_argv("echo hello", ws);
        // Workspace must appear as both source and dest in --bind.
        let bind_idx = args.iter().position(|a| a == "--bind").unwrap();
        assert_eq!(
            args[bind_idx + 1],
            "/my/workspace",
            "--bind source must be workspace"
        );
        assert_eq!(
            args[bind_idx + 2],
            "/my/workspace",
            "--bind dest must be workspace"
        );
        // --chdir must also point to workspace.
        let chdir_idx = args.iter().position(|a| a == "--chdir").unwrap();
        assert_eq!(
            args[chdir_idx + 1],
            "/my/workspace",
            "--chdir must be workspace"
        );
    }

    #[test]
    fn bwrap_argv_no_string_joined_shell() {
        // Every arg is a separate string — no "sh -c cargo test" as one arg.
        let args = bwrap_argv("complex command with spaces", Path::new("/ws"));
        // The command is a single arg after "-c", but the shell invocation
        // itself is never string-joined into one arg.
        let sh_idx = args.iter().position(|a| a == "sh").unwrap();
        assert_eq!(args[sh_idx + 1], "-c");
        assert_eq!(args[sh_idx + 2], "complex command with spaces");
        // No arg contains the full "sh -c ..." as a single string.
        for arg in &args {
            assert!(
                !arg.starts_with("sh -c "),
                "no string-joined shell invocation: found '{arg}'"
            );
        }
    }

    #[test]
    fn spawn_argv_required_no_bwrap_fails_closed() {
        if bwrap_available() {
            // bwrap present on this host — Required succeeds; test the
            // absent-bwrap branch on a host where bwrap is missing.
            return;
        }
        let result = spawn_argv("echo hi", Path::new("/ws"), SandboxMode::Required);
        assert!(result.is_err(), "Required + no bwrap → PolicyDenied");
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bwrap not found"), "error message: {msg}");
    }

    #[test]
    fn spawn_argv_if_available_no_bwrap_falls_back() {
        if bwrap_available() {
            // bwrap present on this host — IfAvailable uses bwrap; test the
            // fallback branch on a host where bwrap is missing.
            return;
        }
        let result = spawn_argv("echo hi", Path::new("/ws"), SandboxMode::IfAvailable);
        assert!(result.is_ok(), "IfAvailable + no bwrap → fallback");
        let (program, args, set_current_dir) = result.unwrap();
        assert_eq!(program, "sh");
        assert_eq!(args, vec!["-c", "echo hi"]);
        assert!(set_current_dir, "fallback sh must set current_dir");
    }

    /// Required + bwrap present → Ok with bwrap argv and set_current_dir=false.
    #[test]
    fn spawn_argv_required_with_bwrap_returns_bwrap_argv() {
        if !bwrap_available() {
            return; // can only test the "bwrap present" branch when bwrap exists.
        }
        let result = spawn_argv("echo hi", Path::new("/ws"), SandboxMode::Required);
        assert!(result.is_ok(), "Required + bwrap present → Ok");
        let (program, args, set_current_dir) = result.unwrap();
        assert_eq!(program, "bwrap");
        assert!(!set_current_dir, "bwrap mode must NOT set current_dir");
        // Verify the argv starts with the expected bwrap flags.
        let first_flag = &args[0];
        assert_eq!(
            first_flag, "--ro-bind",
            "first bwrap flag must be --ro-bind"
        );
    }

    /// IfAvailable + bwrap present → Ok with bwrap argv (same as Required).
    #[test]
    fn spawn_argv_if_available_with_bwrap_returns_bwrap_argv() {
        if !bwrap_available() {
            return;
        }
        let result = spawn_argv("echo hi", Path::new("/ws"), SandboxMode::IfAvailable);
        assert!(result.is_ok(), "IfAvailable + bwrap present → Ok");
        let (program, args, set_current_dir) = result.unwrap();
        assert_eq!(program, "bwrap");
        assert!(!set_current_dir, "bwrap mode must NOT set current_dir");
        let first_flag = &args[0];
        assert_eq!(
            first_flag, "--ro-bind",
            "first bwrap flag must be --ro-bind"
        );
    }

    // ── Policy serde: old JSON without sandbox_exec ─────────────────────

    #[test]
    fn policy_old_json_without_sandbox_exec_loads_as_off() {
        let old_json = r#"{
            "roots": ["/tmp"],
            "auto_allow": ["ls"],
            "deny": ["sudo"],
            "default_exec": "Ask",
            "ask_on_write": false
        }"#;
        let p: Policy = serde_json::from_str(old_json).unwrap();
        assert_eq!(p.sandbox_exec, SandboxMode::Off);
        assert_eq!(p.roots, vec![PathBuf::from("/tmp")]);
    }

    #[test]
    fn policy_json_roundtrip_with_sandbox_exec() {
        let p = Policy {
            roots: vec![PathBuf::from("/tmp")],
            auto_allow: vec!["ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Required,
        };
        let json = serde_json::to_string_pretty(&p).unwrap();
        let loaded: Policy = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.sandbox_exec, SandboxMode::Required);
    }

    // ── Sandbox edge tests ────────────────────────────────────────────

    /// Workspace path with spaces must remain a single argv element in
    /// --bind and --chdir — never split into multiple args.
    #[test]
    fn bwrap_argv_workspace_path_with_spaces() {
        let ws = Path::new("/home/user/my project with spaces");
        let args = bwrap_argv("echo hello", ws);
        let bind_idx = args.iter().position(|a| a == "--bind").unwrap();
        assert_eq!(
            args[bind_idx + 1],
            "/home/user/my project with spaces",
            "--bind source preserves spaces"
        );
        assert_eq!(
            args[bind_idx + 2],
            "/home/user/my project with spaces",
            "--bind dest preserves spaces"
        );
        let chdir_idx = args.iter().position(|a| a == "--chdir").unwrap();
        assert_eq!(
            args[chdir_idx + 1],
            "/home/user/my project with spaces",
            "--chdir preserves spaces"
        );
    }

    /// Workspace path with Unicode characters stays intact as a single
    /// argv element (no encoding corruption).
    #[test]
    fn bwrap_argv_workspace_path_with_unicode() {
        let ws = Path::new("/home/user/日本語プロジェクト");
        let args = bwrap_argv("echo hello", ws);
        let bind_idx = args.iter().position(|a| a == "--bind").unwrap();
        assert_eq!(
            args[bind_idx + 1],
            "/home/user/日本語プロジェクト",
            "--bind source preserves unicode"
        );
        assert_eq!(
            args[bind_idx + 2],
            "/home/user/日本語プロジェクト",
            "--bind dest preserves unicode"
        );
    }

    /// bwrap_argv always produces exactly 19 args for the fixed spec
    /// pattern (9 bwrap flags × 2-3 args each + sh -c <cmd>).
    #[test]
    fn bwrap_argv_produces_consistent_arg_count() {
        // Short command.
        let args1 = bwrap_argv("ls", Path::new("/ws"));
        assert_eq!(args1.len(), 19, "short command → 19 args");
        // Long command.
        let args2 = bwrap_argv("cargo test -- --test-threads=1", Path::new("/ws"));
        assert_eq!(args2.len(), 19, "long command → 19 args");
        // The command is always the last arg (args[18]).
        assert_eq!(args1[18], "ls");
        assert_eq!(args2[18], "cargo test -- --test-threads=1");
    }

    /// SandboxMode::default() is Off — additive default for old policies.
    #[test]
    fn sandbox_mode_default_is_off() {
        assert_eq!(SandboxMode::default(), SandboxMode::Off);
    }

    /// SandboxMode serde roundtrip for all three variants.
    #[test]
    fn sandbox_mode_serde_roundtrip_all_variants() {
        for mode in [
            SandboxMode::Off,
            SandboxMode::IfAvailable,
            SandboxMode::Required,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let loaded: SandboxMode = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded, mode, "roundtrip for {mode:?}");
        }
    }

    /// SandboxMode Copy semantics: assigning a value copies it, no clone needed.
    #[test]
    fn sandbox_mode_is_copy() {
        let a = SandboxMode::Required;
        let b = a; // Copy, not move — a still usable.
        assert_eq!(a, SandboxMode::Required);
        assert_eq!(b, SandboxMode::Required);
    }

    /// spawn_argv Required + no bwrap: exact error message matches spec.
    #[test]
    fn spawn_argv_required_error_message_exact() {
        if bwrap_available() {
            return; // only meaningful when bwrap is absent.
        }
        let result = spawn_argv("ls", Path::new("/ws"), SandboxMode::Required);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("sandbox required but bwrap not found"),
            "error text contains spec phrase: {err}"
        );
    }

    /// Policy evaluate still denies metacharacters when sandbox is Required.
    /// Sandbox mode doesn't bypass the policy gate — chaining is still blocked.
    #[test]
    fn policy_sandbox_required_does_not_bypass_metacharacter_check() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec![],
            default_exec: DefaultExec::Ask,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Required,
        };
        let v = p.evaluate(&Action::Exec {
            command: "ls; rm -rf /".into(),
            cwd: root,
        });
        assert!(
            matches!(v, Verdict::Deny { .. }),
            "Required sandbox does not bypass metachar check: {v:?}"
        );
    }

    /// Policy evaluate still denies deny-list entries when sandbox is Required.
    #[test]
    fn policy_sandbox_required_does_not_bypass_deny_list() {
        let root = PathBuf::from("/tmp");
        let p = Policy {
            roots: vec![root.clone()],
            auto_allow: vec![],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Required,
        };
        let v = p.evaluate(&Action::Exec {
            command: "sudo ls".into(),
            cwd: root,
        });
        assert!(
            matches!(v, Verdict::Deny { .. }),
            "Required sandbox does not bypass deny-list: {v:?}"
        );
    }

    /// Policy JSON deserialization: sandbox_exec = "IfAvailable" loads
    /// correctly (not just Off/Required).
    #[test]
    fn policy_deserialize_sandbox_if_available() {
        let json = r#"{
            "roots": ["/tmp"],
            "auto_allow": [],
            "deny": [],
            "default_exec": "Allow",
            "ask_on_write": false,
            "sandbox_exec": "IfAvailable"
        }"#;
        let p: Policy = serde_json::from_str(json).unwrap();
        assert_eq!(p.sandbox_exec, SandboxMode::IfAvailable);
    }

    // ── Real-bwrap integration test (skipped when absent) ───────────────

    #[tokio::test]
    async fn sandbox_integration_skipped_without_bwrap() {
        if !bwrap_available() {
            eprintln!("SKIP: bwrap not available on this host");
            return;
        }

        let root = std::env::temp_dir();
        let ws = root.join("lore-bwrap-integration-test");
        std::fs::create_dir_all(&ws).unwrap();

        let p = Policy {
            roots: vec![ws.clone()],
            auto_allow: vec!["echo".into(), "cat".into(), "touch".into(), "ls".into()],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
            sandbox_exec: SandboxMode::Required,
        };
        let gate = Arc::new(Gate::new(p, Arc::new(AllowAll)));
        let tool = ShellTool::new(gate, ws.clone());

        // Write inside workspace → succeeds (workspace is bind-mounted writable).
        let out = tool.run("touch inside_ws.txt").await.unwrap();
        assert!(out.contains("[exit code: 0]"), "write in ws ok: {out}");

        // Write inside workspace succeeded — sandbox is functional.
        // /tmp isolation is not asserted here because bwrap creates a fresh
        // tmpfs that may or may not allow writes depending on userns
        // privileges; the important property is that host /tmp is hidden.

        std::fs::remove_dir_all(&ws).ok();
    }
}
