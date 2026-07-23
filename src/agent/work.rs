//! Work loop: plan → apply → verify → iterate.
//!
//! [`Agent::work`] repeatedly calls `solve` then runs verification commands
//! via [`ShellTool`] until ALL verify commands exit 0. The agent never
//! declares victory on its own — only the verify commands' exit codes decide.
//! Non-zero verify exit is data (fed into next iteration), not an error.
//! Only policy denial or spawn failure aborts the loop.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::MAX_SOLVE_STEPS;
use crate::error::{LoreError, Result};
use crate::policy::approval::Gate;
use crate::tool::{ShellTool, Tool, ToolContext};

/// Maximum tail kept from verify output per command (8 KiB).
const VERIFY_TAIL_CAP: usize = 8 * 1024;

/// Truncation marker when verify output exceeds the tail cap.
const TAIL_TRUNCATION_MARKER: &str = "\n[... output truncated]";

/// Work specification: goal, workspace, verify commands, budgets.
#[derive(Clone, Debug)]
pub struct WorkSpec {
    /// What to achieve.
    pub goal: String,
    /// Sandbox root for this task (must exist; canonicalized once).
    pub workspace: PathBuf,
    /// Shell commands to run after each solve — ALL must exit 0 for success.
    pub verify: Vec<String>,
    /// Max work-loop iterations (default 5, clamped 1..=20).
    pub max_iterations: usize,
    /// Per-iteration solve budget (default [`MAX_SOLVE_STEPS`]).
    pub max_solve_steps: usize,
}

/// Outcome of a work loop run.
#[derive(Clone, Debug)]
pub struct WorkReport {
    /// Whether ALL verify commands passed on the final iteration.
    pub success: bool,
    /// Iterations actually used (1..=max_iterations).
    pub iterations: usize,
    /// Last iteration's solve answer.
    pub answer: String,
    /// Tail of the last verify run (≤ 8 KiB per command).
    pub verify_log: String,
}

impl WorkSpec {
    /// New spec with explicit verify commands (required — empty verify means
    /// "no verification"; use `solve` for single-shot tasks).
    pub fn new(goal: impl Into<String>, workspace: PathBuf, verify: Vec<String>) -> Result<Self> {
        let workspace = workspace
            .canonicalize()
            .map_err(|e| LoreError::InvalidInput(format!("workspace does not exist: {e}")))?;
        Ok(Self {
            goal: goal.into(),
            workspace,
            verify,
            max_iterations: 5,
            max_solve_steps: MAX_SOLVE_STEPS,
        })
    }

    /// Convenience: detects default verify commands from workspace contents.
    ///
    /// - `Cargo.toml` → `["cargo test"]`
    /// - `package.json` → `["npm test"]`
    /// - `pyproject.toml` or `requirements.txt` → `["python -m pytest"]`
    /// - None found → empty verify (caller decides; empty means "single
    ///   solve, success = model finished").
    pub fn for_workspace(goal: impl Into<String>, workspace: PathBuf) -> Result<Self> {
        let workspace = workspace
            .canonicalize()
            .map_err(|e| LoreError::InvalidInput(format!("workspace does not exist: {e}")))?;

        let verify = if workspace.join("Cargo.toml").exists() {
            vec!["cargo test".to_string()]
        } else if workspace.join("package.json").exists() {
            vec!["npm test".to_string()]
        } else if workspace.join("pyproject.toml").exists()
            || workspace.join("requirements.txt").exists()
        {
            vec!["python -m pytest".to_string()]
        } else {
            Vec::new()
        };

        Ok(Self {
            goal: goal.into(),
            workspace,
            verify,
            max_iterations: 5,
            max_solve_steps: MAX_SOLVE_STEPS,
        })
    }

    /// Builder: set max iterations (clamped 1..=20 on use).
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Builder: set per-iteration solve step budget.
    pub fn with_max_solve_steps(mut self, n: usize) -> Self {
        self.max_solve_steps = n;
        self
    }
}

/// Clamp max_iterations into 1..=20.
fn clamp_iterations(n: usize) -> usize {
    n.clamp(1, 20)
}

/// Keep only the last `cap` bytes of a string, with a truncation marker
/// if truncated. Uses char-boundary-safe truncation.
fn tail(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    // Walk backwards to find the nearest char boundary ≤ cap from end.
    let start = s.len() - cap;
    let mut i = start;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}{}", TAIL_TRUNCATION_MARKER, &s[i..])
}

use crate::agent::Agent;

impl Agent {
    /// Work loop: solve → verify → iterate until verification passes or budget exhausted.
    ///
    /// Each iteration is a fresh `solve` call (bounded context). Cross-iteration
    /// state is only the bounded failure tail. Non-zero verify exit is data, not
    /// error — only policy denial or spawn failure aborts.
    pub async fn work(
        &self,
        ctx: &ToolContext,
        gate: Arc<Gate>,
        spec: &WorkSpec,
    ) -> Result<WorkReport> {
        let max_iterations = clamp_iterations(spec.max_iterations);
        let shell = ShellTool::new(gate, spec.workspace.clone());

        let mut answer = String::new();
        let mut last_verify_log = String::new();

        for i in 0..max_iterations {
            // Build iteration input.
            let input = if i == 0 {
                spec.goal.clone()
            } else {
                format!(
                    "{}\n\nPrevious attempt FAILED verification. Output (tail):\n{}\nFix the failure, then verify again.",
                    spec.goal,
                    last_verify_log
                )
            };

            answer = self.solve(ctx, &input, spec.max_solve_steps).await?;

            // Run every verify command; collect combined output with per-command tails.
            let mut combined = String::new();
            let mut all_passed = true;

            for cmd in &spec.verify {
                let output = shell.run(cmd).await?;
                // Extract exit code from ShellTool output format: "[exit code: N]".
                let code = extract_exit_code(&output);
                let tailed = tail(&output, VERIFY_TAIL_CAP);
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&tailed);
                if code != Some(0) {
                    all_passed = false;
                }
            }

            last_verify_log = combined;

            if all_passed {
                return Ok(WorkReport {
                    success: true,
                    iterations: i + 1,
                    answer,
                    verify_log: last_verify_log,
                });
            }

            // Non-zero verify → next iteration (failure is data, not error).
        }

        // Budget exhausted.
        Ok(WorkReport {
            success: false,
            iterations: max_iterations,
            answer,
            verify_log: last_verify_log,
        })
    }
}

/// Extract exit code from ShellTool output format `"[exit code: N]"`.
/// Returns `None` if the pattern is not found.
fn extract_exit_code(output: &str) -> Option<i32> {
    // ShellTool always appends "\n[exit code: N]" at the end.
    let marker = "[exit code: ";
    let start = output.rfind(marker)?;
    let rest = &output[start + marker.len()..];
    // The number ends with ']'.
    let end = rest.find(']')?;
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::model::{Completion, Model, Prompt};
    use crate::policy::approval::{AllowAll, DenyAll, Gate};
    use crate::policy::{DefaultExec, Policy};
    use crate::tool::{ToolContext, ToolRegistry};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Create a unique temp dir (no external tempfile crate needed).
    fn make_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lore-work-test-{label}-{pid}",
            pid = std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Clean up a temp dir created by `make_temp_dir`.
    fn cleanup(dir: &PathBuf) {
        std::fs::remove_dir_all(dir).ok();
    }

    // ── ScriptedModel: queue of canned completions, captures prompts ──────

    /// Test model that returns scripted replies in sequence and captures
    /// the user input of each prompt (for assertion on failure tail injection).
    struct ScriptedModel {
        replies: Mutex<VecDeque<String>>,
        /// Captured user texts from each `complete` call.
        captured_inputs: Mutex<Vec<String>>,
    }

    impl ScriptedModel {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
                captured_inputs: Mutex::new(Vec::new()),
            }
        }

        fn captured_inputs(&self) -> Vec<String> {
            self.captured_inputs.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Model for ScriptedModel {
        async fn complete(&self, p: &Prompt) -> Result<Completion> {
            self.captured_inputs.lock().unwrap().push(p.user.clone());
            let text = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "no reply left".into());
            Ok(Completion::new(text))
        }
    }

    fn empty_ctx() -> ToolContext {
        ToolContext {
            registry: ToolRegistry::new(),
            router: Arc::new(crate::tool::KeywordRouter::new()),
        }
    }

    fn allow_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root],
            auto_allow: vec![],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Allow,
            ask_on_write: false,
        };
        Arc::new(Gate::new(p, Arc::new(AllowAll)))
    }

    fn deny_all_gate(root: PathBuf) -> Arc<Gate> {
        let p = Policy {
            roots: vec![root],
            auto_allow: vec![],
            deny: vec!["sudo".into()],
            default_exec: DefaultExec::Deny,
            ask_on_write: false,
        };
        Arc::new(Gate::new(p, Arc::new(DenyAll)))
    }

    fn agent_with_model(model: Arc<dyn Model>) -> Agent {
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        Agent::new(persona, store, model)
    }

    // ── success on first iteration ───────────────────────────────────────

    #[tokio::test]
    async fn work_success_on_first_iteration() {
        let root = make_temp_dir("success-first");
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let agent = agent_with_model(model);

        let spec = WorkSpec::new("do something", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(3);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success, "should succeed on first iteration");
        assert_eq!(report.iterations, 1, "iterations = 1");
        assert_eq!(report.answer, "done");
        cleanup(&root);
    }

    // ── fail then fix (failure tail injection) ───────────────────────────

    #[tokio::test]
    async fn work_fail_then_fix_failure_tail_injection() {
        let root = make_temp_dir("fail-then-fix");
        let model = Arc::new(ScriptedModel::new(&["attempt1", "attempt2"]));
        let agent = agent_with_model(model.clone());

        // Verify always fails (exit 1) — we test that the failure tail
        // appears in the second solve input.
        let spec = WorkSpec::new("do something", root.clone(), vec!["exit 1".to_string()])
            .unwrap()
            .with_max_iterations(2);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(!report.success, "should fail (verify always exits 1)");
        assert_eq!(report.iterations, 2, "should use all iterations");

        let inputs = model.captured_inputs();
        assert_eq!(inputs.len(), 2, "two solve calls");
        assert_eq!(inputs[0], "do something", "first input is goal as-is");
        assert!(
            inputs[1].contains("FAILED verification"),
            "second input contains failure tail: {}",
            inputs[1]
        );
        cleanup(&root);
    }

    // ── fail then actually fix (marker file verify) ──────────────────────

    #[tokio::test]
    async fn work_fail_then_fix_marker_file() {
        let root = make_temp_dir("marker-fix");
        let marker = root.join("marker.txt");
        // Ensure marker doesn't exist initially.
        let _ = std::fs::remove_file(&marker);

        let model = Arc::new(ScriptedModel::new(&["attempt1", "attempt2_fixed"]));
        let agent = agent_with_model(model.clone());

        // Verify: if marker absent, create it AND exit 1 (report failure).
        // If marker present, exit 0 (success). With default_exec: Allow,
        // metacharacters like || are permitted.
        let verify_cmd = format!(
            "test -f {} || (touch {} && exit 1)",
            marker.display(),
            marker.display()
        );

        let spec = WorkSpec::new("create the marker file", root.clone(), vec![verify_cmd])
            .unwrap()
            .with_max_iterations(3);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success, "should succeed on second iteration");
        assert_eq!(report.iterations, 2, "iterations = 2");
        assert_eq!(report.answer, "attempt2_fixed");

        let inputs = model.captured_inputs();
        assert_eq!(inputs.len(), 2, "two solve calls");
        assert!(
            inputs[1].contains("FAILED verification"),
            "second input contains failure tail: {}",
            inputs[1]
        );
        cleanup(&root);
    }

    // ── budget exhausted ──────────────────────────────────────────────────

    #[tokio::test]
    async fn work_budget_exhausted() {
        let root = make_temp_dir("budget-exhausted");
        let model = Arc::new(ScriptedModel::new(&["a1", "a2", "a3", "a4"]));
        let agent = agent_with_model(model);

        let spec = WorkSpec::new("impossible task", root.clone(), vec!["exit 1".to_string()])
            .unwrap()
            .with_max_iterations(4);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(!report.success, "budget exhausted → success=false");
        assert_eq!(report.iterations, 4, "all iterations used");
        assert_eq!(report.answer, "a4", "last iteration's answer");
        cleanup(&root);
    }

    // ── policy denial → Err(PolicyDenied) ────────────────────────────────

    #[tokio::test]
    async fn work_policy_denial_aborts() {
        let root = make_temp_dir("policy-denial");
        let model = Arc::new(ScriptedModel::new(&["answer"]));
        let agent = agent_with_model(model);

        let spec = WorkSpec::new(
            "task",
            root.clone(),
            vec!["some_blocked_command".to_string()],
        )
        .unwrap()
        .with_max_iterations(3);

        let result = agent
            .work(&empty_ctx(), deny_all_gate(root.clone()), &spec)
            .await;
        assert!(result.is_err(), "policy denial → Err");
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoreError::PolicyDenied(_)),
            "must be PolicyDenied: {err:?}"
        );
        cleanup(&root);
    }

    // ── for_workspace detection ───────────────────────────────────────────

    #[test]
    fn for_workspace_detects_cargo_toml() {
        let dir = make_temp_dir("cargo-detect");
        std::fs::write(dir.join("Cargo.toml"), "").unwrap();
        let canon = dir.canonicalize().unwrap();

        let spec = WorkSpec::for_workspace("build it", canon.clone()).unwrap();
        assert_eq!(spec.verify, vec!["cargo test"]);
        assert_eq!(spec.workspace, canon);
        cleanup(&dir);
    }

    #[test]
    fn for_workspace_detects_package_json() {
        let dir = make_temp_dir("npm-detect");
        std::fs::write(dir.join("package.json"), "{}").unwrap();
        let canon = dir.canonicalize().unwrap();

        let spec = WorkSpec::for_workspace("build it", canon).unwrap();
        assert_eq!(spec.verify, vec!["npm test"]);
        cleanup(&dir);
    }

    #[test]
    fn for_workspace_detects_python_project() {
        let dir = make_temp_dir("pyproject-detect");
        std::fs::write(dir.join("pyproject.toml"), "").unwrap();
        let canon = dir.canonicalize().unwrap();

        let spec = WorkSpec::for_workspace("build it", canon).unwrap();
        assert_eq!(spec.verify, vec!["python -m pytest"]);
        cleanup(&dir);

        // Also requirements.txt.
        let dir2 = make_temp_dir("requirements-detect");
        std::fs::write(dir2.join("requirements.txt"), "").unwrap();
        let canon2 = dir2.canonicalize().unwrap();

        let spec2 = WorkSpec::for_workspace("build it", canon2).unwrap();
        assert_eq!(spec2.verify, vec!["python -m pytest"]);
        cleanup(&dir2);
    }

    #[test]
    fn for_workspace_no_detector_returns_empty() {
        let dir = make_temp_dir("no-detect");
        let canon = dir.canonicalize().unwrap();

        let spec = WorkSpec::for_workspace("build it", canon).unwrap();
        assert!(spec.verify.is_empty(), "no detector → empty verify");
        cleanup(&dir);
    }

    #[test]
    fn for_workspace_nonexistent_dir_errors() {
        let result = WorkSpec::for_workspace("task", PathBuf::from("/nonexistent/path"));
        assert!(result.is_err(), "nonexistent workspace → error");
    }

    // ── clamping ──────────────────────────────────────────────────────────

    #[test]
    fn clamp_iterations_bounds() {
        assert_eq!(clamp_iterations(0), 1);
        assert_eq!(clamp_iterations(1), 1);
        assert_eq!(clamp_iterations(5), 5);
        assert_eq!(clamp_iterations(20), 20);
        assert_eq!(clamp_iterations(999), 20);
    }

    // ── tail helper ───────────────────────────────────────────────────────

    #[test]
    fn tail_no_truncation_when_short() {
        let s = "hello";
        assert_eq!(tail(s, 1024), s);
    }

    #[test]
    fn tail_truncates_long_output() {
        let s = "A".repeat(10000);
        let t = tail(&s, 8192);
        assert!(t.contains(TAIL_TRUNCATION_MARKER));
        assert!(t.len() < s.len());
    }

    // ── clamping applied in work() ─────────────────────────────────────────

    #[tokio::test]
    async fn work_clamping_max_iterations_0_becomes_1() {
        let root = make_temp_dir("clamp-0");
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let agent = agent_with_model(model);

        // max_iterations = 0 should be clamped to 1.
        let spec = WorkSpec::new("task", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(0);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success);
        assert_eq!(report.iterations, 1, "clamped 0 → 1");
        cleanup(&root);
    }

    #[tokio::test]
    async fn work_clamping_max_iterations_999_becomes_20() {
        let root = make_temp_dir("clamp-999");
        // 20 replies for 20 iterations; verify always fails.
        let reply_strs: Vec<String> = (0..20).map(|i| format!("a{i}")).collect();
        let reply_refs: Vec<&str> = reply_strs.iter().map(|s| s.as_str()).collect();
        let model = Arc::new(ScriptedModel::new(&reply_refs));
        let agent = agent_with_model(model);

        // max_iterations = 999 should be clamped to 20.
        let spec = WorkSpec::new("task", root.clone(), vec!["exit 1".to_string()])
            .unwrap()
            .with_max_iterations(999);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(!report.success);
        assert_eq!(report.iterations, 20, "clamped 999 → 20");
        cleanup(&root);
    }

    // ── extract_exit_code ─────────────────────────────────────────────────

    #[test]
    fn extract_exit_code_from_shell_output() {
        let output = "hello world\n[exit code: 0]";
        assert_eq!(extract_exit_code(output), Some(0));

        let output2 = "error\n[exit code: 2]";
        assert_eq!(extract_exit_code(output2), Some(2));

        let output3 = "no exit code marker";
        assert_eq!(extract_exit_code(output3), None);
    }
}
