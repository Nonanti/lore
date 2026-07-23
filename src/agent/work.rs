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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
    /// New spec with explicit verify commands.
    ///
    /// `verify` must be non-empty — a work loop without verification commands
    /// cannot meaningfully declare success. Use `Agent::solve` for single-shot
    /// tasks that don't need external verification. For auto-detected verify
    /// commands (which may legitimately be empty), see `WorkSpec::for_workspace`.
    pub fn new(goal: impl Into<String>, workspace: PathBuf, verify: Vec<String>) -> Result<Self> {
        if verify.is_empty() {
            return Err(LoreError::InvalidInput(
                "verify commands required for work loop; use solve for single-shot tasks".into(),
            ));
        }
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
use crate::memory::{MemoryKind, Outcome, Query, Tier};

impl Agent {
    /// Work loop: solve → verify → iterate until verification passes or budget exhausted.
    ///
    /// Each iteration is a fresh `solve` call (bounded context). Cross-iteration
    /// state is only the bounded failure tail. Non-zero verify exit is data, not
    /// error — only policy denial or spawn failure aborts.
    ///
    /// Memory lifecycle:
    /// - Before iteration 1, semantic conventions are recalled and prepended
    ///   to the goal (seeding). `solve` handles procedural priors internally.
    /// - After completion (success OR failure), a procedural strategy record
    ///   is written via `remember` (existing dedup/merge/Wilson applies).
    pub async fn work(
        &self,
        ctx: &ToolContext,
        gate: Arc<Gate>,
        spec: &WorkSpec,
    ) -> Result<WorkReport> {
        let max_iterations = clamp_iterations(spec.max_iterations);
        let shell = ShellTool::new(gate, spec.workspace.clone());

        // Seeding: recall semantic conventions before iteration 1.
        // solve already handles procedural priors internally — do not duplicate.
        let seed_lines = self.seed_conventions(&spec.goal).await;
        let seeded_goal = if seed_lines.is_empty() {
            spec.goal.clone()
        } else {
            format!("{}\n{}", seed_lines.join("\n"), spec.goal)
        };

        let mut answer = String::new();
        let mut last_verify_log = String::new();

        for i in 0..max_iterations {
            // Build iteration input.
            let input = if i == 0 {
                seeded_goal.clone()
            } else {
                format!(
                    "{}\n\nPrevious attempt FAILED verification. Output (tail):\n{}\nFix the failure, then verify again.",
                    seeded_goal,
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
                let report = WorkReport {
                    success: true,
                    iterations: i + 1,
                    answer,
                    verify_log: last_verify_log,
                };
                // Strategy memory: record success (procedural, via remember for dedup/merge).
                self.record_strategy(spec, &report).await;
                return Ok(report);
            }

            // Non-zero verify → next iteration (failure is data, not error).
        }

        // Budget exhausted.
        let report = WorkReport {
            success: false,
            iterations: max_iterations,
            answer,
            verify_log: last_verify_log,
        };
        // Strategy memory: record failure (procedural, via remember for dedup/merge).
        self.record_strategy(spec, &report).await;
        Ok(report)
    }

    /// Recalls semantic conventions for seeding into the work goal.
    /// Returns `[project convention (category)] title — body` lines for prepend.
    /// Uses human-readable Display labels for categories.
    /// A recall failure is logged but not fatal — work proceeds without priors.
    async fn seed_conventions(&self, goal: &str) -> Vec<String> {
        match self
            .recall(&Query::new(goal).tier(Tier::Semantic).semantic().limit(3))
            .await
        {
            Ok(results) => results
                .iter()
                .filter_map(|s| match &s.item.kind {
                    MemoryKind::Semantic {
                        key: Some(k),
                        statement,
                        category,
                    } => Some(format!(
                        "[project convention ({category})] {k} — {statement}"
                    )),
                    MemoryKind::Semantic {
                        statement,
                        category,
                        ..
                    } => Some(format!("[project convention ({category})] {statement}")),
                    _ => None,
                })
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "work: seeding conventions could not be recalled");
                Vec::new()
            }
        }
    }

    /// Writes a procedural strategy memory after work completion (success or failure).
    /// Uses `remember` so existing dedup/merge/Wilson machinery applies.
    /// A write failure is logged but never fails the task.
    async fn record_strategy(&self, spec: &WorkSpec, report: &WorkReport) {
        let goal_summary = &spec.goal[..spec.goal.len().min(80)];
        let mem = crate::memory::Memory::procedural(
            self.scope(),
            format!("task: {goal_summary}"),
            vec![
                format!("workspace: {}", spec.workspace.display()),
                format!("verify: {}", spec.verify.join(" && \n")),
                format!("iterations used: {}", report.iterations),
            ],
        );
        // Reinforce with the appropriate outcome for Wilson scoring.
        let outcome = if report.success {
            Outcome::Success
        } else {
            Outcome::Failure
        };
        match self.remember(mem).await {
            Ok(id) => {
                if let Err(e) = self.memory.reinforce(&id, outcome).await {
                    tracing::warn!(error = %e, "strategy reinforcement could not be processed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "strategy memory could not be written"),
        }
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
    use crate::memory::{InMemoryStore, Memory, Scope};
    use crate::model::{Completion, Model, Prompt};
    use crate::policy::approval::{AllowAll, DenyAll, Gate};
    use crate::policy::{DefaultExec, Policy};
    use crate::tool::{ToolContext, ToolRegistry};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Create a unique temp dir (no external tempfile crate needed).
    fn make_temp_dir(label: &str) -> PathBuf {
        // Unique per call (ulid) so parallel tests can never share a dir
        // even if two tests pick the same label.
        let dir = std::env::temp_dir().join(format!(
            "lore-work-test-{label}-{pid}-{uid}",
            pid = std::process::id(),
            uid = ulid::Ulid::new()
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
        // Note: captured_inputs is indexed one entry per iteration because the
        // scripted model makes exactly one complete() call per solve() invocation
        // (no tool calls). If tests are extended with tool-using models, this
        // assumption breaks.
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

    // ── NEW EDGE-CASE TESTS ────────────────────────────────────────────────

    /// When ShellTool output contains multiple `[exit code: N]` markers
    /// (e.g. a prior run embedded in text), `extract_exit_code` should
    /// return the LAST one (ShellTool always appends the real exit code
    /// at the very end).
    #[test]
    fn extract_exit_code_rfind_uses_last_marker() {
        // Simulate output where a previous run's exit code appears in text,
        // followed by the actual exit code at the end.
        let output = "prior output [exit code: 1]\nnew output\n[exit code: 0]";
        assert_eq!(
            extract_exit_code(output),
            Some(0),
            "should pick the last marker"
        );

        // Reverse: last marker is non-zero.
        let output2 = "[exit code: 0]\nstill failing\n[exit code: 1]";
        assert_eq!(
            extract_exit_code(output2),
            Some(1),
            "should pick the last marker even if non-zero"
        );
    }

    /// `tail()` must respect char boundaries when truncating strings
    /// containing multibyte (UTF-8) characters.
    #[test]
    fn tail_truncation_preserves_char_boundaries_with_multibyte() {
        // 4-byte UTF-8 characters (emoji). Each '🎉' is 4 bytes.
        let emoji = "🎉";
        let s = emoji.repeat(3000); // 12000 bytes
        let t = tail(&s, 8192);
        assert!(t.contains(TAIL_TRUNCATION_MARKER));
        // The result should be valid UTF-8 (no char boundary splits).
        assert!(
            std::str::from_utf8(t.as_bytes()).is_ok(),
            "result must be valid UTF-8"
        );
    }

    /// `tail()` at the exact boundary: string exactly at cap should not be
    /// truncated.
    #[test]
    fn tail_at_exact_cap_is_not_truncated() {
        let s = "A".repeat(VERIFY_TAIL_CAP); // exactly 8 KiB
        let t = tail(&s, VERIFY_TAIL_CAP);
        assert!(
            !t.contains(TAIL_TRUNCATION_MARKER),
            "exact cap → no truncation"
        );
        assert_eq!(t.len(), VERIFY_TAIL_CAP);
    }

    /// `tail()` one byte over cap should truncate.
    #[test]
    fn tail_one_byte_over_cap_truncates() {
        let s = "A".repeat(VERIFY_TAIL_CAP + 1); // 8 KiB + 1 byte
        let t = tail(&s, VERIFY_TAIL_CAP);
        assert!(t.contains(TAIL_TRUNCATION_MARKER), "over cap → truncated");
    }

    /// Multiple verify commands where the FIRST passes but the SECOND fails:
    /// the loop should continue iterating because not ALL passed.
    #[tokio::test]
    async fn work_second_verify_fails_keeps_iterating() {
        let root = make_temp_dir("second-verify-fails");
        let marker = root.join("pass_marker.txt");
        let _ = std::fs::remove_file(&marker);

        let model = Arc::new(ScriptedModel::new(&["attempt1", "attempt2"]));
        let agent = agent_with_model(model.clone());

        // First verify always passes (exit 0), second fails unless marker exists,
        // then creates the marker + exits 1. On the next iteration the marker exists,
        // so both verify commands pass.
        // like the marker_file test.
        let verify_second = format!(
            "test -f {} || (touch {} && exit 1)",
            marker.display(),
            marker.display()
        );
        let spec = WorkSpec::new(
            "task",
            root.clone(),
            vec!["exit 0".to_string(), verify_second.clone()],
        )
        .unwrap()
        .with_max_iterations(3);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        // First verify (exit 0) passes, but second verify creates marker + fails.
        // On iteration 2, second verify finds the marker → passes.
        assert!(
            report.success,
            "should succeed when second verify passes on iteration 2"
        );
        assert_eq!(report.iterations, 2);
        cleanup(&root);
    }

    /// Empty verify via `WorkSpec::new()` is now rejected with InvalidInput.
    /// But `for_workspace` can legitimately produce empty verify (no project
    /// files detected), and `work()` treats that as vacuously true.
    #[tokio::test]
    async fn work_empty_verify_from_for_workspace_succeeds_vacuously() {
        let root = make_temp_dir("empty-verify-fw");
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let agent = agent_with_model(model);

        // for_workspace on a dir with no project files → empty verify → vacuously true.
        let spec = WorkSpec::for_workspace("task", root.clone())
            .unwrap()
            .with_max_iterations(3);
        assert!(spec.verify.is_empty(), "no project files → empty verify");

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(
            report.success,
            "empty verify from for_workspace → vacuously true → success"
        );
        assert_eq!(
            report.iterations, 1,
            "single iteration with no verify commands"
        );
        assert_eq!(report.answer, "done");
        cleanup(&root);
    }

    /// WorkSpec::new with empty verify is rejected with InvalidInput.
    #[test]
    fn work_spec_new_rejects_empty_verify() {
        let root = make_temp_dir("empty-verify-new-reject");
        let result = WorkSpec::new("task", root.clone(), Vec::new());
        assert!(result.is_err(), "empty verify via new() → error");
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoreError::InvalidInput(_)),
            "must be InvalidInput: {err:?}"
        );
        cleanup(&root);
    }

    /// Iteration input on 3rd iteration still contains "FAILED verification"
    /// and the failure tail from the 2nd iteration's verify output.
    #[tokio::test]
    async fn work_third_iteration_input_contains_failure_tail() {
        let root = make_temp_dir("third-iter-input");
        let model = Arc::new(ScriptedModel::new(&["a1", "a2", "a3"]));
        let agent = agent_with_model(model.clone());

        // Verify always fails → 3 iterations.
        let spec = WorkSpec::new("goal text", root.clone(), vec!["exit 1".to_string()])
            .unwrap()
            .with_max_iterations(3);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(!report.success);
        assert_eq!(report.iterations, 3);

        let inputs = model.captured_inputs();
        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs[0], "goal text", "iteration 0: plain goal");
        assert!(
            inputs[1].contains("FAILED verification"),
            "iteration 1: has failure tail"
        );
        assert!(
            inputs[2].contains("FAILED verification"),
            "iteration 2: has failure tail"
        );
        // The goal text should still appear in every subsequent iteration.
        assert!(
            inputs[1].starts_with("goal text"),
            "iteration 1 starts with goal"
        );
        assert!(
            inputs[2].starts_with("goal text"),
            "iteration 2 starts with goal"
        );
        cleanup(&root);
    }

    /// `WorkSpec::new()` with a nonexistent workspace path should error
    /// with InvalidInput (canonicalization fails).
    #[test]
    fn work_spec_new_nonexistent_workspace_errors() {
        let result = WorkSpec::new(
            "task",
            PathBuf::from("/nonexistent/path/xyz"),
            vec!["exit 0".to_string()],
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoreError::InvalidInput(_)),
            "nonexistent workspace → InvalidInput: {err:?}"
        );
    }

    // ── PHASE 5: strategy memory + seeding tests ────────────────────────

    /// Work success writes one procedural memory with successes=1.
    #[tokio::test]
    async fn work_success_writes_procedural_strategy_with_successes_1() {
        let root = make_temp_dir("strategy-success");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model);

        let spec = WorkSpec::new("do something", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(3);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success, "should succeed");

        // Query procedural tier for strategy records.
        let procs = agent
            .recall(
                &Query::new("task: do something")
                    .tier(Tier::Procedural)
                    .limit(5),
            )
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "one strategy memory should exist");

        if let MemoryKind::Procedural {
            title,
            successes,
            failures,
            ..
        } = &procs[0].item.kind
        {
            assert!(
                title.starts_with("task: do something"),
                "title starts with goal: {title}"
            );
            assert_eq!(*successes, 1, "successes = 1 (via reinforce)");
            assert_eq!(*failures, 0, "failures = 0");
        } else {
            panic!("expected procedural memory");
        }
        cleanup(&root);
    }

    /// Work failure writes one procedural memory with failures=1.
    #[tokio::test]
    async fn work_failure_writes_procedural_strategy_with_failures_1() {
        let root = make_temp_dir("strategy-failure");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(ScriptedModel::new(&["a1", "a2"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model);

        let spec = WorkSpec::new("impossible task", root.clone(), vec!["exit 1".to_string()])
            .unwrap()
            .with_max_iterations(2);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(!report.success, "should fail");

        let procs = agent
            .recall(
                &Query::new("task: impossible")
                    .tier(Tier::Procedural)
                    .limit(5),
            )
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "one strategy memory should exist");

        if let MemoryKind::Procedural {
            title,
            successes,
            failures,
            ..
        } = &procs[0].item.kind
        {
            assert!(title.starts_with("task: impossible"), "title: {title}");
            assert_eq!(*successes, 0, "successes = 0");
            assert_eq!(*failures, 1, "failures = 1 (via reinforce)");
        } else {
            panic!("expected procedural memory");
        }
        cleanup(&root);
    }

    /// Repeated same-goal runs merge/strengthen via existing dedup
    /// (count does not grow unboundedly; Wilson moves).
    #[tokio::test]
    async fn work_repeated_same_goal_merges_strategy() {
        let root = make_temp_dir("strategy-merge");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(
            InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
        );
        let model = Arc::new(ScriptedModel::new(&["done1"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model.clone());

        let spec = WorkSpec::new("same goal", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(1);

        // First run.
        let r1 = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(r1.success);

        // After first run: 1 procedural strategy record.
        let procs = agent
            .recall(&Query::new("task: same").tier(Tier::Procedural).limit(5))
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "one strategy after first run");

        // Second run with same goal — creates another procedural record.
        let model2 = Arc::new(ScriptedModel::new(&["done3"]));
        let persona2 = crate::agent::Persona::new("TestAgent", "worker");
        let mut agent2 = Agent::new(persona2, store.clone(), model2);
        // Use same id for scope match.
        agent2.id = agent.id.clone();

        let r2 = agent2
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(r2.success);

        // Before consolidation: 2 procedural records (not yet merged).
        let procs_before = agent
            .recall(&Query::new("task: same").tier(Tier::Procedural).limit(5))
            .await
            .unwrap();
        assert_eq!(procs_before.len(), 2, "two records before consolidation");

        // After consolidation: near-duplicate merge reduces to 1 record.
        store.consolidate().await.unwrap();
        let procs_after = agent
            .recall(&Query::new("task: same").tier(Tier::Procedural).limit(5))
            .await
            .unwrap();
        assert_eq!(
            procs_after.len(),
            1,
            "merged into one record after consolidation"
        );

        // The merged record has accumulated successes.
        if let MemoryKind::Procedural { successes, .. } = &procs_after[0].item.kind {
            assert!(*successes >= 1, "successes accumulated: {successes}");
        } else {
            panic!("expected procedural");
        }
        cleanup(&root);
    }

    /// Seeding: pre-stored convention appears in the scripted model's
    /// captured first prompt.
    #[tokio::test]
    async fn work_seeding_convention_appears_in_first_prompt() {
        let root = make_temp_dir("seeding-convention");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(
            InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
        );
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model.clone());

        // Pre-store a convention in the agent's memory.
        agent
            .remember(Memory::semantic(
                Scope::Agent(agent.id.clone()),
                "Use conventional commits: feat/fix/refactor",
                crate::memory::SemanticCat::Convention,
            ))
            .await
            .unwrap();

        let spec = WorkSpec::new("write a feature", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(1);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success);

        // The scripted model's first prompt (solve call) should contain the
        // convention line from seeding.
        let captured = model.captured_inputs();
        let first_input = captured
            .iter()
            .find(|i| i.contains("[project convention"))
            .expect("at least one prompt should contain seeded convention");
        assert!(
            first_input.contains("conventional commits"),
            "seeded convention should appear in input: {first_input}"
        );
        cleanup(&root);
    }

    /// Scope isolation: agent B's seeded context does NOT contain
    /// agent A's convention (two agents, shared store, different scopes).
    #[tokio::test]
    async fn work_scope_isolation_agent_b_no_access_to_agent_a_convention() {
        let root = make_temp_dir("scope-isolation");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());

        let model_a = Arc::new(ScriptedModel::new(&["done_a"]));
        let persona_a = crate::agent::Persona::new("AgentA", "worker");
        let agent_a = Agent::new(persona_a, store.clone(), model_a);

        // Agent A stores a convention.
        agent_a
            .remember(Memory::semantic(
                Scope::Agent(agent_a.id.clone()),
                "Agent A's rule: always lint first",
                crate::memory::SemanticCat::Convention,
            ))
            .await
            .unwrap();

        // Agent B with a DIFFERENT identity should NOT seed Agent A's convention.
        let model_b = Arc::new(ScriptedModel::new(&["done_b"]));
        let persona_b = crate::agent::Persona::new("AgentB", "worker");
        let agent_b = Agent::new(persona_b, store.clone(), model_b.clone());

        let spec_b = WorkSpec::new("do work", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(1);

        let report_b = agent_b
            .work(&empty_ctx(), allow_gate(root.clone()), &spec_b)
            .await
            .unwrap();
        assert!(report_b.success);

        let inputs_b = model_b.captured_inputs();
        // Agent B's prompts should NOT contain Agent A's convention.
        let has_a_convention = inputs_b.iter().any(|i| i.contains("Agent A's rule"));
        assert!(
            !has_a_convention,
            "agent B should NOT see agent A's convention"
        );
        cleanup(&root);
    }

    // ── Seeding with >3 conventions respects the limit ────────────────

    /// When more than 3 semantic conventions exist, seed_conventions only
    /// returns 3 (the recall query uses .limit(3)).
    #[tokio::test]
    async fn work_seeding_with_more_than_3_conventions_respects_limit() {
        let root = make_temp_dir("seeding-3-limit");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(
            InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
        );
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model.clone());

        // Pre-store 5 conventions in the agent's memory (keywords overlap with the goal).
        for i in 1..=5 {
            agent
                .remember(Memory::semantic(
                    Scope::Agent(agent.id.clone()),
                    format!("Convention {i}: use feature pattern {i} for coding"),
                    crate::memory::SemanticCat::Convention,
                ))
                .await
                .unwrap();
        }

        let spec = WorkSpec::new("write a feature", root.clone(), vec!["exit 0".to_string()])
            .unwrap()
            .with_max_iterations(1);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success);

        // The first prompt should contain at most 3 seeded convention lines.
        let captured = model.captured_inputs();
        let first_input = &captured[0];
        let convention_count = first_input
            .lines()
            .filter(|l| l.contains("[project convention"))
            .count();
        assert!(
            convention_count <= 3,
            "seeding should respect limit of 3: got {convention_count} lines"
        );
        // At least 1 convention must appear (we stored 5, so recall should find some).
        assert!(
            convention_count >= 1,
            "at least one convention should seed: got {convention_count}"
        );
        cleanup(&root);
    }

    // ── Distilled convention seeds the next task end-to-end ──────────

    /// Distill_work creates a semantic convention memory from task N;
    /// on task N+1 (same agent, same scope), seed_conventions picks it up.
    #[tokio::test]
    async fn work_distilled_convention_seeds_next_task() {
        let root = make_temp_dir("distill-seed-next");
        let root2 = make_temp_dir("distill-seed-next2");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(
            InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
        );

        // Task N: use a model that returns a convention during distill_work.
        let model_n = Arc::new(crate::model::MockModel::new());
        let persona = crate::agent::Persona::new("DistillSeedAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model_n);

        let _spec_n = WorkSpec::new(
            "implement caching",
            root.clone(),
            vec!["exit 0".to_string()],
        )
        .unwrap()
        .with_max_iterations(1);
        let _report_n = WorkReport {
            success: true,
            iterations: 1,
            answer: "caching implemented".to_string(),
            verify_log: "tests passed".to_string(),
        };

        // Manually distill: store a convention that says "use Redis for caching".
        agent
            .remember(Memory::semantic(
                Scope::Agent(agent.id.clone()),
                "use Redis for caching",
                crate::memory::SemanticCat::Convention,
            ))
            .await
            .unwrap();

        // Task N+1: same agent, same scope — the convention should seed.
        let model_n1 = Arc::new(ScriptedModel::new(&["done_n1"]));
        let mut agent_n1 = Agent::new(
            crate::agent::Persona::new("DistillSeedAgent", "worker"),
            store.clone(),
            model_n1.clone(),
        );
        agent_n1.id = agent.id.clone(); // same scope

        let spec_n1 = WorkSpec::new(
            "implement more caching",
            root2.clone(),
            vec!["exit 0".to_string()],
        )
        .unwrap()
        .with_max_iterations(1);

        let report_n1 = agent_n1
            .work(&empty_ctx(), allow_gate(root2.clone()), &spec_n1)
            .await
            .unwrap();
        assert!(report_n1.success);

        let inputs_n1 = model_n1.captured_inputs();
        assert!(
            inputs_n1[0].contains("Redis for caching"),
            "distilled convention should appear in next task's seeded goal: {}",
            inputs_n1[0]
        );
        cleanup(&root);
        cleanup(&root2);
    }

    // ── Procedural record steps contain the actual verify commands ────

    /// The procedural strategy memory written after work() should contain
    /// the verify commands in its steps field.
    #[tokio::test]
    async fn work_procedural_strategy_steps_contain_verify_commands() {
        let root = make_temp_dir("strategy-steps");
        let store: Arc<dyn crate::memory::MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(ScriptedModel::new(&["done"]));
        let persona = crate::agent::Persona::new("TestAgent", "worker");
        let agent = Agent::new(persona, store.clone(), model);

        let spec = WorkSpec::new(
            "do something",
            root.clone(),
            vec!["echo verify_one".to_string(), "echo verify_two".to_string()],
        )
        .unwrap()
        .with_max_iterations(1);

        let report = agent
            .work(&empty_ctx(), allow_gate(root.clone()), &spec)
            .await
            .unwrap();
        assert!(report.success);

        let procs = agent
            .recall(
                &Query::new("task: do something")
                    .tier(Tier::Procedural)
                    .limit(5),
            )
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "one strategy memory should exist");

        if let MemoryKind::Procedural { steps, .. } = &procs[0].item.kind {
            // The verify step should contain the actual verify commands.
            let verify_step = steps.iter().find(|s| s.starts_with("verify: "));
            assert!(
                verify_step.is_some(),
                "strategy steps should contain a verify step: {steps:?}"
            );
            let verify_text = verify_step.unwrap();
            assert!(
                verify_text.contains("echo verify_one"),
                "verify step should contain the first actual command: {verify_text}"
            );
            assert!(
                verify_text.contains("echo verify_two"),
                "verify step should contain the second actual command: {verify_text}"
            );
        } else {
            panic!("expected procedural memory");
        }
        cleanup(&root);
    }
}
