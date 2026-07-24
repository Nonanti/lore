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
pub(crate) const VERIFY_TAIL_CAP: usize = 8 * 1024;

/// Truncation marker when verify output exceeds the tail cap.
pub(crate) const TAIL_TRUNCATION_MARKER: &str = "\n[... output truncated]";

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

        let verify = if workspace.join("Cargo.toml").is_file() {
            vec!["cargo test".to_string()]
        } else if workspace.join("package.json").is_file() {
            vec!["npm test".to_string()]
        } else if workspace.join("pyproject.toml").is_file()
            || workspace.join("requirements.txt").is_file()
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
pub(crate) fn clamp_iterations(n: usize) -> usize {
    n.clamp(1, 20)
}

/// Keep only the last `cap` bytes of a string, with a truncation marker
/// if truncated. Uses char-boundary-safe truncation.
///
/// Shared helper for both work and distill modules — each passes its own
/// marker string.
pub(crate) fn tail_bytes(s: &str, cap: usize, marker: &str) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    // Walk backwards to find the nearest char boundary ≤ cap from end.
    let start = s.len() - cap;
    let mut i = start;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}{}", marker, &s[i..])
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
                    "{}\n\nPrevious attempt FAILED verification. Output (tail):\n<verify_output>\n{}\n</verify_output>\nFix the failure, then verify again.",
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
                let tailed = tail_bytes(&output, VERIFY_TAIL_CAP, TAIL_TRUNCATION_MARKER);
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&tailed);
                if code != Some(0) {
                    if code.is_none() {
                        tracing::warn!(cmd = %cmd, "extract_exit_code returned None — treating as failure");
                    }
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
    /// Uses `remember` so existing dedup/merge/Wilson machinery applies;
    /// near-duplicate dedup happens at [`MemoryStore::consolidate`] time.
    /// A write failure is logged but never fails the task.
    async fn record_strategy(&self, spec: &WorkSpec, report: &WorkReport) {
        let goal_summary: String = spec.goal.chars().take(80).collect();
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
pub(crate) fn extract_exit_code(output: &str) -> Option<i32> {
    // ShellTool always appends "\n[exit code: N]" at the end.
    let marker = "[exit code: ";
    let start = output.rfind(marker)?;
    let rest = &output[start + marker.len()..];
    // The number ends with ']'.
    let end = rest.find(']')?;
    rest[..end].parse().ok()
}
#[cfg(test)]
mod tests;
