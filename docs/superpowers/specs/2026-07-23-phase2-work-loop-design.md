# Phase 2 — Work Loop: plan → apply → verify → iterate

Part of the "AI coworkers" roadmap. Phase 1 (policy-gated shell/write/edit
tools) is done and verified end-to-end (`examples/hands_demo.rs`).

## Goal

Turn `Agent::solve` (single-shot, bounded steps, no external truth) into a
**work loop** that keeps iterating until an objective verification passes —
the difference between "an agent that answers" and "a coworker that finishes
work". The agent must never get to declare victory on its own: only the
verify command's exit code decides.

## Design

### New module `src/agent/work.rs`

```rust
pub struct WorkSpec {
    pub goal: String,                 // what to do
    pub workspace: PathBuf,           // sandbox root for this task
    pub verify: Vec<String>,          // shell commands; ALL must exit 0
    pub max_iterations: usize,        // default 5, clamped 1..=20
    pub max_solve_steps: usize,       // per-iteration solve budget, default MAX_SOLVE_STEPS
}

pub struct WorkReport {
    pub success: bool,
    pub iterations: usize,            // iterations actually used
    pub answer: String,               // last iteration's final answer
    pub verify_log: String,           // tail of last verify run (≤ 8 KiB)
}
```

- `WorkSpec::new(goal, workspace, verify)` — verify is **required** here
  (no verification = no work loop; use `solve` for that).
- `WorkSpec::for_workspace(goal, workspace)` — convenience that detects
  defaults: `Cargo.toml` → `["cargo test"]`; `package.json` → `["npm test"]`;
  `pyproject.toml`/`requirements.txt` → `["python -m pytest"]`;
  none found → empty verify (caller decides; document that empty means
  "single solve, success = model finished"). Builder: `with_max_iterations`,
  `with_max_solve_steps`.

### Loop — `impl Agent { pub async fn work(...) }`

```rust
pub async fn work(&self, ctx: &ToolContext, gate: Arc<Gate>, spec: &WorkSpec) -> Result<WorkReport>
```

1. Iteration `i` (0-based): build the iteration input —
   - first: `spec.goal` as-is.
   - later: `goal + "\n\nPrevious attempt FAILED verification. Output (tail):\n<tail>\nFix the failure, then verify again."`
2. `let answer = self.solve(ctx, &input, spec.max_solve_steps).await?;`
   — solve() unchanged; it already plans, calls tools, and records
   procedures into procedural memory.
3. Run every `spec.verify` command via a locally built
   `ShellTool::new(gate.clone(), spec.workspace.clone())`. Collect combined
   output (bounded tail: keep last 8 KiB per command).
4. All exit 0 → `WorkReport { success: true, .. }`. Any non-zero → next
   iteration with the failure tail. Budget exhausted → `success: false`.
5. Verify-command failures are data, not errors: a non-zero exit never
   aborts the loop. Only tool-level `Err` (policy denial, spawn failure,
   timeout) aborts — policy denial means a human said no.

Context management: each iteration is a fresh `solve` call, so the model's
context is bounded per iteration; the only cross-iteration state is the
bounded failure tail. Deliberately no transcript summarization in Phase 2
(YAGNI — MAX_SOLVE_STEPS=10 × truncated observations fits).

### Wiring

- `src/agent/mod.rs`: `pub mod work;` + re-export `WorkSpec, WorkReport`.
- `src/lib.rs`: re-export both next to `Agent`.
- No changes to `solve`, tools, or policy.

## Error handling

- `Gate` denial of a verify command → the loop stops and returns
  `Err(PolicyDenied)` (a human decision must not be retried).
- `spec.verify` empty + built via `new` → `InvalidInput` at construction.
- Workspace must exist; canonicalize once at entry.

## Testing (scripted model, no network)

A `ScriptedModel` test double (queue of canned completions, like the existing
`StubModel` pattern in `src/tool/mod.rs` tests):

- success on iteration 1: solve emits a write tool call + final answer;
  verify (`exit 0` command) passes → report success, iterations=1.
- fail-then-fix: verify scripted as a marker file check — first verify run
  fails (file absent), model "fixes" (write tool call creates the marker),
  second verify passes → success, iterations=2, and the second solve input
  contains the failure tail (assert via captured prompts in the scripted
  model).
- budget exhausted: verify always fails → success=false, iterations=max.
- policy denial: gate with DenyAll on verify command → Err(PolicyDenied).
- `for_workspace` detection: tempdir with Cargo.toml / package.json / none.
- WorkSpec clamping: max_iterations 0→1, 999→20.

## Non-goals (later phases)

Daemon/queue (Phase 3), per-agent models + PM (Phase 4), memory distillation
of work outcomes (Phase 5), parallel iterations, transcript summarization.
