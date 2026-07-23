# Phase 1 — "Hands": Write/Exec Tools + Policy Engine

Part of the "AI coworkers" roadmap (approved 2026-07-23):

1. **Phase 1 — Hands: write/exec tools + policy engine** ← this spec
2. Phase 2 — Work loop: plan → apply → verify → iterate
3. Phase 3 — Daemon + task queue + CLI client
4. Phase 4 — Team: roles, per-agent models, PM agent
5. Phase 5 — Distilling work experience into memory

Decisions driving this design: Lore is the complete platform (native Rust,
self-contained, no new heavy deps); personal-use-first pragmatic security;
general-purpose work (shell + file ops as the universal core); **policy-based
autonomy** — agents run freely inside the policy, anything outside falls to an
approval gate.

## Goal

Give Lore agents the ability to **write/edit files and run shell commands**,
with every dangerous action passing through a single **policy gate** that can
allow, deny, or escalate to human approval.

## Components

### 1. Policy engine — `src/policy/mod.rs`

Pure decision core, unit-testable, no I/O in evaluation.

```rust
pub enum Action {
    Exec { command: String, cwd: PathBuf },
    Write { path: PathBuf },            // covers write + edit
}

pub enum Verdict { Allow, Ask { reason: String }, Deny { reason: String } }

pub struct Policy {
    pub roots: Vec<PathBuf>,        // allowed workspace roots
    pub auto_allow: Vec<String>,    // command prefixes allowed without approval ("cargo test", "ls", ...)
    pub deny: Vec<String>,          // command substrings always denied ("sudo", "rm -rf /", "git push --force", ...)
    pub default_exec: DefaultExec,  // Ask (default) | Allow | Deny — for commands matching neither list
    pub ask_on_write: bool,         // false (default): writes inside roots are allowed
}
```

- `Policy::evaluate(&self, action: &Action) -> Verdict`:
  - **Deny list always wins** (checked first, substring match on the command).
  - Exec: `auto_allow` prefix match → Allow; else `default_exec`.
  - Exec cwd must be inside a root, else Deny.
  - Write: path inside a root → Allow (or Ask if `ask_on_write`); outside → Deny.
  - Path containment uses canonicalization; for files that don't exist yet,
    canonicalize the nearest existing ancestor. Rejects `..` traversal and
    symlink escapes.
- `Policy::default_for(root)` — sensible personal-use defaults (deny list
  pre-seeded with `sudo`, `rm -rf /`, `git push --force`, shutdown/reboot,
  writes to `~/.ssh` etc.).
- Persistence: JSON via serde (`Policy::load(path)` / `save(path)`) — no new
  deps (no toml). Default location: `<LORE_DATA>/policy.json`.

### 2. Approval gate — `src/policy/approval.rs`

```rust
pub struct ApprovalRequest { pub action: Action, pub reason: String, pub agent: Option<String> }

#[async_trait]
pub trait Approver: Send + Sync {
    async fn decide(&self, req: &ApprovalRequest) -> Result<bool>;
}
```

- Implementations now: `DenyAll` (safe default), `AllowAll` (tests/full-auto),
  `CliApprover` (interactive y/N prompt on the terminal; uses
  `tokio::task::spawn_blocking` for stdin).
- Phase 3 will add a queue-backed approver behind the same trait — the trait is
  the seam; nothing else changes.
- `Gate { policy: Policy, approver: Arc<dyn Approver> }`:
  `async fn check(&self, action: &Action) -> Result<()>` — Allow → Ok,
  Deny → `LoreError::PolicyDenied(reason)` (new error variant),
  Ask → `approver.decide(...)`, rejection → `PolicyDenied`.

### 3. Shell tool — `src/tool/shell.rs`

`ShellTool { gate: Arc<Gate>, cwd: PathBuf, timeout: Duration, max_output: usize }`

- Tool name `"shell"`; args = the raw command line (args_hint documents this).
- Gate check (`Action::Exec`) before running.
- Runs via `tokio::process::Command` (`sh -c <command>`, cwd = workspace root).
  Requires adding the `process` feature to tokio in Cargo.toml.
- Captures stdout + stderr + exit status; output truncated to `max_output`
  (default 32 KiB) with a truncation marker; default timeout 120 s — on
  timeout the child is killed and an error returned.
- Builder-style config (`with_timeout`, `with_max_output`) matching existing
  tool conventions.

### 4. File write/edit tools — `src/tool/fs_write.rs`

Both scoped to a workspace root exactly like the existing `FileReadTool`
(relative paths only, no `..`, symlink-escape check), plus a `Gate` check
(`Action::Write`) before touching disk.

- `FileWriteTool` — name `"write"`, args JSON
  `{"path":"rel/path","content":"..."}`. Creates parent dirs, atomic write
  (tmp file + rename). Overwrites existing files.
- `FileEditTool` — name `"edit"`, args JSON
  `{"path":"rel/path","old":"exact text","new":"replacement"}`. `old` must
  match **exactly once**; 0 or >1 matches → descriptive error (count included).

JSON args because content is multiline; `parse_tool_call`-style lenient JSON
extraction is not needed — `serde_json::from_str` on the raw args, with a clear
error message on parse failure.

### 5. Wiring

- `src/tool/mod.rs`: `pub mod shell; pub mod fs_write;` + re-exports.
- `src/lib.rs`: re-export `Policy`, `Gate`, `Approver`, `CliApprover`, new tools.
- `src/error.rs`: add `LoreError::PolicyDenied(String)`.
- `Cargo.toml`: tokio `process` feature.

## Error handling

- Policy denial is a first-class, descriptive error (`PolicyDenied`) so the
  agent loop can report *why* an action was refused (and later phases can
  surface it in the task log).
- Shell failures are not tool errors: non-zero exit returns Ok with the exit
  code + output in the text (the model needs to read compiler/test errors).
  Only spawn failures/timeouts/policy denials are `Err`.

## Testing

- Policy unit tests: deny-over-allow precedence, prefix vs substring matching,
  default_exec branches, path containment (traversal, symlink escape,
  not-yet-existing file, cwd outside roots).
- Gate tests with `AllowAll`/`DenyAll` covering all three verdicts.
- ShellTool: echo roundtrip, non-zero exit reported in text, timeout kill,
  denied command, output truncation.
- Fs tools: write + read-back, atomic overwrite, parent-dir creation, edit
  single-match/zero-match/multi-match, root escape rejection, bad JSON args.

## Non-goals (later phases)

Work loop/verification (Phase 2), daemon + queued approvals (Phase 3),
per-agent model config and PM (Phase 4), memory distillation (Phase 5).
