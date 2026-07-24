# Sandbox (bwrap) + Parallel Daemon — combined design

Closing out the two remaining design-level items from the "AI coworkers"
roadmap. Two independent workstreams, one spec file (disjoint files:
sandbox → `src/policy/` + `src/tool/shell.rs`; parallel → `src/task/` +
`src/daemon.rs` + `src/main.rs`).

## A. OS-level sandbox for ShellTool (bubblewrap)

Threat model stays personal-use: this hardens the exec path against
accidental damage and prompt-injection-fueled mischief; it is not a
multi-tenant boundary.

### Policy — `src/policy/mod.rs`

```rust
pub enum SandboxMode { #[default] Off, IfAvailable, Required }  // serde, additive
// Policy gains: pub sandbox_exec: SandboxMode  (#[serde(default)] — old
// policy.json files load unchanged)
```

### ShellTool — `src/tool/shell.rs`

- When `gate.policy.sandbox_exec != Off`, the command runs as:
  `bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp \
         --bind <workspace> <workspace> --unshare-pid --die-with-parent \
         --chdir <workspace> sh -c <command>`
  Built via `Command::new("bwrap").args(...)` — argv, never string
  interpolation (no quoting bugs). Network stays shared on purpose:
  package managers (cargo/npm) need it; the deny-list + metacharacter
  checks remain as the network-misuse layer.
- bwrap detection: spawn `bwrap --version` once, cache in a
  `std::sync::OnceLock<bool>`.
- `IfAvailable` + bwrap missing → `tracing::warn!` once, run plain.
  `Required` + missing → `LoreError::PolicyDenied("sandbox required but
  bwrap not found")` — fail closed.
- Timeout/kill/truncation behavior unchanged (bwrap forwards exit codes;
  `--die-with-parent` reaps the sandbox on kill).

### Tests

- Argv construction unit tests (mode off → plain `sh -c`; on → bwrap argv
  exact, workspace bind present, no string-joined shell).
- Policy serde: old JSON without `sandbox_exec` loads as Off.
- Integration test runs a real bwrap exec only when `bwrap` is present
  (`bwrap --version` probe; skip otherwise): write inside workspace ok,
  write to /etc fails, tmpfs /tmp isolation.

## B. Parallel task execution in the daemon

### Atomic claim — `src/task/mod.rs`

```rust
pub fn claim_next_queued(&self) -> Result<Option<Task>>
```

Single statement (SQLite ≥3.35 RETURNING, rusqlite 0.32 bundled):

```sql
UPDATE tasks SET status='Running', updated_at=?1
WHERE id = (SELECT id FROM tasks WHERE status='Queued'
            ORDER BY created_at ASC, id ASC LIMIT 1)
  AND status='Queued'
RETURNING *
```

Atomic under WAL: two workers can never claim the same task.

### Daemon — `src/daemon.rs`

- `run_daemon(data_dir, db_path, concurrency: usize)` — spawns
  `concurrency` worker loops (tokio tasks). Each worker: claim → none:
  sleep 2 s → some: `run_task` → repeat. Idle shutdown: ctrl_c broadcast
  (`tokio::sync::watch`) — each worker finishes or re-queues its current
  task exactly like today's single loop, then exits; daemon returns when
  all workers join.
- `recover_orphaned` + `recover_stuck_parents` run once at startup before
  workers spawn (single daemon per data dir remains the assumption —
  documented).
- Same-workspace concurrency: allowed, but when a claimed task's workspace
  equals another currently-running task's, log a `tracing::warn`
  (informational; tracking via a shared `Mutex<Vec<(String, PathBuf)>>`
  of in-flight tasks).
- Team flows are unchanged — `maybe_complete_parent` already triggers on
  any child terminal state, safe under parallel children.

### CLI — `src/main.rs`

`lore daemon --concurrency N` (default 1 = today's behavior, clamp 1..=8).

### Tests

- claim_next_queued: two concurrent claims (two connections) yield
  different tasks / second gets None; FIFO order preserved.
- run_daemon with concurrency=3 processes 3 tasks concurrently (measure
  overlap via scripted-model barrier/AtomicUsize, generous timeout);
  results all Completed.
- Shutdown: SIGINT simulation via watch channel → in-flight task
  re-queued, workers join.
- clap parsing for --concurrency; clamp test.

## Non-goals

Multi-daemon fleets, cgroups/resource limits, network namespacing,
sandboxing file-write tools (they are already path-contained; bwrap only
wraps exec), making sandbox the default (opt-in via policy).
