# Phase 4 — Team: Roles, Per-Agent Models, PM Agent

Roadmap: Phase 1 (hands) ✔, Phase 2 (work loop), Phase 3 (daemon+queue).
This phase turns one worker into a **team**: each agent has a role and its
own model/provider, and an optional PM agent decomposes big goals and routes
subtasks to specialists.

## 1. Per-agent model config — `src/model/factory.rs` + agent file extension

Today the model comes from process-wide env (LORE_PROVIDER, LORE_AUTH,
LORE_LLM_MODEL, ...). Coworkers need their own brains.

```rust
pub struct ModelConfig {
    pub provider: ProviderKind,   // Anthropic | OpenAI | OpenAiCompat | Mock
    pub model: String,            // e.g. "claude-sonnet-4-5-20250929", "qwen3:8b"
    pub auth: Option<AuthKind>,   // Key | Subs — None = auto-detect (existing logic)
    pub base_url: Option<String>, // OpenAiCompat only (e.g. http://localhost:11434/v1)
}

pub fn build_model(cfg: &ModelConfig, data_dir: &Path) -> Result<Arc<dyn Model>>;
```

- `build_model` factors out the env-based wiring already in `src/main.rs`/
  provider modules; env config becomes `ModelConfig::from_env()` — the CLI
  keeps working exactly as today (backward compatible).
- Agent JSON (existing `Agent::save_to`/`load_from` schema) gains an
  **optional** `model` field (`#[serde(default, skip_serializing_if =
  "Option::is_none")]`) holding ModelConfig. Old files load unchanged;
  missing field → daemon falls back to `ModelConfig::from_env()`.
- New CLI: `lore agent create <name> --role backend --provider anthropic
  --model claude-sonnet-4-5-20250929 [--auth subs]` and `lore agent list`
  (name, role, provider/model, memory size). `lore agent create` writes the
  persona+model JSON under `<LORE_DATA>/agents/<name>.json`.

## 2. Role presets — `src/agent/roles.rs`

```rust
pub struct Role { pub name: &'static str, pub role: &'static str,
                  pub traits: &'static [&'static str], pub identity_extra: &'static str }
pub fn preset(name: &str) -> Option<Role>;        // "backend", "frontend", "reviewer", "pm", ...
pub fn presets() -> &'static [Role];
```

`identity_extra` is appended to `Persona::identity_prompt()` output (Persona
gains an optional `extra: Vec<String>` field, additive serde). Presets ship
with verification-minded instructions (e.g. backend: "Run the project's tests
before claiming done; read failures fully before editing."), frontend:
component/a11y basics, reviewer: adversarial read-only mindset, pm:
decompose-then-delegate JSON contract (below). Data-driven, not hardcoded
behavior — presets are prompts, the loop is unchanged.

## 3. PM agent — `src/orchestrator/pm.rs`

Entry point: `lore task add --team <goal>` (team flag on the existing
subcommand). The daemon recognizes a team task and runs the PM flow instead
of direct work:

1. **Decompose**: PM agent (preset "pm", its own ModelConfig) is prompted
   with the goal + `lore agent list` roster (names/roles) and must return
   JSON: `[{"agent": "<name>", "goal": "...", "verify": ["..."]}]`.
   Parse leniently (reuse parse_tool_call-style extraction); unknown agent
   names → task fails with a clear message; empty/invalid JSON → one retry
   with a corrective prompt, then fail.
2. **Enqueue**: each subtask becomes a normal task row (parent_id column —
   additive: `ALTER TABLE ... ADD COLUMN parent_id TEXT` guarded by
   user_version migration) assigned to the named agent. Team task waits
   (status: WaitingApproval is for approvals; add status `WaitingSubtasks`).
3. **Complete**: when all children reach Completed/Failed (daemon checks
   after each task finishes — parent lookup by parent_id), the PM gets a
   synthesis prompt: children reports → combined summary; any Failed child →
   team task Failed with the failing report attached.
4. **Review pass** (only when all children succeed): if a "reviewer" agent
   exists in the roster, one final subtask is auto-enqueued for it: goal =
   "Review the completed work: <children reports + verify logs>. Look for
   gaps, contradictions, missing verification." Its report appends to the
   team task's report. No git diff plumbing in Phase 4 — the reviewer works
   from reports and can use tools in the shared workspace.

Execution stays **sequential** (Phase 3 loop); subtasks run in enqueue order.
Parallel teams are a future phase.

### Wiring

- `src/task/mod.rs`: `parent_id: Option<String>` on Task (+ migration),
  `children_of(id)`, `all_children_done(id)`.
- Daemon: after each task completion, `maybe_complete_parent(store, task)`.
- CLI: `lore task add --team`, `lore task status <id>` shows children.

## Error handling

- PM decomposition failure (2 bad JSON rounds) → team task Failed, reason
  recorded; no children enqueued.
- Child policy-denied → child Failed; parent synthesis still runs and
  surfaces it (PM decides in summary; human sees the report).
- Reviewer absent → review pass skipped silently (not an error).

## Testing

- ModelConfig serde roundtrip; agent JSON backward compat (old file without
  `model` loads; new file with model loads).
- build_model: Mock + OpenAiCompat construct without network; Anthropic/OpenAI
  config → correct type (no live calls).
- Roles: preset lookup, identity_extra appears in identity_prompt.
- PM: ScriptedModel as PM returning a decomposition JSON → daemon enqueues
  children with correct agents/verify; invalid JSON retry then fail; parent
  completes only after all children done; Failed child propagates.
- Migration: open a v1 tasks.db (fixture created in test) → parent_id added.

## Non-goals

Parallel subtask execution, inter-agent messaging during tasks (blackboard
collab during a task), git-integration for the reviewer, cost tracking.
