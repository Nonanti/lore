# Native Tool Calling — Design (approved 2026-07-24)

## 1. Goal & Motivation

Tool calls currently ride a text protocol: the model is instructed to emit
`{"tool":..,"args":..}` JSON as plain text and `parse_tool_call` extracts it.
Two production bugs in a row came from exactly this seam (multi-call replies
mistaken for final answers; object-form args collapsing to `""` — agents'
edit/write silently no-oping while verify failed). Providers train their
models on **native** tool protocols (Anthropic `tool_use`/`tool_result`
content blocks, OpenAI `tool_calls`/`role:"tool"` messages); Lore should
speak them.

Long-term decision (user-approved): **full native conversation threading**
(Approach 2) — not a stateless shim. A structured message/block model becomes
the core; flat text becomes a *rendering* for text-only models, not the
foundation.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| N1 | Both protocols supported, selected by config `tool_mode: auto\|native\|text` | Gemma family, DeepSeek-R1 distills, and misconfigured vLLM servers have no native tool support; positioning is "works with any OpenAI-compatible endpoint" |
| N2 | Default `tool_mode = auto`: native when the provider supports it, runtime downgrade to text on "does not support tools" errors | Zero-config users get the best available behavior; nobody breaks |
| N3 | Provider scope this phase: **Anthropic + OpenAI + OpenAiCompat**. Codex (Responses API function items) is a follow-up. Mock stays text-only (`supports_native_tools = false`) | One wire format per phase; Responses item shape is its own project |
| N4 | `Tool::run(&str)` signature unchanged; native inputs are converted to the existing args-string form | Zero churn across tool impls; procedural memories (`ToolCall{tool,args}`) stay format-compatible between modes |
| N5 | Text mode behavior stays **bit-identical** to today (locked by the existing test suite) | 570 green tests are the regression insurance for the refactor |
| N6 | Solve keeps "step = one model roundtrip"; a native step may execute several parallel tool calls | Budget semantics stay simple; parallel calls are a native bonus |

## 3. Architecture

### 3.1 New types — `src/model/thread.rs`

```rust
/// One content block — the unit all provider wire formats share.
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
}

pub enum ChatRole { User, Assistant }          // system carried on Thread
pub struct ChatMessage { pub role: ChatRole, pub blocks: Vec<ContentBlock> }

/// Conversation for tool-loop completions.
pub struct Thread { pub system: String, pub messages: Vec<ChatMessage> }

/// A tool the model may call natively.
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,        // JSON Schema object
}

pub enum StopReason { EndTurn, ToolUse, Other }

pub struct ThreadReply {
    pub blocks: Vec<ContentBlock>,              // assistant text and/or tool_use
    pub stop: StopReason,
    pub reasoning_fallback: bool,
}
```

Helpers: `ThreadReply::text()` (joined Text blocks), `ThreadReply::tool_uses()`,
`ChatMessage::user_text(..)`, `ChatMessage::tool_results(..)`.

### 3.2 `Model` trait extension — `src/model/mod.rs`

```rust
/// Native tool-loop completion. Providers with native tool APIs override.
/// Default: Err(LoreError::NativeToolsUnsupported) — the solve loop routes
/// text-only models through the existing flat-prompt path instead.
async fn complete_thread(&self, thread: &Thread, tools: &[ToolSpec])
    -> Result<ThreadReply>
{ Err(LoreError::NativeToolsUnsupported("provider has no native tool support".into())) }

/// Capability probe driving `auto` mode (avoids a doomed roundtrip for
/// providers that never support tools: Mock, Codex for now).
fn supports_native_tools(&self) -> bool { false }
```

Streaming stays `complete_stream` (plain chat); streaming tool use is out of
scope (enabled later by this block model).

### 3.3 `Tool` trait extension — `src/tool/mod.rs`

```rust
/// JSON Schema for native calling. Default wraps the args string:
/// {"type":"object","properties":{"args":{"type":"string","description":<args_hint or description>}},"required":["args"]}
/// — every existing tool works natively with no changes.
fn input_schema(&self) -> serde_json::Value { ... }

/// Converts a native tool-use `input` object to the args string `run()`
/// expects. Default: unwrap `{"args": "..."}`; a sole non-string `args`
/// value is serialized; anything else serializes the whole object.
fn args_from_input(&self, input: &serde_json::Value) -> String { ... }
```

Overrides: **write** → real schema `{path, content}` (required both);
**edit** → `{path, old, new}`. Their `args_from_input` default already
produces the JSON string their `run()` parses. `shell`/`calc`/`time`/`web`/
`file` keep the default single-string schema (args_hint as description).
`tool_specs(&ToolRegistry) -> Vec<ToolSpec>` joins `catalog()` as the second
registry view (sorted by name, deterministic).

### 3.4 `ToolMode` config — `src/model/factory.rs` + agent plumbing

```rust
#[derive(..., Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolMode { #[default] Auto, Native, Text }
```

- `ModelConfig` gains `#[serde(default)] pub tool_mode: ToolMode` (old config
  files load unchanged; `skip_serializing_if` default keeps files tidy).
- `Agent` gains a `tool_mode: ToolMode` field + `with_tool_mode(..)` builder
  (default Auto); daemon/main plumb it from the agent's `ModelConfig`.
- Semantics in `solve`:
  - **Text** → always the existing flat-prompt path. `complete_thread` never called.
  - **Native** → native only; `NativeToolsUnsupported` is a hard error (user
    explicitly demanded native).
  - **Auto** → if `!model.supports_native_tools()` → text. Else native; on
    `NativeToolsUnsupported` (runtime, e.g. ollama 400) → `tracing::warn!`,
    set per-agent `native_downgraded: AtomicBool`, redo the step in text mode;
    all later steps/runs of this Agent instance go text directly.

## 4. Provider wire mappings

### 4.1 Anthropic (`anthropic.rs`) — native

Request: existing body + `"tools": [{name, description, input_schema}]`;
messages built from `Thread` (system via existing `build_system`, OAuth
Claude-Code identity prefix preserved):

- `Text` → `{"type":"text","text":..}`
- `ToolUse` → `{"type":"tool_use","id":..,"name":..,"input":{..}}` (assistant)
- `ToolResult` → `{"type":"tool_result","tool_use_id":..,"content":..,"is_error":bool}` (user)

Response: parse `content[]` into blocks (`text` + `tool_use`; thinking blocks
ignored as today), `stop_reason: "tool_use"` → `StopReason::ToolUse`,
`"end_turn"` → `EndTurn`, else `Other`. Existing private response structs are
extended (renamed `RespBlock` to avoid clashing with the public `ContentBlock`).

### 4.2 OpenAI + OpenAiCompat (`openai.rs`) — native

Request: `"tools": [{"type":"function","function":{name, description, parameters}}]`.
Thread → messages: system → `{"role":"system"}`; assistant message with
ToolUse blocks → `{"role":"assistant","content":<text or null>,"tool_calls":
[{"id":..,"type":"function","function":{"name":..,"arguments":"<json string>"}}]}`;
each ToolResult → its own `{"role":"tool","tool_call_id":..,"content":..}`
message (order preserved).

Response: `choices[0].message` → text from `content` (reasoning_content
fallback logic reused), `tool_calls[]` → ToolUse blocks (`arguments` is a
JSON **string** → parse to Value; unparseable → `Value::String(raw)` so the
tool still sees something), `finish_reason: "tool_calls"` → `ToolUse`.

**Unsupported detection (auto downgrade):** HTTP 400 whose body mentions
tools (ollama: `"does not support tools"`; generic: `"tools"` +
`unsupported`/`unknown`) maps to `LoreError::NativeToolsUnsupported` instead
of `Model`. Detection is substring-based and documented at the mapping site.

### 4.3 Error variant — `src/error.rs`

```rust
/// Provider/endpoint cannot do native tool calling (drives auto downgrade).
#[error("native tool calling unsupported: {0}")]
NativeToolsUnsupported(String),
```

## 5. Solve loop — `src/agent/mod.rs`

One shared skeleton, two drivers. Shared: prior-procedure recall + Wilson
hints, scratchpad observation strings, `calls` vec (successful only),
`had_tool_error`, step budget, final-answer memory/note/procedure logic —
all **unchanged**. Per step the driver returns
`StepOutcome::Final(text) | StepOutcome::Calls(Vec<ExecutedCall>)`.

**Text driver** (exactly today's code): flat Prompt with instruction +
catalog, `parse_tool_call`, single call per step, last step forces
plain-text final.

**Native driver**:

- Thread build: `system` = persona identity (+ hints folded in, same as the
  flat path folds context into system for Anthropic); first user message =
  the task. **No tool-JSON instruction** — tools travel in `tools[]`.
- Step: `complete_thread(&thread, &specs)`:
  - Reply has ToolUse blocks → execute **all** in order (registry lookup;
    unknown tool or run error → `ToolResult{is_error:true, content:"ERROR: .."}`;
    same strings also pushed to scratchpad as `[observation] tool(args) → ..`).
    Append assistant message (reply blocks verbatim) + one user message with
    the matching ToolResult blocks. Continue.
  - No ToolUse → final = joined text blocks; done.
- Last step: append a user nudge ("No more tool calls — give the final
  answer based on the results above.") and call with the **same** `tools[]`
  (Anthropic rejects tool-blocked threads without a `tools` param, so tools
  cannot simply be dropped). Any ToolUse in that reply is **not executed**:
  the fell-back guard answers with the last observation instead (mirror of
  today's text-mode last step — raw blocks never leak to the user).
- Memory compat: each executed native call is recorded as
  `ToolCall{tool: name, args: args_from_input(input)}` — the same string
  shape text mode produces, so learned procedures stay interchangeable.
- Auto downgrade (§3.4) wraps the driver choice; the downgraded step re-runs
  through the text driver so no step budget is consumed by the failed probe.

`work()` and the daemon/team flows delegate to `solve` and inherit all of
this untouched.

## 6. Testing strategy

- **Types/registry**: ToolSpec generation (default schema from args_hint;
  write/edit real schemas; deterministic order), `args_from_input` matrix
  (string args / sole-object args / structured object / garbage).
- **Anthropic**: request build with tools + block round-trip (tool_use,
  tool_result, is_error), response parse (text+tool_use mix, stop_reason
  variants, thinking ignored), OAuth system prefix preserved with tools.
- **OpenAI**: message building (assistant tool_calls, role:"tool" ordering),
  arguments string→Value parse incl. unparseable fallback, finish_reason
  mapping, reasoning_content fallback with tools, unsupported-error mapping
  (ollama 400 body → NativeToolsUnsupported; unrelated 400 → Model).
- **Solve native loop** (scripted `Model` with `complete_thread`): happy
  path (tool_use → result → final), parallel calls in one step, tool error
  → is_error result → model corrects, unknown tool, last-step nudge with
  unexecuted ToolUse → fell-back guard, procedure learning parity with text
  mode.
- **Modes**: Text never calls `complete_thread`; Native hard-errors on
  unsupported; Auto downgrades once per Agent (AtomicBool observed), step
  budget not consumed by the failed probe.
- **Regression**: entire existing suite must stay green untouched — that is
  the N5 guarantee.
- Optional: 1 `#[ignore]`d live test (Anthropic native solve), matching the
  existing live-test pattern.

## 7. Phases & gates

| Phase | Content | Gate |
|-------|---------|------|
| 1 | `thread.rs` types; `Model::complete_thread` default + `supports_native_tools`; `NativeToolsUnsupported` variant; `Tool::input_schema`/`args_from_input`; `tool_specs()`; write/edit schema overrides | fmt + clippy `-D warnings` + full suite green (zero behavior change) |
| 2 | Solve skeleton split (text driver = existing code verbatim); native driver; `ToolMode` in ModelConfig + Agent builder + daemon plumbing; auto downgrade; scripted-model tests | same + new solve tests |
| 3 | Anthropic native (`complete_thread` + tools in request + block parse) | same + provider tests |
| 4 | OpenAI/Compat native + unsupported-error mapping | same + provider tests |
| 5 | Docs (README endpoint/feature notes, DESIGN.md decision entry, CHANGELOG) + optional live test | same |

Conventional commits, one phase per commit minimum. 5× stress not required
(no new concurrency primitives beyond one AtomicBool).

## 8. Out of scope (explicit)

- **Codex native** (Responses API `function_call` items) — follow-up phase.
  *Addendum 2026-07-24: completed — see §9.*
- **Streaming tool use** — enabled by this block model, not built now.
  *Addendum 2026-07-24: deliberately deferred as decision D20 — no consumer
  exists (chat streaming carries no tools; solve is batch). Building SSE
  tool-use assembly without a consumer is dead code; the block model makes
  it a bounded add when a consumer appears.*
- **Thinking-block passthrough** (Anthropic `thinking` + tools) — blocks
  remain ignored in parsing, as today. Correct while Lore never sends a
  `thinking` parameter; if extended thinking is ever enabled alongside
  tools, thinking blocks must round-trip in assistant messages.
- **LlmRouter migration** — the router keeps using the text catalog (it
  picks one tool for `act()`; it is not part of the tool-call protocol).

## 9. Addendum (2026-07-24): Codex native follow-up — done

Codex (Responses API over the ChatGPT subscription backend) now implements
`complete_thread`:

- Request: Responses `tools` entries are **flat** (`name`/`description`/
  `parameters` at top level — no nested `function` object). Thread blocks
  map to input items: Text → `message` (`input_text`⁄`output_text`),
  ToolUse → `function_call` (arguments as a JSON string), ToolResult →
  `function_call_output` correlated by `call_id`.
- Response: completed calls are read from `response.output_item.done`
  events (`function_call_arguments.delta` events are skipped — the done
  item is the single source of truth); text via `output_text.delta` as
  before. Same argument tolerance as chat completions (unparseable → raw
  string, missing ids synthesized).
- Unsupported classification is shared with `OpenAiModel` and applied
  **only when the request carried tools**, so plain-chat 400s can never
  masquerade as downgrades. `supports_native_tools = true`; the backend
  remains live-unverified (as all of `codex.rs`), and `auto` mode
  downgrades safely if the backend rejects tools.

Mock remains text-only by design (deterministic test double).
