//! Agent: identity (persona) + personal memory + model.
//!
//! Two things that make an agent "that agent" converge here: **identity** (persona)
//! and **memory** (personal scope). The agent is linked to the memory engine and model
//! via `Arc` handles — both are behind traits, swappable.

mod conversation;
pub mod distill;
mod persona;
pub mod roles;
pub mod work;

pub use conversation::{Conversation, DEFAULT_CONVERSATION_CAP};
pub use persona::Persona;
pub use roles::{preset, presets, Role};
pub use work::{WorkReport, WorkSpec};

use crate::error::{LoreError, Result};
use crate::id::{AgentId, MemoryId};
use crate::memory::retrieval::wilson_lower_bound;
use crate::memory::{
    Memory, MemoryKind, MemoryStore, Outcome, Query, Scope, Scored, SemanticCat, Tier,
};
use crate::model::{Model, Prompt, TokenStream, Turn};
use crate::tool::{catalog, parse_tool_call, ToolCall, ToolContext, ToolRegistry, ToolRouter};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Number of memories included in the prompt during the `respond` loop (top-N most relevant).
const RESPOND_RECALL_LIMIT: usize = 5;
/// Max characters per recalled context line injected into a prompt (keeps the
/// prompt lean when a recalled body is long, e.g. a prior exchange reply).
const RECALL_CONTEXT_CHARS: usize = 400;
/// Importance floor for memories injected as prompt context: keeps low-value
/// automatic traces (exchange/tool/board logs at `AUTO_IMPORTANCE` = 0.2) out,
/// while admitting explicit `remember`/`experience` (0.5) and distilled facts.
/// Recent conversation is already supplied separately via `history`.
const CONTEXT_MIN_IMPORTANCE: f32 = 0.35;

/// Maximum length (characters) of reasoning fallback replies stored in memory.
/// Raw CoT can be thousands of characters; the memory summary is kept short.
const REASONING_MEMORY_CAP: usize = 500;

/// Reflect: minimum access count required for a memory to become a distillation candidate.
/// Frequently recalled = importance signal; cold memories are not promoted.
const REFLECT_MIN_ACCESS: u32 = 2;

/// Reflect: maximum memories distilled in a single run (model cost limit).
const REFLECT_MAX_PER_RUN: usize = 5;

/// Importance of distilled semantic facts (higher than user records:
/// these are the essence of proven, frequently used knowledge).
const REFLECT_IMPORTANCE: f32 = 0.75;

/// Default step limit for the `solve` loop.
pub const DEFAULT_SOLVE_STEPS: usize = 5;

/// Upper ceiling for the `solve` step limit (rogue loop / cost protection).
pub const MAX_SOLVE_STEPS: usize = 10;

/// Limit for prior procedure search before `solve` (dedup + hint candidates).
const SOLVE_PRIOR_LIMIT: usize = 3;

/// Required Wilson lower bound for a procedure to enter the solve prompt as "proven"
/// (~1 success is enough; unproven procedures are not used as hints).
const SOLVE_PRIOR_MIN_WILSON: f64 = 0.2;

/// Serializable identity of an agent (excluding Arc handles).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct AgentRecord {
    id: AgentId,
    persona: Persona,
    /// Per-agent model configuration (optional; absent → env fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<crate::model::ModelConfig>,
    /// Identity extra lines stored separately for backward compat.
    /// (Already part of Persona.extra, but kept for migration.
    ///  This field is purely additive: loaded values are merged into persona.extra.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    extra: Vec<String>,
    /// Whether this agent distills knowledge after each task.
    /// None (absent) = true (default). Set to Some(false) to opt out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    distill: Option<bool>,
}

/// An agent with identity, memory, and model.
#[derive(Clone)]
pub struct Agent {
    /// Unique identity.
    pub id: AgentId,
    /// Identity/character.
    pub persona: Persona,
    /// Personal memory engine (shareable).
    pub memory: Arc<dyn MemoryStore>,
    /// Reasoning engine (LLM abstraction).
    pub model: Arc<dyn Model>,
    /// Per-agent model config (optional; absent → env fallback).
    model_config: Option<crate::model::ModelConfig>,
    /// Whether this agent distills knowledge after each task.
    /// None = true (default). Some(false) = opt-out.
    distill: Option<bool>,
    /// Optional tool context (registry + router).
    tools: Option<Arc<ToolContext>>,
}

impl Agent {
    /// Creates a new agent (generates a new AgentId).
    pub fn new(persona: Persona, memory: Arc<dyn MemoryStore>, model: Arc<dyn Model>) -> Self {
        Self {
            id: AgentId::new(),
            persona,
            memory,
            model,
            model_config: None,
            distill: None, // None = true (default)
            tools: None,
        }
    }

    /// Creates an agent with a specific AgentId (for loading from persistence).
    pub fn with_id(
        id: AgentId,
        persona: Persona,
        memory: Arc<dyn MemoryStore>,
        model: Arc<dyn Model>,
    ) -> Self {
        Self {
            id,
            persona,
            memory,
            model,
            model_config: None,
            distill: None,
            tools: None,
        }
    }

    /// Attaches a tool registry and router to the agent (builder pattern).
    pub fn with_tools(mut self, registry: ToolRegistry, router: Arc<dyn ToolRouter>) -> Self {
        self.tools = Some(Arc::new(ToolContext { registry, router }));
        self
    }

    /// Sets per-agent model config (builder pattern).
    pub fn with_model_config(mut self, cfg: crate::model::ModelConfig) -> Self {
        self.model_config = Some(cfg);
        self
    }

    /// Sets distill opt-out (builder pattern). `false` disables post-task distillation.
    pub fn with_distill(mut self, v: bool) -> Self {
        self.distill = Some(v);
        self
    }

    /// Returns the per-agent model config (None → env fallback).
    pub fn model_config(&self) -> Option<&crate::model::ModelConfig> {
        self.model_config.as_ref()
    }

    /// Whether this agent distills after each task. None/Some(true) → distill enabled;
    /// Some(false) → opt-out.
    pub fn should_distill(&self) -> bool {
        self.distill.unwrap_or(true)
    }

    /// This agent's personal memory scope.
    pub fn scope(&self) -> Scope {
        Scope::Agent(self.id.clone())
    }

    /// Serializes identity (id + persona + model_config + extra + distill) to JSON (Arc handles not included).
    pub fn to_json(&self) -> Result<String> {
        // Merge persona.extra into the record's extra field for backward compat.
        // persona.extra already holds these values; the record's extra is a
        // redundant store that allows migration.
        let extra = self.persona.extra.clone();
        let rec = AgentRecord {
            id: self.id.clone(),
            persona: self.persona.clone(),
            model: self.model_config.clone(),
            extra,
            distill: self.distill,
        };
        Ok(serde_json::to_string_pretty(&rec)?)
    }

    /// Reconstructs an agent from JSON identity (with the given memory + model).
    /// Per-agent model_config, if present, is stored but the provided model
    /// (usually env-based or pre-built) is used as-is. The daemon will
    /// rebuild the model from model_config when needed.
    pub fn from_json(
        json: &str,
        memory: Arc<dyn MemoryStore>,
        model: Arc<dyn Model>,
    ) -> Result<Self> {
        let rec: AgentRecord = serde_json::from_str(json)?;
        let mut persona = rec.persona;
        for line in &rec.extra {
            if !persona.extra.contains(line) {
                persona.extra.push(line.clone());
            }
        }
        Ok(Self {
            id: rec.id,
            persona,
            memory,
            model,
            model_config: rec.model,
            distill: rec.distill,
            tools: None,
        })
    }

    /// Saves identity to a file.
    /// Atomic write: write to `<id>.tmp` first + flush, then `rename`.
    /// If a crash/SIGKILL arrives mid-write, either the old or the new complete file remains —
    /// never a half-written JSON. (The truncate-then-write approach caused agents
    /// to silently disappear on restart: corrupt JSON → load error → skip.)
    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let json = self.to_json()?;
        std::fs::write(&tmp, json).map_err(|e| LoreError::Storage(e.to_string()))?;
        std::fs::rename(&tmp, path).map_err(|e| LoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Loads identity from a file (reborn with the given memory + model).
    pub fn load_from(
        path: impl AsRef<Path>,
        memory: Arc<dyn MemoryStore>,
        model: Arc<dyn Model>,
    ) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| LoreError::Storage(e.to_string()))?;
        Self::from_json(&json, memory, model)
    }

    /// Experiences an event: writes to episodic memory.
    pub async fn experience(
        &self,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<()> {
        self.memory
            .remember(Memory::episodic(self.scope(), title, body))
            .await?;
        Ok(())
    }

    /// System-generated (automatic) episodic note: exchange/tool/message traces.
    /// Born with [`Memory::AUTO_IMPORTANCE`] — unused, decay may reclaim it;
    /// accessed, `access_count` preserves it. For explicit user records,
    /// use [`Agent::experience`] (default importance).
    pub async fn note(&self, title: impl Into<String>, body: impl Into<String>) -> Result<()> {
        self.memory
            .remember(
                Memory::episodic(self.scope(), title, body)
                    .with_importance(Memory::AUTO_IMPORTANCE),
            )
            .await?;
        Ok(())
    }

    /// Stores an arbitrary memory record (in own scope).
    pub async fn remember(&self, mut mem: Memory) -> Result<()> {
        mem.scope = self.scope();
        self.memory.remember(mem).await?;
        Ok(())
    }

    /// **Reflection: episodic → semantic distillation.**
    ///
    /// Frequently recalled episodic memories (access ≥ [`REFLECT_MIN_ACCESS`]) are
    /// distilled by the model into a single-sentence permanent fact and promoted
    /// to the semantic tier (high importance, `distilled:<source-id>` key); the
    /// original memory is archived via soft-delete (audit trail remains). Thus,
    /// highly repeated experiences become abstract knowledge — the learning face of memory evolution.
    /// A single memory/model error does not kill the run; the distilled count is returned.
    pub async fn reflect(&self) -> Result<usize> {
        // Candidates: live episodic, sufficiently accessed. Browse recall does NOT
        // count access — candidate selection produces no false signal.
        let candidates = self
            .memory
            .recall(
                &self.scope(),
                &Query::new("").tier(Tier::Episodic).limit(200),
            )
            .await?;
        let mut distilled = 0usize;
        for cand in candidates
            .iter()
            .filter(|s| s.item.access_count >= REFLECT_MIN_ACCESS)
            .take(REFLECT_MAX_PER_RUN)
        {
            let prompt = Prompt {
                system: format!(
                    "{}\n\nTask: Distill the given memory into a single-sentence permanent fact. \
                     Write only the category on the first line: FACT (objective fact) or PREFERENCE \
                     (preference/habit). Write the distilled sentence on the second line.",
                    self.persona.identity_prompt()
                ),
                user: cand.item.searchable_text(),
                ..Default::default()
            };
            let raw = match self.model.complete(&prompt).await {
                Ok(c) => c.text.trim().to_string(),
                Err(e) => {
                    tracing::warn!(error = %e, "reflect: distillation skipped (model error)");
                    continue;
                }
            };
            // Parse category marker: "PREFERENCE\n<sentence>" → Preference;
            // unmarked/unknown output → Fact, text as-is (robust).
            let (category, fact) = match raw.split_once('\n') {
                Some((tag, body)) if tag.trim().eq_ignore_ascii_case("preference") => {
                    (SemanticCat::Preference, body.trim().to_string())
                }
                Some((tag, body)) if tag.trim().eq_ignore_ascii_case("fact") => {
                    (SemanticCat::Fact, body.trim().to_string())
                }
                _ => (SemanticCat::Fact, raw),
            };
            if fact.is_empty() {
                continue;
            }
            let mem = Memory::semantic(self.scope(), fact, category)
                .with_importance(REFLECT_IMPORTANCE)
                .with_key(format!("distilled:{}", cand.item.id));
            if let Err(e) = self.memory.remember(mem).await {
                tracing::warn!(error = %e, "reflect: semantic record could not be written");
                continue;
            }
            if let Err(e) = self.memory.forget(&cand.item.id).await {
                tracing::warn!(error = %e, "reflect: source memory could not be archived");
            }
            distilled += 1;
        }
        if distilled > 0 {
            tracing::info!(distilled, "reflect: episodic → semantic promotion complete");
        }
        Ok(distilled)
    }

    /// Recalls from personal memory (+ World).
    ///
    /// Records returned from textual queries are counted as "accessed" and reinforced
    /// (`last_access` refreshed, `access_count` incremented) — retrieval itself
    /// feeds the decay signal; frequently recalled records are not forgotten. Browse (textless)
    /// bulk scans (graph construction, board reading) do NOT reinforce: otherwise
    /// every full scan would mark every record as "used" and decay would die.
    pub async fn recall(&self, query: &Query) -> Result<Vec<Scored<Memory>>> {
        let res = self.memory.recall(&self.scope(), query).await?;
        if !query.text.trim().is_empty() && !res.is_empty() {
            // Soft-deleted entries (arriving via include_deleted) are not revived.
            let ids: Vec<MemoryId> = res
                .iter()
                .filter(|s| s.item.deleted_at.is_none())
                .map(|s| s.item.id.clone())
                .collect();
            if !ids.is_empty() {
                // A reinforcement error does NOT lose recalled items — logged and skipped.
                if let Err(e) = self.memory.reinforce_many(&ids, Outcome::Accessed).await {
                    tracing::warn!(error = %e, "access reinforcement could not be written");
                }
            }
        }
        Ok(res)
    }

    /// Acts on input: if the router selects a tool, runs it (and remembers),
    /// otherwise reasons via `respond`.
    pub async fn act(&self, input: &str) -> Result<String> {
        if let Some(ctx) = &self.tools {
            if let Some(call) = ctx.router.route(input, &ctx.registry).await {
                if let Some(tool) = ctx.registry.get(&call.tool) {
                    let result = tool.run(&call.args).await?;
                    self.memory
                        .remember(
                            Memory::episodic(
                                self.scope(),
                                format!("used {} tool", call.tool),
                                format!("input: {input} → result: {result}"),
                            )
                            .with_importance(Memory::AUTO_IMPORTANCE),
                        )
                        .await?;
                    return Ok(result);
                }
            }
        }
        self.respond(input).await
    }

    /// Multi-step tool loop (ReAct): **think → call tool → feed observation back →
    /// think again → ... → final response.** The model either returns a tool call JSON
    /// (executed, observation added to scratchpad) or plain text as the final response
    /// at each step. Tool errors are also fed back as observations — the model can self-correct.
    /// On the last step, tool rights expire and a final response is requested (loop guaranteed to terminate).
    ///
    /// Procedure learning: before the loop, similar past solutions (procedural tier)
    /// are retrieved; proven ones (Wilson ≥ [`SOLVE_PRIOR_MIN_WILSON`]) enter
    /// the prompt as guiding hints. On successful completion, a past procedure
    /// following the same tool sequence is reinforced with `Success` instead of
    /// creating a new record; otherwise a new procedure is learned. In a fallback
    /// run (ended by step limit), injected procedures receive `Failure` — Wilson evidence accumulates bidirectionally.
    pub async fn solve(&self, ctx: &ToolContext, input: &str, max_steps: usize) -> Result<String> {
        let max_steps = max_steps.clamp(1, MAX_SOLVE_STEPS);
        let catalog = catalog(&ctx.registry);

        // Prior procedures: both dedup candidates and (if proven) hints.
        // A recall failure is logged but not fatal — solve proceeds without priors.
        let priors = match self
            .recall(
                &Query::new(input)
                    .tier(Tier::Procedural)
                    .semantic()
                    .limit(SOLVE_PRIOR_LIMIT),
            )
            .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "solve: prior procedures could not be recalled");
                Vec::new()
            }
        };
        let mut hints: Vec<String> = Vec::new();
        let mut injected: Vec<MemoryId> = Vec::new();
        for p in &priors {
            if let MemoryKind::Procedural {
                title,
                steps,
                successes,
                failures,
            } = &p.item.kind
            {
                if wilson_lower_bound(*successes, *failures) >= SOLVE_PRIOR_MIN_WILSON {
                    hints.push(format!(
                        "[prior solution] {title} — steps followed: {}",
                        steps.join(" → ")
                    ));
                    injected.push(p.item.id.clone());
                }
            }
        }

        // `hints` and `scratchpad` (observations) are kept separate: the
        // final record/note logic only considers actual observations.
        let mut scratchpad: Vec<String> = Vec::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        // Was a tool error seen along the procedure path? (For Failure attribution —
        // the model's inability to stop is not the procedure's fault)
        let mut had_tool_error = false;
        for step in 0..max_steps {
            let last = step + 1 == max_steps;
            let instruction = if last {
                "No more tool calls. Give the FINAL answer based on observations as plain text."
                    .to_string()
            } else {
                format!(
                    "If using a tool, return ONLY this JSON: \
                     {{\"tool\":\"<name>\",\"args\":\"<argument — in tool args format>\"}}\n\
                     Available tools:\n{catalog}\n\
                     If the answer is ready, do not call a tool; give the final response as plain text."
                )
            };
            let prompt = Prompt {
                system: format!("{}\n\n{instruction}", self.persona.identity_prompt()),
                context: hints.iter().chain(scratchpad.iter()).cloned().collect(),
                user: input.to_string(),
                ..Default::default()
            };
            let completion = self.model.complete(&prompt).await?;

            // Tool call? (no tool rights on the last step — text is accepted as final.)
            if !last {
                if let Some(call) = parse_tool_call(&completion.text) {
                    let (obs, ok) = match ctx.registry.get(&call.tool) {
                        Some(tool) => match tool.run(&call.args).await {
                            Ok(o) => (o, true),
                            // An error is also an observation: the model can correct in the next step.
                            Err(e) => (format!("ERROR: {e}"), false),
                        },
                        None => (format!("ERROR: no such tool '{}'", call.tool), false),
                    };
                    // Only SUCCESSFUL calls enter the learned procedure —
                    // failed attempts remain in observations, do not pollute the procedure.
                    if ok {
                        calls.push(call.clone());
                    } else {
                        had_tool_error = true;
                    }
                    scratchpad.push(format!(
                        "[observation] {}({}) → {}",
                        call.tool, call.args, obs
                    ));
                    continue;
                }
            }

            // Final response: if tool traces exist, a procedural trace is also remembered.
            // If the model ignores the instruction on the last step and returns a tool JSON again,
            // do not leak raw JSON to the user: respond with the latest observation (reachable
            // only from the `last` branch — on prior steps, JSON goes through `continue`
            // into the tool loop).
            let fell_back = parse_tool_call(&completion.text).is_some();
            let text = if fell_back {
                match scratchpad.last() {
                    Some(obs) => format!("step limit reached; last info: {obs}"),
                    None => "step limit reached; no final response generated.".to_string(),
                }
            } else {
                completion.text
            };
            if fell_back && !injected.is_empty() && had_tool_error {
                // Failure is processed ONLY if a tool error was seen along the procedure path.
                // Hitting the step limit alone is not evidence against the procedure —
                // the model may simply "not know when to stop" (no unfair penalty).
                if let Err(e) = self
                    .memory
                    .reinforce_many(&injected, Outcome::Failure)
                    .await
                {
                    tracing::warn!(error = %e, "procedure failure could not be processed");
                }
            }
            if scratchpad.is_empty() {
                self.remember_exchange(input, &text).await?;
            } else {
                // Automatic trace: unused, decay reclaims it; accessed, it is preserved.
                self.note(
                    format!(
                        "completed task '{input}' with {} tool steps",
                        scratchpad.len()
                    ),
                    format!("{}\nResult: {text}", scratchpad.join("\n")),
                )
                .await?;
                if !fell_back && !calls.is_empty() {
                    self.learn_procedure(input, &calls, &priors).await;
                }
            }
            return Ok(text);
        }
        // Unreachable: last step always returns; safety belt just in case.
        Err(crate::error::LoreError::Model(
            "solve step limit exceeded".into(),
        ))
    }

    /// Learns a successful tool sequence as a procedure.
    ///
    /// If a past procedure follows the same tool sequence (ordered tool names),
    /// no duplicate is created — the existing record is reinforced with `Success` (Wilson
    /// evidence accumulates, decay protection strengthens). Otherwise, a new
    /// procedural record is opened with steps in `tool: args` format and its first success is processed.
    /// A learning error never corrupts the solve result (logged and skipped).
    async fn learn_procedure(&self, input: &str, calls: &[ToolCall], priors: &[Scored<Memory>]) {
        let seq: Vec<&str> = calls.iter().map(|c| c.tool.as_str()).collect();
        for p in priors {
            if let MemoryKind::Procedural { steps, .. } = &p.item.kind {
                let prior_seq: Vec<&str> = steps
                    .iter()
                    .map(|s| s.split(':').next().unwrap_or(s.as_str()).trim())
                    .collect();
                if prior_seq == seq {
                    if let Err(e) = self
                        .memory
                        .reinforce_many(std::slice::from_ref(&p.item.id), Outcome::Success)
                        .await
                    {
                        tracing::warn!(error = %e, "procedure could not be reinforced");
                    }
                    return;
                }
            }
        }
        let steps: Vec<String> = calls
            .iter()
            .map(|c| format!("{}: {}", c.tool, c.args))
            .collect();
        let mem = Memory::procedural(self.scope(), format!("task '{input}'"), steps);
        match self.memory.remember(mem).await {
            Ok(id) => {
                if let Err(e) = self.memory.reinforce(&id, Outcome::Success).await {
                    tracing::warn!(error = %e, "procedure first success could not be processed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "procedure could not be saved"),
        }
    }

    /// Recalls via HyDE: generates a hypothetical answer with the model, then searches by embedding it.
    ///
    /// A question and its answer have different shapes in embedding space; a hypothetical answer
    /// is closer to real records as an embedding (even if wrong, it is "answer-shaped").
    pub async fn recall_hyde(&self, input: &str) -> Result<Vec<Scored<Memory>>> {
        let prompt = Prompt {
            system: "Answer the question with a short, factual single sentence (guess if unsure)."
                .into(),
            user: input.to_string(),
            ..Default::default()
        };
        let hypo = self.model.complete(&prompt).await?.text;
        let q = Query::new(input).semantic().embed_text(hypo);
        self.recall(&q).await
    }

    /// Responds to an input with identity + recalled memories; remembers the interaction.
    ///
    /// Loop: **recall → identity+context prompt → model → remember new memory.**
    pub async fn respond(&self, input: &str) -> Result<String> {
        self.respond_with(input, &[]).await
    }

    /// Like `respond`; extra context lines (e.g. team responses, board summary) are
    /// prepended to the prompt. Foundation of supervisor/synthesis flows.
    pub async fn respond_with(&self, input: &str, extra: &[String]) -> Result<String> {
        self.think(input, extra, Vec::new()).await
    }

    /// Multi-turn chat: the conversation's working memory (last N turns, verbatim) is
    /// included in the prompt, and the exchange is recorded to the window after the response.
    /// The long-term memory loop (same as `respond`) also closes — turns that fall out
    /// of the window can return via retrieval.
    pub async fn converse(&self, convo: &mut Conversation, input: &str) -> Result<String> {
        let reply = self.think(input, &[], convo.history()).await?;
        convo.record(input, &reply);
        Ok(reply)
    }

    /// Streams the response in chunks; when the stream ends, the full response is
    /// saved as an episodic memory (memory loop still closes — just later).
    pub async fn respond_stream(&self, input: &str) -> Result<TokenStream> {
        self.think_stream(input, Vec::new()).await
    }

    /// Shared reasoning loop: **recall → identity+context+history prompt → model →
    /// remember new memory.**
    async fn think(&self, input: &str, extra: &[String], history: Vec<Turn>) -> Result<String> {
        let prompt = self.build_prompt(input, extra, history).await?;
        let completion = self.model.complete(&prompt).await?;
        // Reasoning fallback (empty content → chain of thought): the user sees
        // the full text, but raw CoT is NOT written to memory — it is stored trimmed
        // (preventing context pollution + prompt bloat on subsequent recalls).
        // A memory write failure does NOT lose the response — logged and continued
        // (consistent with think_stream, where post-stream memory errors are also
        // non-fatal: the user already received the answer).
        if completion.reasoning_fallback {
            let mut capped: String = completion.text.chars().take(REASONING_MEMORY_CAP).collect();
            if completion.text.chars().count() > REASONING_MEMORY_CAP {
                capped.push('…');
            }
            if let Err(e) = self.remember_exchange(input, &capped).await {
                tracing::warn!(error = %e, "post-respond memory could not be saved");
            }
        } else {
            if let Err(e) = self.remember_exchange(input, &completion.text).await {
                tracing::warn!(error = %e, "post-respond memory could not be saved");
            }
        }
        Ok(completion.text)
    }

    /// Streaming version of `think`: chunks are published as they arrive; the accumulated
    /// full response is written to memory at stream end. If an error arrives, it
    /// propagates and the stream ends (partial responses are NOT saved as memories).
    pub(crate) async fn think_stream(
        &self,
        input: &str,
        history: Vec<Turn>,
    ) -> Result<TokenStream> {
        let prompt = self.build_prompt(input, &[], history).await?;
        let inner = self.model.complete_stream(&prompt).await?;
        let agent = self.clone();
        let input = input.to_string();
        let wrapped = futures::stream::unfold(
            Some((inner, String::new(), agent, input)),
            |state| async move {
                let (mut inner, mut acc, agent, input) = state?;
                match inner.next().await {
                    Some(Ok(chunk)) => {
                        acc.push_str(&chunk);
                        Some((Ok(chunk), Some((inner, acc, agent, input))))
                    }
                    Some(Err(e)) => Some((Err(e), None)),
                    None => {
                        // Stream ended: memory loop closes.
                        if let Err(e) = agent.remember_exchange(&input, &acc).await {
                            tracing::warn!(error = %e, "post-stream memory could not be saved");
                        }
                        None
                    }
                }
            },
        );
        Ok(Box::pin(wrapped))
    }

    /// Builds a prompt from identity + recalled context + conversation history.
    async fn build_prompt(
        &self,
        input: &str,
        extra: &[String],
        history: Vec<Turn>,
    ) -> Result<Prompt> {
        let recalled = self
            .recall(
                &Query::new(input)
                    .limit(RESPOND_RECALL_LIMIT)
                    .semantic()
                    .min_importance(CONTEXT_MIN_IMPORTANCE),
            )
            .await?;
        let mut context: Vec<String> = extra.to_vec();
        // Inject the FULL recalled content (title + body / statement / steps), not
        // just a title — otherwise remembered facts never reach the model. Long
        // bodies are capped so a single memory can't bloat the prompt.
        context.extend(recalled.iter().map(|s| {
            let line = s.item.recall_context();
            if line.chars().count() > RECALL_CONTEXT_CHARS {
                line.chars().take(RECALL_CONTEXT_CHARS).collect::<String>() + "…"
            } else {
                line
            }
        }));
        Ok(Prompt {
            system: self.persona.identity_prompt(),
            context,
            history,
            user: input.to_string(),
        })
    }

    /// Records an exchange as an episodic memory (automatic importance — high-volume
    /// records: decay must be able to reclaim them, or memory grows without bound).
    async fn remember_exchange(&self, input: &str, reply: &str) -> Result<()> {
        self.note(format!("responded to '{input}': "), reply.to_string())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{InMemoryStore, SemanticCat};
    use crate::model::MockModel;

    fn agent_with(name: &str, store: Arc<dyn MemoryStore>) -> Agent {
        let persona = Persona::new(name, "researcher").with_trait("curious");
        Agent::new(persona, store, Arc::new(MockModel::new()))
    }

    #[tokio::test]
    async fn respond_uses_recalled_memory_and_records_episode() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());

        agent
            .experience("Learned Rust", "Studied ownership and borrow checker")
            .await
            .unwrap();

        let before = agent.recall(&Query::new("rust")).await.unwrap().len();
        assert_eq!(before, 1);

        let reply = agent.respond("what do you know about rust").await.unwrap();
        // Recalled context should be reflected in the response.
        assert!(reply.contains("recalling"));

        // A new episodic memory was added after respond.
        let after = agent.recall(&Query::new("rust")).await.unwrap().len();
        assert!(after > before);
    }

    #[tokio::test]
    async fn converse_carries_working_memory_across_turns() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store);
        let mut convo = Conversation::new();

        // First turn: no history.
        let first = agent.converse(&mut convo, "hi").await.unwrap();
        assert!(!first.contains("chat history"), "no history on first turn");
        assert_eq!(convo.len(), 2, "exchange recorded to window");

        // Second turn: previous 2 messages (user+assistant) are included in the prompt.
        let second = agent.converse(&mut convo, "what did I say?").await.unwrap();
        assert!(
            second.contains("chat history: 2 messages"),
            "model saw history: {second}"
        );
        assert_eq!(convo.len(), 4);

        // respond, however, remains without history (behavior unchanged).
        let plain = agent.respond("hello again").await.unwrap();
        assert!(!plain.contains("chat history"));
    }

    /// Test model that returns scripted replies in sequence (for ReAct scenarios).
    struct SeqModel(std::sync::Mutex<std::collections::VecDeque<String>>);
    impl SeqModel {
        fn new(replies: &[&str]) -> Self {
            Self(std::sync::Mutex::new(
                replies.iter().map(|s| s.to_string()).collect(),
            ))
        }
    }
    #[async_trait::async_trait]
    impl Model for SeqModel {
        async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
            let text = self
                .0
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "no reply left".into());
            Ok(crate::model::Completion::new(text))
        }
    }

    fn calc_ctx() -> ToolContext {
        use crate::tool::{CalcTool, KeywordRouter};
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));
        ToolContext {
            registry: reg,
            router: Arc::new(KeywordRouter::new()),
        }
    }

    #[tokio::test]
    async fn solve_chains_tools_and_feeds_observations_back() {
        // Scenario: (3+4)*6 — model chains two tool steps, then gives a final response.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            r#"{"tool":"calc","args":"7 * 6"}"#,
            "The result is 42.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

        let out = agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();
        assert_eq!(out, "The result is 42.");

        // Two tool steps were remembered as a procedural trace.
        let mems = agent.recall(&Query::new("task")).await.unwrap();
        assert!(!mems.is_empty());
        assert!(mems
            .iter()
            .find(|m| m.item.summary().contains("2 tool steps"))
            .expect("should find tool steps note")
            .item
            .summary()
            .contains("2 tool steps"));
    }

    #[tokio::test]
    async fn solve_recovers_from_tool_error_and_bad_tool() {
        // Model tries a non-existent tool first, then bad arguments; corrects via observations.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"google","args":"x"}"#,
            r#"{"tool":"calc","args":"5 5"}"#,
            r#"{"tool":"calc","args":"5 + 5"}"#,
            "The answer is 10.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

        let out = agent.solve(&calc_ctx(), "5+5", 6).await.unwrap();
        assert_eq!(out, "The answer is 10.");
    }

    /// Scripted model that records Prompt.context on each call (injection test).
    struct CaptureCtxModel {
        replies: std::sync::Mutex<std::collections::VecDeque<String>>,
        seen: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    }
    impl CaptureCtxModel {
        fn new(replies: &[&str], seen: Arc<std::sync::Mutex<Vec<Vec<String>>>>) -> Self {
            Self {
                replies: std::sync::Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
                seen,
            }
        }
    }
    #[async_trait::async_trait]
    impl Model for CaptureCtxModel {
        async fn complete(&self, p: &Prompt) -> Result<crate::model::Completion> {
            self.seen.lock().unwrap().push(p.context.clone());
            let text = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "no reply left".into());
            Ok(crate::model::Completion::new(text))
        }
    }

    #[tokio::test]
    async fn solve_success_learns_procedure_with_wilson() {
        // Successful tool chain becomes a Procedural record (H1: Wilson is fed).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            r#"{"tool":"calc","args":"7 * 6"}"#,
            "The result is 42.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

        agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();

        let procs = agent
            .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "one procedure should be learned");
        let m = &procs[0].item;
        assert!(
            m.summary().contains("1\u{2713}/0\u{2717}"),
            "first success processed: {}",
            m.summary()
        );
        let crate::memory::MemoryKind::Procedural { steps, .. } = &m.kind else {
            panic!("expected procedural");
        };
        assert_eq!(
            steps,
            &vec!["calc: 3 + 4".to_string(), "calc: 7 * 6".to_string()]
        );
    }

    #[tokio::test]
    async fn repeated_solve_reinforces_instead_of_duplicating() {
        // Similar task using the same tool sequence: existing procedure is reinforced
        // with Success instead of creating a new record (Wilson evidence accumulates, no dup bloat).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            // First task.
            r#"{"tool":"calc","args":"3 + 4"}"#,
            r#"{"tool":"calc","args":"7 * 6"}"#,
            "The result is 42.",
            // Similar task — same tool sequence, different arguments.
            r#"{"tool":"calc","args":"5 + 2"}"#,
            r#"{"tool":"calc","args":"7 * 6"}"#,
            "The result is again 42.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

        agent.solve(&calc_ctx(), "(3+4)*6?", 5).await.unwrap();
        agent.solve(&calc_ctx(), "(5+2)*6?", 5).await.unwrap();

        let procs = agent
            .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
            .await
            .unwrap();
        assert_eq!(procs.len(), 1, "no duplicate procedure should be created");
        assert!(
            procs[0].item.summary().contains("2\u{2713}/0\u{2717}"),
            "repeated success should reinforce existing procedure: {}",
            procs[0].item.summary()
        );
    }

    #[tokio::test]
    async fn proven_procedure_steps_enter_solve_prompt() {
        // Proven procedure (Wilson ≥ threshold) enters the next solve's prompt
        // as a guiding hint — also feeds Wilson behavior.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            "The answer is 7.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
        agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

        // Second solve: capture prompts (SAME identity — required for scope isolation).
        let seen: Arc<std::sync::Mutex<Vec<Vec<String>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let cap = Arc::new(CaptureCtxModel::new(
            &["The answer is again 7."],
            seen.clone(),
        ));
        let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, cap);
        agent2.id = agent.id.clone();
        agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

        let seen = seen.lock().unwrap();
        let first = seen.first().expect("at least one prompt");
        assert!(
            first.iter().any(|c| c.contains("calc: 3 + 4")),
            "prior procedure steps should enter prompt: {first:?}"
        );
    }

    #[tokio::test]
    async fn fallback_without_tool_errors_does_not_penalize_procedure() {
        // If the model keeps calling tools and hits the limit (tools all SUCCESSFUL),
        // this is not the procedure's fault — Failure is NOT applied (no noisy signal).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            "The answer is 7.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
        agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

        // Second solve: model never gives a final response, but tools are all SUCCESSFUL.
        let stuck = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
        ]));
        let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, stuck);
        agent2.id = agent.id.clone();
        let out = agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();
        assert!(
            out.contains("step limit reached"),
            "should end with fallback: {out}"
        );

        let procs = agent2
            .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
            .await
            .unwrap();
        assert!(
            procs[0].item.summary().contains("1\u{2713}/0\u{2717}"),
            "procedure not penalized without tool errors: {}",
            procs[0].item.summary()
        );
    }

    #[tokio::test]
    async fn fallback_with_tool_errors_marks_injected_procedure_failure() {
        // Fallback + tool error along the procedure path: evidence against the procedure —
        // Failure is applied (Wilson penalizes, decay protection weakens).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            "The answer is 7.",
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store.clone(), model);
        agent.solve(&calc_ctx(), "3+4?", 5).await.unwrap();

        // Second solve: non-existent tool (ERROR observation) + unstoppable model.
        let stuck = Arc::new(SeqModel::new(&[
            r#"{"tool":"nonexistent","args":"x"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
            r#"{"tool":"calc","args":"1 + 1"}"#,
        ]));
        let mut agent2 = Agent::new(Persona::new("Aria", "solver"), store, stuck);
        agent2.id = agent.id.clone();
        let out = agent2.solve(&calc_ctx(), "3+4?", 5).await.unwrap();
        assert!(
            out.contains("step limit reached"),
            "should end with fallback: {out}"
        );

        let procs = agent2
            .recall(&Query::new("?").tier(crate::memory::Tier::Procedural))
            .await
            .unwrap();
        assert!(
            procs[0].item.summary().contains("1\u{2713}/1\u{2717}"),
            "failed path should apply Failure to procedure: {}",
            procs[0].item.summary()
        );
    }

    #[tokio::test]
    async fn exchange_records_are_auto_importance() {
        // Exchange records must be born with automatic importance so decay can reclaim them;
        // explicit experience() records keep their default (higher) importance.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());
        agent.respond("hi").await.unwrap();
        agent
            .experience("important", "explicit record")
            .await
            .unwrap();

        let all = agent.recall(&Query::new("").limit(10)).await.unwrap();
        let auto = all
            .iter()
            .find(|s| s.item.summary().contains("responded"))
            .expect("exchange record must exist");
        assert_eq!(auto.item.importance, Memory::AUTO_IMPORTANCE);
        let explicit = all
            .iter()
            .find(|s| s.item.summary().contains("important"))
            .expect("explicit record must exist");
        assert!(explicit.item.importance > Memory::AUTO_IMPORTANCE);
    }

    #[tokio::test]
    async fn solve_prompt_includes_tool_args_format() {
        /// Test model that captures the prompt system and returns a direct final response.
        struct CaptureModel(std::sync::Mutex<String>);
        #[async_trait::async_trait]
        impl Model for CaptureModel {
            async fn complete(&self, p: &Prompt) -> Result<crate::model::Completion> {
                *self.0.lock().unwrap() = p.system.clone();
                Ok(crate::model::Completion::new("done"))
            }
        }

        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(CaptureModel(std::sync::Mutex::new(String::new())));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model.clone());
        let _ = agent.solve(&calc_ctx(), "23+17", 3).await.unwrap();
        let sys = model.0.lock().unwrap().clone();
        assert!(
            sys.contains("args format:"),
            "solve tells model the args format: {sys}"
        );
    }

    #[tokio::test]
    async fn solve_last_step_never_leaks_raw_tool_json() {
        // If the model ignores the instruction on the last step and returns a tool JSON again,
        // raw JSON must not leak to the user as the "final response" — respond with observations.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[
            r#"{"tool":"calc","args":"3 + 4"}"#,
            r#"{"tool":"calc","args":"7 * 6"}"#, // last step: still JSON
        ]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);
        let out = agent.solve(&calc_ctx(), "(3+4) then?", 2).await.unwrap();
        assert!(
            !out.contains(r#"{"tool""#),
            "raw tool JSON must not leak: {out}"
        );
        assert!(
            out.contains('7'),
            "available observation carried to response: {out}"
        );
    }

    #[tokio::test]
    async fn solve_last_step_forces_final_answer() {
        // Step limit 1: even if the model wants to call a tool, it WON'T run; raw JSON
        // is also not leaked — an explanatory final text is returned.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(SeqModel::new(&[r#"{"tool":"calc","args":"1+1"}"#]));
        let agent = Agent::new(Persona::new("Aria", "solver"), store, model);

        let out = agent.solve(&calc_ctx(), "1+1", 1).await.unwrap();
        assert!(
            !out.contains(r#"{"tool""#),
            "raw tool JSON does not leak: {out}"
        );
        assert!(out.contains("limit"), "explanatory message returned: {out}");
    }

    #[tokio::test]
    async fn respond_stream_yields_chunks_and_remembers_at_end() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store);

        let mut stream = agent.respond_stream("a strange topic").await.unwrap();
        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            full.push_str(&chunk.unwrap());
        }
        drop(stream);
        assert!(
            full.contains("a strange topic"),
            "stream carried full response"
        );

        // When the stream ended, the exchange was recorded as episodic.
        let mems = agent.recall(&Query::new("strange")).await.unwrap();
        assert!(!mems.is_empty(), "post-stream memory cycle closed");
    }

    #[tokio::test]
    async fn reasoning_fallback_reply_is_truncated_in_memory() {
        // L7: when content is empty, reasoning_content is used — the user sees the full text
        // but raw CoT MUST NOT be written to memory (preventing context pollution
        // + prompt bloat on subsequent recalls). It is stored trimmed.
        struct ReasoningModel;
        #[async_trait::async_trait]
        impl Model for ReasoningModel {
            async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
                Ok(crate::model::Completion::reasoning_fallback(
                    "x".repeat(2000),
                ))
            }
        }
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(
            Persona::new("Aria", "role"),
            store,
            Arc::new(ReasoningModel),
        );
        let reply = agent.respond("question").await.unwrap();
        assert_eq!(reply.len(), 2000, "user sees full text");

        let mems = agent.recall(&Query::new("question")).await.unwrap();
        assert_eq!(mems.len(), 1);
        let crate::memory::MemoryKind::Episodic { body, .. } = &mems[0].item.kind else {
            panic!("expected episodic");
        };
        assert!(
            body.chars().count() <= 600,
            "CoT should be truncated before storing: {}",
            body.chars().count()
        );
    }

    #[tokio::test]
    async fn save_to_writes_atomically_without_tmp_leftover() {
        // M3: persona write must be atomic via tmp+rename — crash/SIGKILL
        // mid-write leaves no corrupt JSON, agent does not silently disappear on restart.
        let dir = std::env::temp_dir().join(format!("lore-atomic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("agent.json");

        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());
        agent.save_to(&path).unwrap();

        // No temporary file left, target file is valid and loadable.
        let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(entries.len(), 1, "no tmp file leftover: {entries:?}");
        let loaded = Agent::load_from(&path, store, Arc::new(MockModel::new())).unwrap();
        assert_eq!(loaded.persona.name, "Aria");

        // Overwriting also works with the same guarantee.
        agent.save_to(&path).unwrap();
        let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(entries.len(), 1, "no tmp leftover on second write either");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn respond_recalls_morphological_variants() {
        // H2: respond's memory access was keyword-only — a "learning" record
        // was filtered out by a "math" query. Flagship morphological capture
        // must also be evident in the agent's own reasoning loop.
        let store: Arc<dyn MemoryStore> = Arc::new(
            InMemoryStore::new().with_embedder(Arc::new(crate::memory::HashingEmbedder::new())),
        );
        let agent = agent_with("Aria", store);
        agent
            .experience("learning", "user is studying math")
            .await
            .unwrap();

        let reply = agent.respond("math").await.unwrap();
        assert!(
            reply.contains("recalling 1 memories") && reply.contains("learning"),
            "morphological variant should be recalled and enter prompt: {reply}"
        );
    }

    #[tokio::test]
    async fn recall_marks_returned_memories_as_accessed() {
        // Textual query: returned records are counted as "accessed" (decay signal is fed).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());
        agent
            .experience("stainless topic", "access should be marked")
            .await
            .unwrap();

        let hits = agent.recall(&Query::new("stainless")).await.unwrap();
        assert_eq!(hits.len(), 1);
        let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
        assert_eq!(mem.access_count, 1, "textual recall should mark access");

        // Second recall increments the counter.
        let _ = agent.recall(&Query::new("stainless")).await.unwrap();
        let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
        assert_eq!(mem.access_count, 2);
    }

    #[tokio::test]
    async fn browse_recall_does_not_touch_memories() {
        // Browse (textless) bulk scans — graph construction, board reading —
        // do not count as access; otherwise every full scan would completely kill decay.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());
        agent.experience("topic", "content").await.unwrap();

        let hits = agent.recall(&Query::new("")).await.unwrap();
        assert_eq!(hits.len(), 1);
        let mem = store.get(&hits[0].item.id).await.unwrap().unwrap();
        assert_eq!(mem.access_count, 0, "browse recall should not touch");
    }

    #[tokio::test]
    async fn freshly_recalled_low_value_memory_survives_decay() {
        // H1 regression: old + low-importance (automatic) record, if accessed via recall,
        // consolidation MUST NOT forget it. Pre-reinforcement behavior:
        // recall was not counting access → record was caught by the 90-day rule and forgotten.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store.clone());

        let mut m = Memory::episodic(
            crate::memory::Scope::World, // remember() pulls into own scope
            "old but recalled topic",
            "still works",
        )
        .with_importance(Memory::AUTO_IMPORTANCE);
        let old = chrono::Utc::now() - chrono::Duration::days(120);
        m.created_at = old;
        m.last_access = old;
        agent.remember(m).await.unwrap();

        // Record is genuinely found and used via textual query.
        let hits = agent.recall(&Query::new("recalled")).await.unwrap();
        assert_eq!(hits.len(), 1);

        // Consolidation runs: accessed record should survive.
        let report = store.consolidate().await.unwrap();
        assert_eq!(
            report.forgotten, 0,
            "accessed record should not be forgotten"
        );
        let still = agent.recall(&Query::new("recalled")).await.unwrap();
        assert_eq!(still.len(), 1, "record still accessible");
    }

    #[tokio::test]
    async fn two_agents_share_store_but_have_separate_memories() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let aria = agent_with("Aria", store.clone());
        let kai = agent_with("Kai", store.clone());

        aria.remember(Memory::semantic(
            Scope::World, // scope is set to own scope inside respond
            "Aria's personal note alpha",
            SemanticCat::Fact,
        ))
        .await
        .unwrap();

        // Kai should not see Aria's personal record.
        assert_eq!(kai.recall(&Query::new("alpha")).await.unwrap().len(), 0);
        // Aria should see her own record.
        assert_eq!(aria.recall(&Query::new("alpha")).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn empty_memory_agent_acknowledges() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store);
        let reply = agent.respond("hi").await.unwrap();
        assert!(reply.contains("memory empty"));
    }

    #[tokio::test]
    async fn act_uses_tool_when_routed() {
        use crate::tool::{CalcTool, KeywordRouter, ToolRegistry, ToolRouter};
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CalcTool::new()));
        let router: Arc<dyn ToolRouter> = Arc::new(KeywordRouter::new().on("calculate", "calc"));
        let agent = Agent::new(
            Persona::new("Aria", "role"),
            store,
            Arc::new(MockModel::new()),
        )
        .with_tools(reg, router);

        let out = agent.act("calculate 12 * 3").await.unwrap();
        assert_eq!(out, "36");
        // Did it remember the tool usage?
        let mem = agent.recall(&Query::new("calc")).await.unwrap();
        assert!(!mem.is_empty());
    }

    #[tokio::test]
    async fn act_falls_back_to_respond_without_tool() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = agent_with("Aria", store);
        let out = agent.act("hi").await.unwrap();
        assert!(out.contains("memory empty"));
    }

    #[tokio::test]
    async fn identity_survives_restart() {
        use crate::id::AgentId;
        use crate::memory::SqliteStore;

        let dir = std::env::temp_dir();
        let stamp = AgentId::new();
        let persona_path = dir.join(format!("lore-agent-{stamp}.json"));
        let db_path = dir.join(format!("lore-agent-{stamp}.db"));
        let persona_path = persona_path.to_str().unwrap().to_string();
        let db_path = db_path.to_str().unwrap().to_string();
        let model: Arc<dyn Model> = Arc::new(MockModel::new());

        // First life: save identity, experience a memory.
        let saved_id = {
            let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db_path).unwrap());
            let agent = Agent::new(
                Persona::new("Aria", "researcher").with_trait("curious"),
                store.clone(),
                model.clone(),
            );
            agent.save_to(&persona_path).unwrap();
            agent
                .experience("important event", "should be recalled after restart")
                .await
                .unwrap();
            agent.id.clone()
        };

        // Rebirth: persona file + same DB → both character and memories restored.
        {
            let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db_path).unwrap());
            let agent = Agent::load_from(&persona_path, store, model.clone()).unwrap();
            assert_eq!(agent.id, saved_id, "same AgentId");
            assert_eq!(agent.persona.name, "Aria");
            assert!(agent.persona.traits.contains(&"curious".to_string()));
            let mem = agent.recall(&Query::new("important")).await.unwrap();
            assert_eq!(mem.len(), 1, "same scope → memories restored");
        }

        let _ = std::fs::remove_file(&persona_path);
        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn recall_hyde_runs_and_searches() {
        use crate::memory::HashingEmbedder;
        let store: Arc<dyn MemoryStore> =
            Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));
        let agent = agent_with("Aria", store);
        agent
            .remember(Memory::semantic(
                Scope::World,
                "thoughts on math",
                SemanticCat::Preference,
            ))
            .await
            .unwrap();

        // HyDE: MockModel generates a hypothesis (includes the input), embed_text is computed from it.
        let res = agent.recall_hyde("math").await.unwrap();
        assert!(!res.is_empty(), "HyDE should return at least one record");
    }
}

#[cfg(test)]
mod reflect_tests {
    use super::*;
    use crate::memory::{InMemoryStore, Outcome, Tier};

    #[tokio::test]
    async fn reflect_distills_frequently_accessed_episodes() {
        // Frequently recalled memory is promoted to a permanent fact: model distills,
        // semantic record is opened, original episodic is archived (soft-delete).
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(crate::model::MockModel::new());
        let agent = Agent::new(Persona::new("Aria", "role"), store.clone(), model);
        agent
            .experience("rust conversation", "user said they started learning Rust")
            .await
            .unwrap();
        // Recalled twice (access signal).
        let hits = agent.recall(&Query::new("rust")).await.unwrap();
        let _ = agent.recall(&Query::new("rust")).await.unwrap();
        assert_eq!(hits.len(), 1);

        let n = agent.reflect().await.unwrap();
        assert_eq!(n, 1, "one memory should be distilled");

        // Semantic promotion: MockModel echoes context → distilled sentence is recorded.
        let sem = agent
            .recall(&Query::new("").tier(Tier::Semantic).limit(10))
            .await
            .unwrap();
        assert_eq!(sem.len(), 1, "promotion record in semantic tier");
        assert!(sem[0].item.importance >= 0.7, "stored with high importance");

        // Original episodic archived (invisible in browse, remains for audit).
        let epi = agent
            .recall(&Query::new("").tier(Tier::Episodic).limit(10))
            .await
            .unwrap();
        assert!(epi.is_empty(), "distilled memory dropped from live list");
        let archived = agent
            .recall(&Query::new("").tier(Tier::Episodic).limit(10).with_deleted())
            .await
            .unwrap();
        assert_eq!(archived.len(), 1, "soft-delete trace remains");
        assert!(archived[0].item.deleted_at.is_some());
    }

    #[tokio::test]
    async fn reflect_skips_cold_memories() {
        // Never-accessed (cold) memories are not distilled — noise is not promoted.
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(
            Persona::new("Aria", "role"),
            store,
            Arc::new(crate::model::MockModel::new()),
        );
        agent
            .experience("cold note", "one-time information")
            .await
            .unwrap();

        let n = agent.reflect().await.unwrap();
        assert_eq!(n, 0, "cold memory not distilled");
        let epi = agent
            .recall(&Query::new("").tier(Tier::Episodic).limit(10))
            .await
            .unwrap();
        assert_eq!(epi.len(), 1, "memory stays in place");
    }

    #[tokio::test]
    async fn reflect_tolerates_model_failure() {
        // Model error does not kill distillation: memory is preserved, 0 is returned.
        struct FailModel;
        #[async_trait::async_trait]
        impl Model for FailModel {
            async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
                Err(crate::error::LoreError::Model("closed".into()))
            }
        }
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(
            Persona::new("Aria", "role"),
            store.clone(),
            Arc::new(FailModel),
        );
        agent
            .experience("hot note", "accessed information")
            .await
            .unwrap();
        // Feed access count directly (since respond's model is broken).
        let hits = agent.recall(&Query::new("hot")).await.unwrap();
        store
            .reinforce(&hits[0].item.id, Outcome::Accessed)
            .await
            .unwrap();

        let n = agent.reflect().await.unwrap();
        assert_eq!(n, 0, "no distillation without model");
        let epi = agent
            .recall(&Query::new("").tier(Tier::Episodic).limit(10))
            .await
            .unwrap();
        assert_eq!(epi.len(), 1, "memory not lost");
    }
}

#[cfg(test)]
mod reflect_category_tests {
    use super::*;
    use crate::memory::{InMemoryStore, SemanticCat, Tier};

    #[tokio::test]
    async fn reflect_learns_preference_category() {
        // If the model distills with a "PREFERENCE" marker, the record is opened
        // with the Preference category (old behavior: everything was Fact). Unmarked text remains Fact.
        struct CatModel;
        #[async_trait::async_trait]
        impl Model for CatModel {
            async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
                Ok(crate::model::Completion::new(
                    "PREFERENCE\nUser drinks coffee plain",
                ))
            }
        }
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(
            Persona::new("Aria", "role"),
            store.clone(),
            Arc::new(CatModel),
        );
        agent
            .experience("coffee", "user did not add sugar to coffee")
            .await
            .unwrap();
        let hits = agent.recall(&Query::new("coffee")).await.unwrap();
        let _ = agent.recall(&Query::new("coffee")).await.unwrap();
        assert_eq!(hits.len(), 1);

        agent.reflect().await.unwrap();
        let sem = agent
            .recall(&Query::new("").tier(Tier::Semantic).limit(10))
            .await
            .unwrap();
        assert_eq!(sem.len(), 1);
        let crate::memory::MemoryKind::Semantic {
            category,
            statement,
            ..
        } = &sem[0].item.kind
        else {
            panic!("expected semantic");
        };
        assert_eq!(*category, SemanticCat::Preference, "category learned");
        assert!(
            !statement.contains("PREFERENCE"),
            "marker line extracted from sentence: {statement}"
        );
    }
}

#[cfg(test)]
mod think_resilience_tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::model::MockModel;

    /// Memory store that accepts reads but fails all writes (remember/reinforce).
    /// Simulates a persistent storage outage while retrieval still works.
    struct FailWriteStore {
        inner: InMemoryStore,
    }
    impl FailWriteStore {
        fn new() -> Self {
            Self {
                inner: InMemoryStore::new(),
            }
        }
    }
    #[async_trait::async_trait]
    impl MemoryStore for FailWriteStore {
        async fn remember(&self, _mem: Memory) -> Result<MemoryId> {
            Err(LoreError::Storage("write disabled".into()))
        }
        async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>> {
            self.inner.recall(scope, query).await
        }
        async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
            self.inner.get(id).await
        }
        async fn forget(&self, _id: &MemoryId) -> Result<()> {
            Err(LoreError::Storage("write disabled".into()))
        }
        async fn reinforce(&self, _id: &MemoryId, _outcome: Outcome) -> Result<()> {
            Err(LoreError::Storage("write disabled".into()))
        }
        async fn reinforce_many(&self, _ids: &[MemoryId], _outcome: Outcome) -> Result<()> {
            Err(LoreError::Storage("write disabled".into()))
        }
        async fn count(&self, scope: &Scope) -> Result<usize> {
            self.inner.count(scope).await
        }
        async fn consolidate(&self) -> Result<crate::memory::ConsolidationReport> {
            Err(LoreError::Storage("write disabled".into()))
        }
        async fn export(&self) -> Result<Vec<Memory>> {
            self.inner.export().await
        }
    }

    fn agent_with_fail_store(name: &str) -> Agent {
        let store: Arc<dyn MemoryStore> = Arc::new(FailWriteStore::new());
        let persona = Persona::new(name, "researcher").with_trait("curious");
        Agent::new(persona, store, Arc::new(MockModel::new()))
    }

    #[tokio::test]
    async fn respond_returns_answer_when_memory_write_fails() {
        // A storage failure in remember_exchange must not lose the
        // entire response: the answer is returned regardless — only a warn is logged.
        let agent = agent_with_fail_store("Aria");
        let reply = agent.respond("hello").await.unwrap();
        assert!(
            !reply.is_empty(),
            "answer must be returned even when memory fails"
        );
    }

    #[tokio::test]
    async fn respond_returns_answer_on_reasoning_fallback_when_memory_write_fails() {
        // Reasoning-fallback branch too: capped CoT memory write fails,
        // but the full text still reaches the user.
        struct ReasoningModel;
        #[async_trait::async_trait]
        impl Model for ReasoningModel {
            async fn complete(&self, _p: &Prompt) -> Result<crate::model::Completion> {
                Ok(crate::model::Completion::reasoning_fallback(
                    "full CoT text",
                ))
            }
        }
        let store: Arc<dyn MemoryStore> = Arc::new(FailWriteStore::new());
        let agent = Agent::new(
            Persona::new("Aria", "researcher").with_trait("curious"),
            store,
            Arc::new(ReasoningModel),
        );
        let reply = agent.respond("deep question").await.unwrap();
        assert_eq!(
            reply, "full CoT text",
            "user sees full text despite memory failure"
        );
    }
}

#[cfg(test)]
mod backward_compat_tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::model::MockModel;

    /// Old schema JSON (no model or extra fields) must load unchanged.
    #[test]
    fn old_agent_json_loads_without_model_or_extra() {
        let old_json = r#"{
  "id": "01HXYZOLDAGENT0",
  "persona": {
    "name": "OldBot",
    "role": "worker",
    "description": "",
    "traits": ["curious", "cautious"],
    "system_prompt": "",
    "version": 1
  }
}"#;
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(old_json, store, model).unwrap();
        assert_eq!(agent.persona.name, "OldBot");
        assert_eq!(agent.persona.role, "worker");
        assert!(
            agent.model_config().is_none(),
            "old schema should have no model config"
        );
        assert!(
            agent.persona.extra.is_empty(),
            "old schema should have no extra"
        );
    }

    /// New schema with model and extra fields roundtrips cleanly.
    #[test]
    fn new_agent_json_with_model_roundtrips() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let persona = Persona::new("NewBot", "backend engineer")
            .with_traits(["verification-minded"])
            .with_extra(["Run the project's tests before claiming done."]);
        let agent =
            Agent::new(persona, store, model).with_model_config(crate::model::ModelConfig {
                provider: crate::model::ProviderKind::Anthropic,
                model: "claude-sonnet-4-5-20250929".to_string(),
                auth: Some(crate::model::AuthKind::Subs),
                base_url: None,
            });

        let json = agent.to_json().unwrap();
        let back = Agent::from_json(
            &json,
            Arc::new(InMemoryStore::new()),
            Arc::new(MockModel::new()),
        )
        .unwrap();
        assert_eq!(back.persona.name, "NewBot");
        assert_eq!(back.persona.role, "backend engineer");
        assert!(back.model_config().is_some());
        assert_eq!(
            back.model_config().unwrap().provider,
            crate::model::ProviderKind::Anthropic
        );
        assert_eq!(
            back.model_config().unwrap().model,
            "claude-sonnet-4-5-20250929"
        );
        assert!(!back.persona.extra.is_empty());
        assert!(back.persona.extra[0].contains("Run the project's tests"));
    }

    /// Agent JSON with extra field but no model also roundtrips.
    #[test]
    fn agent_json_with_extra_no_model_roundtrips() {
        let old_json = r#"{
  "id": "01HXYZAGENTEXTRA0",
  "persona": {
    "name": "ExtraBot",
    "role": "helper",
    "description": "",
    "traits": [],
    "system_prompt": "",
    "extra": ["custom identity line"],
    "version": 1
  },
  "extra": ["custom identity line"]
}"#;
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(old_json, store, model).unwrap();
        assert_eq!(agent.persona.name, "ExtraBot");
        assert!(agent
            .persona
            .extra
            .contains(&"custom identity line".to_string()));
        assert!(agent.model_config().is_none());
    }

    /// identity_prompt includes extra lines from role presets.
    #[test]
    fn identity_extra_appears_in_identity_prompt() {
        let r = crate::agent::roles::preset("backend").unwrap();
        let persona = Persona::new("Dev", r.role)
            .with_traits(r.traits.iter().map(|s| s.to_string()))
            .with_extra([r.identity_extra.to_string()]);
        let ip = persona.identity_prompt();
        assert!(
            ip.contains("Run the project's tests before claiming done"),
            "identity_extra should appear in prompt: {ip}"
        );
    }

    /// save_to/load_from preserves model_config and extra.
    #[tokio::test]
    async fn save_load_preserves_model_config_and_extra() {
        let dir = std::env::temp_dir().join(format!("lore-agent-persist-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("persist_agent.json");

        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let persona = Persona::new("PersistentBot", "reviewer")
            .with_extra(["Read code critically: look for logic gaps."]);
        let agent =
            Agent::new(persona, store, model).with_model_config(crate::model::ModelConfig {
                provider: crate::model::ProviderKind::Mock,
                model: "mock".to_string(),
                auth: None,
                base_url: None,
            });
        agent.save_to(&path).unwrap();

        let loaded = Agent::load_from(
            &path,
            Arc::new(InMemoryStore::new()),
            Arc::new(MockModel::new()),
        )
        .unwrap();
        assert_eq!(loaded.persona.name, "PersistentBot");
        assert!(loaded.model_config().is_some());
        assert!(!loaded.persona.extra.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}
