//! Agent: identity (persona) + personal memory + model.
//!
//! Two things that make an agent "that agent" converge here: **identity** (persona)
//! and **memory** (personal scope). The agent is linked to the memory engine and model
//! via `Arc` handles — both are behind traits, swappable.

mod conversation;
pub mod distill;
mod persona;
pub mod roles;
mod solve;
pub mod work;

pub use conversation::{Conversation, DEFAULT_CONVERSATION_CAP};
pub use persona::Persona;
pub use roles::{preset, presets, Role};
pub use work::{WorkReport, WorkSpec};

use crate::error::{LoreError, Result};
use crate::id::{AgentId, MemoryId};
use crate::memory::{Memory, MemoryStore, Outcome, Query, Scope, Scored, SemanticCat, Tier};
use crate::model::{Model, Prompt, TokenStream, ToolMode, Turn};
use crate::tool::{ToolContext, ToolRegistry, ToolRouter};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::AtomicBool;
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
    /// Whether distilled conventions/constraints are shared to team memory
    /// (`Scope::World`). None (absent) = true. Set to Some(false) to keep
    /// every lesson personal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    share: Option<bool>,
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
    /// Whether distilled conventions/constraints are shared to team memory
    /// (`Scope::World`). None = true (default). Some(false) = personal only.
    share: Option<bool>,
    /// Optional tool context (registry + router).
    tools: Option<Arc<ToolContext>>,
    /// Explicit tool-mode override (builder). Precedence:
    /// builder > model_config.tool_mode > `LORE_TOOL_MODE` env > `Auto`.
    tool_mode: Option<ToolMode>,
    /// `auto` downgrade latch: once the endpoint proves it cannot do native
    /// tools, later solves skip the doomed probe. Arc — shared by clones.
    /// `Relaxed` ordering everywhere: a standalone set-once flag guarding
    /// no dependent memory — stronger orderings would imply synchronization
    /// that does not exist.
    native_downgraded: Arc<AtomicBool>,
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
            share: None,   // None = true (default)
            tools: None,
            tool_mode: None,
            native_downgraded: Arc::new(AtomicBool::new(false)),
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
            share: None,
            tools: None,
            tool_mode: None,
            native_downgraded: Arc::new(AtomicBool::new(false)),
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

    /// Sets the tool-call protocol explicitly (builder pattern). Overrides
    /// model-config and env selection.
    pub fn with_tool_mode(mut self, mode: ToolMode) -> Self {
        self.tool_mode = Some(mode);
        self
    }

    /// Effective tool mode: builder override > per-agent model config
    /// (non-auto) > `LORE_TOOL_MODE` env > `Auto`. The env value is read
    /// once per process (review #8) — solve is a hot path, and a mid-run
    /// env flip silently changing agent behavior would be a misfeature.
    fn effective_tool_mode(&self) -> ToolMode {
        if let Some(m) = self.tool_mode {
            return m;
        }
        if let Some(cfg) = &self.model_config {
            if !cfg.tool_mode.is_auto() {
                return cfg.tool_mode;
            }
        }
        static ENV_MODE: std::sync::OnceLock<ToolMode> = std::sync::OnceLock::new();
        *ENV_MODE.get_or_init(ToolMode::from_env)
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

    /// Sets team-sharing opt-out (builder pattern). `false` keeps distilled
    /// conventions/constraints in personal scope.
    pub fn with_share(mut self, v: bool) -> Self {
        self.share = Some(v);
        self
    }

    /// Whether distilled conventions/constraints go to team memory
    /// (`Scope::World`). None/Some(true) → share; Some(false) → personal only.
    pub fn should_share(&self) -> bool {
        self.share.unwrap_or(true)
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
            share: self.share,
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
        // Validate persona: agents loaded from disk must also pass sanitization.
        // An agent file written before sanitization (or edited on disk with control
        // chars) would bypass the HTTP-level validate() gate. This closes the
        // bypass — loaded agents are rejected just like newly created ones.
        let bad = persona.validate();
        if !bad.is_empty() {
            return Err(LoreError::InvalidInput(format!(
                "persona loaded from disk has invalid fields: {}",
                bad.join(", ")
            )));
        }
        Ok(Self {
            id: rec.id,
            persona,
            memory,
            model,
            model_config: rec.model,
            distill: rec.distill,
            share: rec.share,
            tools: None,
            tool_mode: None,
            native_downgraded: Arc::new(AtomicBool::new(false)),
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
    pub async fn remember(&self, mut mem: Memory) -> Result<MemoryId> {
        mem.scope = self.scope();
        self.memory.remember(mem).await
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
            if let Err(e) = self.remember(mem).await {
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
                    // Graph leg: entity-bridged neighbors join the context —
                    // multi-hop recall ("X's owner's job") measured on the
                    // golden set: MultiHop 2/4 → 4/4, zero regression.
                    // min_importance below still floors pulled neighbors.
                    .graph()
                    // Rerank: native lexical pass is measured-neutral and
                    // cheap; a store-attached neural cross-encoder
                    // (LORE_RERANKER=neural) upgrades precision in place.
                    .rerank()
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
mod tests;

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
        assert!(
            agent.should_distill() && agent.should_share(),
            "absent distill/share fields default to enabled"
        );
    }

    #[test]
    fn share_flag_round_trips_through_record() {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let agent = Agent::new(
            Persona::new("NoShareBot", "worker"),
            store.clone(),
            Arc::new(MockModel::new()),
        )
        .with_share(false);
        assert!(!agent.should_share());
        let json = agent.to_json().unwrap();
        let back = Agent::from_json(&json, store, Arc::new(MockModel::new())).unwrap();
        assert!(!back.should_share(), "share opt-out survives persistence");
        assert!(back.should_distill(), "distill untouched");
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
                tool_mode: Default::default(),
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
                tool_mode: Default::default(),
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

    /// Opt-out default (missing field) = distill ON. When AgentRecord JSON
    /// lacks the `distill` field, `should_distill()` must return true.
    #[test]
    fn agent_record_missing_distill_field_defaults_to_true() {
        // JSON without `distill` field (None → unwrap_or(true) = true).
        let json_without_distill = r#"{
  "id": "01HXYZAGENTNODISTILL0",
  "persona": {
    "name": "DefaultBot",
    "role": "worker",
    "description": "",
    "traits": [],
    "system_prompt": "",
    "extra": [],
    "version": 1
  }
}"#;
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(json_without_distill, store, model).unwrap();
        assert!(
            agent.should_distill(),
            "missing distill field → defaults to true (ON)"
        );
    }

    /// Explicit `distill: false` opts out.
    #[test]
    fn agent_record_explicit_false_distill_opts_out() {
        let json_with_false = r#"{
  "id": "01HXYZAGENTNODISTILL0",
  "persona": {
    "name": "OptOutBot",
    "role": "worker",
    "description": "",
    "traits": [],
    "system_prompt": "",
    "extra": [],
    "version": 1
  },
  "distill": false
}"#;
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(json_with_false, store, model).unwrap();
        assert!(!agent.should_distill(), "distill=false → opts out");
    }

    /// Explicit `distill: true` stays ON.
    #[test]
    fn agent_record_explicit_true_distill_stays_on() {
        let json_with_true = r#"{
  "id": "01HXYZAGENTNODISTILL0",
  "persona": {
    "name": "OnBot",
    "role": "worker",
    "description": "",
    "traits": [],
    "system_prompt": "",
    "extra": [],
    "version": 1
  },
  "distill": true
}"#;
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(json_with_true, store, model).unwrap();
        assert!(agent.should_distill(), "distill=true → ON");
    }

    /// M-1: from_json must reject personas with control chars / newlines,
    /// closing the load-from-disk bypass that the reviewer flagged.
    #[test]
    fn from_json_rejects_persona_with_control_chars() {
        let json_with_newline_name = serde_json::to_string_pretty(&serde_json::json!({
            "id": "01ARYZ6S19Q2VTMRZ",
            "persona": {
                "name": "Aria\nEvil",
                "role": "researcher",
                "description": "",
                "traits": [],
                "system_prompt": "",
                "extra": [],
                "version": 1
            }
        }))
        .unwrap();
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let result = Agent::from_json(&json_with_newline_name, store, model);
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("newline in name must be rejected on load"),
        };
        assert!(
            err.contains("invalid fields") && err.contains("name"),
            "error should mention name: {err}"
        );
    }

    #[test]
    fn from_json_rejects_persona_with_esc_in_role() {
        let json_with_esc_role = serde_json::to_string_pretty(&serde_json::json!({
            "id": "01ARYZ6S19Q2VTMRZ",
            "persona": {
                "name": "Aria",
                "role": "researcher\u{001B}hidden",
                "description": "",
                "traits": [],
                "system_prompt": "",
                "extra": [],
                "version": 1
            }
        }))
        .unwrap();
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let result = Agent::from_json(&json_with_esc_role, store, model);
        assert!(result.is_err(), "ESC in role must be rejected on load");
    }

    #[test]
    fn from_json_accepts_clean_persona() {
        let json_clean = serde_json::to_string_pretty(&serde_json::json!({
            "id": "01ARYZ6S19Q2VTMRZ",
            "persona": {
                "name": "Aria",
                "role": "researcher",
                "description": "",
                "traits": ["curious"],
                "system_prompt": "",
                "extra": [],
                "version": 1
            }
        }))
        .unwrap();
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let model = Arc::new(MockModel::new());
        let agent = Agent::from_json(&json_clean, store, model).unwrap();
        assert_eq!(agent.persona.name, "Aria");
    }
}
