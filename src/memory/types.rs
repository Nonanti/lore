//! Memory data model: tiers, scope, query and score types.

use crate::id::{AgentId, MemoryId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Which agent (or the shared world) a memory belongs to.
///
/// Retrieval-level isolation derives from this: an agent sees only its own
/// `Agent(id)` records plus shared `World` records.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// Personal memory belonging to a specific agent.
    Agent(AgentId),
    /// Shared memory accessible to all agents (blackboard substrate).
    World,
}

/// Semantic (factual) memory category. Mirrors Alaz's `core_memory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticCat {
    /// Objective fact.
    Fact,
    /// Preference.
    Preference,
    /// Convention / rule.
    Convention,
    /// Constraint.
    Constraint,
}

impl std::fmt::Display for SemanticCat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SemanticCat::Fact => "fact",
            SemanticCat::Preference => "preference",
            SemanticCat::Convention => "convention",
            SemanticCat::Constraint => "constraint",
        })
    }
}

/// 5W cues for episodic memory: who/what/where/when/why.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FiveW {
    /// Who.
    #[serde(default)]
    pub who: Vec<String>,
    /// What.
    #[serde(default)]
    pub what: Vec<String>,
    /// Where.
    #[serde(default, rename = "where")]
    pub where_: Vec<String>,
    /// When.
    #[serde(default)]
    pub when: Vec<String>,
    /// Why.
    #[serde(default)]
    pub why: Vec<String>,
}

impl FiveW {
    /// Flattens all cues into a single searchable string.
    pub fn flatten(&self) -> String {
        [&self.who, &self.what, &self.where_, &self.when, &self.why]
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Content of the three memory tiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Experienced event (timestamped, with 5W cues).
    Episodic {
        /// Short title.
        title: String,
        /// Detail.
        body: String,
        /// 5W cues.
        cues: FiveW,
    },
    /// General knowledge / fact.
    Semantic {
        /// Optional key (e.g. "preferred_language").
        key: Option<String>,
        /// Statement.
        statement: String,
        /// Category.
        category: SemanticCat,
    },
    /// Learned skill / procedure (with success tracking).
    Procedural {
        /// Title.
        title: String,
        /// Steps.
        steps: Vec<String>,
        /// Success count (for Wilson score).
        successes: u32,
        /// Failure count (for Wilson score).
        failures: u32,
    },
}

/// Lightweight tag for tier filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    /// Episodic.
    Episodic,
    /// Semantic.
    Semantic,
    /// Procedural.
    Procedural,
}

/// A memory record — common envelope wrapping all tiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier.
    pub id: MemoryId,
    /// Ownership / visibility.
    pub scope: Scope,
    /// Content (tier).
    pub kind: MemoryKind,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last access time (recency/decay signal).
    pub last_access: DateTime<Utc>,
    /// Access count (frequency signal).
    pub access_count: u32,
    /// Importance (0..1, forgetting signal).
    pub importance: f32,
    /// Soft-delete time (Some means "forgotten").
    pub deleted_at: Option<DateTime<Utc>>,
    /// Embedding vector (Phase 2+; currently None).
    pub embedding: Option<Vec<f32>>,
}

impl Memory {
    fn base(scope: Scope, kind: MemoryKind) -> Self {
        let now = Utc::now();
        Self {
            id: MemoryId::new(),
            scope,
            kind,
            created_at: now,
            last_access: now,
            access_count: 0,
            importance: 0.5,
            deleted_at: None,
            embedding: None,
        }
    }

    /// Creates an episodic record.
    /// Birth importance for system-generated (automatic) records: exchange logs,
    /// board notes, tool usage traces. Must be BELOW
    /// [`ForgetPolicy::min_importance`](0.25) so that decay can reclaim these
    /// high-volume records over time — otherwise memory grows without bound
    /// (user-explicit `remember`/`experience` records are protected at the
    /// default 0.5).
    ///
    /// [`ForgetPolicy::min_importance`]: super::evolution::ForgetPolicy
    pub const AUTO_IMPORTANCE: f32 = 0.2;

    pub fn episodic(scope: Scope, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self::base(
            scope,
            MemoryKind::Episodic {
                title: title.into(),
                body: body.into(),
                cues: FiveW::default(),
            },
        )
    }

    /// Creates a semantic record.
    pub fn semantic(scope: Scope, statement: impl Into<String>, category: SemanticCat) -> Self {
        Self::base(
            scope,
            MemoryKind::Semantic {
                key: None,
                statement: statement.into(),
                category,
            },
        )
    }

    /// Creates a procedural record.
    pub fn procedural(scope: Scope, title: impl Into<String>, steps: Vec<String>) -> Self {
        Self::base(
            scope,
            MemoryKind::Procedural {
                title: title.into(),
                steps,
                successes: 0,
                failures: 0,
            },
        )
    }

    /// Sets the importance value (builder).
    pub fn with_importance(mut self, v: f32) -> Self {
        self.importance = v.clamp(0.0, 1.0);
        self
    }

    /// Adds a key to a semantic record (builder).
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        if let MemoryKind::Semantic { key: k, .. } = &mut self.kind {
            *k = Some(key.into());
        }
        self
    }

    /// Adds 5W cues to an episodic record (builder).
    pub fn with_cues(mut self, cues: FiveW) -> Self {
        if let MemoryKind::Episodic { cues: c, .. } = &mut self.kind {
            *c = cues;
        }
        self
    }

    /// This record's tier.
    pub fn tier(&self) -> Tier {
        match &self.kind {
            MemoryKind::Episodic { .. } => Tier::Episodic,
            MemoryKind::Semantic { .. } => Tier::Semantic,
            MemoryKind::Procedural { .. } => Tier::Procedural,
        }
    }

    /// Plain-text representation for search (input to keyword scoring).
    pub fn searchable_text(&self) -> String {
        match &self.kind {
            MemoryKind::Episodic { title, body, cues } => {
                format!("{} {} {}", title, body, cues.flatten())
            }
            MemoryKind::Semantic { key, statement, .. } => match key {
                Some(k) => format!("{} {}", k, statement),
                None => statement.clone(),
            },
            MemoryKind::Procedural { title, steps, .. } => {
                format!("{} {}", title, steps.join(" "))
            }
        }
    }

    /// Human-readable short summary (for demo/logging).
    pub fn summary(&self) -> String {
        match &self.kind {
            MemoryKind::Episodic { title, .. } => format!("[episodic] {title}"),
            MemoryKind::Semantic {
                statement,
                category,
                ..
            } => format!("[semantic/{category:?}] {statement}"),
            MemoryKind::Procedural {
                title,
                successes,
                failures,
                ..
            } => format!("[procedural] {title} ({successes}✓/{failures}✗)"),
        }
    }

    /// Full content for injecting into a model prompt. Unlike [`summary`](Self::summary)
    /// (a compact one-liner for listings), this includes the episodic **body**,
    /// the semantic statement, or the procedural steps — so recalled knowledge
    /// actually reaches the model.
    pub fn recall_context(&self) -> String {
        match &self.kind {
            MemoryKind::Episodic { title, body, .. } => {
                let t = title.trim();
                let b = body.trim();
                if b.is_empty() {
                    t.to_string()
                } else if t.is_empty() {
                    b.to_string()
                } else if t.ends_with(':') {
                    // Exchange traces already end with a colon ("responded to 'x': ").
                    format!("{t} {b}")
                } else {
                    format!("{t}: {b}")
                }
            }
            MemoryKind::Semantic { statement, .. } => statement.clone(),
            MemoryKind::Procedural { title, steps, .. } => {
                if steps.is_empty() {
                    title.clone()
                } else {
                    format!("{title}: {}", steps.join(" → "))
                }
            }
        }
    }
}

/// A memory query.
#[derive(Clone, Debug)]
pub struct Query {
    /// Free-text query (empty triggers "browse" mode: recency+importance ranking).
    pub text: String,
    /// Tier filter (None = all).
    pub tiers: Option<Vec<Tier>>,
    /// Maximum number of results.
    pub limit: usize,
    /// Whether to include soft-deleted records.
    pub include_deleted: bool,
    /// Semantic recall: retrieve via cosine even without keyword match.
    pub semantic: bool,
    /// Diversity: diversify results with MMR (suppresses similar records).
    pub diverse: bool,
    /// Vector leg override: if provided, embedding is computed from this text
    /// (keywords still come from `text`). HyDE injects a hypothetical answer here.
    pub embed_text: Option<String>,
    /// Re-rank first-pass candidates with the native reranker.
    pub rerank: bool,
    /// Minimum importance a record must have to be returned (None = no floor).
    /// Used to keep low-value automatic traces (exchange/tool/board logs, born
    /// at [`Memory::AUTO_IMPORTANCE`]) out of prompt context.
    pub min_importance: Option<f32>,
}

impl Query {
    /// New query from text (limit=10).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tiers: None,
            limit: 10,
            include_deleted: false,
            semantic: false,
            diverse: false,
            min_importance: None,
            embed_text: None,
            rerank: false,
        }
    }

    /// Filters by a single tier (builder).
    pub fn tier(mut self, t: Tier) -> Self {
        self.tiers.get_or_insert_with(Vec::new).push(t);
        self
    }

    /// Sets the limit (builder).
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Includes soft-deleted records (builder).
    pub fn with_deleted(mut self) -> Self {
        self.include_deleted = true;
        self
    }

    /// Enables semantic recall (builder): matches via embedding cosine without keywords.
    pub fn semantic(mut self) -> Self {
        self.semantic = true;
        self
    }

    /// Sets a minimum-importance floor (builder): records below `v` are excluded.
    pub fn min_importance(mut self, v: f32) -> Self {
        self.min_importance = Some(v);
        self
    }

    /// Enables MMR diversification (builder).
    pub fn diverse(mut self) -> Self {
        self.diverse = true;
        self
    }

    /// Overrides the embedding text for the vector leg (builder; HyDE).
    pub fn embed_text(mut self, text: impl Into<String>) -> Self {
        self.embed_text = Some(text.into());
        self
    }

    /// Enables native reranking (builder).
    pub fn rerank(mut self) -> Self {
        self.rerank = true;
        self
    }
}

impl Default for Query {
    fn default() -> Self {
        Self::new("")
    }
}

/// A scored result (score + which signals contributed — explainability).
#[derive(Clone, Debug)]
pub struct Scored<T> {
    /// The item.
    pub item: T,
    /// Final score.
    pub score: f32,
    /// Contributing signals.
    pub signals: Vec<Signal>,
}

/// A single signal contributing to the score.
#[derive(Clone, Debug)]
pub struct Signal {
    /// Signal name (e.g. "keyword", "recency").
    pub name: String,
    /// Value.
    pub value: f32,
}

/// Outcome reported in a `reinforce` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Record was accessed (access_count++).
    Accessed,
    /// Procedural success (successes++).
    Success,
    /// Procedural failure (failures++).
    Failure,
}

/// Report from a `consolidate` call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Number of records scanned.
    pub scanned: usize,
    /// Number of records merged.
    pub merged: usize,
    /// Number of records forgotten.
    pub forgotten: usize,
}

#[cfg(test)]
mod recall_context_tests {
    use super::*;

    #[test]
    fn episodic_recall_context_includes_body() {
        let m = Memory::episodic(Scope::World, "favori dil", "Rust, ownership için");
        assert_eq!(m.recall_context(), "favori dil: Rust, ownership için");
        // summary stays compact (title only).
        assert_eq!(m.summary(), "[episodic] favori dil");
    }

    #[test]
    fn exchange_trace_context_reads_naturally() {
        // Exchange traces have a title ending in ':'.
        let m = Memory::episodic(Scope::World, "responded to 'hi':", "hello there");
        assert_eq!(m.recall_context(), "responded to 'hi': hello there");
    }

    #[test]
    fn semantic_and_procedural_recall_context() {
        let s = Memory::semantic(Scope::World, "Rust is memory-safe", SemanticCat::Fact);
        assert_eq!(s.recall_context(), "Rust is memory-safe");
        let p = Memory::procedural(
            Scope::World,
            "solve math",
            vec!["parse".into(), "compute".into()],
        );
        assert_eq!(p.recall_context(), "solve math: parse → compute");
    }

    #[test]
    fn empty_body_falls_back_to_title() {
        let m = Memory::episodic(Scope::World, "just a title", "");
        assert_eq!(m.recall_context(), "just a title");
    }
}
