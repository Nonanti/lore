//! `InMemoryStore`: zero-dependency, thread-safe memory engine.
//!
//! Phase 1 implementation. Can take/restore JSON snapshots for persistence.
//! Retrieval uses `retrieval::score` (keyword + recency + importance + Wilson).

use super::embed::Embedder;
use super::retrieval;
use super::types::{ConsolidationReport, Memory, Outcome, Query, Scope, Scored};
use super::MemoryStore;
use crate::error::{LoreError, Result};
use crate::id::MemoryId;
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// In-memory memory store (identity → record).
#[derive(Default)]
pub struct InMemoryStore {
    inner: RwLock<HashMap<String, Memory>>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl InMemoryStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches an embedder (builder): embeds on remember, enables hybrid on recall.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Finds whether a new record conflicts with existing ones (same scope).
    /// Write-time conflict detection: band is embedder-specific (`conflict_band`).
    pub async fn conflicts(&self, mem: &Memory) -> Vec<MemoryId> {
        let band = self
            .embedder
            .as_ref()
            .map(|e| e.conflict_band())
            .unwrap_or(retrieval::CONFLICT_BAND);
        let emb = match &mem.embedding {
            Some(e) => e.clone(),
            None => match &self.embedder {
                Some(em) => em.embed(&mem.searchable_text()),
                None => return Vec::new(),
            },
        };
        let guard = self.inner.read().await;
        let mut out = Vec::new();
        for other in guard.values() {
            if other.id == mem.id || other.deleted_at.is_some() || other.scope != mem.scope {
                continue;
            }
            if let Some(oe) = &other.embedding {
                if retrieval::is_conflict_in(&emb, oe, band) {
                    out.push(other.id.clone());
                }
            }
        }
        out
    }

    /// Takes a JSON snapshot of current records.
    pub async fn snapshot(&self) -> Vec<Memory> {
        self.inner.read().await.values().cloned().collect()
    }

    /// Serializes the JSON snapshot.
    pub async fn to_json(&self) -> Result<String> {
        let items = self.snapshot().await;
        Ok(serde_json::to_string_pretty(&items)?)
    }

    /// Creates a store from a list of records.
    pub fn from_memories(memories: Vec<Memory>) -> Self {
        let map = memories
            .into_iter()
            .map(|m| (m.id.to_string(), m))
            .collect();
        Self {
            inner: RwLock::new(map),
            embedder: None,
        }
    }

    /// Loads a store from a JSON snapshot.
    pub fn from_json(json: &str) -> Result<Self> {
        let items: Vec<Memory> = serde_json::from_str(json)?;
        Ok(Self::from_memories(items))
    }

    /// Total record count in the store (including soft-deleted).
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Whether the store is empty.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

/// Determines whether a query scope can see a record's scope.
/// Agent(a) → own records + World; World → only World.
fn scope_visible(query_scope: &Scope, mem_scope: &Scope) -> bool {
    match (query_scope, mem_scope) {
        (_, Scope::World) => true,
        (Scope::Agent(a), Scope::Agent(b)) => a == b,
        (Scope::World, Scope::Agent(_)) => false,
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        Ok(self.inner.read().await.get(&id.to_string()).cloned())
    }

    async fn remember(&self, mut mem: Memory) -> Result<MemoryId> {
        if let Some(e) = &self.embedder {
            if mem.embedding.is_none() {
                mem.embedding = Some(e.embed(&mem.searchable_text()));
            }
        }
        let id = mem.id.clone();
        self.inner.write().await.insert(id.to_string(), mem);
        Ok(id)
    }

    async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>> {
        let now = Utc::now();
        let has_text = !query.text.trim().is_empty();
        let embed_src = query.embed_text.as_deref().unwrap_or(&query.text);
        let q_emb = match (&self.embedder, has_text) {
            (Some(e), true) => Some(e.embed(embed_src)),
            _ => None,
        };
        let guard = self.inner.read().await;

        // Single Scorer for the scan: token embedding cache is shared across candidates.
        let mut scorer = retrieval::Scorer::new(self.embedder.as_deref());
        let mut hits: Vec<(f32, Vec<crate::memory::types::Signal>, &Memory)> = Vec::new();
        for mem in guard.values() {
            if !scope_visible(scope, &mem.scope) {
                continue;
            }
            if mem.deleted_at.is_some() && !query.include_deleted {
                continue;
            }
            if let Some(tiers) = &query.tiers {
                if !tiers.contains(&mem.tier()) {
                    continue;
                }
            }
            if let Some(min_imp) = query.min_importance {
                if mem.importance < min_imp {
                    continue;
                }
            }

            let (score, signals) = scorer.score(mem, query, q_emb.as_deref(), now);
            // With-text query: filter out irrelevant hits (score 0).
            if has_text && score <= 0.0 {
                continue;
            }
            hits.push((score, signals, mem));
        }

        // In the plain path (no rerank/diverse), clone AFTER truncation, not before:
        // a 10k-match limit-10 query clones 10 `Memory` instead of 10k.
        // Rerank/MMR needs the full candidate list — no early truncation there.
        if !query.rerank && !query.diverse {
            hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            hits.truncate(query.limit);
        }
        let scored = hits
            .into_iter()
            .map(|(score, signals, mem)| Scored {
                item: mem.clone(),
                score,
                signals,
            })
            .collect();

        Ok(retrieval::finalize(scored, query))
    }

    async fn reinforce(&self, id: &MemoryId, outcome: Outcome) -> Result<()> {
        use super::types::MemoryKind;
        let mut guard = self.inner.write().await;
        let mem = guard
            .get_mut(&id.to_string())
            .ok_or_else(|| LoreError::NotFound(id.to_string()))?;

        mem.last_access = Utc::now();
        mem.access_count = mem.access_count.saturating_add(1);
        match outcome {
            Outcome::Accessed => {}
            Outcome::Success => {
                if let MemoryKind::Procedural { successes, .. } = &mut mem.kind {
                    *successes = successes.saturating_add(1);
                }
            }
            Outcome::Failure => {
                if let MemoryKind::Procedural { failures, .. } = &mut mem.kind {
                    *failures = failures.saturating_add(1);
                }
            }
        }
        Ok(())
    }

    async fn forget(&self, id: &MemoryId) -> Result<()> {
        let mut guard = self.inner.write().await;
        let mem = guard
            .get_mut(&id.to_string())
            .ok_or_else(|| LoreError::NotFound(id.to_string()))?;
        mem.deleted_at = Some(Utc::now()); // soft-delete
        Ok(())
    }

    async fn count(&self, scope: &Scope) -> Result<usize> {
        Ok(self
            .inner
            .read()
            .await
            .values()
            .filter(|m| scope_visible(scope, &m.scope) && m.deleted_at.is_none())
            .count())
    }

    async fn export(&self) -> Result<Vec<Memory>> {
        Ok(self
            .inner
            .read()
            .await
            .values()
            .filter(|m| m.deleted_at.is_none())
            .cloned()
            .collect())
    }

    async fn consolidate(&self) -> Result<ConsolidationReport> {
        let now = Utc::now();
        let policy = super::evolution::ForgetPolicy::default();
        let mut guard = self.inner.write().await;
        let live: Vec<Memory> = guard
            .values()
            .filter(|m| m.deleted_at.is_none())
            .cloned()
            .collect();
        let scanned = live.len();
        let p = super::evolution::plan(&live, &policy, now);
        for id in &p.to_forget {
            if let Some(m) = guard.get_mut(&id.to_string()) {
                m.deleted_at = Some(now);
            }
        }
        Ok(ConsolidationReport {
            scanned,
            merged: p.merged,
            forgotten: p.forgotten,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::AgentId;
    use crate::memory::types::{Memory, SemanticCat, Tier};

    fn agent_scope() -> (AgentId, Scope) {
        let a = AgentId::new();
        let s = Scope::Agent(a.clone());
        (a, s)
    }

    #[tokio::test]
    async fn remember_then_recall_ranks_relevant_first() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        store
            .remember(Memory::semantic(
                scope.clone(),
                "User likes Rust and math",
                SemanticCat::Preference,
            ))
            .await
            .unwrap();
        store
            .remember(Memory::episodic(scope.clone(), "Dinner", "Pizza yendi"))
            .await
            .unwrap();

        let res = store
            .recall(&scope, &Query::new("rust math"))
            .await
            .unwrap();

        assert_eq!(res.len(), 1, "only relevant record should be returned");
        assert!(res[0].item.summary().contains("Rust"));
        assert!(res[0].score > 0.0);
    }

    #[tokio::test]
    async fn scope_isolation_blocks_other_agents_but_shares_world() {
        let store = InMemoryStore::new();
        let (_a, scope_a) = agent_scope();
        let (_b, scope_b) = agent_scope();

        store
            .remember(Memory::semantic(
                scope_a.clone(),
                "secret agent alpha note",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store
            .remember(Memory::semantic(
                Scope::World,
                "public world alpha note",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        // Agent B should not see A's personal record but should see World.
        let res_b = store.recall(&scope_b, &Query::new("alpha")).await.unwrap();
        assert_eq!(res_b.len(), 1);
        assert!(res_b[0].item.summary().contains("world"));

        // Agent A should see both its own record and World.
        let res_a = store.recall(&scope_a, &Query::new("alpha")).await.unwrap();
        assert_eq!(res_a.len(), 2);
    }

    #[tokio::test]
    async fn get_fetches_by_id_and_reinforce_many_batches() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();
        let a = store
            .remember(Memory::semantic(
                scope.clone(),
                "alfa notu",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        let b = store
            .remember(Memory::semantic(
                scope.clone(),
                "beta notu",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        assert!(store.get(&a).await.unwrap().is_some());
        assert!(store.get(&MemoryId::new()).await.unwrap().is_none());

        // Batch: multiple identities in one call; missing identities are skipped.
        store
            .reinforce_many(&[a.clone(), b.clone(), MemoryId::new()], Outcome::Accessed)
            .await
            .unwrap();
        assert_eq!(store.get(&a).await.unwrap().unwrap().access_count, 1);
        assert_eq!(store.get(&b).await.unwrap().unwrap().access_count, 1);
    }

    #[tokio::test]
    async fn reinforce_success_boosts_procedural() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        let id = store
            .remember(Memory::procedural(
                scope.clone(),
                "run cargo test",
                vec!["cargo build".into(), "cargo test".into()],
            ))
            .await
            .unwrap();

        store.reinforce(&id, Outcome::Success).await.unwrap();
        store.reinforce(&id, Outcome::Success).await.unwrap();
        store.reinforce(&id, Outcome::Failure).await.unwrap();

        let res = store
            .recall(&scope, &Query::new("cargo test").tier(Tier::Procedural))
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert!(res[0].item.summary().contains("2✓/1✗"));
    }

    #[tokio::test]
    async fn forget_soft_deletes_and_hidden_by_default() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        let id = store
            .remember(Memory::semantic(
                scope.clone(),
                "secret to forget",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        assert_eq!(
            store
                .recall(&scope, &Query::new("secret"))
                .await
                .unwrap()
                .len(),
            1
        );

        store.forget(&id).await.unwrap();

        // Default: hidden.
        assert_eq!(
            store
                .recall(&scope, &Query::new("secret"))
                .await
                .unwrap()
                .len(),
            0
        );
        // Visible with include_deleted (auditability).
        assert_eq!(
            store
                .recall(&scope, &Query::new("secret").with_deleted())
                .await
                .unwrap()
                .len(),
            1
        );
        // Record is still in the store (not hard-deleted).
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn tier_filter_restricts_kind() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        store
            .remember(Memory::semantic(
                scope.clone(),
                "shared word alpha",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store
            .remember(Memory::episodic(
                scope.clone(),
                "shared word alpha",
                "event",
            ))
            .await
            .unwrap();

        let only_sem = store
            .recall(&scope, &Query::new("alpha").tier(Tier::Semantic))
            .await
            .unwrap();
        assert_eq!(only_sem.len(), 1);
        assert_eq!(only_sem[0].item.tier(), Tier::Semantic);
    }

    #[tokio::test]
    async fn json_snapshot_roundtrip() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();
        store
            .remember(Memory::semantic(
                scope.clone(),
                "persistent knowledge",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        let json = store.to_json().await.unwrap();
        let restored = InMemoryStore::from_json(&json).unwrap();
        assert_eq!(restored.len().await, 1);

        let res = restored
            .recall(&scope, &Query::new("persistent"))
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
    }

    #[tokio::test]
    async fn browse_mode_ranks_by_recency_and_importance() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        store
            .remember(
                Memory::semantic(scope.clone(), "low importance", SemanticCat::Fact)
                    .with_importance(0.1),
            )
            .await
            .unwrap();
        store
            .remember(
                Memory::semantic(scope.clone(), "high importance", SemanticCat::Fact)
                    .with_importance(0.9),
            )
            .await
            .unwrap();

        // Empty query → browse mode.
        let res = store.recall(&scope, &Query::new("")).await.unwrap();
        assert_eq!(res.len(), 2);
        assert!(res[0].item.summary().contains("high"));
    }

    #[tokio::test]
    async fn min_importance_excludes_low_value_traces() {
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();
        store
            .remember(
                Memory::episodic(scope.clone(), "responded to 'x':", "echo about rust")
                    .with_importance(Memory::AUTO_IMPORTANCE),
            )
            .await
            .unwrap();
        store
            .remember(Memory::episodic(scope.clone(), "favori dil", "rust")) // default 0.5
            .await
            .unwrap();
        // Without a floor both match; with a 0.35 floor the auto trace is dropped.
        let all = store
            .recall(&scope, &Query::new("rust").semantic())
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        let floored = store
            .recall(&scope, &Query::new("rust").semantic().min_importance(0.35))
            .await
            .unwrap();
        assert_eq!(floored.len(), 1);
        assert!(floored[0].item.recall_context().contains("favori dil"));
    }

    #[tokio::test]
    async fn embed_text_overrides_vector_leg() {
        use crate::memory::embed::HashingEmbedder;
        let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
        let (_a, scope) = agent_scope();
        store
            .remember(Memory::semantic(
                scope.clone(),
                "interest in mathematics",
                SemanticCat::Preference,
            ))
            .await
            .unwrap();

        // text is unrelated ("zzz") but embed_text is "math" → vector leg catches it (HyDE mechanism).
        let q = Query::new("zzz").semantic().embed_text("math");
        let res = store.recall(&scope, &q).await.unwrap();
        assert_eq!(res.len(), 1, "embed_text should drive the vector leg");
    }

    #[tokio::test]
    async fn consolidate_merges_near_duplicates() {
        use crate::memory::embed::HashingEmbedder;
        let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
        let (_a, scope) = agent_scope();
        store
            .remember(Memory::semantic(
                scope.clone(),
                "repeated knowledge record",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store
            .remember(Memory::semantic(
                scope.clone(),
                "repeated knowledge record",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        assert_eq!(
            store
                .recall(&scope, &Query::new("repeated"))
                .await
                .unwrap()
                .len(),
            2
        );
        let report = store.consolidate().await.unwrap();
        assert_eq!(report.merged, 1);
        assert_eq!(
            store
                .recall(&scope, &Query::new("repeated"))
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn consolidate_forgets_decayed() {
        use chrono::Duration as ChronoDuration;
        let store = InMemoryStore::new();
        let (_a, scope) = agent_scope();

        store
            .remember(
                Memory::semantic(
                    scope.clone(),
                    "current important knowledge",
                    SemanticCat::Fact,
                )
                .with_importance(0.9),
            )
            .await
            .unwrap();
        let mut old = Memory::semantic(
            scope.clone(),
            "very old unimportant knowledge",
            SemanticCat::Fact,
        )
        .with_importance(0.1);
        let past = Utc::now() - ChronoDuration::days(120);
        old.created_at = past;
        old.last_access = past;
        store.remember(old).await.unwrap();

        let report = store.consolidate().await.unwrap();
        assert_eq!(report.scanned, 2);
        assert_eq!(report.forgotten, 1);

        let alive = store
            .recall(&scope, &Query::new("knowledge"))
            .await
            .unwrap();
        assert_eq!(alive.len(), 1);
        assert!(alive[0].item.summary().contains("current"));
    }
}
