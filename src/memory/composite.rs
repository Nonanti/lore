//! Composite store: personal + shared (team) memory behind one `MemoryStore`.
//!
//! Daemon agents live in separate SQLite files, so `Scope::World` records —
//! "visible to all agents" by definition — were trapped inside each private
//! file. `CompositeStore` frees them: writes route by scope (World → the
//! shared store, Agent → the personal store), recalls merge both sides.
//! Everything above the trait (agents, distillation, recall legs) stays
//! store-agnostic.
//!
//! Design: `docs/superpowers/specs/2026-07-24-team-memory-design.md`.

use super::types::{ConsolidationReport, Memory, Outcome, Query, Scope, Scored};
use super::MemoryStore;
use crate::error::{LoreError, Result};
use crate::id::MemoryId;
use async_trait::async_trait;
use std::sync::Arc;

/// Personal + shared store pair. See module docs.
pub struct CompositeStore {
    personal: Arc<dyn MemoryStore>,
    shared: Arc<dyn MemoryStore>,
}

impl CompositeStore {
    /// Composes a personal store with a shared (team) store.
    pub fn new(personal: Arc<dyn MemoryStore>, shared: Arc<dyn MemoryStore>) -> Self {
        Self { personal, shared }
    }

    /// Which store owns an EXISTING record id (personal wins ties).
    async fn owner_of(&self, id: &MemoryId) -> Result<Option<&Arc<dyn MemoryStore>>> {
        if self.personal.get(id).await?.is_some() {
            return Ok(Some(&self.personal));
        }
        if self.shared.get(id).await?.is_some() {
            return Ok(Some(&self.shared));
        }
        Ok(None)
    }
}

#[async_trait]
impl MemoryStore for CompositeStore {
    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        if let Some(m) = self.personal.get(id).await? {
            return Ok(Some(m));
        }
        self.shared.get(id).await
    }

    async fn remember(&self, mem: Memory) -> Result<MemoryId> {
        match mem.scope {
            Scope::World => self.shared.remember(mem).await,
            Scope::Agent(_) => self.personal.remember(mem).await,
        }
    }

    async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>> {
        // Both sides run the full query (each store enforces `Scope::sees`
        // itself: the shared store answers with World records, the personal
        // store with own + any local World records). Per-store finalize
        // (rerank/MMR/graph) runs before the merge; MMR diversity is not
        // re-applied across stores — documented tradeoff.
        let mut hits = self.personal.recall(scope, query).await?;
        hits.extend(self.shared.recall(scope, query).await?);
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(query.limit);
        Ok(hits)
    }

    async fn reinforce(&self, id: &MemoryId, outcome: Outcome) -> Result<()> {
        match self.owner_of(id).await? {
            Some(store) => store.reinforce(id, outcome).await,
            None => Err(LoreError::NotFound(id.to_string())),
        }
    }

    async fn reinforce_many(&self, ids: &[MemoryId], outcome: Outcome) -> Result<()> {
        // Split by owning store, batch each side (missing ids are skipped —
        // same contract as the single-store implementations).
        let mut personal_ids = Vec::new();
        let mut shared_ids = Vec::new();
        for id in ids {
            if self.personal.get(id).await?.is_some() {
                personal_ids.push(id.clone());
            } else if self.shared.get(id).await?.is_some() {
                shared_ids.push(id.clone());
            }
        }
        if !personal_ids.is_empty() {
            self.personal.reinforce_many(&personal_ids, outcome).await?;
        }
        if !shared_ids.is_empty() {
            self.shared.reinforce_many(&shared_ids, outcome).await?;
        }
        Ok(())
    }

    async fn forget(&self, id: &MemoryId) -> Result<()> {
        match self.owner_of(id).await? {
            Some(store) => store.forget(id).await,
            None => Err(LoreError::NotFound(id.to_string())),
        }
    }

    async fn count(&self, scope: &Scope) -> Result<usize> {
        Ok(self.personal.count(scope).await? + self.shared.count(scope).await?)
    }

    async fn consolidate(&self) -> Result<ConsolidationReport> {
        // Both sides. Shared-file consolidation under concurrent workers is
        // safe: WAL + IMMEDIATE transactions, and merges are idempotent.
        let a = self.personal.consolidate().await?;
        let b = self.shared.consolidate().await?;
        Ok(ConsolidationReport {
            scanned: a.scanned + b.scanned,
            merged: a.merged + b.merged,
            forgotten: a.forgotten + b.forgotten,
        })
    }

    async fn export(&self) -> Result<Vec<Memory>> {
        let mut all = self.personal.export().await?;
        all.extend(self.shared.export().await?);
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::AgentId;
    use crate::memory::types::SemanticCat;
    use crate::memory::InMemoryStore;

    fn composite() -> (CompositeStore, Arc<dyn MemoryStore>, Arc<dyn MemoryStore>) {
        let personal: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let shared: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        (
            CompositeStore::new(personal.clone(), shared.clone()),
            personal,
            shared,
        )
    }

    #[tokio::test]
    async fn writes_route_by_scope() {
        let (c, personal, shared) = composite();
        let scope = Scope::Agent(AgentId::new());

        c.remember(Memory::semantic(
            scope.clone(),
            "personal fact",
            SemanticCat::Fact,
        ))
        .await
        .unwrap();
        c.remember(Memory::semantic(
            Scope::World,
            "team convention",
            SemanticCat::Convention,
        ))
        .await
        .unwrap();

        assert_eq!(personal.count(&scope).await.unwrap(), 1);
        assert_eq!(personal.count(&Scope::World).await.unwrap(), 0);
        assert_eq!(shared.count(&Scope::World).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn recall_merges_both_sides_and_respects_limit() {
        let (c, _p, _s) = composite();
        let scope = Scope::Agent(AgentId::new());
        c.remember(Memory::semantic(
            scope.clone(),
            "rust ownership basics",
            SemanticCat::Fact,
        ))
        .await
        .unwrap();
        c.remember(Memory::semantic(
            Scope::World,
            "rust formatting convention",
            SemanticCat::Convention,
        ))
        .await
        .unwrap();

        let res = c
            .recall(&scope, &Query::new("rust").limit(5))
            .await
            .unwrap();
        assert_eq!(res.len(), 2, "both sides merged");

        let res1 = c
            .recall(&scope, &Query::new("rust").limit(1))
            .await
            .unwrap();
        assert_eq!(res1.len(), 1, "merge respects limit");
    }

    #[tokio::test]
    async fn shared_records_are_not_visible_to_other_worlds_only_scope_rules() {
        let (c, _p, _s) = composite();
        let a = Scope::Agent(AgentId::new());
        let b = Scope::Agent(AgentId::new());
        c.remember(Memory::semantic(
            a.clone(),
            "agent a private note",
            SemanticCat::Fact,
        ))
        .await
        .unwrap();
        c.remember(Memory::semantic(
            Scope::World,
            "shared team note",
            SemanticCat::Convention,
        ))
        .await
        .unwrap();

        // Agent B sees the shared note but never A's private one.
        let res = c.recall(&b, &Query::new("note").limit(5)).await.unwrap();
        assert!(res
            .iter()
            .any(|s| s.item.searchable_text().contains("shared team")));
        assert!(!res
            .iter()
            .any(|s| s.item.searchable_text().contains("private")));
    }

    #[tokio::test]
    async fn get_reinforce_forget_fall_back_to_shared() {
        let (c, _p, _s) = composite();
        let shared_id = c
            .remember(Memory::semantic(
                Scope::World,
                "team convention",
                SemanticCat::Convention,
            ))
            .await
            .unwrap();

        assert!(c.get(&shared_id).await.unwrap().is_some());
        c.reinforce(&shared_id, Outcome::Accessed).await.unwrap();
        assert_eq!(
            c.get(&shared_id).await.unwrap().unwrap().access_count,
            1,
            "reinforce reached the shared store"
        );
        c.forget(&shared_id).await.unwrap();
        assert!(
            c.get(&shared_id)
                .await
                .unwrap()
                .unwrap()
                .deleted_at
                .is_some(),
            "forget reached the shared store"
        );

        // Unknown id surfaces NotFound.
        let missing = MemoryId::new();
        assert!(matches!(
            c.reinforce(&missing, Outcome::Accessed).await,
            Err(LoreError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn reinforce_many_splits_by_owner_and_skips_missing() {
        let (c, _p, _s) = composite();
        let scope = Scope::Agent(AgentId::new());
        let pid = c
            .remember(Memory::semantic(
                scope.clone(),
                "personal fact",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        let sid = c
            .remember(Memory::semantic(
                Scope::World,
                "shared convention",
                SemanticCat::Convention,
            ))
            .await
            .unwrap();
        let missing = MemoryId::new();

        c.reinforce_many(&[pid.clone(), sid.clone(), missing], Outcome::Accessed)
            .await
            .unwrap();
        assert_eq!(c.get(&pid).await.unwrap().unwrap().access_count, 1);
        assert_eq!(c.get(&sid).await.unwrap().unwrap().access_count, 1);
    }

    #[tokio::test]
    async fn count_export_consolidate_aggregate() {
        let (c, _p, _s) = composite();
        let scope = Scope::Agent(AgentId::new());
        c.remember(Memory::semantic(scope.clone(), "one", SemanticCat::Fact))
            .await
            .unwrap();
        c.remember(Memory::semantic(
            Scope::World,
            "two",
            SemanticCat::Convention,
        ))
        .await
        .unwrap();

        assert_eq!(c.count(&scope).await.unwrap(), 2, "personal + shared World");
        assert_eq!(c.export().await.unwrap().len(), 2);
        let report = c.consolidate().await.unwrap();
        assert_eq!(report.scanned, 2, "both sides consolidated");
    }
}
