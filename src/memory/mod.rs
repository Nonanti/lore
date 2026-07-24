//! Memory subsystem: `MemoryStore` trait, types, and implementations.
//!
//! Design decision (D1): memory is abstracted behind a `trait` — not to plug in
//! external backends, but to make our native implementations (`InMemoryStore` ↔
//! `SqliteStore`) swappable/testable. Lore is fully self-contained; there is no
//! external service/HTTP/API.

pub mod embed;
pub mod evolution;
pub mod graph;
mod in_memory;
pub mod rerank;
pub mod retrieval;
mod sqlite;
mod types;

#[cfg(feature = "neural")]
pub use embed::NeuralEmbedder;
pub use embed::{Embedder, HashingEmbedder};
pub use evolution::ForgetPolicy;
pub use graph::MemoryGraph;
pub use in_memory::InMemoryStore;
#[cfg(feature = "neural")]
pub use rerank::NeuralReranker;
pub use rerank::{NativeReranker, Reranker};
pub use sqlite::SqliteStore;
pub use types::{
    ConsolidationReport, FiveW, Memory, MemoryKind, Outcome, Query, Scope, Scored, SemanticCat,
    Signal, Tier,
};

use crate::error::Result;
use crate::id::MemoryId;
use async_trait::async_trait;

/// Interface through which an agent talks to its personal memory engine.
///
/// All implementations are native (in-memory, sqlite, ...); no external dependencies.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Retrieves a record by identity (None if not found).
    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>>;

    /// Stores a new memory, returns its identity.
    async fn remember(&self, mem: Memory) -> Result<MemoryId>;

    /// Runs the query in the given `scope` visibility, returns scored results.
    /// Retrieval-level isolation is enforced here: no leakage outside scope.
    async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>>;

    /// Reinforces a record: processes access/success/failure signal.
    async fn reinforce(&self, id: &MemoryId, outcome: Outcome) -> Result<()>;

    /// Reinforces multiple records in one call (e.g. bulk access marking
    /// for recall hits). Missing identities are skipped — does not kill the
    /// batch. Default calls `reinforce` sequentially; stores may override with
    /// a single transaction/lock.
    async fn reinforce_many(&self, ids: &[MemoryId], outcome: Outcome) -> Result<()> {
        for id in ids {
            if self.get(id).await?.is_some() {
                self.reinforce(id, outcome).await?;
            }
        }
        Ok(())
    }

    /// Soft-deletes a record (soft-delete + timestamp).
    async fn forget(&self, id: &MemoryId) -> Result<()>;

    /// Count of live (not deleted) records visible in the given `scope`.
    /// Cheap count without full-scan recall (for metrics/monitoring).
    async fn count(&self, scope: &Scope) -> Result<usize>;

    /// Maintenance pass: returns scan/merge/forget report (background task).
    async fn consolidate(&self) -> Result<ConsolidationReport>;

    /// Exports all live (not deleted) records — for backup/migration.
    async fn export(&self) -> Result<Vec<Memory>>;
}
