//! Identity types: `AgentId` and `MemoryId`.
//!
//! ULID-based: time-ordered (lexicographically sortable), collision-free,
//! URL-safe. Behaves like a plain `String` in serialization.

use serde::{Deserialize, Serialize};
use ulid::Ulid;

/// Persistent, unique identity of an agent.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(String);

impl AgentId {
    /// Generates a new, time-ordered identity.
    pub fn new() -> Self {
        Self(Ulid::new().to_string())
    }

    /// Raw string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for AgentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for AgentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identity of a memory record.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(String);

impl MemoryId {
    /// Generates a new, time-ordered identity.
    pub fn new() -> Self {
        Self(Ulid::new().to_string())
    }

    /// Raw string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for MemoryId {
    fn from(s: String) -> Self {
        Self(s)
    }
}
