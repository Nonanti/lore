//! Lore error types.

use thiserror::Error;

/// Error type used across Lore.
#[derive(Debug, Error)]
pub enum LoreError {
    /// Requested resource (agent, memory record...) not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Invalid user input (maps to HTTP 422; not a server error).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Serialization error (JSON snapshots, etc.).
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// HTTP / network error (real model call).
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Model/provider error (API response, empty reply, etc.).
    #[error("model error: {0}")]
    Model(String),

    /// Storage error (sqlite, etc.).
    #[error("storage error: {0}")]
    Storage(String),

    /// Service/server error (bind, listen, etc.).
    #[error("server error: {0}")]
    Server(String),

    /// Conflict: the action cannot be applied to the current state
    /// (e.g. deciding an already-decided approval). Maps to HTTP 409.
    #[error("conflict: {0}")]
    Conflict(String),

    /// Action denied by policy gate.
    #[error("policy denied: {0}")]
    PolicyDenied(String),

    /// Other, wrapped errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Shortcut `Result` type for Lore.
pub type Result<T> = std::result::Result<T, LoreError>;
