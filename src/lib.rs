//! # Lore
//!
//! **Identity + orchestration + memory** core for AI agents.
//!
//! Lore is fully standalone: it does not depend on external services, HTTP, or APIs.
//! The memory engine is written from scratch in native Rust within this crate.
//!
//! ## Subsystems
//! - [`id`] — `AgentId`, `MemoryId` (ULID-based)
//! - [`memory`] — three-tier memory (`Episodic`/`Semantic`/`Procedural`),
//!   [`memory::MemoryStore`] trait and native stores
//!   ([`memory::InMemoryStore`], [`memory::SqliteStore`] — FTS5-indexed),
//!   hybrid retrieval (keyword + cosine + recency/importance/Wilson),
//!   consolidation (decay + near-duplicate merge)
//! - [`agent`] — [`agent::Agent`] (identity + memory + model) and [`agent::Persona`]
//! - [`model`] — [`model::Model`] trait, [`model::MockModel`] and
//!   [`model::OpenAiModel`] (OpenAI-compatible: including Ollama, streaming + `<think>` extraction)
//! - [`orchestrator`] — [`orchestrator::Orchestrator`] (supervisor + mailbox + messaging)
//! - [`server`] — HTTP/WS API (auth + rate limit + federation + observability)
//! - [`tool`] — tool registry + router (`KeywordRouter`, `LlmRouter`)

pub mod agent;
pub mod auth;
pub mod daemon;
pub mod error;
pub mod id;
pub mod memory;
pub mod model;
pub mod orchestrator;
pub mod policy;
pub mod server;
pub mod task;
pub mod tool;

pub use agent::roles::{preset, presets};
pub use agent::{Agent, Conversation, Persona, WorkReport, WorkSpec};
pub use auth::{AccessTokenProvider, Credential, RefreshingToken, StaticToken, TokenStore};
pub use daemon::{run_daemon, run_task, TaskDeps};
pub use error::{LoreError, Result};
pub use id::{AgentId, MemoryId};
#[cfg(feature = "neural")]
pub use memory::NeuralEmbedder;
pub use memory::{
    ConsolidationReport, Embedder, FiveW, ForgetPolicy, HashingEmbedder, InMemoryStore, Memory,
};
pub use memory::{
    MemoryGraph, MemoryKind, MemoryStore, NativeReranker, Outcome, Query, Reranker, Scope, Scored,
    SemanticCat, Signal, SqliteStore, Tier,
};
pub use model::{
    build_model, build_model_from_env, AnthropicAuth, AnthropicModel, AuthKind, CodexModel,
    Completion, MockModel, Model, ModelConfig, OpenAiModel, Prompt, ProviderKind, Role, Turn,
};
pub use orchestrator::{Delivery, Envelope, MessageKind, Orchestrator, Party, Recipient, Registry};
pub use policy::approval::{AllowAll, Approver, CliApprover, DenyAll, Gate};
pub use policy::{Action, DefaultExec, Policy, SandboxMode, Verdict};
pub use server::{
    ActResp, AgentView, AppState, AskResp, DeliberateReply, DeliberateResp, MemoryView,
    PersonaPatch,
};
pub use task::approver::QueueApprover;
pub use task::{ApprovalEntry, ApprovalStatus, NewTask, Task, TaskStatus, TaskStore};
pub use tool::{
    parse_tool_call, CalcTool, FileEditTool, FileReadTool, FileWriteTool, KeywordRouter, LlmRouter,
    ShellTool, TimeTool, Tool, ToolCall, ToolContext, ToolRegistry, ToolRouter, WebFetchTool,
};
