//! HTTP API: turns Lore into a service (axum).
//!
//! Core logic lives in async methods on `AppState` (offline-testable);
//! axum handlers are thin wrappers. Modules:
//!
//! - [`state`](self) — `AppState`: agent lifecycle, ask/act/message, board, metrics
//! - `deliberate` — collective deliberation: team poll, supervisor synthesis, federation
//! - `security` — API key validation + rate-limit middleware
//! - `api` — router, `serve()`, and handlers
//! - `types` — external views and request/response DTOs
//!
//! Endpoints:
//!
//! - `GET  /health`                      → "ok" (open)
//! - `GET  /metrics`                     → Prometheus-style metrics (protected)
//! - `POST /agents`                      → create agent (name, role, traits)
//! - `GET  /agents`                      → list agents
//! - `PATCH/DELETE /agents/:id`          → update persona / delete agent
//! - `POST /agents/:id/ask`              → ask agent (message) → reply
//! - `POST /agents/:id/ask/stream`       → stream response as SSE
//! - `POST /agents/:id/act`              → run tool if matched, otherwise respond
//! - `POST /agents/:id/message`          → inter-agent message (ask → reply, tell → 204)
//! - `POST /agents/:id/experience`       → add episodic memory (title, body)
//! - `GET  /agents/:id/recall?q=&limit=` → recall (`semantic=true` for semantic)
//! - `POST /deliberate`                  → collective deliberation (+synthesizer, +local)
//! - `GET  /deliberate/live`             → WebSocket live deliberate
//! - `GET  /board?limit=`                → read shared board
//!
//! All endpoints except `/health` go through auth + rate-limit middleware.

mod api;
mod deliberate;
mod security;
mod state;
mod types;

/// Prevents log forging: newlines in user-controlled text are replaced with spaces.
/// Every user field that enters logs (agent name, question fragment, etc.) must pass
/// through this.
pub(crate) fn log_safe(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests;

pub use api::{router, serve};
pub use state::AppState;
pub use types::{
    ActResp, AgentView, AskResp, DeliberateReply, DeliberateResp, MemoryView, PersonaPatch,
};
