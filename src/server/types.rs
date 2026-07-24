//! HTTP API data types: external views and request/response DTOs.

use serde::{Deserialize, Serialize};

/// External view of an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentView {
    /// Agent identity (ULID string).
    pub id: String,
    /// Name.
    pub name: String,
    /// Role.
    pub role: String,
    /// Character traits.
    pub traits: Vec<String>,
    /// Persona version (increments on every update — identity evolution).
    pub version: u32,
}

/// Partial update for persona (only provided fields change).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PersonaPatch {
    /// New name.
    pub name: Option<String>,
    /// New role.
    pub role: Option<String>,
    /// New character description.
    pub description: Option<String>,
    /// New trait list (replaced wholesale).
    pub traits: Option<Vec<String>>,
    /// New additional system instructions.
    pub system_prompt: Option<String>,
}

/// External view of a memory record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryView {
    /// Record identity.
    pub id: String,
    /// Retrieval score.
    pub score: f32,
    /// Short summary.
    pub summary: String,
}

/// An `ask` response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AskResp {
    /// Agent's reply.
    pub reply: String,
}

/// A single agent's reply in collective deliberation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliberateReply {
    /// Agent identity.
    pub id: String,
    /// Agent name.
    pub name: String,
    /// Reply.
    pub reply: String,
    /// Node the reply came from (None = local; Some(url) = federation peer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
}

/// A `deliberate` result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliberateResp {
    /// Team replies.
    pub replies: Vec<DeliberateReply>,
    /// Supervisor synthesis (only if `synthesizer` was provided).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<String>,
}

/// An `act` result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActResp {
    /// Tool result or reply.
    pub result: String,
}

// --- Request DTOs (handlers only) ---

#[derive(Deserialize)]
pub(super) struct CreateReq {
    pub(super) name: String,
    pub(super) role: String,
    #[serde(default)]
    pub(super) traits: Vec<String>,
}

#[derive(Deserialize)]
pub(super) struct AskReq {
    pub(super) message: String,
    /// Session name: if provided, conversation history (working memory) is preserved —
    /// subsequent questions with the same `session` remember previous turns.
    #[serde(default)]
    pub(super) session: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct ActReq {
    pub(super) input: String,
}

#[derive(Deserialize)]
pub(super) struct SolveReq {
    pub(super) input: String,
    /// Tool loop step limit (default 5, cap 10).
    pub(super) max_steps: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum MsgKind {
    Ask,
    Tell,
}

#[derive(Deserialize)]
pub(super) struct MessageReq {
    /// Sender agent (optional; defaults to "System" if absent).
    pub(super) from: Option<String>,
    /// "ask" (expects a reply) or "tell" (provides info). Default: tell.
    pub(super) kind: Option<MsgKind>,
    pub(super) content: String,
}

#[derive(Deserialize)]
pub(super) struct ExperienceReq {
    pub(super) title: String,
    pub(super) body: String,
}

/// Reflect response: number of distilled memories (episodic → semantic promotion).
#[derive(Serialize)]
pub(super) struct ReflectResp {
    pub(super) distilled: usize,
}

/// Full task view: the task itself + child tasks (when present).
/// Used for GET /tasks/:id — includes report + children.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskFullView {
    /// The task record.
    pub task: crate::task::Task,
    /// Children (subtasks) of this task, empty for standalone tasks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<crate::task::Task>,
}

/// Outcome field for `reinforce` request (lowercase JSON: "accessed" etc.).
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum OutcomeReq {
    Accessed,
    Success,
    Failure,
}

impl From<OutcomeReq> for crate::memory::Outcome {
    fn from(o: OutcomeReq) -> Self {
        match o {
            OutcomeReq::Accessed => Self::Accessed,
            OutcomeReq::Success => Self::Success,
            OutcomeReq::Failure => Self::Failure,
        }
    }
}

#[derive(Deserialize)]
pub(super) struct ReinforceReq {
    /// Record identity to reinforce.
    pub(super) memory_id: String,
    /// Outcome: accessed | success | failure.
    pub(super) outcome: OutcomeReq,
}

#[derive(Deserialize)]
pub(super) struct DeliberateReq {
    pub(super) question: String,
    /// If true, only the local team responds (breaks the federation loop).
    #[serde(default)]
    pub(super) local: bool,
    /// If provided, this agent does not participate in the poll; it synthesizes
    /// all replies (hierarchical team).
    pub(super) synthesizer: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct BoardParams {
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize)]
pub(super) struct RecallParams {
    pub(super) q: Option<String>,
    pub(super) limit: Option<usize>,
    /// Semantic (morphology/synonym) recall — retrieves even without keyword match.
    pub(super) semantic: Option<bool>,
}

#[derive(Deserialize)]
pub(super) struct EnqueueTaskReq {
    /// Agent name (persona file stem).
    pub(super) agent: String,
    /// Goal description (required, non-empty).
    pub(super) goal: String,
    /// Workspace root (optional; defaults to <data>/workspaces/<agent>).
    #[serde(default)]
    pub(super) workspace: Option<String>,
    /// Verification commands (optional).
    #[serde(default)]
    pub(super) verify: Vec<String>,
}

#[derive(Deserialize)]
pub(super) struct TaskListParams {
    /// Maximum number of tasks to return (default 20, max 1000).
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize)]
pub(super) struct TaskLogParams {
    /// Number of lines from the end of the log ("tail" mode).
    pub(super) tail: Option<usize>,
}
