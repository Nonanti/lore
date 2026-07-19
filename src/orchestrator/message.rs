//! Orchestration message types: who → whom, what kind, what content.

use crate::id::AgentId;

/// Party that can be the source/destination of a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Party {
    /// Orchestration / user (system).
    System,
    /// A specific agent.
    Agent(AgentId),
}

/// Message destination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Recipient {
    /// A single agent.
    Agent(AgentId),
    /// All agents except the sender.
    Broadcast,
}

/// Message kind — determines how the recipient behaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    /// Question: recipient produces a response (`respond`), reply returns to
    /// sender.
    Ask,
    /// Notification: recipient only perceives (`experience`), no reply.
    Tell,
}

/// A message envelope.
#[derive(Clone, Debug)]
pub struct Envelope {
    /// Sender.
    pub from: Party,
    /// Destination.
    pub to: Recipient,
    /// Content.
    pub content: String,
    /// Kind.
    pub kind: MessageKind,
}

impl Envelope {
    /// Ask envelope to an agent.
    pub fn ask(from: Party, to: AgentId, content: impl Into<String>) -> Self {
        Self {
            from,
            to: Recipient::Agent(to),
            content: content.into(),
            kind: MessageKind::Ask,
        }
    }

    /// Tell envelope to an agent.
    pub fn tell(from: Party, to: AgentId, content: impl Into<String>) -> Self {
        Self {
            from,
            to: Recipient::Agent(to),
            content: content.into(),
            kind: MessageKind::Tell,
        }
    }

    /// Broadcast envelope to everyone.
    pub fn broadcast(from: Party, content: impl Into<String>) -> Self {
        Self {
            from,
            to: Recipient::Broadcast,
            content: content.into(),
            kind: MessageKind::Tell,
        }
    }
}

/// Record of a processed message (transcript entry).
#[derive(Clone, Debug)]
pub struct Delivery {
    /// Agent that received the message.
    pub to: AgentId,
    /// Sender.
    pub from: Party,
    /// Kind.
    pub kind: MessageKind,
    /// Content.
    pub content: String,
    /// Response produced if Ask (`None` on error).
    pub reply: Option<String>,
    /// Error during delivery (model/memory) — if `Some`, there is no `reply`.
    /// A single agent's error does not stop orchestration; it is marked in
    /// the transcript.
    pub error: Option<String>,
}
