//! Structured conversation thread for native tool calling.
//!
//! Every provider tool API is message/block based (Anthropic content
//! blocks, OpenAI messages + `tool_calls`, Responses items). [`Thread`]
//! is the provider-neutral core those wire formats are built from; the
//! flat [`Prompt`](super::Prompt) render remains the path for
//! text-protocol models. Providers translate blocks explicitly — these
//! types deliberately do NOT derive serde, so no wire format can lean on
//! an accidental default tagging.
//!
//! Design: `docs/superpowers/specs/2026-07-24-native-tool-calling-design.md`.

/// One content block — the unit all provider wire formats share.
#[derive(Clone, Debug, PartialEq)]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text itself.
        text: String,
    },
    /// The model wants a tool run (appears in assistant messages).
    ToolUse {
        /// Provider-issued call id (correlates the eventual result).
        id: String,
        /// Tool name (matches `Tool::name`).
        name: String,
        /// Structured arguments as the model produced them.
        input: serde_json::Value,
    },
    /// Result of a tool run (appears in user messages).
    ToolResult {
        /// Id of the `ToolUse` this answers.
        tool_use_id: String,
        /// Tool output (or `ERROR: ..` text).
        content: String,
        /// Whether the run failed — providers surface this natively.
        is_error: bool,
    },
}

/// Party owning a chat message. System text lives on [`Thread`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    /// The asking / tool-result-carrying party.
    User,
    /// The model.
    Assistant,
}

/// A message: role + ordered content blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    /// Owner of the message.
    pub role: ChatRole,
    /// Ordered content blocks.
    pub blocks: Vec<ContentBlock>,
}

impl ChatMessage {
    /// Plain-text user message.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            blocks: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Plain-text assistant message.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            blocks: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Assistant message carrying reply blocks verbatim (text + tool_use).
    pub fn assistant_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: ChatRole::Assistant,
            blocks,
        }
    }

    /// User message carrying tool results.
    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        debug_assert!(
            results
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. })),
            "tool_results expects only ToolResult blocks"
        );
        Self {
            role: ChatRole::User,
            blocks: results,
        }
    }
}

/// Conversation state for a tool-loop completion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Thread {
    /// System / identity text (providers place it in their native slot).
    pub system: String,
    /// Messages, oldest first.
    pub messages: Vec<ChatMessage>,
}

impl Thread {
    /// New thread with the given system text and no messages.
    pub fn new(system: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            messages: Vec::new(),
        }
    }

    /// Appends a message.
    pub fn push(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }
}

/// A tool offered to the model natively.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    /// Tool name (matches `Tool::name`).
    pub name: String,
    /// What the tool does.
    pub description: String,
    /// JSON Schema object describing the call input.
    pub input_schema: serde_json::Value,
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end of turn — the reply text is the final answer.
    EndTurn,
    /// The model stopped to have tools run.
    ToolUse,
    /// Anything else (length cut, provider-specific reasons).
    Other,
}

/// Borrowed view of one `ToolUse` block.
#[derive(Clone, Copy, Debug)]
pub struct ToolUseRef<'a> {
    /// Provider-issued call id.
    pub id: &'a str,
    /// Tool name.
    pub name: &'a str,
    /// Structured arguments.
    pub input: &'a serde_json::Value,
}

/// Assistant reply to a thread completion.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadReply {
    /// Assistant content blocks (text and/or tool_use), provider order.
    pub blocks: Vec<ContentBlock>,
    /// Stop reason.
    pub stop: StopReason,
    /// Same semantics as [`Completion::reasoning_fallback`](super::Completion):
    /// text came from chain-of-thought because content was empty.
    pub reasoning_fallback: bool,
}

impl ThreadReply {
    /// Joined text blocks (empty string if none).
    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Tool-use blocks in reply order.
    pub fn tool_uses(&self) -> Vec<ToolUseRef<'_>> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(ToolUseRef {
                    id: id.as_str(),
                    name: name.as_str(),
                    input,
                }),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_helpers_build_expected_shapes() {
        let u = ChatMessage::user_text("hi");
        assert_eq!(u.role, ChatRole::User);
        assert_eq!(u.blocks, vec![ContentBlock::Text { text: "hi".into() }]);

        let a = ChatMessage::assistant_text("hello");
        assert_eq!(a.role, ChatRole::Assistant);

        let r = ChatMessage::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "t1".into(),
            content: "4".into(),
            is_error: false,
        }]);
        assert_eq!(r.role, ChatRole::User);
    }

    #[test]
    fn thread_reply_text_joins_only_text_blocks() {
        let reply = ThreadReply {
            blocks: vec![
                ContentBlock::Text {
                    text: "I'll check. ".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "calc".into(),
                    input: serde_json::json!({"args": "2+2"}),
                },
                ContentBlock::Text {
                    text: "One moment.".into(),
                },
            ],
            stop: StopReason::ToolUse,
            reasoning_fallback: false,
        };
        assert_eq!(reply.text(), "I'll check. One moment.");
    }

    #[test]
    fn thread_reply_tool_uses_preserves_order() {
        let reply = ThreadReply {
            blocks: vec![
                ContentBlock::ToolUse {
                    id: "a".into(),
                    name: "calc".into(),
                    input: serde_json::json!({"args": "1"}),
                },
                ContentBlock::Text { text: "..".into() },
                ContentBlock::ToolUse {
                    id: "b".into(),
                    name: "time".into(),
                    input: serde_json::json!({"args": ""}),
                },
            ],
            stop: StopReason::ToolUse,
            reasoning_fallback: false,
        };
        let uses = reply.tool_uses();
        assert_eq!(uses.len(), 2);
        assert_eq!((uses[0].id, uses[0].name), ("a", "calc"));
        assert_eq!((uses[1].id, uses[1].name), ("b", "time"));
    }

    #[test]
    fn thread_push_appends_in_order() {
        let mut t = Thread::new("sys");
        t.push(ChatMessage::user_text("q"));
        t.push(ChatMessage::assistant_text("a"));
        assert_eq!(t.system, "sys");
        assert_eq!(t.messages.len(), 2);
        assert_eq!(t.messages[0].role, ChatRole::User);
        assert_eq!(t.messages[1].role, ChatRole::Assistant);
    }
}
