//! Model layer: abstraction for how an agent "thinks".
//!
//! The `Model` trait hides which LLM is underneath. In M2, we close
//! the end-to-end loop with the deterministic [`MockModel`]; the real
//! OpenAI-compatible client (including Ollama) will sit behind this trait in M4.

mod anthropic;
mod codex;
mod factory;
mod mock;
mod openai;

pub use anthropic::{AnthropicAuth, AnthropicModel};
pub use codex::CodexModel;
pub use factory::{build_model, build_model_from_env, AuthKind, ModelConfig, ProviderKind};
pub use mock::MockModel;
pub use openai::OpenAiModel;

use crate::error::Result;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

/// Streaming response: each item is a text fragment (token/word chunk).
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Party in a conversation turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// User (the asking party).
    User,
    /// Agent (the responding party).
    Assistant,
}

/// A conversation turn: who said what.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Owner of the turn.
    pub role: Role,
    /// Spoken text.
    pub text: String,
}

impl Turn {
    /// User turn.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            text: text.into(),
        }
    }

    /// Agent turn.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            text: text.into(),
        }
    }
}

/// Prompt to the model: identity (system) + recalled memories (context) +
/// conversation history (history, ordered turns) + input (user).
#[derive(Clone, Debug, Default)]
pub struct Prompt {
    /// Identity injection (from persona).
    pub system: String,
    /// Context lines recalled from memory.
    pub context: Vec<String>,
    /// Conversation history — recent turns in working memory (oldest first).
    pub history: Vec<Turn>,
    /// Actual user input / task.
    pub user: String,
}

impl Prompt {
    /// Renders the prompt into a single flat text (for sending to real models).
    pub fn render(&self) -> String {
        let mut s = String::new();
        if !self.system.is_empty() {
            s.push_str(&self.system);
            s.push_str("\n\n");
        }
        if !self.context.is_empty() {
            s.push_str("What you recall:\n");
            for c in &self.context {
                s.push_str("- ");
                s.push_str(c);
                s.push('\n');
            }
            s.push('\n');
        }
        if !self.history.is_empty() {
            s.push_str("Conversation:\n");
            for t in &self.history {
                let who = match t.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                };
                s.push_str(who);
                s.push_str(": ");
                s.push_str(&t.text);
                s.push('\n');
            }
            s.push('\n');
        }
        s.push_str(&self.user);
        s
    }
}

/// Response produced by the model.
#[derive(Clone, Debug)]
pub struct Completion {
    /// Generated text.
    pub text: String,
    /// Reasoning fallback: text comes from the model's chain-of-thought (content
    /// was empty). May be shown to the user but should not be written to memory
    /// as raw CoT — the storing side trims it (context pollution/prompt bloat).
    pub reasoning_fallback: bool,
}

impl Completion {
    /// Normal completion.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reasoning_fallback: false,
        }
    }

    /// Response fallen from chain-of-thought (reasoning model, empty content).
    pub fn reasoning_fallback(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reasoning_fallback: true,
        }
    }
}

/// A language model abstraction. All providers sit behind this trait.
#[async_trait]
pub trait Model: Send + Sync {
    /// Completes the given prompt.
    async fn complete(&self, prompt: &Prompt) -> Result<Completion>;

    /// Streams the response in chunks. Default: `complete` result as a single
    /// chunk — correct fallback for models that do not support streaming (Mock,
    /// etc.).
    async fn complete_stream(&self, prompt: &Prompt) -> Result<TokenStream> {
        let c = self.complete(prompt).await?;
        Ok(Box::pin(futures::stream::iter(vec![Ok(c.text)])))
    }
}
