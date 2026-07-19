//! `MockModel`: deterministic test/demo model.
//!
//! Does not call a real LLM; uses identity and recalled memories to produce a
//! predictable response. This way the memory → identity → response loop can be
//! tested.

use super::{Completion, Model, Prompt};
use crate::error::Result;
use async_trait::async_trait;

/// Deterministic model. Reflects recalled context in the response.
#[derive(Clone, Debug, Default)]
pub struct MockModel;

impl MockModel {
    /// New mock model.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Model for MockModel {
    async fn complete(&self, prompt: &Prompt) -> Result<Completion> {
        let mut text = if prompt.context.is_empty() {
            format!("(memory empty) reflecting on '{}'.", prompt.user)
        } else {
            format!(
                "recalling {} memories, responding to '{}': {}",
                prompt.context.len(),
                prompt.user,
                prompt.context.join(" | ")
            )
        };
        // Conversation history is reflected deterministically (testability).
        if !prompt.history.is_empty() {
            text.push_str(&format!(
                " (chat history: {} messages)",
                prompt.history.len()
            ));
        }
        Ok(Completion::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_context_is_acknowledged() {
        let m = MockModel::new();
        let p = Prompt {
            system: "You are Aria.".into(),
            user: "hello".into(),
            ..Default::default()
        };
        let c = m.complete(&p).await.unwrap();
        assert!(c.text.contains("memory empty"));
        assert!(c.text.contains("hello"));
        assert!(
            !c.text.contains("chat history"),
            "no history suffix when absent"
        );
    }

    #[tokio::test]
    async fn history_is_reflected_deterministically() {
        use super::super::Turn;
        let m = MockModel::new();
        let p = Prompt {
            system: "You are Aria.".into(),
            history: vec![Turn::user("hello"), Turn::assistant("hello to you too")],
            user: "what did I say?".into(),
            ..Default::default()
        };
        let c = m.complete(&p).await.unwrap();
        assert!(c.text.contains("chat history: 2 messages"), "{}", c.text);
    }

    #[tokio::test]
    async fn context_is_reflected_in_reply() {
        let m = MockModel::new();
        let p = Prompt {
            system: "You are Aria.".into(),
            context: vec!["[semantic] Rust is fast".into()],
            user: "how is rust".into(),
            ..Default::default()
        };
        let c = m.complete(&p).await.unwrap();
        assert!(c.text.contains("recalling 1 memories"));
        assert!(c.text.contains("Rust is fast"));
    }
}
