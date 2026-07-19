//! `Conversation`: working memory of a chat — last N turns, verbatim.
//!
//! Two-tier memory model: **working memory** (this type; bounded, verbatim,
//! ephemeral) + **long-term memory** (`MemoryStore`; every exchange is already
//! recorded as episodic, retrievable later). When the window is full, the oldest
//! turns drop out — they are not lost, they are already in long-term memory.

use crate::model::Turn;
use std::collections::VecDeque;

/// Default number of turns kept in working memory (10 exchanges).
pub const DEFAULT_CONVERSATION_CAP: usize = 20;

/// Bounded, ordered conversation window (oldest to newest).
#[derive(Clone, Debug)]
pub struct Conversation {
    turns: VecDeque<Turn>,
    cap: usize,
}

impl Conversation {
    /// Empty conversation with the default window.
    pub fn new() -> Self {
        Self::with_cap(DEFAULT_CONVERSATION_CAP)
    }

    /// Empty conversation with a specific window size (minimum 2; odd numbers are rounded up —
    /// the window always holds COMPLETE exchanges, a user turn is never dropped without
    /// its pair, and history never starts with an assistant turn).
    pub fn with_cap(cap: usize) -> Self {
        let cap = cap.max(2);
        Self {
            turns: VecDeque::new(),
            cap: cap + (cap & 1), // round up to even
        }
    }

    /// Copy of the history (for prompt construction; oldest to newest).
    pub fn history(&self) -> Vec<Turn> {
        self.turns.iter().cloned().collect()
    }

    /// Number of turns in the window.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Is the conversation empty?
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Adds an exchange (user + agent) to the window; overflowing oldest turns are dropped.
    pub(crate) fn record(&mut self, input: &str, reply: &str) {
        self.turns.push_back(Turn::user(input));
        self.turns.push_back(Turn::assistant(reply));
        while self.turns.len() > self.cap {
            self.turns.pop_front();
        }
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Role;

    #[test]
    fn records_pairs_in_order() {
        let mut c = Conversation::new();
        assert!(c.is_empty());
        c.record("hi", "hi to you too");
        let h = c.history();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].role, Role::User);
        assert_eq!(h[0].text, "hi");
        assert_eq!(h[1].role, Role::Assistant);
        assert_eq!(h[1].text, "hi to you too");
    }

    #[test]
    fn cap_trims_oldest_turns() {
        let mut c = Conversation::with_cap(4);
        c.record("1", "a");
        c.record("2", "b");
        c.record("3", "c"); // oldest exchange (1/a) drops out
        assert_eq!(c.len(), 4);
        let h = c.history();
        assert_eq!(h[0].text, "2", "oldest dropped");
        assert_eq!(h[3].text, "c");
    }

    #[test]
    fn cap_has_sane_minimum() {
        let mut c = Conversation::with_cap(0); // raised to 2
        c.record("x", "y");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn odd_cap_rounds_up_and_never_splits_exchange() {
        // Odd cap is rounded up to even: exchanges are never split, history never starts with assistant.
        let mut c = Conversation::with_cap(3); // 4 olur
        c.record("1", "a");
        c.record("2", "b");
        c.record("3", "c");
        assert_eq!(c.len(), 4);
        let h = c.history();
        assert_eq!(h[0].role, Role::User, "window starts with full pair");
        assert_eq!(h[0].text, "2");
    }
}
