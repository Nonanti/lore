//! Orchestration: supervisor + registry + mailbox + routing.
//!
//! Pattern: **supervisor**. The orchestrator registers agents, gives each a
//! tokio mpsc **mailbox**, routes messages to their destinations, and drives
//! turns in a deterministic loop. Single-writer safety is maintained:
//! each agent owns its own memory.
//!
//! Flow: `Ask` → recipient `respond`s, reply returns to sender as `Tell`
//! (thus terminating). `Tell` → recipient `experience`s, no reply.

mod message;
mod registry;

pub use message::{Delivery, Envelope, MessageKind, Party, Recipient};
pub use registry::Registry;

use crate::agent::Agent;
use crate::error::Result;
use crate::id::AgentId;
use crate::memory::{Memory, MemoryStore, Query, Scope, Scored};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

/// Core that registers agents and routes messages between them.
pub struct Orchestrator {
    registry: Registry,
    senders: HashMap<AgentId, UnboundedSender<Envelope>>,
    inboxes: HashMap<AgentId, UnboundedReceiver<Envelope>>,
    blackboard: Option<Arc<dyn MemoryStore>>,
    max_steps: usize,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self {
            registry: Registry::new(),
            senders: HashMap::new(),
            inboxes: HashMap::new(),
            blackboard: None,
            max_steps: 256,
        }
    }
}

impl Orchestrator {
    /// New orchestrator (default max_steps = 256).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the upper bound against infinite message loops (builder).
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Attaches a shared blackboard (World-scope board) (builder).
    pub fn with_blackboard(mut self, store: Arc<dyn MemoryStore>) -> Self {
        self.blackboard = Some(store);
        self
    }

    /// Writes a note to the board (World scope; no-op if no blackboard).
    /// Automatic recording: same retention class as server-side `board_note`
    /// ([`Memory::AUTO_IMPORTANCE`]) — if unused, decay may reclaim it.
    pub async fn post_to_board(&self, from: Party, content: impl Into<String>) -> Result<()> {
        if let Some(bb) = &self.blackboard {
            let author = self.party_name(&from);
            bb.remember(
                Memory::episodic(
                    Scope::World,
                    format!("{author} wrote to board"),
                    content.into(),
                )
                .with_importance(Memory::AUTO_IMPORTANCE),
            )
            .await?;
        }
        Ok(())
    }

    /// Reads from the board (empty if no blackboard).
    pub async fn read_board(&self, query: &Query) -> Result<Vec<Scored<Memory>>> {
        match &self.blackboard {
            Some(bb) => bb.recall(&Scope::World, query).await,
            None => Ok(Vec::new()),
        }
    }

    /// Asks the entire team (except the sender), collects responses.
    /// Responses are collected IN PARALLEL (`join_all` preserves input order —
    /// id-sorted, deterministic); a single agent's error does not kill the
    /// poll, it is logged and skipped (behavior parity with server-side
    /// `poll_team`).
    pub async fn poll(&self, from: Party, question: &str) -> Result<Vec<(AgentId, String)>> {
        let mut ids = self.registry.ids();
        ids.sort_by_key(|id| id.to_string());
        let futs = ids
            .into_iter()
            .filter(|id| !matches!(&from, Party::Agent(asker) if asker == id))
            .filter_map(|id| self.registry.get(&id).map(|agent| (id, agent)))
            .map(|(id, agent)| async move { (id, agent.respond(question).await) });
        let mut out = Vec::new();
        for (id, res) in futures::future::join_all(futs).await {
            match res {
                Ok(reply) => out.push((id, reply)),
                Err(e) => tracing::warn!(%id, error = %e, "orchestration poll: no response"),
            }
        }
        Ok(out)
    }

    /// Blackboard pattern: write question to board → ask entire team → write
    /// responses to board.
    pub async fn deliberate(&self, question: &str) -> Result<Vec<(AgentId, String)>> {
        self.post_to_board(Party::System, format!("Question: {question}"))
            .await?;
        let replies = self.poll(Party::System, question).await?;
        for (id, reply) in &replies {
            self.post_to_board(Party::Agent(id.clone()), reply.clone())
                .await?;
        }
        Ok(replies)
    }

    /// Registers an agent and opens a mailbox for it.
    pub fn register(&mut self, agent: Agent) -> AgentId {
        let id = agent.id.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        self.senders.insert(id.clone(), tx);
        self.inboxes.insert(id.clone(), rx);
        self.registry.register(agent);
        id
    }

    /// Accesses an agent by identity.
    pub fn agent(&self, id: &AgentId) -> Option<&Agent> {
        self.registry.get(id)
    }

    /// Number of registered agents.
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Whether the orchestrator is empty.
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }

    /// Display name of a party (for transcript/log).
    pub fn party_name(&self, p: &Party) -> String {
        match p {
            Party::System => "system".to_string(),
            Party::Agent(id) => self
                .registry
                .get(id)
                .map(|a| a.persona.name.clone())
                .unwrap_or_else(|| id.to_string()),
        }
    }

    /// Sends a question to an agent (drops into queue/mailbox).
    pub fn ask(&self, from: Party, to: &AgentId, content: impl Into<String>) {
        self.dispatch(Envelope::ask(from, to.clone(), content));
    }

    /// Sends a notification to an agent.
    pub fn tell(&self, from: Party, to: &AgentId, content: impl Into<String>) {
        self.dispatch(Envelope::tell(from, to.clone(), content));
    }

    /// Broadcasts a notification to everyone except the sender.
    pub fn broadcast(&self, from: Party, content: impl Into<String>) {
        self.dispatch(Envelope::broadcast(from, content));
    }

    /// Routes an envelope to the target mailbox(es).
    fn dispatch(&self, env: Envelope) {
        match &env.to {
            Recipient::Agent(id) => {
                if let Some(tx) = self.senders.get(id) {
                    let _ = tx.send(env);
                }
            }
            Recipient::Broadcast => {
                for (id, tx) in &self.senders {
                    if env.from != Party::Agent(id.clone()) {
                        let _ = tx.send(env.clone());
                    }
                }
            }
        }
    }

    /// Drives turns until all mailboxes are empty (or max_steps is reached).
    /// Returns a record (transcript) of each processed message.
    ///
    /// Resilience: a single agent's model/memory error does NOT STOP the run
    /// and does not lose the transcript accumulated so far — failed delivery
    /// is marked with [`Delivery::error`]. Batch is processed in agent id
    /// order (deterministic).
    pub async fn run(&mut self) -> Result<Vec<Delivery>> {
        let mut transcript = Vec::new();
        let mut steps = 0usize;

        loop {
            // 1) Collect all pending envelopes (id-sorted — HashMap order
            //    varies across runs, transcript order must be stable).
            let mut ids: Vec<AgentId> = self.inboxes.keys().cloned().collect();
            ids.sort_by_key(|id| id.to_string());
            let mut batch: Vec<(AgentId, Envelope)> = Vec::new();
            for id in ids {
                if let Some(rx) = self.inboxes.get_mut(&id) {
                    while let Ok(env) = rx.try_recv() {
                        batch.push((id.clone(), env));
                    }
                }
            }
            if batch.is_empty() {
                break;
            }

            // 2) Process.
            for (target, env) in batch {
                steps += 1;
                if steps > self.max_steps {
                    // No silent truncation: remaining messages were
                    // unprocessed, leave a trace.
                    tracing::warn!(
                        max_steps = self.max_steps,
                        "orchestration: step limit exceeded, remaining messages unprocessed"
                    );
                    return Ok(transcript);
                }

                let agent = match self.registry.get(&target) {
                    Some(a) => a.clone(),
                    None => continue,
                };
                let from_name = self.party_name(&env.from);

                match env.kind {
                    MessageKind::Tell => {
                        let err = agent
                            .experience(format!("{from_name} sent a message"), env.content.clone())
                            .await
                            .err();
                        if let Some(e) = &err {
                            tracing::warn!(agent = %target, error = %e, "orchestration: message could not be received");
                        }
                        transcript.push(Delivery {
                            to: target,
                            from: env.from,
                            kind: env.kind,
                            content: env.content,
                            reply: None,
                            error: err.map(|e| e.to_string()),
                        });
                    }
                    MessageKind::Ask => match agent.respond(&env.content).await {
                        Ok(reply) => {
                            transcript.push(Delivery {
                                to: target.clone(),
                                from: env.from.clone(),
                                kind: env.kind,
                                content: env.content,
                                reply: Some(reply.clone()),
                                error: None,
                            });
                            // Send the reply back to the asker as Tell
                            // (terminates).
                            if let Party::Agent(sender_id) = &env.from {
                                if self.senders.contains_key(sender_id) {
                                    self.dispatch(Envelope::tell(
                                        Party::Agent(target),
                                        sender_id.clone(),
                                        reply,
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(agent = %target, error = %e, "orchestration: no response");
                            transcript.push(Delivery {
                                to: target,
                                from: env.from,
                                kind: env.kind,
                                content: env.content,
                                reply: None,
                                error: Some(e.to_string()),
                            });
                        }
                    },
                }
            }
        }

        Ok(transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, Persona};
    use crate::memory::{InMemoryStore, MemoryStore, Query};
    use crate::model::{MockModel, Model};
    use std::sync::Arc;

    fn make_agent(name: &str, store: Arc<dyn MemoryStore>, model: Arc<dyn Model>) -> Agent {
        Agent::new(Persona::new(name, "role"), store, model)
    }

    fn shared() -> (Arc<dyn MemoryStore>, Arc<dyn Model>) {
        (
            Arc::new(InMemoryStore::new()) as Arc<dyn MemoryStore>,
            Arc::new(MockModel::new()) as Arc<dyn Model>,
        )
    }

    #[tokio::test]
    async fn register_tracks_agents() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        assert!(orch.is_empty());
        let id = orch.register(make_agent("Aria", store, model));
        assert_eq!(orch.len(), 1);
        assert!(orch.agent(&id).is_some());
        assert_eq!(orch.party_name(&Party::Agent(id)), "Aria");
        assert_eq!(orch.party_name(&Party::System), "system");
    }

    #[tokio::test]
    async fn ask_produces_reply_and_records_transcript() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let kai = orch.register(make_agent("Kai", store, model));

        orch.ask(Party::System, &kai, "give status report");
        let t = orch.run().await.unwrap();

        assert_eq!(t.len(), 1);
        assert_eq!(t[0].to, kai);
        assert_eq!(t[0].kind, MessageKind::Ask);
        assert!(t[0].reply.is_some());
        assert!(t[0].reply.as_ref().unwrap().contains("status report"));
    }

    /// Model that errors on every call (resilience tests).
    struct FailModel;
    #[async_trait::async_trait]
    impl Model for FailModel {
        async fn complete(
            &self,
            _p: &crate::model::Prompt,
        ) -> crate::error::Result<crate::model::Completion> {
            Err(crate::error::LoreError::Model("model broken".into()))
        }
    }

    #[tokio::test]
    async fn run_tolerates_failing_agent_and_keeps_transcript() {
        // Kai's model error must NOT lose Aria's delivery:
        // transcript returns complete, failed delivery is marked with error
        // field.
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let aria = orch.register(make_agent("Aria", store.clone(), model));
        let kai = orch.register(make_agent("Kai", store, Arc::new(FailModel)));

        orch.ask(Party::System, &aria, "hello");
        orch.ask(Party::System, &kai, "hello");
        let t = orch.run().await.unwrap();

        assert_eq!(t.len(), 2, "both deliveries recorded");
        assert!(
            t.iter()
                .any(|d| d.to == aria && d.reply.is_some() && d.error.is_none()),
            "healthy agent's response persists"
        );
        assert!(
            t.iter()
                .any(|d| d.to == kai && d.reply.is_none() && d.error.is_some()),
            "failed agent marked with error"
        );
    }

    #[tokio::test]
    async fn run_processes_batch_in_deterministic_order() {
        // Pending envelopes processed in id-sorted order — stable across runs.
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let a = orch.register(make_agent("Aria", store.clone(), model.clone()));
        let b = orch.register(make_agent("Kai", store, model));
        orch.ask(Party::System, &a, "question");
        orch.ask(Party::System, &b, "question");
        let t = orch.run().await.unwrap();
        let mut ids: Vec<String> = vec![a.to_string(), b.to_string()];
        ids.sort();
        assert_eq!(t[0].to.to_string(), ids[0], "first delivery to smaller id");
        assert_eq!(t[1].to.to_string(), ids[1], "second delivery to larger id");
    }

    #[tokio::test]
    async fn agent_to_agent_ask_makes_both_remember() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let aria = orch.register(make_agent("Aria", store.clone(), model.clone()));
        let kai = orch.register(make_agent("Kai", store, model));

        orch.ask(Party::Agent(aria.clone()), &kai, "what is alpha protocol");
        let t = orch.run().await.unwrap();

        // Ask went to Kai (with reply), reply Tell returned to Aria.
        assert!(t
            .iter()
            .any(|d| d.to == kai && d.kind == MessageKind::Ask && d.reply.is_some()));
        assert!(t
            .iter()
            .any(|d| d.to == aria && d.kind == MessageKind::Tell));

        // Kai remembers responding, Aria remembers Kai's answer.
        let kai_mem = orch
            .agent(&kai)
            .unwrap()
            .recall(&Query::new("alpha"))
            .await
            .unwrap();
        let aria_mem = orch
            .agent(&aria)
            .unwrap()
            .recall(&Query::new("alpha"))
            .await
            .unwrap();
        assert!(!kai_mem.is_empty(), "Kai should remember");
        assert!(!aria_mem.is_empty(), "Aria should remember Kai's response");
    }

    #[tokio::test]
    async fn broadcast_reaches_others_not_sender() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let aria = orch.register(make_agent("Aria", store.clone(), model.clone()));
        let kai = orch.register(make_agent("Kai", store, model));

        orch.broadcast(Party::Agent(aria.clone()), "meeting in beta room");
        orch.run().await.unwrap();

        let kai_mem = orch
            .agent(&kai)
            .unwrap()
            .recall(&Query::new("beta"))
            .await
            .unwrap();
        let aria_mem = orch
            .agent(&aria)
            .unwrap()
            .recall(&Query::new("beta"))
            .await
            .unwrap();
        assert!(!kai_mem.is_empty(), "Kai should receive announcement");
        assert!(
            aria_mem.is_empty(),
            "Sender should not receive own announcement"
        );
    }

    #[tokio::test]
    async fn poll_collects_replies_from_all_agents() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let a = orch.register(make_agent("Aria", store.clone(), model.clone()));
        let b = orch.register(make_agent("Kai", store, model));
        let replies = orch.poll(Party::System, "status?").await.unwrap();
        assert_eq!(replies.len(), 2);
        let ids: Vec<_> = replies.iter().map(|(id, _)| id.clone()).collect();
        assert!(ids.contains(&a) && ids.contains(&b));
    }

    #[tokio::test]
    async fn poll_excludes_the_asker() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let a = orch.register(make_agent("Aria", store.clone(), model.clone()));
        orch.register(make_agent("Kai", store, model));
        let replies = orch.poll(Party::Agent(a.clone()), "status?").await.unwrap();
        assert_eq!(replies.len(), 1);
        assert!(replies.iter().all(|(id, _)| id != &a));
    }

    #[tokio::test]
    async fn blackboard_post_and_read() {
        let (store, model) = shared();
        let board: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let mut orch = Orchestrator::new().with_blackboard(board);
        orch.register(make_agent("Aria", store, model));
        orch.post_to_board(Party::System, "meeting plan alpha")
            .await
            .unwrap();
        let res = orch.read_board(&Query::new("alpha")).await.unwrap();
        assert_eq!(res.len(), 1);
    }

    #[tokio::test]
    async fn deliberate_posts_question_and_replies_to_board() {
        let (store, model) = shared();
        let board: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let mut orch = Orchestrator::new().with_blackboard(board);
        orch.register(make_agent("Aria", store.clone(), model.clone()));
        orch.register(make_agent("Kai", store, model));
        let replies = orch.deliberate("what is the plan").await.unwrap();
        assert_eq!(replies.len(), 2);
        // Board: 1 question + 2 replies = at least 3 records.
        let board_items = orch.read_board(&Query::new("").limit(100)).await.unwrap();
        assert!(board_items.len() >= 3);
    }

    #[tokio::test]
    async fn read_board_empty_without_blackboard() {
        let orch = Orchestrator::new();
        assert!(orch.read_board(&Query::new("x")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn tell_is_perceived_without_reply() {
        let (store, model) = shared();
        let mut orch = Orchestrator::new();
        let kai = orch.register(make_agent("Kai", store, model));

        orch.tell(Party::System, &kai, "gamma log updated");
        let t = orch.run().await.unwrap();

        assert_eq!(t.len(), 1);
        assert_eq!(t[0].kind, MessageKind::Tell);
        assert!(t[0].reply.is_none());
        let mem = orch
            .agent(&kai)
            .unwrap()
            .recall(&Query::new("gamma"))
            .await
            .unwrap();
        assert!(!mem.is_empty());
    }
}
