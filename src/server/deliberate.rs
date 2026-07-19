//! Collective deliberation: local team poll, supervisor synthesis, and federation fanout.
//!
//! Flow: question lands on the board → local team responds in parallel → (if any) peer nodes
//! respond in parallel → (if any) supervisor synthesizes all responses.

use super::state::AppState;
use super::types::{DeliberateReply, DeliberateResp};
use crate::agent::Agent;
use crate::error::{LoreError, Result};
use crate::id::AgentId;
use futures::StreamExt;

/// Peer node response body upper bound — prevents a compromised/faulty peer from
/// inflating memory with a huge body (normal deliberate replies are in the KB range).
const PEER_MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

impl AppState {
    /// Polls the local team (optional `skip`: agent excluded from the poll — e.g. supervisor).
    /// Replies are collected in parallel (`join_all` — input order preserved, deterministic);
    /// a single agent failure does not kill the poll: a warning is logged and the error is skipped.
    async fn poll_team(
        &self,
        question: &str,
        skip: Option<&AgentId>,
    ) -> Result<Vec<DeliberateReply>> {
        let team: Vec<(AgentId, Agent)> = self
            .team()
            .await
            .into_iter()
            .filter(|(id, _)| Some(id) != skip)
            .collect();
        let futs = team.iter().map(|(id, agent)| async move {
            (
                id.clone(),
                agent.persona.name.clone(),
                agent.respond(question).await,
            )
        });
        let mut out = Vec::new();
        for (id, name, res) in futures::future::join_all(futs).await {
            match res {
                Ok(reply) => {
                    // Board write failure MUST NOT lose collected replies — logged and
                    // skipped like agent errors (consistent with peer_fanout).
                    if let Err(e) = self
                        .board_note(format!("{name} response"), reply.clone())
                        .await
                    {
                        tracing::warn!(error = %e, "deliberate: board note could not be written");
                    }
                    out.push(DeliberateReply {
                        id: id.to_string(),
                        name,
                        reply,
                        node: None,
                    });
                }
                Err(e) => tracing::warn!(
                    agent = %super::log_safe(&name),
                    error = %e,
                    "deliberate: no response"
                ),
            }
        }
        Ok(out)
    }

    /// Polls peer nodes (in parallel; node label + local board note; dead peers are
    /// warned and skipped — resilience). Also used by WebSocket live deliberate.
    pub(crate) async fn peer_fanout(&self, question: &str) -> Vec<DeliberateReply> {
        let futs = self
            .peers
            .iter()
            .map(|peer| async move { (peer.clone(), self.ask_peer(peer, question).await) });
        let mut out = Vec::new();
        for (peer, res) in futures::future::join_all(futs).await {
            match res {
                Ok(mut replies) => {
                    for r in &mut replies {
                        r.node = Some(peer.clone());
                        if let Err(e) = self
                            .board_note(format!("{}@{} response", r.name, peer), r.reply.clone())
                            .await
                        {
                            tracing::warn!(error = %e, "federation: board note could not be written");
                        }
                    }
                    out.append(&mut replies);
                }
                Err(e) => {
                    tracing::warn!(peer = %peer, error = %e, "federation: peer did not respond")
                }
            }
        }
        out
    }

    /// Local collective deliberation: writes the question to the board (World), polls the
    /// local team (deterministic order), writes each reply to the board, and returns the replies.
    pub async fn deliberate_local(&self, question: &str) -> Result<Vec<DeliberateReply>> {
        self.board_note("Question", question).await?;
        self.poll_team(question, None).await
    }

    /// Federated collective deliberation: local team + all peer nodes. Sends `local:true`
    /// to peers (breaks the loop); unreachable peers are warned and skipped (resilience).
    pub async fn deliberate(&self, question: &str) -> Result<Vec<DeliberateReply>> {
        let mut out = self.deliberate_local(question).await?;
        out.extend(self.peer_fanout(question).await);
        Ok(out)
    }

    /// Hierarchical team: the supervisor does not participate in the poll; takes local + peer
    /// replies as additional context and produces the final synthesis. The synthesis lands on
    /// the board as "Synthesis". `local:true` also skips peer fanout here — the `local` contract
    /// remains valid even with a synthesizer (depth-1 guarantee stays consistent).
    pub async fn deliberate_synth(
        &self,
        question: &str,
        synthesizer: &AgentId,
        local: bool,
    ) -> Result<(Vec<DeliberateReply>, String)> {
        let synth_agent = self.get_agent(synthesizer).await?; // validate first
        self.board_note("Question", question).await?;
        let mut replies = self.poll_team(question, Some(synthesizer)).await?;
        if !local {
            replies.extend(self.peer_fanout(question).await);
        }

        let lines: Vec<String> = replies
            .iter()
            .map(|r| match &r.node {
                Some(n) => format!("{}@{}: {}", r.name, n, r.reply),
                None => format!("{}: {}", r.name, r.reply),
            })
            .collect();
        let synthesis = synth_agent
            .respond_with(
                &format!("Synthesize the team's responses and give the final decision: {question}"),
                &lines,
            )
            .await?;
        self.board_note("Synthesis", synthesis.clone()).await?;
        Ok((replies, synthesis))
    }

    /// Forwards the question to a single peer node and collects its replies (shared client).
    /// The body is read with a limit ([`PEER_MAX_BODY_BYTES`]) and the status code is checked —
    /// a faulty peer is logged with a clear diagnosis, not silently turned into a JSON error.
    async fn ask_peer(&self, peer: &str, question: &str) -> Result<Vec<DeliberateReply>> {
        let mut req = self
            .http
            .post(format!("{peer}/deliberate"))
            .json(&serde_json::json!({ "question": question, "local": true }));
        if let Some(k) = &self.peer_key {
            req = req.header("x-api-key", k);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(LoreError::Server(format!("peer returned {status}")));
        }
        let mut body = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            if buf.len() + chunk.len() > PEER_MAX_BODY_BYTES {
                return Err(LoreError::Server(format!(
                    "peer response too large (> {PEER_MAX_BODY_BYTES} bytes)"
                )));
            }
            buf.extend_from_slice(&chunk);
        }
        let resp: DeliberateResp = serde_json::from_slice(&buf)?;
        Ok(resp.replies)
    }
}
