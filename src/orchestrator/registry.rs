//! `Registry`: AgentId → Agent lookup table.

use crate::agent::Agent;
use crate::id::AgentId;
use std::collections::HashMap;

/// Table holding registered agents.
#[derive(Default)]
pub struct Registry {
    agents: HashMap<AgentId, Agent>,
}

impl Registry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an agent, returns its identity.
    pub fn register(&mut self, agent: Agent) -> AgentId {
        let id = agent.id.clone();
        self.agents.insert(id.clone(), agent);
        id
    }

    /// Accesses an agent by identity.
    pub fn get(&self, id: &AgentId) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// Removes an agent from the registry.
    pub fn remove(&mut self, id: &AgentId) -> Option<Agent> {
        self.agents.remove(id)
    }

    /// All registered agent identities.
    pub fn ids(&self) -> Vec<AgentId> {
        self.agents.keys().cloned().collect()
    }

    /// Number of registered agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Whether this identity is registered.
    pub fn contains(&self, id: &AgentId) -> bool {
        self.agents.contains_key(id)
    }
}
