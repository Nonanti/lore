//! Native knowledge graph: connects memories via shared entities.
//!
//! Independent counterpart to Alaz's graph signal. Each memory enters an inverted
//! index keyed by its entities (5W cues + content tokens). Two memories sharing
//! an entity are neighbors. Multi-hop traversal (BFS) and shortest path (path)
//! allow extracting multi-step relationships not contained in any single record.

use super::retrieval::tokenize;
use super::types::{Memory, MemoryKind, Query, Scope};
use super::MemoryStore;
use crate::error::Result;
use crate::id::MemoryId;
use std::collections::{HashMap, HashSet, VecDeque};

/// Common function words filtered out to prevent over-connectivity (len>=4).
const STOPWORDS: &[&str] = &[
    "its", "like", "more", "as", "after", "before", "but", "or", "not", "that", "was", "one",
    "this", "that2", "with", "both", "the", "and", "for", "with", "that",
];

/// Extracts entities from a memory: 5W cues + content tokens (len>=4).
/// `pub(crate)`: the stores maintain incremental entity indexes with the
/// SAME rules (recall's graph leg and this analysis graph cannot drift).
pub(crate) fn extract_entities(mem: &Memory) -> HashSet<String> {
    let mut set = HashSet::new();

    // Episodic 5W cues are a strong entity source (no length requirement).
    if let MemoryKind::Episodic { cues, .. } = &mem.kind {
        for group in [&cues.who, &cues.what, &cues.where_, &cues.when, &cues.why] {
            for c in group {
                for t in tokenize(c) {
                    set.insert(t);
                }
            }
        }
    }

    // Content tokens (meaningful words).
    let text = mem.searchable_text();
    for t in tokenize(&text) {
        if t.chars().count() >= 4 && !STOPWORDS.contains(&t.as_str()) {
            set.insert(t);
        }
    }
    // Short ALL-CAPS acronyms (NAS, TLS, GPU, 8TB…) are high-value entities
    // the ≥4-char floor would drop — they are exactly the tokens that bridge
    // technical records. Detected on the RAW text (case carries the signal),
    // stored lowercased like every other entity.
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        let n = raw.chars().count();
        if (2..=4).contains(&n)
            && raw
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && raw.chars().filter(|c| c.is_ascii_uppercase()).count() >= 2
        {
            set.insert(raw.to_lowercase());
        }
    }
    set
}

/// Graph connecting memory nodes via entity sharing.
#[derive(Debug, Default)]
pub struct MemoryGraph {
    entity_index: HashMap<String, HashSet<MemoryId>>,
    memory_entities: HashMap<MemoryId, HashSet<String>>,
}

impl MemoryGraph {
    /// Builds a graph from a list of records (soft-deleted ones are skipped).
    pub fn build(mems: &[Memory]) -> Self {
        let mut entity_index: HashMap<String, HashSet<MemoryId>> = HashMap::new();
        let mut memory_entities: HashMap<MemoryId, HashSet<String>> = HashMap::new();

        for m in mems {
            if m.deleted_at.is_some() {
                continue;
            }
            let ents = extract_entities(m);
            for e in &ents {
                entity_index
                    .entry(e.clone())
                    .or_default()
                    .insert(m.id.clone());
            }
            memory_entities.insert(m.id.clone(), ents);
        }

        Self {
            entity_index,
            memory_entities,
        }
    }

    /// Builds a graph from the live records in a store's given scope.
    ///
    /// Note: loads ALL records in the scope into memory — intended for
    /// analysis/exploration, not for the hot request path.
    pub async fn from_store(store: &dyn MemoryStore, scope: &Scope) -> Result<Self> {
        let scored = store
            .recall(scope, &Query::new("").limit(usize::MAX))
            .await?;
        let mems: Vec<Memory> = scored.into_iter().map(|s| s.item).collect();
        Ok(Self::build(&mems))
    }

    /// Node (memory) count.
    pub fn node_count(&self) -> usize {
        self.memory_entities.len()
    }

    /// Unique entity count.
    pub fn entity_count(&self) -> usize {
        self.entity_index.len()
    }

    /// Entities of a record.
    pub fn entities_of(&self, id: &MemoryId) -> Option<&HashSet<String>> {
        self.memory_entities.get(id)
    }

    /// Direct neighbors: sorted by descending shared entity count.
    pub fn neighbors(&self, id: &MemoryId) -> Vec<(MemoryId, usize)> {
        let mut counts: HashMap<MemoryId, usize> = HashMap::new();
        if let Some(ents) = self.memory_entities.get(id) {
            for e in ents {
                if let Some(ids) = self.entity_index.get(e) {
                    for other in ids {
                        if other != id {
                            *counts.entry(other.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let mut v: Vec<(MemoryId, usize)> = counts.into_iter().collect();
        v.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| a.0.to_string().cmp(&b.0.to_string()))
        });
        v
    }

    /// BFS up to `depth` hops (excluding source, up to `limit`).
    /// Neighbor lists are memoized during traversal — `neighbors` allocs and
    /// sorts on each call; recomputing the same node is expensive on large graphs.
    pub fn related(&self, id: &MemoryId, depth: usize, limit: usize) -> Vec<MemoryId> {
        let mut nb_cache: HashMap<MemoryId, Vec<(MemoryId, usize)>> = HashMap::new();
        let mut visited = HashSet::new();
        visited.insert(id.clone());
        let mut frontier = vec![id.clone()];
        let mut out = Vec::new();

        for _ in 0..depth {
            let mut next = Vec::new();
            for node in &frontier {
                let nbs = nb_cache
                    .entry(node.clone())
                    .or_insert_with(|| self.neighbors(node));
                for (nb, _) in nbs.clone() {
                    if visited.insert(nb.clone()) {
                        out.push(nb.clone());
                        next.push(nb);
                        if out.len() >= limit {
                            return out;
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        out
    }

    /// Shortest path between two records via entity sharing (record chain).
    /// Reveals multi-step relationships ("how is A connected to C?").
    pub fn path(&self, from: &MemoryId, to: &MemoryId) -> Option<Vec<MemoryId>> {
        if from == to {
            return Some(vec![from.clone()]);
        }
        let mut nb_cache: HashMap<MemoryId, Vec<(MemoryId, usize)>> = HashMap::new();
        let mut visited = HashSet::new();
        visited.insert(from.clone());
        let mut queue = VecDeque::new();
        queue.push_back(from.clone());
        let mut parent: HashMap<MemoryId, MemoryId> = HashMap::new();

        while let Some(node) = queue.pop_front() {
            let nbs = nb_cache
                .entry(node.clone())
                .or_insert_with(|| self.neighbors(&node))
                .clone();
            for (nb, _) in nbs {
                if visited.insert(nb.clone()) {
                    parent.insert(nb.clone(), node.clone());
                    if &nb == to {
                        let mut path = vec![to.clone()];
                        let mut cur = to.clone();
                        while &cur != from {
                            let p = parent.get(&cur).expect("parent chain").clone();
                            path.push(p.clone());
                            cur = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(nb);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::SemanticCat;

    fn sem(text: &str) -> Memory {
        Memory::semantic(Scope::World, text, SemanticCat::Fact)
    }

    #[test]
    fn neighbors_share_entities() {
        let a = sem("aylin codes in rust");
        let b = sem("rust uses tokio");
        let (ida, idb) = (a.id.clone(), b.id.clone());
        let g = MemoryGraph::build(&[a, b]);

        let nb = g.neighbors(&ida);
        assert!(
            nb.iter().any(|(id, c)| id == &idb && *c >= 1),
            "rust should be shared"
        );
    }

    #[test]
    fn multi_hop_path_and_related() {
        // A—(rust)—B—(tokio)—C ; A and C share no entity.
        let a = sem("aylin codes in rust");
        let b = sem("rust uses tokio");
        let c = sem("tokio runs async");
        let (ida, idb, idc) = (a.id.clone(), b.id.clone(), c.id.clone());
        let g = MemoryGraph::build(&[a, b, c]);

        // A and C are not directly connected.
        assert!(!g.neighbors(&ida).iter().any(|(id, _)| id == &idc));

        // But reachable in 2 hops.
        let rel = g.related(&ida, 2, 10);
        assert!(rel.contains(&idc), "C reachable from A in 2 hops");

        // Shortest path A→B→C.
        let path = g.path(&ida, &idc).expect("path should exist");
        assert_eq!(path, vec![ida, idb, idc]);
    }

    #[tokio::test]
    async fn build_from_store_works() {
        use crate::memory::InMemoryStore;
        let store = InMemoryStore::new();
        store.remember(sem("aylin codes in rust")).await.unwrap();
        store.remember(sem("rust uses tokio")).await.unwrap();

        let g = MemoryGraph::from_store(&store, &Scope::World)
            .await
            .unwrap();
        assert_eq!(g.node_count(), 2);
        assert!(g.entity_count() >= 3); // aylin, codes, rust, uses, tokio
    }
}
