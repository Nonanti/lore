//! Rerank: re-scores first-pass candidates via query×doc shared features.
//!
//! A real cross-encoder requires a neural model; here we have a standalone/offline
//! native reranker that refines the first-pass score with coverage, exact-phrase,
//! and ordered-bigram signals. An LLM/cross-encoder reranker can later plug behind
//! the same `Reranker` trait.

use super::retrieval::tokenize;
use super::types::{Memory, Scored, Signal};
use std::collections::HashSet;

/// Abstraction that re-ranks a candidate list.
pub trait Reranker: Send + Sync {
    /// Re-ranks candidates according to the query.
    fn rerank(&self, query: &str, items: Vec<Scored<Memory>>) -> Vec<Scored<Memory>>;
}

/// Native reranker using query×doc shared lexical features.
#[derive(Clone, Debug, Default)]
pub struct NativeReranker;

impl Reranker for NativeReranker {
    fn rerank(&self, query: &str, items: Vec<Scored<Memory>>) -> Vec<Scored<Memory>> {
        native_rerank(query, items)
    }
}

struct CrossFeatures {
    coverage: f32,
    phrase: f32,
    bigram: f32,
}

fn cross_features(q_terms: &[String], q_lower: &str, mem: &Memory) -> CrossFeatures {
    if q_terms.is_empty() {
        return CrossFeatures {
            coverage: 0.0,
            phrase: 0.0,
            bigram: 0.0,
        };
    }
    let doc_lower = mem.searchable_text().to_lowercase();
    let doc_terms = tokenize(&doc_lower);
    let doc_set: HashSet<&str> = doc_terms.iter().map(|s| s.as_str()).collect();

    // 1) Term coverage.
    let covered = q_terms
        .iter()
        .filter(|t| doc_set.contains(t.as_str()))
        .count();
    let coverage = covered as f32 / q_terms.len() as f32;

    // 2) Exact phrase (substring) bonus.
    let phrase = if !q_lower.trim().is_empty() && doc_lower.contains(q_lower.trim()) {
        1.0
    } else {
        0.0
    };

    // 3) Ordered bigram overlap (word order).
    let q_bigrams: Vec<(&str, &str)> = q_terms
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();
    let bigram = if q_bigrams.is_empty() {
        // Single-term query: no bigram signal, reflect coverage.
        coverage
    } else {
        let mut hits = 0;
        for bg in &q_bigrams {
            if doc_terms.windows(2).any(|w| w[0] == bg.0 && w[1] == bg.1) {
                hits += 1;
            }
        }
        hits as f32 / q_bigrams.len() as f32
    };

    CrossFeatures {
        coverage,
        phrase,
        bigram,
    }
}

/// Native rerank: blends first-pass score with shared features and re-sorts.
pub fn native_rerank(query: &str, mut items: Vec<Scored<Memory>>) -> Vec<Scored<Memory>> {
    if items.len() <= 1 {
        return items;
    }
    let max_base = items
        .iter()
        .map(|s| s.score)
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let q_lower = query.to_lowercase();
    let q_terms = tokenize(&q_lower);

    for it in &mut items {
        let base_norm = it.score / max_base;
        let f = cross_features(&q_terms, &q_lower, &it.item);
        let r = 0.45 * base_norm + 0.25 * f.coverage + 0.20 * f.phrase + 0.10 * f.bigram;
        // First-pass score is preserved as a signal before being overwritten (explainability).
        it.signals.push(Signal {
            name: "base".into(),
            value: it.score,
        });
        it.score = r;
        it.signals.push(Signal {
            name: "rerank".into(),
            value: r,
        });
    }
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{Memory, Scope, SemanticCat};

    fn scored(text: &str, base: f32) -> Scored<Memory> {
        Scored {
            item: Memory::semantic(Scope::World, text, SemanticCat::Fact),
            score: base,
            signals: vec![],
        }
    }

    #[test]
    fn rerank_promotes_exact_phrase_over_higher_base() {
        // A: lower base but exact phrase match; B: higher base but partial.
        let a = scored("alfa beta harika bir konu", 0.50);
        let b = scored("beta gama delta", 0.90);
        let out = NativeReranker.rerank("alfa beta", vec![b, a]);
        assert!(
            out[0].item.summary().contains("alfa beta"),
            "full phrase + scope should beat high base"
        );
    }

    #[test]
    fn single_item_unchanged() {
        let out = native_rerank("x", vec![scored("tek", 0.3)]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn rerank_preserves_base_score_as_signal() {
        let out = native_rerank("alfa", vec![scored("alfa bir", 0.5), scored("beta", 0.9)]);
        for it in &out {
            assert!(
                it.signals.iter().any(|s| s.name == "base"),
                "first-pass score preserved on signal: {:?}",
                it.signals
            );
        }
    }
}
