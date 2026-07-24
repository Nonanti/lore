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

/// Neural cross-encoder reranker (fastembed/ONNX; compiled only with the
/// `neural` feature). Scores query×document PAIRS jointly — true relevance
/// ordering the bi-encoder first pass cannot see. Downloads the model on
/// first use (then cached). Default model: BGE reranker base;
/// [`NeuralReranker::with_model`] selects others (e.g. the multilingual
/// Jina v2 for Turkish-heavy corpora).
#[cfg(feature = "neural")]
pub struct NeuralReranker {
    model: std::sync::Mutex<fastembed::TextRerank>,
    /// Warn-once latch: a permanently broken model must not spam a warn
    /// per recall on the hot path (review #10).
    warned: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "neural")]
impl NeuralReranker {
    /// Initializes with the default model (BGE reranker base).
    pub fn new() -> crate::error::Result<Self> {
        Self::with_model(fastembed::RerankerModel::BGERerankerBase)
    }

    /// Initializes with a specific fastembed reranker model.
    pub fn with_model(model: fastembed::RerankerModel) -> crate::error::Result<Self> {
        use fastembed::{RerankInitOptions, TextRerank};
        let m =
            TextRerank::try_new(RerankInitOptions::new(model).with_show_download_progress(false))
                .map_err(|e| crate::error::LoreError::Model(e.to_string()))?;
        Ok(Self {
            model: std::sync::Mutex::new(m),
            warned: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

#[cfg(feature = "neural")]
impl Reranker for NeuralReranker {
    fn rerank(&self, query: &str, items: Vec<Scored<Memory>>) -> Vec<Scored<Memory>> {
        if items.is_empty() || query.trim().is_empty() {
            return items;
        }
        let docs: Vec<String> = items.iter().map(|s| s.item.searchable_text()).collect();
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        // Poison recovery mirrors NeuralEmbedder: a panicked past call must
        // not brick the reranker.
        let m = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let results = match m.rerank(query, doc_refs, false, None) {
            Ok(r) => r,
            Err(e) => {
                // Fail open: first-pass order is a valid answer; losing the
                // whole recall to a rerank hiccup is not. Warn once, then
                // drop to debug — no log storms on the recall hot path.
                if !self.warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(error = %e, "neural rerank error (keeping first-pass order; further errors logged at debug)");
                } else {
                    tracing::debug!(error = %e, "neural rerank error (keeping first-pass order)");
                }
                return items;
            }
        };
        // fastembed returns (index, score) sorted by score — rebuild in that
        // order, annotate the signal, keep the cross-encoder score.
        let mut out: Vec<Scored<Memory>> = Vec::with_capacity(items.len());
        let mut taken: Vec<Option<Scored<Memory>>> = items.into_iter().map(Some).collect();
        for r in results {
            if let Some(slot) = taken.get_mut(r.index) {
                if let Some(mut s) = slot.take() {
                    s.score = r.score;
                    s.signals.push(Signal {
                        name: "neural_rerank".into(),
                        value: r.score,
                    });
                    out.push(s);
                }
            }
        }
        // Defensive: anything the model did not score keeps its old position
        // at the tail (should not happen, but records must never vanish).
        for slot in taken.into_iter().flatten() {
            out.push(slot);
        }
        out
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
