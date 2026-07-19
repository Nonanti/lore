//! Retrieval QUALITY harness: accuracy, not speed.
//!
//! Benchmarks measure latency; this suite measures hit@5 on a golden set. Semantic
//! gate, fusion weights, embedder, or token-level fallback calibration
//! changes — this catches quality regressions that unit tests
//! cannot see.
//!
//! Thresholds are kept BELOW current behavior (regression alarm, not a target);
//! if quality improves, thresholds are ONLY updated upward.

use lore::memory::HashingEmbedder;
use lore::{InMemoryStore, Memory, MemoryStore, Query, Scope, SemanticCat};
use std::sync::Arc;

/// Golden corpus: (record text). Turkish-weighted, mimicking real usage.
const CORPUS: &[&str] = &[
    "User started learning Rust programming language, studying ownership",
    "Their cat Paspas went to the vet yesterday, got vaccinated",
    "Project deadline set for next Friday",
    "Favorite food is dumplings, especially Kayseri style",
    "Over the weekend we visited Galata Tower in Istanbul",
    "We use tokio runtime in async Rust, opening tasks with spawn",
    "Talks to their mom on the phone every Sunday",
    "We set up a volume backup strategy for Docker containers",
    "We applied Newton's second law during physics exercise",
    "Arch Linux was installed on the new work laptop, hyprland configured",
    "User drinks coffee without sugar, plain",
    "Derivatives and integrals will be on the math exam",
    "Tomato and pepper seedlings were planted in the garden",
    "Remote work policy was updated at the company meeting",
    "The bike's rear derailleur was replaced with Shimano Deore",
];

/// Golden queries: (query, expected corpus index, note).
/// Covers morphology, single-token, multi-word, and synonym scenarios.
const QUERIES: &[(&str, usize, &str)] = &[
    ("rust learning", 0, "exact match"),
    (
        "learning",
        0,
        "morphology: learning/learn (single token fallback)",
    ),
    ("vet", 1, "single token"),
    ("cat vaccine", 1, "multi-word, partial coverage"),
    ("deadline when", 2, "paraphrase context"),
    ("dumplings", 3, "single token exact match"),
    ("galata", 4, "proper noun"),
    ("tokio spawn", 5, "technical terms"),
    ("phone", 6, "single token"),
    ("backup", 7, "single token"),
    ("coffee", 10, "single token"),
    ("mathematics", 11, "single token"),
    ("derivatives", 11, "subtopic"),
    ("tomato", 12, "garden"),
    ("derailleur", 14, "bike part"),
];

#[tokio::test]
async fn retrieval_golden_set_hit_at_5() {
    let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
    for text in CORPUS {
        store
            .remember(Memory::semantic(Scope::World, *text, SemanticCat::Fact))
            .await
            .unwrap();
    }

    let mut hits = 0;
    let mut misses = Vec::new();
    for (q, want, note) in QUERIES {
        let res = store
            .recall(&Scope::World, &Query::new(*q).semantic().limit(5))
            .await
            .unwrap();
        let want_text = CORPUS[*want];
        let found = res
            .iter()
            .any(|s| s.item.searchable_text().contains(want_text));
        if found {
            hits += 1;
        } else {
            let got: Vec<String> = res
                .iter()
                .take(3)
                .map(|s| s.item.searchable_text().chars().take(40).collect())
                .collect();
            misses.push(format!("'{q}' ({note}) → expected #{want}, got: {got:?}"));
        }
    }

    let total = QUERIES.len();
    let rate = hits as f64 / total as f64;
    // Regression threshold: current quality is above this; alarm if it drops.
    assert!(
        rate >= 0.80,
        "hit@5 = {hits}/{total} PLACEHOLDER — misses:\n{}",
        misses.join("\n")
    );
    // Report (visible in test output — raise the threshold when quality improves).
    if !misses.is_empty() {
        eprintln!("[eval] missed queries:\n{}", misses.join("\n"));
    }
    eprintln!("[eval] hit@5 = {hits}/{total} ({:.0}%)", rate * 100.0);
}

/// Keyword-only mode: basic matches should remain solid even with semantic off.
#[tokio::test]
async fn retrieval_keyword_only_baseline() {
    let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
    for text in CORPUS {
        store
            .remember(Memory::semantic(Scope::World, *text, SemanticCat::Fact))
            .await
            .unwrap();
    }
    // Exact-token-match queries should also be found with keyword-only.
    // (Deliberately queries that do NOT require morphology: "veteriner"↛"veterinere"
    // variants like that are semantic's job, not keyword-only's.)
    for (q, want) in [
        ("dumplings", 3usize),
        ("galata", 4),
        ("tomato", 12),
        ("shimano", 14),
    ] {
        let res = store
            .recall(&Scope::World, &Query::new(q).limit(5))
            .await
            .unwrap();
        assert!(
            res.iter()
                .any(|s| s.item.searchable_text().contains(CORPUS[want])),
            "keyword baseline missed: '{q}'"
        );
    }
}
