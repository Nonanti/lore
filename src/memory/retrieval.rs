//! Phase 1 retrieval: keyword + recency + importance + Wilson scoring.
//!
//! By design, "start flat": a single combined score. This module will later be
//! extended with hybrid (BM25 + vector), HyDE and rerank signals — the `Signal`
//! list returned by `score` is ready for that expansion.

use super::embed::Embedder;
use super::types::{Memory, MemoryKind, Query, Scored, Signal};
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};

/// Lowercases the text and splits on non-alphanumeric characters; keeps tokens
/// with >=2 chars. Non-ASCII Unicode letters are preserved via
/// `is_alphanumeric`.
pub fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(|t| t.to_string())
        .collect()
}

/// Wilson score lower bound (95% confidence, z=1.96).
///
/// Penalizes procedural memory reliability by sample size:
/// 1/1 success is less trustworthy than 100/100 successes.
pub fn wilson_lower_bound(successes: u32, failures: u32) -> f64 {
    let n = (successes + failures) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let z = 1.96_f64;
    let phat = successes as f64 / n;
    let z2 = z * z;
    (phat + z2 / (2.0 * n) - z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt())
        / (1.0 + z2 / n)
}

/// Cosine similarity between two vectors (returns 0 for zero-length/mismatched).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Recency score: exponential decay with a 7-day half-life (Ebbinghaus-style).
/// age=0 → ~1.0; approaches 0 as age increases.
pub fn recency_score(age_seconds: f64) -> f32 {
    let half_life = 7.0 * 24.0 * 3600.0; // 7 days
    let lambda = std::f64::consts::LN_2 / half_life;
    (-lambda * age_seconds.max(0.0)).exp() as f32
}

/// Keyword score (0..1): coverage of query terms in the document, with a mild
/// tf boost. Coverage is the dominant signal; term frequency is secondary.
pub fn keyword_score(query_terms: &[String], doc_text: &str) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let doc_terms = tokenize(doc_text);
    if doc_terms.is_empty() {
        return 0.0;
    }

    let mut tf: HashMap<&str, u32> = HashMap::new();
    for t in &doc_terms {
        *tf.entry(t.as_str()).or_insert(0) += 1;
    }

    let unique_q: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();
    let mut found = 0u32;
    let mut tf_sum = 0f32;
    for q in &unique_q {
        if let Some(&c) = tf.get(*q) {
            found += 1;
            tf_sum += (c as f32).ln_1p(); // ln(1 + tf)
        }
    }
    if found == 0 {
        return 0.0;
    }

    let coverage = found as f32 / unique_q.len() as f32; // 0..1
    let avg_tf = tf_sum / found as f32;
    let tf_norm = avg_tf / (avg_tf + 1.0); // 0..1 saturation
    coverage * (0.7 + 0.3 * tf_norm)
}

/// Semantic candidate gate: when keywords are absent, a cosine above this
/// threshold qualifies. Calibrated for `HashingEmbedder`; for
/// embedder-specific thresholds use
/// [`super::embed::Embedder::semantic_gate`].
pub const SEMANTIC_GATE: f32 = 0.40;

// --- Score fusion calibration ---
// Relevance: keyword-dominant (coverage is the most reliable signal), cosine as support.
// Boost: recency > wilson > importance — fresh and proven knowledge comes first;
// importance is a fragile user signal, so it gets the lowest weight.
// In browse (textless) mode there is no relevance: freshness + importance + confidence rank.

/// Keyword weight in relevance fusion.
const W_KEYWORD: f32 = 0.6;
/// Cosine weight in relevance fusion.
const W_COSINE: f32 = 0.4;
/// Boost: recency coefficient.
const B_RECENCY: f32 = 0.25;
/// Boost: importance coefficient.
const B_IMPORTANCE: f32 = 0.15;
/// Boost: Wilson (procedural confidence) coefficient.
const B_WILSON: f32 = 0.30;
/// Browse mode weights (recency, importance, wilson).
const BROWSE_W: (f32, f32, f32) = (0.6, 0.4, 0.2);

/// Process-wide retrieval counters (metrics feed). Global atomic: metrics are
/// inherently process-wide; no lock, cost is a single fetch_add on the hot path.
pub mod stats {
    use std::sync::atomic::AtomicU64;

    /// Number of records that passed the score threshold (candidates).
    pub static RECALL_CANDIDATES: AtomicU64 = AtomicU64::new(0);
    /// Number of matches produced by the token-level fallback (short-query morphology).
    pub static TOKEN_FALLBACK_HITS: AtomicU64 = AtomicU64::new(0);
}

/// Default conflict band (`HashingEmbedder` calibration).
pub const CONFLICT_BAND: (f32, f32) = (0.6, 0.9);

// --- Graph expansion leg (Query.graph) ---

/// How many top first-pass candidates seed the entity-neighbor expansion.
pub const GRAPH_SEED_K: usize = 3;
/// Max neighbors pulled per expansion (before filtering).
pub const GRAPH_NEIGHBOR_CAP: usize = 16;
/// Damping applied to the best seed score when scoring a pulled neighbor —
/// a hop is supporting evidence, never stronger than its source.
pub const GRAPH_DAMP: f32 = 0.5;

/// Appends graph-pulled neighbors to a first-pass candidate list with the
/// damped seed score and a `graph` signal. `hits` need not be pre-sorted;
/// the caller runs [`finalize`] afterwards (which sorts). Neighbors are
/// assumed pre-filtered (visibility/tier/importance) and deduplicated
/// against `hits` by the caller.
pub fn append_graph_neighbors(
    hits: &mut Vec<Scored<Memory>>,
    neighbors: Vec<Memory>,
    best_seed_score: f32,
) {
    let damped = best_seed_score * GRAPH_DAMP;
    for mem in neighbors {
        hits.push(Scored {
            item: mem,
            score: damped,
            signals: vec![Signal {
                name: "graph".into(),
                value: damped,
            }],
        });
    }
}

/// Whether two embeddings fall in the conflict range (default band): similar
/// topic, likely different information. Used for write-time conflict detection
/// (too low=unrelated, too high=duplicate).
pub fn is_conflict(a: &[f32], b: &[f32]) -> bool {
    is_conflict_in(a, b, CONFLICT_BAND)
}

/// Like `is_conflict`; takes band (lower, upper) as parameter
/// (embedder-specific calibration).
pub fn is_conflict_in(a: &[f32], b: &[f32], band: (f32, f32)) -> bool {
    let c = cosine(a, b);
    (band.0..band.1).contains(&c)
}

/// Scores a record against a query; returns the score + contributing signals.
///
/// Hybrid: keyword + (if available) cosine fusion, topped with
/// recency/importance/wilson boost. Default is keyword-gated (kw=0 →
/// eliminated); if `query.semantic` is on, records whose cosine passes
/// `SEMANTIC_GATE` become candidates even without keywords
/// (morphology/synonymy). In textless "browse" mode, recency + importance
/// (+ wilson) rank.
pub fn score(
    mem: &Memory,
    query: &Query,
    q_emb: Option<&[f32]>,
    now: DateTime<Utc>,
) -> (f32, Vec<Signal>) {
    score_impl(mem, query, q_emb, now, SEMANTIC_GATE, None, None)
}

/// Like `score`; takes the semantic candidate gate as a parameter
/// (embedder-specific calibration — see [`super::embed::Embedder::semantic_gate`]).
#[deprecated(
    since = "0.1.0",
    note = "skips token-level fallback — use `Scorer` or `score_with_embedder`"
)]
pub fn score_with_gate(
    mem: &Memory,
    query: &Query,
    q_emb: Option<&[f32]>,
    now: DateTime<Utc>,
    gate: f32,
) -> (f32, Vec<Signal>) {
    score_impl(mem, query, q_emb, now, gate, None, None)
}

/// Like `score`; takes the gate AND token-level fallback, including all
/// calibration, from the embedder. Intended for one-shot scoring — if many
/// records will be scored across a scan, prefer [`Scorer`] which shares a
/// token cache.
pub fn score_with_embedder(
    mem: &Memory,
    query: &Query,
    q_emb: Option<&[f32]>,
    now: DateTime<Utc>,
    embedder: Option<&dyn Embedder>,
) -> (f32, Vec<Signal>) {
    Scorer::new(embedder).score(mem, query, q_emb, now)
}

/// A scorer that lives across a recall scan: embedder calibration +
/// query-lifetime token embedding cache for token-level fallback. In natural
/// language, tokens are heavily repeated across candidates — the cache avoids
/// embedding the same token twice (significant savings in large stores).
pub struct Scorer<'a> {
    embedder: Option<&'a dyn Embedder>,
    gate: f32,
    token_fb: bool,
    cache: HashMap<String, Vec<f32>>,
}

impl<'a> Scorer<'a> {
    /// New scorer with embedder calibration (defaults if no embedder).
    pub fn new(embedder: Option<&'a dyn Embedder>) -> Self {
        Self {
            embedder,
            gate: embedder.map(|e| e.semantic_gate()).unwrap_or(SEMANTIC_GATE),
            token_fb: embedder.is_some_and(|e| e.token_fallback()),
            cache: HashMap::new(),
        }
    }

    /// Scores a record — see [`score`] (same contract + token-level fallback).
    pub fn score(
        &mut self,
        mem: &Memory,
        query: &Query,
        q_emb: Option<&[f32]>,
        now: DateTime<Utc>,
    ) -> (f32, Vec<Signal>) {
        let (fb, cache) = if self.token_fb {
            (self.embedder, Some(&mut self.cache))
        } else {
            (None, None)
        };
        score_impl(mem, query, q_emb, now, self.gate, fb, cache)
    }
}

/// Token-level fallback: for short queries, full-document cosine is diluted by
/// the document's other tokens (dilution — see TEST_REPORT §5.3).
/// The query vector is compared against each INDIVIDUAL token of the document,
/// taking the highest cosine (a one-sided variant of ColBERT-style late
/// interaction). Cost limit: only short queries
/// (≤ [`TOKEN_FALLBACK_MAX_QUERY`]), at most [`TOKEN_FALLBACK_MAX_DOC`]
/// unique document tokens, and bounded cache ([`TOKEN_CACHE_MAX`] — if
/// exceeded, embeds without caching).
const TOKEN_FALLBACK_MAX_QUERY: usize = 2;
const TOKEN_FALLBACK_MAX_DOC: usize = 128;
const TOKEN_CACHE_MAX: usize = 4096;

/// Pre-filter predicate for the SQL lightweight scan: computes the SUPERSET of
/// scoring's candidate condition (kw>0 ∨ semantic gate) **without loading full
/// rows**. Uses the same gate + same token fallback as `score_impl` — the two
/// logics MUST NOT drift from each other (parity test:
/// `sqlite_recall_matches_in_memory_reference`).
/// `search_text` is a pre-normalized token sequence (space-separated).
pub(crate) fn semantic_prefilter_hit(
    q_emb: Option<&[f32]>,
    emb: Option<&[f32]>,
    search_text: &str,
    q_terms_len: usize,
    embedder: Option<&dyn Embedder>,
    cache: &mut HashMap<String, Vec<f32>>,
) -> bool {
    let (Some(q), Some(d)) = (q_emb, emb) else {
        return false; // without embeddings, semantic candidacy is impossible (same as score_impl)
    };
    let gate = embedder.map(|e| e.semantic_gate()).unwrap_or(SEMANTIC_GATE);
    if cosine(q, d) >= gate {
        return true;
    }
    if q_terms_len <= TOKEN_FALLBACK_MAX_QUERY {
        if let Some(e) = embedder.filter(|e| e.token_fallback()) {
            let mut opt = Some(&mut *cache);
            return token_max_cosine(q, search_text, e, &mut opt) >= gate;
        }
    }
    false
}

fn token_max_cosine(
    q_emb: &[f32],
    doc_text: &str,
    embedder: &dyn Embedder,
    cache: &mut Option<&mut HashMap<String, Vec<f32>>>,
) -> f32 {
    let mut seen = HashSet::new();
    let mut best = 0.0f32;
    for tok in tokenize(doc_text) {
        if !seen.insert(tok.clone()) {
            continue;
        }
        if seen.len() > TOKEN_FALLBACK_MAX_DOC {
            break;
        }
        let c = match cache {
            Some(c) if c.contains_key(&tok) => cosine(q_emb, &c[&tok]),
            Some(c) if c.len() < TOKEN_CACHE_MAX => {
                let emb = embedder.embed(&tok);
                let c_val = cosine(q_emb, &emb);
                c.insert(tok, emb);
                c_val
            }
            _ => cosine(q_emb, &embedder.embed(&tok)),
        };
        if c > best {
            best = c;
        }
    }
    best
}

/// Common scoring core — all public variants funnel down here.
fn score_impl(
    mem: &Memory,
    query: &Query,
    q_emb: Option<&[f32]>,
    now: DateTime<Utc>,
    gate: f32,
    token_fb: Option<&dyn Embedder>,
    mut token_cache: Option<&mut HashMap<String, Vec<f32>>>,
) -> (f32, Vec<Signal>) {
    let mut signals = Vec::new();

    // Use last_access (not created_at) — aligns with should_forget: a
    // recently-accessed old memory is "fresh" for retrieval purposes.
    let age = (now - mem.last_access).num_seconds().max(0) as f64;
    let recency = recency_score(age);
    let importance = mem.importance.clamp(0.0, 1.0);
    let wilson = match &mem.kind {
        MemoryKind::Procedural {
            successes,
            failures,
            ..
        } => wilson_lower_bound(*successes, *failures) as f32,
        _ => 0.0,
    };

    signals.push(Signal {
        name: "recency".into(),
        value: recency,
    });
    signals.push(Signal {
        name: "importance".into(),
        value: importance,
    });
    if wilson > 0.0 {
        signals.push(Signal {
            name: "wilson".into(),
            value: wilson,
        });
    }

    let q_terms = tokenize(&query.text);
    if q_terms.is_empty() {
        // Browse mode: no relevance, rank by freshness + importance + confidence.
        let s = BROWSE_W.0 * recency + BROWSE_W.1 * importance + BROWSE_W.2 * wilson;
        signals.push(Signal {
            name: "score".into(),
            value: s,
        });
        return (s, signals);
    }

    let kw = keyword_score(&q_terms, &mem.searchable_text());
    signals.push(Signal {
        name: "keyword".into(),
        value: kw,
    });

    let has_emb = q_emb.is_some() && mem.embedding.is_some();
    let cos = match (q_emb, mem.embedding.as_deref()) {
        (Some(q), Some(d)) => cosine(q, d),
        _ => 0.0,
    };
    if has_emb {
        signals.push(Signal {
            name: "cosine".into(),
            value: cos,
        });
    }

    // Candidacy: keyword matched OR (semantic on & cosine passed gate).
    let semantic = query.semantic && has_emb;

    // Short query + no keyword + full-doc cosine below gate → token-level fallback.
    // Only runs on embedders that request it (n-gram hashing); on success it
    // enters the score as effective cosine (discovery signal "cosine_tok").
    let mut eff_cos = cos;
    if semantic && kw == 0.0 && cos < gate && q_terms.len() <= TOKEN_FALLBACK_MAX_QUERY {
        if let (Some(e), Some(q)) = (token_fb, q_emb) {
            let tok_cos = token_max_cosine(q, &mem.searchable_text(), e, &mut token_cache);
            if tok_cos >= gate {
                eff_cos = tok_cos;
                stats::TOKEN_FALLBACK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                signals.push(Signal {
                    name: "cosine_tok".into(),
                    value: tok_cos,
                });
            }
        }
    }

    let is_candidate = kw > 0.0 || (semantic && eff_cos >= gate);

    let final_score = if !is_candidate {
        0.0
    } else {
        let relevance = if has_emb {
            W_KEYWORD * kw + W_COSINE * eff_cos
        } else {
            kw
        };
        let boost = 1.0 + B_RECENCY * recency + B_IMPORTANCE * importance + B_WILSON * wilson;
        relevance * boost
    };
    if final_score > 0.0 {
        stats::RECALL_CANDIDATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    signals.push(Signal {
        name: "score".into(),
        value: final_score,
    });
    (final_score, signals)
}

/// Finalizes the candidate list: sorts by score, diversifies with MMR if
/// `query.diverse`, otherwise truncates to limit.
pub fn finalize(mut scored: Vec<Scored<Memory>>, query: &Query) -> Vec<Scored<Memory>> {
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Second pass: native rerank (query×doc shared features).
    if query.rerank && !query.text.trim().is_empty() {
        scored = super::rerank::native_rerank(&query.text, scored);
    }
    if query.diverse {
        mmr(scored, 0.7, query.limit)
    } else {
        scored.truncate(query.limit);
        scored
    }
}

fn scored_sim(a: &Scored<Memory>, b: &Scored<Memory>) -> f32 {
    match (&a.item.embedding, &b.item.embedding) {
        (Some(x), Some(y)) => cosine(x, y),
        _ => 0.0,
    }
}

/// Maximal Marginal Relevance: selects the best `k` by balancing relevance and
/// diversity. High `lambda` → relevance-dominant; low → diversity-dominant.
/// Records without embeddings incur no diversity penalty (relevance only).
pub fn mmr(candidates: Vec<Scored<Memory>>, lambda: f32, k: usize) -> Vec<Scored<Memory>> {
    if k == 0 {
        return Vec::new();
    }
    if candidates.len() <= 1 {
        return candidates;
    }
    let max_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let mut remaining = candidates;
    let mut selected: Vec<Scored<Memory>> = Vec::new();
    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        for (i, cand) in remaining.iter().enumerate() {
            let rel = cand.score / max_score;
            let div = selected
                .iter()
                .map(|s| scored_sim(cand, s))
                .fold(0.0f32, f32::max);
            let val = lambda * rel - (1.0 - lambda) * div;
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        selected.push(remaining.remove(best_idx));
    }
    selected
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Tokenizer must not panic on arbitrary input; all tokens must satisfy
        /// the contract (lowercase, ≥ 2 characters, alphanumeric).
        #[test]
        fn tokenize_upholds_contract(s in "\\PC*") {
            for t in tokenize(&s) {
                prop_assert!(t.chars().count() >= 2);
                prop_assert!(t.chars().all(|c| c.is_alphanumeric()));
                prop_assert_eq!(t.clone(), t.to_lowercase());
            }
        }

        /// Tool call parser must not panic on arbitrary input.
        #[test]
        fn parse_tool_call_never_panics(s in "\\PC*") {
            let _ = crate::tool::parse_tool_call(&s);
        }
    }
}

#[cfg(test)]
mod score_tests {
    use super::*;
    use crate::memory::embed::{Embedder, HashingEmbedder};
    use crate::memory::types::{Scope, SemanticCat};

    #[test]
    fn recency_ranks_by_last_access_not_created_at() {
        // Fix 1: old-created/recently-accessed must outrank
        // fresh-created/never-accessed.
        let now = Utc::now();
        let mut old_but_accessed =
            Memory::semantic(Scope::World, "old accessed", SemanticCat::Fact);
        old_but_accessed.created_at = now - chrono::Duration::days(60);
        old_but_accessed.last_access = now - chrono::Duration::hours(1);

        let mut fresh_never = Memory::semantic(Scope::World, "fresh never", SemanticCat::Fact);
        fresh_never.created_at = now - chrono::Duration::hours(2);
        fresh_never.last_access = now - chrono::Duration::hours(2);

        let q = Query::new("").limit(10); // browse mode (recency-dominant)
        let (s_old, _) = score(&old_but_accessed, &q, None, now);
        let (s_fresh, _) = score(&fresh_never, &q, None, now);
        assert!(
            s_old > s_fresh,
            "recently-accessed old record ({s_old}) should outrank \
             stale fresh record ({s_fresh})"
        );
    }

    #[test]
    fn short_query_token_fallback_recalls_morphology() {
        // TEST_REPORT §5.3: query "learning" could not find the "Learned Rust"
        // memory — multi-token document drowns out the single-token query
        // (dilution). Token-level fallback must compensate.
        let e = HashingEmbedder::new();
        let mut m = Memory::episodic(
            Scope::World,
            "Learned Rust",
            "learned ownership and borrow checker",
        );
        m.embedding = Some(e.embed(&m.searchable_text()));
        let q = Query::new("learning").semantic();
        let q_emb = e.embed("learning");
        let now = Utc::now();

        // Full-document cosine falls below the gate (the finding itself) — if
        // this passes, there is no dilution and the fallback's precondition has changed.
        #[allow(deprecated)]
        let (plain, _) = score_with_gate(&m, &q, Some(&q_emb), now, e.semantic_gate());
        assert_eq!(
            plain, 0.0,
            "dilution precondition: full-doc below cosine gate"
        );

        // Token-level fallback catches the morphological match.
        let (s, sig) = score_with_embedder(&m, &q, Some(&q_emb), now, Some(&e));
        assert!(s > 0.0, "token fallback should catch morphology");
        assert!(
            sig.iter().any(|x| x.name == "cosine_tok"),
            "token signal should be reported: {sig:?}"
        );
    }

    #[test]
    fn scorer_matches_one_shot_scoring_and_reuses_cache() {
        // Scorer (cached) and one-shot score_with_embedder must produce the
        // same result — cache is purely a cost optimization, it does not
        // change behavior.
        let e = HashingEmbedder::new();
        let mut mems = Vec::new();
        for (t, b) in [
            ("Learned Rust", "learned ownership and borrow checker"),
            ("Learned Go", "learned goroutine and channel"),
        ] {
            let mut m = Memory::episodic(Scope::World, t, b);
            m.embedding = Some(e.embed(&m.searchable_text()));
            mems.push(m);
        }
        let q = Query::new("learning").semantic();
        let q_emb = e.embed("learning");
        let now = Utc::now();
        let mut scorer = Scorer::new(Some(&e));
        for m in &mems {
            let (a, _) = scorer.score(m, &q, Some(&q_emb), now);
            let (b, _) = score_with_embedder(m, &q, Some(&q_emb), now, Some(&e));
            assert!((a - b).abs() < 1e-6, "Scorer == single-shot: {a} vs {b}");
            assert!(a > 0.0, "both memories should match morphologically");
        }
    }

    #[test]
    fn token_fallback_hit_feeds_metrics_counter() {
        // Resilient to parallel tests: measuring delta, not absolute value.
        use std::sync::atomic::Ordering;
        let before = stats::TOKEN_FALLBACK_HITS.load(Ordering::Relaxed);
        let e = HashingEmbedder::new();
        // Dilution-verified scenario (same as
        // short_query_token_fallback_recalls_morphology): full-doc cosine
        // below gate → match comes ONLY from fallback.
        let mut m = Memory::episodic(
            Scope::World,
            "Learned Rust",
            "learned ownership and borrow checker",
        );
        m.embedding = Some(e.embed(&m.searchable_text()));
        let q = Query::new("learning").semantic();
        let q_emb = e.embed("learning");
        let (s, _) = score_with_embedder(&m, &q, Some(&q_emb), Utc::now(), Some(&e));
        assert!(s > 0.0);
        assert!(
            stats::TOKEN_FALLBACK_HITS.load(Ordering::Relaxed) > before,
            "fallback match should increment counter"
        );
    }

    #[test]
    fn token_fallback_rejects_unrelated_short_query() {
        // Fallback must not turn unrelated short queries into false positives.
        let e = HashingEmbedder::new();
        let mut m = Memory::episodic(
            Scope::World,
            "Learned Rust",
            "learned ownership and borrow checker",
        );
        m.embedding = Some(e.embed(&m.searchable_text()));
        let q = Query::new("cats").semantic();
        let q_emb = e.embed("cats");
        let (s, _) = score_with_embedder(&m, &q, Some(&q_emb), Utc::now(), Some(&e));
        assert_eq!(
            s, 0.0,
            "unrelated query should not match even with token fallback"
        );
    }

    #[test]
    fn long_query_skips_token_fallback() {
        // Cost limit: fallback only activates for short (≤2 token) queries.
        let e = HashingEmbedder::new();
        let mut m = Memory::episodic(Scope::World, "Learned Rust", "ownership");
        m.embedding = Some(e.embed(&m.searchable_text()));
        let q = Query::new("want to learn right now today").semantic();
        let q_emb = e.embed("want to learn right now today");
        let (_, sig) = score_with_embedder(&m, &q, Some(&q_emb), Utc::now(), Some(&e));
        assert!(
            sig.iter().all(|x| x.name != "cosine_tok"),
            "token fallback should not activate for long query: {sig:?}"
        );
    }

    #[test]
    fn mmr_prefers_diversity_over_near_duplicate() {
        let mk = |txt: &str, sc: f32, e: Vec<f32>| {
            let mut m = Memory::semantic(Scope::World, txt, SemanticCat::Fact);
            m.embedding = Some(e);
            Scored {
                item: m,
                score: sc,
                signals: vec![],
            }
        };
        let a = mk("alfa", 1.00, vec![1.0, 0.0]);
        let b = mk("beta", 0.95, vec![1.0, 0.0]); // same direction as alpha (near-dup)
        let c = mk("gamma", 0.80, vec![0.0, 1.0]); // different
        let out = mmr(vec![a, b, c], 0.7, 2);
        assert_eq!(out.len(), 2);
        assert!(out[0].item.summary().contains("alfa"));
        assert!(out[1].item.summary().contains("gamma")); // diversity: not beta
    }

    #[test]
    fn is_conflict_only_in_mid_similarity_band() {
        // cos = 1.0 (duplicate) → not a conflict
        assert!(!is_conflict(&[1.0, 0.0], &[1.0, 0.0]));
        // cos ~ 0.7 → conflict
        assert!(is_conflict(&[1.0, 0.0], &[0.7, 0.714]));
        // cos = 0.5 → not a conflict (unrelated)
        assert!(!is_conflict(&[1.0, 0.0], &[0.5, 0.866]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{Scope, SemanticCat};

    #[test]
    fn tokenize_lowercases_splits_and_filters_short() {
        let t = tokenize("Rust, math and .NET!");
        assert!(t.contains(&"rust".to_string()));
        assert!(t.contains(&"math".to_string()));
        assert!(t.contains(&"net".to_string()));
        // "and" is 3 chars → kept; single-char tokens are filtered
        assert!(t.contains(&"and".to_string()));
    }

    #[test]
    fn tokenize_keeps_unicode_letters() {
        let t = tokenize("αβγδεζ");
        assert_eq!(t, vec!["αβγδεζ".to_string()]);
    }

    #[test]
    fn wilson_zero_when_no_samples() {
        assert_eq!(wilson_lower_bound(0, 0), 0.0);
    }

    #[test]
    fn wilson_rewards_more_evidence() {
        // Same 100% success rate but more evidence → higher lower bound.
        assert!(wilson_lower_bound(100, 0) > wilson_lower_bound(5, 0));
        assert!(wilson_lower_bound(5, 0) > wilson_lower_bound(1, 0));
    }

    #[test]
    fn wilson_penalizes_failures() {
        assert!(wilson_lower_bound(9, 1) > wilson_lower_bound(5, 5));
    }

    #[test]
    fn recency_is_high_when_fresh_and_decays() {
        let fresh = recency_score(0.0);
        let week = recency_score(7.0 * 24.0 * 3600.0);
        let month = recency_score(30.0 * 24.0 * 3600.0);
        assert!(fresh > 0.99);
        // 7 days = half-life → ~0.5
        assert!((week - 0.5).abs() < 0.01);
        assert!(month < week);
    }

    #[test]
    fn keyword_exact_beats_partial_beats_none() {
        let q = tokenize("rust math");
        let full = keyword_score(&q, "rust and math awesome");
        let partial = keyword_score(&q, "rust and games");
        let none = keyword_score(&q, "completely unrelated text");
        assert!(full > partial);
        assert!(partial > none);
        assert_eq!(none, 0.0);
        assert!(full <= 1.0);
    }

    #[test]
    fn recency_clamps_future_last_access_to_zero_age() {
        // Edge test: if last_access is in the future (clock skew),
        // (now - last_access) would be negative. The .max(0) clamp must
        // produce age=0 → recency_score(0) ≈ 1.0 (freshest possible).
        let now = Utc::now();
        let mut future_mem = Memory::semantic(Scope::World, "clock-skewed", SemanticCat::Fact);
        future_mem.last_access = now + chrono::Duration::seconds(300); // 5 min in future

        let q = Query::new("").limit(10); // browse mode
        let (s_future, _) = score(&future_mem, &q, None, now);
        let (s_now, _) = {
            let mut fresh = Memory::semantic(Scope::World, "fresh now", SemanticCat::Fact);
            fresh.last_access = now;
            score(&fresh, &q, None, now)
        }; // age=0
        assert!((s_future - s_now).abs() < 1e-6,
            "future last_access should be clamped to age=0, same score as now: {s_future} vs {s_now}");
    }
}
