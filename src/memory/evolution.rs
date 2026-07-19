//! Evolution: memory maintenance over time — consolidation, decay, forgetting.
//!
//! "Forgetting is as important as remembering." This module provides pure decision
//! functions (`should_forget`, `duplicates`, `plan`); stores call them and apply
//! soft-delete. Also runs a periodic background task (`spawn_periodic`).

use super::retrieval::{cosine, wilson_lower_bound};
use super::types::{Memory, MemoryKind};
use super::MemoryStore;
use crate::id::MemoryId;
use chrono::{DateTime, Utc};
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

/// Process-wide consolidation counters (metrics feed).
pub mod stats {
    use std::sync::atomic::AtomicU64;

    /// Total run count.
    pub static RUNS: AtomicU64 = AtomicU64::new(0);
    /// Total merged (near-duplicate) records.
    pub static MERGED: AtomicU64 = AtomicU64::new(0);
    /// Total forgotten (decay) records.
    pub static FORGOTTEN: AtomicU64 = AtomicU64::new(0);
    /// Duration of the last run (ms).
    pub static LAST_MS: AtomicU64 = AtomicU64::new(0);
}

/// Forgetting/merge thresholds.
#[derive(Clone, Debug)]
pub struct ForgetPolicy {
    /// Records not accessed for longer than this become forgetting candidates (seconds).
    pub max_idle_secs: f64,
    /// Importance below this threshold is required for forgetting.
    pub min_importance: f32,
    /// Records accessed at least this many times are protected.
    pub min_access: u32,
    /// Cosine at or above this → near-duplicate (merge).
    pub dedup_cosine: f32,
}

impl Default for ForgetPolicy {
    fn default() -> Self {
        Self {
            max_idle_secs: 90.0 * 24.0 * 3600.0, // 90 days
            min_importance: 0.25,
            min_access: 1,
            dedup_cosine: 0.92,
        }
    }
}

/// Whether a record is a forgetting candidate (decay + low value + unused).
pub fn should_forget(mem: &Memory, now: DateTime<Utc>, p: &ForgetPolicy) -> bool {
    if mem.deleted_at.is_some() {
        return false;
    }
    let idle = (now - mem.last_access).num_seconds().max(0) as f64;
    if idle <= p.max_idle_secs {
        return false; // fresh enough
    }
    if mem.importance >= p.min_importance {
        return false; // important
    }
    if mem.access_count >= p.min_access {
        return false; // in use
    }
    // Proven procedure is protected.
    if let MemoryKind::Procedural {
        successes,
        failures,
        ..
    } = &mem.kind
    {
        if wilson_lower_bound(*successes, *failures) > 0.5 {
            return false;
        }
    }
    true
}

/// Near-duplicate pairs: `(keep_id, drop_id)`.
/// Same scope + both have embeddings + cosine ≥ threshold. Kept: the more important
/// one (on tie, the newer one).
///
/// Comparison is per-scope (different scopes can never match) — cost drops from
/// O(n²) for the whole store to Σ per-scope O(n²); records without embeddings
/// or that are soft-deleted never enter the loop.
/// Finds near-duplicate pairs: `(keep, drop)` — the one with higher importance
/// (on tie, the fresher one) stays.
///
/// Scale: multi-probe sort-LSH instead of full O(n²) pairwise scan. 64-bit
/// random-hyperplane signatures are sorted with [`SORT_ROUNDS`] independent bit
/// permutations; each item is a candidate only against its sorted neighborhood
/// window ([`SORT_WINDOW`]) and verified with exact cosine. Candidate count is
/// linear regardless of data clustering (band-bucket approach exploded on
/// clustered corpora). For cos ≥ 0.92 pairs, capture probability is 99%+;
/// no false positives (exact verification), rare false negatives are acceptable
/// (consolidation re-runs hourly). Hyperplanes and permutations use a fixed seed:
/// offline + deterministic.
pub fn duplicates(mems: &[Memory], p: &ForgetPolicy) -> Vec<(MemoryId, MemoryId)> {
    use std::collections::HashMap;
    let mut by_scope: HashMap<&super::types::Scope, Vec<&Memory>> = HashMap::new();
    for m in mems {
        if m.deleted_at.is_none() && m.embedding.is_some() {
            by_scope.entry(&m.scope).or_default().push(m);
        }
    }
    let mut pairs = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for group in by_scope.values() {
        // Hyperplane matrix is built ONCE per scope group.
        let dim = group
            .iter()
            .find_map(|m| m.embedding.as_ref().map(|e| e.len()))
            .unwrap_or(0);
        let planes = plane_matrix(dim, LSH_BITS);
        let sigs: Vec<u64> = group
            .iter()
            .map(|m| lsh_signature_with(m.embedding.as_deref().unwrap_or(&[]), &planes))
            .collect();
        // Multi-probe: in each round, bits are reshuffled and sorted; only
        // within-window pairs are candidates → candidate count is linear (clustering-independent).
        seen.clear();
        for round in 0..SORT_ROUNDS {
            let perm = bit_permutation(round as u64);
            let mut order: Vec<usize> = (0..group.len()).collect();
            order.sort_by_cached_key(|&i| permute_bits(sigs[i], &perm));
            for w in 0..order.len() {
                let end = (w + 1 + SORT_WINDOW).min(order.len());
                for j in (w + 1)..end {
                    let (i, k) = (order[w], order[j]);
                    let key = if i < k { (i, k) } else { (k, i) };
                    if !seen.insert(key) {
                        continue;
                    }
                    let (a, b) = (group[i], group[k]);
                    let (Some(ea), Some(eb)) = (&a.embedding, &b.embedding) else {
                        continue;
                    };
                    if cosine(ea, eb) >= p.dedup_cosine {
                        let a_wins = a.importance > b.importance
                            || (a.importance == b.importance && a.created_at >= b.created_at);
                        let (keep, drop) = if a_wins { (a, b) } else { (b, a) };
                        pairs.push((keep.id.clone(), drop.id.clone()));
                    }
                }
            }
        }
    }
    pairs
}

/// LSH signature bit count (64 hyperplane bits).
const LSH_BITS: usize = 64;
/// Multi-probe rounds: each round sorts signature bits with a different
/// (deterministic) permutation. In a single sort, near-dups that diverge in
/// the high-order bit land far apart; a few independent rounds compensate.
const SORT_ROUNDS: usize = 4;
/// Sorted-index neighbor window for each item. Candidate count =
/// SORT_ROUNDS × n × SORT_WINDOW — INDEPENDENT of data clustering
/// (no band-bucket explosion).
const SORT_WINDOW: usize = 16;

/// Random-hyperplane LSH signature: each bit encodes which side of a
/// hyperplane the vector falls on. Similar vectors share most bits.
/// Hyperplane values are derived from splitmix64 hash (deterministic).
/// (The production path builds the matrix once and uses `lsh_signature_with`;
/// this wrapper is for test/benchmark convenience.)
#[cfg(test)]
fn lsh_signature(emb: &[f32]) -> u64 {
    lsh_signature_with(emb, &plane_matrix(emb.len(), LSH_BITS))
}

/// Signature with a pre-built hyperplane matrix (hot path — `duplicates`).
fn lsh_signature_with(emb: &[f32], planes: &[Vec<f32>]) -> u64 {
    let mut sig = 0u64;
    for (b, p) in planes.iter().enumerate() {
        let mut dot = 0.0f32;
        for (pv, x) in p.iter().zip(emb) {
            dot += pv * x;
        }
        if dot >= 0.0 {
            sig |= 1 << b;
        }
    }
    sig
}

/// `bits` hyperplanes (each `dim`-dimensional) — built once per consolidation
/// run and reused across all records.
fn plane_matrix(dim: usize, bits: usize) -> Vec<Vec<f32>> {
    (0..bits)
        .map(|b| (0..dim).map(|d| plane_value(b, d)).collect())
        .collect()
}

/// Applies bit permutation: `perm[new_pos]` = old bit position to read from.
fn permute_bits(sig: u64, perm: &[u8; 64]) -> u64 {
    let mut out = 0u64;
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        out |= ((sig >> old_pos) & 1) << new_pos;
    }
    out
}

/// Deterministic bit permutation (Fisher-Yates, splitmix64 seeded).
/// Derived from round index — same rounds in every run.
fn bit_permutation(round: u64) -> [u8; 64] {
    let mut perm: [u8; 64] = std::array::from_fn(|i| i as u8);
    let mut state = round.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..64usize).rev() {
        state = splitmix64(state);
        let j = (state % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm
}

/// splitmix64 mixer (for hyperplane and permutation seeds).
fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hyperplane matrix cell (plane, dim): splitmix64 to (-1, 1).
/// Fixed-seed hash instead of randomness — same planes in every process.
fn plane_value(plane: usize, dim: usize) -> f32 {
    let mut z = (plane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (dim as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ 0x1656_67B1_9E37_79F9;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
}

/// A consolidation plan: which records to forget + counters.
#[derive(Clone, Debug, Default)]
pub struct ConsolidationPlan {
    /// Identities to soft-delete.
    pub to_forget: Vec<MemoryId>,
    /// Count dropped due to merge (near-duplicate).
    pub merged: usize,
    /// Count dropped due to decay (forgetting).
    pub forgotten: usize,
}

/// Produces a consolidation plan for the given records (pure; does not apply).
pub fn plan(mems: &[Memory], p: &ForgetPolicy, now: DateTime<Utc>) -> ConsolidationPlan {
    let mut drop_set: HashSet<MemoryId> = HashSet::new();

    // 1) Near-duplicate merge.
    for (_keep, drop) in duplicates(mems, p) {
        drop_set.insert(drop);
    }
    let merged = drop_set.len();

    // 2) Decay forgetting (excluding already merged).
    let mut forgotten = 0;
    for mem in mems {
        if drop_set.contains(&mem.id) {
            continue;
        }
        if should_forget(mem, now, p) {
            drop_set.insert(mem.id.clone());
            forgotten += 1;
        }
    }

    ConsolidationPlan {
        to_forget: drop_set.into_iter().collect(),
        merged,
        forgotten,
    }
}

/// Runs periodic consolidation in the background (tokio task).
/// The returned handle can be stopped via `abort`.
pub fn spawn_periodic(
    store: Arc<dyn MemoryStore>,
    period: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(period);
        loop {
            ticker.tick().await;
            let t0 = std::time::Instant::now();
            // Errors are not silently swallowed: persistent DB issues must be visible (janitor
            // should not appear "running" while spinning forever).
            match store.consolidate().await {
                Ok(r) => {
                    let ms = t0.elapsed().as_millis() as u64;
                    stats::RUNS.fetch_add(1, Ordering::Relaxed);
                    stats::MERGED.fetch_add(r.merged as u64, Ordering::Relaxed);
                    stats::FORGOTTEN.fetch_add(r.forgotten as u64, Ordering::Relaxed);
                    stats::LAST_MS.store(ms, Ordering::Relaxed);
                    tracing::info!(
                        scanned = r.scanned,
                        merged = r.merged,
                        forgotten = r.forgotten,
                        ms,
                        "consolidation done"
                    );
                }
                Err(e) => tracing::error!(error = %e, "consolidation error"),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::AgentId;
    use crate::memory::embed::{Embedder, HashingEmbedder};
    use crate::memory::types::{Memory, Scope, SemanticCat};
    use chrono::Duration as ChronoDuration;

    fn old_low_value(scope: Scope) -> Memory {
        let mut m =
            Memory::semantic(scope, "old unimportant note", SemanticCat::Fact).with_importance(0.1);
        let old = Utc::now() - ChronoDuration::days(120);
        m.created_at = old;
        m.last_access = old;
        m
    }

    #[test]
    fn forgets_old_low_value_unused() {
        let m = old_low_value(Scope::World);
        assert!(should_forget(&m, Utc::now(), &ForgetPolicy::default()));
    }

    #[test]
    fn keeps_recent_or_important_or_used() {
        let p = ForgetPolicy::default();
        let now = Utc::now();

        // Fresh
        let fresh = Memory::semantic(Scope::World, "fresh", SemanticCat::Fact).with_importance(0.1);
        assert!(!should_forget(&fresh, now, &p));

        // Old but important
        let mut important = old_low_value(Scope::World);
        important.importance = 0.9;
        assert!(!should_forget(&important, now, &p));

        // Old, unimportant but accessed
        let mut used = old_low_value(Scope::World);
        used.access_count = 3;
        assert!(!should_forget(&used, now, &p));
    }

    #[test]
    fn forgets_old_auto_exchange_records() {
        // Exchange/board records are born with AUTO_IMPORTANCE — decay must be
        // able to reclaim them over time (otherwise memory grows unbounded).
        let p = ForgetPolicy::default();
        assert!(
            Memory::AUTO_IMPORTANCE < p.min_importance,
            "auto-record importance should be below forgetting threshold"
        );
        let mut m = Memory::episodic(Scope::World, "responded to 'question': ", "response")
            .with_importance(Memory::AUTO_IMPORTANCE);
        let old = Utc::now() - ChronoDuration::days(120);
        m.created_at = old;
        m.last_access = old;
        assert!(should_forget(&m, Utc::now(), &p));
    }

    #[test]
    fn lsh_candidates_find_near_dups_and_skip_unrelated() {
        // LSH signature property: similar vectors share most bits,
        // unrelated ones differ in about half.
        let e = HashingEmbedder::new();
        let a = e.embed("rust ownership model ensures memory safety");
        let near = e.embed("rust ownership model ensures memory safety!"); // ~copy
        let far = e.embed("where to buy cat food today");
        let sig_a = lsh_signature(&a);
        let sig_near = lsh_signature(&near);
        let sig_far = lsh_signature(&far);
        let hamming_near = (sig_a ^ sig_near).count_ones();
        let hamming_far = (sig_a ^ sig_far).count_ones();
        assert!(
            hamming_near <= 6,
            "near-dup bits should be close: {hamming_near}"
        );
        assert!(
            hamming_far > hamming_near * 2,
            "unrelated clearly distant: near={hamming_near} far={hamming_far}"
        );
        // Multi-probe: a pair missed in one round's window is caught in another —
        // permutations must be truly different and valid (0..64 bijection).
        let p0 = bit_permutation(0);
        let p1 = bit_permutation(1);
        assert_ne!(p0, p1, "rounds are independent permutations");
        let mut sorted = p0;
        sorted.sort();
        assert_eq!(sorted, std::array::from_fn::<u8, 64, _>(|i| i as u8));
    }

    #[test]
    fn duplicates_scales_without_full_pairwise_scan() {
        // M2: 2000 records + 5 planted near-dup pairs — via LSH candidates
        // instead of O(n²) scan. All pairs should be found, unrelated ones filtered.
        let e = HashingEmbedder::new();
        // Texts that are also mutually DISTANT at the n-gram level: hex blocks
        // diverge in early characters (shared prefix = shared 3-gram = near-dup;
        // in the previous version, signatures clustered and band buckets exploded).
        let mut mems: Vec<Memory> = (0..2000u64)
            .map(|i| {
                let mut m = Memory::semantic(
                    Scope::World,
                    format!(
                        "{:x} {:x} {:x}",
                        i.wrapping_mul(7919) % 1_000_003,
                        i.wrapping_mul(104729) % 1_000_033,
                        i.wrapping_mul(1299709) % 1_000_039
                    ),
                    SemanticCat::Fact,
                );
                m.embedding = Some(e.embed(&m.searchable_text()));
                m
            })
            .collect();
        let mut planted: Vec<(MemoryId, MemoryId)> = Vec::new();
        for k in 0..5 {
            let mut a = Memory::semantic(
                Scope::World,
                format!("custom procedure step {k}: compile then test carefully"),
                SemanticCat::Fact,
            );
            a.embedding = Some(e.embed(&a.searchable_text()));
            let mut b = Memory::semantic(
                Scope::World,
                format!("custom procedure step {k}: compile then test carefully."),
                SemanticCat::Fact,
            );
            b.embedding = Some(e.embed(&b.searchable_text()));
            planted.push((a.id.clone(), b.id.clone()));
            mems.push(a);
            mems.push(b);
        }
        let p = ForgetPolicy::default();
        let t0 = std::time::Instant::now();
        let dups = duplicates(&mems, &p);
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "fast with LSH: {elapsed:?}"
        );
        for (a, b) in &planted {
            assert!(
                dups.iter()
                    .any(|(x, y)| (x == a && y == b) || (x == b && y == a)),
                "embedded pair should be found"
            );
        }
        // Strong correctness check: every reported pair must actually exceed the
        // threshold (LSH only generates candidates; the decision is exact cosine).
        let by_id: std::collections::HashMap<_, _> =
            mems.iter().map(|m| (m.id.clone(), m)).collect();
        for (keep, drop) in &dups {
            let (a, b) = (&by_id[keep], &by_id[drop]);
            let c = cosine(
                a.embedding.as_deref().unwrap(),
                b.embedding.as_deref().unwrap(),
            );
            assert!(c >= p.dedup_cosine, "no false positives: cos={c}");
        }
        // Candidate pool has not exploded: small candidate pool instead of 2M pairs.
        assert!(dups.len() < 200, "candidate pool bounded: {}", dups.len());
    }

    #[test]
    fn duplicates_flags_identical_embeddings() {
        let s = Scope::Agent(AgentId::new());
        let emb = vec![1.0, 0.0, 0.0];
        let mut a = Memory::semantic(s.clone(), "same", SemanticCat::Fact).with_importance(0.8);
        let mut b = Memory::semantic(s.clone(), "same", SemanticCat::Fact).with_importance(0.2);
        a.embedding = Some(emb.clone());
        b.embedding = Some(emb);

        let pairs = duplicates(&[a.clone(), b.clone()], &ForgetPolicy::default());
        assert_eq!(pairs.len(), 1);
        // The more important one (a) is kept, b is dropped.
        assert_eq!(pairs[0].0, a.id);
        assert_eq!(pairs[0].1, b.id);
    }
}
