//! Performance benchmarks: retrieval (keyword / semantic / browse), embed.
//!
//! Run: `cargo bench` — results land under `target/criterion/`.
//! For regression tracking on calibration changes (scoring, FTS, prefilter)
//! reference numbers are recorded in the README.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lore::{Embedder, HashingEmbedder, InMemoryStore, Memory, MemoryStore, Query, Scope};
use std::sync::Arc;

/// Deterministic fake corpus: n records, mixed topics (keyword selectivity
/// is realistic because the query term appears in ~2% of records).
fn corpus(n: usize) -> Vec<Memory> {
    let topics = [
        "rust ownership borrow checker",
        "go goroutine channel scheduler",
        "python asyncio event loop",
        "linux kernel scheduling",
        "database index btree page",
        "network tcp congestion window",
        "compiler parser lexer token",
        "graphics shader pipeline vertex",
        "audio dsp filter resonance",
        "garden tomato watering soil",
    ];
    (0..n)
        .map(|i| {
            let t = topics[i % topics.len()];
            let title = format!("not {i}: {t}");
            let body = format!("{t} observation {i} and details");
            Memory::episodic(Scope::World, title, body)
        })
        .collect()
}

fn bench_recall(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("recall");
    for &n in &[1_000usize, 10_000] {
        // Setup is excluded from measurement: the store is populated once.
        let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
        rt.block_on(async {
            for m in corpus(n) {
                store.remember(m).await.unwrap();
            }
        });

        group.bench_with_input(BenchmarkId::new("inmem_keyword", n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(store.recall(&Scope::World, &Query::new("rust ownership")))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("inmem_semantic_short", n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(store.recall(&Scope::World, &Query::new("learning").semantic()))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("inmem_browse", n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(store.recall(&Scope::World, &Query::new("")))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_sqlite(c: &mut Criterion) {
    use lore::SqliteStore;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("sqlite_recall");
    group.sample_size(20); // file IO benchmark — keep sample count low
    for &n in &[1_000usize, 10_000] {
        let path = std::env::temp_dir().join(format!("lore-bench-{n}.db"));
        let _ = std::fs::remove_file(&path);
        let store = SqliteStore::open(path.to_str().unwrap())
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        rt.block_on(async {
            for m in corpus(n) {
                store.remember(m).await.unwrap();
            }
        });

        group.bench_with_input(BenchmarkId::new("keyword_fts", n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(store.recall(&Scope::World, &Query::new("rust ownership")))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("semantic_short", n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(store.recall(&Scope::World, &Query::new("learning").semantic()))
                    .unwrap()
            })
        });
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", path.display(), suffix));
        }
    }
    group.finish();
}

fn bench_embed(c: &mut Criterion) {
    let e = HashingEmbedder::new();
    c.bench_function("hashing_embed_sentence", |b| {
        b.iter(|| e.embed("a long example sentence about ownership and borrow checker"))
    });
}

/// Consolidation dedup (LSH banding): near-dup scan across n embedded records.
/// Reference for old O(n²) full scan: 10k records ~75s+ (50M pairs × 512-dim).
fn bench_dedup(c: &mut Criterion) {
    use lore::memory::evolution::duplicates;
    let e = HashingEmbedder::new();
    let mut group = c.benchmark_group("dedup_lsh");
    group.sample_size(10); // expensive setup — few samples suffice
    for &n in &[1_000usize, 10_000] {
        // Realistic distribution: topic clusters EXIST but texts diverge at n-gram level
        // (each record has a unique identifier). Note: with degenerate clusters
        // in the corpus (thousands of nearly identical texts), band buckets naturally
        // grow — in that case, cost approaches the actual near-dup pair count
        // (the output itself consists of pairs; still faster than the old full scan).
        let topics = [
            "rust ownership borrow checker",
            "go goroutine channel scheduler",
            "python asyncio event loop",
            "linux kernel scheduling",
            "database index btree page",
        ];
        let mems: Vec<Memory> = (0..n)
            .map(|i| {
                let mut m = Memory::episodic(
                    Scope::World,
                    format!("not {:x}", i.wrapping_mul(7919) % 1_000_003),
                    format!(
                        "{} {:x}",
                        topics[i % topics.len()],
                        i.wrapping_mul(104729) % 1_000_033
                    ),
                );
                m.embedding = Some(e.embed(&m.searchable_text()));
                m
            })
            .collect();
        let policy = lore::ForgetPolicy::default();
        group.bench_with_input(BenchmarkId::new("duplicates", n), &n, |b, _| {
            b.iter(|| duplicates(&mems, &policy))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_recall,
    bench_sqlite,
    bench_embed,
    bench_dedup
);
criterion_main!(benches);
