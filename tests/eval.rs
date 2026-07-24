//! Retrieval QUALITY harness: accuracy, not speed.
//!
//! Benchmarks measure latency; this suite measures hit@1 / hit@5 / MRR@5 on a
//! golden set with distractors, entity-bridged clusters (multi-hop), zero-
//! overlap paraphrases, and a Turkish subset. Semantic gate, fusion weights,
//! embedder, graph, rerank, or token-level fallback calibration changes —
//! this catches quality regressions that unit tests cannot see, and gives
//! every new signal a number to earn.
//!
//! Thresholds are kept BELOW measured behavior (regression alarm, not a
//! target); when quality improves, thresholds are ONLY updated upward.
//! Per-category rates print each run — paraphrase and multi-hop are the
//! documented headroom for the neural layer and the graph leg respectively.
//!
//! Spec: docs/superpowers/specs/2026-07-24-memory-deepening-design.md

use lore::memory::HashingEmbedder;
use lore::{InMemoryStore, Memory, MemoryStore, Query, Scope, SemanticCat};
use std::sync::Arc;

/// Golden corpus. Structure (do not shuffle — queries reference indices):
/// - 0..=2   cluster A: Aylin → cat Paspas → vet (entity bridge)
/// - 3..=5   cluster B: Deniz → gravel bike → derailleur (entity bridge)
/// - 6..=8   cluster C: Proxmox host → NAS backups → raid disks (bridge)
/// - 9..=25  singles: tech + personal
/// - 26..=35 distractors: share surface tokens with queries, wrong answers
/// - 36..=39 Turkish subset
/// - 40..=55 singles: breadth/noise floor
const CORPUS: &[&str] = &[
    // cluster A
    "Aylin adopted a tabby cat and named it Paspas",
    "Paspas the cat was vaccinated at the veterinary clinic on Tuesday",
    "Aylin works as a backend engineer at a fintech startup",
    // cluster B
    "Deniz rides a gravel bike on weekend trips along the coast",
    "The gravel bike's rear derailleur was replaced with Shimano Deore",
    "Deniz packed panniers for the Ayvalik coastal cycling route",
    // cluster C
    "The homelab server runs Proxmox with three Debian virtual machines",
    "Nightly backups of the Proxmox host go to an external NAS",
    "The NAS raid array uses four 8TB disks in raid5",
    // tech singles
    "We use tokio runtime in async Rust, spawning tasks with spawn",
    "User started learning the Rust programming language, studying ownership",
    "Docker containers get a volume backup strategy with restic",
    "The team migrated the API gateway from nginx to traefik",
    "Postgres query planner was tuned by adding a partial index on orders",
    "Grafana dashboards alert when p99 latency exceeds 800 milliseconds",
    "The mobile app crash was traced to a null pointer in the payment flow",
    // personal singles
    "Their favorite food is manti, especially Kayseri style dumplings",
    "User drinks coffee without sugar, plain filter brew",
    "Talks to their mom on the phone every Sunday evening",
    "Over the weekend we visited Galata Tower in Istanbul",
    "Planted tomato and pepper seedlings in the garden beds",
    "The math exam will cover derivatives and integrals",
    "Applied Newton's second law during the physics exercise session",
    "Arch Linux was installed on the new work laptop with hyprland",
    "Project deadline was set for next Friday by the steering committee",
    "Remote work policy was updated at the company all-hands meeting",
    // distractors
    "The vet clinic parking lot was repaved last month",
    "A documentary about cats aired on television last night",
    "Backup dancers rehearsed for the concert tour",
    "Rust stains on the balcony railing were scrubbed off",
    "The coffee shop on the corner changed its opening hours",
    "Shimano also manufactures fishing reels and rods",
    "The garden hose was left running overnight by mistake",
    "Deadline pressure at the newspaper made the editor cranky",
    "Tomato soup recipe calls for basil and cream",
    "The physics of bicycle balance involves gyroscopic effects",
    // Turkish subset
    "Haftasonu Kapadokya'da balon turuna katildik",
    "Annemin dogum gunu icin cicek siparisi verildi",
    "Toplantida sprint planlamasi carsambaya alindi",
    "Kedi mamasi stoklari azaldi, yeni paket siparis edildi",
    // breadth singles
    "The espresso machine descaling is due every three months",
    "Kubernetes ingress was switched to cert-manager for TLS renewal",
    "The piano teacher assigned a Chopin nocturne for practice",
    "Marathon training plan starts with 5k runs on weekdays",
    "The library book about stoicism is due back on the 15th",
    "Solar panels on the roof cut the electricity bill in half",
    "The aquarium filter pump started making a rattling noise",
    "Friday game night features Catan and Codenames with neighbors",
    "The tax return deadline for freelancers is end of March",
    "A sourdough starter needs feeding twice a day in summer",
    "The drone footage of the coastline was edited in DaVinci",
    "Winter tires were mounted before the trip to Uludag",
    "The standing desk motor stopped working at half height",
    "Beekeeping course covers hive inspection and queen spotting",
    "The neighborhood gym replaced the rowing machines",
    "Passwords were rotated after the phishing attempt at work",
];

/// Query category — per-category rates print each run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cat {
    /// Exact/near-exact token match — must stay near-perfect.
    Exact,
    /// Morphological variants (stemming-free tokenizer stress).
    Morphology,
    /// Zero token overlap — HashingEmbedder's honest weak spot (neural headroom).
    Paraphrase,
    /// Answer requires an entity bridge across records (graph-leg headroom).
    MultiHop,
    /// Surface-token distractors compete — precision under noise.
    Distractor,
}

/// Golden queries: (query, expected corpus index, category, note).
const QUERIES: &[(&str, usize, Cat, &str)] = &[
    // exact
    ("tokio spawn", 9, Cat::Exact, "technical terms"),
    ("galata", 19, Cat::Exact, "proper noun"),
    ("manti dumplings", 16, Cat::Exact, "food"),
    ("traefik migration", 12, Cat::Exact, "infra change"),
    ("hyprland laptop", 23, Cat::Exact, "setup"),
    ("proxmox virtual machines", 6, Cat::Exact, "homelab"),
    ("kapadokya balon", 36, Cat::Exact, "turkish exact"),
    // morphology
    ("learning", 10, Cat::Morphology, "learning/learn variants"),
    (
        "vaccination",
        1,
        Cat::Morphology,
        "vaccinated → vaccination",
    ),
    (
        "planting seedlings",
        20,
        Cat::Morphology,
        "planted → planting",
    ),
    ("tuning postgres", 13, Cat::Morphology, "tuned → tuning"),
    ("descale", 40, Cat::Morphology, "descaling → descale"),
    (
        "sprint plani",
        38,
        Cat::Morphology,
        "tr: planlamasi → plani",
    ),
    (
        "cicek siparis",
        37,
        Cat::Morphology,
        "tr: siparisi → siparis",
    ),
    ("kedi mama", 39, Cat::Morphology, "tr: mamasi → mama"),
    // paraphrase (zero token overlap with the answer)
    (
        "feline immunization",
        1,
        Cat::Paraphrase,
        "cat vaccine, no shared tokens",
    ),
    (
        "two wheeler gearing swap",
        4,
        Cat::Paraphrase,
        "derailleur replacement",
    ),
    ("caffeine preference", 17, Cat::Paraphrase, "coffee habits"),
    (
        "weekly call with parents",
        18,
        Cat::Paraphrase,
        "mom phone sundays",
    ),
    (
        "photovoltaic savings",
        45,
        Cat::Paraphrase,
        "solar panels bill",
    ),
    (
        "bread fermentation care",
        49,
        Cat::Paraphrase,
        "sourdough feeding",
    ),
    (
        "credential reset security incident",
        55,
        Cat::Paraphrase,
        "password rotation phishing",
    ),
    // multi-hop (entity bridge across records)
    ("aylin cat health", 1, Cat::MultiHop, "aylin→paspas→vet"),
    (
        "paspas owner job",
        2,
        Cat::MultiHop,
        "paspas→aylin→engineer",
    ),
    (
        "deniz bike repair",
        4,
        Cat::MultiHop,
        "deniz→gravel bike→derailleur",
    ),
    (
        "proxmox disk redundancy",
        8,
        Cat::MultiHop,
        "proxmox→NAS→raid",
    ),
    // distractor resistance
    (
        "cat vaccine",
        1,
        Cat::Distractor,
        "vs cat documentary + vet parking",
    ),
    ("rust ownership", 10, Cat::Distractor, "vs rust stains"),
    ("docker backup", 11, Cat::Distractor, "vs backup dancers"),
    (
        "shimano derailleur",
        4,
        Cat::Distractor,
        "vs shimano fishing",
    ),
    ("garden tomato", 20, Cat::Distractor, "vs hose + soup"),
    (
        "project deadline friday",
        24,
        Cat::Distractor,
        "vs newspaper + tax deadlines",
    ),
];

struct CatStats {
    hits1: usize,
    hits5: usize,
    total: usize,
}

async fn seeded_store() -> InMemoryStore {
    let store = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
    for text in CORPUS {
        store
            .remember(Memory::semantic(Scope::World, *text, SemanticCat::Fact))
            .await
            .unwrap();
    }
    store
}

#[tokio::test]
async fn retrieval_golden_set_metrics() {
    let store = seeded_store().await;

    let mut hits1 = 0usize;
    let mut hits5 = 0usize;
    let mut mrr = 0.0f64;
    let mut misses: Vec<String> = Vec::new();
    let mut cats: Vec<(Cat, CatStats)> = [
        Cat::Exact,
        Cat::Morphology,
        Cat::Paraphrase,
        Cat::MultiHop,
        Cat::Distractor,
    ]
    .into_iter()
    .map(|c| {
        (
            c,
            CatStats {
                hits1: 0,
                hits5: 0,
                total: 0,
            },
        )
    })
    .collect();

    for (q, want, cat, note) in QUERIES {
        let res = store
            .recall(&Scope::World, &Query::new(*q).semantic().graph().limit(5))
            .await
            .unwrap();
        let want_text = CORPUS[*want];
        let rank = res
            .iter()
            .position(|s| s.item.searchable_text().contains(want_text));

        let stats = &mut cats.iter_mut().find(|(c, _)| c == cat).unwrap().1;
        stats.total += 1;
        match rank {
            Some(0) => {
                hits1 += 1;
                hits5 += 1;
                mrr += 1.0;
                stats.hits1 += 1;
                stats.hits5 += 1;
            }
            Some(r) => {
                hits5 += 1;
                mrr += 1.0 / (r as f64 + 1.0);
                stats.hits5 += 1;
            }
            None => {
                let got: Vec<String> = res
                    .iter()
                    .take(3)
                    .map(|s| s.item.searchable_text().chars().take(44).collect())
                    .collect();
                misses.push(format!(
                    "'{q}' [{cat:?}] ({note}) → expected #{want}, got: {got:?}"
                ));
            }
        }
    }

    let total = QUERIES.len();
    let h5 = hits5 as f64 / total as f64;
    let h1 = hits1 as f64 / total as f64;
    let mrr5 = mrr / total as f64;

    eprintln!(
        "[eval] ── golden set ({total} queries, {} records) ──",
        CORPUS.len()
    );
    eprintln!(
        "[eval] hit@1 = {hits1}/{total} ({:.0}%)  hit@5 = {hits5}/{total} ({:.0}%)  MRR@5 = {mrr5:.3}",
        h1 * 100.0,
        h5 * 100.0
    );
    for (c, s) in &cats {
        eprintln!(
            "[eval]   {c:?}: hit@5 {}/{} (hit@1 {})",
            s.hits5, s.total, s.hits1
        );
    }
    if !misses.is_empty() {
        eprintln!("[eval] missed:\n{}", misses.join("\n"));
    }

    // ── Regression alarms (BELOW measured baseline; only ever raised) ──
    // Baseline history (HashingEmbedder):
    //   2026-07-24 pre-graph:  hit@1 66% · hit@5 72% · MRR .682 · MultiHop 2/4
    //   2026-07-24 graph leg:  hit@1 66% · hit@5 78% · MRR .701 · MultiHop 4/4
    //     (acronym entities NAS/TLS-style + damped 1-hop expansion;
    //      Exact 7/7 · Morphology 8/8 · Distractor 6/6 unchanged)
    // Paraphrase 0/7 is the neural layer's documented headroom.
    assert!(
        h5 >= 0.75,
        "hit@5 = {hits5}/{total} — regression below floor:\n{}",
        misses.join("\n")
    );
    assert!(
        h1 >= 0.60,
        "hit@1 = {hits1}/{total} — regression below floor"
    );
    assert!(mrr5 >= 0.65, "MRR@5 = {mrr5:.3} — regression below floor");
    let floor = |c: Cat| {
        let s = &cats.iter().find(|(cc, _)| *cc == c).unwrap().1;
        s.hits5 as f64 / s.total as f64
    };
    assert!(floor(Cat::Exact) >= 0.85, "Exact category regressed");
    assert!(
        floor(Cat::Morphology) >= 0.85,
        "Morphology category regressed"
    );
    assert!(
        floor(Cat::Distractor) >= 0.80,
        "Distractor category regressed"
    );
    assert!(
        floor(Cat::MultiHop) >= 0.75,
        "MultiHop category regressed (graph leg)"
    );
}

/// Keyword-only mode: basic matches should remain solid even with semantic off.
#[tokio::test]
async fn retrieval_keyword_only_baseline() {
    let store = seeded_store().await;
    // Exact-token-match queries must be found with keyword-only.
    for (q, want) in [
        ("manti", 16usize),
        ("galata", 19),
        ("traefik", 12),
        ("derailleur", 4),
    ] {
        let res = store
            .recall(&Scope::World, &Query::new(q).limit(5))
            .await
            .unwrap();
        assert!(
            res.iter()
                .any(|s| s.item.searchable_text().contains(CORPUS[want])),
            "keyword-only should find '{q}'"
        );
    }
}
