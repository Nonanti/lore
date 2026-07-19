//! Native, offline embedding: character n-gram feature hashing.
//!
//! Why native? "From scratch" philosophy + no model download/network dependency +
//! fully deterministic/testable. Character n-grams capture morphological
//! variants: "organize" and "organizing" share many 3-grams →
//! high cosine. A real neural embedder (e.g. fastembed) can later plug behind
//! the same `Embedder` trait.

use super::retrieval::tokenize;

/// Abstraction that embeds text into a dense vector.
///
/// Cosine thresholds are embedder-specific: with a hashing embedder, unrelated texts
/// yield ~0 cosine, while neural models like e5 yield ~0.7. Therefore
/// `semantic_gate` / `conflict_band` live in the trait; each embedder
/// overrides them according to its own distribution.
pub trait Embedder: Send + Sync {
    /// Vector dimension.
    fn dim(&self) -> usize;
    /// Embeds text into an L2-normalized vector.
    fn embed(&self, text: &str) -> Vec<f32>;
    /// Semantic candidacy threshold: when no keyword matches, a cosine above this makes it a candidate.
    fn semantic_gate(&self) -> f32 {
        0.40
    }
    /// Conflict band `(low, high)`: similar topic but possibly different information range.
    fn conflict_band(&self) -> (f32, f32) {
        (0.6, 0.9)
    }
    /// Whether token-level cosine fallback is desired for short queries.
    /// In "bag-of-direction" embedders like n-gram hashing, multi-token documents
    /// dilute short queries — token-level comparison compensates for this.
    /// Unnecessary and expensive with true semantic (neural) embedders:
    /// off by default.
    fn token_fallback(&self) -> bool {
        false
    }
    /// Embedder signature: model + dimension. Vectors with different signatures MUST NOT be mixed
    /// (different space/dimension → cosine is meaningless). Persistent stores record the signature and
    /// warn on mismatch; transition via `reembed`.
    fn signature(&self) -> String {
        format!("emb-{}", self.dim())
    }
}

/// Character n-gram feature hashing embedder (signed hashing trick).
#[derive(Clone, Debug)]
pub struct HashingEmbedder {
    dim: usize,
    n: usize,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self { dim: 512, n: 3 }
    }
}

impl HashingEmbedder {
    /// Default (dim=512, n=3).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets dimension and n-gram length.
    pub fn with_params(dim: usize, n: usize) -> Self {
        Self {
            dim: dim.max(1),
            n: n.max(1),
        }
    }
}

/// FNV-1a 64-bit hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Embedder for HashingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn signature(&self) -> String {
        format!("hash-{}-n{}", self.dim, self.n)
    }

    /// N-gram summation dilutes single tokens as the document grows — fallback enabled.
    fn token_fallback(&self) -> bool {
        true
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];

        for tok in tokenize(text) {
            // Word boundary markers: "#matematik#"
            let padded: Vec<char> = format!("#{tok}#").chars().collect();
            if padded.len() < self.n {
                // Very short token: count itself as a single feature.
                let h = fnv1a(tok.as_bytes());
                let bucket = (h % self.dim as u64) as usize;
                let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
                v[bucket] += sign;
                continue;
            }
            for w in padded.windows(self.n) {
                let g: String = w.iter().collect();
                let h = fnv1a(g.as_bytes());
                let bucket = (h % self.dim as u64) as usize;
                let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
                v[bucket] += sign;
            }
        }

        // L2-normalize.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

/// Neural embedder (fastembed/ONNX; compiled only with the `neural` feature).
/// Multilingual model (multilingual-e5-small, 384 dimensions) — genuine semantic
/// similarity including Turkish. Downloads the model on first use (then cached locally).
#[cfg(feature = "neural")]
pub struct NeuralEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    dim: usize,
}

#[cfg(feature = "neural")]
impl NeuralEmbedder {
    /// Initializes with the multilingual default model.
    pub fn new() -> crate::error::Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(false),
        )
        .map_err(|e| crate::error::LoreError::Model(e.to_string()))?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
            dim: 384,
        })
    }
}

#[cfg(feature = "neural")]
impl Embedder for NeuralEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn signature(&self) -> String {
        format!("e5-small-{}", self.dim)
    }

    /// The e5 family has a narrow cosine distribution (unrelated ~0.7): threshold is raised.
    fn semantic_gate(&self) -> f32 {
        0.80
    }

    fn conflict_band(&self) -> (f32, f32) {
        (0.90, 0.97)
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let m = self.model.lock().unwrap();
        let out = match m.embed(vec![text.to_string()], None) {
            Ok(o) => o,
            Err(e) => {
                // Visible warning instead of silent zero vector — record is still written
                // but won't match in semantic recall; the operator should notice.
                tracing::warn!(error = %e, "neural embed error (falling back to zero vector)");
                Vec::new()
            }
        };
        let mut v = out
            .into_iter()
            .next()
            .unwrap_or_else(|| vec![0.0; self.dim]);
        // L2 normalize (safety net — for cosine comparisons).
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::retrieval::cosine;

    #[test]
    fn identical_text_cosine_is_one() {
        let e = HashingEmbedder::new();
        let a = e.embed("User likes math");
        let b = e.embed("User likes math");
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn morphological_variants_are_close() {
        let e = HashingEmbedder::new();
        let q = e.embed("math");
        let doc = e.embed("math");
        let unrelated = e.embed("programming");
        let cos_related = cosine(&q, &doc);
        let cos_unrelated = cosine(&q, &unrelated);
        // Agglutinative variant should be significantly closer than an unrelated word.
        assert!(cos_related > 0.5, "cos_related={cos_related}");
        assert!(
            cos_related > cos_unrelated + 0.3,
            "related={cos_related} unrelated={cos_unrelated}"
        );
    }

    #[test]
    fn signatures_identify_space() {
        assert_eq!(HashingEmbedder::new().signature(), "hash-512-n3");
        assert_eq!(
            HashingEmbedder::with_params(64, 3).signature(),
            "hash-64-n3"
        );
        assert_ne!(
            HashingEmbedder::new().signature(),
            HashingEmbedder::with_params(64, 3).signature(),
            "different spaces carry different signatures"
        );
    }

    #[test]
    fn dim_is_respected() {
        let e = HashingEmbedder::with_params(128, 3);
        assert_eq!(e.dim(), 128);
        assert_eq!(e.embed("hello").len(), 128);
    }

    /// Skipped by default because it downloads a model:
    /// `cargo test --features neural -- --ignored`
    #[cfg(feature = "neural")]
    #[test]
    #[ignore = "downloads model (network + ~100MB)"]
    fn neural_embedder_semantics() {
        let e = NeuralEmbedder::new().expect("model must initialize");
        assert_eq!(e.dim(), 384);
        let q = e.embed("math");
        let rel = e.embed("interest in mathematics");
        let unrel = e.embed("where to buy cat food");
        assert!(cosine(&q, &rel) > cosine(&q, &unrel), "semantic ranking");
    }
}
