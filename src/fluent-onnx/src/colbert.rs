//! ColBERT late-interaction retriever — per-token 128-d vectors with MaxSim
//! scoring (ROADMAP_20260827_ORT §5.2).
//!
//! The model produces per-token embeddings (after a dense projection from the
//! base encoder's hidden states), and scoring uses **MaxSim**: for each query
//! token, find the most similar document token (cosine similarity), then
//! average those maxima. This is the standard ColBERT similarity metric.
//!
//! The pure scoring math is unit-tested without a model; the session-facing
//! glue lives behind the `onnx` feature.

use std::collections::HashMap;
use std::sync::Mutex;
use common_core::vector_math::cosine_similarity_f32;

/// Default ColBERT per-token dimension (after the Dense projection: 1024 → 128).
#[cfg(feature = "onnx")]
const DEFAULT_COLBERT_DIMS: u32 = 128;

/// Maximum sequence length for ColBERT (the `sentence_bert_config.json`
/// contract: `max_seq_length: 511`).
#[cfg(feature = "onnx")]
const COLBERT_MAX_SEQ_LEN: usize = 511;

// ── Pure math (ort-free) ───────────────────────────────────────────────

/// L2-normalize a vector in place. A zero vector stays zero (never NaN).
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// MaxSim score: for each query token, find the maximum cosine similarity
/// against all document tokens, then average those maxima.
///
/// Both `query_tokens` and `doc_tokens` are expected to be L2-normalized
/// (the ColBERT model's output is L2-normalized per-token). When either
/// side is empty, returns 0.0.
pub fn maxsim_score(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let q_len = query_tokens.len();
    let mut sum = 0.0f32;
    for qt in query_tokens {
        let max_sim = doc_tokens
            .iter()
            .map(|dt| cosine_similarity_f32(qt, dt))
            .fold(f32::NEG_INFINITY, f32::max);
        sum += max_sim;
    }
    sum / q_len as f32
}

/// MaxSim score with normalization to [0, 1] range. The raw MaxSim average
/// is in [-1, 1] for cosine; this maps it to [0, 1] for consistent ranking.
pub fn maxsim_score_normalized(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    let raw = maxsim_score(query_tokens, doc_tokens);
    f32::midpoint(raw, 1.0)
}

// ── CachedColbert — LRU cache over encoded doc tokens ──────────────────

/// Bounded LRU cache for encoded ColBERT document tokens.
///
/// Keys are content-hashed strings (the same convention as
/// `CachedEmbeddingProvider`); values are per-token 128-d vectors. The cache
/// is keyed by a content hash + model name to avoid cross-model collisions.
pub struct CachedColbert {
    cache: Mutex<HashMap<String, Vec<Vec<f32>>>>,
    capacity: usize,
    insertion_order: Mutex<Vec<String>>,
}

impl CachedColbert {
    /// Create a cache with the given capacity (number of documents).
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::with_capacity(capacity)),
            capacity,
            insertion_order: Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    /// Look up cached token embeddings for a key.
    pub fn get(&self, key: &str) -> Option<Vec<Vec<f32>>> {
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.get(key).cloned()
    }

    /// Insert token embeddings into the cache, evicting the oldest entry when
    /// at capacity.
    #[allow(clippy::map_entry)]
    pub fn insert(&self, key: String, tokens: Vec<Vec<f32>>) {
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut order = self
            .insertion_order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // If the key already exists, just update it.
        if cache.contains_key(&key) {
            cache.insert(key, tokens);
            return;
        }

        // Evict oldest when at capacity.
        while cache.len() >= self.capacity {
            if let Some(oldest) = order.first().cloned() {
                order.remove(0);
                cache.remove(&oldest);
            } else {
                break;
            }
        }

        order.push(key.clone());
        cache.insert(key, tokens);
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── ColbertRetriever (behind `onnx` feature) ───────────────────────────

/// Remove [CLS], [SEP], and [PAD] tokens from the encoded sequence.
/// These are special tokens whose ids are typically 0, 1, 2 in the
/// LFM tokenizer. We detect them via the attention mask (PAD) and the
/// tokenizer's special token handling.
#[cfg(feature = "onnx")]
fn strip_special_tokens(
    tokens: Vec<Vec<f32>>,
    encoding: &crate::tokenizer::LfmEncoding,
) -> Vec<Vec<f32>> {
    tokens
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            let mask = encoding.attention_mask.get(*i).copied().unwrap_or(0);
            let id = encoding.ids.get(*i).copied().unwrap_or(0);
            // Keep only real (non-pad) tokens that are not special tokens.
            // Special tokens have id 0 (PAD), 1 (CLS), 2 (SEP) in the
            // LFM tokenizer. We filter by mask == 1 AND id >= 3 as a
            // heuristic for "content token".
            mask == 1 && id >= 3
        })
        .map(|(_, tokens)| tokens)
        .collect()
}

#[cfg(feature = "onnx")]
mod ort_colbert {
    use std::sync::{Arc, Mutex};

    use crate::config::OnnxConfig;
    use crate::error::OrtError;
    use crate::session::SessionHandle;
    use crate::tokenizer::{LfmEncoding, LfmTokenizer};

    use super::{
        l2_normalize, maxsim_score, strip_special_tokens, ConceptEncoding, EntitySimilarityIndex,
        COLBERT_MAX_SEQ_LEN, DEFAULT_COLBERT_DIMS,
    };

    /// ColBERT late-interaction retriever: encodes text into per-token 128-d
    /// vectors and scores document-query pairs via MaxSim.
    ///
    /// The model architecture is the LFM base encoder + a Dense projection
    /// (1024 → 128, no bias). The session outputs `token_embeddings` — the
    /// per-token projected vectors.
    pub struct ColbertRetriever {
        session: Arc<Mutex<ort::session::Session>>,
        tokenizer: Arc<LfmTokenizer>,
        dims: u32,
        name: &'static str,
        model_key: String,
    }

    impl ColbertRetriever {
        /// Build a retriever from an already-loaded registry session handle.
        pub fn from_handle(
            handle: &SessionHandle,
            config: &OnnxConfig,
            model_key: &str,
        ) -> Result<Self, OrtError> {
            let session = handle
                .downcast_arc::<Mutex<ort::session::Session>>()
                .ok_or_else(|| {
                    OrtError::Other("session handle does not hold an ort session".to_string())
                })?;
            let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
                OrtError::Other(format!("colbert retriever '{model_key}' missing tokenizer_path"))
            })?;
            let tokenizer = LfmTokenizer::from_file(tokenizer_path, COLBERT_MAX_SEQ_LEN)?;
            let dims = config.dims.unwrap_or(DEFAULT_COLBERT_DIMS);
            let name: &'static str = Box::leak(format!("colbert:{model_key}").into_boxed_str());
            Ok(Self {
                session,
                tokenizer,
                dims,
                name,
                model_key: model_key.to_string(),
            })
        }

        /// The per-token embedding dimension.
        pub fn dims(&self) -> u32 {
            self.dims
        }

        /// The model name (for cache keys).
        pub fn name(&self) -> &'static str {
            self.name
        }

        /// Encode a single text into per-token L2-normalized 128-d vectors.
        ///
        /// Returns one `Vec<Vec<f32>>` where each inner vec is a `dims`-length
        /// token embedding. Special tokens ([CLS], [SEP], [PAD]) are included
        /// — the caller decides whether to strip them.
        fn run_encoding(&self, encoding: &LfmEncoding) -> Result<Vec<Vec<f32>>, OrtError> {
            let seq = encoding.len().max(1);
            let ids: Vec<i64> = encoding.ids.iter().map(|&i| i64::from(i)).collect();
            let mask: Vec<i64> = encoding.attention_mask.iter().map(|&m| i64::from(m)).collect();
            let shape = [1usize, seq];
            let input = ort::value::Tensor::from_array((shape, ids))
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let mask_t = ort::value::Tensor::from_array((shape, mask))
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;

            // The ColBERT export produces `token_embeddings` — per-token
            // projected vectors of shape [1, seq, dims].
            let (_, embeddings) = outputs["token_embeddings"]
                .try_extract_tensor::<f32>()
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;

            let dims = self.dims as usize;
            let mut tokens = Vec::with_capacity(seq);
            for t in 0..seq {
                let start = t * dims;
                let end = start + dims;
                let mut row: Vec<f32> = embeddings[start..end].to_vec();
                l2_normalize(&mut row);
                tokens.push(row);
            }
            Ok(tokens)
        }

        /// Encode a single text, returning per-token embeddings (including
        /// specials). The caller should strip [CLS]/[SEP] for MaxSim scoring.
        pub fn encode(&self, text: &str) -> Result<Vec<Vec<f32>>, OrtError> {
            let encoding = self
                .tokenizer
                .encode(text)
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            self.run_encoding(&encoding)
        }

        /// Encode a query text for scoring. Strips special tokens ([CLS],
        /// [SEP], [PAD]) so only content tokens participate in MaxSim.
        pub fn encode_query(&self, text: &str) -> Result<Vec<Vec<f32>>, OrtError> {
            let encoding = self
                .tokenizer
                .encode(text)
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let tokens = self.run_encoding(&encoding)?;
            Ok(strip_special_tokens(tokens, &encoding))
        }

        /// Encode multiple documents, returning per-document per-token
        /// embeddings. Dynamic batch: pads to the longest document in the
        /// batch.
        pub fn encode_docs(
            &self,
            texts: &[&str],
        ) -> Result<Vec<Vec<Vec<f32>>>, OrtError> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let encodings: Vec<LfmEncoding> = texts
                .iter()
                .map(|t| self.tokenizer.encode(t))
                .collect::<Result<_, OrtError>>()
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;

            let batch = encodings.len();
            let max_len = encodings.iter().map(LfmEncoding::len).max().unwrap_or(0).max(1);
            let dims = self.dims as usize;

            // Build batched tensors.
            let mut ids = vec![0i64; batch * max_len];
            let mut mask = vec![0i64; batch * max_len];
            for (bi, enc) in encodings.iter().enumerate() {
                for (si, &id) in enc.ids.iter().enumerate() {
                    ids[bi * max_len + si] = i64::from(id);
                }
                for (si, &m) in enc.attention_mask.iter().enumerate() {
                    mask[bi * max_len + si] = i64::from(m);
                }
            }

            let shape = [batch, max_len];
            let input = ort::value::Tensor::from_array((shape, ids))
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let mask_t = ort::value::Tensor::from_array((shape, mask))
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;
            let (_, embeddings) = outputs["token_embeddings"]
                .try_extract_tensor::<f32>()
                .map_err(|e| OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: e.to_string(),
                })?;

            let mut all_docs = Vec::with_capacity(batch);
            for (bi, enc) in encodings.iter().enumerate() {
                let mut doc_tokens = Vec::with_capacity(max_len);
                for t in 0..max_len {
                    let start = (bi * max_len + t) * dims;
                    let end = start + dims;
                    if start < embeddings.len() && end <= embeddings.len() {
                        let mut row: Vec<f32> = embeddings[start..end].to_vec();
                        l2_normalize(&mut row);
                        doc_tokens.push(row);
                    }
                }
                // Strip specials per-document.
                doc_tokens = strip_special_tokens(doc_tokens, enc);
                all_docs.push(doc_tokens);
            }
            Ok(all_docs)
        }

        /// Score a query against a document via MaxSim.
        pub fn score(&self, query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
            maxsim_score(query_tokens, doc_tokens)
        }
    }

    /// Build a `ColbertRetriever` from the registry's session for `model_key`,
    /// if the model is registered and its task is `LateInteraction`.
    pub fn build_colbert(
        config: &OnnxConfig,
        model_key: &str,
        handle: &SessionHandle,
    ) -> Result<ColbertRetriever, OrtError> {
        ColbertRetriever::from_handle(handle, config, model_key)
    }

    /// Build the ColBERT retriever from the registry for `model_key`.
    /// Returns `Ok(None)` when the model is not registered or not a
    /// `LateInteraction` task.
    pub fn build_colbert_from_registry(
        registry: &crate::session::OrtSessionRegistry,
        model_key: &str,
    ) -> Result<Option<ColbertRetriever>, OrtError> {
        let Some(config) = registry.config(model_key) else {
            return Ok(None);
        };
        if config.task != crate::config::OnnxTask::LateInteraction {
            return Ok(None);
        }
        let Some(handle) = registry.ensure_loaded(model_key)? else {
            return Ok(None);
        };
        build_colbert(&config, model_key, &handle).map(Some)
    }

    /// Bake an [`EntitySimilarityIndex`] over concept labels (ROADMAP
    /// 20260828_ORT_FIXES M3.1). **Data-time:** each concept's **label** is
    /// encoded once via batched [`ColbertRetriever::encode_docs`] (which L2-
    /// normalizes per token and strips [CLS]/[SEP]/[PAD]); the resulting
    /// `(namespace, canonical, token_embeddings)` triples are stored and the
    /// index is read-only at query time.
    ///
    /// `concepts` is `(namespace, canonical, label)` triples — the label is
    /// the surface text encoded for MaxSim matching, while a hit surfaces the
    /// `(namespace, canonical)` identity (which the caller resolves to an
    /// `InterlinguaId` through its store). Empty input yields an empty index
    /// (fail-open).
    pub fn bake_entity_index(
        retriever: &ColbertRetriever,
        concepts: &[(String, String, String)],
        threshold: f32,
    ) -> Result<EntitySimilarityIndex, OrtError> {
        if concepts.is_empty() {
            return Ok(EntitySimilarityIndex::empty(threshold));
        }
        let labels: Vec<&str> = concepts.iter().map(|(_, _, label)| label.as_str()).collect();
        let encoded = retriever.encode_docs(&labels)?;
        let entries = concept_encodings_from_docs(concepts, encoded);
        Ok(EntitySimilarityIndex::new(entries, threshold))
    }

    /// Assemble [`ConceptEncoding`] entries from already-encoded per-doc
    /// tokens. Pure — split out from [`bake_entity_index`] so the triple→entry
    /// mapping is hermetically testable without a session.
    pub(super) fn concept_encodings_from_docs(
        concepts: &[(String, String, String)],
        encoded: Vec<Vec<Vec<f32>>>,
    ) -> Vec<ConceptEncoding> {
        concepts
            .iter()
            .zip(encoded)
            .map(|((namespace, canonical, _), token_embeddings)| ConceptEncoding {
                namespace: namespace.clone(),
                canonical: canonical.clone(),
                token_embeddings,
            })
            .collect()
    }
}

#[cfg(feature = "onnx")]
pub use ort_colbert::{
    bake_entity_index, build_colbert, build_colbert_from_registry, ColbertRetriever,
};

// ── Entity-similarity fallback (M5.4, data-time) ───────────────────────

/// Boot-baked concept-label ColBERT encodings for entity-similarity lookup.
///
/// At boot, concept labels (YagoEntity/YagoClass names) are encoded into
/// per-token ColBERT vectors and stored in this index. At query time,
/// a text span is encoded and scored against the baked set via MaxSim.
/// Matches above the configured `threshold` are returned as entity-link
/// candidates.
///
/// This is a **data-time** artifact: the encodings are baked once (never
/// re-encoded at runtime), and the index is read-only at query time. The
/// M6 overlay/candidate table is the durable surface; this index is the
/// in-memory lookup structure.
///
/// **Not yet wired into the overlay pipeline** — the full entity-link
/// overlay worker (M6.2) consumes this index when the M6 candidate table
/// infrastructure lands.
pub struct EntitySimilarityIndex {
    /// Pre-encoded concept labels: (namespace, canonical_name, per_token_embeddings).
    entries: Vec<ConceptEncoding>,
    /// Maximum cosine similarity threshold for a match.
    threshold: f32,
}

/// A pre-encoded concept label with its namespace and canonical name.
#[derive(Debug, Clone)]
pub struct ConceptEncoding {
    /// The interlingua namespace (e.g., `YagoEntity`, `YagoClass`).
    pub namespace: String,
    /// The canonical concept name (e.g., `"schema:Person"`).
    pub canonical: String,
    /// Per-token ColBERT embeddings (L2-normalized, specials stripped).
    pub token_embeddings: Vec<Vec<f32>>,
}

/// The result of an entity-similarity lookup.
#[derive(Debug, Clone)]
pub struct EntitySimilarityHit {
    pub namespace: String,
    pub canonical: String,
    pub score: f32,
}

impl EntitySimilarityIndex {
    /// Create an empty index (no concepts baked). Fail-open: all lookups
    /// return empty.
    pub fn empty(threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            threshold,
        }
    }

    /// Build from pre-baked concept encodings (data-time: encoded once at
    /// boot or build time, stored as a versioned data artifact).
    pub fn new(entries: Vec<ConceptEncoding>, threshold: f32) -> Self {
        Self { entries, threshold }
    }

    /// Look up the closest concept(s) for a text span's ColBERT token
    /// embeddings. Returns candidates above `self.threshold`, sorted by
    /// descending score. First-wins: the first match per namespace is the
    /// canonical candidate.
    pub fn lookup(&self, query_tokens: &[&[f32]]) -> Vec<EntitySimilarityHit> {
        if query_tokens.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<EntitySimilarityHit> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let doc_refs: Vec<&[f32]> =
                    entry.token_embeddings.iter().map(Vec::as_slice).collect();
                let score = maxsim_score(query_tokens, &doc_refs);
                if score >= self.threshold {
                    Some(EntitySimilarityHit {
                        namespace: entry.namespace.clone(),
                        canonical: entry.canonical.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Number of baked concept encodings.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────
// ── Factory tests (hermetic, onnx-gated) ───────────────────────────────

#[cfg(all(test, feature = "onnx"))]
mod factory_tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::{OnnxConfig, OnnxTask, Quant};
    use crate::error::OrtError;
    use crate::session::{OrtSessionRegistry, SessionHandle, SessionLoader};

    #[derive(Default)]
    struct StubLoader;

    impl SessionLoader for StubLoader {
        fn load(&self, _config: &OnnxConfig, _model_key: &str) -> Result<SessionHandle, OrtError> {
            Ok(SessionHandle::new("not an ort session"))
        }
    }

    fn config_for(task: OnnxTask) -> OnnxConfig {
        OnnxConfig::new()
            .model_path("/models/test.onnx")
            .tokenizer_path("/models/tokenizer.json")
            .task(task)
            .quantization(Quant::Q8)
            .build()
    }

    #[test]
    fn unregistered_model_yields_none() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        let colbert = build_colbert_from_registry(&registry, "missing").expect("no error");
        assert!(colbert.is_none());
    }

    #[test]
    fn non_colbert_task_yields_none() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        registry
            .register("encoder", config_for(OnnxTask::FillMask))
            .expect("register");
        let colbert = build_colbert_from_registry(&registry, "encoder").expect("no error");
        assert!(colbert.is_none());
    }

    #[test]
    fn wrong_handle_type_is_a_loud_error() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        registry
            .register("colbert", config_for(OnnxTask::LateInteraction))
            .expect("register");
        let result = build_colbert_from_registry(&registry, "colbert");
        assert!(result.is_err());
    }

    #[test]
    fn concept_encodings_preserve_identity_order() {
        let concepts = vec![
            ("YagoEntity".into(), "yago:Paris".into(), "Paris".into()),
            ("YagoClass".into(), "schema:Person".into(), "a person".into()),
        ];
        let encoded = vec![vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0], vec![1.0, 0.0]]];
        let entries = super::ort_colbert::concept_encodings_from_docs(&concepts, encoded);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].namespace, "YagoEntity");
        assert_eq!(entries[0].canonical, "yago:Paris");
        assert_eq!(entries[0].token_embeddings, vec![vec![1.0, 0.0]]);
        assert_eq!(entries[1].canonical, "schema:Person");
        assert_eq!(entries[1].token_embeddings.len(), 2);
    }

    #[test]
    fn concept_encodings_zip_mismatch_is_truncated() {
        // The doc count and triple count must agree in production; a shorter
        // encoded side just truncates (defensive) rather than panicking.
        let concepts = vec![
            ("YagoEntity".into(), "a".into(), "a".into()),
            ("YagoEntity".into(), "b".into(), "b".into()),
        ];
        let encoded = vec![vec![vec![1.0]]];
        let entries = super::ort_colbert::concept_encodings_from_docs(&concepts, encoded);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].canonical, "a");
    }

    #[test]
    fn baked_index_lookup_round_trip() {
        // The full bake path minus the session: entries built from encoded
        // tokens land in an index whose lookup returns the canonical identity.
        let concepts = vec![("YagoEntity".into(), "yago:Paris".into(), "Paris".into())];
        let encoded = vec![vec![vec![1.0, 0.0]]];
        let entries = super::ort_colbert::concept_encodings_from_docs(&concepts, encoded);
        let index = EntitySimilarityIndex::new(entries, 0.8);
        let q: Vec<&[f32]> = vec![&[0.99, 0.14]];
        let hits = index.lookup(&q);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].canonical, "yago:Paris");
        assert!(hits[0].score >= 0.8);
    }
}

#[cfg(test)]
#[path = "../tests/colbert.rs"]
mod tests;
