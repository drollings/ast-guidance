//! Embedding vector math.
//!
//! The canonical home for cosine similarity, brute-force KNN, vector↔byte
//! encoding, and int8 quantization. This module was moved verbatim from
//! `search-vector::math` so that the dependency direction stays acyclic:
//! `search-vector` and `coral` depend on `fluent-db`, never the reverse
//! (M4 deleted the `search-vector::math` / `search-vector::error` shims;
//! consume this module directly).
//!
//! The generic scored top-K select lives in `common-core::score`
//! (spacy-rs composes it from there — `fluent-db`'s sqlite/HNSW weight must
//! not leak into the deterministic spine); it is re-exported here so
//! `fluent-db` consumers have one import surface.

pub use common_core::score::top_k_by_score;
pub use common_core::vector_math::cosine_similarity_f32;

/// Vector index abstraction (M7) — `knn` via cosine distance.
pub trait VectorIndex {
    fn knn(&self, query: &[f32], k: usize) -> Vec<fluent_types::KnnHit>;
}

/// Brute-force index wrapping `knn_brute_force`.
pub struct BruteForceIndex {
    pub entries: Vec<(fluent_types::NodeId, Vec<f32>, String)>,
}
impl VectorIndex for BruteForceIndex {
    fn knn(&self, query: &[f32], k: usize) -> Vec<fluent_types::KnnHit> {
        let results = knn_brute_force(query, self.entries.iter().map(|(id, emb, _)| (*id, emb.as_slice())), k);
        results.into_iter().map(|(id, dist)| {
            let name = self.entries.iter().find(|(nid,_,_)| *nid==id).map(|(_,_,n)| n.clone()).unwrap_or_default();
            fluent_types::KnnHit { node_id: id, distance: dist, name: name.into() }
        }).collect()
    }
}

/// Brute-force KNN search: compute cosine distance from `query` to every
/// candidate, sort ascending, and return the top `k` as `(id, distance)`.
///
/// `candidates` yields `(id, embedding_slice)` pairs borrowing the caller's
/// data — no full candidate-list clone. Candidates whose embedding length
/// differs from `query` are silently skipped.
pub fn knn_brute_force<'a, Id: Clone>(
    query: &[f32],
    candidates: impl Iterator<Item = (Id, &'a [f32])>,
    k: usize,
) -> Vec<(Id, f32)> {
    if query.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut results: Vec<(Id, f32)> = candidates
        .filter_map(|(id, emb)| {
            if emb.len() != query.len() {
                return None;
            }
            let distance = 1.0 - cosine_similarity_f32(query, emb);
            Some((id, distance))
        })
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Map a canonical cosine *distance* (`1 - cosine`, as returned by
/// `HnswIndex`/`hnsw_lookup`/`knn_brute_force`) to cosine *similarity*.
///
/// Presentation-only and deliberately unclamped: out-of-range distances stay
/// out-of-range so raw-distance shapes (`KnnHit.distance`, node-store probes)
/// are preserved bit-for-bit. Call sites that already clamp compose
/// [`distance_to_similarity_clamped`] instead — no new semantics either way.
pub fn distance_to_similarity(d: f32) -> f32 {
    1.0 - d
}

/// Clamped twin of [`distance_to_similarity`]: floors at `0.0` for the chart
/// path that already clamps (`(1.0 - d).max(0.0)`). Exists *only* for that
/// path — new call sites default to the unclamped mapping.
pub fn distance_to_similarity_clamped(d: f32) -> f32 {
    (1.0 - d).max(0.0)
}

/// Thin `hnsw_lookup` post-mapper: applies `map` (one of the two mappings
/// above) to each resolved `(Id, distance)` pair, preserving ids and order.
///
/// Retires the per-site `1.0 - d` closures so the mapping lives in exactly
/// one place. Sort/truncate/filter stay call-site code.
pub fn scored_hits<Id: Clone>(resolved: Vec<(Id, f32)>, map: impl Fn(f32) -> f32) -> Vec<(Id, f32)> {
    resolved.into_iter().map(|(id, d)| (id, map(d))).collect()
}

pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn try_bytes_to_vec(b: &[u8]) -> Option<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return None;
    }
    Some(bytes_to_vec(b))
}

#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    pub values: Vec<i8>,
    pub scale: f32,
    pub dimensions: usize,
}

impl QuantizedEmbedding {
    pub fn from_f32(vec: &[f32]) -> Self {
        if vec.is_empty() {
            return Self {
                values: Vec::new(),
                scale: 1.0,
                dimensions: 0,
            };
        }
        let max_abs = vec.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
        let values: Vec<i8> = vec
            .iter()
            .map(|v| (v * scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        Self {
            dimensions: vec.len(),
            values,
            scale,
        }
    }
    pub fn to_f32(&self) -> Vec<f32> {
        if self.scale == 0.0 {
            return vec![0.0; self.dimensions];
        }
        let inv = 1.0 / self.scale;
        self.values.iter().map(|&v| f32::from(v) * inv).collect()
    }
}

pub fn cosine_similarity_q8(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> f32 {
    if a.dimensions != b.dimensions || a.dimensions == 0 {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0i64, 0i64, 0i64);
    for (x, y) in a.values.iter().zip(b.values.iter()) {
        let (xi, yi) = (i64::from(*x), i64::from(*y));
        dot += xi * yi;
        na += xi * xi;
        nb += yi * yi;
    }
    let mag = ((na as f64) * (nb as f64)).sqrt();
    if mag == 0.0 {
        0.0
    } else {
        (dot as f64 / mag) as f32
    }
}

/// Reciprocal Rank Fusion (RRF): merges two ranked candidate lists into a
/// single fused ranking.
///
/// RRF score = sum(1 / (k + rank(engine))) for each id appearing in either
/// list; ids not present in a list contribute 0. On id collision the item
/// from the first (`keyword_results`) list wins and the scores are summed.
/// Results are returned sorted descending by score.
///
/// The inputs are `(id, item)` pairs in rank order (index 0 = best rank), so
/// this is generic over the candidate item type — `search-vector`'s
/// `SearchResult` and coral's `KnnHit` both fuse through this single
/// implementation.
/// Task relevance threshold (axis B) — cosine distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskRelevanceThreshold(pub f32);
impl TaskRelevanceThreshold {
    pub fn new(v: f32) -> Result<Self,String>{ if (0.0..=1.0).contains(&v){ Ok(Self(v))} else { Err(format!("TaskRelevanceThreshold {v} out of range")) } }
    pub fn get(self)->f32{ self.0 }
}
/// Producer confidence threshold (axis A).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProducerConfidenceThreshold(pub f32);
impl ProducerConfidenceThreshold {
    pub fn new(v: f32) -> Result<Self,String>{ if (0.0..=1.0).contains(&v){ Ok(Self(v))} else { Err(format!("ProducerConfidenceThreshold {v} out of range")) } }
    pub fn get(self)->f32{ self.0 }
}

pub fn rrf_merge<T>(
    keyword_results: Vec<(i64, T)>,
    vector_results: Vec<(i64, T)>,
    k_constant: f64,
) -> Vec<(f64, T)> {
    use std::collections::HashMap;

    let mut rrf_scores: HashMap<i64, (f64, T)> = HashMap::new();

    for (rank, (id, item)) in keyword_results.into_iter().enumerate() {
        rrf_scores.insert(id, (1.0 / (k_constant + rank as f64), item));
    }

    for (rank, (id, item)) in vector_results.into_iter().enumerate() {
        let score = 1.0 / (k_constant + rank as f64);
        let entry = rrf_scores.entry(id).or_insert_with(|| (0.0, item));
        entry.0 += score;
    }

    let mut merged: Vec<(f64, T)> = rrf_scores.into_values().collect();
    merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/vector.rs"]
mod tests;
