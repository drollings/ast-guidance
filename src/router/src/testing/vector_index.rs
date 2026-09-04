//! Test-only vector indices — the `VectorIndex` store seam and its two
//! implementations (M5.7).
//!
//! Moved here from the production path: no production caller ever consumed
//! this trait (`ranking`/`retrieval` use ledger-backed signals, not vectors),
//! so leaving it as a public production module invited dead-seam reuse. When
//! a production vector backend is actually needed, promote one implementation
//! back behind a real call site — do not add a third test-only shape here.
//!
//! M5.2: this trait is the stateful **store seam** (`insert`/`search`/`len`,
//! `NodeId`-keyed, similarity-mapped) — deliberately distinct from
//! `fluent_db::vector::VectorIndex::knn`, the stateless **math seam**. Both
//! implementations compose the shared primitives
//! (`fluent_db::hnsw::HnswIndex` + `fluent_db::vector::knn_brute_force`)
//! instead of re-implementing top-K or cosine math.

use std::sync::Mutex;

use fluent_db::hnsw::HnswIndex;
use fluent_types::NodeId;

pub trait VectorIndex: Send + Sync {
    fn insert(&self, id: NodeId, emb: &[f32]) -> Result<(), fluent_db::error::DbError>;
    fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f64)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Byte-identical brute-force implementation — the current `ranking.rs` behavior.
pub struct BruteForceIndex {
    entries: Mutex<Vec<(NodeId, Vec<f32>)>>,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
}

impl Default for BruteForceIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for BruteForceIndex {
    fn insert(&self, id: NodeId, emb: &[f32]) -> Result<(), fluent_db::error::DbError> {
        common_core::sync::lock(&self.entries).push((id, emb.to_vec()));
        Ok(())
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f64)> {
        let entries = common_core::sync::lock(&self.entries);
        let results = fluent_db::vector::knn_brute_force(
            query,
            entries.iter().map(|(id, e)| (*id, e.as_slice())),
            k,
        );
        fluent_db::vector::scored_hits(results, fluent_db::vector::distance_to_similarity)
            .into_iter()
            .map(|(id, similarity)| (id, f64::from(similarity)))
            .collect()
        // distance = 1 - cosine_similarity, so similarity = 1 - dist
    }
    fn len(&self) -> usize {
        common_core::sync::lock(&self.entries).len()
    }
}

/// HNSW-backed ANN index composing `fluent_db::hnsw::HnswIndex`.
pub struct HnswVectorIndex {
    inner: HnswIndex,
}

impl HnswVectorIndex {
    pub fn new() -> Self {
        Self { inner: HnswIndex::new() }
    }
    pub fn is_built(&self) -> bool {
        self.inner.is_built()
    }
}

impl Default for HnswVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for HnswVectorIndex {
    fn insert(&self, id: NodeId, emb: &[f32]) -> Result<(), fluent_db::error::DbError> {
        self.inner.insert(id.as_int(), emb);
        Ok(())
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f64)> {
        // M5: the HNSW probe + id resolution is the shared
        // `fluent_db::hnsw::hnsw_lookup` (`None` = no fallback here — the
        // test seam returns whatever the graph yields, empty when unbuilt).
        fluent_db::hnsw::hnsw_lookup(&self.inner, query, k)
            .map(|resolved| {
                fluent_db::vector::scored_hits(resolved, fluent_db::vector::distance_to_similarity)
                    .into_iter()
                    .map(|(raw, similarity)| (NodeId::from_int(raw), f64::from(similarity)))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
#[path = "../../tests/vector_index.rs"]
mod tests;
