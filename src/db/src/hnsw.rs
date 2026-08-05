//! The canonical HNSW-backed vector index store (D6).
//!
//! `HnswIndex` owns the cosine-distance `Hnsw` index plus its external-id
//! mapping (`id_map`: HNSW `d_id` → caller node id), and exposes
//! `insert`/`rebuild_from`/`search`/`is_built`/`len`/`id_map_snapshot`. It
//! generalizes the two near-verbatim copies that `GuidanceDb`
//! (`search-vector/src/db.rs`) and coral's `Library`
//! (`coral/src/db/hnsw.rs`) previously each maintained by hand.
//!
//! ## Lock ordering (R9)
//!
//! **`hnsw` → `id_map`, never inverted.** The index guard is acquired first,
//! the id-map guard second. This mirrors the documented discipline in
//! `coral/src/db/hnsw.rs`; the database connection lock, when held, is taken
//! *after* both (`hnsw → id_map → conn`). The `search` method returns
//! `(external idx, distance)` pairs after releasing both index guards, so
//! consumers resolve ids through `id_map` and then touch the connection
//! without ever inverting the order.

use std::sync::{Mutex, RwLock};

use anndists::dist::DistCosine;
use common_core::constants::HnswParams;
use common_core::sync::lock;
use hnsw_rs::hnsw::Hnsw;

use crate::error::DbError;

/// A generic HNSW-backed vector index with external-id mapping.
///
/// The index is built lazily on first `insert` (mirroring the historical
/// `get_or_insert_with` behavior) and wholesale-rebuilt by `rebuild_from`.
#[derive(Default)]
pub struct HnswIndex {
    hnsw: RwLock<Option<Hnsw<'static, f32, DistCosine>>>,
    id_map: Mutex<Vec<i64>>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("built", &self.is_built())
            .field("len", &self.len())
            .finish()
    }
}

impl HnswIndex {
    /// An empty, unbuilt index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `node_id` with embedding `embedding`, creating the index on
    /// first use. Returns the external index (`d_id`) assigned to the point.
    ///
    /// Lock order: `hnsw` (write) → `id_map`.
    pub fn insert(&self, node_id: i64, embedding: &[f32]) -> usize {
        let mut guard = self
            .hnsw
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let hnsw = guard.get_or_insert_with(|| {
            let p = HnswParams::default();
            common_core::sqlite::make_hnsw(&p, p.initial_capacity)
        });

        let external_id = {
            let mut id_map = lock(&self.id_map);
            let idx = id_map.len();
            id_map.push(node_id);
            idx
        };

        hnsw.insert((embedding, external_id));
        external_id
    }
    /// Rebuild the index from a table scan of `(node_id, embedding_blob)`
    /// rows. `decode` turns each blob into an `Option<Vec<f32>>` embedding —
    /// `None` (or an empty vector) skips the row. Returns the total number of
    /// rows examined (matching the historical `rows.len()` return).
    ///
    /// Lock order: `hnsw` (write) → `id_map`.
    pub fn rebuild_from(
        &self,
        rows: impl Iterator<Item = (i64, Vec<u8>)>,
        decode: fn(&[u8]) -> Option<Vec<f32>>,
    ) -> Result<usize, DbError> {
        let rows: Vec<(i64, Vec<u8>)> = rows.collect();
        let count = rows.len();

        let p = HnswParams::default();
        let hnsw = common_core::sqlite::make_hnsw(&p, count.max(p.initial_capacity));

        let mut id_map = Vec::with_capacity(count);
        for (node_id, blob) in rows {
            if let Some(embedding) = decode(&blob) {
                if !embedding.is_empty() {
                    hnsw.insert((&embedding, id_map.len()));
                    id_map.push(node_id);
                }
            }
        }

        *self
            .hnsw
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(hnsw);
        *lock(&self.id_map) = id_map;

        Ok(count)
    }

    /// Approximate nearest-neighbour search. Returns `(external idx, distance)`
    /// pairs — resolve each `idx` through `id_map` to recover the caller's
    /// node id.
    ///
    /// Lock order: `hnsw` (read) → `id_map`. Both guards are released before
    /// the caller touches its connection, so the `hnsw → id_map → conn`
    /// ordering is preserved.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let guard = match self.hnsw.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let Some(hnsw) = guard.as_ref() else {
            return Vec::new();
        };
        // Hold the id-map guard for the duration of the index traversal so a
        // concurrent rebuild cannot swap the index out from under the id
        // resolution. Lock order: hnsw (read) → id_map.
        let _id_map = lock(&self.id_map);

        hnsw.search(query, k, k)
            .into_iter()
            .map(|n| (n.d_id, n.distance))
            .collect()
    }

    /// Whether the index has been built (any `insert`/`rebuild_from` ran).
    pub fn is_built(&self) -> bool {
        self.hnsw.read().is_ok_and(|g| g.is_some())
    }

    /// Number of points currently in the index.
    pub fn len(&self) -> usize {
        self.hnsw
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(Hnsw::get_nb_point))
            .unwrap_or(0)
    }

    /// Whether the index contains no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A snapshot of the external-id map (`d_id` → caller node id).
    pub fn id_map_snapshot(&self) -> Vec<i64> {
        lock(&self.id_map).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::vec_to_bytes;

    #[test]
    fn empty_index_is_not_built() {
        let idx = HnswIndex::new();
        assert!(!idx.is_built());
        assert_eq!(idx.len(), 0);
        assert!(idx.search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn insert_then_search_returns_nearest() {
        let idx = HnswIndex::new();
        let id_a = idx.insert(101, &[1.0, 0.0, 0.0]);
        let id_b = idx.insert(202, &[0.0, 1.0, 0.0]);
        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert!(idx.is_built());
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.id_map_snapshot(), vec![101, 202]);

        let nearest = idx.search(&[1.0, 0.1, 0.0], 1);
        assert_eq!(nearest.len(), 1);
        let (external, _dist) = nearest[0];
        let node_id = idx.id_map_snapshot()[external];
        assert_eq!(node_id, 101);
    }

    #[test]
    fn rebuild_from_round_trips() {
        let idx = HnswIndex::new();
        let rows = vec![
            (7, vec_to_bytes(&[1.0, 0.0])),
            (8, vec_to_bytes(&[0.0, 1.0])),
            (9, vec_to_bytes(&[0.8, 0.2])),
        ];
        let count = idx
            .rebuild_from(rows.into_iter(), |b| Some(crate::vector::bytes_to_vec(b)))
            .unwrap();
        assert_eq!(count, 3);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.id_map_snapshot(), vec![7, 8, 9]);

        let nearest = idx.search(&[1.0, 0.05], 1);
        let external = nearest[0].0;
        assert_eq!(idx.id_map_snapshot()[external], 7);
    }

    #[test]
    fn rebuild_from_skips_bad_blobs() {
        let idx = HnswIndex::new();
        let rows = vec![
            (1, vec_to_bytes(&[1.0, 0.0])),
            (2, vec![0u8; 3]), // not a multiple of 4 -> decode None
            (3, Vec::new()),   // empty -> skipped
        ];
        let count = idx
            .rebuild_from(rows.into_iter(), crate::vector::try_bytes_to_vec)
            .unwrap();
        assert_eq!(count, 3, "total rows examined");
        assert_eq!(idx.len(), 1, "only the valid blob is indexed");
        assert_eq!(idx.id_map_snapshot(), vec![1]);
    }

    #[test]
    fn insert_after_rebuild_extends_id_map() {
        let idx = HnswIndex::new();
        idx.insert(1, &[1.0, 0.0]);
        idx.rebuild_from(std::iter::once((5, vec_to_bytes(&[0.0, 1.0]))), |b| {
            Some(crate::vector::bytes_to_vec(b))
        })
        .unwrap();
        assert_eq!(idx.id_map_snapshot(), vec![5]);
        idx.insert(9, &[1.0, 0.0]);
        assert_eq!(idx.id_map_snapshot(), vec![5, 9]);
        assert_eq!(idx.len(), 2);
    }
}
