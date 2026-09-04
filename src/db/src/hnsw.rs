//! The canonical HNSW-backed vector index store.
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
use common_core::constants::{HnswParams, DEFAULT_HNSW_THRESHOLD};
use common_core::sync::{lock, lock_read, lock_write};
use hnsw_rs::hnsw::Hnsw;

use crate::error::DbError;

/// Single adaptive-dispatch policy for HNSW-vs-brute-force routing (M6).
///
/// This is a policy, not an index: it owns the threshold and answers whether
/// a store of `len` vectors should consult a built HNSW graph. The one
/// threshold source is [`DEFAULT_HNSW_THRESHOLD`]; there is no second
/// threshold type — call sites reuse `HnswParams`/`DEFAULT_HNSW_THRESHOLD`
/// through this struct.
///
/// Axis: **[B] cost/recall only.** This gate measures corpus scale (where the
/// HNSW approximation pays off — recall≥0.95 at N≥512 per M5a). It is NOT a
/// confidence signal and must never gate verification, persistence, or
/// frontier escalation (those are [A]/[B]-outcome decisions owned by the
/// workflow/rubric gates calibrated in M5c–M5d).
///
/// [`DEFAULT_HNSW_THRESHOLD`]: common_core::constants::DEFAULT_HNSW_THRESHOLD
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveHnsw {
    /// Corpus size above which a built HNSW graph is consulted. Strict `>`:
    /// `len == threshold` stays brute-force.
    pub threshold: usize,
}

impl AdaptiveHnsw {
    /// Policy with an explicit threshold (tests, unusual corpora).
    pub fn new(threshold: usize) -> Self {
        Self { threshold }
    }

    /// Whether a store holding `len` vectors should use its built HNSW graph
    /// (`len > threshold`).
    pub fn should_use_built(&self, len: usize) -> bool {
        len > self.threshold
    }

    /// Query-time dispatch: probe HNSW iff the index is built *and* the corpus
    /// is above threshold. `false` means "take the call-site brute-force
    /// fallback" — the fallback bodies stay call-site code (M2), this only
    /// answers which path to take.
    pub fn dispatch(&self, hnsw_built: bool, len: usize) -> bool {
        hnsw_built && self.should_use_built(len)
    }
}

impl Default for AdaptiveHnsw {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_HNSW_THRESHOLD,
        }
    }
}

/// A single HNSW index handle — the named index path owned by a store (e.g.
/// the router's `ChartStore` workflow_library index). Moved here from
/// `fluent_router::hnsw` (D7, router shim deleted in M3): the `db` crate is
/// the canonical home for the HNSW surface (`HnswIndex`, `hnsw_lookup`), and
/// this handle is plain data (no router dependency) so it belongs with that
/// surface. Consume it as `fluent_db::hnsw::HnswIndexHandle` directly.
#[derive(Debug, Clone)]
pub struct HnswIndexHandle {
    pub name: String,
    pub path: String,
}

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
        let mut guard = lock_write(&self.hnsw);
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

        *lock_write(&self.hnsw) = Some(hnsw);
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
        let guard = lock_read(&self.hnsw);
        let Some(hnsw) = guard.as_ref() else {
            return Vec::new();
        };
        // Hold the id-map guard for the duration of the index traversal so a
        // concurrent rebuild cannot swap the index out from under the id
        // resolution. Lock order: hnsw (read) → id_map.
        let _id_map = lock(&self.id_map);

        let ef = (k * 4).max(64);
        hnsw.search(query, k, ef)
            .into_iter()
            .map(|n| (n.d_id, n.distance))
            .collect()
    }

    /// Whether the index has been built (any `insert`/`rebuild_from` ran).
    pub fn is_built(&self) -> bool {
        lock_read(&self.hnsw).is_some()
    }

    /// Number of points currently in the index.
    pub fn len(&self) -> usize {
        lock_read(&self.hnsw).as_ref().map_or(0, Hnsw::get_nb_point)
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

/// Probe a possibly-unbuilt [`HnswIndex`], resolving external ids to the
/// caller keys the index was built with.
///
/// M5: the one shared HNSW-then-brute-force stanza. Returns `Some((key,
/// distance))` pairs in HNSW order (cosine distance, NOT similarity — each
/// site applies its own `1 - dist` mapping), or `None` when the caller must
/// use its exact brute-force fallback: the index is unbuilt, the probe is
/// empty (`k == 0`, malformed query), or no hit resolves through the id map.
///
/// Deliberately lookup-only: the fallback bodies stay call-site code because
/// sort/truncate/re-score/filter semantics differ per store (unifying them
/// would be a behavior change). Compose as
/// `hnsw_lookup(...).map_or_else(fallback, scored)` — see
/// `node_store::knn_search`, `charts::store::search`,
/// `ledger::workflow_store::nearest` for the three production shapes.
pub fn hnsw_lookup(hnsw: &HnswIndex, query: &[f32], k: usize) -> Option<Vec<(i64, f32)>> {
    if !hnsw.is_built() {
        return None;
    }
    if query.is_empty() || k == 0 {
        // Malformed probe: an empty query panics inside the distance kernel
        // (dim assertion), so short-circuit to the brute-force fallback
        // signal instead of touching the index.
        return None;
    }
    let hits = hnsw.search(query, k);
    if hits.is_empty() {
        return None;
    }
    let id_map = hnsw.id_map_snapshot();
    let resolved: Vec<(i64, f32)> = hits
        .into_iter()
        .filter_map(|(d_id, distance)| id_map.get(d_id).map(|key| (*key, distance)))
        .collect();
    if resolved.is_empty() { None } else { Some(resolved) }
}

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/hnsw.rs"]
mod tests;
