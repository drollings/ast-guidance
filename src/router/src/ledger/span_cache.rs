//! Span-level detail cache as a **read-through view over the shared ledger
//! sqlite** (ROADMAP_20260831_ARCEAGER M6.1).
//!
//! No parallel store: the same `interlingua_index` table that backs
//! `SqliteCorrectionIndex` backs the span cache, with `role = 'span_cache'`
//! and the content-addressed `span_key` stored as fixed-width hex TEXT
//! (`{:016x}`) in the `span_key` column (the `i64` cast silently corrupts
//! the upper half of the `u64` space — F7). The same connection/transaction
//! is used, so a ledger snapshot contains both correction patterns and span
//! details.
//!
//! # Tenancy (M2b)
//!
//! This module is the ledger view owner for the whole span-cache surface:
//! the [`SpanCache`] trait, the [`InMemorySpanCache`] hermetic double, and
//! the [`SqliteSpanCache`] production view (moved here from `spacy-rs`,
//! which keeps only the pure [`span_key`](spacy_rs::cache::span_key)
//! discipline and the dependency-free
//! [`SpanCacheSeam`](spacy_rs::cache::SpanCacheSeam) callback bundle the
//! ladder consults). [`span_cache_seam`] adapts an `Arc<dyn SpanCache>`
//! into that bundle at the wiring site, so no router edge ever enters
//! spacy-rs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fluent_db::store::SqliteStore;
use rusqlite::params;
use spacy_rs::review::Correction;

/// Content-addressed span cache for refiner corrections.
///
/// `Send + Sync` so the async ladder (shared `Arc` across `ResultPool`
/// workers) and the sync ladder (single-threaded) can share one instance.
pub trait SpanCache: Send + Sync {
    /// Return the cached corrections for `key`, if present.
    fn get(&self, key: u64) -> Option<Vec<Correction>>;
    /// Store `corrections` under `key` (overwrite).
    fn put(&self, key: u64, corrections: Vec<Correction>);
    /// Invalidate the entry for `key` (called when a `CorrectionIndex` write
    /// for the same span lands).
    fn invalidate(&self, key: u64);
    /// Number of entries (for tests/metrics).
    fn len(&self) -> usize;
    /// Whether the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hermetic in-memory span cache (tests + non-ledger callers).
///
/// `Mutex<HashMap>` — the cache is not a hot loop; a single lock per
/// get/put is negligible compared to a model call.
#[derive(Debug, Default)]
pub struct InMemorySpanCache {
    map: Mutex<HashMap<u64, Vec<Correction>>>,
}

impl InMemorySpanCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }
}

impl SpanCache for InMemorySpanCache {
    fn get(&self, key: u64) -> Option<Vec<Correction>> {
        self.map.lock().expect("cache lock").get(&key).cloned()
    }

    fn put(&self, key: u64, corrections: Vec<Correction>) {
        if key == 0 {
            return;
        }
        self.map
            .lock()
            .expect("cache lock")
            .insert(key, corrections);
    }

    fn invalidate(&self, key: u64) {
        self.map.lock().expect("cache lock").remove(&key);
    }

    fn len(&self) -> usize {
        self.map.lock().expect("cache lock").len()
    }
}

/// Sentinel node id for span-cache rows (orthogonal to correction rows).
pub const SPAN_CACHE_NODE: i64 = 0;
const SPAN_CACHE_ROLE: &str = "span_cache";

/// `SpanCache` over the shared ledger sqlite (same file as
/// `interlingua_index`, no parallel store).
pub struct SqliteSpanCache {
    store: Arc<SqliteStore>,
}

impl SqliteSpanCache {
    /// A span cache over the shared connection.
    pub fn new(store: Arc<SqliteStore>) -> Self {
        Self { store }
    }

    /// Invalidate through the correction-index contract: call this when a
    /// `CorrectionIndex::record_correction` for the same span lands, so a
    /// stale cached correction never replays.
    pub fn invalidate_for_corrections(&self, span_key: u64) {
        self.invalidate(span_key);
    }
}

fn hex_key(key: u64) -> String {
    format!("{key:016x}")
}

impl SpanCache for SqliteSpanCache {
    fn get(&self, key: u64) -> Option<Vec<Correction>> {
        if key == 0 {
            return None;
        }
        let hk = hex_key(key);
        let row = self
            .store
            .query_row(
                "SELECT corrections FROM interlingua_index \
                 WHERE node_id = ?1 AND span_key = ?2 AND role = ?3 AND entity_id = 0",
                params![SPAN_CACHE_NODE, hk, SPAN_CACHE_ROLE],
                |r| r.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
            .flatten()?;
        serde_json::from_str(&row).ok()
    }

    fn put(&self, key: u64, corrections: Vec<Correction>) {
        if key == 0 {
            return;
        }
        let Ok(json) = serde_json::to_string(&corrections) else {
            return;
        };
        let hk = hex_key(key);
        let _ = self.store.with_conn(|conn| {
            fluent_db::query::execute(
                conn,
                "INSERT INTO interlingua_index \
                 (node_id, interlingua_id, interlingua_source, role, entity_id, review_status, corrections, span_key) \
                 VALUES (?1, 0, 'spacy_lemma', ?2, 0, 'cached', ?3, ?4) \
                 ON CONFLICT(node_id, interlingua_id, span_key, role, entity_id) DO UPDATE SET \
                    review_status = excluded.review_status, corrections = excluded.corrections",
                params![SPAN_CACHE_NODE, SPAN_CACHE_ROLE, json, hk],
            )
        });
    }

    fn invalidate(&self, key: u64) {
        if key == 0 {
            return;
        }
        let hk = hex_key(key);
        let _ = self.store.with_conn(|conn| {
            fluent_db::query::execute(
                conn,
                "DELETE FROM interlingua_index WHERE node_id = ?1 AND span_key = ?2 AND role = ?3",
                params![SPAN_CACHE_NODE, hk, SPAN_CACHE_ROLE],
            )
        });
    }

    fn len(&self) -> usize {
        self.store
            .query_row(
                "SELECT COUNT(*) FROM interlingua_index WHERE node_id = ?1 AND role = ?2",
                params![SPAN_CACHE_NODE, SPAN_CACHE_ROLE],
                |r| r.get(0),
            )
            .ok()
            .flatten()
            .unwrap_or(0)
    }
}

/// Adapt a ledger-owned `Arc<dyn SpanCache>` into the dependency-free
/// [`SpanCacheSeam`](spacy_rs::cache::SpanCacheSeam) the spacy-rs ladder
/// consults (M2b). Each leg captures an `Arc` clone, so the seam shares the
/// backend instead of copying it.
#[must_use]
pub fn span_cache_seam(cache: &Arc<dyn SpanCache>) -> spacy_rs::cache::SpanCacheSeam {
    let get_h = Arc::clone(cache);
    let put_h = Arc::clone(cache);
    let inv_h = Arc::clone(cache);
    spacy_rs::cache::SpanCacheSeam::new(
        Arc::new(move |key| get_h.get(key)),
        Arc::new(move |key, corrections| put_h.put(key, corrections)),
        Arc::new(move |key| inv_h.invalidate(key)),
    )
}
#[cfg(test)]
#[path = "../../tests/ledger_span_cache.rs"]
mod tests;
