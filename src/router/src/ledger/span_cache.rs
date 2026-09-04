//! Span-level detail cache as a **read-through view over the shared ledger
//! sqlite** (ROADMAP_20260831_ARCEAGER M6.1).
//!
//! No parallel store: the same `interlingua_index` table that backs
//! `SqliteCorrectionIndex` backs the span cache, with `role = 'span_cache'`
//! and the content-addressed `span_key` stored in the `interlingua_id`
//! column (the 64-bit hash). The same connection/transaction is used, so a
//! ledger snapshot contains both correction patterns and span details.

use std::sync::Arc;

use fluent_db::store::SqliteStore;
use rusqlite::params;
use spacy_rs::cache::SpanCache;
use spacy_rs::review::Correction;

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
#[cfg(test)]
#[path = "../../tests/ledger_span_cache.rs"]
mod tests;
