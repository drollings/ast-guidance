//! Embedding-cache DDL — the single owner (ROADMAP_20260903_LLM M5).
//!
//! Moved verbatim from `common_core::sqlite` (`EMBEDDING_CACHE_SCHEMA` +
//! `init_embedding_cache`). This schema was already deduped once (from
//! `coral` + `search-vector` into `common-core`); this finishes the move to
//! the domain owner, so the DDL string has exactly one source of truth.
//! Generic helpers (`open_wal`, `open_in_memory`, `open_shared_in_memory`,
//! `run_batch`, `make_hnsw`, `is_unique_violation`, `in_clause`) stay in
//! `common_core::sqlite`.
//!
//! M11 deleted the `common-core::sqlite` byte-identical shim copies (kept
//! through M10 under `#[deprecated]`); the owner goldens in
//! `tests/embeddings_cache.rs` are the lasting contract.
//!
//! Calibration (roadmap §1, M10): `UNIQUE(query_hash)` is identity, not
//! quality — cached-embedding reuse is key-equality, never similarity.
//! The DDL moves unchanged; existing databases are unaffected.

use rusqlite::{Connection, Result};

/// Canonical `embedding_cache` table DDL.
///
/// This schema was previously duplicated verbatim in `search-vector` and
/// `coral`. It lives here as the single source of truth.
pub const EMBEDDING_CACHE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS embedding_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
)";

/// Create the `embedding_cache` table if it doesn't already exist.
pub fn init_embedding_cache(conn: &Connection) -> Result<()> {
    conn.execute_batch(EMBEDDING_CACHE_SCHEMA)
}
