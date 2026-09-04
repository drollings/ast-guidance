//! Shared SQLite helpers — connection setup, WAL mode, schema init, and the
//! canonical `embedding_cache` table definition.
//!
//! All functions are gated on the `sqlite` Cargo feature so the crate stays
//! zero-domain by default.

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, Result};

use crate::constants::HnswParams;
use anndists::dist::DistCosine;
use hnsw_rs::hnsw::Hnsw;

/// Open a connection to `path` with WAL journal mode and a busy timeout.
///
/// `PRAGMA journal_mode=WAL` allows concurrent readers while one writer holds
/// the lock. `PRAGMA busy_timeout=5000` prevents `SQLITE_BUSY` from racing
/// the Tokio blocking pool.
pub fn open_wal(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
    Ok(conn)
}

/// Open an in-memory connection with WAL mode enabled.
///
/// WAL mode on an in-memory database is a no-op, but we keep it so callers
/// don't need to branch on whether the connection is file-backed.
pub fn open_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
    Ok(conn)
}

/// Open a connection to a **named shared-cache in-memory** database.
///
/// Every connection opened with the same `name` participates in one shared
/// in-memory database, so a pool of connections to `memdb{N}` sees the same
/// tables and data. This is the fix for the pool-isolation bug where each
/// checkout silently saw a different private empty DB: `open_in_memory()`
/// (and SQLite's bare `:memory:`) create a *per-connection* database, which is
/// incoherent under concurrent pooling.
///
/// The database lives exactly as long as at least one connection to it stays
/// open and is destroyed when the last connection closes — callers must keep a
/// connection (the pool's idle set does) alive for the DB's lifetime.
///
/// `PRAGMA busy_timeout=5000` prevents `SQLITE_BUSY` from racing the Tokio
/// blocking pool. `journal_mode=WAL` is deliberately **not** set: WAL is
/// unsupported on shared-cache memory databases and errors; the default
/// (memory) journal is correct here.
pub fn open_shared_in_memory(name: &str) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        format!("file:{name}?mode=memory&cache=shared"),
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.busy_timeout(Duration::from_millis(5000))?;
    Ok(conn)
}

/// Execute a multi-statement SQL batch (e.g. DDL) on `conn`.
pub fn run_batch(conn: &Connection, schema: &str) -> Result<()> {
    conn.execute_batch(schema)
}

/// NOTE (ROADMAP_20260903_LLM M11): the `embedding_cache` table DDL
/// (`EMBEDDING_CACHE_SCHEMA` + `init_embedding_cache`) lived here through
/// M10 as deprecated byte-identical shims of
/// `fluent_llm::embeddings_cache`; M11 deleted them.
///
/// Build a cosine-distance HNSW index from `HnswParams` with a pluggable
/// `initial_capacity`.
///
/// Centralizes the positional-argument unpacking so the
/// (max_nb_connection, initial_capacity, max_layer, ef_construction,
/// DistCosine) ordering is decided in exactly one place across the
/// workspace.  Previously duplicated in coral and search-vector.
pub fn make_hnsw(p: &HnswParams, initial_capacity: usize) -> Hnsw<'static, f32, DistCosine> {
    Hnsw::<f32, DistCosine>::new(
        p.max_nb_connection,
        initial_capacity,
        p.max_layer,
        p.ef_construction,
        DistCosine,
    )
}

/// True when `e` is a SQLite `UNIQUE` or `PRIMARY KEY` constraint violation.
///
/// Generalizes coral's `db/mod.rs` classifier so any consumer can surface a
/// duplicate-key error as a typed variant instead of a raw `SqliteError`.
/// Extended codes: 2067 = `SQLITE_CONSTRAINT_UNIQUE`, 1555 =
/// `SQLITE_CONSTRAINT_PRIMARYKEY`.
pub fn is_unique_violation(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(ffi, _)
            if ffi.code == rusqlite::ErrorCode::ConstraintViolation
                && (ffi.extended_code == 2067 || ffi.extended_code == 1555)
    )
}

/// Build the `?,?,?,…` placeholder list for a parameterized `WHERE x IN (...)`.
///
/// Returns an empty string for `n == 0` (an `IN ()` clause matches nothing and
/// callers typically guard against it, but the empty string is a valid no-op).
pub fn in_clause(n: usize) -> String {
    "?,".repeat(n).trim_end_matches(',').to_string()
}

