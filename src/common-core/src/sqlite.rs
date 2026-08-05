//! Shared SQLite helpers — connection setup, WAL mode, schema init, and the
//! canonical `embedding_cache` table definition.
//!
//! All functions are gated on the `sqlite` Cargo feature so the crate stays
//! zero-domain by default.

use std::path::Path;

use rusqlite::{Connection, Result};

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

/// Execute a multi-statement SQL batch (e.g. DDL) on `conn`.
pub fn run_batch(conn: &Connection, schema: &str) -> Result<()> {
    conn.execute_batch(schema)
}

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

/// Create the `embedding_cache` table if it doesn't already exist.
pub fn init_embedding_cache(conn: &Connection) -> Result<()> {
    conn.execute_batch(EMBEDDING_CACHE_SCHEMA)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_wal_sets_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let conn = open_wal(&path).unwrap();
        let journal: String = conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(journal, "wal");
    }

    #[test]
    fn open_in_memory_works() {
        let conn = open_in_memory().unwrap();
        // In-memory databases return "memory" for journal_mode — that's expected.
        // The important thing is that the connection works.
        conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
    }

    #[test]
    fn run_batch_executes_ddl() {
        let conn = open_in_memory().unwrap();
        run_batch(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY)").unwrap();
        conn.execute_batch("INSERT INTO t (id) VALUES (1)").unwrap();
    }

    #[test]
    fn init_embedding_cache_creates_table() {
        let conn = open_in_memory().unwrap();
        init_embedding_cache(&conn).unwrap();
        // Insert a row to prove the table exists and has the right columns.
        conn.execute(
            "INSERT INTO embedding_cache (query_hash, query_text, embedding) VALUES (?1, ?2, ?3)",
            rusqlite::params!["abc123", "hello world", vec![0u8; 32]],
        )
        .unwrap();
    }

    #[test]
    fn embedding_cache_schema_is_idempotent() {
        let conn = open_in_memory().unwrap();
        init_embedding_cache(&conn).unwrap();
        // Second call must not fail.
        init_embedding_cache(&conn).unwrap();
    }

    #[test]
    fn is_unique_violation_detects_unique_and_primary_key() {
        let conn = open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            .unwrap();

        let err = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![1, "x"],
            )
            .and_then(|_| {
                conn.execute(
                    "INSERT INTO t (id, name) VALUES (?1, ?2)",
                    rusqlite::params![1, "y"],
                )
            })
            .expect_err("duplicate primary key must fail");
        assert!(is_unique_violation(&err), "PK: {err}");

        let err = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![2, "x"],
            )
            .expect_err("duplicate unique name must fail");
        assert!(is_unique_violation(&err), "UNIQUE: {err}");
    }

    #[test]
    fn is_unique_violation_rejects_other_errors() {
        let conn = open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            .unwrap();
        // NOT NULL violation is not a unique/primary-key violation.
        let not_null = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, NULL)",
                rusqlite::params![3],
            )
            .expect_err("NOT NULL violation");
        assert!(!is_unique_violation(&not_null), "NOT NULL: {not_null}");
    }

    #[test]
    fn in_clause_builds_placeholders() {
        assert_eq!(in_clause(1), "?");
        assert_eq!(in_clause(3), "?,?,?");
        assert_eq!(in_clause(0), "");
    }
}
