//! ROADMAP_20260903_LLM M5.3 — embedding-cache DDL goldens (moved, not copied).
//!
//! Canonical home for the `embedding_cache` DDL goldens: the
//! creates-table + idempotent-re-init assertions moved from
//! `src/common-core/tests/sqlite.rs`, plus the `UNIQUE(query_hash)`
//! violation control and the non-embedding-tables-untouched control.
//! The DDL string is byte-identical to the removed `common_core::sqlite`
//! shims (M11 deleted them with the dual-path tests).
//!
//! Calibration (roadmap §1, M10): `UNIQUE(query_hash)` is identity, not
//! quality — cached-embedding reuse is key-equality, never similarity.
//! Generic `open_*` / `run_batch` / `is_unique_violation` / `in_clause`
//! helpers stay in `common_core::sqlite` and are composed here, not
//! re-implemented.

use fluent_llm::embeddings_cache::*;

// ── Moved from common-core/tests/sqlite.rs ───────────────────────────────

#[test]
fn init_embedding_cache_creates_table() {
    let conn = common_core::sqlite::open_in_memory().unwrap();
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
    let conn = common_core::sqlite::open_in_memory().unwrap();
    init_embedding_cache(&conn).unwrap();
    // Second call must not fail.
    init_embedding_cache(&conn).unwrap();
}

// ── Controls ─────────────────────────────────────────────────────────────

#[test]
fn control_unique_query_hash_violation_surfaces() {
    // `UNIQUE(query_hash)` is the identity contract: a second row with the
    // same hash fails, and the generic `is_unique_violation` classifier
    // (which stays in `common-core`) recognizes it.
    let conn = common_core::sqlite::open_in_memory().unwrap();
    init_embedding_cache(&conn).unwrap();
    conn.execute(
        "INSERT INTO embedding_cache (query_hash, query_text, embedding) VALUES (?1, ?2, ?3)",
        rusqlite::params!["dup", "first", vec![1u8; 8]],
    )
    .unwrap();
    let err = conn
        .execute(
            "INSERT INTO embedding_cache (query_hash, query_text, embedding) VALUES (?1, ?2, ?3)",
            rusqlite::params!["dup", "second", vec![2u8; 8]],
        )
        .expect_err("duplicate query_hash must fail");
    assert!(
        common_core::sqlite::is_unique_violation(&err),
        "UNIQUE(query_hash): {err}"
    );
}

#[test]
fn control_non_embedding_tables_untouched() {
    // The DDL creates exactly one table and no indexes/triggers of its own;
    // a pre-existing table survives init untouched with its rows intact.
    let conn = common_core::sqlite::open_in_memory().unwrap();
    common_core::sqlite::run_batch(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY)")
        .unwrap();
    conn.execute("INSERT INTO t (id) VALUES (1)", [])
        .unwrap();
    init_embedding_cache(&conn).unwrap();
    let tables: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(tables, vec!["embedding_cache", "sqlite_sequence", "t"]);
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
}

// NOTE (ROADMAP_20260903_LLM M11): the dual-path `schema_matches_old` /
// `parity_new_eq_old` tests died with the `common_core::sqlite` shims they
// pinned. The owner goldens above (DDL behavior, idempotence, UNIQUE,
// non-embedding-tables-untouched) are the lasting contract.
