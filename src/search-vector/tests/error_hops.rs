//! Error-hop tests (M7): every `rusqlite::Error` reaches the domain
//! `VectorDbError` only through `DbError::from` — the single
//! `From<rusqlite::Error>` impl in `fluent_db::error` — with `VectorDbError`
//! kept a thin wrapper (`Db` hop preserves the classified variant).

use super::*;
use fluent_db::error::DbError;

#[test]
fn rusqlite_unique_violation_reaches_db_variant_classified() {
    // Full chain on a real UNIQUE violation: rusqlite → `DbError::from`
    // (classifies `DuplicateEntry`) → `VectorDbError::Db` (thin wrapper).
    let conn = common_core::sqlite::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
        .unwrap();
    conn.execute("INSERT INTO t (id, name) VALUES (?1, ?2)", rusqlite::params![1, "x"])
        .unwrap();
    let err = conn
        .execute("INSERT INTO t (id, name) VALUES (?1, ?2)", rusqlite::params![1, "y"])
        .expect_err("duplicate primary key must fail");
    match VectorDbError::from(DbError::from(err)) {
        VectorDbError::Db(DbError::DuplicateEntry(_)) => {}
        other => panic!("expected Db(DuplicateEntry), got {other:?}"),
    }
}

#[test]
fn busy_hops_through_thin_wrapper() {
    // `Busy` is preserved verbatim through the `Db` hop — the wrapper adds
    // no lossy re-mapping.
    match VectorDbError::from(DbError::Busy("locked".into())) {
        VectorDbError::Db(DbError::Busy(_)) => {}
        other => panic!("expected Db(Busy), got {other:?}"),
    }
}
