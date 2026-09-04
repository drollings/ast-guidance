//! Error-hop tests (M7): every `rusqlite::Error` reaches the domain
//! `LibraryError` only through `DbError::from` — the single
//! `From<rusqlite::Error>` impl in `fluent_db::error` — with the
//! `DuplicateEntry → DuplicateNode` hop preserved end to end.

use super::*;
use crate::tests::common::make_node;
use fluent_db::error::DbError;
use fluent_types::ContentNode;

#[test]
fn unique_violation_hops_to_duplicate_node_end_to_end() {
    // Full chain: sqlite UNIQUE violation → `DbError::DuplicateEntry` →
    // `LibraryError::DuplicateNode` through a real duplicate-name insert
    // (the legacy autoincrement path issues a plain `INSERT`).
    let lib = crate::db::Library::open_in_memory().expect("in-memory db");
    lib.insert_node(&ContentNode { ..make_node("dup", "s") })
        .expect("first insert");
    let err = lib
        .insert_node(&ContentNode { ..make_node("dup", "s") })
        .expect_err("duplicate name must fail");
    match err {
        LibraryError::DuplicateNode(_) => {}
        other => panic!("expected DuplicateNode, got {other:?}"),
    }
}

#[test]
fn duplicate_entry_maps_to_duplicate_node_while_busy_stays_db() {
    // The domain hop: only `DuplicateEntry` becomes `DuplicateNode`; every
    // other `DbError` (here `Busy`) stays wrapped as `Db`.
    match LibraryError::from(DbError::DuplicateEntry("dup".into())) {
        LibraryError::DuplicateNode(_) => {}
        other => panic!("expected DuplicateNode, got {other:?}"),
    }
    match LibraryError::from(DbError::Busy("locked".into())) {
        LibraryError::Db(DbError::Busy(_)) => {}
        other => panic!("expected Db(Busy), got {other:?}"),
    }
}
