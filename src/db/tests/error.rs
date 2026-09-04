use super::*;

#[test]
fn unique_violation_maps_to_duplicate_entry() {
    let conn = common_core::sqlite::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
        .unwrap();
    conn.execute(
        "INSERT INTO t (id, name) VALUES (?1, ?2)",
        rusqlite::params![1, "x"],
    )
    .unwrap();
    let err = conn
        .execute(
            "INSERT INTO t (id, name) VALUES (?1, ?2)",
            rusqlite::params![1, "y"],
        )
        .expect_err("duplicate primary key must fail");
    match DbError::from(err) {
        DbError::DuplicateEntry(_) => {}
        other => panic!("expected DuplicateEntry, got {other:?}"),
    }
}

#[test]
fn unique_name_violation_maps_to_duplicate_entry() {
    let conn = common_core::sqlite::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
        .unwrap();
    conn.execute(
        "INSERT INTO t (id, name) VALUES (?1, ?2)",
        rusqlite::params![1, "x"],
    )
    .unwrap();
    let err = conn
        .execute(
            "INSERT INTO t (id, name) VALUES (?1, ?2)",
            rusqlite::params![2, "x"],
        )
        .expect_err("duplicate unique name must fail");
    match DbError::from(err) {
        DbError::DuplicateEntry(_) => {}
        other => panic!("expected DuplicateEntry, got {other:?}"),
    }
}

#[test]
fn busy_maps_to_busy() {
    // SQLITE_BUSY == 5 — `ffi::Error::new` derives `ErrorCode::DatabaseBusy`
    // from the raw result code.
    let err = rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(5), None);
    match DbError::from(err) {
        DbError::Busy(_) => {}
        other => panic!("expected Busy, got {other:?}"),
    }
}

#[test]
fn other_error_maps_to_sqlite() {
    // SQLITE_ERROR == 1 — a generic failure, must map to `Sqlite`.
    let err = rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(1), None);
    match DbError::from(err) {
        DbError::Sqlite(_) => {}
        other => panic!("expected Sqlite, got {other:?}"),
    }
}

#[test]
fn io_error_maps_to_other() {
    let io_err = common_core::error::IoError(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "missing file",
    ));
    match DbError::from(io_err) {
        DbError::Other(msg) => assert!(msg.contains("missing file")),
        other => panic!("expected Other, got {other:?}"),
    }
}
