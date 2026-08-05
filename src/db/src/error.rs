//! The single database error taxonomy for the workspace.
//!
//! Consumers' error enums wrap `DbError` via a single `#[from]` variant instead
//! of hand-writing `From<rusqlite::Error>`. The `From<rusqlite::Error>` impl here
//! centralizes the `is_unique_violation` → `DuplicateEntry` mapping and the
//! `SQLITE_BUSY` → `Busy` mapping (generalizing coral's `db/mod.rs:36-45` and the
//! memory-plugin inline check).

/// Canonical database error.
///
/// Variants mirror the failure classes that the workspace's hand-rolled error
/// enums each re-derived separately: raw sqlite failures, missing rows/keys,
/// duplicate-key constraints, busy/timeout conditions, pool exhaustion, and
/// schema-version mismatches.
#[cfg(feature = "sqlite")]
#[derive(thiserror::Error, Debug)]
pub enum DbError {
    /// A raw `rusqlite` failure (wrapped in the shared `SqliteError` leaf).
    #[error("sqlite error: {0}")]
    Sqlite(#[from] common_core::error::SqliteError),
    /// A row or key that was expected to exist did not.
    #[error("not found: {0}")]
    NotFound(String),
    /// A `UNIQUE`/`PRIMARY KEY` constraint violation.
    #[error("duplicate entry: {0}")]
    DuplicateEntry(String),
    /// The database reported `SQLITE_BUSY`.
    #[error("database busy: {0}")]
    Busy(String),
    /// The connection pool had no free connections.
    #[error("connection pool exhausted")]
    PoolExhausted,
    /// The database schema version was not what a migration expected.
    #[error("invalid schema version: {0}")]
    InvalidSchemaVersion(u32),
    /// Any other database-layer error.
    #[error("database error: {0}")]
    Other(String),
}

#[cfg(feature = "sqlite")]
impl From<rusqlite::Error> for DbError {
    fn from(e: rusqlite::Error) -> Self {
        // A UNIQUE/PRIMARY KEY constraint violation is a typed duplicate-key
        // error, not a raw sqlite failure — consumers surface it as such.
        if common_core::sqlite::is_unique_violation(&e) {
            return DbError::DuplicateEntry(e.to_string());
        }
        // `SQLITE_BUSY` means a lock could not be acquired; classify it so
        // callers can back off and retry rather than treating it as a generic
        // failure.
        if let rusqlite::Error::SqliteFailure(ffi, _) = &e {
            if ffi.code == rusqlite::ErrorCode::DatabaseBusy {
                return DbError::Busy(e.to_string());
            }
        }
        DbError::Sqlite(common_core::error::SqliteError(e))
    }
}

#[cfg(feature = "sqlite")]
impl From<common_core::error::IoError> for DbError {
    fn from(e: common_core::error::IoError) -> Self {
        DbError::Other(e.to_string())
    }
}

#[cfg(feature = "sqlite")]
#[cfg(test)]
mod tests {
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
}
