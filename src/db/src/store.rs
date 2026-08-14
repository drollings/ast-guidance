//! The canonical single-connection SQLite store.
//!
//! `SqliteStore` owns the connection lifecycle (open / WAL / schema init) and
//! exposes typed statement helpers (`query_row`/`query_rows`/`execute`/
//! `execute_batch`/`transaction`). It is the base shape that `GuidanceDb`,
//! `Library`, `ContentNodeLedger`, `HolographicStore`, and the charts store all
//! collapse to — each of those hand-rolled `Mutex<Connection>` + `open_wal`
//! before this component existed.
//!
//! Locks are taken via `common_core::sync::lock` (poison-safe). A statement that
//! returns no rows maps to `Ok(None)` for `query_row` and `Ok(vec![])` for
//! `query_rows` — the exact shape consumers already re-implement inline.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::Connection;

use common_core::sync::lock;

use crate::error::DbError;

/// A single-connection SQLite store with WAL mode enabled.
///
/// The connection is wrapped in a `Mutex` because `rusqlite::Connection` is not
/// `Sync`; every operation serializes on it. Consumers that need to hold the lock
/// across multiple statements use `with_conn`/`with_conn_mut` and the typed
/// helpers inside one closure.
#[derive(Debug)]
pub struct SqliteStore {
    pub(crate) conn: Mutex<Connection>,
}

impl SqliteStore {
    /// Open (or create) the database at `path` with WAL mode enabled.
    pub fn open(path: &Path) -> Result<Self, DbError> {
        let conn = common_core::sqlite::open_wal(path).map_err(DbError::from)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory database with WAL mode enabled.
    pub fn open_in_memory() -> Result<Self, DbError> {
        let conn = common_core::sqlite::open_in_memory().map_err(DbError::from)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Execute a multi-statement DDL batch (schema init) on the connection.
    pub fn init_schema(&self, ddl: &str) -> Result<(), DbError> {
        common_core::sqlite::run_batch(&lock(&self.conn), ddl).map_err(DbError::from)
    }

    /// Run a closure against the shared connection.
    ///
    /// The connection lock is held for the duration of the closure, so callers
    /// can run multiple statements atomically (from this thread's perspective).
    pub fn with_conn<R>(
        &self,
        f: impl FnOnce(&Connection) -> Result<R, DbError>,
    ) -> Result<R, DbError> {
        f(&lock(&self.conn))
    }

    /// Run a closure against the shared connection with `&mut` access.
    ///
    /// Needed for `Connection::transaction` and `Connection::prepare`-with-write
    /// shapes.
    pub fn with_conn_mut<R>(
        &self,
        f: impl FnOnce(&mut Connection) -> Result<R, DbError>,
    ) -> Result<R, DbError> {
        f(&mut lock(&self.conn))
    }

    /// Execute a DML statement and return the number of rows affected.
    pub fn execute(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Result<usize, DbError> {
        crate::query::execute(&lock(&self.conn), sql, params)
    }

    /// Execute a multi-statement SQL batch.
    pub fn execute_batch(&self, sql: &str) -> Result<(), DbError> {
        crate::query::execute_batch(&lock(&self.conn), sql)
    }

    /// Query at most one row.
    ///
    /// A statement that matches no rows returns `Ok(None)` (not an error).
    pub fn query_row<T>(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        map: impl FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<Option<T>, DbError> {
        crate::query::query_row(&lock(&self.conn), sql, params, map)
    }

    /// Query all rows.
    ///
    /// A statement that matches no rows returns `Ok(vec![])`.
    pub fn query_rows<T>(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        map: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<Vec<T>, DbError> {
        crate::query::query_rows(&lock(&self.conn), sql, params, map)
    }

    /// Run a closure inside a transaction.
    ///
    /// Commits on `Ok`, rolls back on `Err` (the transaction is dropped while
    /// uncommitted, which rolls it back).
    pub fn transaction<T>(
        &self,
        f: impl FnOnce(&mut rusqlite::Transaction<'_>) -> Result<T, DbError>,
    ) -> Result<T, DbError> {
        self.with_conn_mut(|conn| crate::query::transaction(conn, f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_and_schema_init() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteStore::open(&dir.path().join("test.db")).unwrap();
        store
            .init_schema("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        assert!(dir.path().join("test.db").exists());
    }

    #[test]
    fn open_in_memory_works() {
        let store = SqliteStore::open_in_memory().unwrap();
        store
            .init_schema("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .unwrap();
    }

    #[test]
    fn execute_and_query_round_trip() {
        let store = SqliteStore::open_in_memory().unwrap();
        store
            .init_schema("CREATE TABLE t (id INTEGER, name TEXT)")
            .unwrap();
        let n = store
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![1, "hello"],
            )
            .unwrap();
        assert_eq!(n, 1);

        let name = store
            .query_row(
                "SELECT name FROM t WHERE id = ?1",
                rusqlite::params![1],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        assert_eq!(name.as_deref(), Some("hello"));

        let rows = store
            .query_rows("SELECT id, name FROM t ORDER BY id", &[], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap();
        assert_eq!(rows, vec![(1, "hello".to_string())]);
    }

    #[test]
    fn query_row_no_rows_maps_to_none() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
        let val = store
            .query_row("SELECT id FROM t WHERE id = 999", &[], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(val, None);
    }

    #[test]
    fn query_rows_empty_maps_to_empty_vec() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
        let rows = store
            .query_rows("SELECT id FROM t", &[], |row| row.get::<_, i64>(0))
            .unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn transaction_commits_on_ok() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
        store
            .transaction(|tx| {
                tx.execute("INSERT INTO t (id) VALUES (?1)", rusqlite::params![7])?;
                Ok(())
            })
            .unwrap();
        let count = store
            .query_row("SELECT COUNT(*) FROM t", &[], |row| row.get::<_, i64>(0))
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn transaction_rolls_back_on_err() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
        let result: Result<(), DbError> = store.transaction(|tx| {
            tx.execute("INSERT INTO t (id) VALUES (?1)", rusqlite::params![7])?;
            Err(DbError::Other("boom".into()))
        });
        assert!(result.is_err());
        let count = store
            .query_row("SELECT COUNT(*) FROM t", &[], |row| row.get::<_, i64>(0))
            .unwrap()
            .unwrap();
        assert_eq!(count, 0, "transaction must roll back on error");
    }

    #[test]
    fn poison_recovery_via_lock() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store
                .with_conn(|_| Err::<(), DbError>(DbError::Other("boom".into())))
                .unwrap_err();
            // Simulate a panic while holding the lock.
            let _guard = lock(&store.conn);
            panic!("boom");
        }));
        assert!(panic.is_err());
        // The store is still usable after poison recovery.
        store.execute("INSERT INTO t (id) VALUES (1)", &[]).unwrap();
    }
}
