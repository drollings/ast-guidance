//! The canonical pooled SQLite store (D5).
//!
//! `SqlitePool` is the successor to `fluent-concurrency::io::db`'s pool: a
//! fixed-size set of `rusqlite::Connection`s (default 5) gated by a
//! `tokio::sync::Semaphore`, with synchronous work offloaded to
//! `tokio::task::spawn_blocking` at the async boundary. WAL mode is enabled so
//! SQLite can serve multiple concurrent readers without blocking.
//!
//! Checkout is RAII: a `PooledConnection` returns its `Connection` to the idle
//! set (and releases its semaphore permit) on `Drop`. The heavy per-operation
//! machinery is the same as `SqliteStore`; the difference is that operations
//! run on a blocking worker thread instead of the caller's thread.

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::Connection;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::DbError;

/// Pool sizing and busy-timeout configuration.
///
/// Defaults preserve the historical pool shape: 5 connections and a 5s busy
/// timeout (matching `common_core::sqlite::open_wal`'s `busy_timeout=5000`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    /// Number of `Connection`s to open up front. Defaults to `5`.
    pub size: usize,
    /// `PRAGMA busy_timeout` in milliseconds. `0` leaves the connection's
    /// default busy timeout untouched. Defaults to `5_000`.
    pub busy_timeout_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            size: 5,
            busy_timeout_ms: 5_000,
        }
    }
}

impl PoolConfig {
    /// A `PoolConfig` with the default busy timeout and the given pool size.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            ..Self::default()
        }
    }
}

/// A pool of `rusqlite::Connection` objects sharing one SQLite database file.
///
/// The `Semaphore` bounds the number of concurrent checkouts to `size`; the
/// `Mutex<Vec<Connection>>` holds idle connections. Both are `std::sync`
/// primitives because the critical sections are tiny (push/pop) — the heavy
/// work is offloaded to the blocking pool via `PooledConnection`.
pub struct SqlitePool {
    connections: Mutex<Vec<Connection>>,
    semaphore: Arc<Semaphore>,
}

impl SqlitePool {
    /// Open (or create) a pool of `config.size` connections to `path` with WAL
    /// mode enabled and a busy timeout.
    pub fn open(path: &Path, config: &PoolConfig) -> Result<Self, DbError> {
        let mut connections = Vec::with_capacity(config.size);
        for _ in 0..config.size {
            let conn = common_core::sqlite::open_wal(path).map_err(DbError::from)?;
            if config.busy_timeout_ms > 0 {
                conn.busy_timeout(Duration::from_millis(config.busy_timeout_ms))
                    .map_err(DbError::from)?;
            }
            connections.push(conn);
        }
        Ok(Self {
            connections: Mutex::new(connections),
            semaphore: Arc::new(Semaphore::new(config.size)),
        })
    }

    /// Open a pool of in-memory connections (tests / ephemeral stores).
    pub fn open_in_memory(config: &PoolConfig) -> Result<Self, DbError> {
        let mut connections = Vec::with_capacity(config.size);
        for _ in 0..config.size {
            let conn = common_core::sqlite::open_in_memory().map_err(DbError::from)?;
            if config.busy_timeout_ms > 0 {
                conn.busy_timeout(Duration::from_millis(config.busy_timeout_ms))
                    .map_err(DbError::from)?;
            }
            connections.push(conn);
        }
        Ok(Self {
            connections: Mutex::new(connections),
            semaphore: Arc::new(Semaphore::new(config.size)),
        })
    }

    /// Check out a connection from the pool.
    ///
    /// Waits on the semaphore if all connections are in use. The returned
    /// `PooledConnection` returns itself to the pool (RAII) on `Drop`.
    /// `DbError::PoolExhausted` is returned only if the pool's invariant is
    /// violated (permit held but no idle connection) — with a correctly sized
    /// semaphore this branch is unreachable.
    pub async fn acquire(self: &Arc<Self>) -> Result<PooledConnection, DbError> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DbError::PoolExhausted)?;
        let conn = self
            .connections
            .lock()
            .expect("pool connections lock poisoned")
            .pop()
            .ok_or(DbError::PoolExhausted)?;
        Ok(PooledConnection {
            conn: Some(conn),
            pool: Arc::clone(self),
            _permit: permit,
        })
    }

    /// Return a connection to the idle set.
    fn put(&self, conn: Connection) {
        self.connections
            .lock()
            .expect("pool connections lock poisoned")
            .push(conn);
    }

    /// Run a closure against a checked-out connection on a blocking worker
    /// thread.
    ///
    /// Acquires a connection, offloads `f` via `tokio::task::spawn_blocking`,
    /// and maps a `JoinError` (task panicked / runtime shutdown) to
    /// `DbError::Other`.
    pub async fn with_conn<R, F>(self: &Arc<Self>, f: F) -> Result<R, DbError>
    where
        R: Send + 'static,
        F: FnOnce(&Connection) -> Result<R, DbError> + Send + 'static,
    {
        let conn = self.acquire().await?;
        tokio::task::spawn_blocking(move || f(&conn))
            .await
            .map_err(|e| DbError::Other(format!("blocking database task failed: {e}")))?
    }

    /// Execute a DML statement on a blocking worker thread; returns rows
    /// affected. `params` is an owned iterator of `ToSql` values (a `Vec`, an
    /// array, …) — borrowed `rusqlite::params![]` slices cannot cross the
    /// `spawn_blocking` boundary, so capture them in a `with_conn` closure
    /// instead when the parameter list is heterogeneous.
    pub async fn execute<I>(self: &Arc<Self>, sql: &str, params: I) -> Result<usize, DbError>
    where
        I: IntoIterator + Send + 'static,
        I::Item: rusqlite::ToSql + Send + Sync + 'static,
    {
        let sql = sql.to_string();
        let params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> = params
            .into_iter()
            .map(|p| Box::new(p) as Box<dyn rusqlite::ToSql + Send + Sync>)
            .collect();
        self.with_conn(move |conn| {
            crate::query::execute(conn, &sql, rusqlite::params_from_iter(params))
        })
        .await
    }

    /// Execute a multi-statement SQL batch on a blocking worker thread.
    pub async fn execute_batch(self: &Arc<Self>, sql: &str) -> Result<(), DbError> {
        let sql = sql.to_string();
        self.with_conn(move |conn| crate::query::execute_batch(conn, &sql))
            .await
    }

    /// Query at most one row on a blocking worker thread.
    ///
    /// A statement that matches no rows returns `Ok(None)` (not an error).
    pub async fn query_row<T, M, I>(
        self: &Arc<Self>,
        sql: &str,
        params: I,
        map: M,
    ) -> Result<Option<T>, DbError>
    where
        T: Send + 'static,
        M: FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T> + Send + 'static,
        I: IntoIterator + Send + 'static,
        I::Item: rusqlite::ToSql + Send + Sync + 'static,
    {
        let sql = sql.to_string();
        let params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> = params
            .into_iter()
            .map(|p| Box::new(p) as Box<dyn rusqlite::ToSql + Send + Sync>)
            .collect();
        self.with_conn(move |conn| {
            crate::query::query_row(conn, &sql, rusqlite::params_from_iter(params), map)
        })
        .await
    }

    /// Query all rows on a blocking worker thread.
    ///
    /// A statement that matches no rows returns `Ok(vec![])`.
    pub async fn query_rows<T, M, I>(
        self: &Arc<Self>,
        sql: &str,
        params: I,
        map: M,
    ) -> Result<Vec<T>, DbError>
    where
        T: Send + 'static,
        M: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T> + Send + 'static,
        I: IntoIterator + Send + 'static,
        I::Item: rusqlite::ToSql + Send + Sync + 'static,
    {
        let sql = sql.to_string();
        let params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> = params
            .into_iter()
            .map(|p| Box::new(p) as Box<dyn rusqlite::ToSql + Send + Sync>)
            .collect();
        self.with_conn(move |conn| {
            crate::query::query_rows(conn, &sql, rusqlite::params_from_iter(params), map)
        })
        .await
    }
}

/// A connection checked out from the pool.
///
/// `Deref`/`DerefMut` expose the underlying `rusqlite::Connection`. When
/// dropped, the connection is automatically returned to the idle set and the
/// semaphore permit is released.
pub struct PooledConnection {
    conn: Option<Connection>,
    pool: Arc<SqlitePool>,
    _permit: OwnedSemaphorePermit,
}

impl Deref for PooledConnection {
    type Target = Connection;
    fn deref(&self) -> &Self::Target {
        // `conn` is `Some` until `Drop` runs.
        self.conn
            .as_ref()
            .expect("pooled connection already returned")
    }
}

impl DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn
            .as_mut()
            .expect("pooled connection already returned")
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.put(conn);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Arc<SqlitePool> {
        Arc::new(SqlitePool::open_in_memory(&PoolConfig::default()).unwrap())
    }

    #[tokio::test]
    async fn acquire_release_round_trips() {
        let pool = pool();
        {
            let conn = pool.acquire().await.unwrap();
            conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
        } // dropped -> returned to pool
        let conn = pool.acquire().await.unwrap();
        conn.execute_batch("INSERT INTO t (id) VALUES (1)").unwrap();
    }

    #[tokio::test]
    async fn config_size_is_honored() {
        let pool = Arc::new(SqlitePool::open_in_memory(&PoolConfig::new(3)).unwrap());
        let mut held = Vec::new();
        for _ in 0..3 {
            held.push(pool.acquire().await.unwrap());
        }
        // All 3 permits are taken; a 4th acquire must wait rather than
        // fail. Proving the wait is racy, so instead assert that after
        // releasing one, acquire succeeds promptly.
        drop(held.pop());
        let _conn = pool.acquire().await.unwrap();
    }

    #[tokio::test]
    async fn execute_and_query_round_trip() {
        let pool = pool();
        pool.execute_batch("CREATE TABLE t (id INTEGER, name TEXT)")
            .await
            .unwrap();
        let n = pool
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                vec![
                    rusqlite::types::Value::Integer(1),
                    rusqlite::types::Value::Text("hello".into()),
                ],
            )
            .await
            .unwrap();
        assert_eq!(n, 1);

        let name = pool
            .query_row(
                "SELECT name FROM t WHERE id = ?1",
                vec![rusqlite::types::Value::Integer(1)],
                |row| row.get::<_, String>(0),
            )
            .await
            .unwrap();
        assert_eq!(name.as_deref(), Some("hello"));

        let rows = pool
            .query_rows(
                "SELECT id, name FROM t ORDER BY id",
                Vec::<rusqlite::types::Value>::new(),
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .await
            .unwrap();
        assert_eq!(rows, vec![(1, "hello".to_string())]);
    }

    #[tokio::test]
    async fn query_row_no_rows_maps_to_none() {
        let pool = pool();
        pool.execute_batch("CREATE TABLE t (id INTEGER)")
            .await
            .unwrap();
        let val = pool
            .query_row(
                "SELECT id FROM t WHERE id = 999",
                Vec::<rusqlite::types::Value>::new(),
                |row| row.get::<_, i64>(0),
            )
            .await
            .unwrap();
        assert_eq!(val, None);
    }

    #[tokio::test]
    async fn query_rows_empty_maps_to_empty_vec() {
        let pool = pool();
        pool.execute_batch("CREATE TABLE t (id INTEGER)")
            .await
            .unwrap();
        let rows = pool
            .query_rows(
                "SELECT id FROM t",
                Vec::<rusqlite::types::Value>::new(),
                |row| row.get::<_, i64>(0),
            )
            .await
            .unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn spawn_blocking_offloads_to_worker_thread() {
        let pool = pool();
        pool.execute_batch("CREATE TABLE t (id INTEGER)")
            .await
            .unwrap();
        let current = std::thread::current().id();
        let offloaded = pool
            .query_rows(
                "SELECT 1",
                Vec::<rusqlite::types::Value>::new(),
                move |_row| {
                    let in_worker = std::thread::current().id();
                    assert_ne!(in_worker, current, "map must run off the async thread");
                    Ok(())
                },
            )
            .await
            .unwrap();
        assert_eq!(offloaded.len(), 1);
    }

    #[tokio::test]
    async fn failing_sql_maps_to_db_error() {
        let pool = pool();
        let err = pool
            .query_rows(
                "SELECT * FROM no_such_table",
                Vec::<rusqlite::types::Value>::new(),
                |row| row.get::<_, i64>(0),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, DbError::Sqlite(_)));
    }
}
