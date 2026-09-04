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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rusqlite::Connection;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use common_core::sync::lock;

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

/// Where the pool's connections point, so a replacement opened after a
/// health-check failure targets the *same* database.
enum Backing {
    /// A file-backed database (WAL mode).
    File(PathBuf),
    /// A named shared-cache in-memory database (all `config.size` connections
    /// share one DB — see `common_core::sqlite::open_shared_in_memory`).
    Memory(Arc<str>),
}

/// A pool of `rusqlite::Connection` objects sharing one SQLite database file.
///
/// The `Semaphore` bounds the number of concurrent checkouts to `size`; the
/// `Mutex<Vec<Connection>>` holds idle connections. Both are `std::sync`
/// primitives because the critical sections are tiny (push/pop) — the heavy
/// work is offloaded to the blocking pool via `PooledConnection`.
pub struct SqlitePool {
    pub(crate) connections: Mutex<Vec<Connection>>,
    semaphore: Arc<Semaphore>,
    /// Where to reopen a fresh connection when a returned one fails its health
    /// check — the same database the pool was opened against, so a replaced
    /// connection never silently becomes a fresh private one.
    backing: Backing,
    /// Sizing / timeout config, retained so replacements are opened with the
    /// same busy timeout as the original connections.
    config: PoolConfig,
}

impl SqlitePool {
    /// Open (or create) a pool of `config.size` connections to `path` with WAL
    /// mode enabled and a busy timeout.
    ///
    /// A literal `:memory:` path is SQLite's *per-connection* private in-memory
    /// database — `config.size` independent empty DBs, so writes vanish across
    /// checkouts. It is routed to the shared-name path so an in-memory pool is
    /// coherent.
    pub fn open(path: &Path, config: &PoolConfig) -> Result<Self, DbError> {
        if path == Path::new(":memory:") {
            return Self::open_in_memory(config);
        }
        let mut connections = Vec::with_capacity(config.size);
        for _ in 0..config.size {
            connections.push(Self::open_wal_configured(path, config)?);
        }
        Ok(Self {
            connections: Mutex::new(connections),
            semaphore: Arc::new(Semaphore::new(config.size)),
            backing: Backing::File(path.to_path_buf()),
            config: *config,
        })
    }

    /// Open a pool of in-memory connections (tests / ephemeral stores).
    ///
    /// All `config.size` connections are opened to one process-unique
    /// shared-cache in-memory database (`memdb{N}`), so concurrent checkouts
    /// see the *same* data. The shared cache lives as long as at least one
    /// connection to it stays open — the idle set keeps `config.size`
    /// connections alive for the pool's lifetime.
    ///
    /// **Writer serialization:** a shared cache has one write lock across all
    /// connections. The semaphore (at most `size` concurrent checkouts) plus
    /// the `busy_timeout` (5s before `SQLITE_BUSY`) already mitigate this for
    /// the pool's traffic; do not re-architect around it.
    pub fn open_in_memory(config: &PoolConfig) -> Result<Self, DbError> {
        let name = shared_memory_name();
        let mut connections = Vec::with_capacity(config.size);
        for _ in 0..config.size {
            connections.push(Self::open_shared_in_memory_configured(&name, config)?);
        }
        Ok(Self {
            connections: Mutex::new(connections),
            semaphore: Arc::new(Semaphore::new(config.size)),
            backing: Backing::Memory(name),
            config: *config,
        })
    }

    /// Open a single WAL connection to `path` with the pool's busy timeout.
    fn open_wal_configured(path: &Path, config: &PoolConfig) -> Result<Connection, DbError> {
        let conn = common_core::sqlite::open_wal(path).map_err(DbError::from)?;
        if config.busy_timeout_ms > 0 {
            conn.busy_timeout(Duration::from_millis(config.busy_timeout_ms))
                .map_err(DbError::from)?;
        }
        Ok(conn)
    }

    /// Open a single connection to the pool's named shared-cache in-memory
    /// database with the pool's busy timeout.
    fn open_shared_in_memory_configured(
        name: &str,
        config: &PoolConfig,
    ) -> Result<Connection, DbError> {
        let conn = common_core::sqlite::open_shared_in_memory(name).map_err(DbError::from)?;
        if config.busy_timeout_ms > 0 {
            conn.busy_timeout(Duration::from_millis(config.busy_timeout_ms))
                .map_err(DbError::from)?;
        }
        Ok(conn)
    }

    /// Open a fresh connection matching this pool's backing store.
    fn open_connection(&self) -> Result<Connection, DbError> {
        match &self.backing {
            Backing::File(path) => Self::open_wal_configured(path, &self.config),
            Backing::Memory(name) => Self::open_shared_in_memory_configured(name, &self.config),
        }
    }

    /// Check out a connection from the pool.
    ///
    /// Requires a `DbCapability` token in the current task-local (the same
    /// gate `with_conn` and the typed helpers enforce) — holding a raw
    /// `Arc<SqlitePool>` does not bypass capability gating. Waits on the
    /// semaphore if all connections are in use. The returned
    /// `PooledConnection` returns itself to the pool (RAII) on `Drop`.
    /// `DbError::PoolExhausted` is returned only if the pool's invariant is
    /// violated (permit held but no idle connection) — with a correctly sized
    /// semaphore this branch is unreachable.
    pub async fn acquire(self: &Arc<Self>) -> Result<PooledConnection, DbError> {
        crate::capability::check_db_capability()?;
        self.acquire_ungated().await
    }

    /// The ungated checkout path: semaphore acquire → pop → `PooledConnection`.
    ///
    /// `pub(crate)` because every effect entry point must gate *once*: callers
    /// that already asserted `check_db_capability` reuse this instead of
    /// double-checking the task-local.
    pub(crate) async fn acquire_ungated(self: &Arc<Self>) -> Result<PooledConnection, DbError> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DbError::PoolExhausted)?;
        let conn = lock(&self.connections)
            .pop()
            .ok_or(DbError::PoolExhausted)?;
        Ok(PooledConnection {
            conn: Some(conn),
            pool: Arc::clone(self),
            _permit: permit,
        })
    }

    /// Return a connection to the idle set, health-checking it first.
    ///
    /// A connection that fails the cheap `SELECT 1` probe (e.g. one whose
    /// underlying handle was closed out from under the pool) is discarded and
    /// replaced with a fresh connection so the pool keeps `config.size` live
    /// connections — a poisoned connection is never re-queued for reuse.
    fn put(&self, conn: Connection) {
        if Self::connection_is_healthy(&conn) {
            lock(&self.connections).push(conn);
        } else {
            tracing::warn!("discarding unhealthy pooled connection");
            // Open the replacement BEFORE dropping the unhealthy connection: a
            // shared-cache in-memory database lives exactly as long as at least
            // one connection to it stays open, so dropping the last holder
            // first would destroy the very DB the replacement must reopen.
            match self.open_connection() {
                Ok(fresh) => {
                    lock(&self.connections).push(fresh);
                    drop(conn);
                }
                Err(e) => {
                    tracing::error!("failed to open replacement connection: {e}");
                    drop(conn);
                }
            }
        }
    }

    /// Cheap guardrail probe: the connection must answer a trivial statement.
    fn connection_is_healthy(conn: &Connection) -> bool {
        conn.query_row("SELECT 1", [], |_| Ok(())).is_ok()
    }

    /// Run a closure against a checked-out connection on a blocking worker
    /// thread.
    ///
    /// Acquires a connection, offloads `f` via `tokio::task::spawn_blocking`,
    /// and maps a `JoinError` (task panicked / runtime shutdown) to
    /// `DbError::Other`. Requires a `DbCapability` token in the current
    /// task-local, so holding a raw `Arc<SqlitePool>` does not bypass
    /// capability gating.
    pub async fn with_conn<R, F>(self: &Arc<Self>, f: F) -> Result<R, DbError>
    where
        R: Send + 'static,
        F: FnOnce(&Connection) -> Result<R, DbError> + Send + 'static,
    {
        crate::capability::check_db_capability()?;
        let conn = self.acquire_ungated().await?;
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

    /// Run a closure inside a transaction on a blocking worker thread.
    ///
    /// Commits on `Ok`, rolls back on `Err` (the transaction is dropped while
    /// uncommitted, which rolls it back) — the API-parity counterpart to
    /// [`crate::store::SqliteStore::transaction`] for the pooled, async store.
    /// Requires a `DbCapability` token in the current task-local (the same gate
    /// `with_conn` and the typed helpers enforce). The closure borrows the
    /// checked-out connection, so no per-call param boxing crosses the
    /// `spawn_blocking` boundary.
    pub async fn transaction<T>(
        self: &Arc<Self>,
        f: impl FnOnce(&mut rusqlite::Transaction<'_>) -> Result<T, DbError> + Send + 'static,
    ) -> Result<T, DbError>
    where
        T: Send + 'static,
    {
        crate::capability::check_db_capability()?;
        let mut conn = self.acquire_ungated().await?;
        tokio::task::spawn_blocking(move || crate::query::transaction(&mut conn, f))
            .await
            .map_err(|e| DbError::Other(format!("blocking database task failed: {e}")))?
    }

    // ── Borrowed-connection helpers ───────────────────────────────
    //
    // The async helpers above box each parameter as
    // `Box<dyn ToSql + Send + Sync>` so an owned `IntoIterator` can cross the
    // `spawn_blocking` boundary. Callers who are *already* inside a blocking op
    // — a `with_conn`/`transaction` closure, a `PooledConnection` deref, or a
    // `DbWorkUnit` store op — hold a `&Connection` on the current thread and
    // should not pay that per-query allocation. These borrowed variants run the
    // canonical `crate::query` helpers against such a connection with borrowed
    // `rusqlite::Params` (`rusqlite::params![...]`, `[]`, a `ParamsFromIter`).
    // No capability token is re-checked: the connection was already checked out
    // through a gated path (`acquire`/`with_conn`/`transaction`).

    /// Query at most one row against an already-borrowed connection.
    /// A statement that matches no rows returns `Ok(None)` (not an error).
    pub fn query_row_borrowed<T, P, M>(
        conn: &Connection,
        sql: &str,
        params: P,
        map: M,
    ) -> Result<Option<T>, DbError>
    where
        P: rusqlite::Params,
        M: FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        crate::query::query_row(conn, sql, params, map)
    }

    /// Query all rows against an already-borrowed connection.
    /// A statement that matches no rows returns `Ok(vec![])`.
    pub fn query_rows_borrowed<T, P, M>(
        conn: &Connection,
        sql: &str,
        params: P,
        map: M,
    ) -> Result<Vec<T>, DbError>
    where
        P: rusqlite::Params,
        M: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        crate::query::query_rows(conn, sql, params, map)
    }

    /// Execute a DML statement against an already-borrowed connection;
    /// returns rows affected.
    pub fn execute_borrowed<P: rusqlite::Params>(
        conn: &Connection,
        sql: &str,
        params: P,
    ) -> Result<usize, DbError> {
        crate::query::execute(conn, sql, params)
    }

    /// Query all rows against an already-borrowed connection with dynamic-arity
    /// params from an iterator — the generalized `in_clause` +
    /// `params_from_iter` combo, exposed on the pool surface for borrowed
    /// connections (the async helpers have no such variant).
    pub fn query_rows_from_iter_borrowed<T, I, M>(
        conn: &Connection,
        sql: &str,
        params: I,
        map: M,
    ) -> Result<Vec<T>, DbError>
    where
        I: IntoIterator,
        I::Item: rusqlite::ToSql,
        M: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        crate::query::query_rows_from_iter(conn, sql, params, map)
    }
}

/// Allocate a process-unique shared-cache in-memory database name (`memdb{N}`).
///
/// Two independent pools never share a cache by accident — the name is unique
/// per pool *instance*, not per class.
fn shared_memory_name() -> Arc<str> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    Arc::from(format!("memdb{}", NEXT.fetch_add(1, Ordering::Relaxed)).as_str())
}

/// A connection checked out from the pool.
///
/// `Deref`/`DerefMut` expose the underlying `rusqlite::Connection`. When
/// dropped, the connection is health-checked (`SELECT 1`) and returned to the
/// idle set — a poisoned connection is discarded and replaced instead of being
/// re-queued — and the semaphore permit is released.
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

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/pool.rs"]
mod tests;
