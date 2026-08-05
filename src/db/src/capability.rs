//! The capability-gated async database surface (D5, D2.2).
//!
//! `DbCapability` is a `fluent_wvr::Capability` token over a shared
//! `SqlitePool`. It is the successor to `fluent-concurrency::io::db`'s
//! `DbCapability`: the pooled connection machinery now lives in
//! `fluent-db::pool`, and the token is re-exported from
//! `fluent-concurrency::io::db` so existing callers keep their module path.
//!
//! Capability gating is **not** reimplemented here: the canonical primitive is
//! `fluent_wvr::capability::check_capability`, which consults the
//! `CURRENT_CAPS` task-local installed by `fluent-concurrency`'s
//! `Scope`/`Zone`. This crate's `DbCapability` is the *token*; the gating
//! seam stays in `fluent-wvr`.
//!
//! The lossy all-values-as-strings `query` / `execute` methods are **deprecated**
//! (M3, §0.5): new code should use the typed `SqlitePool::query_row` /
//! `query_rows` / `execute` helpers with typed row mappers.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::error::IoError;
use fluent_wvr::capability::{check_capability, Capability};

use crate::error::DbError;
use crate::pool::{PoolConfig, SqlitePool};

/// Capability token for pooled SQLite database access.
pub struct DbCapability {
    pool: Arc<SqlitePool>,
}

impl Capability for DbCapability {
    fn name(&self) -> &'static str {
        "db"
    }
}

impl DbCapability {
    /// Opens a database at the given path (or `:memory:` for an ephemeral
    /// in-memory pool) with the default pool configuration (5 connections).
    pub fn open(path: &str) -> Result<Self, DbError> {
        Self::open_with_config(path, PoolConfig::default())
    }

    /// Opens a database with a custom pool configuration.
    pub fn open_with_config(path: &str, config: PoolConfig) -> Result<Self, DbError> {
        let pool = Arc::new(SqlitePool::open(Path::new(path), &config)?);
        Ok(Self { pool })
    }

    /// Access the underlying connection pool.
    pub fn pool(&self) -> &Arc<SqlitePool> {
        &self.pool
    }

    /// Executes a SQL query and returns rows as `Vec<HashMap<String, String>>`
    /// (all values stringified).
    ///
    /// **Deprecated**: this lossy string-map shape is superseded by the typed
    /// `SqlitePool::query_row` / `query_rows` helpers. Kept so existing
    /// callers compile unchanged (§0.5 M3).
    #[deprecated = "use SqlitePool::query_rows/query_row with typed row mappers instead"]
    pub async fn query(&self, sql: &str) -> Result<Vec<HashMap<String, String>>, IoError> {
        check_capability(self)?;
        let sql = sql.to_string();
        let pool = Arc::clone(&self.pool);
        let conn = pool
            .acquire()
            .await
            .map_err(|e| IoError(std::io::Error::other(e.to_string())))?;
        let result = tokio::task::spawn_blocking(move || {
            let mut stmt = conn.prepare(&sql).map_err(std::io::Error::other)?;

            let columns: Vec<String> = stmt
                .column_names()
                .iter()
                .map(ToString::to_string)
                .collect();

            let mut rows = Vec::new();
            let mut rows_iter = stmt.query([]).map_err(std::io::Error::other)?;

            while let Some(row) = rows_iter.next().map_err(std::io::Error::other)? {
                let mut map = HashMap::new();
                for (i, col) in columns.iter().enumerate() {
                    let value: String = match row.get::<_, rusqlite::types::Value>(i) {
                        Ok(rusqlite::types::Value::Integer(n)) => n.to_string(),
                        Ok(rusqlite::types::Value::Real(f)) => f.to_string(),
                        Ok(rusqlite::types::Value::Text(s)) => s,
                        Ok(rusqlite::types::Value::Blob(b)) => format!("<blob {} bytes>", b.len()),
                        _ => String::new(),
                    };
                    map.insert(col.clone(), value);
                }
                rows.push(map);
            }

            Ok(rows)
        })
        .await;

        match result {
            Ok(inner) => inner,
            Err(e) => Err(IoError(std::io::Error::other(e.to_string()))),
        }
    }

    /// Executes a SQL statement (INSERT, UPDATE, DELETE) and returns the
    /// number of rows affected.
    ///
    /// **Deprecated**: superseded by the typed `SqlitePool::execute`. Kept so
    /// existing callers compile unchanged (§0.5 M3).
    #[deprecated = "use SqlitePool::execute with typed params instead"]
    pub async fn execute(&self, sql: &str) -> Result<usize, IoError> {
        check_capability(self)?;
        let sql = sql.to_string();
        let pool = Arc::clone(&self.pool);
        let conn = pool
            .acquire()
            .await
            .map_err(|e| IoError(std::io::Error::other(e.to_string())))?;
        let result = tokio::task::spawn_blocking(move || {
            let rows_affected = conn.execute(&sql, []).map_err(std::io::Error::other)?;
            Ok(rows_affected)
        })
        .await;

        match result {
            Ok(inner) => inner,
            Err(e) => Err(IoError(std::io::Error::other(e.to_string()))),
        }
    }
}

#[cfg(test)]
mod tests {
    // The deprecated `query`/`execute` are exercised here as the behavior
    // oracle for legacy callers (§0.5 M3).
    #![allow(deprecated)]
    use super::*;
    use fluent_wvr::capability::CURRENT_CAPS;
    use fluent_wvr::CapabilitySet;

    fn db() -> DbCapability {
        DbCapability::open(":memory:").unwrap()
    }

    fn db_caps() -> CapabilitySet {
        CapabilitySet::new().with(db())
    }

    #[tokio::test]
    async fn open_and_pool_round_trip() {
        let db = db();
        let conn = db.pool().acquire().await.unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
    }

    #[tokio::test]
    async fn query_execute_round_trip_within_capability() {
        let db = db();
        let caps = db_caps();
        CURRENT_CAPS
            .scope(caps, async {
                db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
                    .await
                    .unwrap();
                db.execute("INSERT INTO t VALUES (1, 'hello')")
                    .await
                    .unwrap();
                let rows = db.query("SELECT * FROM t").await.unwrap();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0]["id"], "1");
                assert_eq!(rows[0]["name"], "hello");
            })
            .await;
    }

    #[tokio::test]
    async fn query_without_capability_denied() {
        let db = db();
        let result = db.query("SELECT 1").await;
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn execute_without_capability_denied() {
        let db = db();
        let result = db.execute("CREATE TABLE t (id INTEGER)").await;
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn capability_name_is_db() {
        assert_eq!(db().name(), "db");
    }
}
