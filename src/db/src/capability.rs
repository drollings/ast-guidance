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
//! `Scope`/`SupervisedBatch`. This crate's `DbCapability` is the *token*; the gating
//! seam stays in `fluent-wvr`.

use std::path::Path;
use std::sync::Arc;

use fluent_wvr::capability::{Capability, CapabilitySet, CURRENT_CAPS};

use crate::error::DbError;
use crate::pool::{PoolConfig, SqlitePool};

/// Validate that a `DbCapability` token is present in the current task-local,
/// without needing a token value.
///
/// The typed `SqlitePool` helpers hold only the pool, not the token, so they
/// use this by-type check rather than `check_capability(&DbCapability)`.
pub(crate) fn check_db_capability() -> Result<(), DbError> {
    let present = CURRENT_CAPS
        .try_with(CapabilitySet::contains::<DbCapability>)
        .unwrap_or(false);
    if present {
        Ok(())
    } else {
        Err(DbError::PermissionDenied("missing capability: db".into()))
    }
}

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
    ///
    /// The `:memory:` path routes through the shared-cache in-memory pool path:
    /// all pool connections share one named in-memory database, so
    /// concurrent checkouts are coherent rather than each seeing a private
    /// empty DB.
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
}

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/capability.rs"]
mod tests;
