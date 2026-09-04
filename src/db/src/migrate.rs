//! Idempotent schema migrations.
//!
//! Generalizes the ad-hoc `ensure_column` sequences that consumers hand-roll
//! (e.g. `router/src/ledger.rs:636-641` and the holographic store's inline
//! `PRAGMA table_info` check) into a versioned framework driven by
//! `PRAGMA user_version`.

use rusqlite::Connection;

use crate::error::DbError;

/// Add a column to `table` if it does not already exist.
///
/// Port of `router/src/ledger.rs:636-641`, made table-generic and idempotent:
/// it checks `pragma_table_info(table)` for `name` and issues
/// `ALTER TABLE {table} ADD COLUMN {ddl}` only when absent. `ddl` is the full
/// column definition (name + type + default), e.g.
/// `"label TEXT NOT NULL DEFAULT ''"`.
pub fn ensure_column(conn: &Connection, table: &str, name: &str, ddl: &str) -> Result<(), DbError> {
    let has_sql = format!("SELECT 1 FROM pragma_table_info('{table}') WHERE name = ?1");
    let has = conn
        .prepare(&has_sql)
        .and_then(|mut stmt| stmt.query_row(rusqlite::params![name], |_| Ok(())))
        .is_ok();
    if !has {
        let alter_sql = format!("ALTER TABLE {table} ADD COLUMN {ddl}");
        conn.execute_batch(&alter_sql).map_err(DbError::from)?;
    }
    Ok(())
}

/// A single idempotent schema migration.
///
/// `version` must be strictly monotonic within a migration list. Each `up`
/// runs inside a transaction; on success the framework bumps
/// `PRAGMA user_version` to the highest applied version.
pub trait Migration {
    /// Monotonic version number (matches the `PRAGMA user_version` it moves
    /// the database to).
    fn version(&self) -> u32;
    /// Human-readable name for logging / diagnostics.
    fn name(&self) -> &str;
    /// Apply the migration's DDL/data changes inside the transaction.
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError>;
}

/// Read the database's schema version (`PRAGMA user_version`).
pub fn schema_version(conn: &Connection) -> Result<u32, DbError> {
    let v: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(DbError::from)?;
    Ok(v as u32)
}

/// Apply all migrations with `version > current` in version order inside a
/// single transaction, then bump `PRAGMA user_version` to the highest applied.
///
/// Already-applied migrations (`version <= current`) are skipped, so repeated
/// calls are no-ops and the sequence is idempotent across processes.
pub fn migrate(conn: &Connection, migrations: &[&dyn Migration]) -> Result<(), DbError> {
    let current = schema_version(conn)?;
    let mut pending: Vec<&dyn Migration> = migrations
        .iter()
        .copied()
        .filter(|m| m.version() > current)
        .collect();
    if pending.is_empty() {
        return Ok(());
    }
    pending.sort_by_key(|m| m.version());

    let mut tx = conn.unchecked_transaction().map_err(DbError::from)?;
    for migration in &pending {
        migration.up(&mut tx)?;
    }
    let new_version = pending.last().expect("non-empty pending list").version();
    tx.pragma_update(None, "user_version", new_version)
        .map_err(DbError::from)?;
    tx.commit().map_err(DbError::from)?;
    Ok(())
}

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/migrate.rs"]
mod tests;
