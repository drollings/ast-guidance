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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::conn;

    struct TestMigration {
        version: u32,
        name: String,
        ddl: String,
    }

    impl TestMigration {
        fn new(version: u32, name: &str, ddl: &str) -> Self {
            Self {
                version,
                name: name.to_string(),
                ddl: ddl.to_string(),
            }
        }
    }

    impl Migration for TestMigration {
        fn version(&self) -> u32 {
            self.version
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
            tx.execute_batch(&self.ddl).map_err(DbError::from)
        }
    }

    #[test]
    fn ensure_column_adds_once_and_is_idempotent() {
        let conn = conn();
        conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .unwrap();
        ensure_column(&conn, "t", "label", "label TEXT NOT NULL DEFAULT ''").unwrap();
        ensure_column(&conn, "t", "label", "label TEXT NOT NULL DEFAULT ''").unwrap();
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(t)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(cols, vec!["id".to_string(), "label".to_string()]);
    }

    #[test]
    fn ensure_column_leaves_existing_columns_untouched() {
        let conn = conn();
        conn.execute_batch(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT NOT NULL DEFAULT '')",
        )
        .unwrap();
        // Data in the existing column must survive an ensure_column no-op.
        conn.execute("INSERT INTO t (id, label) VALUES (1, 'keep')", [])
            .unwrap();
        ensure_column(&conn, "t", "label", "label TEXT NOT NULL DEFAULT ''").unwrap();
        let label = conn
            .query_row("SELECT label FROM t WHERE id = 1", [], |row| {
                row.get::<_, String>(0)
            })
            .unwrap();
        assert_eq!(label, "keep");
    }

    #[test]
    fn migrate_applies_in_order_and_tracks_version() {
        let conn = conn();
        let migrations: [&dyn Migration; 3] = [
            &TestMigration::new(1, "base", "CREATE TABLE t (id INTEGER PRIMARY KEY)"),
            &TestMigration::new(2, "col_a", "ALTER TABLE t ADD COLUMN a TEXT"),
            &TestMigration::new(3, "col_b", "ALTER TABLE t ADD COLUMN b TEXT"),
        ];
        assert_eq!(schema_version(&conn).unwrap(), 0);
        migrate(&conn, &migrations).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), 3);

        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(t)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            cols,
            vec!["id".to_string(), "a".to_string(), "b".to_string(),]
        );
    }

    #[test]
    fn migrate_skips_already_applied() {
        let conn = conn();
        let migrations: [&dyn Migration; 2] = [
            &TestMigration::new(1, "base", "CREATE TABLE t (id INTEGER PRIMARY KEY)"),
            &TestMigration::new(2, "col_a", "ALTER TABLE t ADD COLUMN a TEXT"),
        ];
        migrate(&conn, &migrations).unwrap();
        // Second run must be a no-op: re-running "CREATE TABLE t" would fail
        // with a duplicate-object error if it were not skipped.
        migrate(&conn, &migrations).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), 2);
    }

    #[test]
    fn migrate_applies_only_pending_migrations_from_later_version() {
        let conn = conn();
        let base: [&dyn Migration; 2] = [
            &TestMigration::new(1, "base", "CREATE TABLE t (id INTEGER PRIMARY KEY)"),
            &TestMigration::new(2, "col_a", "ALTER TABLE t ADD COLUMN a TEXT"),
        ];
        migrate(&conn, &base).unwrap();

        // A later migration set re-applies only what is above user_version.
        let more: [&dyn Migration; 3] = [
            &TestMigration::new(1, "base", "CREATE TABLE t (id INTEGER PRIMARY KEY)"),
            &TestMigration::new(2, "col_a", "ALTER TABLE t ADD COLUMN a TEXT"),
            &TestMigration::new(3, "col_b", "ALTER TABLE t ADD COLUMN b TEXT"),
        ];
        migrate(&conn, &more).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), 3);

        let has_b: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('t') WHERE name = 'b'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_b, 1);
    }
}
