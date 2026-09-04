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
