use super::*;
use std::sync::Arc;

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

#[tokio::test]
async fn execute_and_query_round_trip() {
    use crate::tests::common::assert_execute_query_round_trip;
    assert_execute_query_round_trip(|| {
        Box::pin(async move {
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
            let name = store
                .query_row(
                    "SELECT name FROM t WHERE id = ?1",
                    rusqlite::params![1],
                    |row| row.get::<_, String>(0),
                )
                .unwrap();
            let rows = store
                .query_rows("SELECT id, name FROM t ORDER BY id", &[], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .unwrap();
            (n, name, rows)
        })
    })
    .await;
}

#[tokio::test]
async fn query_row_no_rows_maps_to_none() {
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

#[tokio::test]
async fn transaction_commit_and_rollback() {
    use crate::tests::common::assert_transaction_commit_rollback;
    assert_transaction_commit_rollback(|commit| {
        Box::pin(async move {
            let store = SqliteStore::open_in_memory().unwrap();
            store.init_schema("CREATE TABLE t (id INTEGER PRIMARY KEY)").unwrap();
            let result: Result<(), DbError> = store.transaction(|tx| {
                tx.execute("INSERT INTO t (id) VALUES (?1)", rusqlite::params![7])?;
                if commit {
                    Ok(())
                } else {
                    Err(DbError::Other("boom".into()))
                }
            });
            if commit {
                result.unwrap();
            } else {
                assert!(result.is_err());
            }
            store
                .query_row(
                    "SELECT COUNT(*) FROM t",
                    &[],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                .unwrap()
        })
    })
    .await;
}

#[tokio::test]
async fn poison_recovery_via_lock() {
    use crate::tests::common::assert_poison_recovery;
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    store
        .init_schema("CREATE TABLE t (id INTEGER)")
        .unwrap();
    let poisoned = Arc::clone(&store);
    assert_poison_recovery(
        move || {
            // Simulate a panic while holding the lock.
            let _guard = lock(&poisoned.conn);
            panic!("boom");
        },
        || {
            Box::pin(async move {
                // The store is still usable after poison recovery.
                store.execute("INSERT INTO t (id) VALUES (1)", &[]).unwrap();
            })
        },
    )
    .await;
}
