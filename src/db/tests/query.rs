use super::*;
use common_core::sqlite::in_clause;
use crate::tests::common::conn;

fn seed(conn: &Connection) {
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        .unwrap();
    conn.execute("INSERT INTO t (id, name) VALUES (1, 'alpha')", [])
        .unwrap();
    conn.execute("INSERT INTO t (id, name) VALUES (2, 'beta')", [])
        .unwrap();
}

#[test]
fn query_row_returns_some_when_row_exists() {
    let conn = conn();
    seed(&conn);
    let name = query_row(
        &conn,
        "SELECT name FROM t WHERE id = ?1",
        rusqlite::params![1],
        |row| row.get::<_, String>(0),
    )
    .unwrap();
    assert_eq!(name.as_deref(), Some("alpha"));
}

#[test]
fn query_row_returns_none_when_no_rows() {
    let conn = conn();
    seed(&conn);
    let val = query_row(
        &conn,
        "SELECT name FROM t WHERE id = ?1",
        rusqlite::params![99],
        |row| row.get::<_, String>(0),
    )
    .unwrap();
    assert_eq!(val, None);
}

#[test]
fn query_rows_returns_all_rows() {
    let conn = conn();
    seed(&conn);
    let rows = query_rows(&conn, "SELECT name FROM t ORDER BY id", [], |row| {
        row.get::<_, String>(0)
    })
    .unwrap();
    assert_eq!(rows, vec!["alpha".to_string(), "beta".to_string()]);
}

#[test]
fn query_rows_empty_when_no_rows() {
    let conn = conn();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        .unwrap();
    let rows = query_rows(&conn, "SELECT id FROM t", [], |row| row.get::<_, i64>(0)).unwrap();
    assert!(rows.is_empty());
}

#[test]
fn execute_returns_rows_affected() {
    let conn = conn();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        .unwrap();
    let n = execute(
        &conn,
        "INSERT INTO t (id) VALUES (?1)",
        rusqlite::params![5],
    )
    .unwrap();
    assert_eq!(n, 1);
    let n = execute(
        &conn,
        "INSERT INTO t (id) VALUES (?1)",
        rusqlite::params![6],
    )
    .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn execute_batch_runs_multi_statement_ddl() {
    let conn = conn();
    execute_batch(
        &conn,
        "CREATE TABLE a (id INTEGER); CREATE TABLE b (id INTEGER);",
    )
    .unwrap();
    assert!(conn.prepare("SELECT * FROM b").is_ok());
}

#[test]
fn query_rows_from_iter_dynamic_arity_in_clause() {
    let conn = conn();
    seed(&conn);
    let names = vec!["alpha", "beta", "gamma"];
    let placeholders = in_clause(names.len());
    let sql = format!("SELECT id, name FROM t WHERE name IN ({placeholders})");
    let rows = query_rows_from_iter(&conn, &sql, names, |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })
    .unwrap();
    assert_eq!(
        rows,
        vec![(1, "alpha".to_string()), (2, "beta".to_string())]
    );
}

#[test]
fn query_rows_from_iter_empty_iterator_matches_nothing() {
    let conn = conn();
    seed(&conn);
    let names: Vec<&str> = Vec::new();
    let placeholders = in_clause(0);
    let sql = format!("SELECT id, name FROM t WHERE name IN ({placeholders})");
    let rows = query_rows_from_iter(&conn, &sql, names, |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })
    .unwrap();
    assert!(rows.is_empty());
}

#[test]
fn last_insert_rowid_tracks_inserts() {
    let conn = conn();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        .unwrap();
    execute(&conn, "INSERT INTO t (id) VALUES (7)", []).unwrap();
    assert_eq!(last_insert_rowid(&conn), 7);
}

#[tokio::test]
async fn transaction_commit_and_rollback() {
    use crate::tests::common::assert_transaction_commit_rollback;
    assert_transaction_commit_rollback(|commit| {
        Box::pin(async move {
            let mut conn = conn();
            conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
                .unwrap();
            let result: Result<(), DbError> = transaction(&mut conn, |tx| {
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
            query_row(&conn, "SELECT COUNT(*) FROM t", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
            .unwrap()
        })
    })
    .await;
}
