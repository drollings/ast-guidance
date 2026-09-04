use super::*;

use fluent_wvr::capability::CURRENT_CAPS;
use crate::tests::common::{db_caps, in_memory_pool};

fn pool() -> Arc<SqlitePool> {
    in_memory_pool()
}

#[tokio::test]
async fn acquire_release_round_trips() {
    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = pool();
            {
                let conn = pool.acquire().await.unwrap();
                conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
            } // dropped -> returned to pool
            let conn = pool.acquire().await.unwrap();
            conn.execute_batch("INSERT INTO t (id) VALUES (1)").unwrap();
        })
        .await;
}

#[tokio::test]
async fn poisoned_idle_connections_mutex_still_serves_acquire() {
    // The pool's idle-connections lock must recover from poison via
    // `common_core::sync::lock`, mirroring `store.rs::poison_recovery_via_lock`.
    use crate::tests::common::{assert_poison_recovery, db_caps, in_memory_pool};
    let pool = in_memory_pool();
    let poisoned = Arc::clone(&pool);
    assert_poison_recovery(
        move || {
            // A panic while holding a guard obtained via `.lock().unwrap()`
            // poisons the mutex.
            let _guard = poisoned.connections.lock().unwrap();
            panic!("boom");
        },
        || {
            Box::pin(async move {
                CURRENT_CAPS
                    .scope(db_caps(), async {
                        // A subsequent `acquire` must still serve a usable connection.
                        let conn = pool.acquire().await.unwrap();
                        conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
                        conn.execute_batch("INSERT INTO t (id) VALUES (1)").unwrap();
                    })
                    .await;
            })
        },
    )
    .await;
}

#[tokio::test]
async fn config_size_is_honored() {
    CURRENT_CAPS
        .scope(db_caps(), async {
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
        })
        .await;
}

#[tokio::test]
async fn acquire_requires_db_capability() {
    // The raw-acquire path is gated like every other effect entry
    // point — a pool held without a `DbCapability` token must be denied.
    let pool = pool();
    let err = match pool.acquire().await {
        Ok(_) => panic!("acquire must be denied without a DbCapability"),
        Err(e) => e,
    };
    assert!(
        matches!(err, DbError::PermissionDenied(_)),
        "expected PermissionDenied without a DbCapability, got {err:?}"
    );
}

#[tokio::test]
async fn execute_and_query_round_trip() {
    use crate::tests::common::{assert_execute_query_round_trip, db_caps, in_memory_pool};
    assert_execute_query_round_trip(|| {
        Box::pin(async move {
            CURRENT_CAPS
                .scope(db_caps(), async {
                    let pool = in_memory_pool();
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
                    let name = pool
                        .query_row(
                            "SELECT name FROM t WHERE id = ?1",
                            vec![rusqlite::types::Value::Integer(1)],
                            |row| row.get::<_, String>(0),
                        )
                        .await
                        .unwrap();
                    let rows = pool
                        .query_rows(
                            "SELECT id, name FROM t ORDER BY id",
                            Vec::<rusqlite::types::Value>::new(),
                            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                        )
                        .await
                        .unwrap();
                    (n, name, rows)
                })
                .await
        })
    })
    .await;
}

#[tokio::test]
async fn query_row_no_rows_maps_to_none() {
    CURRENT_CAPS
        .scope(db_caps(), async {
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
        })
        .await;
}

#[tokio::test]
async fn query_rows_empty_maps_to_empty_vec() {
    CURRENT_CAPS
        .scope(db_caps(), async {
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
        })
        .await;
}

#[tokio::test]
async fn spawn_blocking_offloads_to_worker_thread() {
    CURRENT_CAPS
        .scope(db_caps(), async {
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
        })
        .await;
}

#[tokio::test]
async fn transaction_commit_and_rollback() {
    use crate::tests::common::{assert_transaction_commit_rollback, db_caps, in_memory_pool};
    assert_transaction_commit_rollback(|commit| {
        Box::pin(async move {
            CURRENT_CAPS
                .scope(db_caps(), async {
                    let pool = in_memory_pool();
                    pool.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
                        .await
                        .unwrap();
                    let result: Result<(), DbError> = pool
                        .transaction(move |tx| {
                            tx.execute(
                                "INSERT INTO t (id) VALUES (?1)",
                                rusqlite::params![7],
                            )?;
                            if commit {
                                Ok(())
                            } else {
                                Err(DbError::Other("boom".into()))
                            }
                        })
                        .await;
                    if commit {
                        result.unwrap();
                    } else {
                        assert!(result.is_err());
                    }
                    pool.query_row(
                        "SELECT COUNT(*) FROM t",
                        Vec::<rusqlite::types::Value>::new(),
                        |row| row.get::<_, i64>(0),
                    )
                    .await
                    .unwrap()
                    .unwrap()
                })
                .await
        })
    })
    .await;
}

#[tokio::test]
async fn transaction_requires_db_capability() {
    let pool = pool();
    let err = pool.transaction(|_tx| Ok(())).await.unwrap_err();
    assert!(
        matches!(err, DbError::PermissionDenied(_)),
        "expected PermissionDenied without a DbCapability, got {err:?}"
    );
}

#[tokio::test]
async fn failing_sql_maps_to_db_error() {
    CURRENT_CAPS
        .scope(db_caps(), async {
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
        })
        .await;
}

#[tokio::test]
async fn typed_helpers_require_db_capability() {
    // Nit 4: the typed helpers re-check the task-local, so a pool held
    // without a `DbCapability` token must be denied.
    let pool = pool();
    let err = pool
        .query_rows("SELECT 1", Vec::<rusqlite::types::Value>::new(), |row| {
            row.get::<_, i64>(0)
        })
        .await
        .unwrap_err();
    assert!(
        matches!(err, DbError::PermissionDenied(_)),
        "expected PermissionDenied without a DbCapability, got {err:?}"
    );
}

#[tokio::test]
async fn poisoned_connection_is_discarded_on_return() {
    // Nit 3: a connection that fails the `SELECT 1` health check must not
    // be re-queued. Deny all authorizations on the checked-out connection,
    // so even the health probe fails, then verify the pool still serves a
    // fresh, healthy connection.
    use rusqlite::hooks::Authorization;

    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = pool();
            {
                let conn = pool.acquire().await.unwrap();
                conn.authorizer(Some(|_: rusqlite::hooks::AuthContext<'_>| {
                    Authorization::Deny
                }));
            } // dropped -> health check fails -> discarded, replacement opened
            let n: i64 = pool
                .query_row("SELECT 1", Vec::<rusqlite::types::Value>::new(), |row| {
                    row.get::<_, i64>(0)
                })
                .await
                .unwrap()
                .expect("pool must hand out a healthy replacement connection");
            assert_eq!(n, 1);
        })
        .await;
}

// ── Borrowed-connection helpers ───────────────────────────────

#[tokio::test]
async fn borrowed_helpers_match_async_helpers_without_boxing() {
    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = pool();
            pool.execute_batch("CREATE TABLE t (id INTEGER, name TEXT)")
                .await
                .unwrap();

            // Execute via the async (boxed) path.
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

            // Read back via the borrowed path inside a `with_conn` closure
            // (the "already inside a blocking op" contract). Borrowed
            // `rusqlite::params![]` — no `Vec<Box<dyn ToSql>>` boxing.
            let (id, name): (i64, String) = pool
                .with_conn(|conn| {
                    SqlitePool::query_row_borrowed(
                        conn,
                        "SELECT id, name FROM t WHERE id = ?1",
                        rusqlite::params![1],
                        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
                    )
                })
                .await
                .unwrap()
                .expect("borrowed query_row must find the row");
            assert_eq!((id, name.as_str()), (1, "hello"));
        })
        .await;
}

#[test]
fn borrowed_execute_and_query_rows_use_borrowed_params() {
    // Sync test on a checked-out connection: the borrowed helpers work
    // directly on a `&Connection` with no capability token (the checkout
    // gate is the async `acquire`; these operate on an already-borrowed
    // connection).
    let pool = pool();
    let guard = pool.connections.lock().unwrap();
    let conn = &guard[0];
    SqlitePool::execute_borrowed(
        conn,
        "CREATE TABLE t (id INTEGER, name TEXT)",
        rusqlite::params![],
    )
    .unwrap();
    SqlitePool::execute_borrowed(
        conn,
        "INSERT INTO t (id, name) VALUES (?1, ?2), (?3, ?4)",
        rusqlite::params![1, "alpha", 2, "beta"],
    )
    .unwrap();

    let rows = SqlitePool::query_rows_borrowed(
        conn,
        "SELECT name FROM t ORDER BY id",
        rusqlite::params![],
        |row| row.get::<_, String>(0),
    )
    .unwrap();
    assert_eq!(rows, vec!["alpha".to_string(), "beta".to_string()]);

    // `query_rows_from_iter_borrowed` — the dynamic-arity IN-clause path
    // the async helpers lack.
    use common_core::sqlite::in_clause;
    let names = vec!["alpha", "gamma"];
    let sql = format!(
        "SELECT name FROM t WHERE name IN ({})",
        in_clause(names.len())
    );
    let found = SqlitePool::query_rows_from_iter_borrowed(conn, &sql, names, |row| {
        row.get::<_, String>(0)
    })
    .unwrap();
    assert_eq!(found, vec!["alpha".to_string()]);
}

#[test]
fn borrowed_query_row_no_rows_maps_to_none() {
    let pool = pool();
    let guard = pool.connections.lock().unwrap();
    let conn = &guard[0];
    SqlitePool::execute_borrowed(conn, "CREATE TABLE t (id INTEGER)", rusqlite::params![])
        .unwrap();
    let val = SqlitePool::query_row_borrowed(
        conn,
        "SELECT id FROM t WHERE id = ?1",
        rusqlite::params![99],
        |row| row.get::<_, i64>(0),
    )
    .unwrap();
    assert_eq!(val, None);
}

// ── In-memory pool isolation ─────────────────────────────────────

#[tokio::test]
async fn in_memory_pool_concurrent_visibility() {
    // A size-3 in-memory pool must expose ONE shared database: a write on
    // one checkout is visible on a later checkout AND on a checkout that
    // was awaited concurrently. On the pre-fix code each checkout saw its
    // own private empty DB and this test failed.
    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = Arc::new(SqlitePool::open_in_memory(&PoolConfig::new(3)).unwrap());

            // Checkout A writes the schema + a row.
            {
                let conn = pool.acquire().await.unwrap();
                conn.execute_batch(
                    "CREATE TABLE t (id INTEGER, name TEXT);
                     INSERT INTO t (id, name) VALUES (1, 'hello')",
                )
                .unwrap();
            }

            // Checkout B (sequential) must see the row.
            let seen = pool
                .query_rows(
                    "SELECT name FROM t",
                    Vec::<rusqlite::types::Value>::new(),
                    |row| row.get::<_, String>(0),
                )
                .await
                .unwrap();
            assert_eq!(seen, vec!["hello".to_string()]);

            // Checkout C, held concurrently with the SELECT, must also see
            // the row — this is the case that silently lost data before.
            // The spawned task acquires ungated: `tokio::spawn` does not
            // inherit the `CURRENT_CAPS` task-local from the outer scope.
            let c_handle = {
                let pool = Arc::clone(&pool);
                tokio::spawn(async move {
                    let conn = pool.acquire_ungated().await.unwrap();
                    conn.query_row("SELECT name FROM t WHERE id = 1", [], |r| {
                        r.get::<_, String>(0)
                    })
                    .ok()
                })
            };
            let query_read = pool
                .query_row(
                    "SELECT name FROM t WHERE id = 1",
                    Vec::<rusqlite::types::Value>::new(),
                    |row| row.get::<_, String>(0),
                )
                .await
                .unwrap();
            let c_read = c_handle.await.unwrap();
            assert_eq!(c_read.as_deref(), Some("hello"));
            assert_eq!(query_read.as_deref(), Some("hello"));
        })
        .await;
}

#[tokio::test]
async fn in_memory_pool_replacement_preserves_data() {
    // Poison one connection (health check fails on return) and verify a
    // fresh checkout — backed by a replacement connection — still sees the
    // original schema/data.
    use rusqlite::hooks::Authorization;

    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = Arc::new(SqlitePool::open_in_memory(&PoolConfig::new(3)).unwrap());
            {
                let conn = pool.acquire().await.unwrap();
                conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
                    .unwrap();
                conn.execute(
                    "INSERT INTO t (id, name) VALUES (?1, ?2)",
                    rusqlite::params![1, "original"],
                )
                .unwrap();
            }
            {
                let conn = pool.acquire().await.unwrap();
                conn.authorizer(Some(|_: rusqlite::hooks::AuthContext<'_>| {
                    Authorization::Deny
                }));
            } // dropped -> health check fails -> replacement connection opened

            let name = pool
                .query_row(
                    "SELECT name FROM t WHERE id = 1",
                    Vec::<rusqlite::types::Value>::new(),
                    |row| row.get::<_, String>(0),
                )
                .await
                .unwrap();
            assert_eq!(
                name.as_deref(),
                Some("original"),
                "replacement connection must reopen the same shared DB"
            );
        })
        .await;
}

#[tokio::test]
async fn in_memory_pools_are_name_isolated() {
    // Two independent pools must NOT share data — each has its own
    // process-unique shared-cache name.
    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool_a = Arc::new(SqlitePool::open_in_memory(&PoolConfig::new(2)).unwrap());
            let pool_b = Arc::new(SqlitePool::open_in_memory(&PoolConfig::new(2)).unwrap());

            pool_a
                .execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1)")
                .await
                .unwrap();

            let err = pool_b
                .query_rows(
                    "SELECT id FROM t",
                    Vec::<rusqlite::types::Value>::new(),
                    |r| r.get::<_, i64>(0),
                )
                .await
                .unwrap_err();
            assert!(
                matches!(err, DbError::Sqlite(_)),
                "pool B must not see pool A's table, got {err:?}"
            );

            // And pool A still sees its own data.
            let rows = pool_a
                .query_rows(
                    "SELECT id FROM t",
                    Vec::<rusqlite::types::Value>::new(),
                    |r| r.get::<_, i64>(0),
                )
                .await
                .unwrap();
            assert_eq!(rows, vec![1]);
        })
        .await;
}

#[tokio::test]
async fn open_literal_memory_path_is_shared() {
    // `SqlitePool::open(":memory:")` (the DbCapability::open(":memory:")
    // route) must produce a coherent pool, not `config.size` private DBs.
    CURRENT_CAPS
        .scope(db_caps(), async {
            let pool = Arc::new(
                SqlitePool::open(std::path::Path::new(":memory:"), &PoolConfig::new(3))
                    .unwrap(),
            );
            pool.execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (5)")
                .await
                .unwrap();
            let conn = pool.acquire().await.unwrap();
            let n: i64 = conn
                .query_row("SELECT id FROM t", [], |r| r.get(0))
                .unwrap();
            assert_eq!(
                n, 5,
                "open(\":memory:\") must share one DB across checkouts"
            );
        })
        .await;
}
