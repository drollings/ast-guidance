use std::sync::Arc;
use std::time::Duration;

use common_core::metrics::LatencyHistogram;
use fluent_wvr::wrapper::Instrumented;

use super::*;
use crate::tests::common::{db_caps, in_memory_pool, store_with_t};

fn store() -> Arc<SqliteStore> {
    store_with_t()
}

#[test]
fn store_unit_select_returns_ok_with_data() {
    let unit = store_unit(store(), "db.select", |conn| {
        let name: String =
            conn.query_row("SELECT name FROM t WHERE id = 1", [], |row| row.get(0))?;
        Ok(WorkOutput::ok_with_data(
            "selected",
            serde_json::json!({ "name": name }),
        ))
    });
    let out = unit
        .execute(&WorkContext::default())
        .expect("execute must succeed");
    assert!(out.success);
    assert_eq!(out.data["name"], "hello");
    assert_eq!(unit.name(), "db.select");
}

#[test]
fn store_unit_failing_sql_maps_to_execution_error() {
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
    let unit = store_unit(store, "db.bad", |conn| {
        let _: i64 = conn.query_row("SELECT missing FROM t", [], |row| row.get(0))?;
        Ok(WorkOutput::ok("unreachable"))
    });
    let err = unit
        .execute(&WorkContext::default())
        .expect_err("failing SQL must produce WorkError::Execution");
    assert!(matches!(err, WorkError::Execution(_)));
}

#[test]
fn store_unit_dry_run_still_runs_op() {
    // DbWorkUnit runs its op unconditionally (the store op is the work);
    // dry-run short-circuits are the caller's concern.
    let unit = store_unit(store(), "db.dryrun", |conn| {
        let name: String =
            conn.query_row("SELECT name FROM t WHERE id = 1", [], |row| row.get(0))?;
        Ok(WorkOutput::ok_with_data(
            "selected",
            serde_json::json!(name),
        ))
    });
    let ctx = WorkContext {
        dry_run: true,
        ..WorkContext::default()
    };
    let out = unit.execute(&ctx).expect("execute");
    assert_eq!(out.data, serde_json::json!("hello"));
}

#[test]
fn instrumented_with_metrics_records_timing() {
    let hist = Arc::new(LatencyHistogram::new());
    let unit = Instrumented::with_metrics(
        store_unit(store(), "db.metrics", |conn| {
            conn.execute("INSERT INTO t (id, name) VALUES (2, 'world')", [])?;
            Ok(WorkOutput::ok("inserted"))
        }),
        "db.metrics",
        Arc::clone(&hist),
    );
    let out = unit.execute(&WorkContext::default()).expect("execute");
    assert!(out.success);
    assert_eq!(hist.count(), 1, "Instrumented must record one observation");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_db_op_does_not_starve_executor() {
    // A genuinely slow DbWorkUnit op runs on a dedicated blocking thread
    // (`block_in_place`), so the async executor keeps making progress while
    // the DB work is in flight.
    //
    // Note: a SupervisedBatch's `tokio::time::timeout` cannot *preempt* a single
    // blocking `execute` (block_in_place parks the worker until the op
    // returns); the SupervisedBatch applies its wall-clock budget across the retry
    // loop instead (see `fluent-concurrency` tests/m2.rs
    // `test_batch_real_timeout`). The guarantee `DbWorkUnit` provides is
    // that other tasks are never starved by the DB op.
    let store = Arc::new(SqliteStore::open_in_memory().unwrap());
    store.init_schema("CREATE TABLE t (id INTEGER)").unwrap();
    let unit = store_unit(store, "db.slow", |_conn| {
        std::thread::sleep(Duration::from_millis(300));
        Ok(WorkOutput::ok("slow done"))
    });

    let heartbeat = tokio::spawn(async {
        let start = std::time::Instant::now();
        let mut ticks = 0u32;
        while start.elapsed() < Duration::from_millis(400) {
            tokio::time::sleep(Duration::from_millis(10)).await;
            ticks += 1;
        }
        ticks
    });

    let out = unit
        .execute(&WorkContext::default())
        .expect("slow op completes on the blocking thread");
    assert!(out.success);

    let ticks = heartbeat.await.expect("heartbeat task must not be starved");
    assert!(
        ticks >= 5,
        "executor was starved by the blocking op: only {ticks} heartbeats"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_op_executes_cleanly_under_batch_style_timeout() {
    // A fast DbWorkUnit runs fine under a simulated `SupervisedBatch` supervisor
    // wrapper (`tokio::time::timeout`), surfacing its result before the
    // budget elapses rather than timing out spuriously.
    let unit = store_unit(store(), "db.fast", |conn| {
        let name: String =
            conn.query_row("SELECT name FROM t WHERE id = 1", [], |row| row.get(0))?;
        Ok(WorkOutput::ok_with_data(
            "selected",
            serde_json::json!(name),
        ))
    });
    let result = tokio::time::timeout(Duration::from_millis(500), async {
        unit.execute(&WorkContext::default())
    })
    .await;
    let out = result
        .expect("fast op must complete within the supervision budget")
        .expect("execute");
    assert_eq!(out.data, serde_json::json!("hello"));
}

#[test]
fn store_unit_impl_component_surface() {
    use fluent_wvr::Component;
    let unit = store_unit(store(), "db.comp", |_conn| Ok(WorkOutput::ok("ok")));
    let erased: Arc<dyn Component> = Arc::new(unit);
    assert_eq!(erased.name(), "db.comp");
    assert!(erased.describe().get("name").is_some());
    assert_eq!(erased.field_names().len(), 0);
}

#[test]
fn db_store_impl_for_pool() {
    let pool = in_memory_pool();
    // `acquire` is capability-gated, so the pool-backed `DbStore`
    // path must run under a `DbCapability` scope (the sync `sync_scope`
    // propagates into the fallback-runtime `block_on`).
    let n: i64 = CURRENT_CAPS.sync_scope(db_caps(), || {
        pool.with_conn_blocking(|conn| {
            Ok(conn.query_row("SELECT 41 + 1", [], |row| row.get(0))?)
        })
        .expect("pool with_conn_blocking")
    });
    assert_eq!(n, 42);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pool_with_conn_blocking_from_bare_multi_thread_task_succeeds() {
    // The pool-backed `DbStore` path routes through the unified
    // `common_core::runtime::block_on`, which wraps a multi-thread
    // worker in `block_in_place`.  Calling `with_conn_blocking` from a
    // bare async task on a multi-thread runtime (NOT wrapped in
    // `block_in_place` by the caller) must now succeed — the old plain
    // `handle.block_on` copy panicked here with "Cannot start a runtime
    // from within a runtime".
    let pool = in_memory_pool();
    let n: i64 = CURRENT_CAPS
        .scope(db_caps(), async {
            pool.with_conn_blocking(|conn| {
                Ok(conn.query_row("SELECT 41 + 1", [], |row| row.get(0))?)
            })
            .expect("pool with_conn_blocking from a bare multi-thread task")
        })
        .await;
    assert_eq!(n, 42);
}

#[tokio::test(flavor = "current_thread")]
async fn current_thread_runtime_supports_pool_backed_op() {
    // Nit 2: on a current-thread runtime there is no `block_in_place`, so a
    // pool-backed op used to run inline and `block_on` inside the runtime
    // worker panics ("cannot block the current thread from within a
    // runtime"). The scoped-OS-thread offload gives the op a fresh thread
    // where the sync→async bridge falls back to `fallback_runtime`.
    let pool = in_memory_pool();
    let unit = DbWorkUnit::builder()
        .name("db.poolct")
        .op(Box::new(move |_ctx: &WorkContext| {
            let n: i64 = pool
                .with_conn_blocking(|conn| {
                    Ok(conn.query_row("SELECT 41 + 1", [], |row| row.get(0))?)
                })
                .map_err(|e| WorkError::Execution(e.to_string()))?;
            WorkOutput::typed("got", &n).map_err(|e| WorkError::Execution(e.to_string()))
        }) as StoreUnitOp)
        .build();
    let ctx = WorkContext {
        caps: db_caps(),
        ..WorkContext::default()
    };
    let out = unit
        .execute(&ctx)
        .expect("pool-backed op must complete on the scoped thread");
    assert!(out.success);
    assert_eq!(out.data, serde_json::json!(42));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pool_backed_db_work_unit_scopes_caps_on_multi_thread_path() {
    // A pool-backed `DbWorkUnit` op calls `with_conn_blocking` →
    // gated `acquire`, which reads the `CURRENT_CAPS` task-local.
    // `execute` must re-scope `ctx.caps` around the `block_in_place`
    // offload (the multi-thread path), so the gate passes when a
    // `DbCapability` is present and fails (wrapped in
    // `WorkError::Execution`) when it is not.
    let pool = in_memory_pool();
    let make_unit = || {
        let pool = Arc::clone(&pool);
        DbWorkUnit::builder()
            .name("db.poolmt")
            .op(Box::new(move |_ctx: &WorkContext| {
                let n: i64 = pool
                    .with_conn_blocking(|conn| {
                        Ok(conn.query_row("SELECT 41 + 1", [], |row| row.get(0))?)
                    })
                    .map_err(|e| WorkError::Execution(e.to_string()))?;
                WorkOutput::typed("got", &n).map_err(|e| WorkError::Execution(e.to_string()))
            }) as StoreUnitOp)
            .build()
    };

    let with_caps = WorkContext {
        caps: db_caps(),
        ..WorkContext::default()
    };
    let out = make_unit()
        .execute(&with_caps)
        .expect("pool-backed op must succeed when a DbCapability is scoped");
    assert!(out.success);
    assert_eq!(out.data, serde_json::json!(42));

    let without_caps = WorkContext::default();
    let err = make_unit()
        .execute(&without_caps)
        .expect_err("pool-backed op must be denied without a DbCapability");
    assert!(
        matches!(&err, WorkError::Execution(msg) if msg.contains("permission denied")),
        "expected wrapped PermissionDenied, got {err:?}"
    );
}
