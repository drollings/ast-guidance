//! Database `Component`/`WorkUnit` adapters (D9).
//!
//! `DbWorkUnit` wraps a synchronous store operation so it can run under a
//! `Zone` supervisor (or any `WorkUnit` orchestrator) without violating the
//! WorkUnit purity contract (`fluent-wvr` SKILL §10): `execute` is
//! synchronous and returns promptly. On a multi-threaded runtime worker (the
//! canonical `Zone`/`ResultPool` context) the blocking rusqlite work is
//! offloaded via `tokio::task::block_in_place`; on a current-thread runtime
//! (or with no runtime active) it runs on a dedicated scoped OS thread via
//! `std::thread::scope`, so a slow op still cannot block the caller's single
//! thread.
//!
//! `DbStore` is the blocking-connection abstraction: `SqliteStore` runs the
//! closure against its `Mutex<Connection>`; `Arc<SqlitePool>` acquires a
//! pooled connection via the sync→async bridge (`Handle::block_on`, safe on
//! `spawn_blocking`/`block_in_place` threads). `fluent-db` owns this tokio
//! composition directly — it must not import `fluent-concurrency` (acyclic,
//! D2), so the bridge mirrors `fluent_llm::client::block_on` rather than
//! reusing it.

use std::sync::Arc;

use internment::ArcIntern;

use fluent_wvr::capability::CURRENT_CAPS;
use fluent_wvr::{impl_component, impl_fieldless, Describable};
use fluent_wvr::{WorkContext, WorkError, WorkOutput, WorkUnit};

use crate::error::DbError;
use crate::pool::SqlitePool;
use crate::store::SqliteStore;

/// A store that can execute a blocking database operation.
///
/// The operation runs against a checked-out connection (single-connection
/// mutex for `SqliteStore`, pooled checkout for `Arc<SqlitePool>`). The
/// caller — typically `DbWorkUnit::execute` on a blocking thread — decides
/// where the closure runs; this trait only supplies the connection.
pub trait DbStore: Send + Sync {
    fn with_conn_blocking<R>(
        &self,
        f: impl FnOnce(&rusqlite::Connection) -> Result<R, DbError> + Send + 'static,
    ) -> Result<R, DbError>
    where
        R: Send + 'static;
}

impl DbStore for SqliteStore {
    fn with_conn_blocking<R>(
        &self,
        f: impl FnOnce(&rusqlite::Connection) -> Result<R, DbError> + Send + 'static,
    ) -> Result<R, DbError>
    where
        R: Send + 'static,
    {
        self.with_conn(f)
    }
}

impl DbStore for Arc<SqlitePool> {
    fn with_conn_blocking<R>(
        &self,
        f: impl FnOnce(&rusqlite::Connection) -> Result<R, DbError> + Send + 'static,
    ) -> Result<R, DbError>
    where
        R: Send + 'static,
    {
        let pool = Arc::clone(self);
        block_on(async move {
            let conn = pool.acquire().await?;
            tokio::task::spawn_blocking(move || f(&conn))
                .await
                .map_err(|e| DbError::Other(format!("blocking database task failed: {e}")))?
        })
    }
}

/// Drive an async `acquire` from a blocking (or otherwise sync) context.
///
/// Mirrors `fluent_llm::client::block_on` (D9: `fluent-db` owns its tokio
/// composition). `Handle::block_on` is safe on `spawn_blocking`/`block_in_place`
/// threads and on the process fallback runtime; it must only be called from a
/// runtime worker task when wrapped in `block_in_place` (which the
/// `DbWorkUnit::execute` path guarantees).
fn block_on<F>(fut: F) -> F::Output
where
    F: std::future::Future + Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(fut),
        Err(_) => fallback_runtime().block_on(fut),
    }
}

fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build fluent-db fallback runtime")
    })
}

/// A `Component`/`WorkUnit` that runs a synchronous database operation on a
/// dedicated blocking thread.
///
/// The op receives the `WorkContext` and returns a `WorkOutput`. `execute`
/// offloads it via `tokio::task::block_in_place` when on a multi-threaded
/// runtime worker, so the async executor is never starved — this is what makes
/// DB work safe to run under `Zone` supervision (timeout/retry/cancellation)
/// and inside `ResultPool` handlers without violating the WorkUnit purity
/// contract.
///
/// Construct with [`DbWorkUnit::builder`], or use the concrete
/// [`store_unit`] factory for store operations over an `Arc<SqliteStore>`.
#[derive(bon::Builder)]
pub struct DbWorkUnit<F> {
    /// The unit's `WorkUnit::name()`.
    #[builder(into)]
    name: ArcIntern<str>,
    /// The blocking operation. Runs on a dedicated blocking thread.
    op: F,
    /// Asset names this unit depends on (`WorkUnit::depends`).
    #[builder(default)]
    depends: Vec<ArcIntern<str>>,
    /// Asset names this unit provides (`WorkUnit::provides`).
    #[builder(default)]
    provides: Vec<ArcIntern<str>>,
    /// Per-unit timeout in ms (`WorkUnit::default_timeout_ms`).
    #[builder(default = 30_000)]
    timeout_ms: u64,
}

impl<F> WorkUnit for DbWorkUnit<F>
where
    F: Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn default_timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        // Offload to a blocking thread when on a multi-threaded runtime worker
        // (the canonical Zone/ResultPool context) so the executor isn't
        // starved. On a current-thread runtime (or with no runtime active), run
        // the op on a dedicated scoped OS thread instead of inline, so a
        // genuinely slow op still cannot block the caller's single thread.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| (self.op)(ctx))
            }
            _ => {
                // Run the op on a dedicated scoped OS thread. `thread::scope`
                // joins it before returning; a thread panic surfaces as the
                // join's `Err` (the op's own `WorkError` propagates as-is).
                // `ctx.caps` is re-scoped into the thread so capability-gated
                // pool helpers work off the executor thread too.
                let caps = ctx.caps.clone();
                match std::thread::scope(|scope| {
                    scope
                        .spawn(|| CURRENT_CAPS.sync_scope(caps, || (self.op)(ctx)))
                        .join()
                }) {
                    Ok(Ok(out)) => Ok(out),
                    Ok(Err(we)) => Err(we),
                    Err(_) => Err(WorkError::Execution("db work unit thread panicked".into())),
                }
            }
        }
    }
}

impl<F> Describable for DbWorkUnit<F> {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "name": &self.name,
            "depends": &self.depends,
            "provides": &self.provides,
            "timeout_ms": self.timeout_ms,
        })
    }
}

impl_fieldless!(generic (F: Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync + 'static) for DbWorkUnit<F>);
impl_component!(generic (F: Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync + 'static) for DbWorkUnit<F>);

/// The boxed op signature of a [`DbWorkUnit`] produced by [`store_unit`].
pub type StoreUnitOp = Box<dyn Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync>;

/// A shared, boxed store operation closure (connection → `WorkOutput`).
type StoreOp = Arc<dyn Fn(&rusqlite::Connection) -> Result<WorkOutput, DbError> + Send + Sync>;

/// Build a `DbWorkUnit` that runs a store operation over a shared
/// `Arc<SqliteStore>`.
///
/// The `op` closure runs against the store's connection on a blocking thread
/// (via `DbWorkUnit::execute`'s offload); a `DbError` is mapped to
/// `WorkError::Execution`. `depends`/`provides` default to empty and the
/// timeout to `30_000` ms.
pub fn store_unit(
    store: Arc<SqliteStore>,
    name: &str,
    op: impl Fn(&rusqlite::Connection) -> Result<WorkOutput, DbError> + Send + Sync + 'static,
) -> DbWorkUnit<StoreUnitOp> {
    let name = name.to_string();
    let op: StoreOp = Arc::new(op);
    DbWorkUnit::builder()
        .name(name.clone())
        .op(Box::new(move |_ctx: &WorkContext| {
            let op = Arc::clone(&op);
            store
                .with_conn_blocking(move |conn| op(conn))
                .map_err(|e| WorkError::Execution(format!("{name}: {e}")))
        }) as StoreUnitOp)
        .build()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use common_core::metrics::LatencyHistogram;
    use fluent_wvr::wrapper::Instrumented;

    use super::*;

    fn store() -> Arc<SqliteStore> {
        let store = Arc::new(SqliteStore::open_in_memory().unwrap());
        store
            .init_schema("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        store
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![1, "hello"],
            )
            .unwrap();
        store
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
        // the DB work is in flight — the property the roadmap's "without
        // blocking the executor" guarantee requires.
        //
        // Note: a Zone's `tokio::time::timeout` cannot *preempt* a single
        // blocking `execute` (block_in_place parks the worker until the op
        // returns); the Zone applies its wall-clock budget across the retry
        // loop instead (see `fluent-concurrency` tests/m2.rs
        // `test_zone_real_timeout`). The guarantee `DbWorkUnit` provides is
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
    async fn fast_op_executes_cleanly_under_zone_style_timeout() {
        // A fast DbWorkUnit runs fine under a simulated `Zone` supervisor
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
        let pool =
            Arc::new(SqlitePool::open_in_memory(&crate::pool::PoolConfig::default()).unwrap());
        let n: i64 = pool
            .with_conn_blocking(|conn| Ok(conn.query_row("SELECT 41 + 1", [], |row| row.get(0))?))
            .expect("pool with_conn_blocking");
        assert_eq!(n, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_thread_runtime_supports_pool_backed_op() {
        // Nit 2: on a current-thread runtime there is no `block_in_place`, so a
        // pool-backed op used to run inline and `block_on` inside the runtime
        // worker panics ("cannot block the current thread from within a
        // runtime"). The scoped-OS-thread offload gives the op a fresh thread
        // where the sync→async bridge falls back to `fallback_runtime`.
        let pool =
            Arc::new(SqlitePool::open_in_memory(&crate::pool::PoolConfig::default()).unwrap());
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
        let out = unit
            .execute(&WorkContext::default())
            .expect("pool-backed op must complete on the scoped thread");
        assert!(out.success);
        assert_eq!(out.data, serde_json::json!(42));
    }
}
