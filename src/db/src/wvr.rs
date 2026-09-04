//! Database `Component`/`WorkUnit` adapters.
//!
//! `DbWorkUnit` wraps a synchronous store operation so it can run under a
//! `SupervisedBatch` supervisor (or any `WorkUnit` orchestrator) without violating the
//! WorkUnit purity contract (`fluent-wvr` SKILL §10): `execute` is
//! synchronous and returns promptly. On a multi-threaded runtime worker (the
//! canonical `SupervisedBatch`/`ResultPool` context) the blocking rusqlite work is
//! offloaded via `tokio::task::block_in_place`; on a current-thread runtime
//! (or with no runtime active) it runs on a dedicated scoped OS thread via
//! `std::thread::scope`, so a slow op still cannot block the caller's single
//! thread.
//!
//! `DbStore` is the blocking-connection abstraction: `SqliteStore` runs the
//! closure against its `Mutex<Connection>`; `Arc<SqlitePool>` acquires a
//! pooled connection via the unified sync→async bridge
//! (`common_core::runtime::block_on`, the workspace's single canonical
//! bridge.

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
        common_core::runtime::block_on(async move {
            let conn = pool.acquire().await?;
            tokio::task::spawn_blocking(move || f(&conn))
                .await
                .map_err(|e| DbError::Other(format!("blocking database task failed: {e}")))?
        })
    }
}

/// A `Component`/`WorkUnit` that runs a synchronous database operation on a
/// dedicated blocking thread.
///
/// The op receives the `WorkContext` and returns a `WorkOutput`. `execute`
/// offloads it via `tokio::task::block_in_place` when on a multi-threaded
/// runtime worker, so the async executor is never starved — this is what makes
/// DB work safe to run under `SupervisedBatch` supervision (timeout/retry/cancellation)
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
        // (the canonical SupervisedBatch/ResultPool context) so the executor isn't
        // starved. On a current-thread runtime (or with no runtime active), run
        // the op on a dedicated scoped OS thread instead of inline, so a
        // genuinely slow op still cannot block the caller's single thread.
        //
        // On **both** paths `ctx.caps` is re-scoped into the worker via
        // `CURRENT_CAPS.sync_scope`, so a pool-backed op (whose
        // `with_conn_blocking` → gated `acquire` reads the task-local) is
        // capability-correct on multi-thread and current-thread runtimes alike.
        let caps = ctx.caps.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                CURRENT_CAPS.sync_scope(caps, || tokio::task::block_in_place(|| (self.op)(ctx)))
            }
            _ => {
                // Run the op on a dedicated scoped OS thread. `thread::scope`
                // joins it before returning; a thread panic surfaces as the
                // join's `Err` (the op's own `WorkError` propagates as-is).
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

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/wvr.rs"]
mod tests;
