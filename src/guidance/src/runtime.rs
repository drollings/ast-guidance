use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use fluent_concurrency::pool::{global_pool_config, ResultPool};
use fluent_concurrency::runtime::tokio::TokioRuntime;
use fluent_concurrency::thread_resource::with_tlr;
use fluent_db::wvr::{DbWorkUnit, StoreUnitOp};
use fluent_wvr::{WorkContext, WorkError, WorkOutput, WorkUnit};

use crate::ast_parser::AstParser;
use crate::sync_engine::{GenConfig, SyncEngine, SyncEngineError};
use search_vector::GuidanceDb;

/// A file for AST generation in the result pool.
pub struct AstGenPayload {
    pub source_path: PathBuf,
    pub source_dir: PathBuf,
    pub guidance_dir: PathBuf,
    pub config: GenConfig,
}

/// A database sync job for the result pool.
pub struct DbSyncPayload {
    pub json_dir: PathBuf,
    pub db_path: PathBuf,
}

use fluent_types::GuidanceDoc;

fluent_concurrency::thread_local_resource!(static PARSER: AstParser);

/// Shared AST generation pool — sized to available cores, backpressure-managed queue.
pub static AST_POOL: LazyLock<Arc<ResultPool<AstGenPayload, GuidanceDoc, SyncEngineError>>> =
    LazyLock::new(|| {
        let (workers, queue_cap) = global_pool_config(4, 4);
        Arc::new(ResultPool::new(
            Arc::new(TokioRuntime),
            workers,
            queue_cap,
            |job: AstGenPayload| async move {
                tokio::task::spawn_blocking(move || {
                    with_tlr(&PARSER, |parser| {
                        let mut engine = SyncEngine::with_parser(
                            job.guidance_dir,
                            job.source_dir,
                            std::mem::take(parser),
                        );
                        let r = engine.gen_with_config(&job.source_path, &job.config);
                        *parser = engine.ast_parser;
                        r
                    })
                })
                .await
                .unwrap_or_else(|e| Err(SyncEngineError::Parse(e.to_string())))
            },
        ))
    });

/// Shared `GuidanceDb` cache for the DB sync pool.
///
/// The first `DbSyncPayload` to reach `DB_POOL` opens the database once; later
/// jobs at the same path reuse it, so the connection/schema lifecycle lives in
/// `fluent-db`/`search-vector` rather than being hand-opened per job.
static SHARED_DB: Mutex<Option<(PathBuf, Arc<GuidanceDb>)>> = Mutex::new(None);

fn shared_guidance_db(path: &Path) -> Result<Arc<GuidanceDb>, String> {
    let mut cache = SHARED_DB
        .lock()
        .map_err(|_| "shared guidance db lock poisoned".to_string())?;
    if let Some((cached_path, db)) = cache.as_ref() {
        if cached_path == path {
            return Ok(Arc::clone(db));
        }
    }
    let db = Arc::new(GuidanceDb::open(path).map_err(|e| e.to_string())?);
    *cache = Some((path.to_path_buf(), Arc::clone(&db)));
    Ok(db)
}

/// Build the `DbWorkUnit` that performs a guidance DB sync.
///
/// The op runs on a dedicated blocking thread via `DbWorkUnit::execute`'s
/// offload, so the `ResultPool` worker never blocks on the rusqlite work.
/// The full `sync_from_dir` (SQL writes + HNSW rebuild) lives on
/// `GuidanceDb`, so the generic op form is used over the shared db rather
/// than a bare-connection `store_unit` op.
fn db_sync_work_unit(job: DbSyncPayload) -> DbWorkUnit<StoreUnitOp> {
    DbWorkUnit::builder()
        .name("db.sync")
        .op(Box::new(move |_ctx: &WorkContext| {
            let db = shared_guidance_db(&job.db_path).map_err(WorkError::Execution)?;
            let count = db
                .sync_from_dir(&job.json_dir)
                .map_err(|e| WorkError::Execution(e.to_string()))?;
            WorkOutput::typed("synced guidance db", &count)
                .map_err(|e| WorkError::Execution(e.to_string()))
        }) as StoreUnitOp)
        .build()
}

/// Shared database sync pool — serializes writes to avoid SQLite contention.
pub static DB_POOL: LazyLock<Arc<ResultPool<DbSyncPayload, usize, String>>> = LazyLock::new(|| {
    Arc::new(ResultPool::new(
        Arc::new(TokioRuntime),
        1,
        100,
        |job: DbSyncPayload| async move {
            let unit = db_sync_work_unit(job);
            let output = unit
                .execute(&WorkContext::default())
                .map_err(|e| e.to_string())?;
            output.data_take::<usize>().map_err(|e| e.to_string())
        },
    ))
});

/// Create a `SupervisedBatch` with the standard guidance configuration (structured
/// concurrency, failure containment, and dependency tracking for batches of
/// AST generation tasks).
///
/// Unlike manual oneshot channel management, a SupervisedBatch:
/// - Automatically cancels dependent tasks when a prerequisite fails
/// - Enforces a poll budget to prevent executor starvation
/// - Provides a typed event summary (completed/panicked/cancelled)
///
/// # Example
/// ```ignore
/// let mut batch = SupervisedBatch::new_with_config(
///     Arc::new(TokioRuntime),
///     CapabilitySet::default(),
///     BatchConfig {
///         poll_budget: 64,
///         ..BatchConfig::default()
///     },
/// );
/// batch.register(build_task_a);  // provides "parsed"
/// batch.register(build_task_b);  // depends on "parsed"
/// let summary = batch.await;
/// // If task_a panics, task_b is automatically cancelled
/// ```
#[cfg(test)]
mod tests {
    use super::*;
    use fluent_concurrency::batch::SupervisedBatch;

    #[tokio::test]
    async fn test_ast_pool_static_init() {
        let _ = &*AST_POOL;
        let _ = &*DB_POOL;
    }

    #[test]
    fn test_sync_batch_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SupervisedBatch>();
    }
}
