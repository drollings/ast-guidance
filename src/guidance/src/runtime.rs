use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use fluent_concurrency::pool::ResultPool;
use fluent_concurrency::runtime::tokio::TokioRuntime;
use fluent_concurrency::zone::{Zone, ZoneConfig};
use fluent_wvr::Runtime;
use fluent_wvr::prelude::*;

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

/// Shared AST generation pool — sized to available cores, backpressure-managed queue.
pub static AST_POOL: LazyLock<Arc<ResultPool<AstGenPayload, GuidanceDoc, SyncEngineError>>> =
    LazyLock::new(|| {
        let workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        Arc::new(ResultPool::new(
            Arc::new(TokioRuntime),
            workers,
            workers * 4,
            |job: AstGenPayload| async move {
                tokio::task::spawn_blocking(move || {
                    thread_local! {
                        static PARSER: RefCell<Option<AstParser>> = const { RefCell::new(None) };
                    }
                    PARSER.with(|cell| {
                        let parser = cell.borrow_mut().take().unwrap_or_else(AstParser::new);
                        let mut engine =
                            SyncEngine::with_parser(job.guidance_dir, job.source_dir, parser);
                        let r = engine.gen_with_config(&job.source_path, &job.config);
                        *cell.borrow_mut() = Some(engine.ast_parser);
                        r
                    })
                })
                .await
                .unwrap_or_else(|e| Err(SyncEngineError::Parse(e.to_string())))
            },
        ))
    });

/// Shared database sync pool — serializes writes to avoid SQLite contention.
pub static DB_POOL: LazyLock<Arc<ResultPool<DbSyncPayload, usize, String>>> =
    LazyLock::new(|| {
        Arc::new(ResultPool::new(
            Arc::new(TokioRuntime),
            1,
            100,
            |job: DbSyncPayload| async move {
                tokio::task::spawn_blocking(move || {
                    let db = GuidanceDb::open(&job.db_path).map_err(|e| e.to_string())?;
                    db.sync_from_dir(&job.json_dir).map_err(|e| e.to_string())
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string()))
            },
        ))
    });

/// Create a Zone that provides structured concurrency, failure containment,
/// and dependency tracking for a batch of AST generation tasks.
///
/// Unlike manual oneshot channel management, a Zone:
/// - Automatically cancels dependent tasks when a prerequisite fails
/// - Enforces a poll budget to prevent executor starvation
/// - Provides a typed event summary (completed/panicked/cancelled)
///
/// # Example
/// ```ignore
/// let mut zone = create_sync_zone(Arc::new(TokioRuntime));
/// zone.register(build_task_a);  // provides "parsed"
/// zone.register(build_task_b);  // depends on "parsed"
/// let summary = zone.await;
/// // If task_a panics, task_b is automatically cancelled
/// ```
pub fn create_sync_zone(runtime: Arc<dyn Runtime>) -> Zone {
    Zone::new_with_config(
        runtime,
        CapabilitySet::default(),
        ZoneConfig { poll_budget: 64 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ast_pool_static_init() {
        let _ = &*AST_POOL;
        let _ = &*DB_POOL;
    }

    #[test]
    fn test_sync_zone_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Zone>();
    }
}
