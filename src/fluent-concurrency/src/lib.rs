#![forbid(unsafe_code)]

pub mod affinity;
// The `capability` module provides the concrete Fs/Net/Db capability tokens
// and the `io` module the gated engines behind them. Both are behind the `db`
// feature so a non-DB consumer pays nothing for the database layer.
#[cfg(feature = "db")]
pub mod capability;
pub mod flow;
#[cfg(feature = "db")]
pub mod io;
pub mod ladder;
pub mod llm_queue;
pub mod pool;
pub mod queue;
pub mod reserve;
pub mod router;
pub mod runtime;
pub mod scope;
pub mod thread_resource;
pub mod batch;

use std::sync::Arc;

/// Returns a new `Arc<dyn Runtime>` wrapping the production Tokio runtime.
/// Use this instead of `crate::tokio_runtime()`.
pub fn tokio_runtime() -> Arc<dyn fluent_wvr::Runtime> {
    Arc::new(runtime::tokio::TokioRuntime)
}

#[cfg(test)]
mod tests;
