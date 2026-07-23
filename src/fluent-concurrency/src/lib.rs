#![forbid(unsafe_code)]

pub mod capability;
pub mod flow;
pub mod io;
pub mod llm_queue;
pub mod pool;
pub mod queue;
pub mod reserve;
pub mod router;
pub mod runtime;
pub mod scope;
pub mod thread_resource;
pub mod zone;

use std::sync::Arc;

/// Returns a new `Arc<dyn Runtime>` wrapping the production Tokio runtime.
/// Use this instead of `crate::tokio_runtime()`.
pub fn tokio_runtime() -> Arc<dyn fluent_wvr::Runtime> {
    Arc::new(runtime::tokio::TokioRuntime)
}

#[cfg(test)]
mod tests;
