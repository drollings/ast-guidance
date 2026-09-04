#![forbid(unsafe_code)]

pub mod affinity;
// The `capability` module provides the concrete Fs/Net/Db capability tokens
// and the `io` module the gated engines behind them. Both are behind the `db`
// feature so a non-DB consumer pays nothing for the database layer.
#[cfg(feature = "db")]
pub mod capability;
pub mod credit_pool;
pub mod feed_worker;
pub mod flow;
#[cfg(feature = "db")]
pub mod io;
pub mod ladder;
// NOTE (ROADMAP_20260903_LLM M11): `llm_queue` (LLM protocol types +
// `LlmRequestQueue`, owned by `fluent_llm::protocol` since M9) lived here
// through M10 as deprecated byte-identical shims; M11 deleted the module.
// The generic executor (`pool::ResultPool`, `stream::StreamAbort`,
// `runtime::TestRuntime`) stays and is composed by the protocol owner.
pub mod pool;
pub mod queue;
pub mod reserve;
pub mod router;
pub mod runtime;
pub mod scope;
pub mod stream;
pub mod thread_resource;
pub mod batch;

use std::sync::Arc;

/// Returns a new `Arc<dyn Runtime>` wrapping the production Tokio runtime.
/// Use this instead of `crate::tokio_runtime()`.
pub fn tokio_runtime() -> Arc<dyn fluent_wvr::Runtime> {
    Arc::new(runtime::tokio::TokioRuntime)
}
#[cfg(test)]
#[path = "../tests/mod.rs"]
mod e2e_tests;

#[cfg(test)]
#[path = "../tests/affinity_calibration.rs"]
mod affinity_calibration_tests;

