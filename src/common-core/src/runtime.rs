//! Sync→async runtime bridge: run a future to completion from a possibly-sync
//! context.
//!
//! This is the workspace's single canonical `block_on`. `fluent-db` and
//! `fluent-llm` delegate their own `block_on` entry points here (module paths
//! preserved for callers), so the sync→async bridging semantics live in
//! exactly one zero-domain place instead of duplicated copies that diverge.
//!
//! Zero-domain rationale: bridging between synchronous and asynchronous
//! execution is generic tokio plumbing — it knows nothing about nodes, LLMs,
//! or databases — so it belongs in `common-core`, the workspace's only
//! zero-domain crate.

use std::sync::OnceLock;

/// Run a future to completion synchronously from a possibly-sync context.
///
/// - Inside a **multi-threaded** tokio runtime worker: drive via
///   `tokio::task::block_in_place(|| handle.block_on(fut))` so the worker is
///   not starved and a bare `handle.block_on` never panics.
/// - Inside a **current-thread** runtime (where `block_in_place` panics):
///   `handle.block_on(fut)`.
/// - With **no active runtime** (e.g. a `spawn_blocking` thread, a scoped OS
///   thread, or a non-tokio thread): the process-wide fallback runtime.
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                handle.block_on(fut)
            } else {
                tokio::task::block_in_place(move || handle.block_on(fut))
            }
        }
        Err(_) => fallback_runtime().block_on(fut),
    }
}

/// Process-wide fallback runtime, used only when no tokio runtime is active on
/// the calling thread (plain `fn main()`, scoped OS threads, blocking threads).
fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build common-core fallback runtime")
    })
}

