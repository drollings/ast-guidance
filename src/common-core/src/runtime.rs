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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn block_on_from_multi_thread_worker_completes() {
        // The case a plain `handle.block_on` fails on: a bare async task on a
        // multi-thread worker must not panic ("Cannot start a runtime from
        // within a runtime").
        let value = block_on(async { 40 + 2 });
        assert_eq!(value, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn block_on_from_current_thread_runtime_completes() {
        // `block_in_place` panics on a current-thread runtime; the bridge must
        // use a plain `handle.block_on` instead. From the runtime's driver
        // task neither is legal, so exercise the current-thread branch from a
        // blocking thread, where the runtime handle is present and
        // `handle.block_on` is safe.
        let value = tokio::task::spawn_blocking(|| block_on(async { 40 + 2 }))
            .await
            .expect("blocking task must not panic");
        assert_eq!(value, 42);
    }

    #[test]
    fn block_on_with_no_runtime_uses_fallback() {
        // A dedicated std::thread with no tokio runtime active must complete
        // via the process-wide fallback runtime.
        std::thread::spawn(|| {
            let value = block_on(async { 40 + 2 });
            assert_eq!(value, 42);
        })
        .join()
        .expect("thread must not panic");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nested_block_in_place_does_not_panic() {
        // `block_on` from inside an outer `tokio::task::block_in_place` nests
        // without panicking.
        let value = tokio::task::block_in_place(|| block_on(async { 40 + 2 }));
        assert_eq!(value, 42);
    }
}
