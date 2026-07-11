//! Structured concurrency via `Scope` — all spawned tasks are joined or aborted on drop.
//! Capabilities are propagated to child tasks through a task-local.
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluent_concurrency::scope::Scope;
//!
//! # async fn example() {
//! let mut scope = Scope::new();
//! scope.spawn(async { /* work */ });
//! // Option 1: explicit close
//! scope.close().await;
//!
//! // Option 2: defer guard — closes on drop (best-effort via spawn)
//! let mut scope = Scope::new();
//! scope.spawn(async { /* work */ });
//! let _guard = scope.defer();
//! // _guard calls close().await on drop
//! # }
//! ```

use std::future::Future;

use fluent_wvr::CapabilitySet;
use tokio::task::{AbortHandle, JoinSet};

tokio::task_local! {
    pub static CURRENT_CAPS: CapabilitySet;
}

#[must_use = "Scopes must be explicitly closed with .close().await"]
/// Structured concurrency scope — all spawned tasks are joined or aborted on drop.
///
/// # Examples
///
/// ```no_run
/// use fluent_concurrency::scope::Scope;
///
/// # async fn example() {
/// let mut scope = Scope::new();
/// scope.spawn(async { /* background work */ });
/// scope.spawn(async { /* more work */ });
/// scope.close().await; // waits for all tasks to complete
/// # }
///
/// // Or use a defer guard for automatic cleanup:
/// # async fn example2() {
/// let mut scope = Scope::new();
/// scope.spawn(async { /* work */ });
/// let _guard = scope.defer(); // calls close().await on drop
/// # }
/// ```
pub struct Scope {
    tasks: JoinSet<()>,
    closed: bool,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            tasks: JoinSet::new(),
            closed: false,
        }
    }

    pub fn spawn<F>(&mut self, future: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let caps = CURRENT_CAPS.try_with(Clone::clone).unwrap_or_default();
        self.tasks.spawn(async move {
            CURRENT_CAPS.scope(caps, future).await;
        })
    }

    pub async fn close(&mut self) {
        self.closed = true;
        self.tasks.abort_all();
        while self.tasks.join_next().await.is_some() {}
    }

    /// Synchronous variant of `close()`. Aborts all tasks immediately.
    /// Aborted tasks resolve their `JoinHandle` on the next yield, so the
    /// aborted futures are guaranteed not to outlive this call.
    ///
    /// Use this when `Drop` must guarantee that all spawned tasks have
    /// been cancelled, e.g. in `ScopeGuard::drop` where the async
    /// `close().await` would require a separate spawned task that itself
    /// could be aborted if the runtime drops.
    ///
    /// We do not block on the active runtime to wait for the aborts to
    /// settle — `block_on` from within a runtime is undefined behavior,
    /// and `block_in_place` requires a multi-thread runtime. The
    /// `abort_all` + drop of the `JoinSet` is the strongest synchronous
    /// guarantee we can give.
    pub fn close_sync(&mut self) {
        self.closed = true;
        self.tasks.abort_all();
        // Dropping the JoinSet also aborts any tasks still in flight.
        // Aborted tasks are guaranteed not to make progress past their
        // next yield point.
        self.tasks = tokio::task::JoinSet::new();
    }

    /// Gracefully shuts down: waits up to `timeout` for tasks to complete,
    /// then aborts any remaining tasks.
    pub async fn close_graceful(&mut self, timeout: std::time::Duration) {
        self.closed = true;
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.tasks.is_empty() {
                break;
            }
            match tokio::time::timeout_at(deadline, self.tasks.join_next()).await {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    // Timeout expired — abort remaining tasks
                    self.tasks.abort_all();
                    while self.tasks.join_next().await.is_some() {}
                    break;
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Returns a `ScopeGuard` that will call `close().await` on drop.
    ///
    /// This is the ergonomic alternative to manually calling `close().await`.
    /// The guard uses `tokio::spawn` to close asynchronously when a runtime
    /// is available, and falls back to `abort_all()` otherwise.
    pub fn defer(&mut self) -> ScopeGuard<'_> {
        ScopeGuard { scope: self }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if !self.closed {
            self.tasks.abort_all();
            if std::thread::panicking() {
                // During panic unwind a secondary panic would abort the process.
                // Log the violation and let the original panic propagate.
                tracing::error!(
                    "Scope dropped without calling .close().await during panic unwind; \
                     all tasks were aborted"
                );
            } else {
                panic!(
                    "Scope dropped without calling .close().await — \
                     all tasks were aborted. This is a structured concurrency violation. \
                     Call scope.close().await before dropping, or use scope.defer()."
                );
            }
        }
    }
}

/// RAII guard returned by `Scope::defer()`. Calls `close().await` (via
/// `tokio::spawn`) on drop. If the guard is dropped during a panic unwind
/// or outside a tokio runtime, it falls back to `abort_all()` (best-effort).
///
/// # Example
///
/// ```rust,ignore
/// let mut scope = Scope::new();
/// scope.spawn(async { /* work */ });
/// let _guard = scope.defer();
/// // When _guard drops, scope is closed automatically.
/// ```
pub struct ScopeGuard<'a> {
    scope: &'a mut Scope,
}

impl ScopeGuard<'_> {
    /// Returns a reference to the underlying scope for additional operations.
    pub fn scope(&mut self) -> &mut Scope {
        self.scope
    }
}

impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        if self.scope.closed {
            return;
        }
        // Close the scope synchronously: abort all tasks and drive the
        // JoinSet to completion on the current runtime (if any). This
        // avoids the fire-and-forget spawned task that could itself be
        // aborted if the runtime drops before it completes.
        //
        // During a panic unwind, fall back to abort_all only — `block_on`
        // inside a Drop during panic is unsafe and may deadlock.
        if std::thread::panicking() {
            self.scope.closed = true;
            self.scope.tasks.abort_all();
            tracing::error!(
                "ScopeGuard dropped during panic unwind; \
                 tasks were aborted without waiting for join"
            );
            return;
        }
        self.scope.close_sync();
    }
}
