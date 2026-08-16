//! Production `Runtime` implementation backed by `tokio::spawn` / `tokio::time::sleep`.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use fluent_wvr::Runtime;
use tokio::task::JoinHandle;

/// Production runtime that delegates to `tokio::spawn`, `tokio::time::sleep`, and `Instant::now()`.
#[derive(Clone)]
pub struct TokioRuntime;

impl TokioRuntime {
    /// Non-boxing spawn for callers who hold `TokioRuntime` concretely.
    ///
    /// The future is inlined into tokio's task allocation instead of boxed
    /// through [`Runtime::spawn`]'s `Pin<Box<dyn Future>>`. Use this on hot
    /// spawn paths; `Arc<dyn Runtime>` callers still go through the trait.
    /// This is the hot-path seam; the trait stays the `Arc<dyn Runtime>`
    /// abstraction.
    #[inline]
    pub fn spawn<F>(&self, future: F) -> JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(future)
    }
}

impl Runtime for TokioRuntime {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()> {
        // Delegate to the inherent method (DRY: one body, two entry points).
        // `Pin<Box<dyn Future<Output = ()> + Send>>` implements `Future`, so
        // the generic inherent `spawn` accepts the boxed future too.
        self.spawn(future)
    }

    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(tokio::time::sleep(duration))
    }

    fn now(&self) -> Instant {
        Instant::now()
    }
}
