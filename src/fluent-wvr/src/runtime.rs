use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;

/// Async runtime abstraction. All async primitives in the workspace accept
/// this trait so that production code uses `tokio` and tests can substitute
/// a deterministic runtime.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_wvr::{Runtime, NoopRuntime};
///
/// let rt: Arc<dyn Runtime> = Arc::new(NoopRuntime);
/// rt.spawn(Box::pin(async { /* background work */ }));
/// ```
pub trait Runtime: Send + Sync + 'static {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()>;
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
    fn now(&self) -> Instant;
}

/// A no-op runtime for contexts where `spawn` and `sleep` are never called.
///
/// `spawn` logs a warning and returns a dummy `JoinHandle`. If called outside
/// a tokio runtime, it panics with a clear message directing the caller to
/// provide a real `Runtime`. `sleep` returns immediately. This runtime is
/// intended for testing or initialization code that doesn't actually need
/// async execution.
pub struct NoopRuntime;

impl Runtime for NoopRuntime {
    fn spawn(&self, _future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()> {
        assert!(
            tokio::runtime::Handle::try_current().is_ok(),
            "NoopRuntime::spawn called outside a tokio runtime. \
             Either supply a real Runtime (e.g. via WorkContext with rt: tokio_runtime()) \
             or use NoopRuntime only for dry-run / init code paths."
        );
        tokio::spawn(async {})
    }

    fn sleep(&self, _duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async {})
    }

    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_runtime_spawn_panics_outside_tokio_with_clear_message() {
        let rt = NoopRuntime;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[allow(clippy::let_underscore_future)]
            let _ = rt.spawn(Box::pin(async {}));
        }));
        assert!(result.is_err(), "should panic outside tokio runtime");
        let payload = result.unwrap_err();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            panic!("unexpected panic payload type");
        };
        assert!(
            msg.contains("NoopRuntime::spawn called outside a tokio runtime"),
            "panic message should be clear, got: {msg}"
        );
        assert!(
            msg.contains("supply a real Runtime"),
            "panic message should suggest fix, got: {msg}"
        );
    }

    #[tokio::test]
    async fn noop_runtime_sleep_returns_immediately() {
        let rt = NoopRuntime;
        let start = std::time::Instant::now();
        rt.sleep(Duration::from_secs(3600)).await;
        // A one-hour sleep must return without waiting (the elapsed time is a
        // tiny fraction of the requested duration).
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "NoopRuntime sleep must not block"
        );
    }

    #[tokio::test]
    async fn noop_runtime_spawn_inside_tokio_completes() {
        let rt = NoopRuntime;
        let handle = rt.spawn(Box::pin(async {}));
        handle.await.expect("spawned future completes");
    }

    #[test]
    fn noop_runtime_now_returns_instant() {
        // `now` just answers the monotonic clock; two calls move forward.
        let rt = NoopRuntime;
        let a = rt.now();
        let b = rt.now();
        assert!(b >= a);
    }
}
