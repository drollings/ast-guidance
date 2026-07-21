//! Test `Runtime` implementation with paused time support and seeded PRNG for reproducibility.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use fluent_wvr::Runtime;
use tokio::task::JoinHandle;

/// Test runtime that delegates to a `tokio::runtime::Handle` for deterministic time control.
/// The seeded PRNG provides reproducible non-determinism for tests.
pub struct TestRuntime {
    handle: tokio::runtime::Handle,
    seed: u64,
    rng: Mutex<fastrand::Rng>,
}

impl TestRuntime {
    pub fn new(handle: tokio::runtime::Handle, seed: u64) -> Self {
        Self {
            handle,
            seed,
            rng: Mutex::new(fastrand::Rng::with_seed(seed)),
        }
    }

    /// Returns a reference to the deterministic PRNG for test assertions.
    pub fn rng(&self) -> &Mutex<fastrand::Rng> {
        &self.rng
    }
}

impl Clone for TestRuntime {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            seed: self.seed,
            rng: Mutex::new(fastrand::Rng::with_seed(self.seed)),
        }
    }
}

impl Runtime for TestRuntime {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()> {
        self.handle.spawn(future)
    }

    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(tokio::time::sleep(duration))
    }

    fn now(&self) -> Instant {
        tokio::time::Instant::now().into_std()
    }
}
