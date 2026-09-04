use std::sync::Arc;

use common_core::metrics::LatencyHistogram;
use fluent_wvr::prelude::*;
use fluent_wvr::wrapper::Middleware;

pub struct TimingMiddleware {
    histogram: Option<Arc<LatencyHistogram>>,
}

impl TimingMiddleware {
    pub fn new() -> Self {
        Self { histogram: None }
    }

    /// Create a `TimingMiddleware` that records execution durations into
    /// the provided `LatencyHistogram` via `Instrumented::with_metrics`.
    pub fn with_histogram(histogram: Arc<LatencyHistogram>) -> Self {
        Self {
            histogram: Some(histogram),
        }
    }
}

impl Default for TimingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for TimingMiddleware {
    fn wrap(&self, inner: Arc<dyn Component>) -> Arc<dyn Component> {
        match &self.histogram {
            Some(hist) => Arc::new(Instrumented::with_metrics(
                inner,
                "middleware",
                Arc::clone(hist),
            )),
            None => Arc::new(Instrumented::new(inner, "middleware")),
        }
    }
}

#[cfg(test)]
#[path = "../tests/middleware.rs"]
mod tests;
