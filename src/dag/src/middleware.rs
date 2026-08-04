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
mod tests {
    use super::*;
    use fluent_wvr_testutil::PassthroughUnit;

    #[test]
    fn test_timing_middleware() {
        let mw = TimingMiddleware::new();
        let wrapped = mw.wrap(Arc::new(PassthroughUnit::new("test")));
        assert!(wrapped.execute(&WorkContext::default()).unwrap().success);
    }

    #[test]
    fn test_timing_middleware_with_histogram() {
        let hist = Arc::new(LatencyHistogram::new());
        let mw = TimingMiddleware::with_histogram(Arc::clone(&hist));
        let wrapped = mw.wrap(Arc::new(PassthroughUnit::new("test")));
        assert!(wrapped.execute(&WorkContext::default()).unwrap().success);
        assert_eq!(hist.count(), 1);
    }

    #[test]
    fn test_middleware_chain() {
        let chain =
            fluent_wvr::wrapper::MiddlewareChain::new().push(Box::new(TimingMiddleware::new()));
        let wrapped = chain.apply(Arc::new(PassthroughUnit::new("chained")));
        assert!(wrapped.execute(&WorkContext::default()).unwrap().success);
    }
}
