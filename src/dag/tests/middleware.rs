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
