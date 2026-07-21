use super::*;

#[tokio::test(start_paused = true)]
async fn test_tokio_runtime_spawn() {
    tokio::time::resume();
    let runtime = TokioRuntime;
    let handle = runtime.spawn(Box::pin(async {}));
    handle.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn test_tokio_runtime_sleep_now() {
    tokio::time::resume();
    let runtime = TokioRuntime;
    let before = runtime.now();
    runtime.sleep(Duration::from_millis(1)).await;
    let after = runtime.now();
    assert!(after >= before);
}

#[tokio::test(start_paused = true)]
async fn test_test_runtime_with_paused_time() {
    tokio::time::resume();
    let handle = tokio::runtime::Handle::current();
    let test_runtime = TestRuntime::new(handle, 42);
    let before = test_runtime.now();
    test_runtime.sleep(Duration::from_millis(5)).await;
    let after = test_runtime.now();
    assert!(after >= before);
}

#[tokio::test(start_paused = true)]
async fn test_test_runtime_spawn() {
    tokio::time::resume();
    let handle = tokio::runtime::Handle::current();
    let test_runtime = TestRuntime::new(handle, 42);
    let join = test_runtime.spawn(Box::pin(async {}));
    join.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn test_test_runtime_deterministic_rng() {
    tokio::time::resume();
    let handle = tokio::runtime::Handle::current();
    let rt1 = TestRuntime::new(handle.clone(), 12345);
    let rt2 = TestRuntime::new(handle, 12345);
    let a = rt1.rng().lock().unwrap().u32(..);
    let b = rt2.rng().lock().unwrap().u32(..);
    assert_eq!(a, b, "same seed must produce same output");
}

#[test]
fn test_capability_set_insert_get() {
    let caps = CapabilitySet::new().with(TestCapA).with(TestCapB);
    assert!(caps.get::<TestCapA>().is_some());
    assert!(caps.get::<TestCapB>().is_some());
    assert_eq!(caps.get::<TestCapA>().unwrap().name(), "cap_a");
}

#[test]
fn test_capability_set_missing_returns_none() {
    let caps = CapabilitySet::new().with(TestCapA);
    assert!(caps.get::<TestCapB>().is_none());
}

#[test]
fn test_capability_set_clone() {
    let caps = CapabilitySet::new().with(TestCapA);
    let cloned = caps.clone();
    assert!(cloned.get::<TestCapA>().is_some());
}

#[test]
fn test_derived_field_access() {
    #[derive(FieldAccess)]
    struct Config {
        label: String,
        count: u32,
    }
    let mut cfg = Config {
        label: "hello".into(),
        count: 5,
    };
    assert_eq!(cfg.get_field("label").unwrap(), "hello");
    assert_eq!(cfg.get_field("count").unwrap(), "5");
    cfg.set_field("label", "world").unwrap();
    cfg.set_field("count", "10").unwrap();
    assert_eq!(cfg.get_field("label").unwrap(), "world");
    assert_eq!(cfg.get_field("count").unwrap(), "10");
    assert!(cfg.set_field("missing", "x").is_err());
    assert_eq!(cfg.field_names(), &["label", "count"]);
}
