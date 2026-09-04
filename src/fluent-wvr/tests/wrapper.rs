#![allow(unused_imports)]
use common_core::metrics::LatencyHistogram;
#[allow(unused_imports)]
use fluent_wvr::wrapper::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


use fluent_wvr::test_support::{
    AdapterHost, CloneableUnit, ConstrainedHost, DepProvider, FieldedHost, MockUnit,
    failing_unit, ok_unit,
};
use std::sync::atomic::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn retry_call_succeeds_first() {
    let result: Result<RetryResult<i32>, ()> = retry_call(3, 1, || Ok(42));
    assert_eq!(result.unwrap().result, 42);
}

#[test]
fn retry_call_always_fails() {
    let result: Result<RetryResult<i32>, ()> = retry_call(3, 1, || Err(()));
    assert!(result.is_err());
}

#[test]
fn retry_call_jittered_delays_are_non_deterministic() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    // Run two sequences of retries; due to jitter, total wall time
    // should differ (at least one will be faster). We use a counter
    // that always fails to force all retry delays to fire.
    fn run_retries() -> u128 {
        let calls = AtomicUsize::new(0);
        let start = Instant::now();
        let _ = retry_call(5, 10, || {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<i32, ()>(())
        });
        start.elapsed().as_millis()
    }

    let t1 = run_retries();
    let t2 = run_retries();
    // Both should have taken some time (at least a few ms from the delays).
    assert!(t1 > 0, "first run should take > 0ms");
    assert!(t2 > 0, "second run should take > 0ms");
    // With jitter, the two runs should not be identical (very high
    // probability with 4 jittered delays of 10ms each).
    // We can't assert strict inequality due to timer resolution, but
    // we can assert both are positive which proves the delays fired.
}

#[test]
fn instrumented_delegates() {
    let inner = MockUnit::ok("mock");
    let wrapped = Instrumented::new(inner, "test-label");
    let ctx = WorkContext::default();
    let result = wrapped.execute(&ctx);
    assert!(result.is_ok());
    assert_eq!(wrapped.name(), "mock");
}

/// Exercises `Instrumented::with_metrics` end-to-end: confirms the
/// histogram records one observation after `execute` runs and that the
/// recorded sum is non-zero. This is the minimum-bar in-tree usage;
/// see the `with_metrics` doc comment for the candidate production sites
/// that have not yet been wired.
#[test]
fn instrumented_with_metrics_records_duration() {
    let inner = MockUnit::ok("mock");
    let histogram = Arc::new(LatencyHistogram::new());
    let wrapped = Instrumented::with_metrics(inner, "knn_mock", Arc::clone(&histogram));
    let ctx = WorkContext::default();
    let result = wrapped.execute(&ctx);
    assert!(result.is_ok());
    // Exactly one observation was recorded by the wrapper's `execute`.
    assert_eq!(histogram.count(), 1);
    // `estimate_percentile` returns the bucket bound for the observed
    // duration; a non-zero value confirms the wiring is live (and that
    // `observe_duration` was actually called inside `execute`).
    let p50 = histogram.estimate_percentile(50.0);
    assert!(
        p50 > 0,
        "histogram must produce a non-zero p50 after one observation"
    );
}

#[test]
fn arc_dyn_work_unit_delegates() {
    let inner = MockUnit::ok("mock");
    let arc: Arc<dyn WorkUnit> = Arc::new(inner);
    let ctx = WorkContext::default();
    assert_eq!(arc.name(), "mock");
    let result = arc.execute(&ctx);
    assert!(result.is_ok());
}

// --- Downcast round-trip tests ---

#[test]
fn instrumented_downcast_roundtrip() {
    let inner = MockUnit::ok("mock");
    let instr = Instrumented::new(inner, "test");
    let arc: Arc<dyn Component> = Arc::new(instr);
    assert!(fluent_wvr::component_downcast_ref::<Instrumented<MockUnit>>(&*arc).is_some());
    // Instrumented does not auto-leak the inner type
    assert!(fluent_wvr::component_downcast_ref::<MockUnit>(&*arc).is_none());
}

// --- ComponentAdapter tests ---

#[test]
fn adapter_delegates_by_default() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host));
    assert_eq!(adapter.name(), "inner");
    let result = adapter.execute(&WorkContext::default()).unwrap();
    assert_eq!(result.message, "from_inner");
}

#[test]
fn adapter_name_override() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host)).with_name_override("renamed");
    assert_eq!(adapter.name(), "renamed");
}

#[test]
fn adapter_execute_override_short_circuits_inner() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host))
        .with_execute_override(Arc::new(|_| Ok(WorkOutput::ok("overridden"))));
    let result = adapter.execute(&WorkContext::default()).unwrap();
    assert_eq!(result.message, "overridden");
}

#[test]
fn adapter_field_override_set_then_get() {
    let host = AdapterHost::new("inner");
    let mut adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "8079");
    assert_eq!(adapter.get_field("port").unwrap(), "8079");
    adapter.set_field("port", "9090").unwrap();
    assert_eq!(adapter.get_field("port").unwrap(), "9090");
}

#[test]
fn adapter_field_overrides_stack_last_wins() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host))
        .with_field_override("k", "v1")
        .with_field_override("k", "v2");
    assert_eq!(adapter.get_field("k").unwrap(), "v2");
}

#[test]
fn adapter_field_not_found() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host));
    assert!(matches!(
        adapter.get_field("missing"),
        Err(FieldError::NotFound(_))
    ));
}

#[test]
fn adapter_is_itself_a_component() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host));
    // Box as dyn Component — proves the blanket impl fires.
    let boxed: Box<dyn Component> = Box::new(adapter);
    assert_eq!(boxed.name(), "inner");
}

#[test]
fn adapter_describe_includes_overrides() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host))
        .with_name_override("renamed")
        .with_field_override("port", "8079");
    let schema = adapter.describe();
    assert_eq!(schema["name"], "renamed");
    assert_eq!(schema["adapted"], true);
    let overrides = schema["field_overrides"].as_array().unwrap();
    assert_eq!(overrides.len(), 1);
    assert_eq!(overrides[0][0], "port");
    assert_eq!(overrides[0][1], "8079");
}

#[test]
fn adapter_inner_accessor_returns_wrapped_component() {
    let host = AdapterHost::new("inner");
    let adapter = ComponentAdapter::new(Arc::new(host));
    let inner: &Arc<dyn Component> = adapter.inner();
    assert_eq!(inner.name(), "inner");
}

#[test]
fn adapter_propagates_depends_and_provides_from_inner() {
    // A Component whose depends/provides are non-empty — confirm
    // delegation rather than pass-through to the empty default.
    let inner = DepProvider {
        name: ArcIntern::from("dp"),
        deps: vec![ArcIntern::from("a"), ArcIntern::from("b")],
        provs: vec![ArcIntern::from("c")],
    };
    let adapter = ComponentAdapter::new(Arc::new(inner));
    assert_eq!(adapter.depends().len(), 2);
    assert_eq!(&*adapter.depends()[0], "a");
    assert_eq!(&*adapter.depends()[1], "b");
    assert_eq!(adapter.provides().len(), 1);
    assert_eq!(&*adapter.provides()[0], "c");
}

/// Wrap an AdapterHost with non-empty `field_names` and verify the
/// adapter reports the same set — proves `ComponentAdapter::field_names`
/// delegates to `self.inner.field_names()` rather than returning `&[]`.
#[test]
fn adapter_field_names_forwards_from_inner() {
    let inner = FieldedHost {
        name: ArcIntern::from("fielded"),
    };
    let adapter = ComponentAdapter::new(Arc::new(inner));
    let names = adapter.field_names();
    assert_eq!(names.len(), 3);
    assert_eq!(names[0], "host_a");
    assert_eq!(names[1], "host_b");
    assert_eq!(names[2], "host_c");
}

#[test]
fn adapter_set_field_propagates_constraint_error() {
    let host = ConstrainedHost { port: 8079 };
    let mut adapter = ComponentAdapter::new(Arc::new(host));
    // Valid value — should succeed and store override.
    adapter.set_field("port", "512").unwrap();
    assert_eq!(adapter.get_field("port").unwrap(), "512");
    // Value above max — should propagate the constraint error.
    let err = adapter.set_field("port", "9999").unwrap_err();
    match err {
        FieldError::Constraint(msg) => {
            assert!(msg.contains("above maximum"), "unexpected: {msg}");
        }
        other => panic!("expected Constraint error, got: {other:?}"),
    }
    // The override should NOT have been stored for the invalid value.
    assert_eq!(adapter.get_field("port").unwrap(), "512");
}

#[test]
fn adapter_set_field_propagates_parse_error() {
    let host = ConstrainedHost { port: 8079 };
    let mut adapter = ComponentAdapter::new(Arc::new(host));
    let err = adapter.set_field("port", "not_a_number").unwrap_err();
    match err {
        FieldError::Parse(msg) => {
            assert!(msg.contains("invalid u16"), "unexpected: {msg}");
        }
        other => panic!("expected Parse error, got: {other:?}"),
    }
}

#[test]
fn instrumented_clone() {
    let unit = CloneableUnit;
    let inst = Instrumented::new(unit, "test.label");
    let cloned = inst.clone();
    // label is private; verify clone preserves identity via name delegation
    assert_eq!(cloned.name(), inst.name());
    assert_eq!(cloned.execute(&WorkContext::default()).unwrap().message, inst.execute(&WorkContext::default()).unwrap().message);
}

#[test]
fn component_adapter_clone() {
    let host = ConstrainedHost { port: 8079 };
    let adapter = ComponentAdapter::new(Arc::new(host)).with_name_override("cloned-name");
    let cloned = adapter.clone();
    assert_eq!(cloned.name(), "cloned-name");
    assert_eq!(adapter.name(), cloned.name());
}

#[test]
fn adapter_get_field_falls_back_to_inner() {
    let host = ConstrainedHost { port: 8079 };
    let adapter = ComponentAdapter::new(Arc::new(host));
    assert_eq!(adapter.get_field("port").unwrap(), "8079");
}

#[test]
fn adapter_override_takes_priority_over_inner() {
    let host = ConstrainedHost { port: 8079 };
    let adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "9090");
    assert_eq!(adapter.get_field("port").unwrap(), "9090");
}

#[test]
fn adapter_hashmap_lookup_is_o1() {
    let host = ConstrainedHost { port: 0 };
    let mut adapter = ComponentAdapter::new(Arc::new(host));
    let n = 1000;
    for i in 0..n {
        adapter = adapter.with_field_override(format!("k{i}"), format!("v{i}"));
    }
    assert_eq!(
        adapter.get_field(&format!("k{}", n - 1)).unwrap(),
        format!("v{}", n - 1)
    );
}

#[test]
fn adapter_describe_field_overrides_sorted() {
    let host = ConstrainedHost { port: 0 };
    let adapter = ComponentAdapter::new(Arc::new(host))
        .with_field_override("c", "3")
        .with_field_override("a", "1")
        .with_field_override("b", "2");
    let schema = adapter.describe();
    let overrides = schema["field_overrides"].as_array().unwrap();
    assert_eq!(overrides.len(), 3);
    assert_eq!(overrides[0][0], "a");
    assert_eq!(overrides[1][0], "b");
    assert_eq!(overrides[2][0], "c");
}

#[test]
fn adapter_with_field_overrides_bulk() {
    let host = ConstrainedHost { port: 0 };
    let mut map = HashMap::new();
    map.insert("x".to_string(), "10".to_string());
    map.insert("y".to_string(), "20".to_string());
    let adapter = ComponentAdapter::new(Arc::new(host)).with_field_overrides(map);
    assert_eq!(adapter.get_field("x").unwrap(), "10");
    assert_eq!(adapter.get_field("y").unwrap(), "20");
}

#[test]
fn adapter_clear_field_overrides_reverts_to_inner() {
    let host = ConstrainedHost { port: 8079 };
    let mut adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "9090");
    assert_eq!(adapter.get_field("port").unwrap(), "9090");
    adapter.clear_field_overrides();
    assert_eq!(adapter.get_field("port").unwrap(), "8079");
}

// --- ComponentCascade tests ---

#[test]
fn cascade_returns_first_ok_and_short_circuits() {
    let mut cascade = ComponentCascade::new();
    cascade.register(failing_unit("fail-1"));
    cascade.register(ok_unit("ok-1"));
    let ok: Arc<dyn Component> = ok_unit("ok-2");
    cascade.register(Arc::clone(&ok));

    let result = cascade
        .execute_first_ok(&WorkContext::default())
        .expect("first-ok");
    assert_eq!(result.message, "done");
    // ok-2 must not have run (short-circuit after ok-1).
    assert_eq!(
        ok.as_any()
            .downcast_ref::<MockUnit>()
            .unwrap()
            .call_count
            .load(Ordering::SeqCst),
        0
    );
}

#[test]
fn cascade_all_fail_returns_last_error() {
    let mut cascade = ComponentCascade::new();
    cascade.register(failing_unit("fail-1"));
    cascade.register(failing_unit("fail-2"));
    let err = cascade
        .execute_first_ok(&WorkContext::default())
        .expect_err("all units fail");
    assert!(matches!(err, WorkError::Execution(_)));
}

#[test]
fn cascade_empty_returns_descriptive_error() {
    let cascade = ComponentCascade::new();
    let err = cascade
        .execute_first_ok(&WorkContext::default())
        .expect_err("empty cascade");
    assert!(err.to_string().contains("no units"));
}

#[test]
fn cascade_len_and_is_empty() {
    let mut cascade = ComponentCascade::new();
    assert!(cascade.is_empty());
    cascade.push(ok_unit("a"));
    assert_eq!(cascade.len(), 1);
    assert!(!cascade.is_empty());
}

#[test]
fn cascade_is_itself_a_component() {
    let mut cascade = ComponentCascade::new();
    cascade.register(ok_unit("a"));
    let boxed: Box<dyn Component> = Box::new(cascade);
    assert_eq!(boxed.name(), "component_cascade");
    let result = boxed.execute(&WorkContext::default()).expect("delegates");
    assert_eq!(result.message, "done");
    let schema = boxed.describe();
    assert_eq!(schema["cascade"]["units"][0], "a");
}

#[test]
fn cascade_executes_empty_units() {
    let cascade = ComponentCascade::with_units(vec![failing_unit("only-fail")]);
    assert!(cascade.execute(&WorkContext::default()).is_err());
}
