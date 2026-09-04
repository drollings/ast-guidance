#![allow(unused_imports)]
use std::time::Duration;
#[allow(unused_imports)]
use fluent_wvr::runtime::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


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
