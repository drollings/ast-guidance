#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::work::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};



#[test]
fn test_work_context_default() {
    let ctx = WorkContext::default();
    assert!(!ctx.dry_run);
    assert_eq!(ctx.timeout_ms, 30000);
    assert!(ctx.outputs.is_empty());
}

#[test]
fn work_context_typed_channel_handoff() {
    let mut ctx = WorkContext::default();
    ctx.set("stage", "deterministic".to_string());
    ctx.set("attempt", 1_i32);

    assert_eq!(
        ctx.get::<String>("stage"),
        Some(&"deterministic".to_string())
    );
    assert_eq!(ctx.get::<i32>("attempt"), Some(&1));
    assert_eq!(ctx.get::<u32>("attempt"), None, "wrong type reads None");
    assert_eq!(ctx.get::<i32>("absent"), None);

    let clone = ctx.clone();
    assert_eq!(
        clone.get::<String>("stage"),
        Some(&"deterministic".to_string()),
        "clone shares the typed allocation"
    );
}

#[test]
fn work_context_typed_channel_overwrite() {
    let mut ctx = WorkContext::default();
    ctx.set("verdict", "passed".to_string());
    ctx.set("verdict", "rerouted".to_string());
    assert_eq!(ctx.get::<String>("verdict"), Some(&"rerouted".to_string()));
}

#[test]
fn test_work_output_helpers() {
    assert!(WorkOutput::ok("done").success);
    assert!(!WorkOutput::fail("error").success);
}

#[test]
fn work_output_typed_and_data_as_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct MyData {
        x: i32,
        label: String,
    }
    let data = MyData {
        x: 42,
        label: "hello".into(),
    };
    let output = WorkOutput::typed("ok", &data).unwrap();
    assert!(output.success);
    assert_eq!(output.message, "ok");

    let recovered: MyData = output.data_as().unwrap();
    assert_eq!(recovered, data);
}

#[test]
fn work_output_typed_returns_err_on_unserializable() {
    /// A type whose `Serialize` impl always fails.
    struct AlwaysFails;
    impl serde::Serialize for AlwaysFails {
        fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("always fails"))
        }
    }
    let result = WorkOutput::typed("fail", &AlwaysFails);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("execution failed") || msg.contains("serialize"),
        "error message should describe serialization failure, got: {msg}"
    );
}

#[test]
fn work_output_data_take_consumes_self() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct MyData {
        x: i32,
    }
    let data = MyData { x: 42 };
    let output = WorkOutput::typed("ok", &data).unwrap();

    let recovered: MyData = output.data_take().unwrap();
    assert_eq!(recovered, data);
    // output is consumed — cannot call data_as anymore
}

#[test]
fn work_output_typed_infallible_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct MyData {
        x: i32,
    }
    let data = MyData { x: 42 };
    let output = WorkOutput::typed_infallible("ok", &data);
    assert!(output.success);
    let recovered: MyData = output.data_as().unwrap();
    assert_eq!(recovered, data);
}

#[test]
fn work_output_display_ok() {
    let out = WorkOutput::ok("done");
    assert_eq!(format!("{}", out), "OK: done");
}

#[test]
fn work_output_display_fail() {
    let out = WorkOutput::fail("oops");
    assert_eq!(format!("{}", out), "FAIL: oops");
}

#[test]
fn work_error_partial_eq() {
    let a = WorkError::Execution("boom".into());
    let b = WorkError::Execution("boom".into());
    assert_eq!(a, b);
}

#[test]
fn work_error_is_retryable_classification() {
    // Execution is a permanent failure — never retry.
    assert!(!WorkError::Execution("boom".into()).is_retryable());
    // Dependency (prerequisite not yet satisfied) is transient — retry.
    assert!(WorkError::Dependency("awaiting asset".into()).is_retryable());
    // Timeout (budget exhausted under load) is transient — retry.
    assert!(WorkError::Timeout {
        duration_ms: 100,
        unit: "u".into()
    }
    .is_retryable());
}

#[test]
fn field_error_partial_eq() {
    let a = FieldError::NotFound("x".into());
    let b = FieldError::NotFound("x".into());
    assert_eq!(a, b);
}
