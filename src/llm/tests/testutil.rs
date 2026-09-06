//! Shared stub-backend contract: route-to-stub, fallback-to-second-stub,
//! unknown-key-`None`. Hermetic: in-process stubs only, no model or network.

use std::sync::Arc;

use fluent_llm::testutil::{CountingStubBackend, StubBackend, StubSessionLoader};
use fluent_llm::{ChatMessage, InferenceRegistry};

fn one_message() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: "hi".into(),
    }]
}

#[test]
fn shared_stub_routes_to_registered_key() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("stub", vec!["model-a"], "hello")));
    let out = registry
        .route_chat("model-a", None)
        .expect("registered key must route")
        .chat_complete(&one_message())
        .unwrap();
    assert_eq!(out, "hello");
}

#[test]
fn shared_stub_falls_through_to_second_stub() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("first", vec!["model-a"], "first")));
    registry.register(Arc::new(StubBackend::new(
        "second",
        vec!["model-b"],
        "second",
    )));
    let out = registry
        .route_chat("model-b", None)
        .expect("second stub must serve its key")
        .chat_complete(&[])
        .unwrap();
    assert_eq!(out, "second");
}

#[test]
fn shared_stub_unknown_key_routes_to_none() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("stub", vec!["model-a"], "hello")));
    assert!(registry.route_chat("model-zzz", None).is_none());
    assert!(registry.route_embed("model-zzz").is_none());
}

#[test]
fn counting_stub_records_consultation_order() {
    let mut registry = InferenceRegistry::new();
    let first = Arc::new(CountingStubBackend::new("first", vec!["model-a"], "first"));
    let second = Arc::new(CountingStubBackend::new(
        "second",
        vec!["model-b"],
        "second",
    ));
    registry.register(first.clone());
    registry.register(second.clone());
    registry
        .route_chat("model-b", None)
        .expect("second stub serves model-b");
    assert!(
        first.consults().is_empty(),
        "non-candidate backend must not be consulted"
    );
    assert_eq!(second.consults(), vec!["model-b".to_string()]);
}

#[test]
fn counting_stub_failed_readiness_skips_with_cause() {
    let mut registry = InferenceRegistry::new();
    let failed = Arc::new(CountingStubBackend::failed("broken", vec!["model-a"]));
    let healthy = Arc::new(CountingStubBackend::new("healthy", vec!["model-a"], "ok"));
    registry.register(failed.clone());
    registry.register(healthy.clone());
    let out = registry
        .route_chat("model-a", None)
        .expect("healthy fallback must serve")
        .chat_complete(&[])
        .unwrap();
    assert_eq!(out, "ok");
    assert_eq!(healthy.consults(), vec!["model-a".to_string()]);
}

#[test]
fn stub_session_loader_returns_canned_handle() {
    use fluent_llm::onnx_session::{OrtSessionRegistry, SessionLoader};
    use fluent_llm::{OnnxConfig, OnnxTask};

    let loader = StubSessionLoader;
    let handle = loader
        .load(
            &OnnxConfig::new()
                .model_path("/models/stub.onnx")
                .task(OnnxTask::FillMask)
                .build(),
            "stub-key",
        )
        .expect("stub load always succeeds");
    assert!(handle.downcast_ref::<&str>().is_some());

    let registry = OrtSessionRegistry::new(Arc::new(StubSessionLoader));
    registry
        .register(
            "stub-key",
            OnnxConfig::new()
                .model_path("/models/stub.onnx")
                .task(OnnxTask::FillMask)
                .build(),
        )
        .expect("register");
    assert!(registry.is_registered("stub-key"));
}
