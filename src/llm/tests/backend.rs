//! Backend plugin layer tests: registry routing, capability defaults,
//! moved neutral-type serde shapes, and old-path re-export equivalence.
//!
//! Hermetic: stub backends only, no model or network.

use std::sync::Arc;

use fluent_llm::backend::{
    BackendCaps, ContextProfile, InferenceBackend, InferenceRegistry, OverlayContribution,
    OverlayError, PiiSpan, Readiness, Residual, ResidualKind, ResidualOverlay, RouteLabel,
};
use fluent_llm::testutil::StubBackend;
use fluent_llm::{ChatMessage, EmbeddingProvider};

// ─── Registry routing ────────────────────────────────────────────────────────

#[test]
fn registry_routes_to_single_registered_backend() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("stub", vec!["model-a"], "hello")));
    let backend = registry
        .route_chat("model-a", None)
        .expect("registered key must route");
    let out = backend
        .chat_complete(&[ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }])
        .unwrap();
    assert_eq!(out, "hello");
}

#[test]
fn registry_returns_none_for_unregistered_key() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("stub", vec!["model-a"], "hello")));
    assert!(registry.route_chat("missing-model", None).is_none());
    assert!(registry.route_embed("missing-model").is_none());
}

#[test]
fn registry_falls_through_to_second_backend() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("first", vec!["model-a"], "first")));
    registry.register(Arc::new(StubBackend::new("second", vec!["model-b"], "second")));
    let out = registry
        .route_chat("model-b", None)
        .expect("second backend must serve its key")
        .chat_complete(&[])
        .unwrap();
    assert_eq!(out, "second");
    // Unknown key consults every backend and still finds nothing.
    assert!(registry.route_chat("model-zzz", None).is_none());
}

#[test]
fn registry_unregister_removes_backend() {
    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(StubBackend::new("stub", vec!["model-a"], "hello")));
    assert!(registry.route_chat("model-a", None).is_some());
    let removed = registry.unregister("stub");
    assert!(removed.is_some());
    assert!(registry.route_chat("model-a", None).is_none());
    assert!(registry.is_empty());
}

// ─── Defaults ────────────────────────────────────────────────────────────────

#[test]
fn backend_caps_defaults_are_all_false() {
    let caps = BackendCaps::default();
    assert!(!caps.named_contexts);
    assert!(!caps.kv_snapshot);
    assert!(!caps.grammar_constrained);
    assert!(!caps.embeddings);
    assert!(!caps.streaming);
}

#[test]
fn inference_backend_default_methods() {
    let stub = StubBackend::new("stub", vec!["model-a"], "hello");
    assert!(stub.embed_provider("model-a").is_none());
    assert_eq!(stub.readiness("model-a"), Readiness::Unloaded);
}

#[test]
fn context_profile_default_shape() {
    let profile = ContextProfile::default();
    assert_eq!(profile.group, "default");
    assert!(!profile.pinned);
    assert!(!profile.resume);
    assert_eq!(profile.max_ctx, None);
}

// ─── Moved-type serde goldens ────────────────────────────────────────────────

#[test]
fn pii_span_serde_golden() {
    let span = PiiSpan::new(5, 12, "contact.email", 1.0);
    let json = serde_json::to_value(&span).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "start": 5,
            "end": 12,
            "label": "contact.email",
            "score": 1.0,
        })
    );
    let back: PiiSpan = serde_json::from_value(json).unwrap();
    assert_eq!(back, span);
    assert_eq!(back.slice("user contact.email here"), Some("contact"));
}

#[test]
fn residual_kind_serde_names() {
    assert_eq!(
        serde_json::to_string(&ResidualKind::Disambiguation).unwrap(),
        "\"disambiguation\""
    );
    assert_eq!(
        serde_json::to_string(&ResidualKind::EntityLink).unwrap(),
        "\"entity_link\""
    );
}

#[test]
fn residual_and_contribution_serde_round_trip() {
    let residual = Residual {
        kind: ResidualKind::Disambiguation,
        span: Some((0, 12)),
        text: "show me the report".into(),
        meta: serde_json::json!({ "overall": 0.3 }),
    };
    let back: Residual =
        serde_json::from_str(&serde_json::to_string(&residual).unwrap()).unwrap();
    assert_eq!(back, residual);

    let contribution = OverlayContribution {
        kind: ResidualKind::Disambiguation,
        score: Some(0.9),
        payload: serde_json::json!({ "route_hints": [{"route": "code", "score": 0.9}] }),
    };
    let back: OverlayContribution =
        serde_json::from_str(&serde_json::to_string(&contribution).unwrap()).unwrap();
    assert_eq!(back, contribution);
}

#[test]
fn route_label_serde_golden() {
    let label = RouteLabel {
        route: "code".into(),
        description: "software questions".into(),
    };
    let json = serde_json::to_value(&label).unwrap();
    assert_eq!(
        json,
        serde_json::json!({ "route": "code", "description": "software questions" })
    );
    let back: RouteLabel = serde_json::from_value(json).unwrap();
    assert_eq!(back, label);
}

// ─── Single-home paths ───────────────────────────────────────────────────────
// The neutral consumer types live only in `fluent_llm::backend` (covered by
// the serde goldens above and the router's build-graph manifest test); there
// is no second `fluent_onnx::…` path to assert equivalence with.


#[test]
fn overlay_error_and_detector_shape() {
    struct NoopOverlay;
    impl ResidualOverlay for NoopOverlay {
        fn kind(&self) -> ResidualKind {
            ResidualKind::Disambiguation
        }
        fn run(
            &self,
            residual: &Residual,
        ) -> Result<OverlayContribution, OverlayError> {
            Err(OverlayError::Rejected(format!("noop: {}", residual.text)))
        }
    }
    let overlay = NoopOverlay;
    assert_eq!(overlay.kind(), ResidualKind::Disambiguation);
    let err = overlay.run(&Residual::disambiguation("hi")).unwrap_err();
    assert!(err.to_string().contains("noop"));

    // Embedding-provider seam exists on the trait and defaults to None.
    struct EmbedStub;
    impl EmbeddingProvider for EmbedStub {
        fn name(&self) -> &'static str {
            "embed-stub"
        }
        fn dimensions(&self) -> u32 {
            3
        }
        fn embed(&self, _text: &str) -> Result<Vec<f32>, fluent_llm::EmbeddingError> {
            Ok(vec![1.0, 0.0, 0.0])
        }
        fn embed_batch(
            &self,
            texts: &[&str],
        ) -> Result<fluent_llm::BatchEmbedding, fluent_llm::EmbeddingError> {
            Ok(fluent_llm::BatchEmbedding {
                flat: vec![0.0; texts.len() * 3],
                count: texts.len(),
                dims: 3,
            })
        }
    }
    let provider = EmbedStub;
    assert_eq!(provider.dimensions(), 3);
}
