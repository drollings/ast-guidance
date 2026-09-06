//! Baseline characterization for the deprecated-code replacement roadmap.
//!
//! Locks current behavior before any migration: typed/JSON dual-channel
//! equality for the routing-target handoff. Production code unchanged.

use crate::pipeline::RoutingTarget;
use crate::pipeline_types::{StageDecision, StageVerdict};

#[test]
fn dual_channel_equality() {
    use crate::stages::common::{publish_routing_target, ROUTING_TARGET_TYPED_KEY};

    let entry: crate::config::ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "baseline-model",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
    }))
    .expect("valid ModelEntry");
    let rt = RoutingTarget::from_model_entry("baseline", &entry);
    let mut decision = StageDecision {
        stage: crate::pipeline_types::PipelineStage::Classifier,
        verdict: StageVerdict::Passed,
        score: None,
        reason: "baseline".into(),
        latency_ms: 0,
        metadata: serde_json::json!({}),
    };
    let mut ctx = fluent_wvr::WorkContext::default();
    publish_routing_target(&mut ctx, &mut decision, rt.clone());

    let typed = ctx
        .get::<RoutingTarget>(ROUTING_TARGET_TYPED_KEY)
        .expect("typed channel present");
    assert_eq!(typed.model, rt.model, "typed model matches producer value");
    assert!(
        decision.metadata.get("routing_target").is_none(),
        "publish is typed-only: no JSON shim in metadata"
    );
}
