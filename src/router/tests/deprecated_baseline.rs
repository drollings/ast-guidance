//! Baseline characterization for the deprecated-code replacement roadmap.
//!
//! Locks current behavior before any migration: tree-only config snapshot,
//! typed/JSON dual-channel equality, `PipelineStage::Router` label census,
//! and the overlay bool×models warning matrix. Production code unchanged.
//!
//! NOTE (M3c): this suite is owned by M7, but the flat→tree break lands here,
//! so the flat snapshot becomes the tree snapshot with this edit. M7's
//! remaining job is deleting obsolete `allow(deprecated)`s, not re-baselining.

use std::path::PathBuf;

use crate::config::{RouterConfig, RouteRef};
use crate::pipeline::RoutingTarget;
use crate::pipeline_types::{StageDecision, StageVerdict};

fn shipped_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json")
}

fn load_shipped_config() -> RouterConfig {
    let path = shipped_config_path();
    let content = std::fs::read_to_string(&path).expect("shipped config readable");
    serde_json::from_str(&content).expect("shipped config deserializes")
}

#[test]
fn tree_only_config_snapshot() {
    let cfg = load_shipped_config();
    // Same load path the server uses (composed, not copied).
    let via_loader: RouterConfig = common_core::config::load_json_or_default(
        std::path::Path::new(&shipped_config_path()),
    );
    // M3c: shipped file is tree-only.
    assert!(
        cfg.classification.is_some(),
        "shipped config must carry a classification tree"
    );
    assert!(
        via_loader.classification.is_some(),
        "loader path must agree: classification tree present"
    );
    cfg.validate_flat_tree_coherence().expect("shipped tree valid");

    let view: std::collections::HashMap<String, RouteRef> = cfg.routes_view();
    let mut keys: Vec<&String> = view.keys().collect();
    keys.sort();
    let key_strs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    assert_eq!(
        key_strs,
        vec!["code", "explain", "explore", "local", "prose", "summarize"],
        "tree routes_view key snapshot"
    );

    let routing = cfg.routing_config();
    let mut rkeys: Vec<&String> = routing.routes.keys().collect();
    rkeys.sort();
    let rkey_strs: Vec<&str> = rkeys.iter().map(|k| k.as_str()).collect();
    assert_eq!(rkey_strs, key_strs, "routing_config routes match routes_view");
    assert_eq!(
        routing.routes.len(),
        view.len(),
        "route count snapshot: {}",
        view.len()
    );

    // Derived system prompt: always tree-derived, non-empty.
    assert!(
        routing.system_prompt.contains("You are a Coral request router."),
        "tree-derived prompt, got: {}",
        routing.system_prompt
    );
}

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

#[test]
fn router_label_census() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "src/pipeline.rs",
        "src/server/dispatch.rs",
        "src/test_stubs.rs",
    ];
    let mut total = 0;
    for rel in files {
        let content =
            std::fs::read_to_string(manifest.join(rel)).expect("source file readable");
        total += content.matches("PipelineStage::Router").count();
    }
    assert_eq!(
        total, 0,
        "fresh code never names PipelineStage::Router (historical payloads map to Classifier at deserialize time)"
    );
}

#[test]
fn overlay_warning_matrix() {
    use crate::config::builder::PipelineParams;

    // The legacy `overlay` bool is deleted: the derived flag is the only
    // switch, and no bool-vs-models warning matrix remains. Both model
    // shapes below build silently.
    for models in [vec![], vec!["m".to_string()]] {
        let params = PipelineParams {
            overlay_models: models.clone(),
            ..PipelineParams::default()
        };
        assert_eq!(
            params.overlay_enabled(),
            !models.is_empty(),
            "overlay_enabled derived from models only"
        );
    }
}
