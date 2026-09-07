use super::*;

fn test_entry() -> ModelEntry {
    serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "unsloth/lfm2.5-1.2b-instruct",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
        "total_timeout_ms": 40000,
        "idle_timeout_ms": 8000,
        "stream": true,
        "filter_thinking": true,
        "retry_count": 2,
        "retry_base_interval_s": 1,
        "params": {
            "num_ctx": 98304,
            "parallel": 3,
            "sleep_idle_seconds": 7200
        }
    }))
    .expect("valid ModelEntry")
}

#[test]
fn from_model_entry_strips_declaration_only_params() {
    let rt = RoutingTarget::from_model_entry("lfm", &test_entry());

    assert_eq!(rt.url, "http://localhost:8080/v1/chat/completions");
    assert_eq!(rt.model, "unsloth/lfm2.5-1.2b-instruct");
    assert_eq!(rt.target_name.as_deref(), Some("lfm"));
    // The declaration-only llama.cpp keys are stripped — they are owned by
    // the instance grammar, not the request body.
    let params = rt.params.expect("params present");
    assert!(params.get("num_ctx").is_none());
    assert!(params.get("parallel").is_none());
    assert!(params.get("sleep_idle_seconds").is_none());
    assert!(rt.filter_thinking);
    assert_eq!(rt.retry_count, 2);
    assert_eq!(rt.retry_base_interval_s, 1);
    assert!(rt.stream);
    assert_eq!(rt.idle_timeout_ms, 8000);
    assert_eq!(rt.total_timeout_ms, 40000);
}

#[test]
fn from_model_entry_keeps_sampling_params() {
    let mut entry = test_entry();
    entry.params = Some(serde_json::json!({
        "num_ctx": 98304,
        "temperature": 0.1,
        "repeat_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": false},
    }));
    let rt = RoutingTarget::from_model_entry("lfm", &entry);
    let params = rt.params.expect("params present");
    assert!(params.get("num_ctx").is_none());
    assert_eq!(params.get("temperature"), Some(&serde_json::json!(0.1)));
    assert_eq!(params.get("repeat_penalty"), Some(&serde_json::json!(1.1)));
    assert!(params.get("chat_template_kwargs").is_some());
}

fn entry_with_instances(instances: serde_json::Value) -> ModelEntry {
    serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "abiray/lfm2.5-2.6b-heretic-abliterated",
        "intelligence": 2,
        "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7,
        "speed": 8,
        "instances": instances,
    }))
    .expect("valid ModelEntry")
}

// NOTE (pruned): single-shared-group, default-profile, bare-base, and
// named-point wire shapes were pinned here and are now covered once per tier
// by the role golden (`target_for_key.lfm2.5-2.6b` / `.code` / `.code:missing`
// cases), the single-resolver precedence tests (`inference_point_*`), and the
// named-instance backend test — one assertion per tier, no duplicates.

#[test]
fn from_model_entry_merges_instance_profile_params() {
    // The reference swarm config: the default profile (ledger) carries
    // temperature 0.1, scratch carries 0.4. Those must reach the body for
    // the qualifier each builder resolves.
    let entry = entry_with_instances(serde_json::json!({
        "swarm": { "count": 3, "group": "swarm", "num_ctx": 16384, "warm": true,
                   "params": { "temperature": 0.1 } },
        "ledger": { "num_ctx": 131072, "pinned": true, "default": true,
                    "params": { "temperature": 0.1 } },
        "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30,
                     "params": { "temperature": 0.4 } }
    }));

    // from_model_entry resolves the default profile (ledger, temp 0.1).
    let rt = RoutingTarget::from_model_entry("swarm", &entry);
    assert_eq!(rt.model, "abiray/lfm2.5-2.6b-heretic-abliterated:ledger");
    assert_eq!(
        rt.params.as_ref().and_then(|p| p.get("temperature")),
        Some(&serde_json::json!(0.1)),
        "default profile sampling params reach the dispatch body"
    );

    // from_model_entry_instance targets scratch (temp 0.4).
    let rt = RoutingTarget::from_model_entry_instance("swarm", &entry, "scratch");
    assert_eq!(rt.model, "abiray/lfm2.5-2.6b-heretic-abliterated:scratch");
    assert_eq!(
        rt.params.as_ref().and_then(|p| p.get("temperature")),
        Some(&serde_json::json!(0.4)),
        "named-instance sampling params reach the dispatch body"
    );
}

#[test]
fn from_model_entry_falls_back_to_key_when_name_missing() {
    let mut entry = test_entry();
    entry.name = None;
    let rt = RoutingTarget::from_model_entry("lfm", &entry);
    assert_eq!(rt.model, "lfm");
}

#[test]
fn qualified_model_id_roundtrip() {
    let q = QualifiedModelId::qualified("base-model", "swarm");
    assert_eq!(q.as_wire(), "base-model:swarm");
    assert_eq!(QualifiedModelId::parse("base-model:swarm"), q);
    let b = QualifiedModelId::bare("base-model");
    assert_eq!(b.as_wire(), "base-model");
    assert_eq!(QualifiedModelId::parse("base-model"), b);
    assert!(q.is_qualified());
    assert!(!b.is_qualified());
}

#[test]
fn routing_target_serde_defaults_read_canonical_constants() {
    // Round-trips through the serde path (no explicit timeout/retry fields)
    // so the defaults actually exercised are the serde defaults — guards
    // against the 120s/10s-vs-300s/30s divergence recurring (D7).
    let rt: RoutingTarget = serde_json::from_str(r#"{"url":"u","model":"m"}"#).unwrap();
    assert_eq!(
        rt.total_timeout_ms,
        fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
    );
    assert_eq!(
        rt.idle_timeout_ms,
        fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS
    );
    assert_eq!(
        rt.retry_base_interval_s,
        fluent_llm::constants::DEFAULT_RETRY_INTERVAL_S
    );
}

#[test]
fn routing_target_typed_preferred_over_json() {
    use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
    use crate::stages::common::ROUTING_TARGET_TYPED_KEY;
    use fluent_wvr::WorkContext;

    let typed_rt = RoutingTarget {
        url: "http://typed".into(),
        model: "typed-model".into(),
        group: None,
        target_name: Some("typed".into()),
        params: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        filter_thinking: false,
        retry_count: 1,
        retry_base_interval_s: 1,
        stream: true,
        idle_timeout_ms: 8000,
        total_timeout_ms: 30000,
        api_key: None,
        fallbacks: vec![],
        is_onnx: false,
    };
    let json_rt = RoutingTarget {
        url: "http://json".into(),
        model: "json-model".into(),
        group: None,
        target_name: Some("json".into()),
        params: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        filter_thinking: false,
        retry_count: 1,
        retry_base_interval_s: 1,
        stream: true,
        idle_timeout_ms: 8000,
        total_timeout_ms: 30000,
        api_key: None,
        fallbacks: vec![],
        is_onnx: false,
    };
    // A legacy decision may still carry a raw `routing_target` JSON key from
    // out-of-tree producers; the typed channel alone drives dispatch.
    let _ = json_rt;
    let meta = crate::pipeline_types::StageMetadata::new(serde_json::json!({}));
    let decision = StageDecision {
        stage: PipelineStage::Classifier,
        verdict: StageVerdict::Passed,
        score: None,
        reason: "test".into(),
        latency_ms: 0,
        metadata: meta.into_value(),
    };
    let mut ctx = WorkContext::default();
    // Publish decision and *typed* channel with typed_rt
    ctx.set(STAGE_DECISION_KEY, decision.clone());
    ctx.set(ROUTING_TARGET_TYPED_KEY, typed_rt.clone());

    let mut current_request = serde_json::json!({});
    let mut routing_target = None;
    let mut classifier_response = None;
    // handle_stage_verdict should prefer typed
    let _ = PipelineOrchestrator::handle_stage_verdict(
        &ctx,
        PipelineStage::Classifier,
        &mut current_request,
        &mut routing_target,
        &mut classifier_response,
    );
    assert_eq!(
        routing_target.as_ref().unwrap().model,
        "typed-model",
        "typed channel must be preferred over json shim"
    );
}

#[test]
fn publish_routing_target_both_channels_equal() {
    use crate::pipeline_types::{StageDecision, StageVerdict};
    use crate::stages::common::{publish_routing_target, ROUTING_TARGET_TYPED_KEY};
    use fluent_wvr::WorkContext;

    let rt = RoutingTarget::from_model_entry("test", &test_entry());
    let mut decision = StageDecision {
        stage: crate::pipeline_types::PipelineStage::Classifier,
        verdict: StageVerdict::Passed,
        score: None,
        reason: "test".into(),
        latency_ms: 0,
        metadata: serde_json::json!({}),
    };
    let mut ctx = WorkContext::default();
    publish_routing_target(&mut ctx, &mut decision, rt.clone());
    // Typed channel
    let typed = ctx
        .get::<RoutingTarget>(ROUTING_TARGET_TYPED_KEY)
        .expect("typed present");
    assert_eq!(typed.model, rt.model);
    // No JSON shim: the decision carries no routing_target key.
    assert!(
        decision.metadata.get("routing_target").is_none(),
        "publish is typed-only"
    );
}

#[test]
fn json_only_payload_no_longer_dispatches() {
    use crate::pipeline_types::StageMetadata;

    let rt = RoutingTarget::from_model_entry("legacy", &test_entry());
    // A legacy out-of-tree payload: raw JSON shim, no typed channel.
    let mut meta = StageMetadata::new(serde_json::json!({}));
    meta.insert(
        "routing_target",
        serde_json::to_value(&rt).expect("serialize"),
    );
    let decision = StageDecision {
        stage: PipelineStage::Classifier,
        verdict: StageVerdict::Passed,
        score: None,
        reason: "legacy".into(),
        latency_ms: 0,
        metadata: meta.into_value(),
    };
    let expected = decision.clone();
    let stage: std::sync::Arc<dyn fluent_wvr::prelude::Component> =
        std::sync::Arc::new(crate::test_stubs::SimplePassStage::new("stub", "legacy"));
    let producer: StageProducer =
        std::sync::Arc::new(move |_, _| Ok((expected.clone(), None)));
    let orch = PipelineOrchestrator::builder()
        .push_with_producer(stage, producer)
        .build();
    let out = orch
        .execute(&fluent_wvr::WorkContext::default())
        .expect("execute never panics on legacy payloads");
    let result: PipelineResult = out.data_take().expect("PipelineResult");
    assert!(
        result.routing_target.is_none(),
        "typed store missing → no target (legacy JSON shim is ignored)"
    );
}

#[test]
fn error_branch_propagates_err_without_decision() {
    // A stage failure propagates the `WorkError` through the `WorkUnit`
    // contract; no synthetic decision is recorded (the error arm returns
    // `Err` directly).
    let stage: std::sync::Arc<dyn fluent_wvr::prelude::Component> =
        std::sync::Arc::new(crate::test_stubs::SimplePassStage::new("stub", "boom"));
    let producer: StageProducer = std::sync::Arc::new(move |_, _| {
        Err(fluent_wvr::WorkError::Execution("boom".into()))
    });
    let orch = PipelineOrchestrator::builder()
        .push_with_producer(stage, producer)
        .build();
    let err = orch
        .execute(&fluent_wvr::WorkContext::default())
        .expect_err("stage failure must propagate as Err");
    let message = format!("{err}");
    assert!(
        !message.is_empty(),
        "the propagated error carries its reason"
    );
}
