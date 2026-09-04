use super::*;

#[test]
fn pipeline_stage_serde_round_trip() {
    for stage in [
        PipelineStage::DeterministicPreFilter,
        PipelineStage::Nlp,
        PipelineStage::Overlay,
        PipelineStage::Classifier,
    ] {
        let json = serde_json::to_string(&stage).expect("serialize");
        let back: PipelineStage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, stage, "round trip for {json}");
    }
}

#[test]
fn stage_verdict_serde_round_trip() {
    for verdict in [
        StageVerdict::Passed,
        StageVerdict::Rejected,
        StageVerdict::Rerouted,
        StageVerdict::Skipped,
        StageVerdict::Error,
    ] {
        let json = serde_json::to_string(&verdict).expect("serialize");
        let back: StageVerdict = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, verdict, "round trip for {json}");
    }
}

#[test]
fn stage_decision_builders_and_serde() {
    let d = StageDecision::new(PipelineStage::DeterministicPreFilter, StageVerdict::Rejected, "blocked")
        .with_score(0.9)
        .with_latency(12)
        .with_metadata(serde_json::json!({"x": 1}));
    assert_eq!(d.score, Some(0.9));
    assert_eq!(d.latency_ms, 12);
    assert_eq!(d.metadata["x"], 1);
    let back: StageDecision =
        serde_json::from_str(&serde_json::to_string(&d).expect("serialize")).expect("deserialize");
    assert_eq!(back.stage, d.stage);
    assert_eq!(back.verdict, d.verdict);
    assert_eq!(back.score, d.score);
}

#[test]
fn pii_verdict_serde_round_trip() {
    let v = PiiVerdict {
        pattern: "email".into(),
        action: FilterAction::Anonymize,
        codewords: [("a".to_string(), "b".to_string())].into_iter().collect(),
        matches: vec![RegexMatch {
            pattern_name: "email".into(),
            matched_text: "x@y.z".into(),
            start: 0,
            end: 6,
            action: FilterAction::Redact,
        }],
    };
    let back: PiiVerdict =
        serde_json::from_str(&serde_json::to_string(&v).expect("serialize")).expect("deserialize");
    assert_eq!(back, v);
}

#[test]
fn stage_metadata_typed_accessors() {
    let mut m = StageMetadata::new(serde_json::json!({}));
    m.set_response("hello");
    m.set_rewritten_request("rewritten");
    m.set_command_result("result");
    m.set_fallback(true);
    assert_eq!(m.response(), Some("hello"));
    assert_eq!(m.rewritten_request(), Some("rewritten"));
    assert_eq!(m.command_result(), Some("result"));
    assert_eq!(m.fallback(), Some(true));

    let pii = PiiVerdict {
        pattern: "ssn".into(),
        action: FilterAction::Redact,
        codewords: Default::default(),
        matches: vec![],
    };
    m.set_pii_filter(&pii);
    assert_eq!(m.pii_filter().expect("pii"), pii);

    // `RoutingTarget` travels through the typed store, not metadata: publish
    // it and read it back by value. Metadata itself carries no shim.
    let rt: crate::pipeline::RoutingTarget = serde_json::from_value(serde_json::json!({
        "url": "http://upstream",
        "model": "fast",
    }))
    .expect("routing target from json");
    assert_eq!(rt.model, "fast");
    let mut decision = StageDecision::new(
        PipelineStage::Classifier,
        StageVerdict::Passed,
        "typed handoff",
    );
    let mut ctx = fluent_wvr::WorkContext::default();
    crate::stages::common::publish_routing_target(&mut ctx, &mut decision, rt);
    assert_eq!(
        ctx.get::<crate::pipeline::RoutingTarget>(
            crate::stages::common::ROUTING_TARGET_TYPED_KEY
        )
        .expect("typed routing target")
        .model,
        "fast"
    );
    assert!(
        decision.metadata.get("routing_target").is_none(),
        "metadata carries no routing_target shim"
    );

    m.insert("custom", serde_json::json!(true));
    assert_eq!(m.as_value()["custom"], true);
    // `into_value` unwraps to the underlying map.
    assert_eq!(m.into_value()["response"], "hello");
}

#[test]
fn stage_metadata_nlp_parse_round_trip() {
    use spacy_rs::routing::RoutingSignal;
    let mut m = StageMetadata::new(serde_json::json!({}));
    let signals = vec![RoutingSignal {
        sentence: "show me the report".into(),
        predicate: "show".into(),
        subject: None,
        direct_object: Some("report".into()),
        indirect_object: Some("me".into()),
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["show".into(), "me".into()],
        lemmas: vec!["show".into(), "me".into()],
        pos: vec!["verb".into(), "pron".into()],
        deps: vec!["root".into(), "iobj".into()],
        heads: vec![0, -1],
        interlingua: None,
    }];
    m.set_nlp_parse(&signals);
    assert_eq!(m.nlp_parse().expect("nlp_parse"), signals);
    assert_eq!(StageMetadata::new(serde_json::json!({})).nlp_parse(), None);
}

#[test]
fn stage_metadata_missing_accessors_return_none() {
    let m = StageMetadata::new(serde_json::json!({}));
    assert_eq!(m.response(), None);
    assert!(m.pii_filter().is_none());
    assert_eq!(m.fallback(), None);
}

#[test]
fn stage_metadata_from_value_and_transparent_serde() {
    // `#[serde(transparent)]` means the wrapper (de)serializes as the
    // underlying JSON value alone.
    let json = serde_json::json!({"routing_target": {"model": "fast", "group": "fast"}});
    let m: StageMetadata = serde_json::from_value(json.clone()).expect("from value");
    let back = serde_json::to_value(m).expect("to value");
    assert_eq!(back, json);
}
#[test]
fn nlp_interlingua_and_confidence_accessors_round_trip() {
    use spacy_rs::routing::InterlinguaSignal;
    let mut m = StageMetadata::new(serde_json::json!({}));
    let sigs = vec![InterlinguaSignal {
        predicate_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0001)),
        subject_id: None,
        direct_object_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0002)),
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: vec![],
        confidence: None,
    }];
    m.set_nlp_interlingua(&sigs);
    assert_eq!(m.nlp_interlingua().expect("il"), sigs);
    assert_eq!(StageMetadata::new(serde_json::json!({})).nlp_interlingua(), None);

    let conf = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::ArcEager,
        overall: 0.6,
        role_coverage: 0.5,
        oracle_tie_count: 2,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    };
    m.set_nlp_confidence(&conf);
    assert_eq!(m.nlp_confidence().expect("conf"), conf);
}

#[test]
fn needs_disambiguation_gates_on_source_and_collisions() {
    let low = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::ArcEager,
        overall: 0.3,
        role_coverage: 0.5,
        oracle_tie_count: 3,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    };
    assert!(low.needs_disambiguation(0.5));
    assert!(!low.needs_disambiguation(0.2), "above a lower threshold → no");
    // LLM parses are never flagged (they are the capable tier).
    let llm = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::Llm,
        overall: 0.1,
        role_coverage: 1.0,
        oracle_tie_count: 0,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    };
    assert!(!llm.needs_disambiguation(0.5));
    // An encoder parse is treated like ArcEager: low confidence flags.
    let encoder = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::Encoder,
        overall: 0.3,
        role_coverage: 0.5,
        oracle_tie_count: 2,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    };
    assert!(encoder.needs_disambiguation(0.5));
    let encoder_high = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::Encoder,
        overall: 0.9,
        role_coverage: 0.9,
        oracle_tie_count: 0,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    };
    assert!(!encoder_high.needs_disambiguation(0.5));
    // A collision flags the request regardless of the model tier.
    let collided = NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::Llm,
        overall: 0.9,
        role_coverage: 1.0,
        oracle_tie_count: 0,
        collision_count: 2,
        semantic_plausibility: None,
        refine_reason: None,
    };
    assert!(collided.needs_disambiguation(0.5));
}

#[test]
fn router_label_deserializes_to_classifier_with_warn() {
    // Historical `"Router"` payloads map to `Classifier` (with a `warn!` at
    // the deserialization site) so stored decisions keep reading.
    let back: PipelineStage =
        serde_json::from_str("\"Router\"").expect("legacy Router label reads");
    assert_eq!(
        back,
        PipelineStage::Classifier,
        "legacy Router deserializes to Classifier"
    );
    // Fresh code never emits the label.
    let fresh = serde_json::to_string(&PipelineStage::Classifier).expect("serialize");
    assert_eq!(fresh, "\"Classifier\"");
}
