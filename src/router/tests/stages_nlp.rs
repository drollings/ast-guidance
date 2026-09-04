use super::*;
use crate::config::builder::NlpDeps;
use crate::config::RouterConfig;
use crate::pipeline::PipelineResult;
use crate::types::RouterRequest;
use spacy_rs::concept_store_mem::InMemoryConceptStore;
use spacy_rs::concept_store::ConceptStore;

/// A context carrying a user message (the `structured["request"]` handoff
/// the stage reads).
fn ctx_for(text: &str) -> WorkContext {
    let request = RouterRequest {
        model: "local".into(),
        messages: vec![crate::types::RouterMessage {
            role: "user".into(),
            content: crate::types::RouterMessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: None,
        agent_id: None,
        adapter: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        metadata: Default::default(),
    };
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", &request);
    ctx
}

fn en_stage() -> NlpStage {
    NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        None,
        None,
    )
}

#[test]
fn parses_and_publishes_signals() {
    let stage = en_stage();
    let (message, decision) = stage.decide(&ctx_for("show me the report"));
    assert_eq!(message, "parsed");
    assert_eq!(decision.stage, PipelineStage::Nlp);
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let signals = StageMetadata::from(decision.metadata)
        .nlp_parse()
        .expect("nlp_parse");
    assert_eq!(signals.len(), 1);
    // Deterministic parser (middle rung): "show" is the verb predicate.
    assert_eq!(signals[0].predicate, "show");
    assert_eq!(signals[0].direct_object.as_deref(), Some("me"));
}

#[test]
fn publishes_interlingua_and_confidence_accessors() {
    // A pipeline wired to a resolver stamps interlingua ids (the boot
    // composition, 13.5); the default `en_stage` has none.
    let vocab = std::sync::Arc::new(spacy_rs::vocab::Vocab::new(
        spacy_rs::lang::en::lexicon_config(),
    ));
    let tokenizer = spacy_rs::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let store = std::sync::Arc::new(spacy_rs::concept_store_mem::InMemoryConceptStore::new());
    let resolver = std::sync::Arc::new(spacy_rs::InterlinguaResolver::new(
        std::sync::Arc::clone(&store) as std::sync::Arc<dyn spacy_rs::ConceptStore>,
        std::sync::Arc::clone(vocab.strings()),
    ));
    let pipeline = std::sync::Arc::new(
        spacy_rs::NlpPipeline::new_with_resolver(
            vocab,
            tokenizer,
            spacy_rs::AnnotationValidator::new(),
            Some(resolver),
        )
        .expect("pipeline"),
    );
    let stage = NlpStage::new(pipeline, None, None);
    let (_, decision) = stage.decide(&ctx_for("show me the report"));
    let meta = StageMetadata::from(decision.metadata);
    // The C1 handoff (13.6): per-sentence interlingua frames + summary.
    let interlingua = meta.nlp_interlingua().expect("nlp_interlingua");
    assert_eq!(interlingua.len(), 1);
    assert!(interlingua[0].predicate_id.is_some(), "predicate id stamped");
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert_eq!(conf.source, spacy_rs::AnnotationSource::ArcEager);
    assert!((0.0..=1.0).contains(&conf.overall));
}

#[test]
fn empty_text_is_skipped() {
    let stage = en_stage();
    let (_, decision) = stage.decide(&ctx_for(""));
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn llm_rung_wins_when_fetch_returns_full_parse() {
    let full = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":2,"lemma":"the"},
        {"text":"sales","pos":"noun","dep":"compound","head":1,"lemma":"sales"},
        {"text":"report","pos":"noun","dep":"dobj","head":-4,"lemma":"report"}
    ]"#;
    let fetch: spacy_rs::pipeline::LlmFetchSync =
        Arc::new(move |_tokens: Vec<String>| Ok(full.to_string()));
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        Some(fetch),
        None,
    )
    .with_refine_policy(spacy_rs::RefinePolicy {
        mode: spacy_rs::RefineMode::Always,
        ..spacy_rs::RefinePolicy::default()
    });
    let (_, decision) = stage.decide(&ctx_for("Show me the sales report"));
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let signals = StageMetadata::from(decision.metadata)
        .nlp_parse()
        .expect("nlp_parse");
    assert_eq!(signals[0].predicate, "show");
    assert_eq!(signals[0].direct_object.as_deref(), Some("report"));
}

#[test]
fn llm_fetch_failure_falls_back_to_star_parse() {
    let fetch: spacy_rs::pipeline::LlmFetchSync =
        Arc::new(move |_tokens: Vec<String>| Err(spacy_rs::pipeline::AnnotateError::Fetch(
            "boom".into(),
        )));
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        Some(fetch),
        None,
    );
    let (_, decision) = stage.decide(&ctx_for("show me the report"));
    assert_eq!(decision.verdict, StageVerdict::Passed, "parser rung answers");
    let signals = StageMetadata::from(decision.metadata)
        .nlp_parse()
        .expect("nlp_parse");
    assert_eq!(signals[0].predicate, "show");
    assert_eq!(signals[0].direct_object.as_deref(), Some("me"));
}

#[test]
fn execute_produces_typed_decision() {
    let stage = en_stage();
    let ctx = ctx_for("show me the report");
    let output = stage.execute(&ctx).expect("execute");
    let decision: StageDecision = output.data_take().expect("typed decision");
    assert_eq!(decision.stage, PipelineStage::Nlp);
}

#[test]
fn encoder_field_reflected_in_describe() {
    let stage = en_stage();
    let desc = stage.describe();
    assert_eq!(desc["encoder_rung"], false, "no encoder → false");

    // A stage with a stub encoder closure (returns error → falls back).
    let encoder: spacy_rs::pipeline::EncoderFetchSync =
        Arc::new(|_doc| Err(spacy_rs::pipeline::AnnotateError::Fetch("stub".into())));
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        None,
        Some(encoder),
    );
    let desc = stage.describe();
    assert_eq!(desc["encoder_rung"], true, "encoder present → true");
}

#[test]
fn encoder_failure_falls_back_to_arc_eager() {
    let encoder: spacy_rs::pipeline::EncoderFetchSync =
        Arc::new(|_doc| Err(spacy_rs::pipeline::AnnotateError::Fetch(
            "heads not trained".into(),
        )));
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        None,
        Some(encoder),
    );
    let (_, decision) = stage.decide(&ctx_for("show me the report"));
    assert_eq!(decision.verdict, StageVerdict::Passed, "falls back to ArcEager");
    let meta = StageMetadata::from(decision.metadata);
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert_eq!(
        conf.source,
        spacy_rs::AnnotationSource::ArcEager,
        "encoder failure → ArcEager provenance"
    );
}

#[test]
fn encoder_rung_wins_with_encoder_provenance() {
    // A stub encoder closure that produces a valid star parse (last token
    // ROOT, every other token attaches to it) aligned to the doc's orth by
    // construction — the shape the real adapter produces. It must WIN the
    // ladder and report `AnnotationSource::Encoder`.
    let encoder: spacy_rs::pipeline::EncoderFetchSync =
        Arc::new(|doc: &spacy_rs::Doc| {
            let n = doc.len();
            if n == 0 {
                return Err(spacy_rs::pipeline::AnnotateError::Encoder("empty doc".into()));
            }
            let root = n - 1;
            let mut records = Vec::with_capacity(n);
            for i in 0..n {
                let text = doc.token_text(i);
                let (pos, dep, head) = if i == root {
                    ("verb".to_string(), "root".to_string(), 0)
                } else {
                    ("noun".to_string(), "nsubj".to_string(), (root as i32) - (i as i32))
                };
                records.push(spacy_rs::AnnotationRecord {
                    text,
                    pos,
                    dep,
                    head,
                    tag: String::new(),
                    lemma: String::new(),
                    morph: String::new(),
                    ent_iob: String::new(),
                    ent_type: String::new(),
                });
            }
            Ok(spacy_rs::AnnotationSet(records))
        });
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        None,
        Some(encoder),
    )
    .with_refine_policy(spacy_rs::RefinePolicy {
        mode: spacy_rs::RefineMode::Always,
        ..spacy_rs::RefinePolicy::default()
    });
    let (_, decision) = stage.decide(&ctx_for("show me the report"));
    assert_eq!(decision.verdict, StageVerdict::Passed, "encoder rung wins");
    let meta = StageMetadata::from(decision.metadata);
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert_eq!(
        conf.source,
        spacy_rs::AnnotationSource::Encoder,
        "encoder rung must report Encoder provenance"
    );
}

#[test]
fn encoder_plus_fetch_both_provided_ladder_tries_in_order() {
    let full = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":2,"lemma":"the"},
        {"text":"sales","pos":"noun","dep":"compound","head":1,"lemma":"sales"},
        {"text":"report","pos":"noun","dep":"dobj","head":-4,"lemma":"report"}
    ]"#;
    let fetch: spacy_rs::pipeline::LlmFetchSync =
        Arc::new(move |_tokens: Vec<String>| Ok(full.to_string()));
    // Encoder stub that would fail — but LLM wins first.
    let encoder: spacy_rs::pipeline::EncoderFetchSync =
        Arc::new(|_doc| Err(spacy_rs::pipeline::AnnotateError::Fetch("stub".into())));
    let stage = NlpStage::new(
        Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline")),
        Some(fetch),
        Some(encoder),
    )
    .with_refine_policy(spacy_rs::RefinePolicy {
        mode: spacy_rs::RefineMode::Always,
        ..spacy_rs::RefinePolicy::default()
    });
    let (_, decision) = stage.decide(&ctx_for("Show me the sales report"));
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let meta = StageMetadata::from(decision.metadata);
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert_eq!(
        conf.source,
        spacy_rs::AnnotationSource::Llm,
        "LLM wins over failing encoder"
    );
}

/// M1.2 (ROADMAP_20260828_ORT): a pipeline built *through the builder* with
/// a concept store (fail-open default absent) stamps real interlingua ids —
/// the boot-wiring guarantee, exercised end-to-end. Previously the resolver
/// was constructed only in direct `NlpStage` tests (G1).
#[test]
fn builder_built_pipeline_publishes_interlingua_ids() {
    let config: RouterConfig = serde_json::from_str(
        r#"{
            "pipelines": {
                "default": {"deterministic_prefilter": false, "nlp": true, "classifier": false}
            }
        }"#,
    )
    .expect("config");
    let store: Arc<dyn ConceptStore> =
        Arc::new(InMemoryConceptStore::new());
    let nlp_deps = NlpDeps {
        concept_store: Some(store),
        strings_path: None,
    };
    let map = config.build_all_pipelines_with_backend_onnx_and_nlp(None, None, &nlp_deps);
    let pipeline = map.get("default").expect("pipeline built through the builder");
    let output = pipeline.execute(&ctx_for("show me the report")).expect("execute");
    let result: PipelineResult = output.data_take().expect("pipeline result");
    let nlp_decision = result
        .decisions
        .iter()
        .find(|d| d.stage == PipelineStage::Nlp)
        .expect("nlp stage ran");
    assert_eq!(nlp_decision.verdict, StageVerdict::Passed);
    let meta = StageMetadata::from(nlp_decision.metadata.clone());
    let interlingua = meta.nlp_interlingua().expect("nlp_interlingua");
    assert_eq!(interlingua.len(), 1);
    assert!(
        interlingua[0].predicate_id.is_some(),
        "predicate id stamped by the resolver (end-to-end)"
    );
    assert!(
        interlingua[0].direct_object_id.is_some(),
        "direct object id stamped by the resolver (end-to-end)"
    );
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert_eq!(conf.source, spacy_rs::AnnotationSource::ArcEager);
}

/// M1.2 (ROADMAP_20260828_ORT): with a concept store absent the builder
/// fails open (a `warn!`) and the pipeline still builds — the NlpStage
/// publishes a parse, but no interlingua ids (resolver `None`).
#[test]
fn builder_built_pipeline_fails_open_without_concept_store() {
    let config: RouterConfig = serde_json::from_str(
        r#"{
            "pipelines": {
                "default": {"deterministic_prefilter": false, "nlp": true, "classifier": false}
            }
        }"#,
    )
    .expect("config");
    let map = config.build_all_pipelines_with_backend_onnx_and_nlp(None, None, &NlpDeps::default());
    let pipeline = map.get("default").expect("pipeline built");
    let output = pipeline.execute(&ctx_for("show me the report")).expect("execute");
    let result: PipelineResult = output.data_take().expect("pipeline result");
    let nlp_decision = result
        .decisions
        .iter()
        .find(|d| d.stage == PipelineStage::Nlp)
        .expect("nlp stage ran");
    let meta = StageMetadata::from(nlp_decision.metadata.clone());
    let interlingua = meta.nlp_interlingua().expect("nlp_interlingua");
    assert_eq!(interlingua.len(), 1);
    assert!(
        interlingua[0].predicate_id.is_none(),
        "no resolver ⇒ no interlingua ids (fail-open)"
    );
}

/// G9 (ROADMAP_20260828_ORT): the confidence summary reports the resolver's
/// surfaced collision count, not a hardcoded zero. A collision is forced by
/// registering a second canonical under a taken id.
#[test]
fn confidence_summary_reports_collision_count() {
    let store = Arc::new(InMemoryConceptStore::new());
    // Prime a canonical for a lemma id so resolving a *different* canonical
    // with the same content hash collides. Use an id the resolver will hit.
    // (The InMemoryConceptStore is content-addressed; inserting a generic
    // YagoClass concept that shares the lemma's hash triggers a collision.)
    let lemma = "show";
    let id = fluent_types::lemma_id_for_str(lemma);
    store
        .insert(fluent_types::ConceptMetadata {
            id,
            canonical_name: "colliding-canonical".into(),
            namespace: fluent_types::InterlinguaNamespace::SpacyLemma,
            yago_iri: None,
            yago_class_iri: None,
            label: None,
            node_id: None,
            parent_class_id: None,
        })
        .expect("insert colliding canonical");
    let store_arc: Arc<dyn ConceptStore> = store;
    let nlp_deps = NlpDeps {
        concept_store: Some(store_arc),
        strings_path: None,
    };
    let config: RouterConfig = serde_json::from_str(
        r#"{
            "pipelines": {
                "default": {"deterministic_prefilter": false, "nlp": true, "classifier": false}
            }
        }"#,
    )
    .expect("config");
    let map = config.build_all_pipelines_with_backend_onnx_and_nlp(None, None, &nlp_deps);
    let pipeline = map.get("default").expect("pipeline built");
    let output = pipeline.execute(&ctx_for("show me the report")).expect("execute");
    let result: PipelineResult = output.data_take().expect("pipeline result");
    let nlp_decision = result
        .decisions
        .iter()
        .find(|d| d.stage == PipelineStage::Nlp)
        .expect("nlp stage ran");
    let meta = StageMetadata::from(nlp_decision.metadata.clone());
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    assert!(
        conf.collision_count > 0,
        "the resolver's surfaced collision must reach the summary (G9)"
    );
}

/// M4.3: refinement products keep `source == Llm/Encoder` (not
/// confidence-bearing) — the `NlpConfidenceSummary` logic treats them as
/// non-gated (overall 1.0) even when a parse confidence is present, and the
/// exhaustive `is_confidence_bearing` match stays honest for future variants.
#[test]
fn refined_products_are_not_confidence_bearing_and_summary_is_well_formed() {
    // Every source must be classified exhaustively (no wildcard).
    assert!(!spacy_rs::AnnotationSource::Llm.is_confidence_bearing());
    assert!(!spacy_rs::AnnotationSource::RuleRung.is_confidence_bearing());
    assert!(!spacy_rs::AnnotationSource::HumanReview.is_confidence_bearing());
    assert!(!spacy_rs::AnnotationSource::Frontier.is_confidence_bearing());
    assert!(spacy_rs::AnnotationSource::ArcEager.is_confidence_bearing());
    assert!(spacy_rs::AnnotationSource::Encoder.is_confidence_bearing());

    // A refined result (Llm) carrying a parse confidence must still report
    // 1.0 — the summary gates only on confidence-bearing sources.
    let vocab = std::sync::Arc::new(spacy_rs::vocab::Vocab::new(
        spacy_rs::lang::en::lexicon_config(),
    ));
    let mut doc =
        spacy_rs::doc::Doc::new(std::sync::Arc::clone(&vocab));
    doc.push_back("hello", false).expect("push");
    let set = spacy_rs::AnnotationSet(vec![spacy_rs::AnnotationRecord {
        text: "hello".into(),
        pos: "noun".into(),
        dep: "root".into(),
        head: 0,
        tag: String::new(),
        lemma: "hello".into(),
        morph: String::new(),
        ent_iob: String::new(),
        ent_type: String::new(),
    }]);
    let mut refined = spacy_rs::AnnotationResult::new(set, spacy_rs::AnnotationSource::Llm);
    // Give it a fake parse confidence that would be gated if it were ArcEager.
    refined.parse_confidence = Some(spacy_rs::arc_eager::ParseConfidence {
        overall: 0.2,
        role_coverage: 0.0,
        oracle_tie_count: 7,
        token_scores: vec![0.2],
        oracle_margins: vec![0.0],
        semantic_plausibility: Some(0.1),
    });
    refined.collision_count = 1;
    let summary = confidence_summary(&refined);
    assert_eq!(summary.source, spacy_rs::AnnotationSource::Llm);
    // Non-bearing source => confidence is the 1.0 convention, not the low value.
    assert!((summary.overall - 1.0).abs() < f64::EPSILON);
    assert!((summary.role_coverage - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.oracle_tie_count, 0);
    assert_eq!(summary.collision_count, 1);
    // Encoder IS bearing — its confidence passes through.
    let mut enc = refined.clone();
    enc.source = spacy_rs::AnnotationSource::Encoder;
    let enc_summary = confidence_summary(&enc);
    assert!((enc_summary.overall - 0.2).abs() < f64::EPSILON);
}

/// M4.3: an `OnUncertain`-refined parse's `NlpConfidenceSummary` is
/// well-formed (bounded, sourced correctly) — the stage publishes it even
/// when the refiner wins.
#[test]
fn on_uncertain_refined_parse_summary_is_well_formed() {
    // A fetch that returns a valid annotation array (same shape the real
    // LlmRung produces). The stage is wired with `OnUncertain`; the text
    // "hello" yields a trivial deterministic base that the policy may or may
    // not refine, but either way the summary must be bounded and the source
    // must be one of the known variants.
    let full = r#"[{"text":"hello","pos":"noun","dep":"root","head":0,"lemma":"hello"}]"#;
    let fetch: spacy_rs::pipeline::LlmFetchSync =
        std::sync::Arc::new(move |_tokens: Vec<String>| Ok(full.to_string()));
    let pipeline =
        std::sync::Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline"));
    let stage = NlpStage::with_strings(pipeline, Some(fetch), None, None)
        .with_refine_policy(spacy_rs::RefinePolicy {
            mode: spacy_rs::RefineMode::OnUncertain,
            ..spacy_rs::RefinePolicy::default()
        });
    let (_, decision) = stage.decide(&ctx_for("hello"));
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let meta = StageMetadata::from(decision.metadata);
    let conf = meta.nlp_confidence().expect("nlp_confidence");
    // Summary is well-formed regardless of whether refinement ran.
    assert!((0.0..=1.0).contains(&conf.overall));
    assert!((0.0..=1.0).contains(&conf.role_coverage));
    assert!(matches!(
        conf.source,
        spacy_rs::AnnotationSource::ArcEager
            | spacy_rs::AnnotationSource::Llm
            | spacy_rs::AnnotationSource::Encoder
            | spacy_rs::AnnotationSource::RuleRung
    ));
}

#[test]
fn nlp_stage_new_with_fetch_defaults_to_off() {
    let fetch: spacy_rs::pipeline::LlmFetchSync =
        std::sync::Arc::new(|_tokens: Vec<String>| Ok("[]".to_string()));
    let pipeline =
        std::sync::Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline"));
    let stage = NlpStage::new(pipeline, Some(fetch), None);
    assert_eq!(
        stage.refine_policy().mode,
        spacy_rs::RefineMode::Off,
        "NlpStage::new must default to Off even when fetch is wired"
    );
    let pipeline2 =
        std::sync::Arc::new(spacy_rs::NlpPipeline::en_default().expect("en pipeline"));
    let encoder: spacy_rs::pipeline::EncoderFetchSync =
        std::sync::Arc::new(|_doc| Err(spacy_rs::pipeline::AnnotateError::Encoder("x".into())));
    let stage2 = NlpStage::new(pipeline2, None, Some(encoder));
    assert_eq!(
        stage2.refine_policy().mode,
        spacy_rs::RefineMode::Off,
        "NlpStage::new must default to Off even when encoder is wired"
    );
}
