use super::*;
use crate::pipeline_types::{PipelineStage, StageDecision};
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};
use fluent_onnx::overlay::{OverlayError, OverlayContribution};

fn interlingua_confidence(conf: Option<f64>) -> spacy_rs::routing::InterlinguaSignal {
    spacy_rs::routing::InterlinguaSignal {
        predicate_id: None,
        subject_id: None,
        direct_object_id: None,
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: vec![],
        confidence: conf,
    }
}

fn signal(sentence: &str, conf: Option<f64>) -> RoutingSignal {
    RoutingSignal {
        sentence: sentence.into(),
        predicate: "show".into(),
        subject: None,
        direct_object: Some("report".into()),
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["show".into(), "me".into(), "the".into(), "report".into()],
        lemmas: vec![],
        pos: vec![],
        deps: vec![],
        heads: vec![],
        interlingua: Some(interlingua_confidence(conf)),
    }
}

fn confidence(source: spacy_rs::AnnotationSource, overall: f64) -> NlpConfidenceSummary {
    NlpConfidenceSummary {
        source,
        overall,
        role_coverage: 0.5,
        oracle_tie_count: 2,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    }
}

// ── ResidualSelector purity ──

#[test]
fn low_confidence_sentence_yields_disambiguation_residual() {
    let selector = ResidualSelector::new(0.5);
    let residuals = selector.select(&[signal("show me the report", Some(0.2))], None);
    assert_eq!(residuals.len(), 1);
    assert_eq!(residuals[0].kind, ResidualKind::Disambiguation);
    assert_eq!(residuals[0].text, "show me the report");
}

#[test]
fn happy_path_yields_no_residuals() {
    let selector = ResidualSelector::new(0.5);
    let residuals = selector.select(
        &[signal("show me the report", Some(0.9))],
        Some(&confidence(spacy_rs::AnnotationSource::ArcEager, 0.9)),
    );
    assert!(residuals.is_empty());
}

#[test]
fn doc_level_needs_disambiguation_flags_all_sentences() {
    let selector = ResidualSelector::new(0.5);
    let residuals = selector.select(
        &[signal("a", Some(0.9)), signal("b", Some(0.9))],
        Some(&confidence(spacy_rs::AnnotationSource::ArcEager, 0.2)),
    );
    assert_eq!(residuals.len(), 2);
}

#[test]
fn sentence_without_confidence_is_below_the_floor() {
    let selector = ResidualSelector::new(0.5);
    let residuals = selector.select(&[signal("show me the report", None)], None);
    assert_eq!(residuals.len(), 1, "None confidence fails closed to a residual");
}

#[test]
fn llm_source_never_flags() {
    let selector = ResidualSelector::new(0.5);
    // A true LLM parse carries no per-sentence confidence and its doc-level
    // summary never flags — a high-confidence sentence stays un-flagged.
    let residuals = selector.select(
        &[signal("x", Some(0.9))],
        Some(&confidence(spacy_rs::AnnotationSource::Llm, 0.1)),
    );
    assert!(residuals.is_empty(), "LLM parses never flag");
}

// ── OverlayStage ──

/// A stub overlay that returns a canned contribution.
struct StubOverlay {
    kind: ResidualKind,
    contribution: OverlayContribution,
}

impl ResidualOverlay for StubOverlay {
    fn kind(&self) -> ResidualKind {
        self.kind
    }

    fn run(&self, _residual: &Residual) -> Result<OverlayContribution, OverlayError> {
        Ok(self.contribution.clone())
    }
}

struct FailingOverlay;

impl ResidualOverlay for FailingOverlay {
    fn kind(&self) -> ResidualKind {
        ResidualKind::Disambiguation
    }

    fn run(&self, _residual: &Residual) -> Result<OverlayContribution, OverlayError> {
        Err(OverlayError::Inference("boom".into()))
    }
}

fn hint_overlay() -> Arc<dyn ResidualOverlay> {
    Arc::new(StubOverlay {
        kind: ResidualKind::Disambiguation,
        contribution: OverlayContribution {
            kind: ResidualKind::Disambiguation,
            score: Some(0.9),
            payload: serde_json::json!({
                "route_hints": [
                    {"route": "code", "score": 0.9},
                    {"route": "prose", "score": 0.7},
                ]
            }),
        },
    })
}

fn stage_with(overlays: Vec<Arc<dyn ResidualOverlay>>) -> OverlayStage {
    OverlayStage::new(ResidualSelector::new(0.5), overlays, Some(2))
}

fn nlp_prior(signals: Vec<RoutingSignal>, conf: Option<NlpConfidenceSummary>) -> Vec<StageDecision> {
    let mut meta = StageMetadata::new(serde_json::json!({}));
    meta.set_nlp_parse(&signals);
    if let Some(c) = conf {
        meta.set_nlp_confidence(&c);
    }
    vec![StageDecision::new(
        PipelineStage::Nlp,
        StageVerdict::Passed,
        "parsed",
    )
    .with_metadata(meta.into_value())]
}

#[test]
fn hints_reach_classifier_metadata_with_stub_overlay() {
    let stage = stage_with(vec![hint_overlay()]);
    let ctx = WorkContext::default();
    let signals = vec![signal("show me the report", Some(0.2))];
    let conf = confidence(spacy_rs::AnnotationSource::ArcEager, 0.3);
    let decision = stage
        .evaluate(&ctx, &nlp_prior(signals, Some(conf)))
        .expect("decision");
    assert_eq!(decision.stage, PipelineStage::Overlay);
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let meta = StageMetadata::from(decision.metadata);
    let hints = meta.overlay_route_hints().expect("route hints");
    assert_eq!(hints.len(), 2);
    assert_eq!(hints[0].route, "code", "highest score first");
    let contributions = meta.overlay_contributions().expect("contributions");
    assert_eq!(contributions.len(), 1);
}

#[test]
fn no_residuals_skips() {
    let stage = stage_with(vec![hint_overlay()]);
    let ctx = WorkContext::default();
    let signals = vec![signal("show me the report", Some(0.9))];
    let conf = confidence(spacy_rs::AnnotationSource::ArcEager, 0.9);
    let decision = stage
        .evaluate(&ctx, &nlp_prior(signals, Some(conf)))
        .expect("decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn no_overlays_skips_with_warning() {
    let stage = stage_with(vec![]);
    let ctx = WorkContext::default();
    let signals = vec![signal("show me the report", Some(0.2))];
    let decision = stage
        .evaluate(&ctx, &nlp_prior(signals, None))
        .expect("decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn missing_nlp_handoff_skips() {
    let stage = stage_with(vec![hint_overlay()]);
    let ctx = WorkContext::default();
    let decision = stage.evaluate(&ctx, &[]).expect("decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn overlay_failure_is_fail_open() {
    // A failing overlay produces no contributions → Skipped, never Error.
    let stage = stage_with(vec![Arc::new(FailingOverlay)]);
    let ctx = WorkContext::default();
    let signals = vec![signal("show me the report", Some(0.2))];
    let decision = stage
        .evaluate(&ctx, &nlp_prior(signals, None))
        .expect("decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn execute_without_handoff_skips() {
    // The WorkUnit path has no prior decisions (the orchestrator's typed
    // `evaluate` is the real path); it degrades to Skipped.
    let stage = stage_with(vec![hint_overlay()]);
    let output = stage.execute(&WorkContext::default()).expect("execute");
    let decision: StageDecision = output.data_take().expect("typed decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[test]
fn request_context_round_trip() {
    // Exercise the RouterRequest handoff shape used by other stages so the
    // ctx plumbing in `evaluate` stays coherent.
    let request = RouterRequest {
        model: "local".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text("show me the report".into()),
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
    let stage = stage_with(vec![hint_overlay()]);
    let decision = stage.evaluate(&ctx, &[]).expect("decision");
    assert_eq!(decision.verdict, StageVerdict::Skipped);
}

#[tokio::test]
async fn overlay_concurrent_scores_all_residuals() {
    let stage = OverlayStage::new(ResidualSelector::new(0.5), vec![hint_overlay()], Some(2));
    let signals: Vec<RoutingSignal> = (0..4).map(|i| signal(&format!("sentence {}", i), Some(0.1))).collect();
    let conf = confidence(spacy_rs::AnnotationSource::ArcEager, 0.1);
    let (msg, decision) = stage.decide_async(Some(&signals), Some(&conf)).await;
    assert_eq!(msg, "overlaid");
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let meta = StageMetadata::from(decision.metadata);
    assert_eq!(meta.overlay_contributions().unwrap().len(), 4);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlay_peak_concurrency_equals_cap() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let in_flight = std::sync::Arc::new(AtomicUsize::new(0));
    let peak = std::sync::Arc::new(AtomicUsize::new(0));
    struct SlowOverlay {
        in_flight: std::sync::Arc<AtomicUsize>,
        peak: std::sync::Arc<AtomicUsize>,
    }
    impl ResidualOverlay for SlowOverlay {
        fn kind(&self) -> ResidualKind { ResidualKind::Disambiguation }
        fn run(&self, _r: &Residual) -> Result<OverlayContribution, OverlayError> {
            let cur = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(cur, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(OverlayContribution { kind: ResidualKind::Disambiguation, score: Some(0.5), payload: serde_json::json!({"route_hints": []}) })
        }
    }
    let overlay: std::sync::Arc<dyn ResidualOverlay> = std::sync::Arc::new(SlowOverlay { in_flight: in_flight.clone(), peak: peak.clone() });
    let stage = OverlayStage::new(ResidualSelector::new(0.5), vec![overlay], Some(2));
    let signals: Vec<RoutingSignal> = (0..8).map(|i| signal(&format!("s {}", i), Some(0.1))).collect();
    let conf = confidence(spacy_rs::AnnotationSource::ArcEager, 0.1);
    let _ = stage.decide_async(Some(&signals), Some(&conf)).await;
    assert_eq!(peak.load(Ordering::SeqCst), 2, "peak should equal cap 2");
}

#[tokio::test]
async fn overlay_single_residual_byte_identical() {
    let stage = OverlayStage::new(ResidualSelector::new(0.5), vec![hint_overlay()], Some(2));
    let signals = vec![signal("show me the report", Some(0.1))];
    let conf = confidence(spacy_rs::AnnotationSource::ArcEager, 0.1);
    let (_, sync_decision) = stage.decide(Some(&signals), Some(&conf));
    let (_, async_decision) = stage.decide_async(Some(&signals), Some(&conf)).await;
    assert_eq!(sync_decision.verdict, async_decision.verdict);
    assert_eq!(sync_decision.metadata, async_decision.metadata);
}
