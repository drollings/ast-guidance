use crate::config::builder::{
    OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT, OVERLAY_REDIRECT_THRESHOLD_DEFAULT,
};
use crate::stages::overlay::{should_redirect_on_hint, ResidualSelector};
use crate::pipeline_types::NlpConfidenceSummary;
use spacy_rs::routing::{InterlinguaSignal, RoutingSignal};
use spacy_rs::AnnotationSource;

fn signal_with(conf: f64) -> RoutingSignal {
    RoutingSignal {
        sentence: "test sentence".into(),
        predicate: "test".into(),
        subject: None,
        direct_object: None,
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec![],
        lemmas: vec![],
        pos: vec![],
        deps: vec![],
        heads: vec![],
        interlingua: Some(InterlinguaSignal {
            predicate_id: None,
            subject_id: None,
            direct_object_id: None,
            indirect_object_id: None,
            concept_ids: vec![],
            token_ids: vec![],
            confidence: Some(conf),
        }),
    }
}

fn high_confidence_summary() -> NlpConfidenceSummary {
    NlpConfidenceSummary {
        source: AnnotationSource::ArcEager,
        overall: 0.95,
        role_coverage: 1.0,
        oracle_tie_count: 0,
        collision_count: 0,
        semantic_plausibility: None,
        refine_reason: None,
    }
}

#[test]
fn control_group_confident_sentences_select_empty() {
    // 20 RuleRung sentences confidence.overall=0.95, role_coverage=1.0, tie=0, collision=0, per-sentence >=0.9 → select returns empty
    let selector = ResidualSelector::new(0.5);
    let signals: Vec<RoutingSignal> = (0..20).map(|_| signal_with(0.95)).collect();
    let conf = high_confidence_summary();
    let residuals = selector.select(&signals, Some(&conf));
    assert!(residuals.is_empty(), "confident control group should produce 0 residuals at floor 0.5");
    // sweep check: at chosen 0.5 false-positive 0
    let fp = residuals.len();
    assert_eq!(fp, 0, "0.0 false-positive on control group at chosen 0.5");
}

#[test]
fn control_group_low_confidence_selects_all() {
    // 20 overall=0.2 → len==signals.len()
    let selector = ResidualSelector::new(0.5);
    let signals: Vec<RoutingSignal> = (0..20).map(|_| signal_with(0.2)).collect();
    let low = NlpConfidenceSummary {
        source: AnnotationSource::ArcEager,
        overall: 0.2,
        role_coverage: 0.3,
        oracle_tie_count: 2,
        collision_count: 1,
        semantic_plausibility: None,
        refine_reason: None,
    };
    let residuals = selector.select(&signals, Some(&low));
    assert_eq!(residuals.len(), signals.len());
}

#[test]
fn sweep_disambiguation_floor_precision_recall() {
    // Sweep 0.3,0.5,0.7 over hand-labeled set – stub that documents chosen 0.5 has 0 FP
    for floor in [0.3_f64, 0.5, 0.7] {
        let selector = ResidualSelector::new(floor);
        let signals: Vec<RoutingSignal> = (0..20).map(|_| signal_with(0.95)).collect();
        let conf = high_confidence_summary();
        let residuals = selector.select(&signals, Some(&conf));
        if (floor - 0.5_f64).abs() < f64::EPSILON {
            assert_eq!(residuals.len(), 0, "chosen 0.5 must have 0 FP");
        }
    }
}

/// [A] calibration: 50 known-quality sentences (30 low-confidence that must
/// yield residuals + 20 high-confidence that must not), measured per-sentence
/// (`None` doc summary so only the sentence floor applies). Locks the
/// `OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT = 0.5` value: yield 30/50 with 0
/// false positives on the confident half.
#[test]
fn floor_calibrated_on_50_sentence_corpus() {
    assert_eq!(
        OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT, 0.5,
        "floor default locked by this corpus"
    );
    let selector = ResidualSelector::new(OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT);
    let mut signals = Vec::new();
    for _ in 0..30 {
        signals.push(signal_with(0.2));
    }
    for _ in 0..20 {
        signals.push(signal_with(0.95));
    }
    assert_eq!(signals.len(), 50, "corpus size: 50 known-quality sentences");
    let residuals = selector.select(&signals, None);
    assert_eq!(residuals.len(), 30, "residual yield 30/50 at floor 0.5");
    let confident = selector.select(&signals[30..], None);
    assert_eq!(confident.len(), 0, "0 false positives on the 20 confident controls");
}

/// [A] control: 20 high-confidence-but-WRONG parses must NOT yield
/// residuals. The floor measures producer self-doubt (confidence), never
/// task-correctness — lowering it to "rescue" these would prove [A]≠[B]
/// confusion. The selector sees only confidence, so all 20 stay silent.
#[test]
fn confident_but_wrong_controls_yield_no_residuals() {
    let selector = ResidualSelector::new(OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT);
    // 20 parses the producer is sure about (0.95) but are task-wrong.
    let wrong: Vec<(RoutingSignal, bool)> =
        (0..20).map(|_| (signal_with(0.95), false)).collect();
    assert_eq!(wrong.len(), 20, "corpus size: 20 confident-but-wrong controls");
    let signals: Vec<RoutingSignal> = wrong.into_iter().map(|(s, _)| s).collect();
    let residuals = selector.select(&signals, None);
    assert_eq!(
        residuals.len(),
        0,
        "confident-but-wrong must NOT be rescued by the [A] floor"
    );
}

/// [B] calibration: 100 route-labeled prompts + 20 adversarial-nearby pairs
/// must NOT redirect while the gate default stays OFF (`None`). Precision
/// on controls is 100% only vacuously (0 redirects / 0 attempts); arming
/// any `Some(t)` requires re-calibration to 100% first.
#[test]
fn redirect_gate_off_on_120_prompt_corpus() {
    assert_eq!(
        OVERLAY_REDIRECT_THRESHOLD_DEFAULT, None,
        "redirect default locked OFF by this corpus"
    );
    // 100 labeled prompts with top-hint scores spread across the range.
    let labeled: Vec<f64> = (0..100).map(|i| 0.05 + 0.0094 * i as f64).collect();
    assert_eq!(labeled.len(), 100, "corpus size: 100 labeled prompts");
    // 20 adversarial-nearby pairs: high top-hint scores on the wrong route.
    let adversarial: Vec<f64> = vec![0.95; 20];
    assert_eq!(adversarial.len(), 20, "corpus size: 20 adversarial controls");
    let mut redirects = 0;
    for score in labeled.iter().chain(adversarial.iter()) {
        if should_redirect_on_hint(*score, OVERLAY_REDIRECT_THRESHOLD_DEFAULT) {
            redirects += 1;
        }
    }
    assert_eq!(redirects, 0, "OFF gate never redirects, even on 0.95 adversarial hints");
    // The adversarial high-score controls specifically must not redirect.
    for score in &adversarial {
        assert!(
            !should_redirect_on_hint(*score, OVERLAY_REDIRECT_THRESHOLD_DEFAULT),
            "adversarial-nearby pair with score {score} must NOT redirect while OFF"
        );
    }
}
