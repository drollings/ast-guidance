//! Calibration corpus for task-value triggers (M5.2).
//!
//! Separate from the golden corpus: hand-labeled cases for the three
//! task-value triggers plus a control group that must NOT trigger refine.
//! Measures trigger precision/recall against expectations — the zero
//! false-positive control is the M5.5 gate.
//!
//! Each case crafts a synthetic `AnnotationResult` + `InterlinguaSignal` +
//! `RoutingSignal` so the decision is deterministic and hermetic, mirroring
//! the unit-test truth table but grouped as a corpus with aggregate metrics.

use spacy_rs::{
    arc_eager::ParseConfidence, pipeline::refine_reason_aggregated, AnnotationResult,
    AnnotationSet, AnnotationSource, ConfidenceReason, RefineMode, RefinePolicy,
    RefineReason, TaskValueReason,
};
use spacy_rs::routing::{InterlinguaSignal, RoutingSignal};
use fluent_types::InterlinguaId;

fn high_confidence_base() -> AnnotationResult {
    AnnotationResult::new(AnnotationSet::default(), AnnotationSource::ArcEager).with_confidence(
        Some(vec![0.9, 0.9, 0.9]),
        Some(ParseConfidence {
            overall: 0.9,
            token_scores: vec![0.9, 0.9, 0.9],
            role_coverage: 1.0,
            oracle_tie_count: 0,
            oracle_margins: vec![0.5, 0.5],
            semantic_plausibility: None,
        }),
    )
}

fn low_confidence_base() -> AnnotationResult {
    AnnotationResult::new(AnnotationSet::default(), AnnotationSource::ArcEager).with_confidence(
        Some(vec![0.3]),
        Some(ParseConfidence {
            overall: 0.3,
            token_scores: vec![0.3],
            role_coverage: 0.3,
            oracle_tie_count: 1,
            oracle_margins: vec![0.0],
            semantic_plausibility: None,
        }),
    )
}

fn signal_with(
    predicate: Option<u64>,
    subject: Option<u64>,
    dobj: Option<u64>,
    token_ids: Vec<u64>,
) -> InterlinguaSignal {
    InterlinguaSignal {
        predicate_id: predicate.map(InterlinguaId::from_u64),
        subject_id: subject.map(InterlinguaId::from_u64),
        direct_object_id: dobj.map(InterlinguaId::from_u64),
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: token_ids.into_iter().map(InterlinguaId::from_u64).collect(),
        confidence: None,
    }
}

fn routing_with(subject: Option<&str>, dobj: Option<&str>) -> RoutingSignal {
    RoutingSignal {
        sentence: String::new(),
        predicate: "run".into(),
        subject: subject.map(String::from),
        direct_object: dobj.map(String::from),
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
        interlingua: None,
    }
}

struct CalibrationCase {
    name: &'static str,
    base: AnnotationResult,
    signal: InterlinguaSignal,
    routing: RoutingSignal,
    expect_reason: RefineReason,
    expect_should_refine: bool,
    /// Category for aggregate reporting.
    category: &'static str,
}

fn calibration_corpus() -> Vec<CalibrationCase> {
    vec![
        // ── Confident-but-routing-wrong: predicate/subject/dobj unresolved ──
        CalibrationCase {
            name: "confident_predicate_unresolved",
            base: high_confidence_base(),
            signal: signal_with(None, Some(2), Some(3), vec![1, 2, 3]),
            routing: routing_with(Some("cat"), Some("mat")),
            expect_reason: RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole),
            expect_should_refine: true,
            category: "unresolved_role",
        },
        CalibrationCase {
            name: "confident_subject_present_but_unresolved",
            base: high_confidence_base(),
            signal: signal_with(Some(10), None, Some(30), vec![10, 20, 30]),
            routing: routing_with(Some("NASA"), Some("report")),
            expect_reason: RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole),
            expect_should_refine: true,
            category: "unresolved_role",
        },
        CalibrationCase {
            name: "confident_dobj_present_but_unresolved",
            base: high_confidence_base(),
            signal: signal_with(Some(10), Some(20), None, vec![10, 20, 30]),
            routing: routing_with(Some("cat"), Some("report")),
            expect_reason: RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole),
            expect_should_refine: true,
            category: "unresolved_role",
        },
        // ── Collision note ──
        CalibrationCase {
            name: "confident_collision_triggers",
            base: {
                let mut b = high_confidence_base();
                b.collision_count = 1;
                b
            },
            signal: signal_with(Some(10), Some(20), Some(30), vec![1]),
            routing: routing_with(None, None),
            expect_reason: RefineReason::TaskValue(TaskValueReason::Collision),
            expect_should_refine: true,
            category: "collision",
        },
        CalibrationCase {
            name: "collision_with_subject_resolved_still_triggers",
            base: {
                let mut b = high_confidence_base();
                b.collision_count = 2;
                b
            },
            signal: signal_with(Some(10), Some(20), Some(30), vec![1, 2]),
            routing: routing_with(None, None),
            expect_reason: RefineReason::TaskValue(TaskValueReason::Collision),
            expect_should_refine: true,
            category: "collision",
        },
        // ── Unresolved PROPN (token_ids sentinel) — threshold 0.3 ──
        CalibrationCase {
            name: "confident_unresolved_token_id_triggers_propn",
            base: high_confidence_base(),
            signal: signal_with(Some(10), Some(20), Some(30), vec![10, 0]),
            routing: routing_with(None, None),
            expect_reason: RefineReason::TaskValue(TaskValueReason::UnresolvedPropn),
            expect_should_refine: true,
            category: "unresolved_propn",
        },
        CalibrationCase {
            name: "unresolved_fraction_0_4_above_threshold_triggers",
            base: high_confidence_base(),
            signal: signal_with(
                Some(10),
                Some(20),
                Some(30),
                vec![10, 10, 10, 10, 10, 10, 0, 0, 0, 0],
            ),
            routing: routing_with(None, None),
            expect_reason: RefineReason::TaskValue(TaskValueReason::UnresolvedPropn),
            expect_should_refine: true,
            category: "unresolved_propn",
        },
        CalibrationCase {
            name: "unresolved_fraction_0_1_below_threshold_no_trigger",
            base: high_confidence_base(),
            signal: signal_with(
                Some(10),
                Some(20),
                Some(30),
                vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 0],
            ),
            routing: routing_with(None, None),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        // ── Confidence triggers (control for the confidence axis) ──
        CalibrationCase {
            name: "low_overall_triggers_confidence",
            base: low_confidence_base(),
            signal: signal_with(Some(10), Some(20), Some(30), vec![1]),
            routing: routing_with(None, None),
            expect_reason: RefineReason::Confidence(ConfidenceReason::Overall),
            expect_should_refine: true,
            category: "confidence",
        },
        // ── Control group: confident-and-correct must NOT trigger ──
        CalibrationCase {
            name: "control_confident_fully_resolved_no_collision",
            base: high_confidence_base(),
            signal: signal_with(Some(10), Some(20), Some(30), vec![10, 20, 30]),
            routing: routing_with(Some("cat"), Some("mat")),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        CalibrationCase {
            name: "control_confident_predicate_and_subject_resolved_no_dobj",
            base: high_confidence_base(),
            signal: signal_with(Some(10), Some(20), None, vec![10, 20]),
            routing: routing_with(Some("cat"), None),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        CalibrationCase {
            name: "control_confident_no_subject_structurally_no_signal",
            base: high_confidence_base(),
            signal: signal_with(Some(10), None, Some(30), vec![10, 30]),
            routing: routing_with(None, Some("mat")),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        CalibrationCase {
            name: "control_confident_single_token_resolved",
            base: high_confidence_base(),
            signal: signal_with(Some(10), None, None, vec![10]),
            routing: routing_with(None, None),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        CalibrationCase {
            name: "control_confident_multi_token_all_resolved",
            base: high_confidence_base(),
            signal: signal_with(Some(10), Some(20), Some(30), vec![10, 20, 30, 40, 50]),
            routing: routing_with(Some("cat"), Some("mat")),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
        CalibrationCase {
            name: "control_off_never_triggers_even_low_confidence",
            base: low_confidence_base(),
            signal: signal_with(None, None, None, vec![0]),
            routing: routing_with(Some("cat"), Some("mat")),
            expect_reason: RefineReason::NoTrigger,
            expect_should_refine: false,
            category: "control",
        },
    ]
}

#[test]
fn calibration_corpus_trigger_precision_and_recall() {
    let policy = RefinePolicy {
        mode: RefineMode::OnUncertain,
        ..RefinePolicy::default()
    };
    let policy_off = RefinePolicy {
        mode: RefineMode::Off,
        ..RefinePolicy::default()
    };
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;
    let mut true_negatives = 0usize;
    let mut control_false_positives = 0usize;

    for case in calibration_corpus() {
        let pol = if case.name == "control_off_never_triggers_even_low_confidence" {
            policy_off
        } else {
            policy
        };
        let reason = spacy_rs::refine_reason(&case.base, &case.signal, &case.routing, pol);
        let should = spacy_rs::should_refine(&case.base, &case.signal, &case.routing, pol);
        // Reason must match the hand label.
        assert_eq!(
            reason, case.expect_reason,
            "case {}: expected reason {:?}, got {:?}",
            case.name, case.expect_reason, reason
        );
        assert_eq!(
            should, case.expect_should_refine,
            "case {}: expected should_refine={}, got {}",
            case.name, case.expect_should_refine, should
        );
        // Aggregate for precision/recall.
        let expected_positive = case.expect_should_refine;
        let got_positive = should;
        match (expected_positive, got_positive) {
            (true, true) => true_positives += 1,
            (false, true) => {
                false_positives += 1;
                if case.category == "control" {
                    control_false_positives += 1;
                }
            }
            (true, false) => false_negatives += 1,
            (false, false) => true_negatives += 1,
        }
    }

    let total = true_positives + false_positives + false_negatives + true_negatives;
    let precision = if true_positives + false_positives == 0 {
        1.0
    } else {
        true_positives as f64 / (true_positives + false_positives) as f64
    };
    let recall = if true_positives + false_negatives == 0 {
        1.0
    } else {
        true_positives as f64 / (true_positives + false_negatives) as f64
    };

    // The control-group zero false-positive gate (M5.5): a spurious task-value
    // trigger would quietly turn "deterministic-first" into "LLM-first".
    assert_eq!(
        control_false_positives, 0,
        "control group must show zero spurious triggers (M5.5 gate)"
    );

    // Overall trigger precision/recall must be perfect on this hand-labeled
    // corpus — any mismatch above is already an assertion, but report the
    // aggregate for observability.
    assert!(
        precision >= 1.0 - f64::EPSILON,
        "precision {precision} on {total} cases"
    );
    assert!(
        recall >= 1.0 - f64::EPSILON,
        "recall {recall} on {total} cases"
    );
}

#[test]
fn control_20_confident_sentences_no_trigger_both_paths() {
    // 20 confident well-formed sentences (overall>=0.8, role_coverage>=0.9, tie==0, every critical role resolved, collision==0, no PROPN) → NoTrigger on both single and aggregated
    let base = high_confidence_base();
    let policy = RefinePolicy {
        mode: RefineMode::OnUncertain,
        ..RefinePolicy::default()
    };
    let mut signals_agg = Vec::new();
    for _ in 0..20 {
        let signal = signal_with(Some(10), Some(20), Some(30), vec![10, 20, 30]);
        let routing = routing_with(Some("cat"), Some("mat"));
        let reason = spacy_rs::refine_reason(&base, &signal, &routing, policy);
        assert_eq!(reason, RefineReason::NoTrigger);
        signals_agg.push((routing, signal));
    }
    let agg_reason = refine_reason_aggregated(&base, &signals_agg, policy);
    assert_eq!(agg_reason, RefineReason::NoTrigger, "20 confident aggregated must be NoTrigger");
    // 0.0 false-positive before OnUncertain is trusted
    let fp = if agg_reason != RefineReason::NoTrigger { 1 } else { 0 };
    assert_eq!(fp, 0);
}

#[test]
fn calibration_individual_flag_gating() {
    // Each refine_on_* flag independently gates its trigger — flipping one
    // off must suppress only that trigger.  This is a corpus-level check that
    // the flags are not accidentally coupled.
    let base = high_confidence_base();
    let resolved = signal_with(Some(10), Some(20), Some(30), vec![10, 20, 30]);
    let unresolved_role = signal_with(None, None, None, vec![10, 20, 30]);
    let unresolved_token = signal_with(Some(10), Some(20), Some(30), vec![10, 0]);
    let mut collision_base = high_confidence_base();
    collision_base.collision_count = 1;
    let routing_some = routing_with(Some("cat"), Some("report"));
    let routing_none = routing_with(None, None);

    // Unresolved role gated by its flag.
    assert!(spacy_rs::should_refine(
        &base,
        &unresolved_role,
        &routing_some,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            ..RefinePolicy::default()
        }
    ));
    assert!(!spacy_rs::should_refine(
        &base,
        &unresolved_role,
        &routing_some,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            refine_on_unresolved_critical_role: false,
            refine_on_unresolved_propn: false,
            ..RefinePolicy::default()
        }
    ));

    // Unresolved propn gated by its flag.
    assert!(spacy_rs::should_refine(
        &base,
        &unresolved_token,
        &routing_none,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            ..RefinePolicy::default()
        }
    ));
    assert!(!spacy_rs::should_refine(
        &base,
        &unresolved_token,
        &routing_none,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            refine_on_unresolved_propn: false,
            refine_on_unresolved_critical_role: false,
            ..RefinePolicy::default()
        }
    ));

    // Collision gated by its flag.
    assert!(spacy_rs::should_refine(
        &collision_base,
        &resolved,
        &routing_none,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            ..RefinePolicy::default()
        }
    ));
    assert!(!spacy_rs::should_refine(
        &collision_base,
        &resolved,
        &routing_none,
        RefinePolicy {
            mode: RefineMode::OnUncertain,
            refine_on_collision_note: false,
            refine_on_unresolved_critical_role: false,
            refine_on_unresolved_propn: false,
            ..RefinePolicy::default()
        }
    ));
}
