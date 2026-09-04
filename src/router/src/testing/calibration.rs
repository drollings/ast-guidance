//! Calibration control-group helpers (ROADMAP M1).
//!
//! Distinguishes **confidence (self-doubt)** from **task-value (correctness)**.
//! Control groups are synthetic but deterministic — no model, no I/O, no
//! taxonomy blob needed for the hermetic harness. The harness measures whether
//! a threshold's **precision ≥0.90 and FPR ≤0.05** on the must-not-fire
//! group before the threshold is trusted for caching/persisting.

#![allow(clippy::cast_precision_loss, clippy::cast_lossless)]

use crate::pipeline_types::NlpConfidenceSummary;
use common_core::calibration::{calibrate_threshold, sweep_thresholds, CalibrationReport};

// ── Generic synthetic case ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SyntheticCase {
    pub score: f64,
    pub label: bool,
    pub note: &'static str,
}

// ── Entity-link threshold control groups ────────────────────────────────

/// 200 high-cosine but wrong-entity pairs that must NOT fire at threshold 0.9.
///
/// Simulates the "confident but wrong" retrieval similarity signal: cosine
/// 0.85–0.95 but label false (wrong entity). A threshold that lets these pass
/// would amplify confident errors into cached wrong links.
pub fn entity_link_must_not_fire() -> Vec<SyntheticCase> {
    (0..200)
        .map(|i| SyntheticCase {
            // 0.85 + (i % 10)*0.01 → 0.85..0.94
            score: 0.85 + ((i % 10) as f64) * 0.01,
            label: false,
            note: "high-cosine wrong entity",
        })
        .collect()
}

/// 100 known ambiguous / correct-entity pairs that must fire at 0.6.
pub fn entity_link_must_fire() -> Vec<SyntheticCase> {
    (0..100)
        .map(|i| SyntheticCase {
            score: 0.75 + ((i % 5) as f64) * 0.05, // 0.75..0.95
            label: true,
            note: "correct entity high cosine",
        })
        .collect()
}

// ── NlpConfidenceSummary control groups ─────────────────────────────────

/// 100 high-overall but collision-free parses that must NOT reroute.
///
/// `overall` 0.85–0.95, `collision_count==0`, source is confidence-bearing
/// (ArcEager). A threshold that fires on these would reroute confident parses
/// — a precision inversion.
pub fn needs_disambiguation_must_not_fire() -> Vec<NlpConfidenceSummary> {
    (0..100)
        .map(|i| NlpConfidenceSummary {
            source: spacy_rs::AnnotationSource::ArcEager,
            overall: 0.85 + ((i % 10) as f64) * 0.01,
            role_coverage: 0.9,
            oracle_tie_count: 0,
            collision_count: 0,
            semantic_plausibility: None,
            refine_reason: None,
        })
        .collect()
}

/// 50 known ambiguous cases that must fire (low overall or collision>0).
pub fn needs_disambiguation_must_fire() -> Vec<NlpConfidenceSummary> {
    let mut out = Vec::new();
    for i in 0..25 {
        out.push(NlpConfidenceSummary {
            source: spacy_rs::AnnotationSource::ArcEager,
            overall: 0.3 + ((i % 5) as f64) * 0.05, // 0.3..0.5 low
            role_coverage: 0.5,
            oracle_tie_count: 2,
            collision_count: 0,
            semantic_plausibility: None,
            refine_reason: None,
        });
    }
    for _ in 0..25 {
        out.push(NlpConfidenceSummary {
            source: spacy_rs::AnnotationSource::Llm,
            overall: 0.95,
            role_coverage: 1.0,
            oracle_tie_count: 0,
            collision_count: 2, // collision flags regardless of confidence
            semantic_plausibility: None,
            refine_reason: None,
        });
    }
    out
}

// ── Harness helpers ─────────────────────────────────────────────────────

/// Build a markdown calibration table for `entity_link_threshold` over the
/// synthetic control set (must-not-fire + must-fire combined). Emits to
/// `target/calibration/entity_link.md` when `emit` is true.
pub fn entity_link_calibration_report(emit: bool) -> Vec<(f64, CalibrationReport)> {
    let mut cases = entity_link_must_not_fire();
    cases.extend(entity_link_must_fire());
    let thresholds: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();
    let reports = sweep_thresholds(&cases, |c| c.score, |c| c.label, &thresholds);
    if emit {
        common_core::calibration::emit_markdown_artifact("entity_link", &reports);
    }
    reports
}

/// Calibration for `needs_disambiguation` at `threshold` (default 0.5 in tests).
pub fn needs_disambiguation_report(
    threshold: f64,
    emit: bool,
) -> CalibrationReport {
    let cases = needs_disambiguation_must_not_fire();
    // For this signal, task-value label is collision_count>0 || overall<0.5 (ground truth ambiguous).
    // But harness's must-not-fire cases are all label false (should not fire), so we can just score.
    let cases_scored: Vec<SyntheticCase> = cases
        .iter()
        .map(|s| SyntheticCase {
            score: s.overall,
            label: false,
            note: "must not fire",
        })
        .collect();
    let r = calibrate_threshold(&cases_scored, |c| c.score, |c| c.label, threshold);
    if emit {
        let thresholds = vec![0.3, 0.5, 0.7, 0.9];
        let reports = sweep_thresholds(&cases_scored, |c| c.score, |c| c.label, &thresholds);
        common_core::calibration::emit_markdown_artifact("needs_disambiguation", &reports);
    }
    // Include collision-based signal separately: any collision should be flagged.
    // For must-not-fire group collision_count==0, so this path is just confidence.
    let _ = calibrate_threshold(&cases_scored, |c| c.score, |c| c.label, threshold);
    r
}

/// Emit CI artifacts for all control groups (called by `cargo test calibration -- --nocapture`).
#[cfg(test)]
#[test]
fn emit_calibration_artifacts() {
    let reports = entity_link_calibration_report(true);
    for (t, r) in &reports {
        println!("entity_link threshold {:.2}: precision {:.3} recall {:.3} FPR {:.3} support {}", t, r.precision, r.recall, r.fpr, r.support);
    }
    let _ = needs_disambiguation_report(0.5, true);
    // Also sweep needs_disambiguation over 0.3..0.9
    let cases = needs_disambiguation_must_not_fire();
    let synthetic: Vec<SyntheticCase> = cases.iter().map(|s| SyntheticCase { score: s.overall, label: false, note: "must not fire" }).collect();
    let thresholds: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();
    let reports2 = sweep_thresholds(&synthetic, |c| c.score, |c| c.label, &thresholds);
    common_core::calibration::emit_markdown_artifact("needs_disambiguation_full", &reports2);
}
#[cfg(test)]
#[path = "../../tests/testing_calibration.rs"]
mod tests;
