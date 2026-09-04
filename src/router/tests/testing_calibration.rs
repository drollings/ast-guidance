use super::*;
use common_core::calibration::calibrate_threshold;

#[test]
fn control_groups_are_nonempty() {
    assert!(!entity_link_must_not_fire().is_empty());
    assert!(!entity_link_must_fire().is_empty());
    assert!(!needs_disambiguation_must_not_fire().is_empty());
    assert!(!needs_disambiguation_must_fire().is_empty());
}

#[test]
fn entity_link_control_must_not_fire_at_0_9() {
    let cases = entity_link_must_not_fire();
    // At threshold 0.9, among 200 high-cosine wrong-entity pairs, none should be labeled true.
    // So if we treat score>=0.9 as "would cache", precision on must-not-fire alone should be:
    // we evaluate with label false => any firing is FP.
    let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.9);
    // label is all false, so TP=0, FP = count where score>=0.9, TN = rest.
    // For must-not-fire alone, precision = 1.0 when FP=0 else 0. With 0.85..0.94, about half fire.
    // The *combined* control with must-fire true cases should have high precision at 0.9 only if
    // must-not-fire doesn't dominate. This test asserts that at 0.9, FP >0 would be visible,
    // so we check that our synthetic distribution indeed has some firing FP — the harness is non-vacuous.
    // The golden assertion is that the *combined* report at 0.6 has precision ≥0.90? For must-not-fire alone
    // at 0.9 we expect FPR >0, so the test must show that the pure must-not-fire group at 0.9 has
    // FPR >0 (i.e., the harness can detect a bad threshold).
    // For the roadmap gate, the relevant check is combined precision at 0.9: we want it to fail if must-not-fire leaks.
    // Here we just assert the group is calibrated to have some high scores above 0.9.
    assert!(r.fp > 0, "must-not-fire should have some FP at 0.9 to be a useful control, got {:?}", r);
    assert!(r.fpr > 0.0);
    // At threshold 0.95, FPR should be lower (monotonic).
    let r2 = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.95);
    assert!(r2.fpr <= r.fpr, "FPR non-increasing with threshold");
}

#[test]
fn entity_link_combined_precision_at_0_9_is_low_without_filter() {
    // Combined must-not-fire (200 FP-prone) + must-fire (100 TP) at 0.9:
    // many must-not-fire still fire, so precision is poor — proving confidence alone is not task-value.
    let mut cases = entity_link_must_not_fire();
    cases.extend(entity_link_must_fire());
    let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.9);
    // With 0.85..0.94 wrong entities, at 0.9 about 50-100 FP, TP ~50-100, precision ~0.5
    assert!(r.precision < 0.90, "combined precision at 0.9 should be <0.90 (confident-but-wrong poisons)",);
    // This is the point of the harness: you cannot trust cosine alone.
}
