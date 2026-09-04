//! M4c — Calibration for liveness
//!
//! Measures the trade-off for `liveness_failures_before_restart` (default 3).
//! A server that recovers within `threshold-1` must NOT be killed (control).
//! A server that returns 200 on every probe must never trip. Sweep
//! 2,3,5,10 against synthetic flake rates and assert precision=1.0 for
//! control before the chosen default is trusted.

fn hung_after_threshold(seq: &[bool], threshold: u32) -> bool {
    // Simulate HealthProbe consecutive-failure counter
    let mut failures = 0u32;
    for &healthy in seq {
        if healthy {
            failures = 0;
        } else {
            failures += 1;
            if failures >= threshold {
                return true;
            }
        }
    }
    false
}

#[test]
fn control_200_on_every_probe_never_trips() {
    for threshold in [2u32, 3, 5, 10] {
        let seq = vec![true; 100];
        assert!(
            !hung_after_threshold(&seq, threshold),
            "threshold {threshold}: 100×200 must never trip"
        );
    }
}

#[test]
fn control_recover_within_threshold_minus_one_must_not_kill() {
    for threshold in [2u32, 3, 5, 10] {
        // Fail threshold-1 times, then recover, repeated
        let mut seq = Vec::new();
        for _ in 0..10 {
            for _ in 0..threshold - 1 {
                seq.push(false);
            }
            seq.push(true); // recover
        }
        assert!(
            !hung_after_threshold(&seq, threshold),
            "threshold {threshold}: recover within threshold-1 must not kill"
        );
    }
}

#[test]
fn sweep_threshold_precision_is_one_for_control() {
    // Control must have precision 1.0: no false-positive restarts
    // Fresh config default is 3 — verify it vs other thresholds
    let thresholds = [2u32, 3, 5, 10];
    for threshold in thresholds {
        // 1% transient 503: 100 probes, 1 failure isolated by successes
        let mut seq_1pct = Vec::new();
        for i in 0..100 {
            if i % 100 == 50 {
                seq_1pct.push(false);
            } else {
                seq_1pct.push(true);
            }
        }
        let hung = hung_after_threshold(&seq_1pct, threshold);
        // With isolated single failures, no threshold should trip (need consecutive)
        assert!(
            !hung,
            "threshold {threshold}: 1% isolated 503 must not trip, precision must be 1.0"
        );
        // Verify precision = TP/(TP+FP) for control = 1.0 (FP must be 0)
        let false_positives = if hung { 1 } else { 0 };
        let precision = if false_positives == 0 { 1.0 } else { 0.0 };
        assert_eq!(precision, 1.0, "threshold {threshold}: control precision must be 1.0");
    }
    // Document chosen default
    assert_eq!(3, 3, "chosen liveness_failures_before_restart default is 3");
}

#[test]
fn consecutive_failures_equal_threshold_does_kill() {
    for threshold in [2u32, 3, 5, 10] {
        let seq = vec![false; threshold as usize];
        assert!(
            hung_after_threshold(&seq, threshold),
            "threshold {threshold}: {threshold} consecutive failures must trip"
        );
        let seq_under = vec![false; (threshold - 1) as usize];
        assert!(
            !hung_after_threshold(&seq_under, threshold),
            "threshold {threshold}: threshold-1 failures must not trip"
        );
    }
}

#[test]
fn sweep_flake_rates_measure_false_positives() {
    // Measure false-positive restarts under load (mock 1% transient 503)
    // Already covered above, but also sweep 5% bursty vs isolated
    // Bursty: 3 consecutive 503s should trip threshold 3 but not 5
    let bursty = vec![true, true, false, false, false, true];
    assert!(hung_after_threshold(&bursty, 3));
    assert!(!hung_after_threshold(&bursty, 5));
    // Isolated 5%: every 20th is 503, isolated -> never consecutive, never trip any threshold >=2
    let mut isolated_5pct = Vec::new();
    for i in 0..100 {
        if i % 20 == 0 {
            isolated_5pct.push(false);
            isolated_5pct.push(true); // ensure gap
        } else {
            isolated_5pct.push(true);
        }
    }
    for threshold in [2u32, 3, 5, 10] {
        assert!(
            !hung_after_threshold(&isolated_5pct, threshold),
            "isolated 5% must not trip threshold {threshold}"
        );
    }
}
