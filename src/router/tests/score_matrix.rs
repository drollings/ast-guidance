use super::*;

#[test]
fn score_matrix_resolve_is_clone_free() {
    let m = ScoreMatrix::default();
    let scores = HashMap::from([("completeness".into(), 0.8), ("risk".into(), 0.6), ("coherence".into(), 0.9), ("complexity".into(), 0.7)]);
    let a = m.resolve(&scores);
    let b = m.resolve(&scores);
    assert_eq!(a.len(), b.len());
    assert_eq!(a[0].route_name, b[0].route_name);
}

#[test]
fn plan_route_low_completeness() {
    let matrix = ScoreMatrix::default();
    let scores = HashMap::from([
        ("coherence".into(), 0.8),
        ("complexity".into(), 0.5),
        ("completeness".into(), 0.3),
        ("risk".into(), 0.2),
    ]);
    let results = matrix.resolve(&scores);
    assert_eq!(results[0].route_name, "plan");
}

#[test]
fn rigor_route_high_risk() {
    let matrix = ScoreMatrix::default();
    let scores = HashMap::from([
        ("coherence".into(), 0.9),
        ("complexity".into(), 0.7),
        ("completeness".into(), 0.8),
        ("risk".into(), 0.6),
    ]);
    let results = matrix.resolve(&scores);
    assert_eq!(results[0].route_name, "rigor");
}

#[test]
fn local_route_low_risk_high_completeness() {
    let matrix = ScoreMatrix::default();
    let scores = HashMap::from([
        ("coherence".into(), 0.9),
        ("complexity".into(), 0.3),
        ("completeness".into(), 0.9),
        ("risk".into(), 0.1),
    ]);
    let results = matrix.resolve(&scores);
    assert_eq!(results[0].route_name, "local");
}

#[test]
fn score_matrix_default_is_router_default() {
    let a = ScoreMatrix::default();
    let b = ScoreMatrix::router_default();
    assert_eq!(a.dimensions(), b.dimensions());
    assert_eq!(a.weights(), b.weights());
    assert_eq!(a.routes().len(), b.routes().len());
}

#[test]
fn non_finite_scores_do_not_panic_and_treat_as_no_match() {
    let matrix = ScoreMatrix::default();
    let scores = HashMap::from([
        ("coherence".into(), f64::NAN),
        ("complexity".into(), f64::INFINITY),
        ("completeness".into(), 0.9),
        ("risk".into(), 0.1),
    ]);
    let results = matrix.resolve(&scores);
    assert!(!results.is_empty(), "routes should still resolve");
    for r in &results {
        assert!(
            r.weighted_score.is_finite(),
            "weighted score must stay finite, got {}",
            r.weighted_score
        );
    }
}
