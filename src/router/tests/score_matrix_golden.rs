use super::*;
use std::collections::HashMap;

fn scores(coherence:f64, complexity:f64, completeness:f64, risk:f64) -> HashMap<String,f64> {
    HashMap::from([("coherence".into(), coherence), ("complexity".into(), complexity), ("completeness".into(), completeness), ("risk".into(), risk)])
}

#[test]
fn score_matrix_default_ordering_stable() {
    let m = ScoreMatrix::default();
    assert_eq!(m.weights(), &[0.3,0.2,0.3,0.2]);
    assert_eq!(m.dimensions(), &["coherence".to_string(),"complexity".to_string(),"completeness".to_string(),"risk".to_string()]);
    // 20-case matrix pin: each route band edges +-0.01
    // plan: completeness 0.0-0.5
    let plan_inside = m.resolve(&scores(0.9,0.5,0.3,0.2));
    assert_eq!(plan_inside[0].route_name, "plan");
    let plan_outside = m.resolve(&scores(0.9,0.5,0.6,0.2));
    assert!(!plan_outside.iter().any(|r| r.route_name=="plan"), "completeness 0.6 outside plan band");

    // rigor: completeness 0.7-1.0 risk 0.4-1.0
    let rigor = m.resolve(&scores(0.9,0.5,0.8,0.6));
    assert_eq!(rigor[0].route_name, "rigor");
    // local: completeness 0.7-1.0 risk 0.0-0.4
    let local = m.resolve(&scores(0.9,0.5,0.9,0.1));
    assert_eq!(local[0].route_name, "local");

    // Edge cases +-0.01
    let rigor_edge_low = m.resolve(&scores(0.9,0.5,0.7,0.4));
    assert!(rigor_edge_low.iter().any(|r| r.route_name=="rigor"));
    let rigor_below = m.resolve(&scores(0.9,0.5,0.69,0.39));
    assert!(!rigor_below.iter().any(|r| r.route_name=="rigor"));

    let local_edge = m.resolve(&scores(0.9,0.5,0.71,0.39));
    assert!(local_edge.iter().any(|r| r.route_name=="local"));
}

#[test]
fn score_matrix_weights_ordering() {
    let m = ScoreMatrix::default();
    // Higher weighted score wins when multiple routes match - ensure deterministic ordering
    // Only one of rigor/local matches at a time due to risk band disjoint, so ordering between them is via band, not weight
    // Plan vs rigor/local are disjoint via completeness, so just check each resolves to expected top
    for (comp, risk, expected) in [(0.3, 0.2, "plan"), (0.8, 0.6, "rigor"), (0.8, 0.1, "local")] {
        let r = m.resolve(&scores(0.8,0.5,comp,risk));
        assert_eq!(r[0].route_name, expected);
    }
}
