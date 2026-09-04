use common_core::score::*;
use std::collections::HashMap;


fn default_matrix() -> ScoreMatrix<String> {
        ScoreMatrix {
            dimensions: vec![
                "coherence".into(),
                "complexity".into(),
                "completeness".into(),
                "risk".into(),
            ],
            weights: vec![0.3, 0.2, 0.3, 0.2],
            routes: {
                let mut routes = HashMap::new();
                routes.insert(
                    "plan".into(),
                    RouteBands {
                        bands: {
                            let mut b = HashMap::new();
                            b.insert("completeness".into(), (0.0, 0.5));
                            b
                        },
                    },
                );
                routes.insert(
                    "rigor".into(),
                    RouteBands {
                        bands: {
                            let mut b = HashMap::new();
                            b.insert("completeness".into(), (0.7, 1.0));
                            b.insert("risk".into(), (0.4, 1.0));
                            b
                        },
                    },
                );
                routes.insert(
                    "local".into(),
                    RouteBands {
                        bands: {
                            let mut b = HashMap::new();
                            b.insert("completeness".into(), (0.7, 1.0));
                            b.insert("risk".into(), (0.0, 0.4));
                            b
                        },
                    },
                );
                routes
            },
        }
}

#[test]
fn plan_route_low_completeness() {
        let matrix = default_matrix();
        let scores = HashMap::from([
            ("coherence".to_string(), 0.8),
            ("complexity".to_string(), 0.5),
            ("completeness".to_string(), 0.3),
            ("risk".to_string(), 0.2),
        ]);
        let results = matrix.resolve(&scores);
        assert_eq!(results[0].route_name, "plan");
}

#[test]
fn rigor_route_high_risk() {
        let matrix = default_matrix();
        let scores = HashMap::from([
            ("coherence".to_string(), 0.9),
            ("complexity".to_string(), 0.7),
            ("completeness".to_string(), 0.8),
            ("risk".to_string(), 0.6),
        ]);
        let results = matrix.resolve(&scores);
        assert_eq!(results[0].route_name, "rigor");
}

#[test]
fn local_route_low_risk_high_completeness() {
        let matrix = default_matrix();
        let scores = HashMap::from([
            ("coherence".to_string(), 0.9),
            ("complexity".to_string(), 0.3),
            ("completeness".to_string(), 0.9),
            ("risk".to_string(), 0.1),
        ]);
        let results = matrix.resolve(&scores);
        assert_eq!(results[0].route_name, "local");
}

#[test]
fn non_finite_scores_do_not_panic_and_treat_as_no_match() {
        let matrix = default_matrix();
        let scores = HashMap::from([
            ("coherence".to_string(), f64::NAN),
            ("complexity".to_string(), f64::INFINITY),
            ("completeness".to_string(), 0.9),
            ("risk".to_string(), 0.1),
        ]);
        let results = matrix.resolve(&scores);
        assert!(!results.is_empty());
        for r in &results {
            assert!(
                r.weighted_score.is_finite(),
                "weighted score must stay finite, got {}",
                r.weighted_score
            );
        }
}

#[test]
fn serde_round_trip() {
        let matrix = default_matrix();
        let json = serde_json::to_string(&matrix).unwrap();
        let back: ScoreMatrix<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(matrix.dimensions, back.dimensions);
        assert_eq!(matrix.weights, back.weights);
        assert_eq!(matrix.routes.len(), back.routes.len());
}

#[test]
fn normalize_score_guards() {
        assert_eq!(normalize_score(f64::NAN), 0.0);
        assert_eq!(normalize_score(f64::INFINITY), 0.0);
        assert_eq!(normalize_score(f64::NEG_INFINITY), 0.0);
        assert_eq!(normalize_score(0.5), 0.5);
}

#[test]
fn weighted_dot_matches_manual() {
        assert_eq!(weighted_dot(&[1.0, 2.0, 3.0], &[0.5, 0.25, 0.1]), 1.3);
        assert_eq!(weighted_dot(&[], &[]), 0.0);
        assert_eq!(weighted_dot(&[], &[1.0]), 0.0);
        // Unequal lengths: dot over the shared prefix.
        assert_eq!(weighted_dot(&[1.0, 2.0, 3.0], &[2.0]), 2.0);
        assert_eq!(weighted_dot(&[1.0], &[2.0, 3.0]), 2.0);
        // NaN propagates (callers sanitize via normalize_score first).
        assert!(weighted_dot(&[f64::NAN], &[1.0]).is_nan());
}

#[test]
fn top_k_empty_and_zero_k() {
        let empty: Vec<(u32, f32)> = Vec::new();
        assert!(top_k_by_score(empty, 3, |t| t.1, true).is_empty());
        assert!(top_k_by_score(vec![(1u32, 0.5f32)], 0, |t| t.1, true).is_empty());
}

#[test]
fn top_k_desc_and_asc() {
        let items = vec![(1u32, 0.2f32), (2, 0.9), (3, 0.5)];
        let desc = top_k_by_score(items.clone(), 2, |t| t.1, true);
        assert_eq!(desc.iter().map(|t| t.0).collect::<Vec<_>>(), vec![2, 3]);
        let asc = top_k_by_score(items, 2, |t| t.1, false);
        assert_eq!(asc.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3]);
}

#[test]
fn top_k_over_len_returns_all_ordered() {
        let items = vec![(1u32, 0.3f32), (2, 0.1)];
        let out = top_k_by_score(items, 9, |t| t.1, true);
        assert_eq!(out.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 2]);
}

#[test]
fn top_k_ties_keep_insertion_order() {
        // Stable sort: equal scores stay in insertion order (locked in —
        // today's call sites rely on insertion-order tie-breaks).
        let items = vec![(1u32, 0.5f32), (2, 0.5), (3, 0.5)];
        let out = top_k_by_score(items, 3, |t| t.1, true);
        assert_eq!(out.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 2, 3]);
}

#[test]
fn top_k_nan_scores_do_not_panic() {
        let items = vec![(1u32, f32::NAN), (2, 0.5f32), (3, f32::NAN)];
        let out = top_k_by_score(items, 3, |t| t.1, true);
        assert_eq!(out.len(), 3);
}
