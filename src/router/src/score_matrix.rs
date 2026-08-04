use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreMatrix {
    pub dimensions: Vec<String>,
    pub weights: Vec<f64>,
    pub routes: HashMap<String, RouteBands>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteBands {
    pub bands: HashMap<String, (f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct ScoredRoute {
    pub route_name: String,
    pub weighted_score: f64,
    pub score_vector: Vec<(String, f64)>,
}

impl ScoreMatrix {
    pub fn resolve(&self, scores: &HashMap<String, f64>) -> Vec<ScoredRoute> {
        let mut results = Vec::new();
        for (route_name, bands) in &self.routes {
            let mut route_vector = Vec::new();
            let mut total = 0.0;
            let mut matches_all = true;

            for (i, dim) in self.dimensions.iter().enumerate() {
                // Non-finite scores (NaN/±Inf from a malformed classifier
                // response) carry no ranking signal — treat them as no-match
                // (zero contribution) so they can never poison `total` and
                // panic the sort below.
                let raw_score = scores.get(dim).copied().unwrap_or(0.0);
                let score = if raw_score.is_finite() {
                    raw_score
                } else {
                    0.0
                };
                let weight = self.weights.get(i).copied().unwrap_or(0.0);
                route_vector.push((dim.clone(), score));

                if let Some(&(min, max)) = bands.bands.get(dim) {
                    if score < min || score > max {
                        matches_all = false;
                        break;
                    }
                }
                total += score * weight;
            }

            if matches_all {
                results.push(ScoredRoute {
                    route_name: route_name.clone(),
                    weighted_score: total,
                    score_vector: route_vector,
                });
            }
        }

        // `total` is always finite after the per-score guard above; the
        // fallback ordering also protects against a non-finite config weight.
        results.sort_by(|a, b| {
            b.weighted_score
                .partial_cmp(&a.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

impl Default for ScoreMatrix {
    fn default() -> Self {
        Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn non_finite_scores_do_not_panic_and_treat_as_no_match() {
        let matrix = ScoreMatrix::default();
        // NaN coherence + Infinity completeness: both carry no signal and are
        // treated as zero, so the ranking still resolves deterministically.
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
}
