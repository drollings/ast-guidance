use std::collections::HashMap;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

/// Normalize a raw score to a finite value: non-finite (NaN/±Inf) becomes 0.0.
pub fn normalize_score(raw: f64) -> f64 {
    if raw.is_finite() { raw } else { 0.0 }
}

pub fn is_below_threshold(value: f64, threshold: f64) -> bool {
    value < threshold
}
pub fn is_above_threshold(value: f64, threshold: f64) -> bool {
    value > threshold
}

/// Dot product over the shared prefix (`min(signals, weights)`); empty or
/// disjoint inputs yield `0.0`. Non-finite lanes propagate per IEEE-754
/// (callers that need sanitizing compose `normalize_score` first).
pub fn weighted_dot(signals: &[f64], weights: &[f64]) -> f64 {
    signals
        .iter()
        .zip(weights.iter())
        .map(|(s, w)| s * w)
        .sum()
}

/// Generic scored top-K select (P2 primitive).
///
/// Stable-sorts `items` by `score` (descending when `descending` is true,
/// ascending otherwise), truncates to `k`, and returns the survivors.
/// Ties keep insertion order (`sort_by` is stable — locked in by test).
/// `k == 0` yields empty; `k >= len` returns all, ordered per comparator.
/// NaN scores fall back to `Equal` (today's `unwrap_or(Equal)` idiom).
///
/// Deliberately **no** score-filter parameter: filtering (`> 0.0`,
/// `>= threshold`) stays call-site code so fail-open/filter semantics never
/// move implicitly.
pub fn top_k_by_score<T>(mut items: Vec<T>, k: usize, score: impl Fn(&T) -> f32, descending: bool) -> Vec<T> {
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    if descending {
        items.sort_by(|a, b| {
            score(b).partial_cmp(&score(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        items.sort_by(|a, b| {
            score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    items.truncate(k);
    items
}
pub fn default_coherence_threshold() -> f64 {
    0.5
}
pub fn default_safety_threshold() -> f64 {
    0.5
}

#[derive(Debug, Clone)]
pub struct ThresholdGate {
    pub coherence: f64,
    pub safety: f64,
}
impl ThresholdGate {
    pub fn new(coherence: f64, safety: f64) -> Self {
        Self { coherence, safety }
    }
    pub fn is_coherent(&self, coherence: f64) -> bool {
        coherence >= self.coherence
    }
    pub fn is_safe(&self, safety: f64) -> bool {
        safety >= self.safety
    }
    pub fn complexity_exceeds(&self, complexity: u8, intelligence: u8) -> bool {
        complexity > intelligence
    }
}
impl Default for ThresholdGate {
    fn default() -> Self {
        Self {
            coherence: default_coherence_threshold(),
            safety: default_safety_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteBands<D>
where
    D: Eq + Hash + Clone,
{
    pub bands: HashMap<D, (f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreMatrix<D>
where
    D: Eq + Hash + Clone,
{
    pub dimensions: Vec<D>,
    pub weights: Vec<f64>,
    pub routes: HashMap<String, RouteBands<D>>,
}

#[derive(Debug, Clone)]
pub struct ScoredRoute<D>
where
    D: Clone,
{
    pub route_name: String,
    pub weighted_score: f64,
    pub score_vector: Vec<(D, f64)>,
}

impl<D> ScoreMatrix<D>
where
    D: Eq + Hash + Clone,
{
    pub fn from_parts(dimensions: Vec<D>, weights: Vec<f64>, routes: HashMap<String, RouteBands<D>>) -> Self {
        Self { dimensions, weights, routes }
    }

    pub fn resolve(&self, scores: &HashMap<D, f64>) -> Vec<ScoredRoute<D>>
    where
        D: Clone,
    {
        let mut results = Vec::new();
        for (route_name, bands) in &self.routes {
            let mut route_vector = Vec::new();
            let mut total = 0.0;
            let mut matches_all = true;

            for (i, dim) in self.dimensions.iter().enumerate() {
                let raw_score = scores.get(dim).copied().unwrap_or(0.0);
                let score = normalize_score(raw_score);
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

        results.sort_by(|a, b| {
            b.weighted_score
                .partial_cmp(&a.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

