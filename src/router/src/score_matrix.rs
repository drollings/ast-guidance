//! Router-local `String` specialization of `common_core::score::ScoreMatrix`.
//! Generic math lives in `common_core::score`; this file is the only place router
//! routes appear as score dimensions (`plan`/`rigor`/`local`). Do not promote
//! `router_default` to `common-core` — it is domain.
//!
//! Confidence (`coherence`) measures producer self-doubt (well-formedness);
//! `completeness`/`risk`/`complexity` are task-value (outcome fitness) — bands
//! measure task-value, not confidence.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub use common_core::score::{normalize_score, RouteBands, ScoredRoute};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScoreMatrix {
    inner: common_core::score::ScoreMatrix<String>,
}

impl ScoreMatrix {
    pub fn resolve(&self, scores: &HashMap<String, f64>) -> Vec<ScoredRoute<String>> {
        self.inner.resolve(scores)
    }

    pub fn dimensions(&self) -> &[String] { &self.inner.dimensions }
    pub fn weights(&self) -> &[f64] { &self.inner.weights }
    pub fn routes(&self) -> &HashMap<String, RouteBands<String>> { &self.inner.routes }
}

impl ScoreMatrix {
    pub fn router_default() -> Self {
        let mut routes = HashMap::new();
        routes.insert("plan".into(), RouteBands { bands: { let mut b = HashMap::new(); b.insert("completeness".into(), (0.0, 0.5)); b }});
        routes.insert("rigor".into(), RouteBands { bands: { let mut b = HashMap::new(); b.insert("completeness".into(), (0.7, 1.0)); b.insert("risk".into(), (0.4, 1.0)); b }});
        routes.insert("local".into(), RouteBands { bands: { let mut b = HashMap::new(); b.insert("completeness".into(), (0.7, 1.0)); b.insert("risk".into(), (0.0, 0.4)); b }});
        Self { inner: common_core::score::ScoreMatrix { dimensions: vec!["coherence".into(),"complexity".into(),"completeness".into(),"risk".into()], weights: vec![0.3,0.2,0.3,0.2], routes } }
    }
}

impl Default for ScoreMatrix {
    fn default() -> Self { Self::router_default() }
}



#[cfg(test)]
#[path = "../tests/score_matrix.rs"]
mod tests;
#[cfg(test)]
#[path = "../tests/score_matrix_golden.rs"]
mod score_matrix_golden;
