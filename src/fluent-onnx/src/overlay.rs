//! Overlay data types and the `ResidualOverlay` seam (ROADMAP_20260827_ORT
//! §2.2).
//!
//! The **overlay plane** is the residual-consultation machinery behind
//! `NlpOrdering::DeterministicFirst`: a deterministic parse leaves *residuals*
//! (low-confidence sentences, PII-shaped spans, unresolved PROPN spans) and a
//! model overlay scores a residual, producing an [`OverlayContribution`]. The
//! router composes `Arc<dyn ResidualOverlay>` implementations; the trait and
//! the data types live here so the router's `OverlayStage` is ort-free.
//!
//! The `PromptRouterOverlay` (feature `onnx`) is the M2 implementation: it
//! scores a disambiguation residual's sentence against its baked route labels
//! and returns route hints the classifier merges as deterministic context.

#[cfg(feature = "onnx")]
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// What kind of parse residual a sentence (or span) carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    /// A sentence whose parse was uncertain below the confidence floor.
    Disambiguation,
    /// A PII-shaped span (M3: PII-Detector / regex pre-filter).
    PiiSpan,
    /// A PROPN span with no resolved entity (M6: entity linking).
    EntityLink,
    /// A parse whose dependency structure wants correction.
    ParseCorrection,
    /// A span worth a concept-level summary.
    ConceptSummary,
}

/// A deterministic parse residual: the sentence/span the deterministic layer
/// was unsure about, plus byte span and structured context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Residual {
    pub kind: ResidualKind,
    /// Optional byte span into the source request text.
    pub span: Option<(usize, usize)>,
    /// The sentence or span text the overlay scores.
    pub text: String,
    /// Structured context the producer attached (e.g. the confidence that
    /// triggered the residual).
    #[serde(default)]
    pub meta: serde_json::Value,
}

impl Residual {
    /// A disambiguation residual over a sentence.
    #[must_use]
    pub fn disambiguation(sentence: impl Into<String>) -> Self {
        Self {
            kind: ResidualKind::Disambiguation,
            span: None,
            text: sentence.into(),
            meta: serde_json::Value::Object(Default::default()),
        }
    }
}

/// The result of running an overlay over a residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayContribution {
    pub kind: ResidualKind,
    /// The overlay's primary score, when the contribution is score-shaped.
    pub score: Option<f64>,
    /// Structured payload (e.g. `{"route_hints": [...]}` for disambiguation).
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// A `ResidualOverlay` consumes residuals of one [`ResidualKind`] and produces
/// contributions. `dyn` at the request boundary — the router composes these
/// behind `Arc<dyn ResidualOverlay>` and runs them under a `Limiter`.
pub trait ResidualOverlay: Send + Sync {
    /// The residual kind this overlay consumes.
    fn kind(&self) -> ResidualKind;

    /// Score the residual. Errors are fail-open at the stage boundary (the
    /// stage skips the contribution, never fails the request).
    fn run(&self, residual: &Residual) -> Result<OverlayContribution, OverlayError>;
}

/// An overlay failure. The router's `OverlayStage` treats every error as
/// skip-and-log (fail-open enrichment, never a gate).
#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("overlay inference failed: {0}")]
    Inference(String),
    #[error("overlay rejected the residual: {0}")]
    Rejected(String),
}

/// A route the disambiguation overlay scores against: its config key and the
/// description the prompt line is built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteLabel {
    pub route: String,
    pub description: String,
}

/// The M2 disambiguation overlay: a `TwoTowerWorker` (Prompt-Router) scoring a
/// residual sentence against the route descriptions it was built with, exposed
/// as route hints in the contribution payload.
#[cfg(feature = "onnx")]
pub struct PromptRouterOverlay {
    worker: Arc<crate::two_tower::TwoTowerWorker>,
    labels: Vec<RouteLabel>,
}

#[cfg(feature = "onnx")]
impl PromptRouterOverlay {
    /// A disambiguation overlay over the worker and its route labels.
    #[must_use]
    pub fn new(worker: Arc<crate::two_tower::TwoTowerWorker>, labels: Vec<RouteLabel>) -> Self {
        Self { worker, labels }
    }

    /// The route labels this overlay scores against (diagnostics/tests).
    #[must_use]
    pub fn labels(&self) -> &[RouteLabel] {
        &self.labels
    }
}

#[cfg(feature = "onnx")]
impl ResidualOverlay for PromptRouterOverlay {
    fn kind(&self) -> ResidualKind {
        ResidualKind::Disambiguation
    }

    fn run(&self, residual: &Residual) -> Result<OverlayContribution, OverlayError> {
        if residual.kind != ResidualKind::Disambiguation {
            return Err(OverlayError::Rejected(format!(
                "PromptRouterOverlay consumes Disambiguation residuals, got {:?}",
                residual.kind
            )));
        }
        let descriptions: Vec<String> = self.labels.iter().map(|l| l.description.clone()).collect();
        let scores = self
            .worker
            .score_labels(&residual.text, &descriptions)
            .map_err(|e| OverlayError::Inference(e.to_string()))?;
        let hints: Vec<serde_json::Value> = self
            .labels
            .iter()
            .zip(scores.iter())
            .map(|(l, s)| serde_json::json!({ "route": l.route, "score": s }))
            .collect();
        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(OverlayContribution {
            kind: ResidualKind::Disambiguation,
            score: if top.is_finite() { Some(top) } else { None },
            payload: serde_json::json!({ "route_hints": hints }),
        })
    }
}

/// Build a `PromptRouterOverlay` for a registered two-tower model (`task:
/// ZeroShotRouting`), or `None` when the model is absent or mis-typed. The
/// router (composition root) supplies the route labels from `RoutingConfig`.
#[cfg(feature = "onnx")]
pub fn build_prompt_router_overlay(
    registry: &crate::session::OrtSessionRegistry,
    model_key: &str,
    labels: Vec<RouteLabel>,
) -> Result<Option<Arc<dyn ResidualOverlay>>, crate::error::OrtError> {
    use crate::config::OnnxTask;
    let Some(config) = registry.config(model_key) else {
        return Ok(None);
    };
    if config.task != OnnxTask::ZeroShotRouting {
        return Ok(None);
    }
    let Some(handle) = registry.ensure_loaded(model_key)? else {
        return Ok(None);
    };
    let worker = crate::two_tower::TwoTowerWorker::from_handle(&handle, &config, model_key)?;
    Ok(Some(Arc::new(PromptRouterOverlay::new(Arc::new(worker), labels))))
}

#[cfg(test)]
#[path = "../tests/overlay.rs"]
mod tests;
