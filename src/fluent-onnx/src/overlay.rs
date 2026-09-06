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

// The neutral consumer types live in `fluent_llm::backend` (imported, not
// re-exported — consumers name the `fluent_llm` paths directly).
// Implementations (`PromptRouterOverlay`, builders) stay here. The import is
// gated: without the `onnx` feature only the hermetic decode tests (which
// name these types through the parent module) need it.
#[cfg(any(feature = "onnx", test))]
use fluent_llm::backend::{OverlayContribution, Residual, ResidualKind};
#[cfg(feature = "onnx")]
use fluent_llm::backend::{OverlayError, ResidualOverlay, RouteLabel};

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
