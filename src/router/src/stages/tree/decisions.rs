//! Tree evaluation outcome types and the `StageDecision` builders that
//! materialize the `tree_path` audit trail (`node_decision` family) and the
//! final pipeline-handoff decision (`final_decision`).

use crate::config::ClassificationNode;
use crate::pipeline::RoutingTarget;
use crate::pipeline_types::{PipelineStage, StageDecision, StageMetadata, StageVerdict};
use crate::target_match::AssessmentRecord;

/// The outcome of evaluating a node in the tree.
#[derive(Debug)]
pub enum TreeOutcome {
    /// Resolved dispatch target (a `terminal` node, possibly reached via
    /// `soft_redirect`). Boxed to keep the enum small (the target carries
    /// per-model routing fields).
    Route(Box<RoutingTarget>),
    /// The request is rejected with a human-readable reason.
    Reject(String),
    /// The node produced no decision (e.g. a filter with no match); the caller
    /// continues evaluating siblings or falls back.
    Pass,
}

/// The result of a full tree evaluation.
#[derive(Debug)]
pub struct TreeEvaluation {
    /// The final classifier `StageDecision` — carries `routing_target` or a
    /// rejection for the pipeline handoff, plus the full `tree_path` of
    /// visited nodes in its metadata.
    pub decision: StageDecision,
}

/// Build the final pipeline-handoff `StageDecision` from the tree outcome,
/// embedding every visited node's decision in `metadata.tree_path`.
pub fn final_decision(outcome: TreeOutcome, visited: Vec<StageDecision>) -> StageDecision {
    // The final decision's score is the last classifier node's coherence.
    let score = visited.iter().rev().find_map(|d| {
        d.metadata
            .get("coherence")
            .and_then(serde_json::Value::as_f64)
    });

    // Consume `visited` explicitly (`json!` would borrow via `to_value(&..)`).
    let tree_path: serde_json::Value = serde_json::Value::Array(
        visited
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_default(),
    );

    let mut metadata = StageMetadata::new(serde_json::json!({
        "tree": true,
        "tree_path": tree_path,
    }));

    let (verdict, reason) = match outcome {
        TreeOutcome::Route(rt) => {
            metadata.set_routing_target(rt.as_ref());
            let name = rt.target_name.as_deref().unwrap_or(&rt.model);
            (
                StageVerdict::Passed,
                format!("tree routed to {name} (model {})", rt.model),
            )
        }
        TreeOutcome::Reject(reason) => (StageVerdict::Rejected, reason),
        TreeOutcome::Pass => (
            StageVerdict::Rejected,
            "classification tree produced no decision".into(),
        ),
    };

    StageDecision {
        stage: PipelineStage::Classifier,
        verdict,
        score,
        reason,
        latency_ms: 0,
        metadata: metadata.into_value(),
    }
}

pub fn fallback_child(
    children: &[crate::config::ClassificationChild],
) -> Option<&crate::config::ClassificationChild> {
    children
        .iter()
        .find(|c| matches!(c.node, ClassificationNode::Fallback { .. }))
}

/// Build the `tree_path` decision for a resolved terminal. Additive over the
/// existing `route`/`group`/`model`/`complexity` fields: when the
/// target-matching ladder ran, the walk's self-assessment records and a
/// `matched_via` marker are appended (M4 — auditability by construction).
pub fn terminal_decision(
    description: &str,
    rt: &RoutingTarget,
    complexity: Option<u8>,
    assessments: Option<Vec<AssessmentRecord>>,
) -> StageDecision {
    let mut extra = serde_json::json!({
        "route": rt.target_name,
        "group": rt.group,
        "model": rt.model,
        "complexity": complexity,
    });
    if let Some(assessments) = assessments {
        extra["matched_via"] = serde_json::json!("self_assess");
        extra["assessments"] = serde_json::json!(assessments);
    }
    node_decision(
        "terminal",
        description,
        StageVerdict::Passed,
        format!(
            "terminal resolved to route '{}'",
            rt.target_name.as_deref().unwrap_or("?")
        ),
        extra,
    )
}

/// Build a per-node `StageDecision` for the `tree_path` audit trail.
pub fn node_decision(
    node_type: &'static str,
    description: &str,
    verdict: StageVerdict,
    reason: String,
    extra: serde_json::Value,
) -> StageDecision {
    let mut metadata = serde_json::json!({
        "node_type": node_type,
        "node_description": description,
    });
    if let serde_json::Value::Object(map) = extra {
        if let Some(m) = metadata.as_object_mut() {
            for (k, v) in map {
                m.insert(k, v);
            }
        }
    }
    StageDecision {
        stage: PipelineStage::Classifier,
        verdict,
        score: None,
        reason,
        latency_ms: 0,
        metadata,
    }
}
