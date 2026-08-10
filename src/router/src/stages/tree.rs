//! Classification-tree engine
//!
//! Evaluates a [`ClassificationTree`] recursively:
//!
//! - `filter` nodes short-circuit deterministically (`hard_reject` /
//!   `soft_redirect` / `output_filter`),
//! - `classifier` nodes auto-build their prompt from their children (key +
//!   description) and the three-axis JSON schema, call the injected backend,
//!   enforce coherence/safety thresholds, and pick a child,
//! - `terminal` nodes resolve a [`RoutingTarget`] through
//!   `RoutingConfig::resolve_route` (complexity-based model selection),
//! - `fallback` children are evaluated when a classifier picks no named child
//!   or its LLM call fails.
//!
//! Every visited node emits a `StageDecision` (the final one carries the
//! `routing_target` / rejection for the pipeline handoff) and a durable audit
//! record via `audit::emit` with `kind = "tree_node"`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use fluent_concurrency::pool::Limiter;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_wvr::prelude::*;
use regex::Regex;

use crate::config::filters::FilterOutcome;
use crate::config::{ClassificationNode, ClassificationTree, RoutingConfig};
use crate::pipeline::RoutingTarget;
use crate::pipeline_types::{PipelineStage, StageDecision, StageMetadata, StageVerdict};
use crate::stages::common::{coerce_float, coerce_string, coerce_u8};
use crate::target_match::{candidates_for_group, AssessmentRecord, TargetMatcher};

/// The three-axis verdict a classifier node's LLM call must return.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TreeClassifierVerdict {
    /// The child key to route to. `None`/empty → the classifier's fallback
    /// child (or a rejection when there is none).
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default = "default_one")]
    pub coherence: f64,
    #[serde(default = "default_one")]
    pub safety: f64,
    #[serde(default = "default_five")]
    pub complexity: u8,
    #[serde(default)]
    pub reason: String,
}

fn default_one() -> f64 {
    1.0
}

fn default_five() -> u8 {
    5
}

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

pub struct ClassificationEngine {
    tree: ClassificationTree,
    routing: RoutingConfig,
    /// Backend used for every classifier node that has no dedicated entry in
    /// `clients` (mock/transcript injection, or the root classifier client).
    default_client: Arc<dyn ChatBackend>,
    /// Per-node backends keyed by `classifier` node `model` key (real mode
    /// only), so a sub-classifier on a different model dispatches to its own
    /// endpoint.
    clients: HashMap<String, Arc<dyn ChatBackend>>,
    /// Bounds concurrent classifier LLM calls (same primitive as the flat
    /// stage).
    limiter: Arc<Limiter>,
    /// Coherence threshold for classifier nodes that don't set their own.
    default_coherence_threshold: f64,
    /// The M3 target-matching ladder, shared with the flat classifier path
    /// (DRY — one climbing implementation). `Some` (pipeline `target_match:
    /// "self_assess"`) resolves terminals with 2+ member groups through
    /// per-candidate self-assessment; `None` keeps the static
    /// cheapest-qualifying pick.
    target_matcher: Option<TargetMatcher>,
}

impl ClassificationEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tree: ClassificationTree,
        routing: RoutingConfig,
        default_client: Arc<dyn ChatBackend>,
        clients: HashMap<String, Arc<dyn ChatBackend>>,
        limiter: Arc<Limiter>,
        default_coherence_threshold: f64,
        target_matcher: Option<TargetMatcher>,
    ) -> Self {
        Self {
            tree,
            routing,
            default_client,
            clients,
            limiter,
            default_coherence_threshold,
            target_matcher,
        }
    }

    /// Evaluate the whole tree against the user message and produce the final
    /// classifier `StageDecision`.
    pub fn evaluate(&self, user_text: &str) -> Result<TreeEvaluation, WorkError> {
        let mut visited: Vec<StageDecision> = Vec::new();
        let siblings = HashMap::new();
        let outcome =
            self.evaluate_node(&self.tree.root, &siblings, user_text, None, &mut visited)?;

        for d in &visited {
            crate::audit::emit(
                "tree_node",
                serde_json::json!({
                    "node_type": d.metadata.get("node_type"),
                    "verdict": format!("{:?}", d.verdict),
                    "reason": d.reason,
                }),
            );
        }

        let decision = final_decision(outcome, visited);
        Ok(TreeEvaluation { decision })
    }

    fn evaluate_node(
        &self,
        node: &ClassificationNode,
        siblings: &HashMap<String, ClassificationNode>,
        user_text: &str,
        complexity: Option<u8>,
        visited: &mut Vec<StageDecision>,
    ) -> Result<TreeOutcome, WorkError> {
        match node {
            ClassificationNode::Classifier {
                description,
                model,
                coherence_threshold,
                safety_threshold,
                children,
            } => self.evaluate_classifier(
                description,
                model,
                *coherence_threshold,
                *safety_threshold,
                children,
                node,
                user_text,
                visited,
            ),
            ClassificationNode::Terminal {
                route,
                group,
                description,
            } => Ok(self.evaluate_terminal(
                route,
                group.as_deref(),
                description,
                complexity,
                user_text,
                visited,
            )),
            ClassificationNode::Filter {
                description,
                patterns,
                outcome,
                redirect_to,
            } => self.evaluate_filter(
                description,
                patterns,
                outcome,
                redirect_to.as_deref(),
                siblings,
                user_text,
                complexity,
                visited,
            ),
            ClassificationNode::Fallback { node, .. } => {
                self.evaluate_node(node, siblings, user_text, complexity, visited)
            }
        }
    }

    /// Classifier node: run deterministic filter children first (short-circuit),
    /// then an LLM call over the auto-built prompt, thresholds, then pick a child.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_classifier(
        &self,
        description: &str,
        model: &str,
        coherence_threshold: Option<f64>,
        safety_threshold: Option<f64>,
        children: &[crate::config::ClassificationChild],
        node: &ClassificationNode,
        user_text: &str,
        visited: &mut Vec<StageDecision>,
    ) -> Result<TreeOutcome, WorkError> {
        // Deterministic filter children short-circuit before any LLM call.
        let siblings: HashMap<String, ClassificationNode> = children
            .iter()
            .map(|c| (c.key.clone(), c.node.clone()))
            .collect();
        for child in children {
            if let ClassificationNode::Filter { .. } = child.node {
                match self.evaluate_node(&child.node, &siblings, user_text, None, visited)? {
                    TreeOutcome::Pass => {}
                    other => return Ok(other),
                }
            }
        }

        let coherence = coherence_threshold.unwrap_or(self.default_coherence_threshold);
        let safety = safety_threshold.unwrap_or(self.routing.safety_threshold);

        let Some(prompt) = node.build_prompt(coherence, safety) else {
            let reason = "classifier node has no routeable children".to_string();
            visited.push(node_decision(
                "classifier",
                description,
                StageVerdict::Rejected,
                reason.clone(),
                serde_json::json!({ "model": model }),
            ));
            return Ok(TreeOutcome::Reject(reason));
        };

        let verdict = match self.call_classifier(model, &prompt, user_text) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    target: "router.pipeline.stage2.tree",
                    model = %model,
                    error = %e,
                    "tree classifier LLM call failed — falling to fallback/reject",
                );
                if let Some(fb) = fallback_child(children) {
                    visited.push(node_decision(
                        "classifier",
                        description,
                        StageVerdict::Rerouted,
                        format!("classifier LLM error, falling back: {e}"),
                        serde_json::json!({ "model": model, "route": fb.key }),
                    ));
                    return self.evaluate_node(&fb.node, &siblings, user_text, None, visited);
                }
                let reason = format!("classifier LLM error: {e}");
                visited.push(node_decision(
                    "classifier",
                    description,
                    StageVerdict::Rejected,
                    reason.clone(),
                    serde_json::json!({ "model": model }),
                ));
                return Ok(TreeOutcome::Reject(reason));
            }
        };

        // Three-axis gating: incoherent / unsafe queries are rejected before
        // any child is picked (VISION §"Three axes of routing").
        if verdict.coherence < coherence || verdict.safety < safety {
            let reason = if verdict.coherence < coherence {
                format!(
                    "rejected: coherence {:.2} below threshold {:.2}",
                    verdict.coherence, coherence
                )
            } else {
                format!(
                    "rejected: safety {:.2} below threshold {:.2}",
                    verdict.safety, safety
                )
            };
            visited.push(node_decision(
                "classifier",
                description,
                StageVerdict::Rejected,
                reason.clone(),
                serde_json::json!({
                    "model": model,
                    "coherence": verdict.coherence,
                    "safety": verdict.safety,
                    "reason": verdict.reason,
                }),
            ));
            return Ok(TreeOutcome::Reject(reason));
        }

        let route = verdict.route.as_deref().filter(|r| !r.is_empty());
        if let Some(route) = route {
            if let Some(child) = children.iter().find(|c| c.key == route) {
                visited.push(node_decision(
                    "classifier",
                    description,
                    StageVerdict::Rerouted,
                    format!("picked child '{route}'"),
                    serde_json::json!({
                        "model": model,
                        "route": route,
                        "coherence": verdict.coherence,
                        "safety": verdict.safety,
                        "complexity": verdict.complexity,
                        "reason": verdict.reason,
                    }),
                ));
                return self.evaluate_node(
                    &child.node,
                    &siblings,
                    user_text,
                    Some(verdict.complexity),
                    visited,
                );
            }
        }

        // No named child picked (or unknown name): fall back if configured.
        if let Some(fb) = fallback_child(children) {
            visited.push(node_decision(
                "classifier",
                description,
                StageVerdict::Rerouted,
                format!(
                    "no named child picked (route {:?}) — falling back",
                    verdict.route
                ),
                serde_json::json!({
                    "model": model,
                    "route": verdict.route,
                    "coherence": verdict.coherence,
                    "safety": verdict.safety,
                    "complexity": verdict.complexity,
                    "reason": verdict.reason,
                }),
            ));
            return self.evaluate_node(
                &fb.node,
                &siblings,
                user_text,
                Some(verdict.complexity),
                visited,
            );
        }

        let reason = "classifier picked no valid child and no fallback is configured".to_string();
        visited.push(node_decision(
            "classifier",
            description,
            StageVerdict::Rejected,
            reason.clone(),
            serde_json::json!({
                "model": model,
                "route": verdict.route,
                "coherence": verdict.coherence,
                "safety": verdict.safety,
                "reason": verdict.reason,
            }),
        ));
        Ok(TreeOutcome::Reject(reason))
    }

    /// Terminal node: resolve a dispatch target. When the pipeline opts into
    /// target-matching (`target_match: "self_assess"`), a 2+ member group
    /// resolves through the shared self-assessment ladder
    /// (`crate::target_match::TargetMatcher`); otherwise the static
    /// cheapest-qualifying pick runs (`RoutingConfig::routing_target` /
    /// [`Self::resolve_group_target`]).
    fn evaluate_terminal(
        &self,
        route: &str,
        group: Option<&str>,
        description: &str,
        complexity: Option<u8>,
        user_text: &str,
        visited: &mut Vec<StageDecision>,
    ) -> TreeOutcome {
        // Strict resolution: a terminal names an explicit route. Unknown route
        // names must not silently divert to the default route — `resolve_route`
        // would fall back; check the flat map first.
        if self.routing.routes.contains_key(route) {
            if let Some((rt, assessments)) =
                self.resolve_route_with_matcher(route, complexity, user_text)
            {
                visited.push(terminal_decision(description, &rt, complexity, assessments));
                return TreeOutcome::Route(Box::new(rt));
            }
        }

        // The terminal carries its own `group`: resolve directly through the
        // group's models (the flat routes map has no entry for this route).
        if let Some(group) = group {
            if let Some((rt, assessments)) =
                self.resolve_group_target(route, group, complexity, user_text)
            {
                visited.push(terminal_decision(description, &rt, complexity, assessments));
                return TreeOutcome::Route(Box::new(rt));
            }
        }

        let reason = format!("terminal route not resolvable: {route}");
        visited.push(node_decision(
            "terminal",
            description,
            StageVerdict::Rejected,
            reason.clone(),
            serde_json::json!({ "route": route }),
        ));
        TreeOutcome::Reject(reason)
    }

    /// Flat-route terminal resolution through the shared target-matching
    /// ladder when available, falling back to the static
    /// `RoutingConfig::routing_target` (defense-in-depth — never fails harder
    /// than today). Single-member groups skip the ladder entirely.
    fn resolve_route_with_matcher(
        &self,
        route: &str,
        complexity: Option<u8>,
        user_text: &str,
    ) -> Option<(RoutingTarget, Option<Vec<AssessmentRecord>>)> {
        if let Some(matcher) = &self.target_matcher {
            if let Some(group) = self.routing.route_group(route) {
                let candidates = candidates_for_group(&self.routing, group);
                if candidates.len() >= 2 {
                    if let Some(tm) = matcher.match_target(
                        route,
                        group,
                        &self.routing,
                        &candidates,
                        complexity,
                        user_text,
                    ) {
                        return Some((tm.primary, Some(tm.assessments)));
                    }
                }
            }
        }
        self.routing
            .routing_target(route, complexity)
            .map(|rt| (rt, None))
    }

    /// Resolve a terminal's own `model_group` when no flat `routes` entry
    /// exists. When the target-matching ladder is available and the group has
    /// 2+ members, runs the self-assessment climb; otherwise picks the
    /// cheapest model in the group whose `intelligence` meets the request
    /// complexity (else cheapest in the group) — today's static behavior.
    fn resolve_group_target(
        &self,
        route: &str,
        group: &str,
        min_complexity: Option<u8>,
        user_text: &str,
    ) -> Option<(RoutingTarget, Option<Vec<AssessmentRecord>>)> {
        if let Some(matcher) = &self.target_matcher {
            let candidates = candidates_for_group(&self.routing, group);
            if candidates.len() >= 2 {
                if let Some(tm) = matcher.match_target(
                    route,
                    group,
                    &self.routing,
                    &candidates,
                    min_complexity,
                    user_text,
                ) {
                    return Some((tm.primary, Some(tm.assessments)));
                }
            }
        }

        let model_keys = self.routing.model_groups.get(group)?.models();
        let candidates: Vec<&String> = model_keys
            .iter()
            .filter(|k| {
                self.routing
                    .models
                    .get(*k)
                    .is_some_and(|m| m.intelligence >= min_complexity.unwrap_or(0))
            })
            .collect();
        let cheapest = |a: &&String, b: &&String| {
            let ca = self.routing.models.get(*a).map_or(f64::MAX, cost);
            let cb = self.routing.models.get(*b).map_or(f64::MAX, cost);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        };
        let entry_key = if candidates.is_empty() {
            model_keys.iter().min_by(cheapest)?
        } else {
            candidates.into_iter().min_by(cheapest)?
        };
        let entry = self.routing.models.get(entry_key)?;
        let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
        let mut rt = RoutingTarget::from_model_entry(&name, entry);
        rt.group = Some(group.to_string());
        rt.target_name = Some(route.to_string());
        Some((rt, None))
    }

    /// Filter node: deterministic short-circuit over the user message.
    fn evaluate_filter(
        &self,
        description: &str,
        patterns: &[String],
        outcome: &FilterOutcome,
        redirect_to: Option<&str>,
        siblings: &HashMap<String, ClassificationNode>,
        user_text: &str,
        complexity: Option<u8>,
        visited: &mut Vec<StageDecision>,
    ) -> Result<TreeOutcome, WorkError> {
        if patterns.is_empty() {
            visited.push(node_decision(
                "filter",
                description,
                StageVerdict::Passed,
                "filter has no patterns — passing through".into(),
                serde_json::json!({}),
            ));
            return Ok(TreeOutcome::Pass);
        }

        let matched = patterns.iter().any(|p| {
            Regex::new(p).map_or_else(
                |e| {
                    tracing::warn!(
                        target: "router.pipeline.stage2.tree",
                        pattern = %p,
                        error = %e,
                        "invalid filter regex — treated as non-match",
                    );
                    false
                },
                |re| re.is_match(user_text),
            )
        });
        if !matched {
            visited.push(node_decision(
                "filter",
                description,
                StageVerdict::Passed,
                "filter did not match — passing through".into(),
                serde_json::json!({}),
            ));
            return Ok(TreeOutcome::Pass);
        }

        match outcome {
            FilterOutcome::HardReject => {
                let reason = format!("blocked by filter: {description}");
                visited.push(node_decision(
                    "filter",
                    description,
                    StageVerdict::Rejected,
                    reason.clone(),
                    serde_json::json!({ "outcome": "hard_reject" }),
                ));
                Ok(TreeOutcome::Reject(reason))
            }
            FilterOutcome::SoftRedirect => {
                if let Some(target) = redirect_to {
                    if let Some(node) = siblings.get(target) {
                        visited.push(node_decision(
                            "filter",
                            description,
                            StageVerdict::Rerouted,
                            format!("soft redirect to '{target}'"),
                            serde_json::json!({ "outcome": "soft_redirect", "redirect_to": target }),
                        ));
                        return self.evaluate_node(node, siblings, user_text, complexity, visited);
                    }
                    tracing::warn!(
                        target: "router.pipeline.stage2.tree",
                        target = %target,
                        "soft_redirect target not found among classifier children",
                    );
                }
                visited.push(node_decision(
                    "filter",
                    description,
                    StageVerdict::Passed,
                    "soft_redirect target missing — passing through".into(),
                    serde_json::json!({ "outcome": "soft_redirect" }),
                ));
                Ok(TreeOutcome::Pass)
            }
            FilterOutcome::OutputFilter => {
                // The tree records the match and continues; wiring the actual
                // redaction into the eventual dispatch response is post-tree.
                visited.push(node_decision(
                    "filter",
                    description,
                    StageVerdict::Passed,
                    "output_filter matched — flagged for redaction".into(),
                    serde_json::json!({ "outcome": "output_filter" }),
                ));
                Ok(TreeOutcome::Pass)
            }
        }
    }

    /// One classifier LLM call: build messages from the auto-constructed
    /// prompt, run through the shared limiter, and parse the three-axis verdict.
    fn call_classifier(
        &self,
        model: &str,
        prompt: &str,
        user_text: &str,
    ) -> Result<TreeClassifierVerdict, WorkError> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: prompt.to_string(),
            },
            ChatMessage {
                role: "user".into(),
                content: user_text.to_string(),
            },
        ];

        let client = self.clients.get(model).unwrap_or(&self.default_client);
        tracing::info!(
            target: "router.pipeline.stage2.tree",
            model = %model,
            input_len = user_text.len(),
            system_prompt_len = prompt.len(),
            "tree classifier LLM request",
        );

        let call_start = Instant::now();
        let mut llm_latency_ms = 0u64;
        let response = self.limiter.run_sync(|| async {
            let llm_start = Instant::now();
            let result = client.chat_complete(&messages);
            llm_latency_ms = llm_start.elapsed().as_millis() as u64;
            result
        });
        let total_latency_ms = call_start.elapsed().as_millis() as u64;
        let limiter_wait_ms = total_latency_ms.saturating_sub(llm_latency_ms);

        let response = match response {
            Ok(r) => {
                tracing::info!(
                    target: "router.pipeline.stage2.tree",
                    model = %model,
                    llm_latency_ms = llm_latency_ms,
                    limiter_wait_ms = limiter_wait_ms,
                    response_len = r.len(),
                    "tree classifier LLM call succeeded",
                );
                r
            }
            Err(e) => {
                return Err(WorkError::Execution(format!(
                    "tree classifier LLM error for model '{model}': {e}"
                )));
            }
        };

        parse_tree_verdict(&response)
    }
}

/// Cost of a model entry for the cheapest-first group resolution.
fn cost(entry: &crate::config::ModelEntry) -> f64 {
    entry.cost_input + entry.cost_output
}

/// Build the final pipeline-handoff `StageDecision` from the tree outcome,
/// embedding every visited node's decision in `metadata.tree_path`.
fn final_decision(outcome: TreeOutcome, visited: Vec<StageDecision>) -> StageDecision {
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

fn fallback_child(
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
fn terminal_decision(
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
fn node_decision(
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

/// Tolerant parse of a classifier node's LLM response into the three-axis
/// verdict: direct deserialize fast path, then the shared fence-strip → parse
/// → extract pipeline, then the shared field coercion.
fn parse_tree_verdict(response: &str) -> Result<TreeClassifierVerdict, WorkError> {
    if let Ok(v) = serde_json::from_str::<TreeClassifierVerdict>(response) {
        return Ok(v);
    }

    let raw = fluent_llm::parse_json_response(response).map_err(|e| {
        WorkError::Execution(format!("tree classifier response was not valid JSON: {e}"))
    })?;
    let mut obj = match raw {
        serde_json::Value::Object(map) => map,
        other => {
            return serde_json::from_value(other).map_err(|e| {
                WorkError::Execution(format!("tree classifier verdict parse error: {e}"))
            })
        }
    };
    coerce_float(&mut obj, "coherence", 1.0);
    coerce_float(&mut obj, "safety", 1.0);
    coerce_u8(&mut obj, "complexity", 5);
    coerce_string(&mut obj, "reason", "");
    serde_json::from_value(serde_json::Value::Object(obj))
        .map_err(|e| WorkError::Execution(format!("tree classifier verdict parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use common_core::sync::lock;
    use fluent_llm::{ChatMessage, LlmError};

    use crate::config::{ModelEntry, ModelGroup, RouteRef, RoutingConfig};
    use crate::pipeline::RoutingTarget;
    use crate::pipeline_types::StageMetadata;
    use crate::target_match::{TargetBackends, TargetMatcher};
    use crate::test_stubs::{CountingBackend, StubChatBackend};

    use super::*;

    fn model_entry(key: &str, intelligence: u8, cost: f64) -> ModelEntry {
        ModelEntry {
            name: Some(key.into()),
            endpoint: "http://localhost:8080/v1/chat/completions".into(),
            intelligence,
            cost_input: cost,
            cost_output: cost * 6.0,
            cost_cached_read: cost * 0.4,
            speed: 8,
            total_timeout_ms: 40_000,
            idle_timeout_ms: 8_000,
            stream: true,
            filter_thinking: true,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: None,
            weights: None,
            hf_repo: None,
            hf_file: None,
        }
    }

    fn test_routing() -> RoutingConfig {
        RoutingConfig {
            routes: HashMap::from([
                (
                    "code".into(),
                    RouteRef {
                        group: "code".into(),
                        pipelines: vec!["default".into()],
                        description: "code".into(),
            always_route: false,
                    },
                ),
                (
                    "translation".into(),
                    RouteRef {
                        group: "translation".into(),
                        pipelines: vec!["default".into()],
                        description: "translation".into(),
            always_route: false,
                    },
                ),
                (
                    "local".into(),
                    RouteRef {
                        group: "question".into(),
                        pipelines: vec!["default".into()],
                        description: "local".into(),
            always_route: false,
                    },
                ),
            ]),
            models: HashMap::from([
                ("fast".into(), model_entry("fast", 1, 1e-6)),
                ("small".into(), model_entry("small", 2, 2e-6)),
                ("code-model".into(), model_entry("code-model", 5, 5e-6)),
                (
                    "translation-model".into(),
                    model_entry("translation-model", 3, 3e-6),
                ),
                (
                    "question-model".into(),
                    model_entry("question-model", 2, 2e-6),
                ),
            ]),
            model_groups: HashMap::from([
                ("code".into(), ModelGroup::Array(vec!["code-model".into()])),
                (
                    "translation".into(),
                    ModelGroup::Array(vec!["translation-model".into()]),
                ),
                (
                    "question".into(),
                    ModelGroup::Array(vec!["question-model".into()]),
                ),
                (
                    "fast".into(),
                    ModelGroup::Array(vec!["fast".into(), "small".into()]),
                ),
            ]),
            system_prompt: String::new(),
            safety_threshold: 0.3,
            default_route: "local".into(),
            score_matrix: None,
        }
    }

    fn engine(tree: &ClassificationTree, backend: Arc<dyn ChatBackend>) -> ClassificationEngine {
        engine_with_matcher(tree, backend, None)
    }

    fn engine_with_matcher(
        tree: &ClassificationTree,
        backend: Arc<dyn ChatBackend>,
        matcher: Option<TargetMatcher>,
    ) -> ClassificationEngine {
        engine_with_routing(tree, backend, test_routing(), matcher)
    }

    fn engine_with_routing(
        tree: &ClassificationTree,
        backend: Arc<dyn ChatBackend>,
        routing: RoutingConfig,
        matcher: Option<TargetMatcher>,
    ) -> ClassificationEngine {
        ClassificationEngine::new(
            tree.clone(),
            routing,
            backend,
            HashMap::new(),
            Arc::new(Limiter::new(4)),
            0.5,
            matcher,
        )
    }

    fn verdict(route: &str, coherence: f64, safety: f64, complexity: u8) -> String {
        serde_json::to_string(&serde_json::json!({
            "route": route,
            "coherence": coherence,
            "safety": safety,
            "complexity": complexity,
            "reason": "test verdict",
        }))
        .unwrap()
    }

    /// A canned self-assessment response for the target-matching ladder.
    fn self_assessment(complexity: u8, reason: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "complexity": complexity,
            "reason": reason,
        }))
        .unwrap()
    }

    /// A ladder matcher whose default backend serves the queued self-assessment
    /// responses (empty per-key map → every candidate routes through the
    /// default, mirroring mock/transcript injection).
    fn ladder_matcher(responses: Vec<String>) -> TargetMatcher {
        TargetMatcher::new(
            TargetBackends::new(
                HashMap::new(),
                Arc::new(StubChatBackend::new(responses)),
            ),
            Arc::new(Limiter::new(4)),
            0,
        )
    }

    fn routed_target(decision: &StageDecision) -> RoutingTarget {
        StageMetadata::from(decision.metadata.clone())
            .routing_target()
            .expect("decision should carry a routing target")
    }

    fn simple_tree() -> ClassificationTree {
        serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "request router",
                "model": "fast",
                "children": [
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    },
                    {
                        "key": "translation",
                        "description": "translation",
                        "node": { "type": "terminal", "route": "translation", "group": "translation" }
                    },
                    {
                        "key": "general",
                        "description": "everything else",
                        "node": {
                            "type": "fallback",
                            "node": { "type": "terminal", "route": "local", "group": "question" }
                        }
                    }
                ]
            }
        }))
        .unwrap()
    }

    // ── Terminal nodes ─────────────────────────────────────────────────

    #[test]
    fn terminal_node_resolves_route() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "code" }
        }))
        .unwrap();
        let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
        let decision = engine.evaluate("write a rust function").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let rt = routed_target(&decision);
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        assert_eq!(rt.model, "code-model");
        assert_eq!(rt.group.as_deref(), Some("code"));
    }

    #[test]
    fn terminal_complexity_selects_model() {
        // complexity 8 > code-model intelligence 5, so the cheapest model in
        // the group whose intelligence meets it — none — falls back to the
        // cheapest in the group (code-model).
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "code" }
        }))
        .unwrap();
        let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
        let decision = engine.evaluate("complex").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(routed_target(&decision).model, "code-model");
    }

    #[test]
    fn terminal_unresolvable_route_rejects() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "does-not-exist" }
        }))
        .unwrap();
        let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
        let decision = engine.evaluate("hi").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("does-not-exist"));
    }

    #[test]
    fn terminal_with_own_group_resolves_without_flat_route() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "fresh", "group": "fast" }
        }))
        .unwrap();
        let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
        let decision = engine.evaluate("hi").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let rt = routed_target(&decision);
        assert_eq!(rt.target_name.as_deref(), Some("fresh"));
        // Cheapest in "fast" group meeting no-complexity: fast (cost 1e-6 vs small 2e-6).
        assert_eq!(rt.model, "fast");
    }

    #[test]
    fn terminal_group_ladder_self_assesses_and_matches() {
        // The "fast" group has 2 members (fast intelligence 1, small
        // intelligence 2). A root terminal on that group climbs: fast
        // self-assesses above its intelligence, small matches.
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "fresh", "group": "fast" }
        }))
        .unwrap();
        let matcher = ladder_matcher(vec![
            self_assessment(7, "too hard for fast"),
            self_assessment(1, "easy for small"),
        ]);
        let engine = engine_with_matcher(
            &tree,
            Arc::new(StubChatBackend::always("{}")),
            Some(matcher),
        );
        let decision = engine.evaluate("some task").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let rt = routed_target(&decision);
        assert_eq!(
            rt.model, "small",
            "ladder climbs past the too-weak cheap member",
        );
        assert_eq!(rt.target_name.as_deref(), Some("fresh"));
        assert_eq!(rt.group.as_deref(), Some("fast"));

        // The terminal's tree_path audit carries the ladder walk (additive over
        // the existing route/group/model/complexity fields).
        let path = decision.metadata["tree_path"].as_array().expect("tree_path");
        let terminal = path
            .iter()
            .find(|d| d["metadata"]["node_type"] == "terminal")
            .expect("terminal node decision");
        assert_eq!(terminal["metadata"]["matched_via"], "self_assess");
        let assessments = terminal["metadata"]["assessments"]
            .as_array()
            .expect("assessments");
        assert_eq!(assessments.len(), 2);
        assert_eq!(assessments[0]["model_name"], "fast");
        assert_eq!(assessments[0]["assessed"], serde_json::json!(7));
        assert_eq!(assessments[0]["matched"], serde_json::json!(false));
        assert_eq!(assessments[1]["model_name"], "small");
        assert_eq!(assessments[1]["assessed"], serde_json::json!(1));
        assert_eq!(assessments[1]["matched"], serde_json::json!(true));
    }

    #[test]
    fn terminal_flat_route_ladder_matches_within_group() {
        // The route's own group ("code" is a single-member group — static).
        // Use a 2-member group via a flat route: "local" → group "question"
        // is single-member too. Build a flat route on the 2-member "fast"
        // group to exercise the resolve_route_with_matcher path.
        let mut routing = test_routing();
        routing.routes.insert(
            "fresh".into(),
            RouteRef {
                group: "fast".into(),
                pipelines: vec!["default".into()],
                description: "fresh".into(),
            always_route: false,
            },
        );
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "fresh" }
        }))
        .unwrap();

        // fast self-assesses 2 > intelligence 1 → escalate to small, which
        // matches at 2 <= 2.
        let matcher = ladder_matcher(vec![
            self_assessment(2, "above fast"),
            self_assessment(2, "ok for small"),
        ]);
        let engine = engine_with_routing(
            &tree,
            Arc::new(StubChatBackend::always("{}")),
            routing,
            Some(matcher),
        );
        let decision = engine.evaluate("a task").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let rt = routed_target(&decision);
        assert_eq!(rt.model, "small");
        assert_eq!(rt.target_name.as_deref(), Some("fresh"));
        assert_eq!(rt.group.as_deref(), Some("fast"));

        let path = decision.metadata["tree_path"].as_array().expect("tree_path");
        let terminal = path
            .iter()
            .find(|d| d["metadata"]["node_type"] == "terminal")
            .expect("terminal node decision");
        assert_eq!(
            terminal["metadata"]["assessments"].as_array().map(Vec::len),
            Some(2),
        );
    }

    #[test]
    fn terminal_single_member_group_never_self_assesses() {
        // A single-member group ("code") has nothing to climb — the ladder is
        // skipped entirely and no self-assessment call is made, even with a
        // matcher present (byte-identical to today's static pick).
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "code", "group": "code" }
        }))
        .unwrap();
        let counting = Arc::new(CountingBackend::new("{}"));
        let matcher = TargetMatcher::new(
            TargetBackends::new(HashMap::new(), Arc::clone(&counting) as Arc<dyn ChatBackend>),
            Arc::new(Limiter::new(4)),
            0,
        );
        let engine = engine_with_matcher(
            &tree,
            Arc::new(StubChatBackend::always("{}")),
            Some(matcher),
        );
        let decision = engine.evaluate("hello").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(routed_target(&decision).model, "code-model");
        assert_eq!(
            counting.calls(),
            0,
            "single-member group must not run the ladder",
        );
    }

    #[test]
    fn terminal_ladder_assessment_failure_escalates_to_last_member() {
        // The "fast" group: fast's self-assessment is unparseable (conservative
        // escalate), small matches as the last member regardless.
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": { "type": "terminal", "route": "fresh", "group": "fast" }
        }))
        .unwrap();
        let matcher = ladder_matcher(vec![
            "not json at all".into(),
            self_assessment(9, "hard even for small"),
        ]);
        let engine = engine_with_matcher(
            &tree,
            Arc::new(StubChatBackend::always("{}")),
            Some(matcher),
        );
        let decision = engine.evaluate("some task").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(routed_target(&decision).model, "small");

        let path = decision.metadata["tree_path"].as_array().expect("tree_path");
        let terminal = path
            .iter()
            .find(|d| d["metadata"]["node_type"] == "terminal")
            .expect("terminal node decision");
        let assessments = terminal["metadata"]["assessments"]
            .as_array()
            .expect("assessments");
        assert_eq!(assessments[0]["assessed"], serde_json::Value::Null);
        assert!(assessments[0]["error"].as_str().is_some());
        assert_eq!(assessments[0]["matched"], serde_json::json!(false));
        assert_eq!(assessments[1]["matched"], serde_json::json!(true));
    }

    // ── Filter nodes ───────────────────────────────────────────────────

    #[test]
    fn filter_hard_reject_short_circuits() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "blocked",
                        "description": "blocks banned content",
                        "node": {
                            "type": "filter",
                            "patterns": ["\\bharmful pattern\\b"],
                            "outcome": "hard_reject"
                        }
                    },
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let engine = engine(
            &tree,
            Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
        );
        let decision = engine
            .evaluate("this is a harmful pattern test")
            .unwrap()
            .decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("blocked"));
    }

    #[test]
    fn filter_non_match_falls_through_to_llm() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "blocked",
                        "description": "blocks banned content",
                        "node": { "type": "filter", "patterns": ["\\bharmful\\b"], "outcome": "hard_reject" }
                    },
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let engine = engine(
            &tree,
            Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
        );
        let decision = engine.evaluate("write a function").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("code")
        );
    }

    #[test]
    fn filter_soft_redirect_jumps_to_sibling() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "redirect",
                        "description": "always code",
                        "node": {
                            "type": "filter",
                            "patterns": [".*"],
                            "outcome": "soft_redirect",
                            "redirect_to": "code"
                        }
                    },
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
        let decision = engine.evaluate("anything at all").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("code")
        );
    }

    #[test]
    fn filter_output_filter_continues() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "redact",
                        "description": "flag pii",
                        "node": { "type": "filter", "patterns": ["\\d{3}-\\d{2}-\\d{4}"], "outcome": "output_filter" }
                    },
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let engine = engine(
            &tree,
            Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
        );
        let decision = engine
            .evaluate("my ssn is 123-45-6789 and I need code")
            .unwrap()
            .decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("code")
        );
    }

    // ── Classifier nodes ───────────────────────────────────────────────

    #[test]
    fn classifier_picks_child_and_routes() {
        let tree = simple_tree();
        let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3)));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("help me debug rust").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("code")
        );
    }

    #[test]
    fn classifier_threshold_rejects_incoherent_query() {
        let tree = simple_tree();
        let backend = Arc::new(StubChatBackend::always(verdict("code", 0.2, 0.9, 3)));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("asdf qwerty").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("coherence"));
    }

    #[test]
    fn classifier_threshold_rejects_unsafe_query() {
        let tree = simple_tree();
        let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.05, 3)));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("something unsafe").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("safety"));
    }

    #[test]
    fn classifier_unknown_route_falls_back() {
        let tree = simple_tree();
        let backend = Arc::new(StubChatBackend::always(verdict("nonexistent", 0.9, 0.9, 3)));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("hello").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("local")
        );
    }

    #[test]
    fn classifier_llm_failure_falls_back() {
        let tree = simple_tree();
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("hello").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("local")
        );
    }

    #[test]
    fn classifier_no_fallback_rejects_on_llm_failure() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("hello").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("LLM error"));
    }

    #[test]
    fn classifier_empty_route_rejects_when_no_fallback() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    }
                ]
            }
        }))
        .unwrap();
        let backend = Arc::new(StubChatBackend::always(verdict("", 0.9, 0.9, 3)));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("hello").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Rejected);
    }

    // ── Multi-level trees ──────────────────────────────────────────────

    #[test]
    fn multi_level_domain_to_subdomain_to_terminal() {
        let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
            "root": {
                "type": "classifier",
                "description": "domain router",
                "model": "fast",
                "children": [
                    {
                        "key": "code",
                        "description": "programming domain",
                        "node": {
                            "type": "classifier",
                            "description": "code subdomain",
                            "model": "small",
                            "children": [
                                {
                                    "key": "debug",
                                    "description": "debugging help",
                                    "node": { "type": "terminal", "route": "code", "group": "code" }
                                },
                                {
                                    "key": "general",
                                    "description": "general programming",
                                    "node": { "type": "terminal", "route": "code", "group": "code" }
                                }
                            ]
                        }
                    },
                    {
                        "key": "prose",
                        "description": "general questions",
                        "node": { "type": "terminal", "route": "local", "group": "question" }
                    }
                ]
            }
        }))
        .unwrap();
        // Call 1: root picks "code". Call 2: subdomain picks "debug".
        let backend = Arc::new(StubChatBackend::new(vec![
            verdict("code", 0.9, 0.9, 5),
            verdict("debug", 0.9, 0.9, 6),
        ]));
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("my program segfaults").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            routed_target(&decision).target_name.as_deref(),
            Some("code")
        );

        // Both visited node types appear in the tree_path.
        let path = decision.metadata["tree_path"]
            .as_array()
            .expect("tree_path");
        let types: Vec<&str> = path
            .iter()
            .filter_map(|d| d["metadata"]["node_type"].as_str())
            .collect();
        assert!(types.contains(&"classifier"));
        assert!(types.contains(&"terminal"));
        assert!(
            path.len() >= 3,
            "root + sub + terminal decisions, got {path:?}"
        );
    }

    // ── Prompt auto-construction ───────────────────────────────────────

    #[test]
    fn auto_generated_prompt_lists_children() {
        let tree = simple_tree();
        let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3)));
        let engine = engine(&tree, backend);
        let _ = engine.evaluate("hello").unwrap();
        // The prompt is only observable via the audit/log stream; assert the
        // pure `build_prompt` output is what the engine would send.
        let prompt = tree
            .root
            .build_prompt(0.5, 0.3)
            .expect("root classifier prompt");
        assert!(prompt.contains("You are a request router."));
        assert!(prompt.contains("- code: programming"));
        assert!(prompt.contains("- translation: translation"));
        assert!(prompt.contains("\"route\": \"<exactly one of: code, translation>\""));
        assert!(prompt.contains("\"coherence\": 0.0-1.0"));
        assert!(prompt.contains("\"complexity\": 0-10"));
    }

    // ── Prompt capture through the backend ─────────────────────────────

    struct RecordingBackend {
        prompts: Arc<Mutex<Vec<String>>>,
        response: String,
    }

    impl ChatBackend for RecordingBackend {
        fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
            lock(&self.prompts).extend(
                messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.clone()),
            );
            Ok(self.response.clone())
        }
    }

    #[test]
    fn engine_sends_auto_generated_prompt_to_backend() {
        let tree = simple_tree();
        let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
        let backend: Arc<dyn ChatBackend> = Arc::new(RecordingBackend {
            prompts: prompts.clone(),
            response: verdict("code", 0.9, 0.9, 3),
        });
        let engine = engine(&tree, backend);
        let decision = engine.evaluate("write code").unwrap().decision;
        assert_eq!(decision.verdict, StageVerdict::Passed);

        let captured = lock(&prompts).clone();
        assert_eq!(captured.len(), 1, "exactly one classifier call");
        assert!(captured[0].contains("You are a request router."));
        assert!(captured[0].contains("- code: programming"));
        assert!(captured[0].contains("- translation: translation"));
        assert!(
            captured[0].contains("\"route\": \"<exactly one of: code, translation>\""),
            "three-axis route enum, got: {}",
            captured[0]
        );
    }
}
