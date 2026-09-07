//! The classification-tree engine walk: evaluates a [`ClassificationTree`]
//! recursively (filter / classifier / terminal / fallback nodes) and produces
//! the final classifier `StageDecision`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use fluent_concurrency::pool::Limiter;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_wvr::prelude::*;
use regex::Regex;

use crate::config::classification::InterlinguaMatch;
use crate::config::filters::FilterOutcome;
use crate::config::{ClassificationNode, ClassificationTree, RoutingConfig};
use crate::pipeline::RoutingTarget;
use crate::stages::classifier::ClassifierBackendResolver;
use crate::pipeline_types::{StageDecision, StageVerdict};
use crate::target_match::{AssessmentRecord, GroupExpansion, TargetMatcher};

use super::decisions::{fallback_child, final_decision, node_decision, terminal_decision, TreeEvaluation, TreeOutcome};
use super::verdict::{parse_tree_verdict, TreeClassifierVerdict};

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
    /// Late-bound backend resolution shared with the owning classifier stage.
    /// `Some` re-resolves each node key per request; a miss falls back to
    /// the boot-built `clients` map and `default_client` unchanged. `None`
    /// (injected mock path) serves the boot-built backends exactly as before.
    backend_resolver: Option<ClassifierBackendResolver>,
    /// Bounds concurrent classifier LLM calls (same primitive as the flat
    /// stage).
    limiter: Arc<Limiter>,
    /// Coherence threshold for classifier nodes that don't set their own.
    default_coherence_threshold: f64,
    /// The target-matching ladder, shared with the flat classifier path
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
        backend_resolver: Option<ClassifierBackendResolver>,
    ) -> Self {
        Self {
            tree,
            routing,
            default_client,
            clients,
            backend_resolver,
            limiter,
            default_coherence_threshold,
            target_matcher,
        }
    }

    /// Evaluate the whole tree against the user message and produce the final
    /// classifier `StageDecision`. `interlingua` is the `NlpStage` handoff
    /// (ROADMAP §14.6, C6): when present, `Filter` nodes with
    /// `match_interlingua` dispatch deterministically on the parse's ids.
    /// `route_hints` is the overlay stage's handoff (ROADMAP_20260827_ORT
    /// §2.6): scored route recommendations appended to classifier-node LLM
    /// context as deterministic routing context.
    pub fn evaluate(
        &self,
        user_text: &str,
        interlingua: Option<&[spacy_rs::routing::InterlinguaSignal]>,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
    ) -> Result<TreeEvaluation, WorkError> {
        self.evaluate_with_expansion(user_text, interlingua, route_hints, &GroupExpansion::default())
    }

    /// [`Self::evaluate`] with the request's availability view for
    /// group-member sentinel expansion. Callers on the serving path (which
    /// carry a request context) pass the context-derived expansion; all other
    /// callers use `evaluate` and get unexpanded behavior.
    pub fn evaluate_with_expansion(
        &self,
        user_text: &str,
        interlingua: Option<&[spacy_rs::routing::InterlinguaSignal]>,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
        expansion: &GroupExpansion,
    ) -> Result<TreeEvaluation, WorkError> {
        let mut visited: Vec<StageDecision> = Vec::new();
        let siblings = HashMap::new();
        let outcome = self.evaluate_node(
            &self.tree.root,
            &siblings,
            user_text,
            interlingua,
            route_hints,
            None,
            &mut visited,
            expansion,
        )?;

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

        let handoff = final_decision(outcome, visited);
        Ok(TreeEvaluation {
            decision: handoff.decision,
            target: handoff.target,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_node(
        &self,
        node: &ClassificationNode,
        siblings: &HashMap<String, ClassificationNode>,
        user_text: &str,
        interlingua: Option<&[spacy_rs::routing::InterlinguaSignal]>,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
        complexity: Option<u8>,
        visited: &mut Vec<StageDecision>,
        expansion: &GroupExpansion,
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
                interlingua,
                route_hints,
                visited,
                expansion,
            ),
            ClassificationNode::Terminal {
                route,
                group,
                description,
                ..
            } => Ok(self.evaluate_terminal(
                route,
                group.as_deref(),
                description,
                complexity,
                user_text,
                visited,
                expansion,
            )),
            ClassificationNode::Filter {
                description,
                patterns,
                outcome,
                redirect_to,
                match_interlingua,
            } => self.evaluate_filter(
                description,
                patterns,
                match_interlingua.as_ref(),
                outcome,
                redirect_to.as_deref(),
                siblings,
                user_text,
                interlingua,
                route_hints,
                complexity,
                visited,
                expansion,
            ),
            ClassificationNode::Fallback { node, .. } => self.evaluate_node(
                node,
                siblings,
                user_text,
                interlingua,
                route_hints,
                complexity,
                visited,
                expansion,
            ),
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
        interlingua: Option<&[spacy_rs::routing::InterlinguaSignal]>,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
        visited: &mut Vec<StageDecision>,
        expansion: &GroupExpansion,
    ) -> Result<TreeOutcome, WorkError> {
        // Deterministic filter children short-circuit before any LLM call.
        let siblings: HashMap<String, ClassificationNode> = children
            .iter()
            .map(|c| (c.key.clone(), c.node.clone()))
            .collect();
        for child in children {
            if let ClassificationNode::Filter { .. } = child.node {
                match self.evaluate_node(
                    &child.node,
                    &siblings,
                    user_text,
                    interlingua,
                    route_hints,
                    None,
                    visited,
                    expansion,
                )? {
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

        let verdict = match self.call_classifier(model, &prompt, user_text, route_hints) {
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
                    return self.evaluate_node(&fb.node, &siblings, user_text, interlingua, route_hints, None, visited, expansion);
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
                    interlingua,
                    route_hints,
                    Some(verdict.complexity),
                    visited,
                    expansion,
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
                interlingua,
                route_hints,
                Some(verdict.complexity),
                visited,
                expansion,
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
        expansion: &GroupExpansion,
    ) -> TreeOutcome {
        // Strict resolution: a terminal names an explicit route. Unknown route
        // names must not silently divert to the default route — `resolve_route`
        // would fall back; check the flat map first.
        if self.routing.routes.contains_key(route) {
            if let Some((rt, assessments)) =
                self.resolve_route_with_matcher(route, complexity, user_text, expansion)
            {
                visited.push(terminal_decision(description, &rt, complexity, assessments));
                return TreeOutcome::Route(Box::new(rt));
            }
        }

        // The terminal carries its own `group`: resolve directly through the
        // group's models (the flat routes map has no entry for this route).
        if let Some(group) = group {
            if let Some((rt, assessments)) =
                self.resolve_group_target(route, group, complexity, user_text, expansion)
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
        expansion: &GroupExpansion,
    ) -> Option<(RoutingTarget, Option<Vec<AssessmentRecord>>)> {
        if let Some(matcher) = &self.target_matcher {
            if let Some(group) = self.routing.route_group(route) {
                let candidates = crate::target_match::expanded_candidates_for_group(
                    &self.routing,
                    group,
                    expansion.recency(),
                    &|base| expansion.supervisor_loaded(base),
                );
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
        expansion: &GroupExpansion,
    ) -> Option<(RoutingTarget, Option<Vec<AssessmentRecord>>)> {
        if let Some(matcher) = &self.target_matcher {
            let candidates = crate::target_match::expanded_candidates_for_group(
                &self.routing,
                group,
                expansion.recency(),
                &|base| expansion.supervisor_loaded(base),
            );
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

        // Static fallback: role members fan out to candidate keys;
        // availability sentinels have no meaning here (no recency/liveness)
        // and unknown literals fall out through the entry lookup, as today.
        let model_keys: Vec<String> = self
            .routing
            .role_expanded_members(group)
            .into_iter()
            .filter(|k| k != "last" && k != "any")
            .collect();
        let candidates: Vec<&String> = model_keys
            .iter()
            .filter(|k| {
                self.routing
                    .entry_for_key(k)
                    .is_some_and(|m| m.intelligence >= min_complexity.unwrap_or(0))
            })
            .collect();
        let cheapest = |a: &&String, b: &&String| {
            let ca = self
                .routing
                .entry_for_key(a)
                .map_or(f64::MAX, |m| m.cost_input + m.cost_output);
            let cb = self
                .routing
                .entry_for_key(b)
                .map_or(f64::MAX, |m| m.cost_input + m.cost_output);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        };
        let entry_key = if candidates.is_empty() {
            model_keys.iter().min_by(cheapest)?
        } else {
            candidates.into_iter().min_by(cheapest)?
        };
        let mut rt = self.routing.target_for_key(entry_key)?;
        rt.group = Some(group.to_string());
        rt.target_name = Some(route.to_string());
        Some((rt, None))
    }

    /// Filter node: deterministic short-circuit over the user message. With
    /// `match_interlingua` set, dispatches on the request's parsed ids instead
    /// of regexes (ROADMAP §14.6, C6) — same phrasing → same ids → same route,
    /// zero tokens. When `interlingua` is `None` (NLP stage absent) an
    /// interlingua filter is a non-match → Pass (graceful degradation,
    /// identical to an invalid regex).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_filter(
        &self,
        description: &str,
        patterns: &[String],
        match_interlingua: Option<&InterlinguaMatch>,
        outcome: &FilterOutcome,
        redirect_to: Option<&str>,
        siblings: &HashMap<String, ClassificationNode>,
        user_text: &str,
        interlingua: Option<&[spacy_rs::routing::InterlinguaSignal]>,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
        complexity: Option<u8>,
        visited: &mut Vec<StageDecision>,
        expansion: &GroupExpansion,
    ) -> Result<TreeOutcome, WorkError> {
        let matched = if let Some(m) = match_interlingua {
            // Interlingua dispatch: AND of the set fields across any sentence,
            // gated by the sentence confidence floor. `confidence_min` is
            // fail-closed: a sentence whose parse carried no confidence is
            // treated as below the floor (low-confidence parses escalate).
            interlingua.is_some_and(|signals| {
                signals.iter().any(|s| {
                    m.predicate_id.is_none_or(|p| s.predicate_id == Some(p))
                        && m.subject_id.is_none_or(|s_| s.subject_id == Some(s_))
                        && m.object_id.is_none_or(|o| s.direct_object_id == Some(o))
                        && m.confidence_min
                            .is_none_or(|min| s.confidence.unwrap_or(0.0) >= min)
                })
            })
        } else {
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
            patterns.iter().any(|p| {
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
            })
        };
        if !matched {
            visited.push(node_decision(
                "filter",
                description,
                StageVerdict::Passed,
                if match_interlingua.is_some() {
                    "interlingua filter did not match — passing through".into()
                } else {
                    "filter did not match — passing through".into()
                },
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
                        return self.evaluate_node(node, siblings, user_text, interlingua, route_hints, complexity, visited, expansion);
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
    /// `route_hints` (the overlay handoff) is appended to the system message as
    /// deterministic routing context (ROADMAP_20260827_ORT §2.6).
    fn call_classifier(
        &self,
        model: &str,
        prompt: &str,
        user_text: &str,
        route_hints: Option<&[crate::pipeline_types::RouteHint]>,
    ) -> Result<TreeClassifierVerdict, WorkError> {
        let mut system = prompt.to_string();
        if let Some(hints) = route_hints.filter(|h| !h.is_empty()) {
            system.push('\n');
            system.push_str(
                &crate::stages::classifier::ClassifierStage::route_hints_prompt_context(hints),
            );
        }
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system,
            },
            ChatMessage {
                role: "user".into(),
                content: user_text.to_string(),
            },
        ];

        // Late-bound first: a backend registered (or rewritten) after boot
        // wins; otherwise the boot-built per-node map and default serve
        // unchanged (injected mock path and unknown keys behave as before).
        let live = self
            .backend_resolver
            .as_ref()
            .and_then(|resolve| resolve(model));
        let client: &Arc<dyn ChatBackend> = live
            .as_ref()
            .or_else(|| self.clients.get(model))
            .unwrap_or(&self.default_client);
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
pub fn cost(entry: &crate::config::ModelEntry) -> f64 {
    entry.cost_input + entry.cost_output
}