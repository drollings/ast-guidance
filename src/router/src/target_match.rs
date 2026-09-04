//! Classifier-driven target matching — the in-group complexity ladder.
//!
//! VISION §"Target selection within a group": after the classifier resolves a
//! route to a `model_group`, each candidate target self-assesses the prompt's
//! complexity and defers to the next, more-intelligent member when the
//! assessed complexity exceeds its configured `intelligence`. The first target
//! whose `intelligence` meets or exceeds the assessed complexity — or the last
//! member of the group — actually answers, and its answer is recorded in the
//! session ledger.
//!
//! This module owns the *matching* concern only (§4.1 of the roadmap): the
//! pure selection core (`start_index` / `is_match`), the self-assessment
//! prompt + tolerant parse, and the I/O-bound `TargetMatcher` that runs the
//! climb. It deliberately knows nothing about HTTP dispatch, escalation, or
//! the ledger — those stay in `server/` and `dispatch/`.
//!
//! Both the flat classifier path (`stages::classifier`) and the
//! classification-tree engine (`stages::tree`) resolve through the *same*
//! `TargetMatcher` (DRY — one climbing implementation).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use common_core::registry::ConcurrentRegistry;
use fluent_concurrency::pool::Limiter;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_wvr::prelude::*;

use crate::config::RoutingConfig;
use crate::pipeline::RoutingTarget;
use crate::stages::common::{coerce_string, coerce_u8};

/// A single target's self-assessment of the request's complexity.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SelfAssessment {
    #[serde(default = "default_five")]
    pub complexity: u8,
    #[serde(default)]
    pub reason: String,
}

fn default_five() -> u8 {
    5
}

/// The fixed self-assessment prompt — NOT auto-constructed from children (the
/// self-assessment is not a tree classifier node). §4.2 verbatim.
pub fn build_self_assessment_prompt(user_text: &str) -> String {
    format!(
        "You are asked to rate the complexity of the user's request on a scale of\n\
         0 (trivial) to 10 (requires the most capable model available).\n\n\
         User request: {user_text}\n\n\
         Output exactly one JSON object:\n\
         \x20 {{\"complexity\": <integer 0-10>, \"reason\": \"<brief justification>\"}}\n\
         Only output JSON, no other text."
    )
}

/// Tolerant parse of a target's self-assessment response. Mirrors
/// `parse_tree_verdict` (`stages/tree.rs`) through the shared
/// `fluent_llm::parse_typed` codec: direct deserialize fast path, then the
/// shared fence-strip → parse → extract pipeline, then the shared field
/// coercion so string-valued numbers ("7") survive.
pub fn parse_self_assessment(response: &str) -> Result<SelfAssessment, WorkError> {
    fluent_llm::parse_typed::<SelfAssessment>(
        response,
        &serde_json::Value::Null,
        |v| {
            if let Some(obj) = v.as_object_mut() {
                coerce_u8(obj, "complexity", 5);
                coerce_string(obj, "reason", "");
            }
        },
    )
    .map_err(|e| WorkError::Execution(format!("self-assessment verdict parse error: {e}")))
}

/// One ordered member of a `model_group` the matcher climbs over.
///
/// `model_key` is the config key (`models`/`model_groups`), `model_name` the
/// resolved display name (`ModelEntry.name` or the key when absent).
#[derive(Debug, Clone)]
pub struct TargetCandidate {
    pub model_key: String,
    pub model_name: String,
    pub intelligence: u8,
    /// `cost_input + cost_output` — informational, used to keep the ordered
    /// group's cost ordering visible to callers and tests.
    pub cost: f64,
}

/// Build the ordered `TargetCandidate` list for a `model_group` — the DRY
/// shape both the flat classifier path and the classification-tree engine feed
/// the matcher. Group members are in config order (ascending cost and
/// intelligence, as shipped); members with no `models` entry are skipped
/// (defense-in-depth — the caller falls back to static resolution when the
/// resulting list is empty or single-member).
pub fn candidates_for_group(routing: &RoutingConfig, group: &str) -> Vec<TargetCandidate> {
    let Some(group_cfg) = routing.model_groups.get(group) else {
        return Vec::new();
    };
    group_cfg
        .models()
        .iter()
        .filter_map(|key| {
            routing.entry_for_key(key).map(|entry| TargetCandidate {
                model_key: key.clone(),
                model_name: entry
                    .name
                    .clone()
                    .unwrap_or_else(|| crate::config::split_model_key(key).0.to_string()),
                intelligence: entry.intelligence,
                cost: entry.cost_input + entry.cost_output,
            })
        })
        .collect()
}

/// One assessed step of the climb — the audit/observability payload.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AssessmentRecord {
    pub model_key: String,
    pub model_name: String,
    /// `None` when the self-assessment call failed or was unparseable
    /// (conservative: escalate to the next candidate).
    pub assessed: Option<u8>,
    pub reason: String,
    pub error: Option<String>,
    pub matched: bool,
}

/// The outcome of a ladder run: the matched primary target plus the full walk.
#[derive(Debug)]
pub struct TargetMatch {
    pub primary: RoutingTarget,
    pub assessments: Vec<AssessmentRecord>,
}

/// Step 1 of §4.1: the cheapest index whose `intelligence` meets the
/// classifier's complexity estimate.
///
/// Candidates are in config order (ascending cost and intelligence). Returns
/// `0` when `classifier_complexity` is absent or no member qualifies — the
/// cheapest candidate self-assesses first; the climb never skips a candidate
/// the classifier already ruled out as too weak when a stronger one is cheaper
/// (cost ordering governs within the group).
pub fn start_index(candidates: &[TargetCandidate], classifier_complexity: Option<u8>) -> usize {
    let Some(c) = classifier_complexity else {
        return 0;
    };
    candidates
        .iter()
        .position(|cand| cand.intelligence >= c)
        .unwrap_or(0)
}

/// The match rule for a single assessed candidate (§4.1 step 2c): a candidate
/// matches when its assessed complexity does not exceed its configured
/// `intelligence`. The "last member always matches" escape is enforced by the
/// matcher loop, not here — this stays a pure per-candidate rule.
pub fn is_match(candidate: &TargetCandidate, assessed: u8) -> bool {
    assessed <= candidate.intelligence
}

/// The ladder's per-candidate backend set, built once by `config::builder`
/// (the single `LlmClient` factory — DIP) and shared by every pipeline.
///
/// `by_key` holds one `ChatBackend` per model key referenced by any
/// `model_groups` member. `default` serves any candidate key absent from
/// `by_key`: the injected mock/transcript backend in mock runs (covering every
/// key), or a defense-in-depth real client in real mode — mirroring the
/// classification-tree engine's `default_client` fallback.
#[derive(Clone)]
pub struct TargetBackends {
    by_key: ConcurrentRegistry<String, Arc<dyn ChatBackend>>,
    default: Arc<dyn ChatBackend>,
}

impl TargetBackends {
    pub fn new(
        by_key: HashMap<String, Arc<dyn ChatBackend>>,
        default: Arc<dyn ChatBackend>,
    ) -> Self {
        let reg = ConcurrentRegistry::new();
        for (key, backend) in by_key {
            reg.insert(key, backend);
        }
        Self { by_key: reg, default }
    }

    /// The backend for `model_key`, falling back to `default` when the key has
    /// no dedicated backend (mock/transcript injection, or an unbuilt model
    /// key). This is the single lookup rule — the matcher and any caller use
    /// exactly this one.
    pub fn get(&self, model_key: &str) -> Arc<dyn ChatBackend> {
        self.by_key
            .get(&model_key.to_string())
            .map_or_else(|| Arc::clone(&self.default), |b| b.as_ref().clone())
    }

    /// The number of dedicated per-key backends.
    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    /// Whether no dedicated per-key backends exist (mock/transcript mode, or
    /// a config with no `model_groups` members).
    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

/// The I/O-bound half of the ladder: holds the per-candidate backend set (one
/// `ChatBackend` per candidate model key plus the `default` fallback — the
/// injected mock/transcript backend or a real client) and runs the climb under
/// a shared `Limiter`, exactly like the classifier stage's sync LLM call.
///
/// `Clone` is cheap (three `Arc`-ish fields + a `u64`), so the same matcher
/// can be shared between the flat classifier stage and the classification-tree
/// engine without re-provisioning backends.
#[derive(Clone)]
pub struct TargetMatcher {
    /// Per-model-key backends + the default for keys absent from the map
    /// (real mode: one `LlmClient` per group member, built by `config::builder`
    /// — the single DIP construction site).
    backends: TargetBackends,
    /// Bounds concurrent self-assessment calls (same primitive as the
    /// classifier stage).
    limiter: Arc<Limiter>,
    /// Per-assessment wall-clock budget.
    timeout_ms: u64,
}

impl TargetMatcher {
    pub fn new(backends: TargetBackends, limiter: Arc<Limiter>, timeout_ms: u64) -> Self {
        Self {
            backends,
            limiter,
            timeout_ms,
        }
    }

    /// Run the §4.1 ladder: start index → per-candidate self-assessment →
    /// match rule → tail fallbacks.
    ///
    /// NOTE (M8 exemption): this loop deliberately does NOT compose
    /// `fluent_concurrency::ladder::first_accept_in_order`. It differs in
    /// three load-bearing ways: (1) start-index skip — iteration begins at
    /// `start_index(complexity)`, not rung 0; (2) last-always-wins — the
    /// final candidate matches unconditionally, so exhaustion yields a
    /// match rather than `Ok(None)`; (3) per-rung audit — every
    /// assessment (including non-matching ones) emits an `AssessmentRecord`
    /// plus an audit event, not just the winner.
    ///
    /// `route` is the resolved route name (the primary's `target_name`),
    /// `group` its model group. `candidates` is the group's ordered member
    /// list. `routing` supplies the `ModelEntry`s for fallback construction and
    /// the cross-group `all_dispatch_targets` resilience list.
    pub fn match_target(
        &self,
        route: &str,
        group: &str,
        routing: &RoutingConfig,
        candidates: &[TargetCandidate],
        classifier_complexity: Option<u8>,
        user_text: &str,
    ) -> Option<TargetMatch> {
        if candidates.is_empty() {
            tracing::warn!(
                target: "router.pipeline.stage2.target_match",
                route = %route,
                group = %group,
                "target-matching ladder: empty candidate group",
            );
            return None;
        }

        let start = start_index(candidates, classifier_complexity);
        let mut assessments: Vec<AssessmentRecord> = Vec::with_capacity(candidates.len());
        let mut matched_index = candidates.len() - 1;

        for (i, candidate) in candidates.iter().enumerate().skip(start) {
            let assessed = self.assess(candidate, user_text);
            let reason = assessed
                .as_ref()
                .ok()
                .map(|s| s.reason.clone())
                .unwrap_or_default();

            // Match rule: assessed <= intelligence, OR this is the last member
            // (the ladder always terminates — the last member always matches).
            let matched = i == candidates.len() - 1
                || assessed
                    .as_ref()
                    .is_ok_and(|s| is_match(candidate, s.complexity));

            let record = AssessmentRecord {
                model_key: candidate.model_key.clone(),
                model_name: candidate.model_name.clone(),
                assessed: assessed.as_ref().ok().map(|s| s.complexity),
                reason,
                // Self-assessment failure → conservative escalate.
                error: assessed.as_ref().err().map(ToString::to_string),
                matched,
            };
            crate::audit::emit(
                "target_match",
                serde_json::json!({
                    "stage": "assessment",
                    "route": route,
                    "group": group,
                    "model_key": candidate.model_key,
                    "model_name": candidate.model_name,
                    "intelligence": candidate.intelligence,
                    "assessed": record.assessed,
                    "reason": record.reason,
                    "error": record.error,
                    "matched": record.matched,
                }),
            );
            assessments.push(record);

            if matched {
                matched_index = i;
                break;
            }
        }

        let matched_candidate = &candidates[matched_index];
        let primary = Self::build_primary(matched_candidate, route, group, routing);
        let fallbacks = Self::build_fallbacks(
            routing,
            route,
            classifier_complexity,
            &primary,
            candidates,
            matched_index,
        );

        let mut match_target = primary;
        match_target.fallbacks = fallbacks;

        crate::audit::emit(
            "target_match",
            serde_json::json!({
                "stage": "match",
                "route": route,
                "group": group,
                "matched_model_key": matched_candidate.model_key,
                "matched_model_name": matched_candidate.model_name,
                "assessed": assessments[matched_index - start].assessed,
                "fallback_count": match_target.fallbacks.len(),
            }),
        );

        Some(TargetMatch {
            primary: match_target,
            assessments,
        })
    }

    /// One self-assessment call: build the fixed prompt, run through the
    /// shared limiter (mirroring `stages::classifier`'s sync call), bounded by
    /// `timeout_ms`. A failed/unparseable call is `Err` — the caller escalates
    /// conservatively.
    fn assess(&self, candidate: &TargetCandidate, user_text: &str) -> Result<SelfAssessment, WorkError> {
        let messages = vec![ChatMessage {
            role: "system".into(),
            content: build_self_assessment_prompt(user_text),
        }];
        let client = self.backends.get(&candidate.model_key);

        tracing::info!(
            target: "router.pipeline.stage2.target_match",
            model_key = %candidate.model_key,
            model_name = %candidate.model_name,
            input_len = user_text.len(),
            "target self-assessment request",
        );

        let call_start = std::time::Instant::now();
        let response = self.limiter.run_sync(|| async {
            let call = client.chat_complete(&messages);
            if self.timeout_ms > 0 {
                let timed = tokio::time::timeout(
                    Duration::from_millis(self.timeout_ms),
                    async { call },
                )
                .await;
                if let Ok(r) = timed {
                    r
                } else {
                    tracing::warn!(
                        target: "router.pipeline.stage2.target_match",
                        model_key = %candidate.model_key,
                        timeout_ms = self.timeout_ms,
                        "target self-assessment timed out",
                    );
                    Err(fluent_llm::LlmError::Http(format!(
                        "self-assessment timed out after {}ms",
                        self.timeout_ms
                    )))
                }
            } else {
                call
            }
        });
        let latency_ms = call_start.elapsed().as_millis() as u64;

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    target: "router.pipeline.stage2.target_match",
                    model_key = %candidate.model_key,
                    error = %e,
                    latency_ms = latency_ms,
                    "target self-assessment call failed — escalating to next candidate",
                );
                return Err(WorkError::Execution(e.to_string()));
            }
        };

        match parse_self_assessment(&response) {
            Ok(s) => {
                tracing::info!(
                    target: "router.pipeline.stage2.target_match",
                    model_key = %candidate.model_key,
                    complexity = s.complexity,
                    latency_ms = latency_ms,
                    "target self-assessment succeeded",
                );
                Ok(s)
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.pipeline.stage2.target_match",
                    model_key = %candidate.model_key,
                    error = %e,
                    latency_ms = latency_ms,
                    "target self-assessment unparseable — escalating to next candidate",
                );
                Err(e)
            }
        }
    }

    /// Primary target for the matched candidate (§4.1 step 3): the matched
    /// model's `RoutingTarget` with `group = G` and `target_name = route`.
    fn build_primary(
        candidate: &TargetCandidate,
        route: &str,
        group: &str,
        routing: &RoutingConfig,
    ) -> RoutingTarget {
        let mut rt = match routing.target_for_key(&candidate.model_key) {
            Some(rt) => rt,
            None => RoutingTarget {
                url: String::new(),
                model: candidate.model_name.clone(),
                group: None,
                target_name: Some(candidate.model_key.clone()),
                params: None,
                instance: None,
                snapshot: None,
                id_slot: None,
                filter_thinking: false,
                retry_count: 0,
                retry_base_interval_s: 1,
                stream: true,
                idle_timeout_ms: fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS,
                total_timeout_ms: fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS,
                api_key: None,
                fallbacks: vec![],
                is_onnx: false,
            },
        };
        rt.group = Some(group.to_string());
        rt.target_name = Some(route.to_string());
        rt
    }

    /// Fallback construction (§4.1 step 3): the matched index's more-intelligent
    /// group tail `G[i+1..=n]` as `RoutingTarget`s (mechanical-failure walk, in
    /// order), then any cross-group models from `all_dispatch_targets` not
    /// already included (dedup by model name, preserving today's cross-group
    /// resilience).
    fn build_fallbacks(
        routing: &RoutingConfig,
        route: &str,
        classifier_complexity: Option<u8>,
        primary: &RoutingTarget,
        candidates: &[TargetCandidate],
        matched_index: usize,
    ) -> Vec<RoutingTarget> {
        let mut result: Vec<RoutingTarget> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(primary.model.clone());

        for candidate in candidates.iter().skip(matched_index + 1) {
            let Some(rt) = routing.target_for_key(&candidate.model_key) else {
                continue;
            };
            if seen.insert(rt.model.clone()) {
                result.push(rt);
            }
        }

        for (name, _) in routing.all_dispatch_targets(route, classifier_complexity) {
            let Some(rt) = routing.target_for_key(&name) else {
                continue;
            };
            if seen.insert(rt.model.clone()) {
                result.push(rt);
            }
        }

        result
    }
}

#[cfg(test)]
#[path = "../tests/target_match.rs"]
mod tests;
