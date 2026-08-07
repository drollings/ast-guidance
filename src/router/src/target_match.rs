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
/// `parse_tree_verdict` (`stages/tree.rs`): direct deserialize fast path, then
/// the shared fence-strip → parse → extract pipeline, then the shared field
/// coercion so string-valued numbers ("7") survive.
pub fn parse_self_assessment(response: &str) -> Result<SelfAssessment, WorkError> {
    if let Ok(v) = serde_json::from_str::<SelfAssessment>(response) {
        return Ok(v);
    }

    let raw = fluent_llm::parse_json_response(response).map_err(|e| {
        WorkError::Execution(format!("self-assessment response was not valid JSON: {e}"))
    })?;
    let mut obj = match raw {
        serde_json::Value::Object(map) => map,
        other => {
            return serde_json::from_value(other).map_err(|e| {
                WorkError::Execution(format!("self-assessment verdict parse error: {e}"))
            })
        }
    };
    coerce_u8(&mut obj, "complexity", 5);
    coerce_string(&mut obj, "reason", "");
    serde_json::from_value(serde_json::Value::Object(obj))
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
            routing.models.get(key).map(|entry| TargetCandidate {
                model_key: key.clone(),
                model_name: entry.name.clone().unwrap_or_else(|| key.clone()),
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
    by_key: HashMap<String, Arc<dyn ChatBackend>>,
    default: Arc<dyn ChatBackend>,
}

impl TargetBackends {
    pub fn new(
        by_key: HashMap<String, Arc<dyn ChatBackend>>,
        default: Arc<dyn ChatBackend>,
    ) -> Self {
        Self { by_key, default }
    }

    /// The backend for `model_key`, falling back to `default` when the key has
    /// no dedicated backend (mock/transcript injection, or an unbuilt model
    /// key). This is the single lookup rule — the matcher and any caller use
    /// exactly this one.
    pub fn get(&self, model_key: &str) -> Arc<dyn ChatBackend> {
        self.by_key
            .get(model_key)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&self.default))
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
        let mut rt = match routing.models.get(&candidate.model_key) {
            Some(entry) => RoutingTarget::from_model_entry(&candidate.model_key, entry),
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
                idle_timeout_ms: common_core::constants::DEFAULT_IDLE_TIMEOUT_MS,
                total_timeout_ms: common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS,
                fallbacks: vec![],
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
            let rt = match routing.models.get(&candidate.model_key) {
                Some(entry) => RoutingTarget::from_model_entry(&candidate.model_key, entry),
                None => continue,
            };
            if seen.insert(rt.model.clone()) {
                result.push(rt);
            }
        }

        for (name, entry) in routing.all_dispatch_targets(route, classifier_complexity) {
            let rt = RoutingTarget::from_model_entry(&name, &entry);
            if seen.insert(rt.model.clone()) {
                result.push(rt);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::{ModelEntry, ModelGroup, RouteRef};
    use crate::test_stubs::StubChatBackend;

    use super::*;

    fn candidate(key: &str, intelligence: u8, cost: f64) -> TargetCandidate {
        TargetCandidate {
            model_key: key.into(),
            model_name: key.into(),
            intelligence,
            cost,
        }
    }

    fn candidates(entries: &[(&str, u8, f64)]) -> Vec<TargetCandidate> {
        entries.iter().map(|(k, i, c)| candidate(k, *i, *c)).collect()
    }

    fn assessment(complexity: u8, reason: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "complexity": complexity,
            "reason": reason,
        }))
        .unwrap()
    }

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
        }
    }

    fn routing_with(group: &str, keys: &[&str]) -> RoutingConfig {
        RoutingConfig {
            routes: HashMap::from([(
                "local".into(),
                RouteRef {
                    group: group.into(),
                    pipelines: vec!["default".into()],
                    description: "local".into(),
                },
            )]),
            models: keys
                .iter()
                .map(|k| (k.to_string(), model_entry(k, 0, 1.0)))
                .collect(),
            model_groups: HashMap::from([(
                group.into(),
                ModelGroup::Array(keys.iter().map(ToString::to_string).collect()),
            )]),
            system_prompt: String::new(),
            safety_threshold: 0.3,
            default_route: "local".into(),
            score_matrix: None,
        }
    }

    // ── Pure selection core ──────────────────────────────────────────────

    #[test]
    fn start_index_with_classifier_complexity() {
        let cands = candidates(&[("swarm", 2, 1.0), ("qwen", 6, 3.0)]);
        // complexity 5 → first member with intelligence >= 5 is qwen (index 1).
        assert_eq!(start_index(&cands, Some(5)), 1);
        // complexity 1 → swarm qualifies at index 0.
        assert_eq!(start_index(&cands, Some(1)), 0);
        // complexity exactly equal → that index.
        assert_eq!(start_index(&cands, Some(6)), 1);
    }

    #[test]
    fn start_index_none_or_unqualified_returns_zero() {
        let cands = candidates(&[("swarm", 2, 1.0), ("qwen", 6, 3.0)]);
        assert_eq!(start_index(&cands, None), 0);
        // No member qualifies (complexity beyond the top intelligence) → 0:
        // the cheapest candidate self-assesses first; the climb proceeds.
        assert_eq!(start_index(&cands, Some(9)), 0);
    }

    #[test]
    fn start_index_empty_group_is_zero() {
        let empty: Vec<TargetCandidate> = vec![];
        assert_eq!(start_index(&empty, Some(3)), 0);
        assert_eq!(start_index(&empty, None), 0);
    }

    #[test]
    fn start_index_skips_weak_cheaper_candidates() {
        // Cost ordering governs within the group: index 0 is cheapest but too
        // weak; the classifier already ruled it out, so the climb starts at
        // the first qualifying member.
        let cands = candidates(&[("tiny", 1, 0.5), ("small", 3, 1.0), ("big", 6, 3.0)]);
        assert_eq!(start_index(&cands, Some(3)), 1);
        assert_eq!(start_index(&cands, Some(2)), 1);
        assert_eq!(start_index(&cands, Some(6)), 2);
    }

    #[test]
    fn is_match_boundary_inclusive() {
        let cand = candidate("qwen", 6, 3.0);
        // assessed == intelligence matches (meets or exceeds).
        assert!(is_match(&cand, 6));
        assert!(is_match(&cand, 5));
        assert!(!is_match(&cand, 7));
    }

    // ── Prompt + parse ───────────────────────────────────────────────────

    #[test]
    fn self_assessment_prompt_embeds_user_text() {
        let prompt = build_self_assessment_prompt("what is 2+2?");
        assert!(prompt.contains("0 (trivial) to 10 (requires the most capable model available)"));
        assert!(prompt.contains("User request: what is 2+2?"));
        assert!(prompt.contains(r#"{"complexity": <integer 0-10>, "reason": "<brief justification>"}"#));
        assert!(prompt.ends_with("Only output JSON, no other text."));
    }

    #[test]
    fn parse_pristine_json() {
        let s = parse_self_assessment(&assessment(4, "simple math")).unwrap();
        assert_eq!(s.complexity, 4);
        assert_eq!(s.reason, "simple math");
    }

    #[test]
    fn parse_fenced_json() {
        let s = parse_self_assessment(&format!("```json\n{}\n```", assessment(7, "complex"))).unwrap();
        assert_eq!(s.complexity, 7);
    }

    #[test]
    fn parse_json_inside_prose() {
        let s = parse_self_assessment(&format!(
            "Here you go: {} Hope that helps!",
            assessment(3, "moderate")
        ))
        .unwrap();
        assert_eq!(s.complexity, 3);
    }

    #[test]
    fn parse_string_number_coerces() {
        let s = parse_self_assessment(r#"{"complexity": "6", "reason": "coerced"}"#).unwrap();
        assert_eq!(s.complexity, 6);
    }

    #[test]
    fn parse_missing_fields_default() {
        let s = parse_self_assessment(r"{}").unwrap();
        assert_eq!(s.complexity, 5, "missing complexity defaults to 5");
        assert_eq!(s.reason, "");
    }

    #[test]
    fn parse_unparseable_is_error() {
        assert!(parse_self_assessment("not json at all").is_err());
        assert!(parse_self_assessment("").is_err());
    }

    // ── Matcher climb ────────────────────────────────────────────────────

    fn matcher_with(default_response: Vec<String>) -> TargetMatcher {
        TargetMatcher::new(
            TargetBackends::new(
                HashMap::new(),
                Arc::new(StubChatBackend::new(default_response)),
            ),
            Arc::new(Limiter::new(4)),
            0,
        )
    }

    #[test]
    fn target_backends_get_prefers_dedicated_then_default() {
        let dedicated: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::always(assessment(2, "dedicated")));
        let default: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::always(assessment(8, "default")));
        let backends = TargetBackends::new(
            HashMap::from([("swarm".to_string(), Arc::clone(&dedicated))]),
            Arc::clone(&default),
        );

        assert!(
            Arc::ptr_eq(&backends.get("swarm"), &dedicated),
            "dedicated backend wins for a mapped key"
        );
        assert!(
            Arc::ptr_eq(&backends.get("qwen3.6-27b"), &default),
            "default backend (injected mock/transcript) serves keys absent from the map"
        );
    }

    #[test]
    fn climb_matches_first_candidate_within_intelligence() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        let matcher = matcher_with(vec![assessment(4, "mid"), assessment(6, "hard")]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, Some(3), "hello")
            .expect("match");
        // start_index(Some(3)) = 1 (qwen3.5-9b, the first member with
        // intelligence >= 3). It self-assesses 4 <= 4 → match.
        assert_eq!(tm.primary.model, "qwen3.5-9b");
        assert_eq!(tm.primary.target_name.as_deref(), Some("local"));
        assert_eq!(tm.primary.group.as_deref(), Some("default"));
        // Exactly 1 self-assessment: swarm skipped by the start index, qwen3.5-9b
        // assessed and matched.
        assert_eq!(tm.assessments.len(), 1);
        let rec = &tm.assessments[0];
        assert_eq!(rec.assessed, Some(4));
        assert!(rec.matched);
    }

    #[test]
    fn climb_escalates_to_more_intelligent_member() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        let matcher = matcher_with(vec![
            assessment(7, "hard for qwen3.5"),
            assessment(6, "ok for 27b"),
        ]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, Some(3), "hello")
            .expect("match");
        // start_index(Some(3)) = 1. qwen3.5-9b self-assesses 7 > 4 → escalate.
        // qwen3.6-27b self-assesses 6 <= 6 → match.
        assert_eq!(tm.primary.model, "qwen3.6-27b");
        assert_eq!(tm.assessments.len(), 2);
        assert_eq!(tm.assessments[0].assessed, Some(7));
        assert!(!tm.assessments[0].matched);
        assert_eq!(tm.assessments[1].assessed, Some(6));
        assert!(tm.assessments[1].matched);
    }

    #[test]
    fn climb_starts_at_classifier_seed() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        let matcher = matcher_with(vec![assessment(1, "easy"), assessment(3, "ok")]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, None, "hello")
            .expect("match");
        // No classifier estimate → start at 0 (swarm). Swarm self-assesses 1 <= 2 → match.
        assert_eq!(tm.primary.model, "swarm");
        assert_eq!(tm.assessments.len(), 1);
    }

    #[test]
    fn parse_failure_escalates_conservatively() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        // swarm's response is unparseable → treated as can't-confirm → escalate.
        let matcher = matcher_with(vec![
            "not json".into(),
            assessment(4, "ok for qwen3.5"),
            assessment(6, "ok for 27b"),
        ]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, None, "hello")
            .expect("match");
        assert_eq!(tm.primary.model, "qwen3.5-9b");
        assert_eq!(tm.assessments.len(), 2);
        assert_eq!(tm.assessments[0].assessed, None);
        assert!(tm.assessments[0].error.is_some());
        assert!(!tm.assessments[0].matched);
    }

    #[test]
    fn llm_error_escalates_conservatively() {
        let cands = candidates(&[("swarm", 2, 1.0), ("qwen3.6-27b", 6, 3.0)]);
        let routing = routing_with("default", &["swarm", "qwen3.6-27b"]);
        // Empty queue → both self-assessment calls fail with NoResponse.
        // Conservative escalation: a candidate that cannot confirm (LLM error)
        // is skipped; the last member still matches (terminate-don't-loop).
        let matcher = matcher_with(vec![]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, None, "hello")
            .expect("match");
        assert_eq!(tm.primary.model, "qwen3.6-27b");
        assert_eq!(tm.assessments.len(), 2);
        assert_eq!(tm.assessments[0].assessed, None);
        assert!(tm.assessments[0].error.is_some());
        assert!(!tm.assessments[0].matched);
    }

    #[test]
    fn last_member_always_matches() {
        let cands = candidates(&[("swarm", 2, 1.0), ("qwen3.6-27b", 6, 3.0)]);
        let routing = routing_with("default", &["swarm", "qwen3.6-27b"]);
        // The last member self-assesses 9 > 6 — still matches (terminate).
        let matcher = matcher_with(vec![assessment(3, "ok for swarm"), assessment(9, "hard")]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, None, "hello")
            .expect("match");
        assert_eq!(tm.primary.model, "qwen3.6-27b");
        assert_eq!(tm.assessments.len(), 2);
        assert!(tm.assessments[1].matched, "last member always matches");
        assert_eq!(tm.assessments[1].assessed, Some(9));
    }

    #[test]
    fn exact_call_count_two_assessments() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        let matcher = matcher_with(vec![assessment(7, "too hard"), assessment(6, "match")]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, Some(3), "hello")
            .expect("match");
        // start index 1 → swarm never self-assesses; exactly 2 calls made.
        assert_eq!(tm.assessments.len(), 2);
        assert_eq!(tm.assessments[0].model_key, "qwen3.5-9b");
        assert_eq!(tm.primary.model, "qwen3.6-27b");
    }

    #[test]
    fn fallbacks_are_group_tail_then_cross_group() {
        let cands = candidates(&[
            ("swarm", 2, 1.0),
            ("qwen3.5-9b", 4, 3.0),
            ("qwen3.6-27b", 6, 5.0),
        ]);
        let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
        let matcher = matcher_with(vec![assessment(1, "easy")]);

        let tm = matcher
            .match_target("local", "default", &routing, &cands, None, "hello")
            .expect("match");
        assert_eq!(tm.primary.model, "swarm");
        // Fallback tail = the more-intelligent members of the group, in order.
        let fb_models: Vec<&str> = tm.primary.fallbacks.iter().map(|f| f.model.as_str()).collect();
        assert_eq!(fb_models, vec!["qwen3.5-9b", "qwen3.6-27b"]);
    }

    #[test]
    fn empty_candidates_is_none() {
        let routing = routing_with("default", &[]);
        let matcher = matcher_with(vec![]);
        assert!(
            matcher
                .match_target("local", "default", &routing, &[], Some(3), "hello")
                .is_none()
        );
    }
}
