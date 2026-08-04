//! Rubric acceptance gate for chart target outputs (ROADMAP M9).
//!
//! Before a chart target's output is promoted to `provides`, a cheap
//! deterministic field-presence rule must pass; an optional LLM judge is
//! consulted **only when the rubric says so** (`judge_model` set) and a judge
//! backend is available. Failure marks the target failed and cancels
//! dependents rather than propagating bad data (VISION: local-first, frontier
//! as a bounded, audited exception).
//!
//! Deterministic gate semantics mirror the repo's `FieldSchema`/`EntityPredicate`
//! convention: dotted `require_fields` paths resolved against the output value,
//! present and non-null. Substring `pattern` is not part of a rubric — the
//! field-presence rule is intentionally simpler (presence + non-null).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use guidance_llm::client::ChatBackend;
use guidance_llm::ChatMessage;

use super::binding::resolve_path;
use super::{ChartError, ChartRubric};

/// Result of running a rubric gate on a target's output.
#[derive(Debug, Clone)]
pub struct RubricVerdict {
    pub accepted: bool,
    /// Human-readable explanation (deterministic-miss message or judge reason).
    pub reason: String,
    /// Judge score in `[0,1]` when the LLM judge was consulted.
    pub score: Option<f64>,
    /// `true` when the LLM judge was consulted (not just the deterministic gate).
    pub judged: bool,
}

impl RubricVerdict {
    fn deterministic(reason: impl Into<String>) -> Self {
        Self {
            accepted: false,
            reason: reason.into(),
            score: None,
            judged: false,
        }
    }

    fn pass() -> Self {
        Self {
            accepted: true,
            reason: "rubric gate passed".into(),
            score: None,
            judged: false,
        }
    }
}

/// In-memory cache of validated rubric/answer pairs (M9).
///
/// A pair `(rubric, output)` that passed the gate is recorded; a later
/// identical check short-circuits without re-running the deterministic rules
/// or the (expensive) judge. In-memory today; promoted to the `rubric_cache`
/// HNSW/SQLite index when a persistent consumer appears (Consolidation
/// Contract — cross-crate limits stay local until a second consumer).
#[derive(Debug, Default)]
pub struct RubricCache {
    accepted: Mutex<HashMap<u64, ()>>,
}

impl RubricCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(rubric: &ChartRubric, output: &serde_json::Value) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        rubric.require_fields.hash(&mut h);
        rubric.judge_model.hash(&mut h);
        rubric.min_score.to_bits().hash(&mut h);
        output.hash(&mut h);
        h.finish()
    }

    /// Record a rubric/answer pair that passed the gate.
    pub fn record_accepted(&self, rubric: &ChartRubric, output: &serde_json::Value) {
        self.accepted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(Self::key(rubric, output), ());
    }

    /// `true` when this exact rubric/answer pair already passed.
    pub fn is_cached_accepted(&self, rubric: &ChartRubric, output: &serde_json::Value) -> bool {
        self.accepted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&Self::key(rubric, output))
    }

    /// Number of recorded pairs (for tests/metrics).
    pub fn len(&self) -> usize {
        self.accepted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Run the rubric gate on a target's output.
///
/// 1. Consult the `RubricCache` (when provided): an identical pair that
///    already passed short-circuits to an accept.
/// 2. Deterministic field-presence rule: every `require_fields` path must
///    resolve to a present, non-null value. A miss fails immediately — no
///    judge is spent on structurally-invalid output.
/// 3. LLM judge: only when `rubric.judge_model` is set **and** a judge backend
///    is provided. The judge returns a score + acceptance; the output is
///    accepted when the judge accepts **and** `score >= rubric.min_score`.
///    A configured-but-unavailable judge logs a `warn!` and degrades to the
///    deterministic result (the judge is an optional escalation, never a
///    hard dependency).
///
/// `owner` names the gated target/chart for error messages and audit logging.
pub fn check_rubric(
    rubric: &ChartRubric,
    output: &serde_json::Value,
    judge: Option<&Arc<dyn ChatBackend>>,
    cache: Option<&RubricCache>,
    owner: &str,
) -> Result<RubricVerdict, ChartError> {
    if let Some(cache) = cache {
        if cache.is_cached_accepted(rubric, output) {
            return Ok(RubricVerdict {
                accepted: true,
                reason: "rubric gate passed (cached)".into(),
                score: None,
                judged: false,
            });
        }
    }

    if let Err(reason) = deterministic_fields_pass(rubric, output) {
        tracing::warn!(
            target: "router.charts.rubric",
            owner = owner,
            reason = %reason,
            "rubric gate rejected output deterministically"
        );
        return Ok(RubricVerdict::deterministic(reason));
    }

    let judge_needed = rubric.judge_model.is_some();
    let Some(judge_backend) = judge.filter(|_| judge_needed) else {
        if judge_needed {
            tracing::warn!(
                target: "router.charts.rubric",
                owner = owner,
                judge_model = ?rubric.judge_model,
                "rubric declares a judge but none is available — accepting on deterministic gate"
            );
        }
        if let Some(cache) = cache {
            cache.record_accepted(rubric, output);
        }
        return Ok(RubricVerdict::pass());
    };

    match judge_output(judge_backend, rubric, owner, output) {
        Ok(verdict) => {
            if verdict.accepted {
                if let Some(cache) = cache {
                    cache.record_accepted(rubric, output);
                }
            }
            Ok(verdict)
        }
        Err(e) => {
            // Judge failure is not a hard rejection — degrade to the
            // deterministic result (fail closed only on a structural miss).
            tracing::warn!(
                target: "router.charts.rubric",
                owner = owner,
                error = %e,
                "LLM judge failed — accepting on deterministic gate"
            );
            if let Some(cache) = cache {
                cache.record_accepted(rubric, output);
            }
            Ok(RubricVerdict::pass())
        }
    }
}

/// Deterministic field-presence rule: every `require_fields` path must resolve
/// to a present, non-null value.
fn deterministic_fields_pass(
    rubric: &ChartRubric,
    output: &serde_json::Value,
) -> Result<(), String> {
    for path in &rubric.require_fields {
        match resolve_path(output, path) {
            Some(v) if !v.is_null() => {}
            Some(_) => return Err(format!("required field '{path}' is null")),
            None => return Err(format!("output is missing required field '{path}'")),
        }
    }
    Ok(())
}

/// One LLM judge call over the gated output. Returns the parsed verdict.
fn judge_output(
    backend: &Arc<dyn ChatBackend>,
    rubric: &ChartRubric,
    owner: &str,
    output: &serde_json::Value,
) -> Result<RubricVerdict, ChartError> {
    let prompt = format!(
        "You are a strict output validator. Evaluate whether the following structured \
         output satisfies the rubric. Output JSON only:\n\
         {{\"score\": 0.0-1.0, \"accepted\": true/false, \
         \"reason\": \"brief explanation\"}}\n\n\
         Rubric:\n- min_score: {}\n- required fields: {:?}\n\n\
         Output to validate:\n{}",
        rubric.min_score,
        rubric.require_fields,
        serde_json::to_string(output).unwrap_or_default(),
    );
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: prompt.clone(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!("Validate the output for {owner}."),
        },
    ];
    tracing::debug!(
        target: "router.charts.rubric",
        owner = owner,
        prompt_len = prompt.len(),
        "rubric judge LLM request"
    );
    let response = backend
        .chat_complete(&messages)
        .map_err(|e| ChartError::Selection {
            reason: format!("rubric judge call failed: {e}"),
        })?;
    parse_judge_output(&response, rubric.min_score)
}

/// Parse a judge response (tolerant: fences + first-object extraction).
fn parse_judge_output(raw: &str, min_score: f64) -> Result<RubricVerdict, ChartError> {
    let trimmed = raw.trim();
    let cleaned = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed)
        .trim_end_matches("```")
        .trim();
    let value = if let Ok(v) = serde_json::from_str::<serde_json::Value>(cleaned) {
        v
    } else {
        let start = cleaned.find('{');
        let end = cleaned.rfind('}');
        match (start, end) {
            (Some(s), Some(e)) if e > s => {
                serde_json::from_str(&cleaned[s..=e]).map_err(|e| ChartError::Selection {
                    reason: format!("rubric judge output unparseable: {e}"),
                })?
            }
            _ => {
                return Err(ChartError::Selection {
                    reason: "rubric judge returned no JSON object".into(),
                });
            }
        }
    };
    let obj = value.as_object().ok_or_else(|| ChartError::Selection {
        reason: "rubric judge output is not a JSON object".into(),
    })?;
    let score = obj
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let accepted = obj
        .get("accepted")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(score >= min_score);
    let reason = obj
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("judge did not provide a reason")
        .to_string();
    let final_accepted = accepted && score >= min_score;
    Ok(RubricVerdict {
        accepted: final_accepted,
        reason: if final_accepted {
            reason
        } else {
            format!("judge score {score:.2} below min {min_score:.2}: {reason}")
        },
        score: Some(score),
        judged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_stubs::StubChatBackend;

    fn rubric(require: &[&str]) -> ChartRubric {
        ChartRubric {
            require_fields: require.iter().map(ToString::to_string).collect(),
            judge_model: None,
            min_score: 0.7,
        }
    }

    #[test]
    fn empty_rubric_passes_any_output() {
        let out = serde_json::json!({"whatever": true});
        let v = check_rubric(&rubric(&[]), &out, None, None, "t").expect("no error");
        assert!(v.accepted);
        assert!(!v.judged);
    }

    #[test]
    fn present_non_null_field_passes() {
        let out = serde_json::json!({"plan": "step 1", "cause": null});
        let v = check_rubric(&rubric(&["plan"]), &out, None, None, "t").expect("no error");
        assert!(v.accepted);
    }

    #[test]
    fn missing_field_fails() {
        let out = serde_json::json!({"plan": "step 1"});
        let v = check_rubric(&rubric(&["cause"]), &out, None, None, "t").expect("no error");
        assert!(!v.accepted);
        assert!(v.reason.contains("cause"));
    }

    #[test]
    fn null_field_fails() {
        let out = serde_json::json!({"plan": null});
        let v = check_rubric(&rubric(&["plan"]), &out, None, None, "t").expect("no error");
        assert!(!v.accepted);
        assert!(v.reason.contains("null"));
    }

    #[test]
    fn nested_path_check() {
        let out = serde_json::json!({"answer": {"steps": ["a"], "verdict": "ok"}});
        let v =
            check_rubric(&rubric(&["answer.verdict"]), &out, None, None, "t").expect("no error");
        assert!(v.accepted);
    }

    #[test]
    fn judge_consulted_only_when_configured() {
        let out = serde_json::json!({"x": 1});
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
            r#"{"score": 0.9, "accepted": true, "reason": "good"}"#,
        ));
        // No judge_model → backend ignored (judged = false, no LLM call).
        let v = check_rubric(&rubric(&["x"]), &out, Some(&backend), None, "t").expect("no error");
        assert!(v.accepted);
        assert!(!v.judged);

        // judge_model set → backend consulted.
        let mut r = rubric(&["x"]);
        r.judge_model = Some("judge".into());
        let v = check_rubric(&r, &out, Some(&backend), None, "t").expect("no error");
        assert!(v.accepted);
        assert!(v.judged);
        assert_eq!(v.score, Some(0.9));
    }

    #[test]
    fn judge_below_min_score_rejects() {
        let out = serde_json::json!({"x": 1});
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
            r#"{"score": 0.4, "accepted": true, "reason": "weak"}"#,
        ));
        let mut r = rubric(&["x"]);
        r.judge_model = Some("judge".into());
        let v = check_rubric(&r, &out, Some(&backend), None, "t").expect("no error");
        assert!(!v.accepted, "judge accept below min_score must reject");
        assert!(v.reason.contains("below min"));
    }

    #[test]
    fn judge_absent_backend_degrades_to_deterministic() {
        let out = serde_json::json!({"x": 1});
        let mut r = rubric(&["x"]);
        r.judge_model = Some("judge".into());
        // No backend provided → warn + accept on the deterministic gate.
        let v = check_rubric(&r, &out, None, None, "t").expect("no error");
        assert!(v.accepted);
        assert!(!v.judged);
    }

    #[test]
    fn judge_parse_tolerates_fences() {
        let v = parse_judge_output("```json\n{\"score\": 0.95, \"accepted\": true}\n```", 0.7)
            .expect("parses");
        assert!(v.accepted);
        assert_eq!(v.score, Some(0.95));
        let v = parse_judge_output("noise {\"score\": 0.2, \"accepted\": false} trailing", 0.7)
            .expect("parses");
        assert!(!v.accepted);
    }

    #[test]
    fn cache_short_circuits_and_records() {
        let cache = RubricCache::new();
        let out = serde_json::json!({"plan": "ok"});
        assert!(!cache.is_cached_accepted(&rubric(&["plan"]), &out));
        cache.record_accepted(&rubric(&["plan"]), &out);
        assert!(cache.is_cached_accepted(&rubric(&["plan"]), &out));
        assert_eq!(cache.len(), 1);

        // Same rubric, different output → not cached.
        let other = serde_json::json!({"plan": "different"});
        assert!(!cache.is_cached_accepted(&rubric(&["plan"]), &other));
    }

    #[test]
    fn cached_pair_short_circuits_gate() {
        let cache = RubricCache::new();
        let out = serde_json::json!({"x": 1});
        let mut r = rubric(&["x"]);
        r.judge_model = Some("judge".into());
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
            r#"{"score": 0.9, "accepted": true, "reason": "good"}"#,
        ));
        let v = check_rubric(&r, &out, Some(&backend), Some(&cache), "t").expect("no error");
        assert!(v.accepted);
        assert!(v.judged, "first run consults the judge");

        let v2 = check_rubric(&r, &out, Some(&backend), Some(&cache), "t").expect("no error");
        assert!(v2.accepted);
        assert!(!v2.judged, "cached pair skips the judge");
        assert!(v2.reason.contains("cached"));
    }
}
