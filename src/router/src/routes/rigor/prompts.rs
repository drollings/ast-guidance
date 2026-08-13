//! Rigor route prompt constants, message builders, and tolerant parse helpers.
//!
//! The fixed-pass blue/red/judge protocol lives in [`super`]. This module owns
//! the prompt strings, the standard system+user message pair, the LLM call
//! wrapper (`chat`), and the parse helpers (`parse_objections` / `parse_judge`)
//! the protocol drives.

use std::collections::HashSet;

use common_core::sync::lock;
use fluent_llm::{parse_json_response, ChatMessage};
use fluent_types::{NodeId, StepStatus};
use serde::Deserialize;

use crate::dispatch::escalation::LocalBackend;

use super::{JudgeVerdict, RedObjection, RigorContext, RigorError};

/// The blue-team system prompt: a direct candidate answer, plain text out.
pub(super) const BLUE_SYSTEM_PROMPT: &str = r"You are the blue team. Produce a direct, correct candidate answer to the user's request.
Be complete and precise. Output only the answer text, no preamble.";

/// The red-team system prompt: a JSON array of objections, each anchored to a
/// claim's ledger node id where applicable.
pub(super) const RED_SYSTEM_PROMPT: &str = r#"You are the red team. Given a blue team's candidate answer and the session material,
find weaknesses: factual errors, missing requirements, unsafe claims, or unsupported assertions.
Output a JSON array of objections, each:
{"category": "a short category", "description": "the objection", "severity": 0.0-1.0, "target_claim": <ledger node id as a number, or null>}
Output only valid JSON, no other text."#;

/// The judge system prompt: a structured verdict.
pub(super) const JUDGE_SYSTEM_PROMPT: &str = r#"You are the judge. Given a blue team's candidate answer and the red team's objections,
decide whether the answer is acceptable.
Output JSON only:
{"verdict": "accept" | "accept_with_caveats" | "reject", "caveats": [...], "reasons": [...], "confidence": 0.0-1.0}
"confidence" is how confident you are in this verdict. Output only valid JSON, no other text."#;

/// Build the standard system+user message pair.
pub(super) fn messages(system: &str, user: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: system.into(),
        },
        ChatMessage {
            role: "user".into(),
            content: user.into(),
        },
    ]
}

/// Run the blue pass: candidate answer as plain text.
pub(super) async fn blue_answer(blue: &LocalBackend, user_message: &str) -> Result<String, RigorError> {
    chat(blue, BLUE_SYSTEM_PROMPT, user_message)
        .await
        .map_err(|e| RigorError::BlueTeam(e.to_string()))
}

/// Run one blocking `ChatBackend::chat_complete` off the executor.
///
/// The role backends are synchronous (the codebase's DIP pattern - the same
/// `Arc<dyn ChatBackend>` used by dispatch). `RigorRoute::execute` is async by
/// locked decision D5, so the blocking calls are offloaded via
/// `spawn_blocking` and awaited, honoring the WorkUnit purity contract (never
/// block a tokio worker).
pub(super) async fn chat(
    backend: &LocalBackend,
    system: &str,
    user: &str,
) -> Result<String, fluent_llm::LlmError> {
    let backend = backend.clone();
    let messages = messages(system, user);
    tokio::task::spawn_blocking(move || backend.chat_complete(&messages))
        .await
        .map_err(|e| fluent_llm::LlmError::Api(format!("chat task failed: {e}")))?
}

/// Fold the red objections into a refocused blue prompt for the second pass.
pub(super) fn blue_retry_prompt(
    user_message: &str,
    prior_answer: &str,
    objections: &[RedObjection],
) -> String {
    use std::fmt::Write as _;

    let mut prompt = format!(
        "Original request:\n{user_message}\n\n\
         Your previous answer was rejected by the red team.\n\
         Previous answer:\n{prior_answer}\n\n\
         Address these objections:\n"
    );
    for o in objections {
        let _ = writeln!(
            prompt,
            "- [{:.2}] {}: {}",
            o.severity, o.category, o.description
        );
    }
    prompt
}

/// Build the judge prompt: the candidate answer plus each objection.
pub(super) fn judge_prompt(answer: &str, objections: &[RedObjection]) -> String {
    use std::fmt::Write as _;

    let mut prompt = format!("Blue team's candidate answer:\n{answer}\n\nRed team objections:\n");
    if objections.is_empty() {
        prompt.push_str("(none)\n");
    }
    for (i, o) in objections.iter().enumerate() {
        let claim = o
            .target_claim
            .map_or_else(|| "null".to_string(), |id| id.as_int().to_string());
        let _ = writeln!(
            prompt,
            "{i}. [{:.2}] {}: {} (target_claim: {claim})",
            o.severity, o.category, o.description
        );
    }
    prompt
}

/// Wire shape of the red output before it is mapped into `RedObjection`.
#[derive(Debug, Deserialize)]
struct RedObjectionWire {
    #[serde(default)]
    category: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    severity: f64,
    #[serde(default)]
    target_claim: Option<NodeId>,
}

/// Tolerant parse of the red-team objections array via
/// `fluent_llm::parse_json_response` (never hand-rolled fence-stripping).
pub(super) fn parse_objections(raw: &str) -> Result<Vec<RedObjection>, String> {
    let value = parse_json_response(raw).map_err(|e| e.to_string())?;
    let arr = match value {
        serde_json::Value::Array(arr) => arr,
        serde_json::Value::Object(map) => {
            // Tolerate a wrapped `{"objections": [...]}` shape.
            map.get("objections")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .ok_or_else(|| "red output is neither an array nor an objections map".to_string())?
        }
        other => return Err(format!("red output is not a JSON array: {other}")),
    };
    arr.into_iter()
        .map(|v| {
            let wire: RedObjectionWire =
                serde_json::from_value(v).map_err(|e| format!("invalid objection: {e}"))?;
            Ok(RedObjection {
                category: wire.category,
                description: wire.description,
                severity: wire.severity,
                target_claim: wire.target_claim,
            })
        })
        .collect()
}

/// Wire shape of the judge output.
#[derive(Debug, Deserialize)]
struct JudgeOutputWire {
    #[serde(default)]
    verdict: String,
    #[serde(default)]
    caveats: Vec<String>,
    #[serde(default)]
    reasons: Vec<String>,
    #[serde(default = "default_judge_confidence")]
    confidence: f64,
}

pub(super) const fn default_judge_confidence() -> f64 {
    0.5
}

/// Tolerant parse of the judge verdict via `fluent_llm::parse_json_response`.
pub(super) fn parse_judge(raw: &str) -> Result<(JudgeVerdict, f64), String> {
    let value = parse_json_response(raw).map_err(|e| e.to_string())?;
    let wire: JudgeOutputWire =
        serde_json::from_value(value).map_err(|e| format!("invalid judge output: {e}"))?;
    let verdict = match wire.verdict.as_str() {
        "accept" => JudgeVerdict::Accept,
        "accept_with_caveats" => JudgeVerdict::AcceptWithCaveats {
            caveats: wire.caveats,
        },
        "reject" => JudgeVerdict::Reject {
            reasons: wire.reasons,
        },
        other => return Err(format!("unknown judge verdict: {other}")),
    };
    Ok((verdict, wire.confidence))
}

/// Whether a verdict is a rejection.
pub(super) fn is_reject(verdict: &JudgeVerdict) -> bool {
    matches!(verdict, JudgeVerdict::Reject { .. })
}

/// Audit-tag form of a verdict.
pub fn verdict_tag(verdict: &JudgeVerdict) -> &'static str {
    match verdict {
        JudgeVerdict::Accept => "accept",
        JudgeVerdict::AcceptWithCaveats { .. } => "accept_with_caveats",
        JudgeVerdict::Reject { .. } => "reject",
    }
}

/// The dead-end node ids for the red-team filtered view: ledger nodes in the
/// session whose `accepted == Some(false)` (the blue dead ends persisted by
/// `record_blue_dead_end`), plus nodes mapped from session steps whose result
/// is rejected (the roadmap's step-level mechanism, where available).
pub(super) fn dead_end_node_ids(ctx: &RigorContext) -> HashSet<NodeId> {
    let Some(ledger) = &ctx.ledger else {
        return HashSet::new();
    };
    let store = ledger.node_store();
    let mut rejected_steps: HashSet<String> = HashSet::new();
    if let Some(session) = &ctx.session {
        let s = lock(session);
        for step_id in s.step_ids() {
            if let Some(step) = s.get_step(step_id) {
                let rejected = matches!(step.status, StepStatus::Failed | StepStatus::Cancelled)
                    || step
                        .result
                        .as_ref()
                        .is_some_and(|r| !r.accepted || r.error.is_some());
                if rejected {
                    rejected_steps.insert(step_id.clone());
                }
            }
        }
    }
    store
        .session_node_ids(&ctx.session_id)
        .into_iter()
        .filter(|nid| {
            store.snapshot(*nid).is_some_and(|node| {
                node.accepted == Some(false)
                    || node
                        .step_id
                        .as_deref()
                        .is_some_and(|sid| rejected_steps.contains(sid))
            })
        })
        .collect()
}

/// The targeted-interview fallback: the <= 3 highest-severity objections,
/// turned into direct clarification questions (VISION: "a targeted interview
/// with the user - not silent escalation").
pub(super) fn derive_interview(objections: &[RedObjection]) -> Vec<String> {
    let mut ranked: Vec<&RedObjection> = objections.iter().collect();
    ranked.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
        .into_iter()
        .take(3)
        .map(|o| {
            format!(
                "{} - {}. Could you clarify this concern?",
                o.category, o.description
            )
        })
        .collect()
}
