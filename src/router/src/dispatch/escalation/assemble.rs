//! The parse/assemble/scorer helpers the escalation modes drive: approach-vote
//! and judge verdict parsing, the question-mode `assemble_answers` synthesizer,
//! the team-mode `BackendDecomposer` adapter, and the turnover `session_ledger_text`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fluent_llm::parse::parse_typed;
use fluent_llm::{ChatMessage, Decomposer, LlmError};

use crate::dag_session::DependencySession;
use crate::stages::prompt_parse::chat_json;
use crate::types::{RouterRequest, RouterResponse};

use super::LocalBackend;

/// Acceptance threshold applied by the question-mode `ResultScorer`.
pub(super) const SCORE_ACCEPTANCE_THRESHOLD: f64 = 0.7;

/// A per-hypothetical frontier job (question mode).
pub(super) struct HypotheticalTask {
    pub(super) request: RouterRequest,
}

/// A per-slot classifier job (team mode).
pub(super) struct ClassifierSlot {
    pub(super) index: usize,
    pub(super) slots: usize,
}

/// Extract the assistant text from a completion.
pub(super) fn response_text(resp: &RouterResponse) -> String {
    resp.choices
        .first()
        .map(|c| c.message.content.to_string_lossy())
        .unwrap_or_default()
}

/// Collapse the parallel classifier votes into a distribution string
/// ("3/3 recommend decomposition") for the draft/judge prompts.
pub(super) fn summarize_votes(votes: &[String]) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for vote in votes {
        let label = match parse_approach(vote) {
            Some(approach) => approach,
            None => vote.trim().to_string(),
        };
        *counts.entry(label).or_insert(0) += 1;
    }
    let total = votes.len();
    let mut parts: Vec<String> = counts
        .into_iter()
        .map(|(label, count)| format!("{count}/{total} say: {label}"))
        .collect();
    parts.sort_unstable();
    parts.join("\n")
}

/// Wire shape of a classifier slot's approach vote.
#[derive(serde::Deserialize)]
struct ApproachWire {
    #[serde(default)]
    approach: Option<String>,
}

/// Parse `{"approach": "...", ...}` from a classifier slot's raw output.
pub(super) fn parse_approach(raw: &str) -> Option<String> {
    parse_typed::<ApproachWire>(raw, &serde_json::Value::Null, |_| {})
        .ok()
        .and_then(|w| w.approach)
}

/// Wire shape of the frontier judge verdict.
#[derive(serde::Deserialize)]
struct JudgeWire {
    #[serde(default)]
    accepted: bool,
}

/// Judge verdict: `{"accepted": bool, "reason": "..."}` over the frontier
/// output. Non-JSON / missing `accepted` → rejected (conservative). The call +
/// tolerant-parse round-trip runs through the shared `chat_json` codec.
pub(super) fn judge_verdict(judge: &LocalBackend, user_text: &str, raw: &str) -> bool {
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a strict judge. Does the answer resolve the request with verified \
                      reasoning? Output JSON only: {\"accepted\": true/false, \"reason\": \"...\"}"
                .into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!("Request:\n{user_text}\n\nAnswer:\n{raw}"),
        },
    ];
    chat_json::<JudgeWire>(judge, &messages, &serde_json::Value::Null, |_| {})
        .is_ok_and(|w| w.accepted)
}

/// Synthesize the independent frontier answers into a final answer.
pub(super) fn assemble_answers(
    assembler: &LocalBackend,
    user_text: &str,
    answers: &[String],
) -> Result<String, LlmError> {
    let body = answers
        .iter()
        .enumerate()
        .map(|(i, a)| format!("{}. {a}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are synthesizing independent frontier answers to a decomposed query \
                      into one coherent final answer. Preserve the key facts. Output only the \
                      final answer."
                .into(),
        },
        ChatMessage {
            role: "user".into(),
            content: format!("Original query:\n{user_text}\n\nFrontier answers:\n{body}"),
        },
    ];
    assembler.chat_complete(&messages)
}

/// A `Decomposer` adapter over an injected local backend (DRY: mirrors
/// `LocalDecomposer`'s prompt/parse, but uses the ladder's injected backend
/// so tests substitute a stub without constructing an `LlmClient`).
pub(super) struct BackendDecomposer {
    backend: LocalBackend,
}

impl BackendDecomposer {
    pub(super) fn new(backend: LocalBackend) -> Self {
        Self { backend }
    }
}

const DECOMPOSER_SYSTEM_PROMPT: &str = "You are a task planner. Given a user query, decompose \
it into at most 5 concrete, ordered sub-tasks. Reply with ONLY a JSON array of strings, no \
preamble, no explanation. Example: [\"Find relevant documents\",\"Filter by date\",\"Summarize results\"]";

impl Decomposer for BackendDecomposer {
    fn decompose(&self, task: &str) -> Vec<String> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: DECOMPOSER_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".into(),
                content: task.to_string(),
            },
        ];
        let Ok(raw) = self.backend.chat_complete(&messages) else {
            return vec![task.to_string()];
        };
        parse_subtask_array(&raw).unwrap_or_else(|| vec![task.to_string()])
    }
}

/// Parse a JSON string array from an LLM response (tolerant — fenced/prose
/// wrapped), mirroring `LocalDecomposer`'s fallback semantics. Any non-string
/// entry or empty array → `None` (the caller falls back to the undecomposed
/// task), matching the historical behavior.
pub(super) fn parse_subtask_array(raw: &str) -> Option<Vec<String>> {
    let arr = parse_typed::<Vec<String>>(raw, &serde_json::Value::Null, |_| {}).ok()?;
    if arr.is_empty() {
        None
    } else {
        Some(arr)
    }
}

/// Render a session's completed steps as ledger text for the turnover
/// handoff. `None` when the session has nothing verified.
pub(super) fn session_ledger_text(session: &Arc<Mutex<DependencySession>>) -> Option<String> {
    let session = session.lock().ok()?;
    let mut parts: Vec<String> = Vec::new();
    for id in session.step_ids() {
        let step = session.get_step(id)?;
        let Some(result) = &step.result else { continue };
        if !result.accepted {
            continue;
        }
        if result.content.trim().is_empty() {
            continue;
        }
        parts.push(format!("[{}] {}", step.description, result.content));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}
