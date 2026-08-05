//! Escalation-ladder dispatch loop (ROADMAP_20260805_REVIEW M3).
//!
//! Implements VISION §"The escalation ladder": after every local model in a
//! `model_group` chain fails, the request escalates through the configured
//! modes — `filter → question → team → turnover` — in order, each a discrete
//! policy governing how much context/agency the frontier model receives.
//!
//! Design rules from the roadmap:
//! - Deterministic-first: a [`fluent_types::ContextCache`] lookup
//!   short-circuits *before* any frontier call.
//! - Frontier transport reuses `dispatch/backend.rs` (`ChatBackend`) — no
//!   third HTTP path.
//! - Every interaction writes a `kind = "escalation"` record via
//!   [`crate::audit::emit`] carrying `mode`/`accepted`/`payload`/
//!   `raw_response`/`trigger`/`timestamp`.
//! - Parallel slots reuse `fluent_concurrency::ResultPool`; local roles reuse
//!   `transforms/`, `summarization::ResultScorer`, and the shared tolerant
//!   `fluent_llm::parse_json_response`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fluent_concurrency::pool::{ResultPool, ResultPoolError};
use fluent_llm::{ChatMessage, Decomposer, LlmError};
use fluent_wvr::prelude::*;
use serde_json::Value;

use crate::config::{EscalationLadderConfig, FrontierConfig};
use crate::dag_session::DependencySession;
use crate::dispatch::backend::ChatBackend as DispatchChatBackend;
use crate::dispatch::frontier::DispatchError;
use crate::frontier::modes::EscalationMode;
use crate::server::dispatch::render_prompt;
use crate::server::responses::{completion_to_response, make_text_completion, HyperResponse};
use crate::stages::deterministic::DeterministicPreFilter;
use crate::summarization::{ResultScorer, ScoredResult};
use crate::transforms::pii_anonymize::PiiAnonymize;
use crate::transforms::TransformStrategy;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse};

/// A sync local-model backend — the role engines (decomposer/assembler/
/// classifier/draft/judge) run against `fluent_llm::ChatBackend`.
pub type LocalBackend = Arc<dyn fluent_llm::client::ChatBackend>;

/// The async frontier backend — every actual frontier HTTP call goes through
/// the canonical `dispatch/backend.rs` transport.
pub type FrontierBackend = Arc<dyn DispatchChatBackend>;

/// Backends for every ladder role. Roles that are `None` disable the modes
/// that require them.
#[derive(Clone)]
pub struct EscalationBackends {
    /// Frontier transport (required for every mode).
    pub frontier: FrontierBackend,
    /// Question-mode: breaks the anonymized query into hypotheticals.
    pub decomposer: Option<LocalBackend>,
    /// Question-mode: synthesizes + scores the final answer.
    pub assembler: Option<LocalBackend>,
    /// Team-mode: parallel classifier slots that vote on approach.
    pub classifier: Option<LocalBackend>,
    /// Team-mode: decomposes into subtasks + attempts a local draft.
    pub draft: Option<LocalBackend>,
    /// Team-mode: crafts the frontier prompt from the gap + judges the result.
    pub judge: Option<LocalBackend>,
}

/// Per-request escalation context (the `ctx_deps` bundle the ladder needs).
pub struct EscalationContext<'a> {
    /// The full normalized request.
    pub request: &'a RouterRequest,
    /// The last user message text.
    pub user_text: &'a str,
    /// The model name the client requested (used on the completion response).
    pub model_name: &'a str,
    /// Deterministic-fact cache consulted before any frontier call.
    pub context_cache: Option<&'a Arc<dyn fluent_types::ContextCache>>,
    /// The session to mark frontier-owned after a turnover handoff.
    pub session: Option<&'a Arc<Mutex<DependencySession>>>,
}

/// A configured escalation ladder for one model group.
pub struct EscalationLadder {
    config: EscalationLadderConfig,
    backends: EscalationBackends,
}

impl EscalationLadder {
    pub fn new(config: EscalationLadderConfig, backends: EscalationBackends) -> Self {
        Self { config, backends }
    }

    pub fn config(&self) -> &EscalationLadderConfig {
        &self.config
    }

    /// Direct frontier dispatch of the full request — the bypass path for a
    /// session the turnover mode already marked frontier-owned (M3.7).
    /// Buffered (multi-step escalation is not streamable); `None` on failure.
    pub async fn dispatch_frontier(&self, ctx: &EscalationContext<'_>) -> Option<HyperResponse> {
        self.turnover_mode(ctx).await.ok().flatten()
    }

    /// Attempt to resolve the request through the ladder.
    ///
    /// Returns `Some(response)` when a rung accepted the request (or the
    /// context cache short-circuited); `None` when every configured mode was
    /// rejected or skipped — the caller then returns `fallback_completion`.
    pub async fn try_escalate(&self, ctx: &EscalationContext<'_>) -> Option<HyperResponse> {
        // Deterministic-first: a context hit short-circuits before any
        // frontier call (VISION §"Post-processing": the same question never
        // pays frontier cost twice).
        if let Some(cache) = ctx.context_cache {
            if let Some(hit) = cache.lookup(ctx.user_text) {
                tracing::info!(
                    target: "router.dispatch.escalation",
                    mode = "context",
                    source = %hit.source,
                    score = hit.score,
                    "context cache short-circuit"
                );
                Self::emit_audit(
                    "context",
                    true,
                    ctx.user_text,
                    &hit.content,
                    "context cache hit",
                    &serde_json::json!({ "source": hit.source, "score": hit.score }),
                );
                let completion = make_text_completion(ctx.model_name, &hit.content);
                return Some(completion_to_response(&completion, ctx.model_name, false, None));
            }
        }

        for mode in &self.config.modes {
            match self.run_mode(*mode, ctx).await {
                Ok(Some(resp)) => return Some(resp),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        target: "router.dispatch.escalation",
                        mode = ?mode,
                        error = %e,
                        retryable = e.is_retryable(),
                        "escalation mode failed"
                    );
                }
            }
        }
        None
    }

    async fn run_mode(
        &self,
        mode: EscalationMode,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        match mode {
            EscalationMode::Filter => self.filter_mode(ctx).await,
            EscalationMode::Question => self.question_mode(ctx).await,
            EscalationMode::Team => self.team_mode(ctx).await,
            EscalationMode::Turnover => self.turnover_mode(ctx).await,
        }
    }

    /// One-shot dispatch of a full request to the frontier backend — the
    /// single frontier HTTP path shared by all four modes.
    async fn frontier_complete(
        &self,
        request: &RouterRequest,
        front: &FrontierConfig,
    ) -> Result<RouterResponse, DispatchError> {
        self.backends
            .frontier
            .complete(
                request.clone(),
                front.model.clone(),
                None,
                common_core::constants::DEFAULT_IDLE_TIMEOUT_MS,
                common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS,
                false,
            )
            .await
    }

    // ── Mode implementations ──────────────────────────────────────────────

    /// filter: PII transform → frontier → stage-1 re-scan; accept if clean.
    async fn filter_mode(
        &self,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        let Some(front) = &self.config.frontier else {
            return Ok(None);
        };

        let filtered = PiiAnonymize
            .transform(ctx.request)
            .map_err(|e| DispatchError::RequestBuild(e.to_string()))?;
        let payload = render_prompt(&filtered);
        let resp = match self.frontier_complete(&filtered, front).await {
            Ok(r) => r,
            Err(e) => {
                Self::emit_audit(
                    "filter",
                    false,
                    &payload,
                    "",
                    &format!("frontier error: {e}"),
                    &serde_json::json!({}),
                );
                return Ok(None);
            }
        };
        let raw = response_text(&resp);

        // Re-scan the frontier output with the stage-1 engine; only a clean
        // response is accepted (VISION: "deterministic PII rules strip
        // sensitive content").
        let prefilter = DeterministicPreFilter::new();
        let accepted = prefilter.scan_output(&raw).is_none();
        let trigger = if accepted {
            "response clean after re-scan"
        } else {
            "response re-flagged by stage-1 filters"
        };
        Self::emit_audit("filter", accepted, &payload, &raw, trigger, &serde_json::json!({}));

        if accepted {
            Ok(Some(completion_to_response(
                &resp,
                ctx.model_name,
                false,
                Some(&resp.model),
            )))
        } else {
            Ok(None)
        }
    }

    /// question: anonymize + decompose → parallel hypothetical frontier
    /// calls (`ResultPool`) → assembler synthesizes → `ResultScorer` gates.
    async fn question_mode(
        &self,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        let Some(front) = &self.config.frontier else {
            return Ok(None);
        };
        let (Some(decomposer), Some(assembler)) =
            (self.backends.decomposer.clone(), self.backends.assembler.clone())
        else {
            return Ok(None);
        };

        // Anonymize first, then break the abstract problem into generic
        // hypothetical questions (frontier sees no personal data).
        let anonymized = fluent_llm::anonymize(ctx.user_text);
        let hypotheticals = BackendDecomposer::new(decomposer).decompose(&anonymized);
        if hypotheticals.is_empty() {
            return Ok(None);
        }
        let payload = format!("hypotheticals: {hypotheticals:?}");

        let mut answers = Vec::new();
        {
            let pool = ResultPool::new(
                fluent_concurrency::tokio_runtime(),
                hypotheticals.len(),
                hypotheticals.len(),
                {
                    let backend = self.backends.frontier.clone();
                    let model = front.model.clone();
                    move |task: HypotheticalTask| {
                        let backend = backend.clone();
                        let model = model.clone();
                        async move {
                            backend
                                .complete(
                                    task.request,
                                    model,
                                    None,
                                    common_core::constants::DEFAULT_IDLE_TIMEOUT_MS,
                                    common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS,
                                    false,
                                )
                                .await
                        }
                    }
                },
            );

            for hyp in &hypotheticals {
                let mut req = ctx.request.clone();
                req.messages = vec![RouterMessage {
                    role: "user".into(),
                    content: RouterMessageContent::Text(hyp.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                }];
                match pool.submit(HypotheticalTask { request: req }).await {
                    Ok(resp) => answers.push(response_text(&resp)),
                    Err(e) => match e {
                        ResultPoolError::Inner(de) => {
                            tracing::warn!(
                                target: "router.dispatch.escalation",
                                mode = "question",
                                error = %de,
                                "hypothetical frontier call failed"
                            );
                        }
                        _ => {
                            tracing::warn!(
                                target: "router.dispatch.escalation",
                                mode = "question",
                                "hypothetical pool submit failed"
                            );
                        }
                    },
                }
            }
            pool.shutdown().await;
        }

        if answers.is_empty() {
            Self::emit_audit(
                "question",
                false,
                &payload,
                "",
                "no frontier answers",
                &serde_json::json!({}),
            );
            return Ok(None);
        }

        // Assemble the independent answers into one coherent final answer.
        let assembled = match assemble_answers(&assembler, ctx.user_text, &answers) {
            Ok(a) => a,
            Err(e) => {
                Self::emit_audit(
                    "question",
                    false,
                    &payload,
                    "",
                    &format!("assembler error: {e}"),
                    &serde_json::json!({}),
                );
                return Ok(None);
            }
        };

        // Acceptance gate: the canonical `ResultScorer` (local model).
        let scorer = ResultScorer::new(assembler.clone(), SCORE_ACCEPTANCE_THRESHOLD);
        let mut wctx = WorkContext::default();
        wctx.set_structured("request", ctx.request);
        wctx.metadata
            .insert("response".into(), MetadataValue::from(assembled.clone()));
        let accepted = match scorer.execute(&wctx) {
            Ok(output) => output.data_as::<ScoredResult>().is_ok_and(|s| s.accepted),
            Err(e) => {
                tracing::warn!(
                    target: "router.dispatch.escalation",
                    mode = "question",
                    error = %e,
                    "result scorer failed"
                );
                false
            }
        };
        let trigger = if accepted {
            "assembled answer accepted by scorer"
        } else {
            "assembled answer rejected by scorer"
        };
        Self::emit_audit(
            "question",
            accepted,
            &payload,
            &assembled,
            trigger,
            &serde_json::json!({ "answers": answers.len() }),
        );

        if accepted {
            let completion = make_text_completion(ctx.model_name, &assembled);
            Ok(Some(completion_to_response(
                &completion,
                ctx.model_name,
                false,
                None,
            )))
        } else {
            Ok(None)
        }
    }

    /// team: parallel classifier votes → draft model attempts subtasks →
    /// judge crafts a precise frontier prompt → frontier → judge verdict.
    async fn team_mode(
        &self,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        let Some(front) = &self.config.frontier else {
            return Ok(None);
        };
        let (Some(classifier), Some(draft), Some(judge)) = (
            self.backends.classifier.clone(),
            self.backends.draft.clone(),
            self.backends.judge.clone(),
        )
        else {
            return Ok(None);
        };
        let slots = self.config.classifier_parallel.max(1);
        let user_text = ctx.user_text.to_string();

        // Parallel classifier slots vote on the approach. Slot diversity comes
        // from a per-slot instruction (the sync `ChatBackend` fixes the
        // temperature at construction; `classifier_parallel` instances of the
        // same model are simulated by varied prompts).
        let votes: Vec<String> = {
            let pool = ResultPool::new(
                fluent_concurrency::tokio_runtime(),
                slots,
                slots,
                {
                    let backend = classifier.clone();
                    let user_text = user_text.clone();
                    move |task: ClassifierSlot| {
                        let backend = backend.clone();
                        let user_text = user_text.clone();
                        async move {
                            let messages = vec![
                                ChatMessage {
                                    role: "system".into(),
                                    content: format!(
                                        "You are analyst vote {} of {}. Analyze the request and \
                                         output JSON only: {{\"approach\": \"...\", \"confidence\": 0.0-1.0}}",
                                        task.index + 1,
                                        task.slots,
                                    ),
                                },
                                ChatMessage {
                                    role: "user".into(),
                                    content: user_text,
                                },
                            ];
                            backend.chat_complete(&messages)
                        }
                    }
                },
            );

            let mut out = Vec::new();
            for index in 0..slots {
                match pool.submit(ClassifierSlot { index, slots }).await {
                    Ok(vote) => out.push(vote),
                    Err(e) => match e {
                        ResultPoolError::Inner(le) => {
                            tracing::warn!(
                                target: "router.dispatch.escalation",
                                mode = "team",
                                slot = index,
                                error = %le,
                                "classifier slot failed"
                            );
                        }
                        _ => {
                            tracing::warn!(
                                target: "router.dispatch.escalation",
                                mode = "team",
                                slot = index,
                                "classifier slot submit failed"
                            );
                        }
                    },
                }
            }
            pool.shutdown().await;
            out
        };

        if votes.is_empty() {
            Self::emit_audit(
                "team",
                false,
                ctx.user_text,
                "",
                "no classifier votes",
                &serde_json::json!({}),
            );
            return Ok(None);
        }

        let distribution = summarize_votes(&votes);

        // Draft model: decompose into subtasks locally, then attempt a draft.
        let subtasks = BackendDecomposer::new(draft.clone()).decompose(&user_text);
        let draft_attempt = draft
            .chat_complete(&[
                ChatMessage {
                    role: "system".into(),
                    content: "You are drafting a local solution. Attempt as much of the request \
                              as you can; leave the unsolved remainder clearly marked."
                        .into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: user_text.clone(),
                },
            ])
            .unwrap_or_default();

        // Judge crafts the frontier prompt: only the unsolved gap + verified
        // partial work crosses the boundary.
        let frontier_prompt = judge
            .chat_complete(&[
                ChatMessage {
                    role: "system".into(),
                    content: "You are a senior reviewer. Given the original request, the team's \
                              approach votes, and the local draft, craft ONE precise prompt for a \
                              frontier model that asks ONLY for the unsolved remainder. Reference \
                              verified partial work. Output only the prompt text."
                        .into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: format!(
                        "Request:\n{user_text}\n\nVotes:\n{distribution}\n\nLocal draft:\n{draft_attempt}"
                    ),
                },
            ])
            .unwrap_or_else(|_| user_text.clone());
        let payload = frontier_prompt.clone();

        // Frontier gets the judge-crafted prompt.
        let mut req = ctx.request.clone();
        req.messages = vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(frontier_prompt),
            tool_calls: None,
            tool_call_id: None,
        }];
        let resp = match self.frontier_complete(&req, front).await {
            Ok(r) => r,
            Err(e) => {
                Self::emit_audit(
                    "team",
                    false,
                    &payload,
                    "",
                    &format!("frontier error: {e}"),
                    &serde_json::json!({ "votes": votes.len(), "subtasks": subtasks.len() }),
                );
                return Ok(None);
            }
        };
        let raw = response_text(&resp);

        // Judge verdict on the frontier output.
        let accepted = judge_verdict(&judge, &user_text, &raw);
        let trigger = if accepted {
            "judge accepted frontier output"
        } else {
            "judge rejected frontier output"
        };
        Self::emit_audit(
            "team",
            accepted,
            &payload,
            &raw,
            trigger,
            &serde_json::json!({ "votes": votes.len(), "subtasks": subtasks.len() }),
        );

        if accepted {
            Ok(Some(completion_to_response(
                &resp,
                ctx.model_name,
                false,
                Some(&resp.model),
            )))
        } else {
            Ok(None)
        }
    }

    /// turnover: the full session (all messages + ledger text) goes to
    /// frontier, and the session is marked frontier-owned so subsequent
    /// requests bypass the pipeline.
    async fn turnover_mode(
        &self,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        let Some(front) = &self.config.frontier else {
            return Ok(None);
        };

        let mut req = ctx.request.clone();
        if let Some(session) = ctx.session {
            if let Some(ledger_text) = session_ledger_text(session) {
                req.messages.push(RouterMessage {
                    role: "system".into(),
                    content: RouterMessageContent::Text(format!(
                        "Session ledger (prior verified work):\n{ledger_text}"
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }
        let payload = render_prompt(&req);

        let resp = match self.frontier_complete(&req, front).await {
            Ok(r) => r,
            Err(e) => {
                Self::emit_audit(
                    "turnover",
                    false,
                    &payload,
                    "",
                    &format!("frontier error: {e}"),
                    &serde_json::json!({}),
                );
                return Ok(None);
            }
        };
        let raw = response_text(&resp);

        // Hand off: subsequent requests in this session go through frontier.
        if let Some(session) = ctx.session {
            if let Ok(mut s) = session.lock() {
                s.set_frontier_owned(true);
            }
        }

        Self::emit_audit(
            "turnover",
            true,
            &payload,
            &raw,
            "frontier handoff",
            &serde_json::json!({}),
        );
        Ok(Some(completion_to_response(
            &resp,
            ctx.model_name,
            false,
            Some(&resp.model),
        )))
    }

    fn emit_audit(
        mode: &str,
        accepted: bool,
        payload: &str,
        raw_response: &str,
        trigger: &str,
        extra: &serde_json::Value,
    ) {
        let mut record = serde_json::json!({
            "mode": mode,
            "accepted": accepted,
            "payload": payload,
            "raw_response": raw_response,
            "trigger": trigger,
            "timestamp": common_core::now_secs(),
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                record[k] = v.clone();
            }
        }
        crate::audit::emit("escalation", record);
    }
}

/// Acceptance threshold applied by the question-mode `ResultScorer`.
const SCORE_ACCEPTANCE_THRESHOLD: f64 = 0.7;

/// A per-hypothetical frontier job (question mode).
struct HypotheticalTask {
    request: RouterRequest,
}

/// A per-slot classifier job (team mode).
struct ClassifierSlot {
    index: usize,
    slots: usize,
}

/// Extract the assistant text from a completion.
fn response_text(resp: &RouterResponse) -> String {
    resp.choices
        .first()
        .map(|c| c.message.content.to_string_lossy())
        .unwrap_or_default()
}

/// Collapse the parallel classifier votes into a distribution string
/// ("3/3 recommend decomposition") for the draft/judge prompts.
fn summarize_votes(votes: &[String]) -> String {
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

/// Parse `{"approach": "...", ...}` from a classifier slot's raw output.
fn parse_approach(raw: &str) -> Option<String> {
    let parsed = fluent_llm::parse::parse_json_response(raw).ok()?;
    let approach = parsed.get("approach")?.as_str()?;
    Some(approach.to_string())
}

/// Judge verdict: `{"accepted": bool, "reason": "..."}` over the frontier
/// output. Non-JSON / missing `accepted` → rejected (conservative).
fn judge_verdict(judge: &LocalBackend, user_text: &str, raw: &str) -> bool {
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
    let Ok(verdict) = judge.chat_complete(&messages) else {
        return false;
    };
    fluent_llm::parse::parse_json_response(&verdict)
        .ok()
        .and_then(|v| v.get("accepted").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Synthesize the independent frontier answers into a final answer.
fn assemble_answers(
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
struct BackendDecomposer {
    backend: LocalBackend,
}

impl BackendDecomposer {
    fn new(backend: LocalBackend) -> Self {
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
/// wrapped), mirroring `LocalDecomposer`'s fallback semantics.
fn parse_subtask_array(raw: &str) -> Option<Vec<String>> {
    let parsed = fluent_llm::parse::parse_json_response(raw).ok()?;
    let Value::Array(arr) = parsed else {
        return None;
    };
    if arr.is_empty() {
        return None;
    }
    arr.into_iter()
        .map(|v| v.as_str().map(ToOwned::to_owned))
        .collect()
}

/// Render a session's completed steps as ledger text for the turnover
/// handoff. `None` when the session has nothing verified.
fn session_ledger_text(session: &Arc<Mutex<DependencySession>>) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag_session::{SessionStep, StepResult};
    use crate::dispatch::backend::ChatBackend;
    use crate::dispatch::backend::StreamResult;
    use crate::testing::mock::TranscriptProvider;
    use fluent_types::ContextHit;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    fn test_request(text: &str) -> RouterRequest {
        RouterRequest {
            model: "fast".into(),
            messages: vec![RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text(text.into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            metadata: Default::default(),
        }
    }

    fn text_response(model: &str, content: &str) -> RouterResponse {
        RouterResponse {
            id: "r".into(),
            object: "chat.completion".into(),
            created: 0,
            model: model.into(),
            choices: vec![crate::types::RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text(content.into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: crate::types::Usage::default(),
        }
    }

    /// Async frontier stub: canned responses (popped FIFO) + a record of
    /// every request it received.
    struct StubFrontier {
        responses: Mutex<VecDeque<RouterResponse>>,
        received: Mutex<Vec<RouterRequest>>,
    }

    impl StubFrontier {
        fn new(responses: Vec<RouterResponse>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into()),
                received: Mutex::new(Vec::new()),
            })
        }
        fn received_texts(&self) -> Vec<String> {
            self.received
                .lock()
                .unwrap()
                .iter()
                .map(render_prompt)
                .collect()
        }
    }

    impl ChatBackend for StubFrontier {
        fn complete(
            &self,
            request: RouterRequest,
            _model: String,
            _params: Option<Value>,
            _idle_timeout_ms: u64,
            _total_timeout_ms: u64,
            _filter_thinking: bool,
        ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
            self.received.lock().unwrap().push(request);
            let resp = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or(DispatchError::AllBackendsFailed);
            Box::pin(async move { resp })
        }

        fn stream_complete(
            &self,
            _request: RouterRequest,
            _model: String,
            _params: Option<Value>,
            _idle_timeout_ms: u64,
            _total_timeout_ms: u64,
            _filter_thinking: bool,
        ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>> {
            Box::pin(async move { Err(DispatchError::AllBackendsFailed) })
        }
    }

    /// Sync local-role stub: routes each call by the user message content.
    fn func_backend<F>(f: F) -> Arc<dyn fluent_llm::client::ChatBackend>
    where
        F: Fn(&[ChatMessage]) -> Result<String, LlmError> + Send + Sync + 'static,
    {
        struct FuncBackend<F>(F);
        impl<F> fluent_llm::client::ChatBackend for FuncBackend<F>
        where
            F: Fn(&[ChatMessage]) -> Result<String, LlmError> + Send + Sync,
        {
            fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
                (self.0)(messages)
            }
        }
        Arc::new(FuncBackend(f))
    }

    /// Read the assistant text out of a `HyperResponse` body.
    async fn read_body_text(resp: HyperResponse) -> String {
        use http_body_util::BodyExt;
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    fn last_user_message(messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default()
    }

    fn filter_config() -> EscalationLadderConfig {
        EscalationLadderConfig {
            modes: vec![EscalationMode::Filter],
            frontier: Some(FrontierConfig {
                endpoint: "http://frontier.test/v1/chat/completions".into(),
                api_key_env: None,
                model: "claude".into(),
            }),
            classifier_parallel: 2,
            ..Default::default()
        }
    }

    fn full_modes_config() -> EscalationLadderConfig {
        EscalationLadderConfig {
            modes: vec![
                EscalationMode::Filter,
                EscalationMode::Question,
                EscalationMode::Team,
                EscalationMode::Turnover,
            ],
            frontier: Some(FrontierConfig {
                endpoint: "http://frontier.test/v1/chat/completions".into(),
                api_key_env: None,
                model: "claude".into(),
            }),
            classifier_parallel: 2,
            ..Default::default()
        }
    }

    #[derive(Default)]
    struct MapContextCache {
        map: Mutex<HashMap<String, String>>,
    }

    impl fluent_types::ContextCache for MapContextCache {
        fn lookup(&self, query: &str) -> Option<ContextHit> {
            self.map.lock().unwrap().get(query).map(|content| ContextHit {
                source: "test-cache".into(),
                content: content.clone(),
                score: 0.99,
                metadata: None,
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn context_cache_short_circuits_before_any_mode() {
        let map_cache = Arc::new(MapContextCache::default());
        map_cache
            .map
            .lock()
            .unwrap()
            .insert("known question".into(), "verified answer".into());
        let cache: Arc<dyn fluent_types::ContextCache> = map_cache;

        let ladder = EscalationLadder::new(
            full_modes_config(),
            EscalationBackends {
                // No frontier backends required: the cache hit returns first.
                frontier: StubFrontier::new(vec![]),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("known question");
        let ctx = EscalationContext {
            request: &request,
            user_text: "known question",
            model_name: "fast",
            context_cache: Some(&cache),
            session: None,
        };
        let resp = ladder.try_escalate(&ctx).await.expect("short-circuits");
        assert_eq!(read_body_text(resp).await, "verified answer");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_context_hit_does_not_short_circuit() {
        let cache: Arc<dyn fluent_types::ContextCache> = Arc::new(MapContextCache::default());
        let frontier = StubFrontier::new(vec![text_response("claude", "local answer")]);
        let ladder = EscalationLadder::new(
            full_modes_config(),
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("what is 2+2?");
        let ctx = EscalationContext {
            request: &request,
            user_text: "what is 2+2?",
            model_name: "fast",
            context_cache: Some(&cache),
            session: None,
        };
        // Filter mode runs: PII transform is identity here, frontier returns
        // a clean answer → accepted.
        let resp = ladder.try_escalate(&ctx).await.expect("filter accepted");
        assert_eq!(read_body_text(resp).await, "local answer");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn filter_mode_accepts_clean_frontier_response() {
        let frontier = StubFrontier::new(vec![text_response("claude", "The answer is 4")]);
        let ladder = EscalationLadder::new(
            filter_config(),
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("What is 2+2?");
        let ctx = EscalationContext {
            request: &request,
            user_text: "What is 2+2?",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        let resp = ladder.try_escalate(&ctx).await.expect("accepted");
        assert_eq!(read_body_text(resp).await, "The answer is 4");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn filter_mode_rejects_pii_flagged_response() {
        let frontier = StubFrontier::new(vec![text_response(
            "claude",
            "Reach me at alice@example.com",
        )]);
        let ladder = EscalationLadder::new(
            filter_config(),
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("email me");
        let ctx = EscalationContext {
            request: &request,
            user_text: "email me",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        assert!(
            ladder.try_escalate(&ctx).await.is_none(),
            "PII-flagged frontier output must not be accepted"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mode_walk_escalates_on_rejection() {
        // Filter rejects (PII in output) → turnover accepts.
        let frontier = StubFrontier::new(vec![
            text_response("claude", "call bob@example.com"),
            text_response("claude", "escalated answer"),
        ]);
        let config = EscalationLadderConfig {
            modes: vec![EscalationMode::Filter, EscalationMode::Turnover],
            frontier: filter_config().frontier,
            ..Default::default()
        };
        let ladder = EscalationLadder::new(
            config,
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("help me");
        let ctx = EscalationContext {
            request: &request,
            user_text: "help me",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        let resp = ladder.try_escalate(&ctx).await.expect("turnover accepted");
        assert_eq!(read_body_text(resp).await, "escalated answer");
        assert_eq!(frontier.received_texts().len(), 2, "both modes hit frontier");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn question_mode_accepts_assembled_answer() {
        let decomposer = func_backend(|_| Ok(r#"["hyp1", "hyp2"]"#.into()));
        let assembler = func_backend(|messages| {
            let msg = last_user_message(messages);
            if msg.contains("Original query") {
                Ok("assembled final answer".into())
            } else {
                // ResultScorer prompt — canonical scorer output shape.
                Ok(r#"{"score": 0.9, "accepted": true, "reason": "good", "summary": "ok"}"#.into())
            }
        });
        let frontier = StubFrontier::new(vec![
            text_response("claude", "answer one"),
            text_response("claude", "answer two"),
        ]);
        let config = EscalationLadderConfig {
            modes: vec![EscalationMode::Question],
            frontier: filter_config().frontier,
            ..Default::default()
        };
        let ladder = EscalationLadder::new(
            config,
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: Some(decomposer),
                assembler: Some(assembler),
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("how do I tune a model?");
        let ctx = EscalationContext {
            request: &request,
            user_text: "how do I tune a model?",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        let resp = ladder.try_escalate(&ctx).await.expect("accepted");
        assert_eq!(read_body_text(resp).await, "assembled final answer");
        assert_eq!(frontier.received_texts().len(), 2, "one frontier call per hypothetical");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn team_mode_accepts_after_judge_verdict() {
        let classifier = func_backend(|_| {
            Ok(r#"{"approach": "decompose", "confidence": 0.9}"#.into())
        });
        let draft = func_backend(|_| Ok(r#"["subtask A", "subtask B"]"#.into()));
        let judge = func_backend(|messages| {
            let msg = last_user_message(messages);
            if msg.contains("\n\nAnswer:\n") {
                Ok(r#"{"accepted": true, "reason": "solved"}"#.into())
            } else {
                Ok("frontier prompt with only the gap".into())
            }
        });
        let frontier = StubFrontier::new(vec![text_response("claude", "frontier solution")]);
        let config = EscalationLadderConfig {
            modes: vec![EscalationMode::Team],
            frontier: filter_config().frontier,
            classifier_parallel: 2,
            ..Default::default()
        };
        let ladder = EscalationLadder::new(
            config,
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: Some(classifier),
                draft: Some(draft),
                judge: Some(judge),
            },
        );
        let request = test_request("solve the hard problem");
        let ctx = EscalationContext {
            request: &request,
            user_text: "solve the hard problem",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        let resp = ladder.try_escalate(&ctx).await.expect("accepted");
        assert_eq!(read_body_text(resp).await, "frontier solution");
        // The frontier received the judge-crafted prompt, not the raw request.
        let received = frontier.received_texts();
        assert!(
            received[0].contains("frontier prompt"),
            "frontier must receive the judge-crafted prompt: {received:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn team_mode_rejects_when_judge_rejects() {
        let classifier = func_backend(|_| Ok(r#"{"approach": "x", "confidence": 0.5}"#.into()));
        let draft = func_backend(|_| Ok(r#"["a"]"#.into()));
        let judge = func_backend(|messages| {
            let msg = last_user_message(messages);
            if msg.contains("\n\nAnswer:\n") {
                Ok(r#"{"accepted": false, "reason": "still gap"}"#.into())
            } else {
                Ok("gap prompt".into())
            }
        });
        let frontier = StubFrontier::new(vec![text_response("claude", "partial")]);
        let config = EscalationLadderConfig {
            modes: vec![EscalationMode::Team],
            frontier: filter_config().frontier,
            classifier_parallel: 1,
            ..Default::default()
        };
        let ladder = EscalationLadder::new(
            config,
            EscalationBackends {
                frontier,
                decomposer: None,
                assembler: None,
                classifier: Some(classifier),
                draft: Some(draft),
                judge: Some(judge),
            },
        );
        let request = test_request("hard");
        let ctx = EscalationContext {
            request: &request,
            user_text: "hard",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        assert!(ladder.try_escalate(&ctx).await.is_none(), "judge reject → no acceptance");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn turnover_marks_session_frontier_owned_and_appends_ledger() {
        let frontier = StubFrontier::new(vec![text_response("claude", "handoff answer")]);
        let config = EscalationLadderConfig {
            modes: vec![EscalationMode::Turnover],
            frontier: filter_config().frontier,
            ..Default::default()
        };
        let ladder = EscalationLadder::new(
            config,
            EscalationBackends {
                frontier: frontier.clone(),
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );

        let session = Arc::new(Mutex::new(DependencySession::new("sess-1")));
        {
            let mut s = session.lock().unwrap();
            s.add_step(SessionStep::new("step-1", "reproduce")).unwrap();
            s.complete_step(
                "step-1",
                StepResult {
                    content: "reproduced the crash".into(),
                    accepted: true,
                    score: Some(0.9),
                    latency_ms: 0,
                    error: None,
                },
            )
            .unwrap();
        }

        let request = test_request("what next?");
        let ctx = EscalationContext {
            request: &request,
            user_text: "what next?",
            model_name: "fast",
            context_cache: None,
            session: Some(&session),
        };
        let resp = ladder.try_escalate(&ctx).await.expect("handoff");
        assert_eq!(read_body_text(resp).await, "handoff answer");

        assert!(
            session.lock().unwrap().is_frontier_owned(),
            "turnover must mark the session frontier-owned"
        );
        let received = frontier.received_texts();
        assert!(
            received[0].contains("reproduced the crash"),
            "turnover must append the session ledger: {received:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exhaustion_returns_none() {
        let frontier = StubFrontier::new(vec![text_response("claude", "call bob@example.com")]);
        let ladder = EscalationLadder::new(
            filter_config(),
            EscalationBackends {
                frontier,
                decomposer: None,
                assembler: None,
                classifier: None,
                draft: None,
                judge: None,
            },
        );
        let request = test_request("email me");
        let ctx = EscalationContext {
            request: &request,
            user_text: "email me",
            model_name: "fast",
            context_cache: None,
            session: None,
        };
        assert!(
            ladder.try_escalate(&ctx).await.is_none(),
            "exhaustion → None so the caller returns fallback_completion"
        );
    }

    #[test]
    fn votes_distribution_counts_approaches() {
        let votes = vec![
            r#"{"approach": "decompose", "confidence": 0.9}"#.to_string(),
            r#"{"approach": "decompose", "confidence": 0.8}"#.to_string(),
            "plain text vote".to_string(),
        ];
        let dist = summarize_votes(&votes);
        assert!(dist.contains("2/3 say: decompose"), "dist: {dist}");
        assert!(dist.contains("1/3 say: plain text vote"), "dist: {dist}");
    }

    #[test]
    fn parse_subtask_array_handles_fenced_json() {
        let arr = parse_subtask_array("```json\n[\"a\", \"b\"]\n```").unwrap();
        assert_eq!(arr, vec!["a", "b"]);
        assert!(parse_subtask_array("not json").is_none());
    }

    #[test]
    fn ladder_builds_from_config_with_transcript_local_roles() {
        // `TranscriptProvider` is a sync `ChatBackend` — the same trait the
        // ladder's local roles use, so builder wiring type-checks.
        let backends = EscalationBackends {
            frontier: StubFrontier::new(vec![]),
            decomposer: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
            assembler: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
            classifier: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
            draft: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
            judge: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
        };
        let ladder = EscalationLadder::new(full_modes_config(), backends);
        assert_eq!(ladder.config().modes.len(), 4);
    }
}
