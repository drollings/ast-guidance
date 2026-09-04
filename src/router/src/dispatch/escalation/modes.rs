//! The four escalation mode implementations (filter / question / team /
//! turnover) plus the single shared frontier transport (`frontier_complete`).
//! All run against the ladder's injected backends and emit `kind="escalation"`
//! audit records via `super::audit::emit_audit`.

use fluent_concurrency::pool::{ResultPool, ResultPoolError};
use fluent_llm::{ChatMessage, Decomposer};
use fluent_wvr::prelude::*;

use crate::config::FrontierConfig;
use crate::dispatch::frontier::DispatchError;
use crate::server::dispatch::render_prompt;
use crate::server::responses::{completion_to_response, make_text_completion, HyperResponse};
use crate::stages::deterministic::DeterministicPreFilter;
use crate::summarization::{ResultScorer, ScoredResult};
use crate::transforms::pii_anonymize::PiiAnonymize;
use crate::transforms::TransformStrategy;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse};

use super::assemble::{
    assemble_answers, judge_verdict, response_text, session_ledger_text, summarize_votes,
    BackendDecomposer, ClassifierSlot, HypotheticalTask, SCORE_ACCEPTANCE_THRESHOLD,
};
use super::audit::emit_audit;
use super::{EscalationContext, Ladder};

impl Ladder {
    /// One-shot dispatch of a full request to the frontier backend — the
    /// single frontier HTTP path shared by all four modes.
    pub(super) async fn frontier_complete(
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
                fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS,
                fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS,
                false,
            )
            .await
    }

    // ── Mode implementations ──────────────────────────────────────────────

    /// filter: PII transform → frontier → stage-1 re-scan; accept if clean.
    pub(super) async fn filter_mode(
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
                emit_audit(
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
        emit_audit(
            "filter",
            accepted,
            &payload,
            &raw,
            trigger,
            &serde_json::json!({}),
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

    /// question: anonymize + decompose → parallel hypothetical frontier
    /// calls (`ResultPool`) → assembler synthesizes → `ResultScorer` gates.
    pub(super) async fn question_mode(
        &self,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        let Some(front) = &self.config.frontier else {
            return Ok(None);
        };
        let (Some(decomposer), Some(assembler)) = (
            self.backends.decomposer.clone(),
            self.backends.assembler.clone(),
        ) else {
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
                                    fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS,
                                    fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS,
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
            emit_audit(
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
                emit_audit(
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
        emit_audit(
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
    pub(super) async fn team_mode(
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
        ) else {
            return Ok(None);
        };
        let slots = self.config.classifier_parallel.max(1);
        let user_text = ctx.user_text.to_string();

        // Parallel classifier slots vote on the approach. Slot diversity comes
        // from a per-slot instruction (the sync `ChatBackend` fixes the
        // temperature at construction; `classifier_parallel` instances of the
        // same model are simulated by varied prompts).
        let votes: Vec<String> = {
            let pool = ResultPool::new(fluent_concurrency::tokio_runtime(), slots, slots, {
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
            });

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
            emit_audit(
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
                emit_audit(
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
        emit_audit(
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
    pub(super) async fn turnover_mode(
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
                emit_audit(
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

        emit_audit(
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
}
