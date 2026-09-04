//! Rigor route - the fixed-pass blue/red/judge protocol.
//!
//! `RigorRoute` wires the VISION's high-stakes verification loop to live
//! backends:
//!
//! 1. **Blue team** produces a candidate answer (plain text).
//! 2. A `DependencySession` checkpoint (`rigor.blue`) is recorded so a
//!    red-team-identified dead end can be **rewound for real**.
//! 3. **Red team** reads the session through `FilteredLedger` at
//!    `Lod::LOD0` (exclusion set = blue's rejected dead ends) and emits a JSON
//!    array of objections.
//! 4. **Judge** emits a structured verdict (`accept` /
//!    `accept_with_caveats` / `reject`) with a confidence.
//! 5. A **material** rejection (any objection severity >= the configured
//!    threshold) rewinds to `rigor.blue` and runs a **second and final** blue
//!    pass with the objections folded in, then judges again.
//! 6. If still rejected, the route resolves to a **targeted interview**
//!    (<= 3 questions derived from the objection descriptions) and, only when
//!    judge confidence is low, marks `frontier_escalation` (an explicit
//!    config value - never "red scored a point").
//!
//! Round count is fixed at `max_passes` (default 2): never a third pass
//! (VISION: terminate, don't loop). Backends are DIP-injected
//! `Arc<dyn ChatBackend>` built exactly once in `main.rs`. There is **no**
//! `Interviewable` trait - the targeted interview is a third, distinct shape
//! from plan's binding-gap closure loop.
//!
//! The prompt constants, message builders, and tolerant parse helpers live in
//! the [`prompts`] submodule.

pub mod prompts;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use common_core::hash::uuid_v4;
use common_core::sync::lock;
use fluent_types::{ContentNode, NodeId};
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};

use crate::config::RigorConfig;
use crate::dag_session::{DependencySession, SessionStep, StepResult};
use crate::dispatch::escalation::LocalBackend;
use crate::ledger::prompt::{LedgerPromptAssembler, LodSpec, PromptBudget, WorkerContext};
use crate::ledger::ContentNodeLedger;
use crate::server::handler::ServerDeps;
use crate::server::responses::{empty_response, HyperResponse};
use crate::views::{FilteredLedger, LedgerView, Lod, ParallelLedger};

use prompts::{
    blue_answer, blue_retry_prompt, chat, dead_end_node_ids, derive_interview, is_reject,
    judge_prompt, parse_judge, parse_objections, verdict_tag, JUDGE_SYSTEM_PROMPT,
    RED_SYSTEM_PROMPT,
};

/// Context passed to the rigor route's execute method. Carries the minimal
/// information needed for the 3-pass blue/red/judge protocol.
#[derive(Clone)]
pub struct RigorContext {
    pub user_message: String,
    pub session_id: String,
    pub model_endpoint: String,
    /// The `DependencySession` for checkpoint/rewind between passes.
    /// `None` degrades to a sessionless run (no checkpoint, no rewind, no
    /// red-team ledger view).
    pub session: Option<Arc<Mutex<DependencySession>>>,
    /// The shared `ContentNodeLedger` whose store the red team reads at LOD0.
    /// `None` degrades the red pass to the blue answer only.
    pub ledger: Option<Arc<ContentNodeLedger>>,
    /// The named instance the blue pass served on. `Some` lets the
    /// blue-pass completion record a fork KV snapshot on it; `None` skips the
    /// save (degrade, never a crash).
    pub kv_instance: Option<String>,
}

pub struct RigorRoute {
    /// Whether to support KV-cache checkpoint/rewind for dead-end recovery.
    pub kv_cache_enabled: bool,
    blue: Option<LocalBackend>,
    red: Option<LocalBackend>,
    judge: Option<LocalBackend>,
    /// Route behavior config: `max_passes`, material-rejection threshold,
    /// escalation trigger.
    cfg: RigorConfig,
    /// Optional `LedgerPromptAssembler`: when a ledger is attached, the
    /// judge renders its review prompt over the session ledger through the
    /// assembler's budget/relevance rules. The red team keeps its LOD0
    /// `FilteredLedger` view unchanged (it dereferences LOD0 by design).
    assembler: Option<LedgerPromptAssembler>,
    budget: PromptBudget,
    lod_spec: LodSpec,
}

#[derive(Debug, Clone)]
pub struct RigoResult {
    pub blue_answer: String,
    pub red_objections: Vec<RedObjection>,
    pub judge_verdict: JudgeVerdict,
    pub frontier_escalation: bool,
    /// The targeted-interview fallback: <= 3 questions derived from the
    /// objection descriptions when the final pass still rejects.
    pub interview_questions: Vec<String>,
    /// Whether a real `rewind_to_checkpoint("rigor.blue")` ran between passes.
    pub rewound: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedObjection {
    pub category: String,
    pub description: String,
    pub severity: f64,
    /// The claim's ledger node id the objection dereferences at LOD0.
    /// `None` when the objection is not claim-anchored.
    #[serde(default)]
    pub target_claim: Option<NodeId>,
}

#[derive(Debug, Clone)]
pub enum JudgeVerdict {
    Accept,
    AcceptWithCaveats { caveats: Vec<String> },
    Reject { reasons: Vec<String> },
}

impl Default for RigorRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl RigorRoute {
    #[must_use]
    pub fn new() -> Self {
        Self {
            kv_cache_enabled: false,
            blue: None,
            red: None,
            judge: None,
            cfg: RigorConfig::default(),
            assembler: None,
            budget: PromptBudget::from_tokens_default(8192),
            lod_spec: LodSpec::full(),
        }
    }

    #[must_use]
    pub fn with_kv_cache(mut self) -> Self {
        self.kv_cache_enabled = true;
        self
    }

    /// Attach the `LedgerPromptAssembler` so the judge's review prompt renders
    /// the session ledger at a budget/relevance-matched fidelity.
    #[must_use]
    pub fn with_prompt_assembler(
        mut self,
        assembler: LedgerPromptAssembler,
        budget: PromptBudget,
        lod_spec: LodSpec,
    ) -> Self {
        self.assembler = Some(assembler);
        self.budget = budget;
        self.lod_spec = lod_spec;
        self
    }

    /// Attach the blue-team (candidate-answer) backend. Mock-injectable.
    #[must_use]
    pub fn with_blue_backend(mut self, backend: LocalBackend) -> Self {
        self.blue = Some(backend);
        self
    }

    /// Attach the red-team (objections) backend. Mock-injectable.
    #[must_use]
    pub fn with_red_backend(mut self, backend: LocalBackend) -> Self {
        self.red = Some(backend);
        self
    }

    /// Attach the judge backend. Mock-injectable.
    #[must_use]
    pub fn with_judge_backend(mut self, backend: LocalBackend) -> Self {
        self.judge = Some(backend);
        self
    }

    /// Attach the route behavior config (passes / thresholds).
    #[must_use]
    pub fn with_config(mut self, cfg: RigorConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Resolve a role backend, degrading to an explicit error when missing
    /// (never a silent no-op).
    fn role_backend(&self, name: &str) -> Result<LocalBackend, RigorError> {
        let backend = match name {
            "blue" => self.blue.clone(),
            "red" => self.red.clone(),
            "judge" => self.judge.clone(),
            _ => return Err(RigorError::Unconfigured(name.to_string())),
        };
        backend.ok_or_else(|| RigorError::Unconfigured(name.to_string()))
    }

    /// Execute the fixed-pass rigor protocol:
    /// blue -> checkpoint -> red -> judge -> (rewind + second blue + judge) ->
    /// interview/escalation.
    ///
    /// A material rejection rewinds the session for real between the first
    /// blue pass and the re-run; the final resolution is never an open loop.
    pub async fn execute(&self, ctx: &RigorContext) -> Result<RigoResult, RigorError> {
        let blue = self.role_backend("blue")?;
        let red = self.role_backend("red")?;
        let judge = self.role_backend("judge")?;

        Self::register_steps(ctx);

        // Pass 1: blue.
        let mut answer = blue_answer(&blue, &ctx.user_message).await?;
        self.complete_blue_step(ctx, &answer);

        // Red + judge over the (no-dead-end) view.
        let objections = Self::red_pass(ctx, &red, &answer).await?;
        let (mut verdict, mut confidence) =
            self.judge_pass(ctx, &judge, &answer, &objections).await?;

        let mut rewound = false;
        let mut interview_questions = Vec::new();
        let mut frontier_escalation = false;

        // A material rejection -> one rewind + a second, final blue pass.
        if is_reject(&verdict) && self.material_rejection(&objections) && self.cfg.max_passes.rounds() > 1 {
            Self::record_blue_dead_end(ctx, &answer);
            rewound = self.rewind_to_blue(ctx);
            let refocused = blue_retry_prompt(&ctx.user_message, &answer, &objections);
            answer = blue_answer(&blue, &refocused).await?;
            self.complete_blue_step(ctx, &answer);
            let objections2 = Self::red_pass(ctx, &red, &answer).await?;
            let (v2, c2) = self.judge_pass(ctx, &judge, &answer, &objections2).await?;
            verdict = v2;
            confidence = c2;
        }

        if is_reject(&verdict) {
            interview_questions = derive_interview(&objections);
            if confidence < self.cfg.escalation_confidence.get() {
                frontier_escalation = true;
            }
        }

        crate::audit::emit(
            "rigor",
            serde_json::json!({
                "passes": if rewound { 2 } else { 1 },
                "rewound": rewound,
                "verdict": verdict_tag(&verdict),
                "objections": objections.len(),
                "frontier_escalation": frontier_escalation,
                "interview_questions": interview_questions.len(),
            }),
        );

        Ok(RigoResult {
            blue_answer: answer,
            red_objections: objections,
            judge_verdict: verdict,
            frontier_escalation,
            interview_questions,
            rewound,
        })
    }

    /// Register the rigor steps on the session idempotently (`add_step`
    /// returns `DuplicateNode` - only add when absent), and set the model if
    /// unset (KV snapshot keying requires one, `dag_session.rs:354-363`).
    fn register_steps(ctx: &RigorContext) {
        let Some(session) = &ctx.session else {
            return;
        };
        let mut s = lock(session);
        if s.model.is_none() {
            s.set_model(ctx.model_endpoint.clone());
        }
        if s.get_step("rigor.blue").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.blue", "blue team candidate answer").with_checkpoint(),
            );
        }
        if s.get_step("rigor.red").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.red", "red team objections")
                    .with_depends(vec!["rigor.blue".into()]),
            );
        }
        if s.get_step("rigor.judge").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.judge", "judge verdict")
                    .with_depends(vec!["rigor.red".into()]),
            );
        }
    }

    /// Complete `rigor.blue` (records the checkpoint via `complete_step`).
    /// When KV-cache is enabled this also saves a fork snapshot on the blue
    /// instance so the subsequent `rewind_to_checkpoint("rigor.blue")` finds
    /// real metadata (D7). The save is best-effort - it logs and never fails
    /// the request.
    fn complete_blue_step(&self, ctx: &RigorContext, answer: &str) {
        let Some(session) = &ctx.session else {
            return;
        };
        let mut s = lock(session);
        if s.get_step("rigor.blue").is_some() {
            let _ = s.complete_step(
                "rigor.blue",
                StepResult {
                    content: answer.to_string(),
                    accepted: true,
                    score: None,
                    latency_ms: 0,
                    error: None,
                },
            );
        }
        if self.kv_cache_enabled {
            Self::save_blue_snapshot(ctx, &mut s);
        }
    }

    /// Save the blue-pass KV snapshot (D7): record a snapshot named after the
    /// blue step on `ctx.kv_instance` so a later `rewind_to_checkpoint` finds
    /// real metadata. Requires a session carrying a `SnapshotStore` (with a
    /// fork handle), a model name for keying, and a `kv_instance`; any missing
    /// piece degrades to a logged no-op - never a crash, never a fabricated
    /// lookup.
    fn save_blue_snapshot(ctx: &RigorContext, s: &mut DependencySession) {
        let Some(kv) = s.kv_cache() else {
            return;
        };
        let Some(instance) = &ctx.kv_instance else {
            return;
        };
        let Some(model) = s.model.clone() else {
            return;
        };
        let name = "rigor-blue-rigor.blue";
        match kv.save_snapshot(&model, s.adapter.as_deref(), &s.session_id, name, instance) {
            Ok(()) => tracing::info!(
                target: "router.rigor",
                snapshot = %name,
                instance = %instance,
                model = %model,
                "kv cache snapshot saved for blue pass",
            ),
            Err(e) => tracing::warn!(
                target: "router.rigor",
                snapshot = %name,
                instance = %instance,
                error = %e,
                "kv cache snapshot save failed - continuing (never fails the request)",
            ),
        }
    }

    /// Rewind to `rigor.blue` (real `rewind_to_checkpoint`). Steps are reset
    /// to `Pending` with result data preserved for audit; the KV snapshot, if
    /// restored, is logged and carried on the session (`pending_kv_fields`) so
    /// the next dispatch sets the fork's `snapshot`/`instance`/`id_slot`
    /// request fields. Returns whether a rewind ran.
    fn rewind_to_blue(&self, ctx: &RigorContext) -> bool {
        let Some(session) = &ctx.session else {
            return false;
        };
        let mut s = lock(session);
        match s.rewind_to_checkpoint("rigor.blue") {
            Ok(snapshot) => {
                if let Some(snap) = snapshot {
                    tracing::info!(
                        target: "router.rigor",
                        snapshot_name = %snap.snapshot_name,
                        instance = ?snap.instance,
                        id_slot = 0,
                        kv_cache_enabled = self.kv_cache_enabled,
                        "kv cache snapshot restored on rigor rewind - snapshot/instance passed to next dispatch"
                    );
                }
                true
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.rigor",
                    error = %e,
                    "rigor rewind failed - continuing with current answer"
                );
                false
            }
        }
    }

    /// Persist the first blue answer as a rejected dead-end ledger node so the
    /// second red pass's filtered view excludes it. Best-effort; a
    /// missing ledger degrades silently.
    fn record_blue_dead_end(ctx: &RigorContext, answer: &str) {
        let Some(ledger) = &ctx.ledger else {
            return;
        };
        let node = ContentNode {
            id: None,
            name: "rigor-blue-dead-end".into(),
            source: "rigor".into(),
            content_hash: 0,
            lod: vec![
                answer.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ],
            embedding: None,
            capabilities: None,
            session_id: Some(ctx.session_id.clone()),
            request_id: None,
            role: Some("blue".into()),
            turn_index: None,
            accepted: Some(false),
            acceptance_score: None,
            active_lod: Some(0),
            parent_id: None,
            step_id: Some("rigor.blue".into()),
            step_status: None,
            metadata: None,
            created_at: Some(common_core::now_secs()),
            annotation: None,
        };
        let _ = ledger.record_content_node(&node);
    }

    /// Red-pass prompt: the blue answer plus the session rendered through
    /// `FilteredLedger` at LOD0, excluding blue's rejected dead ends.
    /// No ledger/session -> the blue answer alone (documented degradation).
    async fn red_pass(
        ctx: &RigorContext,
        red: &LocalBackend,
        answer: &str,
    ) -> Result<Vec<RedObjection>, RigorError> {
        let mut material = format!("Blue team's candidate answer:\n{answer}\n");
        let rendered = Self::red_team_view(ctx);
        if !rendered.is_empty() {
            material.push_str("\nSession ledger (LOD0, dead ends excluded):\n");
            material.push_str(&rendered);
        }
        let raw = chat(red, RED_SYSTEM_PROMPT, &material)
            .await
            .map_err(|e| RigorError::RedTeam(e.to_string()))?;
        parse_objections(&raw).map_err(RigorError::RedTeam)
    }

    /// A red-team view: `ParallelLedger::for_session(...).with_default_lod
    /// (Lod::LOD0)` wrapped in `FilteredLedger` excluding dead-end node ids.
    fn red_team_view(ctx: &RigorContext) -> String {
        let Some(ledger) = &ctx.ledger else {
            return String::new();
        };
        let store = ledger.node_store().clone();
        let base = ParallelLedger::for_session(store, &ctx.session_id).with_default_lod(Lod::LOD0);
        let excluded = dead_end_node_ids(ctx);
        FilteredLedger::new(base, excluded).render()
    }

    async fn judge_pass(
        &self,
        ctx: &RigorContext,
        judge: &LocalBackend,
        answer: &str,
        objections: &[RedObjection],
    ) -> Result<(JudgeVerdict, f64), RigorError> {
        let prompt = judge_prompt(answer, objections);
        // Fold the session ledger (assembled through the budget/relevance
        // rules) into the judge's review material when a ledger + assembler are
        // attached. Additive — a route without them keeps today's prompt.
        let ledger_ctx = self.assembled_judge_context(ctx);
        let prompt = if ledger_ctx.is_empty() {
            prompt
        } else {
            format!("Session ledger context:\n{ledger_ctx}\n\n{prompt}")
        };
        let raw = chat(judge, JUDGE_SYSTEM_PROMPT, &prompt)
            .await
            .map_err(|e| RigorError::Judge(e.to_string()))?;
        parse_judge(&raw).map_err(RigorError::Judge)
    }

    /// Render the session ledger through the `LedgerPromptAssembler` for the
    /// judge's review prompt. Empty when no assembler or ledger is
    /// attached (the judge degrades to today's prompt). The rendered `body` is
    /// the assembled context; the per-node fidelity `node_plan` is audited.
    fn assembled_judge_context(&self, ctx: &RigorContext) -> String {
        let Some(assembler) = self.assembler else {
            return String::new();
        };
        let Some(ledger) = &ctx.ledger else {
            return String::new();
        };
        let store = ledger.node_store().clone();
        let view = ParallelLedger::for_session(store, &ctx.session_id);
        let assembled = assembler.assemble(
            &view,
            &WorkerContext::new("judge", "Review the candidate answer against the ledger."),
            &self.budget,
            None,
            &self.lod_spec,
        );
        crate::audit::emit(
            "prompt",
            serde_json::json!({
                "session_id": ctx.session_id,
                "role": "rigor_judge",
                "budget_used": assembled.budget_used,
                "node_plan": assembled
                    .node_plan
                    .iter()
                    .map(|(id, lod)| serde_json::json!([id.as_int(), lod.as_u8()]))
                    .collect::<Vec<_>>(),
            }),
        );
        assembled.body
    }

    /// Whether any objection is material (severity >= the configured
    /// threshold) - the trigger for rewind, not "the judge rejected".
    fn material_rejection(&self, objections: &[RedObjection]) -> bool {
        objections
            .iter()
            .any(|o| o.severity >= self.cfg.severity_threshold.get())
    }
}


#[derive(Debug, thiserror::Error)]
pub enum RigorError {
    #[error("rigor role backend not configured: {0}")]
    Unconfigured(String),
    #[error("blue team error: {0}")]
    BlueTeam(String),
    #[error("red team error: {0}")]
    RedTeam(String),
    #[error("judge error: {0}")]
    Judge(String),
}
pub async fn handle_rigor_request(
    req: hyper::Request<hyper::body::Incoming>,
    deps: ServerDeps,
) -> Result<HyperResponse, std::convert::Infallible> {
    let ServerDeps {
        stats,
        max_payload,
        rigor_route,
        sessions,
        ledger,
        classifier,
        models,
        ..
    } = &deps;
    let Some(route) = rigor_route else {
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(crate::server::responses::error_response(
            hyper::StatusCode::SERVICE_UNAVAILABLE,
            "rigor route not configured",
        ));
    };

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(crate::server::responses::error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("body read error: {e}"),
            ));
        }
    };
    if body_bytes.len() > *max_payload {
        return Ok(empty_response(hyper::StatusCode::PAYLOAD_TOO_LARGE));
    }
    let body: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(crate::server::responses::error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {e}"),
            ));
        }
    };

    let message = body
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if message.is_empty() {
        return Ok(crate::server::responses::error_response(
            hyper::StatusCode::BAD_REQUEST,
            "missing 'message'",
        ));
    }
    let session_id = body
        .get("session_id")
        .and_then(serde_json::Value::as_str)
        .map_or_else(uuid_v4, ToOwned::to_owned);

    // The session model key: the classifier model when known, else a stable
    // placeholder (KV snapshot keying needs *a* model, `dag_session.rs:354`).
    let model_endpoint = classifier
        .as_ref()
        .map_or_else(|| "fast".into(), |(name, _)| name.clone());

    // Thread the registry session + shared ledger into the context so
    // checkpoint/rewind and the red-team LOD0 view are load-bearing.
    let session = sessions.as_ref().map(|s| s.get_or_create(&session_id));
    // The session model is set so KV snapshot save/rewind can key by it
    // (`dag_session.rs` refuses to key without a model). The blue instance is
    // the model's internal work group (the pool).
    if let Some(session) = &session {
        let mut s = lock(session);
        s.set_model(model_endpoint.clone());
    }
    let kv_instance = models
        .get(&model_endpoint)
        .and_then(crate::config::ModelEntry::pool_qualifier);
    let ledger = ledger.clone();

    let ctx = RigorContext {
        user_message: message.to_string(),
        session_id,
        model_endpoint,
        session,
        ledger,
        kv_instance,
    };

    match route.execute(&ctx).await {
        Ok(result) => {
            let response = if matches!(
                &result.judge_verdict,
                crate::routes::rigor::JudgeVerdict::Reject { .. }
            ) {
                // A final rejection resolves to a targeted interview.
                serde_json::json!({
                    "status": "clarify",
                    "questions": result.interview_questions,
                    "rewound": result.rewound,
                })
            } else {
                let mut executed = serde_json::json!({
                    "status": "executed",
                    "answer": result.blue_answer,
                    "verdict": verdict_tag(&result.judge_verdict),
                    "rewound": result.rewound,
                });
                if let crate::routes::rigor::JudgeVerdict::AcceptWithCaveats { ref caveats } =
                    result.judge_verdict
                {
                    executed["caveats"] = serde_json::to_value(caveats).unwrap_or_default();
                }
                if result.frontier_escalation {
                    executed["frontier_escalation"] = serde_json::json!(true);
                }
                executed
            };
            Ok(crate::server::responses::json_response(
                hyper::StatusCode::OK,
                &response,
            ))
        }
        Err(RigorError::Unconfigured(name)) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            Ok(crate::server::responses::error_response(
                hyper::StatusCode::SERVICE_UNAVAILABLE,
                &format!("rigor role backend not configured: {name}"),
            ))
        }
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            Ok(crate::server::responses::error_response(
                hyper::StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
            ))
        }
    }
}
#[cfg(test)]
#[path = "../../../tests/routes_rigor_mod.rs"]
mod tests;
