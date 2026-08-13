use std::sync::atomic::Ordering;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_llm::client::ChatBackend;
use http_body_util::BodyExt;

use crate::charts::binding::Entity;
use crate::charts::binding::ENTITIES_META_KEY;
use crate::charts::execute::{ChartExecOptions, ChartExecutionPlan, ChartExecutionSummary};
use crate::charts::extract::WorkflowExtractor;
use crate::charts::select::{ChartFit, ChartSelector};
use crate::charts::store::ChartStore;
use crate::config::ChartsConfig;
use crate::ledger::prompt::{LedgerPromptAssembler, LodSpec, PromptBudget, WorkerContext};
use crate::server::responses::{empty_response, HyperResponse, ServerStats};
use crate::views::ParallelLedger;

pub struct PlanRoute {
    /// The chart store — the single owner of the workflow_library index
    /// path. Shared via `Arc` so the M7 `ChartSelector` and the route read
    /// from the same boot-loaded store.
    charts: Arc<ChartStore>,
    /// Adjudicator backend for chart selection (M7 step 3). `None` degrades
    /// selection to deterministic + HNSW only. Also doubles as the rubric
    /// judge backend for chart execution (M4.1).
    selector_backend: Option<Arc<dyn ChatBackend>>,
    /// Reranker backend for chart selection (M7 step 2.5). `None` skips
    /// candidate re-ranking (Step 2 → Step 3 directly).
    reranker_backend: Option<Arc<dyn ChatBackend>>,
    /// Backend that executes a selected chart's targets (M4.1 server-side
    /// execution). `None` degrades an exact fit to a fresh draft.
    execution_backend: Option<Arc<dyn ChatBackend>>,
    /// Bounds concurrent chart-target LLM calls during execution.
    limiter: Arc<Limiter>,
    /// Chart-selection configuration (thresholds, max candidates).
    cfg: ChartsConfig,
    /// M10 dispatch post-processing hook: distills successful dispatches into
    /// draft charts. `None` when extraction is not configured (opt-in).
    extractor: Option<Arc<WorkflowExtractor>>,
    /// Optional session-context renderer (M5.3): when a ledger store + prompt
    /// assembler are attached, the selector/adjudicator models receive the
    /// session ledger rendered through the assembler's budget/relevance rules.
    /// `None` keeps today's blank-slate plan prompts (byte-identical).
    prompt_ctx: Option<PromptAssemblerCtx>,
}

/// The session-context renderer for the plan route (M5.3): a shared `ContentNodeStore`
/// plus a `LedgerPromptAssembler` and its budget/fidelity band. Pure — it only
/// renders, it never triggers LOD derivation.
#[derive(Clone)]
pub struct PromptAssemblerCtx {
    store: Arc<crate::node_store::ContentNodeStore>,
    assembler: LedgerPromptAssembler,
    budget: PromptBudget,
    lod_spec: LodSpec,
}

impl PromptAssemblerCtx {
    pub fn new(
        store: Arc<crate::node_store::ContentNodeStore>,
        assembler: LedgerPromptAssembler,
        budget: PromptBudget,
        lod_spec: LodSpec,
    ) -> Self {
        Self {
            store,
            assembler,
            budget,
            lod_spec,
        }
    }

    /// Render a session's ledger through the assembler into a context block
    /// (`""` for an empty session — the caller keeps the blank-slate prompt).
    /// Audits the fidelity plan (`kind = "prompt"`, role = `"plan_selector"`).
    fn render(&self, session_id: &str) -> String {
        let view = ParallelLedger::for_session(Arc::clone(&self.store), session_id);
        let assembled = self.assembler.assemble(
            &view,
            &WorkerContext::new("chart selector", "Select a chart against the session context."),
            &self.budget,
            None,
            &self.lod_spec,
        );
        crate::audit::emit(
            "prompt",
            serde_json::json!({
                "session_id": session_id,
                "role": "plan_selector",
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
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub source: PlanSource,
    pub interview_questions: Vec<String>,
    /// Raw gap dep names behind the rendered questions (M8 round-trip: the
    /// handler echoes these back so the interview stays exactly one round).
    pub gaps: Vec<String>,
    pub gaps_filled: Vec<String>,
    /// Execution summary when a chart was compiled + executed server-side
    /// (Exact / interviewed-Exact hit). `None` for clarify and fresh-draft.
    pub summary: Option<ChartExecutionSummary>,
    /// Selection provenance label (`"exact"` / `"partial"` / `"mismatch"`).
    pub fit: Option<String>,
    /// Selection confidence in `[0, 1]` (audit trail provenance).
    pub score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanSource {
    HnswHit,
    TemplateAdapted,
    FreshDraft,
}

impl Default for PlanRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanRoute {
    pub fn new() -> Self {
        Self {
            charts: Arc::new(ChartStore::new(None)),
            selector_backend: None,
            reranker_backend: None,
            execution_backend: None,
            limiter: Arc::new(Limiter::new(4)),
            cfg: ChartsConfig::default(),
            extractor: None,
            prompt_ctx: None,
        }
    }

    /// Attach the session-context renderer (M5.3). When set, the plan route's
    /// `plan_for_session`/`plan_interviewed_for_session` fold the session
    /// ledger (rendered via the `LedgerPromptAssembler`) into the
    /// selector/adjudicator prompts. Opt-in — a route without it is
    /// byte-identical to today.
    #[must_use]
    pub fn with_prompt_assembler(mut self, ctx: PromptAssemblerCtx) -> Self {
        self.prompt_ctx = Some(ctx);
        self
    }

    /// Attach the boot-loaded chart store. The store is shared (`Arc`) so the
    /// M7 `ChartSelector` can be built over the same instance.
    #[must_use]
    pub fn with_chart_store(mut self, store: Arc<ChartStore>) -> Self {
        self.charts = store;
        self
    }

    /// Attach the adjudicator backend used by chart selection (M7 step 3).
    /// Mock-injectable.
    #[must_use]
    pub fn with_selector_backend(mut self, backend: Arc<dyn ChatBackend>) -> Self {
        self.selector_backend = Some(backend);
        self
    }

    /// Attach the reranker backend used by chart selection (M7 step 2.5).
    /// Mock-injectable.
    #[must_use]
    pub fn with_reranker_backend(mut self, backend: Arc<dyn ChatBackend>) -> Self {
        self.reranker_backend = Some(backend);
        self
    }

    /// Attach the backend that executes a selected chart's targets (M4.1).
    /// Mock-injectable.
    #[must_use]
    pub fn with_execution_backend(mut self, backend: Arc<dyn ChatBackend>) -> Self {
        self.execution_backend = Some(backend);
        self
    }

    /// Attach a limiter bounding concurrent chart-target LLM calls.
    #[must_use]
    pub fn with_limiter(mut self, limiter: Arc<Limiter>) -> Self {
        self.limiter = limiter;
        self
    }

    /// Attach the chart-selection configuration.
    #[must_use]
    pub fn with_charts_config(mut self, cfg: ChartsConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Attach the M10 dispatch post-processing hook. `None` disables the
    /// learning loop for this route.
    #[must_use]
    pub fn with_workflow_extractor(mut self, extractor: Arc<WorkflowExtractor>) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// The dispatch post-processing hook, if configured.
    pub fn workflow_extractor(&self) -> Option<&Arc<WorkflowExtractor>> {
        self.extractor.as_ref()
    }

    /// Borrow the chart store.
    pub fn chart_store(&self) -> &ChartStore {
        self.charts.as_ref()
    }

    /// Plan a request against the chart library (M7), executing server-side.
    ///
    /// Selection outcome drives the returned plan:
    ///
    /// - `Exact`: compile + execute the chart under Zone supervision, `source
    ///   = HnswHit`, with the execution summary populated.
    /// - `Partial { gaps }`: `source = TemplateAdapted` with the gaps turned
    ///   into `interview_questions` (≤ `CHART_MAX_INTERVIEW_QUESTIONS`),
    ///   `summary = None`.
    /// - `Mismatch`: `source = FreshDraft`, `summary = None` (fall through to
    ///   blank-slate planning).
    ///
    /// `gaps_filled` is reserved for the M8 interview loop.
    pub async fn plan(&self, user_message: &str, entities: &[Entity]) -> PlanResult {
        self.plan_inner(user_message, entities, false).await
    }

    /// Plan against a session ledger (M5.3): when the route has a prompt
    /// assembler attached and a `session_id` is given, the session context is
    /// rendered through the assembler and folded into the selector/adjudicator
    /// prompt so chart selection follows the same budget/relevance rules.
    /// Without a session id (or assembler) it is identical to [`Self::plan`].
    pub async fn plan_for_session(
        &self,
        session_id: Option<&str>,
        user_message: &str,
        entities: &[Entity],
    ) -> PlanResult {
        let enriched = self.enrich_with_context(session_id, user_message);
        self.plan_inner(&enriched, entities, false).await
    }

    /// Render the session context (M5.3) and prepend it to the selector's user
    /// message. Returns the message unchanged when no session id or assembler
    /// is attached (byte-identical to today).
    fn enrich_with_context(&self, session_id: Option<&str>, user_message: &str) -> String {
        let (Some(session_id), Some(ctx)) = (session_id, &self.prompt_ctx) else {
            return user_message.to_string();
        };
        let rendered = ctx.render(session_id);
        if rendered.is_empty() {
            return user_message.to_string();
        }
        format!("Session ledger context:\n{rendered}\n\nRequest:\n{user_message}")
    }

    /// Session-aware variant of [`Self::plan_interviewed`] (M5.3). See
    /// [`Self::plan_for_session`].
    pub async fn plan_interviewed_for_session(
        &self,
        session_id: Option<&str>,
        user_message: &str,
        entities: &[Entity],
        prior_gaps: &[String],
    ) -> PlanResult {
        let enriched = self.enrich_with_context(session_id, user_message);
        let mut result = self.plan_inner(&enriched, entities, true).await;
        if result.source == PlanSource::HnswHit {
            result.source = PlanSource::TemplateAdapted;
            result.gaps_filled = prior_gaps.to_vec();
        }
        result
    }

    /// Round-2 entry for the one-round interview loop (M8).
    ///
    /// The client's answers have been turned into `entities` (kind = the gap
    /// dep name). Re-binds and:
    ///
    /// - `Exact` now → `source = TemplateAdapted` with `gaps_filled` set to
    ///   the previously-asked gaps, and the chart executed server-side.
    /// - Still `Partial`/`Mismatch` → `source = FreshDraft`. The interview is
    ///   one round, never open-ended (VISION: terminate, don't loop).
    pub async fn plan_interviewed(
        &self,
        user_message: &str,
        entities: &[Entity],
        prior_gaps: &[String],
    ) -> PlanResult {
        let mut result = self.plan_inner(user_message, entities, true).await;
        if result.source == PlanSource::HnswHit {
            result.source = PlanSource::TemplateAdapted;
            result.gaps_filled = prior_gaps.to_vec();
        }
        result
    }

    /// Shared selection+binding+fit pipeline for both interview rounds.
    async fn plan_inner(&self, user_message: &str, entities: &[Entity], retry: bool) -> PlanResult {
        let mut selector = ChartSelector::new(
            self.charts.clone(),
            self.selector_backend.clone(),
            self.cfg.clone(),
        );
        if let Some(reranker) = &self.reranker_backend {
            selector = selector.with_reranker(reranker.clone());
        }
        let selection = match selector.select(user_message, entities) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(
                    target: "router.plan",
                    error = %e,
                    "chart selection failed — falling through to fresh draft"
                );
                return fresh_draft();
            }
        };

        match selection.fit {
            ChartFit::Exact => {
                let Some(chart) = self.charts.get(&selection.chart) else {
                    tracing::error!(
                        target: "router.plan",
                        chart = %selection.chart,
                        "selected chart is no longer in the store"
                    );
                    return fresh_draft();
                };
                self.execute_chart(&chart, user_message, entities, "exact", selection.score)
                    .await
            }
            ChartFit::Partial { gaps } => {
                if retry {
                    // Second failure → terminate the interview, FreshDraft.
                    tracing::warn!(
                        target: "router.plan",
                        chart = %selection.chart,
                        remaining_gaps = ?gaps,
                        "interview round did not close all gaps — fresh draft"
                    );
                    fresh_draft()
                } else {
                    let mut questions: Vec<String> = gaps.iter().map(|g| gap_prompt(g)).collect();
                    questions.truncate(crate::charts::CHART_MAX_INTERVIEW_QUESTIONS);
                    PlanResult {
                        source: PlanSource::TemplateAdapted,
                        interview_questions: questions,
                        gaps,
                        gaps_filled: Vec::new(),
                        summary: None,
                        fit: Some("partial".into()),
                        score: Some(selection.score),
                    }
                }
            }
            ChartFit::Mismatch => fresh_draft(),
        }
    }

    /// Compile + execute an exact-selected chart under Zone supervision.
    ///
    /// A missing `execution_backend` or a compile error degrades to a fresh
    /// draft (never a crash): the chart library is advisory, not mandatory.
    async fn execute_chart(
        &self,
        chart: &crate::charts::ChartDef,
        user_message: &str,
        entities: &[Entity],
        fit: &str,
        score: f64,
    ) -> PlanResult {
        let Some(backend) = self.execution_backend.clone() else {
            tracing::error!(
                target: "router.plan",
                chart = %chart.name,
                "no execution backend configured — exact fit degrades to fresh draft"
            );
            return fresh_draft();
        };
        let base_ctx = plan_ctx(user_message, entities);
        let plan = match ChartExecutionPlan::compile(chart, entities, &backend, &self.limiter) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    target: "router.plan",
                    chart = %chart.name,
                    error = %e,
                    "exact-selected chart failed to compile"
                );
                return fresh_draft();
            }
        };
        let opts = ChartExecOptions {
            runtime: fluent_concurrency::tokio_runtime(),
            judge: self.selector_backend.clone(),
            cache: None,
            metrics: None,
            health: Some(self.charts.clone()),
            fit: Some(fit.into()),
            score: Some(score),
            ..Default::default()
        };
        match plan.execute(&base_ctx, &opts).await {
            Ok(summary) => PlanResult {
                source: PlanSource::HnswHit,
                interview_questions: Vec::new(),
                gaps: Vec::new(),
                gaps_filled: Vec::new(),
                summary: Some(summary),
                fit: Some(fit.into()),
                score: Some(score),
            },
            Err(e) => {
                tracing::error!(
                    target: "router.plan",
                    chart = %chart.name,
                    error = %e,
                    "chart execution failed"
                );
                fresh_draft()
            }
        }
    }
}

/// A `FreshDraft` plan: no chart hit, planning falls through to a blank slate.
fn fresh_draft() -> PlanResult {
    PlanResult {
        source: PlanSource::FreshDraft,
        interview_questions: Vec::new(),
        gaps: Vec::new(),
        gaps_filled: Vec::new(),
        summary: None,
        fit: None,
        score: None,
    }
}

/// Render an interview question for a missing binding gap.
fn gap_prompt(gap: &str) -> String {
    format!("Please provide the missing input: {gap}")
}

/// Build the base execution `WorkContext` carrying the request + bound
/// entities (the chart stages re-bind from the structured `entities` at
/// execution time — see `ChartPromptStage`).
fn plan_ctx(user_message: &str, entities: &[Entity]) -> fluent_wvr::WorkContext {
    let request_json = serde_json::json!({
        "model": "chart",
        "messages": [{"role": "user", "content": user_message}]
    });
    let mut ctx = fluent_wvr::WorkContext::default();
    ctx.set_structured("request", &request_json);
    if !entities.is_empty() {
        ctx.set_structured(ENTITIES_META_KEY, &entities);
    }
    ctx
}



pub async fn handle_plan_request(
    req: hyper::Request<hyper::body::Incoming>,
    plan_route: Option<Arc<PlanRoute>>,
    max_payload: usize,
    stats: &ServerStats,
) -> Result<HyperResponse, std::convert::Infallible> {
    let Some(route) = plan_route else {
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(crate::server::responses::error_response(
            hyper::StatusCode::SERVICE_UNAVAILABLE,
            "plan route not configured",
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
    if body_bytes.len() > max_payload {
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

    let entities: Vec<crate::charts::binding::Entity> = body
        .get("entities")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|e| serde_json::from_value(e.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    let retry = body
        .get("retry")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let prior_gaps: Vec<String> = body
        .get("gaps")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    // M5.3: when the plan body carries a session_id and the route has a prompt
    // assembler attached, the selector/adjudicator reads the session ledger
    // through the assembler's budget/relevance rules.
    let session_id = body.get("session_id").and_then(serde_json::Value::as_str);

    let result = if retry {
        route
            .plan_interviewed_for_session(session_id, message, &entities, &prior_gaps)
            .await
    } else {
        route
            .plan_for_session(session_id, message, &entities)
            .await
    };

    let response = match result.source {
        PlanSource::FreshDraft => {
            serde_json::json!({ "status": "fresh_draft", "source": "fresh_draft" })
        }
        PlanSource::HnswHit => plan_executed_response("hnsw_hit", &result),
        PlanSource::TemplateAdapted => {
            if result.interview_questions.is_empty() {
                plan_executed_response("template_adapted", &result)
            } else {
                serde_json::json!({
                    "status": "clarify",
                    "source": "template_adapted",
                    "questions": result.interview_questions,
                    "gaps": result.gaps,
                })
            }
        }
    };
    Ok(crate::server::responses::json_response(
        hyper::StatusCode::OK,
        &response,
    ))
}

/// Build the D3 `/v1/plan` "executed" response: execution results, not a
/// compiled graph. Carries selection provenance (`fit`/`score`) and the
/// execution summary (`final_output`/`accepted`/`audit`) when the chart ran.
pub fn plan_executed_response(
    source: &str,
    result: &PlanResult,
) -> serde_json::Value {
    let mut executed = serde_json::json!({
        "status": "executed",
        "source": source,
        "gaps_filled": result.gaps_filled,
    });
    if let Some(fit) = &result.fit {
        executed["fit"] = serde_json::Value::String(fit.clone());
    }
    if let Some(score) = result.score {
        executed["score"] = serde_json::json!(score);
    }
    if let Some(summary) = &result.summary {
        executed["accepted"] = serde_json::json!(summary.accepted);
        if let Some(output) = &summary.final_output {
            executed["final_output"] = output.clone();
        }
        executed["audit"] = serde_json::to_value(&summary.audit).unwrap_or_default();
        executed["completed"] = serde_json::to_value(&summary.completed).unwrap_or_default();
    }
    executed
}

/// Handle `POST /v1/rigor` - the fixed-pass blue/red/judge protocol (M3).
///
/// Body: `{ "message", "session_id"?, "entities"? }`. A configured route with
/// all three role backends executes and returns `executed` (accepted answer)
/// or `clarify` (a material rejection resolved to a targeted interview). An
/// unconfigured route (no `rigor` section / missing backends) returns an
/// explicit error - never a crash.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::store::{chart_from_str, ChartStore};
    use crate::hnsw::HnswIndexHandle;
    use crate::test_stubs::{HashEmbedder, StubChatBackend};
    use tempfile::TempDir;

    fn triage_chart_json() -> String {
        r#"{
            "name": "bug_triage",
            "description": "Triage a bug report into reproduction, root cause, and fix plan",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                {
                    "name": "reproduce",
                    "provides": ["repro_plan"],
                    "depends": [],
                    "template": "reproduce {{ request }}",
                    "essential": true
                },
                {
                    "name": "root_cause",
                    "provides": ["root_cause"],
                    "depends": [
                        { "kind": "capability", "name": "repro_plan" },
                        { "kind": "entity_match", "name": "report",
                          "description": "the bug report",
                          "predicate": {
                            "fields": [
                                { "path": "title", "ty": "string", "required": true }
                            ]
                          },
                          "required": true }
                    ],
                    "template": "cause {{ request }}",
                    "essential": true
                }
            ]
        }"#
        .to_string()
    }

    fn report_entity() -> Entity {
        Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on startup"}),
        }
    }

    fn indexed_store() -> (Arc<ChartStore>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let handle = HnswIndexHandle {
            name: "workflow_library".into(),
            path: tmp
                .path()
                .join("workflow_library.sqlite")
                .display()
                .to_string(),
        };
        let store = ChartStore::new(Some(handle));
        let chart = chart_from_str(&triage_chart_json()).unwrap();
        store.upsert(chart).unwrap();
        store
            .build_index(Arc::new(HashEmbedder::new(256)))
            .expect("index builds");
        (Arc::new(store), tmp)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_partial_returns_interview_questions_for_gaps() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "partial"}"#,
            )))
            .with_charts_config(ChartsConfig::default());
        // No report entity → root_cause is unbound → Partial.
        let result = route
            .plan("Triage a bug report into reproduction", &[])
            .await;
        assert_eq!(result.source, PlanSource::TemplateAdapted);
        assert!(
            result
                .interview_questions
                .iter()
                .any(|q| q.contains("report")),
            "interview questions must cover the missing dep, got {:?}",
            result.interview_questions
        );
        assert!(result.summary.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_mismatch_falls_through_to_fresh_draft() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": null, "fit": "mismatch"}"#,
            )))
            .with_charts_config(ChartsConfig::default());
        let result = route.plan("how do I cook pasta", &[]).await;
        assert_eq!(result.source, PlanSource::FreshDraft);
        assert!(result.summary.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_exact_hit_executes_chart_to_golden() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store.clone())
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "exact"}"#,
            )))
            .with_execution_backend(Arc::new(StubChatBackend::new(vec![
                r#"{"plan": "minimal repro"}"#.to_string(),
                golden().to_string(),
            ])))
            .with_charts_config(ChartsConfig::default());

        let entities = vec![report_entity()];
        let request = "Triage a bug report into reproduction, root cause, and fix plan";

        let result = route.plan(request, &entities).await;
        assert_eq!(result.source, PlanSource::HnswHit);
        let summary = result.summary.expect("executed chart summary");
        assert_eq!(
            summary.completed.len(),
            2,
            "topo order: reproduce → root_cause"
        );
        assert_eq!(
            summary.final_output,
            Some(golden()),
            "executed result equals the golden transcript"
        );
        assert!(summary.accepted);
    }

    fn golden() -> serde_json::Value {
        serde_json::json!({"cause": "null pointer deref in async task"})
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_exact_without_execution_backend_degrades_to_fresh_draft() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "exact"}"#,
            )))
            .with_charts_config(ChartsConfig::default());
        let entities = vec![report_entity()];
        let result = route
            .plan(
                "Triage a bug report into reproduction, root cause, and fix plan",
                &entities,
            )
            .await;
        assert_eq!(
            result.source,
            PlanSource::FreshDraft,
            "an exact fit with no execution backend cannot execute — degrade, don't crash"
        );
        assert!(result.summary.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_with_reranker_backend_still_selects() {
        // The reranker backend (M7 step 2.5) is threaded into the selector.
        // A stub that returns the correct ranking, plus an adjudicator that
        // picks the chart, must yield an HnswHit exactly as without a
        // reranker — the rerank stage is additive.
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_reranker_backend(Arc::new(StubChatBackend::always(r#"["bug_triage"]"#)))
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "exact"}"#,
            )))
            .with_execution_backend(Arc::new(StubChatBackend::new(vec![
                r#"{"plan": "minimal repro"}"#.to_string(),
                golden().to_string(),
            ])))
            .with_charts_config(ChartsConfig::default());

        let entities = vec![report_entity()];
        let request = "Triage a bug report into reproduction, root cause, and fix plan";
        let result = route.plan(request, &entities).await;
        assert_eq!(result.source, PlanSource::HnswHit);
        assert!(result.summary.is_some());
    }

    // ── M8: one-round interview loop ─────────────────────────────────────

    /// A route whose selector always returns Partial with a `report` gap.
    fn partial_route() -> PlanRoute {
        let (store, _tmp) = indexed_store();
        PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "partial", "gaps": ["report"]}"#,
            )))
            .with_charts_config(ChartsConfig::default())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interview_questions_are_capped_at_max() {
        let route = partial_route();
        let result = route.plan("Triage a bug report", &[]).await;
        assert_eq!(result.source, PlanSource::TemplateAdapted);
        assert!(
            result.interview_questions.len() <= crate::charts::CHART_MAX_INTERVIEW_QUESTIONS,
            "questions must be capped at {}, got {}",
            crate::charts::CHART_MAX_INTERVIEW_QUESTIONS,
            result.interview_questions.len()
        );
        assert!(
            result.gaps.contains(&"report".to_string()),
            "raw gaps must be echoed for the round-trip: {:?}",
            result.gaps
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interview_round_trip_binds_answer_and_executes() {
        // HNSW-backed route with NO selector backend: the binding is the sole
        // authority on executability, so round 2 re-bind closes the gap.
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_execution_backend(Arc::new(StubChatBackend::new(vec![
                r#"{"plan": "minimal repro"}"#.to_string(),
                golden().to_string(),
            ])))
            .with_charts_config(ChartsConfig::default());
        let request = "Triage a bug report into reproduction, root cause, and fix plan";

        // Round 1: no report entity → the binding leaves `report` unmatched →
        // Partial with one targeted question.
        let round1 = route.plan(request, &[]).await;
        assert_eq!(round1.source, PlanSource::TemplateAdapted);
        assert_eq!(round1.interview_questions.len(), 1);
        assert_eq!(round1.gaps, vec!["report".to_string()]);
        let gaps = round1.gaps.clone();

        // Round 2: the answer arrives as an entity (kind = gap dep name) and
        // is re-bound → the chart becomes executable.
        let round2 = route
            .plan_interviewed(request, &[report_entity()], &gaps)
            .await;
        assert_eq!(
            round2.source,
            PlanSource::TemplateAdapted,
            "an interviewed chart is TemplateAdapted, not a fresh HNSW hit"
        );
        assert_eq!(round2.gaps_filled, vec!["report".to_string()]);
        assert!(
            round2.summary.is_some(),
            "interviewed chart executes into a summary"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn second_interview_failure_terminates_as_fresh_draft() {
        let route = partial_route();
        // Round 1 asks for `report`; round 2 answers with an entity that does
        // NOT satisfy the predicate (wrong kind) → still Partial → FreshDraft.
        let round1 = route.plan("Triage a bug report", &[]).await;
        let gaps = round1.gaps.clone();
        // An entity whose value does NOT satisfy the `report` predicate
        // (title is missing) → binding still leaves `report` unmatched.
        let bad_entity = Entity {
            id: "note-1".into(),
            kind: "note".into(),
            value: serde_json::json!({"body": "no title field"}),
        };
        let round2 = route
            .plan_interviewed("Triage a bug report", &[bad_entity], &gaps)
            .await;
        assert_eq!(
            round2.source,
            PlanSource::FreshDraft,
            "a second failure terminates the interview, never a second round of questions"
        );
        assert!(round2.interview_questions.is_empty());
    }

    // -- M5.3: session context via the LedgerPromptAssembler --------------

    /// A selector backend that captures the user message it receives.
    struct RecordingSelector {
        captured: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl ChatBackend for RecordingSelector {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            let user = messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            self.captured.lock().unwrap().push(user);
            Ok(r#"{"chart": null, "fit": "mismatch"}"#.to_string())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_for_session_folds_ledger_context_into_selector_prompt() {
        // M5.3: with a ledger store + assembler attached, `plan_for_session`
        // renders the session ledger and prepends it to the selector prompt.
        use crate::node_store::ContentNodeStore;
        let dir = std::env::temp_dir().join(format!(
            "coral-router-plan-ctx-{}",
            common_core::hash::uuid_v4()
        ));
        let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);
        store
            .record_request("sess-plan", "r1", "PLAN LEDGER CONTEXT at LOD0")
            .unwrap();
        let (chart_store, _tmp) = indexed_store();

        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let route = PlanRoute::new()
            .with_chart_store(chart_store)
            .with_selector_backend(Arc::new(RecordingSelector {
                captured: Arc::clone(&captured),
            }))
            .with_charts_config(ChartsConfig {
                min_score: 0.0,
                ..Default::default()
            })
            .with_prompt_assembler(PromptAssemblerCtx::new(
                store,
                LedgerPromptAssembler,
                PromptBudget::new(10_000),
                LodSpec::full(),
            ));

        let result = route
            .plan_for_session(Some("sess-plan"), "the request", &[])
            .await;
        assert_eq!(result.source, PlanSource::FreshDraft);

        let prompt = captured.lock().unwrap().last().cloned().unwrap_or_default();
        assert!(
            prompt.contains("PLAN LEDGER CONTEXT at LOD0"),
            "selector prompt must include the assembled ledger context, got: {prompt}"
        );
        assert!(
            prompt.contains("the request"),
            "selector prompt must still carry the request"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plan_without_session_keeps_blank_slate_prompt() {
        // M5.3 degradation: no session_id → identical to today's prompt (no
        // ledger context prepended), even with an assembler attached.
        use crate::node_store::ContentNodeStore;
        let dir = std::env::temp_dir().join(format!(
            "coral-router-plan-nosess-{}",
            common_core::hash::uuid_v4()
        ));
        let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);
        store
            .record_request("sess-plan", "r1", "CONTEXT SHOULD NOT APPEAR")
            .unwrap();
        let (chart_store, _tmp) = indexed_store();

        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let route = PlanRoute::new()
            .with_chart_store(chart_store)
            .with_selector_backend(Arc::new(RecordingSelector {
                captured: Arc::clone(&captured),
            }))
            .with_charts_config(ChartsConfig {
                min_score: 0.0,
                ..Default::default()
            })
            .with_prompt_assembler(PromptAssemblerCtx::new(
                store,
                LedgerPromptAssembler,
                PromptBudget::new(10_000),
                LodSpec::full(),
            ));

        let _ = route.plan("the request", &[]).await;
        let prompt = captured.lock().unwrap().last().cloned().unwrap_or_default();
        assert!(
            !prompt.contains("CONTEXT SHOULD NOT APPEAR"),
            "no session_id -> no ledger context prepended, got: {prompt}"
        );
        assert!(prompt.contains("the request"));
    }
}
