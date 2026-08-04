use std::sync::Arc;

use guidance_llm::client::ChatBackend;

use crate::charts::binding::Entity;
use crate::charts::compile::compile_chart;
use crate::charts::extract::WorkflowExtractor;
use crate::charts::select::{ChartFit, ChartSelector};
use crate::charts::store::ChartStore;
use crate::config::ChartsConfig;
use crate::workflow_config::WorkflowConfig;

pub struct PlanRoute {
    /// HNSW index for prior workflows.
    workflow_index: Option<crate::hnsw::HnswIndexHandle>,
    /// The chart store — the single owner of the workflow_library index
    /// path. Shared via `Arc` so the M7 `ChartSelector` and the route read
    /// from the same boot-loaded store.
    charts: Arc<ChartStore>,
    /// Adjudicator backend for chart selection (M7 step 3). `None` degrades
    /// selection to deterministic + HNSW only.
    selector_backend: Option<Arc<dyn ChatBackend>>,
    /// Reranker backend for chart selection (M7 step 2.5). `None` skips
    /// candidate re-ranking (Step 2 → Step 3 directly).
    reranker_backend: Option<Arc<dyn ChatBackend>>,
    /// Chart-selection configuration (thresholds, max candidates).
    cfg: ChartsConfig,
    /// M10 dispatch post-processing hook: distills successful dispatches into
    /// draft charts. `None` when extraction is not configured (opt-in).
    extractor: Option<Arc<WorkflowExtractor>>,
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub workflow: WorkflowConfig,
    pub source: PlanSource,
    pub interview_questions: Vec<String>,
    /// Raw gap dep names behind the rendered questions (M8 round-trip: the
    /// handler echoes these back so the interview stays exactly one round).
    pub gaps: Vec<String>,
    pub gaps_filled: Vec<String>,
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
            workflow_index: None,
            charts: Arc::new(ChartStore::new(None)),
            selector_backend: None,
            reranker_backend: None,
            cfg: ChartsConfig::default(),
            extractor: None,
        }
    }

    #[must_use]
    pub fn with_index(mut self, index: crate::hnsw::HnswIndexHandle) -> Self {
        self.workflow_index = Some(index);
        self.charts = Arc::new(ChartStore::new(self.workflow_index.clone()));
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

    pub fn register_template(&mut self, _task_class: impl Into<String>, _workflow: WorkflowConfig) {
        // Retained for API compatibility with the stub; the chart store is
        // the backing store now.
    }

    /// Plan a request against the chart library (M7).
    ///
    /// Selection outcome drives the returned plan:
    ///
    /// - `Exact`: compile the chart into its `WorkflowConfig`, `source =
    ///   HnswHit`.
    /// - `Partial { gaps }`: `source = TemplateAdapted` with the gaps turned
    ///   into `interview_questions` (≤ `CHART_MAX_INTERVIEW_QUESTIONS`),
    ///   `workflow` unset.
    /// - `Mismatch`: `source = FreshDraft`, `workflow` unset (fall through to
    ///   blank-slate planning).
    ///
    /// The chart's `WorkflowConfig` is compiled (not executed) here — the
    /// caller owns the `ChatBackend`/`Limiter` and executes it downstream.
    /// `gaps_filled` is reserved for the M8 interview loop.
    pub fn plan(&self, user_message: &str, entities: &[Entity]) -> PlanResult {
        self.plan_inner(user_message, entities, false)
    }

    /// Round-2 entry for the one-round interview loop (M8).
    ///
    /// The client's answers have been turned into `entities` (kind = the gap
    /// dep name). Re-binds and:
    ///
    /// - `Exact` now → `source = TemplateAdapted` with `gaps_filled` set to
    ///   the previously-asked gaps (the chart became executable).
    /// - Still `Partial`/`Mismatch` → `source = FreshDraft`. The interview is
    ///   one round, never open-ended (VISION: terminate, don't loop).
    pub fn plan_interviewed(
        &self,
        user_message: &str,
        entities: &[Entity],
        prior_gaps: &[String],
    ) -> PlanResult {
        let mut result = self.plan_inner(user_message, entities, true);
        if result.source == PlanSource::HnswHit {
            result.source = PlanSource::TemplateAdapted;
            result.gaps_filled = prior_gaps.to_vec();
        }
        result
    }

    /// Shared selection+binding+fit pipeline for both interview rounds.
    fn plan_inner(&self, user_message: &str, entities: &[Entity], retry: bool) -> PlanResult {
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
                match compile_chart(&chart, entities) {
                    Ok(workflow) => PlanResult {
                        workflow,
                        source: PlanSource::HnswHit,
                        interview_questions: Vec::new(),
                        gaps: Vec::new(),
                        gaps_filled: Vec::new(),
                    },
                    Err(e) => {
                        tracing::error!(
                            target: "router.plan",
                            chart = %chart.name,
                            error = %e,
                            "exact-selected chart failed to compile"
                        );
                        fresh_draft()
                    }
                }
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
                        workflow: WorkflowConfig::default(),
                        source: PlanSource::TemplateAdapted,
                        interview_questions: questions,
                        gaps,
                        gaps_filled: Vec::new(),
                    }
                }
            }
            ChartFit::Mismatch => fresh_draft(),
        }
    }
}

/// A `FreshDraft` plan: no chart hit, planning falls through to a blank slate.
fn fresh_draft() -> PlanResult {
    PlanResult {
        workflow: WorkflowConfig::default(),
        source: PlanSource::FreshDraft,
        interview_questions: Vec::new(),
        gaps: Vec::new(),
        gaps_filled: Vec::new(),
    }
}

/// Render an interview question for a missing binding gap.
fn gap_prompt(gap: &str) -> String {
    format!("Please provide the missing input: {gap}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::binding::ENTITIES_META_KEY;
    use crate::charts::store::{chart_from_str, ChartStore};
    use crate::hnsw::HnswIndexHandle;
    use crate::test_stubs::{HashEmbedder, StubChatBackend};
    use fluent_concurrency::pool::Limiter;
    use fluent_wvr::prelude::*;
    use guidance_llm::client::ChatBackend;
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

    fn request_ctx(text: &str, entities: &[Entity]) -> WorkContext {
        let ctx_json = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": text}]
        });
        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(ctx_json.to_string()),
        );
        if !entities.is_empty() {
            ctx.metadata.insert(
                ENTITIES_META_KEY.into(),
                MetadataValue::String(serde_json::to_string(entities).unwrap()),
            );
        }
        ctx
    }

    #[test]
    fn plan_partial_returns_interview_questions_for_gaps() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "partial"}"#,
            )))
            .with_charts_config(ChartsConfig::default());
        // No report entity → root_cause is unbound → Partial.
        let result = route.plan("Triage a bug report into reproduction", &[]);
        assert_eq!(result.source, PlanSource::TemplateAdapted);
        assert!(
            result
                .interview_questions
                .iter()
                .any(|q| q.contains("report")),
            "interview questions must cover the missing dep, got {:?}",
            result.interview_questions
        );
    }

    #[test]
    fn plan_mismatch_falls_through_to_fresh_draft() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": null, "fit": "mismatch"}"#,
            )))
            .with_charts_config(ChartsConfig::default());
        let result = route.plan("how do I cook pasta", &[]);
        assert_eq!(result.source, PlanSource::FreshDraft);
        assert!(result.workflow.workflows.is_empty());
    }

    #[test]
    fn plan_exact_hit_compiles_chart_and_executes_to_golden() {
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store.clone())
            .with_selector_backend(Arc::new(StubChatBackend::always(
                r#"{"chart": "bug_triage", "fit": "exact"}"#,
            )))
            .with_charts_config(ChartsConfig::default());

        let entities = vec![report_entity()];
        let request = "Triage a bug report into reproduction, root cause, and fix plan";

        let result = route.plan(request, &entities);
        assert_eq!(result.source, PlanSource::HnswHit);
        let workflow = &result.workflow.workflows["bug_triage"];
        assert_eq!(workflow.stages.len(), 2, "compiled chart has 2 targets");

        // Execute the compiled chart through the pipeline builder with a stub
        // backend; the final target's output must equal the golden transcript.
        let golden = serde_json::json!({"cause": "null pointer deref in async task"});
        let exec_backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"plan": "minimal repro"}"#.to_string(),
            golden.to_string(),
        ]));
        let config = crate::config::RouterConfig::default();
        let limiter = Arc::new(Limiter::new(4));
        let chart = store.get("bug_triage").expect("chart in store");
        let graph = config
            .build_chart_pipeline(&chart, &entities, &exec_backend, &limiter)
            .expect("chart compiles into a runnable pipeline");
        let output = graph
            .execute(&request_ctx(request, &entities))
            .expect("chart pipeline executes");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("pipeline result");
        assert_eq!(
            result.decisions.len(),
            2,
            "topo order: reproduce → root_cause"
        );
        let final_decision = result.decisions.last().unwrap();
        assert_eq!(
            final_decision.metadata["output"], golden,
            "executed result equals the golden transcript"
        );
    }

    #[test]
    fn plan_with_reranker_backend_still_selects() {
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
            .with_charts_config(ChartsConfig::default());

        let entities = vec![report_entity()];
        let request = "Triage a bug report into reproduction, root cause, and fix plan";
        let result = route.plan(request, &entities);
        assert_eq!(result.source, PlanSource::HnswHit);
        assert!(result.workflow.workflows.contains_key("bug_triage"));
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

    #[test]
    fn interview_questions_are_capped_at_max() {
        let route = partial_route();
        let result = route.plan("Triage a bug report", &[]);
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

    #[test]
    fn interview_round_trip_binds_answer_and_executes() {
        // HNSW-backed route with NO selector backend: the binding is the sole
        // authority on executability, so round 2 re-bind closes the gap.
        let (store, _tmp) = indexed_store();
        let route = PlanRoute::new()
            .with_chart_store(store)
            .with_charts_config(ChartsConfig::default());
        let request = "Triage a bug report into reproduction, root cause, and fix plan";

        // Round 1: no report entity → the binding leaves `report` unmatched →
        // Partial with one targeted question.
        let round1 = route.plan(request, &[]);
        assert_eq!(round1.source, PlanSource::TemplateAdapted);
        assert_eq!(round1.interview_questions.len(), 1);
        assert_eq!(round1.gaps, vec!["report".to_string()]);
        let gaps = round1.gaps.clone();

        // Round 2: the answer arrives as an entity (kind = gap dep name) and
        // is re-bound → the chart becomes executable.
        let round2 = route.plan_interviewed(request, &[report_entity()], &gaps);
        assert_eq!(
            round2.source,
            PlanSource::TemplateAdapted,
            "an interviewed chart is TemplateAdapted, not a fresh HNSW hit"
        );
        assert_eq!(round2.gaps_filled, vec!["report".to_string()]);
        assert!(
            round2.workflow.workflows.contains_key("bug_triage"),
            "interviewed chart compiles into a runnable workflow"
        );
    }

    #[test]
    fn second_interview_failure_terminates_as_fresh_draft() {
        let route = partial_route();
        // Round 1 asks for `report`; round 2 answers with an entity that does
        // NOT satisfy the predicate (wrong kind) → still Partial → FreshDraft.
        let round1 = route.plan("Triage a bug report", &[]);
        let gaps = round1.gaps.clone();
        // An entity whose value does NOT satisfy the `report` predicate
        // (title is missing) → binding still leaves `report` unmatched.
        let bad_entity = Entity {
            id: "note-1".into(),
            kind: "note".into(),
            value: serde_json::json!({"body": "no title field"}),
        };
        let round2 = route.plan_interviewed("Triage a bug report", &[bad_entity], &gaps);
        assert_eq!(
            round2.source,
            PlanSource::FreshDraft,
            "a second failure terminates the interview, never a second round of questions"
        );
        assert!(round2.interview_questions.is_empty());
    }
}
