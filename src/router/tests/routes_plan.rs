use super::*;
use crate::charts::store::{chart_from_str, ChartStore};
use fluent_db::hnsw::HnswIndexHandle;
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
    // The reranker backend is threaded into the selector.
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

// ── One-round interview loop ─────────────────────────────────────

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

// -- Session context via the LedgerPromptAssembler --------------

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
    // With a ledger store + assembler attached, `plan_for_session`
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
    // Degradation: no session_id → identical to today's prompt (no
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
