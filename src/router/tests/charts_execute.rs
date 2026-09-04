use super::*;
use crate::charts::binding::Entity;
use crate::charts::store::chart_from_str;
use crate::test_stubs::StubChatBackend;
use common_core::sync::lock;
use fluent_llm::{ChatMessage, LlmError};
use std::sync::Mutex;

/// A deterministic test backend that keys responses on a substring of the
/// rendered system prompt. `"__error__"` maps to `LlmError::NoResponse`.
/// Unmatched prompts error — so a stage that never should run fails loudly.
struct KeyedBackend {
    map: HashMap<String, String>,
}

impl KeyedBackend {
    fn new(entries: Vec<(String, String)>) -> Self {
        Self {
            map: entries.into_iter().collect(),
        }
    }
}

impl fluent_llm::client::ChatBackend for KeyedBackend {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        for (key, resp) in &self.map {
            if system.contains(key) {
                if resp == "__error__" {
                    return Err(LlmError::NoResponse);
                }
                return Ok(resp.clone());
            }
        }
        Err(LlmError::NoResponse)
    }
}

/// Fails exactly the first `chat_complete` call, then succeeds forever.
/// Exercises the SupervisedBatch's retry-with-backoff over a transient target failure.
struct RetryOnceBackend {
    failures_left: Mutex<usize>,
    response: String,
}

impl RetryOnceBackend {
    fn new(response: String) -> Self {
        Self {
            failures_left: Mutex::new(1),
            response,
        }
    }
}

impl fluent_llm::client::ChatBackend for RetryOnceBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        let mut left = lock(&self.failures_left);
        if *left > 0 {
            *left -= 1;
            return Err(LlmError::NoResponse);
        }
        Ok(self.response.clone())
    }
}

/// A 2-target linear chart: `a` (no deps) → `b` (depends on `a_out`).
fn linear_chart_json() -> String {
    r#"{
        "name": "linear",
        "description": "linear chain",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "a", "provides": ["a_out"], "depends": [],
              "template": "a {{ request }}", "essential": true },
            { "name": "b", "provides": ["b_out"], "depends": [
                { "kind": "capability", "name": "a_out" }
              ], "template": "b {{ upstream.a.output }}", "essential": true }
        ]
    }"#
    .to_string()
}

/// A diamond chart: base → {left, right} → join. `left` is expected to fail.
fn diamond_chart_json() -> String {
    r#"{
        "name": "diamond",
        "description": "diamond",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "base", "provides": ["base_out"], "depends": [],
              "template": "base {{ request }}", "essential": true },
            { "name": "left", "provides": ["left_out"], "depends": [
                { "kind": "capability", "name": "base_out" }
              ], "template": "left {{ upstream.base.output }}", "essential": false },
            { "name": "right", "provides": ["right_out"], "depends": [
                { "kind": "capability", "name": "base_out" }
              ], "template": "right {{ upstream.base.output }}", "essential": false },
            { "name": "join", "provides": ["join_out"], "depends": [
                { "kind": "capability", "name": "left_out" },
                { "kind": "capability", "name": "right_out" }
              ], "template": "join {{ upstream.left.output }} {{ upstream.right.output }}",
              "essential": true }
        ]
    }"#
    .to_string()
}

fn make_ctx(text: &str) -> WorkContext {
    let request_json = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": text}]
    });
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", &request_json);
    ctx
}

/// Default execution options: tokio runtime, no judge, no retries.
fn default_opts() -> ChartExecOptions {
    ChartExecOptions {
        runtime: fluent_concurrency::tokio_runtime(),
        ..ChartExecOptions::default()
    }
}

fn build_plan(
    chart_json: &str,
    backend: &Arc<dyn ChatBackend>,
    entities: &[Entity],
) -> ChartExecutionPlan {
    let chart = chart_from_str(chart_json).expect("chart parses");
    let limiter = Arc::new(Limiter::new(4));
    ChartExecutionPlan::compile(&chart, entities, backend, &limiter).expect("compiles")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn linear_chart_completes_in_order() {
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"out": "a-done"}"#.into(),
        r#"{"out": "b-done"}"#.into(),
    ]));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    assert_eq!(plan.order(), &["a".to_string(), "b".to_string()]);

    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert_eq!(summary.completed.len(), 2);
    assert_eq!(summary.completed[0].reason, "chart target 'a' completed");
    assert_eq!(summary.completed[1].reason, "chart target 'b' completed");
    assert!(summary.failed.is_empty());
    assert!(summary.cancelled.is_empty());
    assert!(summary.accepted);
    assert_eq!(
        summary.final_output,
        Some(serde_json::json!({"out": "b-done"}))
    );
    // Audit trail has 2 completed entries (a, b).
    let completed_audits: Vec<_> = summary
        .audit
        .iter()
        .filter(|e| e.verdict == ChartTargetVerdict::Completed)
        .collect();
    assert_eq!(completed_audits.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failing_mid_chart_target_cancels_dependents_and_keeps_independent_branch() {
    // Deterministic per-stage script: base ok, left errors, right ok.
    // Keys are prefix-tagged ("base ", "left ", ...) so the rendered
    // prompts never collide (left's prompt embeds base's output, but
    // "base " with the trailing space does not match inside a JSON value).
    let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
        ("base ".to_string(), r#"{"out": "base"}"#.to_string()),
        ("left ".to_string(), "__error__".to_string()),
        ("right ".to_string(), r#"{"out": "right-done"}"#.to_string()),
        ("join ".to_string(), r#"{"out": "join"}"#.to_string()),
    ]));
    let plan = build_plan(&diamond_chart_json(), &backend, &[]);

    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");

    // base + right completed; left failed (LLM error); join cancelled.
    let names: Vec<&str> = summary
        .completed
        .iter()
        .map(|d| d.metadata["chart_target"].as_str().unwrap_or("?"))
        .collect();
    assert!(names.contains(&"base"), "base completed, got {names:?}");
    assert!(
        names.contains(&"right"),
        "independent branch survives, got {names:?}"
    );
    assert!(!names.contains(&"left"));
    assert!(!names.contains(&"join"));
    assert!(summary.failed.contains(&"left".to_string()), "left failed");
    assert!(
        summary.cancelled.contains(&"join".to_string()),
        "join cancelled: {:?}",
        summary.cancelled
    );
    // join is essential → whole chart not accepted.
    assert!(!summary.accepted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn essential_failure_aborts_chart() {
    // base fails (NoResponse) → essential → nothing else runs.
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert!(summary.failed.contains(&"a".to_string()));
    assert!(summary.cancelled.contains(&"b".to_string()));
    assert!(!summary.accepted);
}

/// An essential failure aborts the chart even when an *independent* branch
/// has a ready-but-unexecuted target: that target is still cancelled (its
/// deps all completed but the chart stopped), so the cancelled set is a
/// "not completed and not failed" sweep — not merely the transitive
/// `dependents_of` the failed essential target. Locks classifier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn essential_failure_cancels_independent_ready_branch() {
    let chart_json = r#"{
        "name": "split",
        "description": "two independent branches",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "a1", "provides": ["a1_out"], "depends": [],
              "template": "a1 {{ request }}", "essential": true },
            { "name": "a2", "provides": ["a2_out"], "depends": [
                { "kind": "capability", "name": "a1_out" }
              ], "template": "a2 {{ upstream.a1.output }}", "essential": true },
            { "name": "b1", "provides": ["b1_out"], "depends": [],
              "template": "b1 {{ request }}", "essential": false },
            { "name": "b2", "provides": ["b2_out"], "depends": [
                { "kind": "capability", "name": "b1_out" }
              ], "template": "b2 {{ upstream.b1.output }}", "essential": false }
        ]
    }"#;
    // Wave 1 schedules {a1, b1}; a1 fails (essential), b1 completes. The
    // abort leaves a2 (dependent of a1) *and* b2 (ready via b1, independent
    // of a1) both cancelled.
    let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
        ("a1 ".to_string(), "__error__".to_string()),
        ("b1 ".to_string(), r#"{"out": "b1"}"#.to_string()),
    ]));
    let plan = build_plan(chart_json, &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");

    let completed: Vec<&str> = summary
        .completed
        .iter()
        .map(|d| d.metadata["chart_target"].as_str().unwrap_or("?"))
        .collect();
    assert!(completed.contains(&"b1"), "b1 completed, got {completed:?}");
    assert!(summary.failed.contains(&"a1".to_string()), "a1 failed");
    // Both the dependent-of-the-failure (a2) and the ready-but-independent
    // (b2) land in cancelled — `dependents_of(a1)` alone would miss b2.
    assert!(
        summary.cancelled.contains(&"a2".to_string()),
        "a2 cancelled: {:?}",
        summary.cancelled
    );
    assert!(
        summary.cancelled.contains(&"b2".to_string()),
        "b2 cancelled: {:?}",
        summary.cancelled
    );
    assert!(!summary.accepted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_retry_recovers_a_transient_target_failure() {
    // `a`'s LLM call errors on the first attempt, then succeeds. With
    // max_retries = 1 the SupervisedBatch retries and the whole chain completes.
    let backend: Arc<dyn ChatBackend> =
        Arc::new(RetryOnceBackend::new(r#"{"out": "a-retried"}"#.to_string()));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    let mut opts = default_opts();
    opts.max_retries = 1;
    let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
    assert_eq!(
        summary.completed.len(),
        2,
        "a recovers via SupervisedBatch retry, then b runs"
    );
    assert!(summary.failed.is_empty());
    assert!(summary.cancelled.is_empty());
    assert!(summary.accepted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_retries_means_transient_failure_fails_target() {
    let backend: Arc<dyn ChatBackend> =
        Arc::new(RetryOnceBackend::new(r#"{"out": "a"}"#.to_string()));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    // max_retries = 0 (default): a's first error is fatal.
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert!(summary.failed.contains(&"a".to_string()));
    assert!(summary.cancelled.contains(&"b".to_string()));
    assert!(!summary.accepted);
}

// ── Rubric gate ──────────────────────────────────────────────────────

fn rubric_chart_json() -> String {
    r#"{
        "name": "gated",
        "description": "rubric-gated chart",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "probe", "provides": ["probe_out"], "depends": [],
              "template": "probe {{ request }}", "essential": true,
              "rubric": { "require_fields": ["answer"], "min_score": 0.7 } },
            { "name": "after", "provides": ["after_out"], "depends": [
                { "kind": "capability", "name": "probe_out" }
              ], "template": "after {{ upstream.probe.output }}", "essential": true }
        ]
    }"#
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rubric_pass_promotes_output() {
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"answer": 42}"#.into(),
        r#"{"done": true}"#.into(),
    ]));
    let plan = build_plan(&rubric_chart_json(), &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert_eq!(summary.completed.len(), 2, "probe + after both promote");
    assert!(summary.failed.is_empty());
    assert!(summary.accepted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rubric_fail_cancels_dependents() {
    // probe's output lacks the required `answer` field → rubric reject →
    // after never becomes ready.
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"no_answer": true}"#.into(),
        r#"{"done": true}"#.into(),
    ]));
    let plan = build_plan(&rubric_chart_json(), &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert!(summary.failed.contains(&"probe".to_string()));
    assert!(summary.cancelled.contains(&"after".to_string()));
    assert!(!summary.accepted);
    let rejected: Vec<_> = summary
        .audit
        .iter()
        .filter(|e| e.verdict == ChartTargetVerdict::RubricRejected)
        .collect();
    assert_eq!(rejected.len(), 1, "probe is rubric-rejected");
    assert_eq!(rejected[0].target, "probe");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn absent_rubric_skips_gate() {
    // No rubric on the chart → output promoted on successful execution
    // even though the field is missing.
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"only_unexpected": true}"#.into(),
        r#"{"done": true}"#.into(),
    ]));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert_eq!(summary.completed.len(), 2);
    assert!(summary.accepted);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chart_level_rubric_gates_final_output() {
    let chart_json = r#"{
        "name": "charted",
        "description": "chart-level rubric",
        "schema_version": 1,
        "author_model": "human",
        "rubric": { "require_fields": ["final_answer"] },
        "targets": [
            { "name": "t", "provides": ["t_out"], "depends": [],
              "template": "t {{ request }}", "essential": true }
        ]
    }"#;
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::always(r#"{"not_the_final_answer": true}"#));
    let plan = build_plan(chart_json, &backend, &[]);
    let summary = plan
        .execute(&make_ctx("run"), &default_opts())
        .await
        .expect("runs");
    assert!(!summary.accepted, "chart-level rubric rejects");
    assert!(summary.failed.is_empty(), "target itself did not fail");
    let rejected: Vec<_> = summary
        .audit
        .iter()
        .filter(|e| e.verdict == ChartTargetVerdict::RubricRejected)
        .collect();
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0].target, "<chart>");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_trail_records_fit_and_score() {
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"out": "a"}"#.into(),
        r#"{"out": "b"}"#.into(),
    ]));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    let mut opts = default_opts();
    opts.fit = Some("exact".into());
    opts.score = Some(0.93);
    let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
    assert_eq!(summary.audit.len(), 2);
    for entry in &summary.audit {
        assert_eq!(entry.chart, "linear");
        assert_eq!(entry.fit.as_deref(), Some("exact"));
        assert_eq!(entry.score, Some(0.93));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_histogram_records_per_target() {
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
        r#"{"out": "a"}"#.into(),
        r#"{"out": "b"}"#.into(),
    ]));
    let plan = build_plan(&linear_chart_json(), &backend, &[]);
    let hist = Arc::new(LatencyHistogram::new());
    let mut opts = default_opts();
    opts.metrics = Some(hist.clone());
    let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
    assert_eq!(summary.completed.len(), 2);
    assert!(
        hist.count() >= 2,
        "per-target latency recorded, got {}",
        hist.count()
    );
}

// ── Golden e2e: real seed chart + rubric through the SupervisedBatch ─────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn golden_rubric_gated_seed_chart_runs_through_batch() {
    // Load the real Appendix A seed chart, add a target rubric, and run it
    // through the SupervisedBatch supervisor with a mock backend. The audit trail
    // must record chart/fit/score/targets.
    let seed_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/workflows/charts");
    let path = seed_dir.join("bug_triage.md.json");
    let json = std::fs::read_to_string(&path).expect("seed chart file exists");
    let mut chart = crate::charts::store::chart_from_str(&json).expect("seed chart parses");

    // Gate `root_cause` on a `cause` field being present.
    chart.targets[1].rubric = Some(crate::charts::ChartRubric {
        require_fields: vec!["cause".into()],
        judge_model: None,
        min_score: 0.7,
    });
    chart.validate().expect("rubric-gated chart validates");

    let entity = Entity {
        id: "issue-42".into(),
        kind: "report".into(),
        value: serde_json::json!({"title": "Segfault on startup"}),
    };
    let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
        (
            "write a minimal reproduction plan".to_string(),
            r#"{"plan": "minimal repro"}"#.to_string(),
        ),
        (
            "identify the root cause".to_string(),
            r#"{"cause": "null pointer deref in async task"}"#.to_string(),
        ),
        (
            "Produce a fix plan".to_string(),
            r#"{"fix": "check for null before deref"}"#.to_string(),
        ),
    ]));
    let limiter = Arc::new(Limiter::new(4));
    let plan =
        ChartExecutionPlan::compile(&chart, std::slice::from_ref(&entity), &backend, &limiter)
            .expect("seed chart compiles");
    assert_eq!(plan.order().len(), 3);

    let mut opts = default_opts();
    opts.fit = Some("exact".into());
    opts.score = Some(0.99);

    // The ctx must carry both the request and the bound entities — the
    // stages re-bind from the structured `entities` at execution time.
    let mut ctx = make_ctx("app crashes on startup");
    ctx.set_structured(
        crate::charts::binding::ENTITIES_META_KEY,
        &std::slice::from_ref(&entity),
    );

    let summary = plan
        .execute(&ctx, &opts)
        .await
        .expect("seed chart executes under SupervisedBatch supervision");

    if summary.completed.len() != 3 {
        eprintln!("FAILED summary: {summary:#?}");
        panic!("seed chart did not complete 3 targets");
    }
    assert!(summary.failed.is_empty());
    assert!(summary.cancelled.is_empty());
    assert!(summary.accepted, "rubric-gated chart accepted");
    assert_eq!(summary.audit.len(), 3);
    for entry in &summary.audit {
        assert_eq!(entry.chart, "bug_triage");
        assert_eq!(entry.fit.as_deref(), Some("exact"));
        assert_eq!(entry.score, Some(0.99));
        assert!(matches!(entry.verdict, ChartTargetVerdict::Completed));
    }
    let target_names: Vec<&str> = summary.audit.iter().map(|e| e.target.as_str()).collect();
    assert_eq!(target_names, vec!["reproduce", "root_cause", "fix_plan"]);
}

// ── Staleness / demotion fed by rubric-gate results ─────────────

/// A 1-target chart whose target rubric requires an `out` field.
fn rubric_failing_chart_json() -> String {
    r#"{
        "name": "gated",
        "description": "rubric-gated single target",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "g", "provides": ["g_out"], "depends": [],
              "template": "g {{ request }}", "essential": true,
              "rubric": { "require_fields": ["out"] } }
        ]
    }"#
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_rubric_failures_demote_chart_in_store() {
    let store = Arc::new(ChartStore::new(None));
    store
        .upsert(chart_from_str(&rubric_failing_chart_json()).unwrap())
        .unwrap();

    // Every run returns output missing `out` → rubric-rejected → a stale
    // failure recorded against the store. `KeyedBackend` repeats the
    // response (its key "g " matches the rendered prompt each run).
    let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![(
        "g ".to_string(),
        r#"{"wrong": true}"#.to_string(),
    )]));
    let plan = build_plan(&rubric_failing_chart_json(), &backend, &[]);

    for i in 0..crate::charts::CHART_STALE_FAILS {
        let mut opts = default_opts();
        opts.health = Some(store.clone());
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert!(
            summary.rubric_rejected(),
            "run {} rejected by rubric",
            i + 1
        );
        if i + 1 < crate::charts::CHART_STALE_FAILS {
            assert!(!store.is_demoted("gated"));
        }
    }
    assert!(
        store.is_demoted("gated"),
        "crossing CHART_STALE_FAILS demotes the chart"
    );
    assert_eq!(store.demoted_charts(), vec!["gated".to_string()]);
    assert!(
        !store.charts_sorted().iter().any(|c| c.name == "gated"),
        "demoted chart is no longer selected"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn passing_run_promotes_draft_and_resets_streak() {
    // Extract the chart through the idempotent path so it is a draft.
    let store = Arc::new(ChartStore::new(None));
    let chart = chart_from_str(&rubric_failing_chart_json()).unwrap();
    store
        .upsert_idempotent(chart, crate::charts::store::CHART_SUBSUME_THRESHOLD)
        .unwrap();
    assert!(store.is_draft("gated"));

    // One stale failure, then a passing run: the streak resets and the
    // draft is promoted to selectable.
    let failing: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::always(r#"{"no_out": true}"#));
    let plan = build_plan(&rubric_failing_chart_json(), &failing, &[]);
    let mut opts = default_opts();
    opts.health = Some(store.clone());
    let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
    assert!(summary.rubric_rejected());
    assert!(store.is_draft("gated"), "still a draft after a failure");

    let passing: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(r#"{"out": "g"}"#));
    let plan = build_plan(&rubric_failing_chart_json(), &passing, &[]);
    let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
    assert!(!summary.rubric_rejected());
    assert!(!store.is_draft("gated"), "a passing run promotes the draft");
    assert!(!store.is_demoted("gated"));
    assert!(
        store.charts_sorted().iter().any(|c| c.name == "gated"),
        "promoted chart is selectable"
    );
}
