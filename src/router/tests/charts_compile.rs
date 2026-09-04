use super::*;
use fluent_wvr::prelude::WorkContext;

fn linear_chart() -> ChartDef {
    serde_json::from_str(
        r#"{
            "name": "bug_triage",
            "description": "triage",
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
                        { "kind": "capability", "name": "repro_plan" }
                    ],
                    "template": "cause {{ upstream.reproduce.output }}",
                    "essential": true
                },
                {
                    "name": "fix_plan",
                    "provides": ["fix_plan"],
                    "depends": [
                        { "kind": "capability", "name": "root_cause" },
                        { "kind": "entity_match", "name": "report",
                          "description": "the report",
                          "predicate": {
                            "fields": [
                                { "path": "title", "ty": "string", "required": true }
                            ]
                          },
                          "required": true }
                    ],
                    "template": "fix {{ upstream.root_cause.output }}",
                    "essential": true
                }
            ]
        }"#,
    )
    .expect("linear chart JSON")
}

fn report_entity() -> Entity {
    Entity {
        id: "issue-42".into(),
        kind: "report".into(),
        value: serde_json::json!({"title": "Segfault on startup"}),
    }
}

fn test_backend() -> Arc<dyn ChatBackend> {
    Arc::new(crate::test_stubs::StubChatBackend::always("{}"))
}

fn test_limiter() -> Arc<Limiter> {
    Arc::new(Limiter::new(4))
}

#[test]
fn linear_chart_compiles_with_dep_edges() {
    let (targets, order) = compile_chart_stages(
        &linear_chart(),
        &[report_entity()],
        &test_backend(),
        &test_limiter(),
    )
    .expect("compiles");
    assert_eq!(targets.len(), 3);
    assert_eq!(order, vec!["reproduce", "root_cause", "fix_plan"]);

    assert!(targets[0].upstream_ids.is_empty());
    assert_eq!(targets[1].upstream_ids, vec!["reproduce".to_string()]);
    // Capability root_cause → edge to root_cause; entity dep → no edge.
    assert_eq!(targets[2].upstream_ids, vec!["root_cause".to_string()]);
    assert!(targets.iter().all(|t| t.essential));
}

#[test]
fn diamond_chart_compiles() {
    let chart: ChartDef = serde_json::from_str(
        r#"{
            "name": "diamond",
            "description": "diamond",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "base", "provides": ["base_out"], "depends": [],
                  "template": "base", "essential": true },
                { "name": "left", "provides": ["left_out"], "depends": [
                    { "kind": "capability", "name": "base_out" }
                  ], "template": "left", "essential": true },
                { "name": "right", "provides": ["right_out"], "depends": [
                    { "kind": "capability", "name": "base_out" }
                  ], "template": "right", "essential": true },
                { "name": "join", "provides": ["join_out"], "depends": [
                    { "kind": "capability", "name": "left_out" },
                    { "kind": "capability", "name": "right_out" }
                  ], "template": "join", "essential": true }
            ]
        }"#,
    )
    .expect("diamond chart JSON");

    let (targets, order) =
        compile_chart_stages(&chart, &[], &test_backend(), &test_limiter()).expect("compiles");
    assert_eq!(targets.len(), 4);
    assert_eq!(
        order,
        vec![
            "base".to_string(),
            "left".to_string(),
            "right".to_string(),
            "join".to_string()
        ]
    );
    assert_eq!(
        targets[3].upstream_ids,
        vec!["left".to_string(), "right".to_string()]
    );
}

#[test]
fn unmatched_required_entity_dep_is_compile_error() {
    // fix_plan requires a "report" entity; none provided.
    let result = compile_chart_stages(&linear_chart(), &[], &test_backend(), &test_limiter());
    assert!(
        matches!(result, Err(ChartError::Compile { reason }) if reason.contains("not fully bound")),
        "expected compile error for unbound chart"
    );
}

#[test]
fn ambiguous_binding_is_compile_error() {
    let chart = linear_chart();
    let two = vec![report_entity(), report_entity()];
    let result = compile_chart_stages(&chart, &two, &test_backend(), &test_limiter());
    assert!(
        matches!(result, Err(ChartError::Compile { reason }) if reason.contains("ambiguous")),
        "expected ambiguous compile error"
    );
}

#[test]
fn cycle_in_capability_deps_is_compile_error() {
    let chart: ChartDef = serde_json::from_str(
        r#"{
            "name": "cycle",
            "description": "cycle",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "a", "provides": ["a_out"], "depends": [
                    { "kind": "capability", "name": "b_out" }
                  ], "template": "a", "essential": true },
                { "name": "b", "provides": ["b_out"], "depends": [
                    { "kind": "capability", "name": "a_out" }
                  ], "template": "b", "essential": true }
            ]
        }"#,
    )
    .expect("cycle chart JSON");

    let result = compile_chart_stages(&chart, &[], &test_backend(), &test_limiter());
    assert!(
        matches!(result, Err(ChartError::Compile { reason }) if reason.contains("cycle")),
        "expected cycle compile error"
    );
}

#[test]
fn optional_entity_dep_unbound_compiles_without_edge() {
    // A non-required entity dep with no match renders without it — no
    // compile error, no stage edge.
    let chart: ChartDef = serde_json::from_str(
        r#"{
            "name": "opt",
            "description": "optional entity dep",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "t", "provides": ["t_out"], "depends": [
                    { "kind": "entity_match", "name": "note",
                      "description": "the note",
                      "predicate": {
                        "fields": [
                          { "path": "note_title", "ty": "string", "required": true }
                        ]
                      },
                      "required": false }
                ], "template": "t", "essential": true }
            ]
        }"#,
    )
    .expect("optional chart JSON");

    let (targets, order) =
        compile_chart_stages(&chart, &[], &test_backend(), &test_limiter()).expect("compiles");
    assert_eq!(targets.len(), 1);
    assert!(targets[0].upstream_ids.is_empty());
    assert_eq!(order, vec!["t".to_string()]);
}

/// End-to-end (single chart path): `compile_chart_stages` output
/// feeds `ChartExecutionPlan`, which executes under SupervisedBatch supervision and
/// yields the compiled order + a completed summary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compiled_stages_feed_chart_execution_plan() {
    use crate::charts::execute::ChartExecOptions;
    use crate::charts::store::chart_from_str;

    let chart_json = r#"{
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
    }"#;
    let chart = chart_from_str(chart_json).expect("chart parses");
    let backend: Arc<dyn ChatBackend> =
        Arc::new(crate::test_stubs::StubChatBackend::new(vec![
            r#"{"out": "a"}"#.into(),
            r#"{"out": "b"}"#.into(),
        ]));

    let (targets, order) =
        compile_chart_stages(&chart, &[], &backend, &test_limiter()).expect("compiles");
    assert_eq!(order, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(targets.len(), 2);

    let plan = crate::charts::execute::ChartExecutionPlan::compile(
        &chart,
        &[],
        &backend,
        &test_limiter(),
    )
    .expect("plan compiles");
    assert_eq!(plan.order(), &order);

    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({"model": "test", "messages": [{"role": "user", "content": "run"}]}),
    );
    let opts = ChartExecOptions {
        runtime: fluent_concurrency::tokio_runtime(),
        ..ChartExecOptions::default()
    };
    let summary = plan.execute(&ctx, &opts).await.expect("executes");
    assert_eq!(summary.completed.len(), 2);
    assert!(summary.failed.is_empty());
    assert!(summary.accepted);
    assert_eq!(summary.final_output, Some(serde_json::json!({"out": "b"})));
}
