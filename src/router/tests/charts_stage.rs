use super::*;
use crate::charts::binding::Entity;
use crate::test_stubs::StubChatBackend;

fn make_ctx(user_text: &str, entities: &[Entity]) -> WorkContext {
    let request_json = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": user_text}]
    });
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", &request_json);
    if !entities.is_empty() {
        ctx.set_structured(super::super::binding::ENTITIES_META_KEY, &entities);
    }
    ctx
}

fn stage_with(
    backend: StubChatBackend,
    template: &str,
    upstream_ids: Vec<String>,
) -> ChartPromptStage {
    ChartPromptStage::new(
        Arc::new(backend),
        Arc::new(Limiter::new(4)),
        "reproduce",
        "bug_triage",
        template,
        vec![],
        upstream_ids,
        vec![],
        vec![ArcIntern::from("repro_plan"), ArcIntern::from("reproduce")],
        vec![],
    )
}

#[test]
fn execute_emits_chart_provenance_metadata() {
    let backend = StubChatBackend::always(r#"{"plan": "minimal repro steps"}"#);
    let stage = stage_with(
        backend,
        "Given the bug report {{ request }}, write a reproduction plan.",
        vec![],
    );
    let ctx = make_ctx("crash on startup", &[]);
    let output = stage.execute(&ctx).unwrap();
    let decision: StageDecision = output.data_take().unwrap();
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        decision.metadata[CHART_NAME_META_KEY],
        serde_json::json!("bug_triage")
    );
    assert_eq!(
        decision.metadata[CHART_TARGET_META_KEY],
        serde_json::json!("reproduce")
    );
    assert_eq!(
        decision.metadata[CHART_OUTPUT_META_KEY]["plan"],
        serde_json::json!("minimal repro steps")
    );
    assert!(decision.metadata[CHART_RESPONSE_META_KEY].is_string());
}

#[test]
fn execute_renders_entities_and_upstream() {
    // First stage output promoted under `stage.reproduce.output`.
    let mut ctx = make_ctx(
        "crash on startup",
        &[Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on load"}),
        }],
    );
    ctx.structured.insert(
        "stage.reproduce".into(),
        serde_json::json!({"output": {"plan": "minimal repro"}}),
    );

    let backend = StubChatBackend::always(r#"{"cause": "null pointer deref"}"#);
    let stage = ChartPromptStage::new(
        Arc::new(backend),
        Arc::new(Limiter::new(4)),
        "root_cause",
        "bug_triage",
        "Using the plan {{ upstream.reproduce.output.plan }}, find the cause of {{ request }}.\n{% for e in deps.report %}Report: {{ e.value.title }}\n{% endfor %}",
        vec![DepSpec::EntityMatch {
            name: "report".into(),
            description: "the bug report".into(),
            predicate: Some(super::super::EntityPredicate {
                fields: vec![super::super::FieldRule {
                    path: "title".into(),
                    ty: super::super::FieldType::String,
                    required: true,
                    min: None,
                    max: None,
                    pattern: None,
                }],
                any_of: vec![],
            }),
            required: true,
        }],
        vec!["reproduce".to_string()],
        vec![ArcIntern::from("repro_plan")],
        vec![ArcIntern::from("root_cause")],
        vec!["repro_plan".to_string()],
    );

    let output = stage.execute(&ctx).unwrap();
    let decision: StageDecision = output.data_take().unwrap();
    assert_eq!(
        decision.metadata[CHART_OUTPUT_META_KEY]["cause"],
        serde_json::json!("null pointer deref")
    );
}

#[test]
fn unmatched_required_dep_fails_closed() {
    // No entities provided; the report dep is required → error.
    let backend = StubChatBackend::always(r#"{"x": 1}"#);
    let stage = ChartPromptStage::new(
        Arc::new(backend),
        Arc::new(Limiter::new(4)),
        "fix_plan",
        "bug_triage",
        "fix {{ request }}",
        vec![DepSpec::EntityMatch {
            name: "report".into(),
            description: "the bug report".into(),
            predicate: Some(super::super::EntityPredicate {
                fields: vec![super::super::FieldRule {
                    path: "title".into(),
                    ty: super::super::FieldType::String,
                    required: true,
                    min: None,
                    max: None,
                    pattern: None,
                }],
                any_of: vec![],
            }),
            required: true,
        }],
        vec![],
        vec![],
        vec![ArcIntern::from("fix_plan")],
        vec![],
    );
    let ctx = make_ctx("help", &[]);
    let err = stage.execute(&ctx).unwrap_err();
    assert!(
        err.to_string().contains("unmatched required deps"),
        "expected unmatched-deps error, got: {err}"
    );
}

#[test]
fn unmatched_capability_dep_fails_closed() {
    // A capability dep with no in-graph provider and no matching entity
    // at runtime fails closed — the stage must not render without
    // the capability's input.
    let backend = StubChatBackend::always(r#"{"x": 1}"#);
    let stage = ChartPromptStage::new(
        Arc::new(backend),
        Arc::new(Limiter::new(4)),
        "fix_plan",
        "bug_triage",
        "fix {{ request }}",
        vec![DepSpec::Capability {
            name: "external_data".into(),
        }],
        vec![],
        vec![],
        vec![ArcIntern::from("fix_plan")],
        vec![],
    );
    let ctx = make_ctx("help", &[]);
    let err = stage.execute(&ctx).unwrap_err();
    assert!(
        err.to_string().contains("unmatched required deps"),
        "expected unmatched-deps error, got: {err}"
    );
}

#[test]
fn graph_satisfied_capability_does_not_fail_closed() {
    // A capability dep satisfied by an in-graph upstream is bound by the
    // graph, not by context entities — the runtime re-bind must not
    // fail-closed on it even when no entity provides it.
    let backend = StubChatBackend::always(r#"{"plan": "minimal repro"}"#);
    let stage = ChartPromptStage::new(
        Arc::new(backend),
        Arc::new(Limiter::new(4)),
        "root_cause",
        "bug_triage",
        "Using the plan {{ upstream.reproduce.output }}, find the cause.",
        vec![DepSpec::Capability {
            name: "repro_plan".into(),
        }],
        vec!["reproduce".to_string()],
        vec![ArcIntern::from("repro_plan")],
        vec![ArcIntern::from("root_cause")],
        vec!["repro_plan".to_string()],
    );
    let mut ctx = make_ctx("crash on startup", &[]);
    ctx.structured.insert(
        "stage.reproduce".into(),
        serde_json::json!({"output": {"plan": "minimal repro"}}),
    );
    stage
        .execute(&ctx)
        .expect("graph-satisfied capability runs");
}

#[test]
fn parse_output_strips_fences_and_falls_back() {
    assert_eq!(
        parse_output("```json\n{\"a\": 1}\n```"),
        serde_json::json!({"a": 1})
    );
    assert_eq!(
        parse_output("plain text answer"),
        serde_json::json!("plain text answer")
    );
    assert_eq!(parse_output("42"), serde_json::json!(42));
}
