use super::*;

fn valid_chart() -> ChartDef {
    serde_json::from_str(
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
                        { "kind": "capability", "name": "repro_plan" }
                    ],
                    "template": "root cause {{ upstream.reproduce.output }}",
                    "essential": true
                },
                {
                    "name": "fix_plan",
                    "provides": ["fix_plan"],
                    "depends": [
                        { "kind": "capability", "name": "root_cause" },
                        { "kind": "entity_match", "name": "report",
                          "description": "the bug report entity",
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
    .expect("valid seed chart JSON")
}

#[test]
fn valid_chart_passes() {
    let chart = valid_chart();
    let warnings = chart.validate().expect("valid chart validates");
    assert!(
        warnings.is_empty(),
        "fully-internal capability chain has no unresolved deps: {warnings:?}"
    );
}

#[test]
fn over_long_description_fails() {
    let mut chart = valid_chart();
    chart.description = "x".repeat(CHART_DESCRIPTION_MAX_CHARS + 1);
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn bad_schema_version_fails() {
    let mut chart = valid_chart();
    chart.schema_version = 99;
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::UnsupportedVersion(99)));
}

#[test]
fn duplicate_provides_fails() {
    let mut chart = valid_chart();
    chart.targets[1].provides = vec!["repro_plan".into()];
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::DuplicateName(ref n) if n == "repro_plan"));
}

#[test]
fn duplicate_target_name_fails() {
    let mut chart = valid_chart();
    let dup = chart.targets[0].clone();
    chart.targets.push(dup);
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::DuplicateName(ref n) if n == "reproduce"));
}

#[test]
fn self_satisfying_capability_chain_passes() {
    // Capability deps reference another target's provides: no warning.
    let warnings = valid_chart().validate().unwrap();
    assert!(warnings.is_empty());
}

#[test]
fn entity_only_dep_produces_warning_not_error() {
    let mut chart = valid_chart();
    // Point fix_plan at a capability no target provides.
    chart.targets[2].depends = vec![DepSpec::Capability {
        name: "external_data".into(),
    }];
    let warnings = chart
        .validate()
        .expect("entity-only dep is a warning, not an error");
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].dep, "external_data");
}

#[test]
fn empty_template_fails() {
    let mut chart = valid_chart();
    chart.targets[0].template = String::new();
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn no_provides_fails() {
    let mut chart = valid_chart();
    for t in &mut chart.targets {
        t.provides.clear();
    }
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn golden_seed_round_trips_and_validates() {
    // Appendix A seed chart: round-trip through serde_json.
    let json = chart_to_json(&valid_chart()).expect("serializes");
    let chart: ChartDef = chart_from_str(&json).expect("round-trips");
    assert_eq!(chart.name, "bug_triage");
    assert_eq!(chart.targets.len(), 3);
    chart.validate().expect("round-tripped chart validates");
}

#[test]
fn parse_helper_validates_content_model() {
    // Parsing a chart must validate it, not just deserialize: an empty
    // template is rejected by `chart_from_str`.
    let mut chart = valid_chart();
    chart.targets[0].template = String::new();
    let json = chart_to_json(&chart).expect("serializes");
    let err = chart_from_str(&json).unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn dep_spec_name_helper() {
    let cap = DepSpec::Capability { name: "x".into() };
    assert_eq!(cap.name(), "x");
    let em = DepSpec::EntityMatch {
        name: "report".into(),
        description: "d".into(),
        predicate: None,
        required: true,
    };
    assert_eq!(em.name(), "report");
}

#[test]
fn target_rubric_round_trips_and_validates() {
    let mut chart = valid_chart();
    chart.targets[0].rubric = Some(ChartRubric {
        require_fields: vec!["plan".into()],
        judge_model: Some("judge".into()),
        min_score: 0.8,
    });
    chart.validate().expect("rubric chart validates");
    let json = chart_to_json(&chart).expect("serializes");
    let back: ChartDef = chart_from_str(&json).expect("round-trips");
    assert_eq!(back.targets[0].rubric, chart.targets[0].rubric);
}

#[test]
fn chart_level_rubric_validates() {
    let mut chart = valid_chart();
    chart.rubric = Some(ChartRubric {
        require_fields: vec!["fix_plan".into()],
        judge_model: None,
        min_score: 0.7,
    });
    chart.validate().expect("chart rubric validates");
}

#[test]
fn rubric_min_score_out_of_range_fails() {
    let mut chart = valid_chart();
    chart.targets[0].rubric = Some(ChartRubric {
        require_fields: vec!["x".into()],
        judge_model: None,
        min_score: 1.5,
    });
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn rubric_empty_require_field_path_fails() {
    let mut chart = valid_chart();
    chart.targets[0].rubric = Some(ChartRubric {
        require_fields: vec!["   ".into()],
        judge_model: None,
        min_score: 0.7,
    });
    let err = chart.validate().unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}
