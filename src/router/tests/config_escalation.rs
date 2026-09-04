use super::*;

#[test]
fn array_form_deserializes_and_lists_models() {
    let group: ModelGroup = serde_json::from_str(r#"["fast", "small"]"#).unwrap();
    assert_eq!(group.models(), &["fast".to_string(), "small".to_string()]);
    assert!(group.escalation().is_none());
}

#[test]
fn object_form_models_only() {
    let group: ModelGroup = serde_json::from_str(r#"{"models": ["code-model"]}"#).unwrap();
    assert_eq!(group.models(), &["code-model".to_string()]);
    assert!(group.escalation().is_none());
}

#[test]
fn object_form_with_escalation() {
    let group: ModelGroup = serde_json::from_str(
        r#"{
            "models": ["code-model"],
            "escalation": {
                "modes": ["filter", "question", "team", "turnover"],
                "frontier": {"endpoint": "https://frontier.example/v1/chat/completions", "model": "claude-sonnet", "api_key_env": "ANTHROPIC_KEY"},
                "decomposer_model": "fast",
                "assembler_model": "fast",
                "classifier_model": "small",
                "classifier_parallel": 5,
                "draft_model": "small",
                "judge_model": "fast"
            }
        }"#,
    )
    .unwrap();
    assert_eq!(group.models(), &["code-model".to_string()]);
    let ladder = group.escalation().expect("escalation present");
    assert_eq!(
        ladder.modes,
        vec![
            EscalationMode::Filter,
            EscalationMode::Question,
            EscalationMode::Team,
            EscalationMode::Turnover
        ]
    );
    let front = ladder.frontier.as_ref().unwrap();
    assert_eq!(
        front.endpoint,
        "https://frontier.example/v1/chat/completions"
    );
    assert_eq!(front.model, "claude-sonnet");
    assert_eq!(front.api_key_env.as_deref(), Some("ANTHROPIC_KEY"));
    assert_eq!(ladder.decomposer_model.as_deref(), Some("fast"));
    assert_eq!(ladder.classifier_parallel, 5);
    assert_eq!(ladder.draft_model.as_deref(), Some("small"));
    assert_eq!(ladder.judge_model.as_deref(), Some("fast"));
}

#[test]
fn array_form_round_trips() {
    let group: ModelGroup = serde_json::from_str(r#"["fast", "small"]"#).unwrap();
    let serialized = serde_json::to_string(&group).unwrap();
    let back: ModelGroup = serde_json::from_str(&serialized).unwrap();
    assert_eq!(back.models(), group.models());
}

#[test]
fn ladder_defaults() {
    let ladder = EscalationLadderConfig::default();
    assert!(ladder.modes.is_empty());
    assert!(ladder.frontier.is_none());
    assert_eq!(ladder.classifier_parallel, 3);
}

#[test]
fn ladder_missing_fields_default() {
    let ladder: EscalationLadderConfig =
        serde_json::from_str(r#"{"frontier": {"endpoint": "u", "model": "m"}}"#).unwrap();
    assert!(ladder.modes.is_empty());
    assert_eq!(ladder.classifier_parallel, 3, "unset parallel defaults");
    assert!(ladder.decomposer_model.is_none());
}

#[test]
fn escalation_mode_list_deserializes() {
    let modes: Vec<EscalationMode> = serde_json::from_str(r#"["filter","team"]"#).unwrap();
    assert_eq!(modes, vec![EscalationMode::Filter, EscalationMode::Team]);
}

#[test]
fn empty_models_object_deserializes() {
    let group: ModelGroup = serde_json::from_str("{}").unwrap();
    assert!(group.models().is_empty());
    assert!(group.escalation().is_none());
}

#[test]
fn router_config_shipped_array_shape_still_parses() {
    // The shipped `env/coral-router.json` shape: array-form model_groups.
    let cfg: crate::config::RouterConfig = serde_json::from_str(
        r#"{
            "model_groups": {"fast": ["fast"], "code": ["code-model"]},
            "models": {}
        }"#,
    )
    .unwrap();
    assert_eq!(cfg.model_groups.len(), 2);
    assert_eq!(cfg.model_groups["fast"].models(), &["fast".to_string()]);
    assert!(cfg.model_groups["fast"].escalation().is_none());
}
