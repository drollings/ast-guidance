use super::*;

#[test]
fn classifier_action_parses_known_variants() {
    assert!(ClassifierAction::from_str("respond").unwrap().is_respond());
    assert!(ClassifierAction::from_str("route").unwrap().is_route());
    assert!(ClassifierAction::from_str("reject").unwrap().is_reject());

    let respond = ClassifierOutput {
        action: "respond".into(),
        response: Some("hello".into()),
        ..Default::default()
    };
    let a = ClassifierAction::from_output(&respond).unwrap();
    assert_eq!(a, ClassifierAction::Respond("hello".into()));

    let route = ClassifierOutput {
        action: "route".into(),
        target: Some("code".into()),
        ..Default::default()
    };
    let b = ClassifierAction::from_output(&route).unwrap();
    assert_eq!(b, ClassifierAction::Route { target: Some("code".into()) });

    let reject = ClassifierOutput {
        action: "reject".into(),
        reason: "unsafe".into(),
        ..Default::default()
    };
    let c = ClassifierAction::from_output(&reject).unwrap();
    assert_eq!(c, ClassifierAction::Reject { reason: "unsafe".into() });
}

#[test]
fn classifier_action_unknown_is_error_not_route() {
    let out = ClassifierOutput {
        action: "nonsense".into(),
        ..Default::default()
    };
    let err = ClassifierAction::from_output(&out).unwrap_err();
    assert_eq!(err.0, "nonsense");
    // FromStr also strict
    assert!(ClassifierAction::from_str("nonsense").is_err());
    assert!(ClassifierAction::from_str("ROUTE").is_err());
}

#[test]
fn classifier_reject_has_typed_reason() {
    let out = ClassifierOutput {
        action: "reject".into(),
        reason: "policy violation".into(),
        ..Default::default()
    };
    match ClassifierAction::from_output(&out).unwrap() {
        ClassifierAction::Reject { reason } => assert_eq!(reason, "policy violation"),
        other => panic!("expected Reject, got {other:?}"),
    }
    // Empty reason still typed but caller can check emptiness
    let empty = ClassifierOutput {
        action: "reject".into(),
        reason: "".into(),
        ..Default::default()
    };
    match ClassifierAction::from_output(&empty).unwrap() {
        ClassifierAction::Reject { reason } => assert!(reason.is_empty()),
        _ => panic!("expected Reject"),
    }
}
