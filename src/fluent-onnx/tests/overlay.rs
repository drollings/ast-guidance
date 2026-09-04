use super::*;

#[test]
fn residual_kind_serde_names() {
    assert_eq!(
        serde_json::to_string(&ResidualKind::Disambiguation).unwrap(),
        "\"disambiguation\""
    );
    assert_eq!(serde_json::to_string(&ResidualKind::EntityLink).unwrap(), "\"entity_link\"");
}

#[test]
fn residual_and_contribution_serde_round_trip() {
    let residual = Residual {
        kind: ResidualKind::Disambiguation,
        span: Some((0, 12)),
        text: "show me the report".into(),
        meta: serde_json::json!({ "overall": 0.3 }),
    };
    let back: Residual =
        serde_json::from_str(&serde_json::to_string(&residual).unwrap()).unwrap();
    assert_eq!(back, residual);

    let contribution = OverlayContribution {
        kind: ResidualKind::Disambiguation,
        score: Some(0.9),
        payload: serde_json::json!({ "route_hints": [{"route": "code", "score": 0.9}] }),
    };
    let back: OverlayContribution =
        serde_json::from_str(&serde_json::to_string(&contribution).unwrap()).unwrap();
    assert_eq!(back, contribution);
}

#[test]
fn disambiguation_constructor_fills_kind() {
    let r = Residual::disambiguation("hello");
    assert_eq!(r.kind, ResidualKind::Disambiguation);
    assert_eq!(r.text, "hello");
    assert_eq!(r.span, None);
}
