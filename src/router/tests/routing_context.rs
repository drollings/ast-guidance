use super::*;

#[test]
fn routing_context_merge_body_wins_over_query() {
    let q = RoutingContext::from_query(&[("instance".into(), "a".into())]);
    let b = RoutingContext {
        instance: Some("b".into()),
        ..Default::default()
    };
    let merged = q.merge(b);
    assert_eq!(merged.instance.as_deref(), Some("b"));
}

#[test]
fn routing_context_into_params_strips_none() {
    let ctx = RoutingContext {
        instance: Some("x".into()),
        ..Default::default()
    };
    let v = ctx.into_params();
    assert_eq!(v.get("instance").and_then(|v| v.as_str()), Some("x"));
    assert!(v.get("snapshot").is_none());
    assert!(v.get("id_slot").is_none());
}

#[test]
fn routing_context_from_body_parses() {
    let body = serde_json::json!({"instance":"scratch","snapshot":"snap","id_slot":3,"num_ctx":8192});
    let ctx = RoutingContext::from_body(&body);
    assert_eq!(ctx.instance.as_deref(), Some("scratch"));
    assert_eq!(ctx.snapshot.as_deref(), Some("snap"));
    assert_eq!(ctx.id_slot, Some(3));
    assert_eq!(ctx.num_ctx, Some(8192));
}
