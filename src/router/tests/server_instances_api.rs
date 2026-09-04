use super::*;

#[test]
fn parse_query_decodes_and_splits() {
    let pairs = parse_query("model=swarm&instance=ledger%3A0&id_slot=2&blank=");
    let map: std::collections::HashMap<&str, &str> = pairs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    assert_eq!(map.get("model"), Some(&"swarm"));
    assert_eq!(map.get("instance"), Some(&"ledger:0"));
    assert_eq!(map.get("id_slot"), Some(&"2"));
    assert_eq!(map.get("blank"), Some(&""));
}

#[test]
fn percent_decode_handles_plus_and_hex() {
    assert_eq!(percent_decode("a+b"), "a b");
    assert_eq!(percent_decode("a%20b"), "a b");
    assert_eq!(percent_decode("%2F"), "/");
    assert_eq!(percent_decode("plain"), "plain");
}

#[test]
fn apply_request_routing_overrides_target_fields() {
    let target = crate::pipeline::RoutingTarget {
        url: "http://x/v1/chat/completions".into(),
        model: "base:ledger".into(),
        group: None,
        target_name: None,
        params: None,
        instance: Some("ledger".into()),
        snapshot: None,
        id_slot: None,
        filter_thinking: false,
        retry_count: 0,
        retry_base_interval_s: 1,
        stream: true,
        idle_timeout_ms: 5000,
        total_timeout_ms: 30000,
        api_key: None,
        fallbacks: vec![],
        is_onnx: false,
    };
    let request = RouterRequest {
        model: "base".into(),
        messages: vec![],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: None,
        agent_id: None,
        adapter: None,
        instance: Some("scratch".into()),
        snapshot: Some("readfiles".into()),
        id_slot: Some(3),
        metadata: Default::default(),
    };
    let overlaid = apply_request_routing_fields(&target, &request);
    assert_eq!(overlaid.instance.as_deref(), Some("scratch"));
    assert_eq!(overlaid.snapshot.as_deref(), Some("readfiles"));
    assert_eq!(overlaid.id_slot, Some(3));
}
