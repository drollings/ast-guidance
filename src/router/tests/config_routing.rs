use super::*;

fn sample_config() -> RoutingConfig {
    serde_json::from_value(serde_json::json!({
        "routes": {
            "fast": {"group": "fast", "pipelines": ["default"], "description": "fast route"},
            "smart": {"group": "smart", "pipelines": ["default"], "always_route": true},
        },
        "models": {
            "tiny": {"endpoint": "http://a", "name": "tiny", "intelligence": 1,
                     "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.0, "speed": 10},
            "big": {"endpoint": "http://b", "name": "big", "intelligence": 5,
                    "cost_input": 9.0, "cost_output": 9.0, "cost_cached_read": 0.0, "speed": 10},
            "huge": {"endpoint": "http://c", "name": "huge", "intelligence": 8,
                     "cost_input": 20.0, "cost_output": 20.0, "cost_cached_read": 0.0, "speed": 10},
        },
        "model_groups": {"fast": ["tiny"], "smart": ["big", "huge"]},
        "system_prompt": "sys",
        "safety_threshold": 0.5,
        "default_route": "fast",
    }))
    .expect("valid routing config")
}

#[test]
fn route_ref_serde_round_trip() {
    let r: RouteRef = serde_json::from_value(serde_json::json!({
        "group": "g", "pipelines": ["a"], "description": "d", "always_route": true
    }))
    .expect("deserialize");
    assert_eq!(r.group, "g");
    assert_eq!(r.pipelines, vec!["a"]);
    assert!(r.always_route);
    let back: RouteRef =
        serde_json::from_str(&serde_json::to_string(&r).expect("serialize")).expect("round trip");
    assert_eq!(back.group, "g");
}

#[test]
fn route_ref_pipelines_defaults_to_default() {
    let r: RouteRef = serde_json::from_value(serde_json::json!({"group": "g"})).expect("deserialize");
    assert_eq!(r.pipelines, vec!["default"]);
    assert!(!r.always_route);
}

#[test]
fn route_group_resolves_route_or_default() {
    let c = sample_config();
    assert_eq!(c.route_group("fast"), Some("fast"));
    assert_eq!(c.route_group("smart"), Some("smart"));
    // Unknown route falls back to the default route's group.
    assert_eq!(c.route_group("nope"), Some("fast"));
}

#[test]
fn route_group_none_when_no_groups() {
    let c: RoutingConfig = serde_json::from_value(serde_json::json!({
        "routes": {"r": {"group": "missing"}},
        "models": {},
        "model_groups": {},
        "system_prompt": "s",
        "safety_threshold": 0.5,
        "default_route": "r",
    }))
    .expect("config");
    assert_eq!(c.route_group("r"), Some("missing"));
    // Unknown route and default route both groupless -> None.
    let c2: RoutingConfig = serde_json::from_value(serde_json::json!({
        "routes": {},
        "models": {},
        "model_groups": {},
        "system_prompt": "s",
        "safety_threshold": 0.5,
        "default_route": "missing",
    }))
    .expect("config");
    assert_eq!(c2.route_group("r"), None);
}

#[test]
fn resolve_route_picks_cheapest_passing_complexity() {
    let c = sample_config();
    // smart group has big(int 5, cost 18) and huge(int 8, cost 40).
    let (entry, name) = c.resolve_route("smart", Some(6)).expect("resolve");
    // big is int 5 < 6 so filtered out; huge (8 >= 6) is the only candidate.
    assert_eq!(name, "huge");
    assert_eq!(entry.intelligence, 8);
    // No min_complexity -> cheapest eligible (big, cost 18).
    let (_, name) = c.resolve_route("smart", None).expect("resolve");
    assert_eq!(name, "big");
}

#[test]
fn resolve_route_falls_back_to_cheapest_when_none_pass() {
    let c = sample_config();
    // min_complexity 9 filters everything; cheapest in group wins.
    let (entry, _) = c.resolve_route("smart", Some(9)).expect("resolve");
    assert_eq!(entry.intelligence, 5, "big is cheapest in the smart group");
}

#[test]
fn resolve_route_unknown_route_uses_default_group() {
    let c = sample_config();
    let (_, name) = c.resolve_route("nope", None).expect("resolve via default");
    assert_eq!(name, "tiny");
}

#[test]
fn resolve_route_direct_model_by_name() {
    // A model name that is not a route resolves as a direct model only
    // when neither the route nor the default route exists.
    let c: RoutingConfig = serde_json::from_value(serde_json::json!({
        "routes": {},
        "models": {
            "big": {"endpoint": "http://b", "name": "big", "intelligence": 5,
                    "cost_input": 9.0, "cost_output": 9.0, "cost_cached_read": 0.0, "speed": 10},
        },
        "model_groups": {},
        "system_prompt": "s",
        "safety_threshold": 0.5,
        "default_route": "missing",
    }))
    .expect("config");
    let (entry, name) = c.resolve_route("big", None).expect("resolve direct");
    assert_eq!(name, "big");
    assert_eq!(entry.intelligence, 5);
    // A completely unknown name with no default route -> None.
    assert!(c.resolve_route("nope", None).is_none());
}

#[test]
fn resolve_route_missing_group_returns_none() {
    let c: RoutingConfig = serde_json::from_value(serde_json::json!({
        "routes": {"r": {"group": "missing"}},
        "models": {"m": {"endpoint": "http://a", "name": "m", "intelligence": 1,
                         "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.0, "speed": 10}},
        "model_groups": {},
        "system_prompt": "s",
        "safety_threshold": 0.5,
        "default_route": "r",
    }))
    .expect("config");
    assert!(c.resolve_route("r", None).is_none());
}

#[test]
fn routing_target_attaches_group_target_and_fallbacks() {
    let c = sample_config();
    let rt = c.routing_target("smart", None).expect("routing target");
    assert_eq!(rt.model, "big");
    assert_eq!(rt.group.as_deref(), Some("smart"));
    assert_eq!(rt.target_name.as_deref(), Some("smart"));
    // Primary (big) is skipped; fallbacks are the remaining dispatch
    // targets in preference order.
    let fallback_models: Vec<&str> = rt.fallbacks.iter().map(|f| f.model.as_str()).collect();
    assert!(fallback_models.contains(&"big"));
    assert!(fallback_models.contains(&"tiny"));
}

#[test]
fn routing_target_unknown_route_uses_default() {
    let c = sample_config();
    let rt = c.routing_target("nope", None).expect("routing target");
    assert_eq!(rt.model, "tiny");
    assert_eq!(rt.group.as_deref(), Some("fast"));
}

/// A route whose `model_groups` member is a configured in-process onnx
/// role (e.g. the generative `onnx/llm` routing model) resolves to an onnx
/// `RoutingTarget` — not a `models` entry — so the dispatch layer serves it
/// through the onnx `ChatBackend` (is_onnx, no HTTP url).
#[test]
fn routing_target_resolves_onnx_role_group_member() {
    let mut c: RoutingConfig = serde_json::from_value(serde_json::json!({
        "routes": {"local": {"group": "default"}},
        "models": {},
        "model_groups": {"default": ["onnx/llm"]},
        "system_prompt": "s",
        "safety_threshold": 0.5,
        "default_route": "local",
    }))
    .expect("config");
    c.onnx_keys.insert("onnx/llm".into());

    let rt = c.routing_target("local", None).expect("routing target");
    assert!(rt.is_onnx, "onnx target must be flagged for the onnx backend");
    assert_eq!(rt.model, "onnx/llm");
    assert!(rt.url.is_empty(), "onnx targets have no HTTP url");
    assert_eq!(rt.group.as_deref(), Some("default"));
    assert_eq!(rt.target_name.as_deref(), Some("local"));
}

#[test]
fn all_dispatch_targets_orders_primary_group_first() {
    let c = sample_config();
    let targets = c.all_dispatch_targets("smart", None);
    let names: Vec<&str> = targets.iter().map(|(n, _)| n.as_str()).collect();
    // Primary group (smart) first, ordered by intelligence descending
    // (huge int 8 before big int 5), then other groups (tiny).
    assert_eq!(&names[..2], &["huge", "big"]);
    assert!(names.contains(&"tiny"));
}
