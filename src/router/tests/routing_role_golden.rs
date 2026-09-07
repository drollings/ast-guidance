//! Role-first resolved-target golden.
//!
//! Pins the dispatch-relevant resolution of the shipped `env/coral-router.json`
//! as the role-first config change lands: every model group (via a
//! representative route at several complexities), every qualifier spelling
//! the config's keys admit, and backend resolvability per member. The corpus
//! lives in `tests/data/routing_role_golden.json`; the test fails on any
//! deviation, so the role-first config must resolve byte-identically.
//!
//! Regeneration is explicit (`UPDATE_GOLDEN=1`), and only meaningful while
//! recording the baseline — after it, the file is frozen and any diff is a
//! regression.

use crate::config::RouterConfig;

fn config_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json")
}

fn golden_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("routing_role_golden.json")
}

fn load_live_config() -> RouterConfig {
    let path = config_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("live config must deserialize: {e}"))
}

/// Representative route per model group (the tree derives one route per group,
//  `local` rides the `default` group).
fn group_routes() -> Vec<(&'static str, &'static str)> {
    vec![
        ("default", "local"),
        ("code", "code"),
        ("summarize", "summarize"),
        ("explore", "explore"),
        ("explain", "explain"),
        ("prose", "prose"),
    ]
}

fn complexities() -> Vec<(String, Option<u8>)> {
    vec![
        ("none".into(), None),
        ("zero".into(), Some(0)),
        ("four".into(), Some(4)),
        ("ten".into(), Some(10)),
    ]
}

fn qualifier_keys() -> Vec<&'static str> {
    vec![
        "code:default",
        "code:latest",
        "code",
        "lfm2.5-2.6b:default",
        "lfm2.5-2.6b",
        "code:missing",
        "nope:default",
    ]
}

fn backend_keys() -> Vec<&'static str> {
    vec!["code:default", "code", "lfm2.5-2.6b", "nope", "last", "any"]
}

fn build_corpus() -> serde_json::Value {
    let config = load_live_config();
    let routing = config.routing_config();

    let mut cases = Vec::new();
    for (group, route) in group_routes() {
        // The group must resolve in the live config, or the corpus is stale.
        assert!(
            routing.model_groups.contains_key(group),
            "shipped config must declare group '{group}'"
        );
        for (label, complexity) in complexities() {
            let target = routing.routing_target(route, complexity);
            cases.push(serde_json::json!({
                "id": format!("group.{group}.route.{route}.complexity.{label}"),
                "target": target,
            }));
        }
    }
    for key in qualifier_keys() {
        cases.push(serde_json::json!({
            "id": format!("target_for_key.{key}"),
            "target": routing.target_for_key(key),
        }));
    }
    let backends: serde_json::Map<String, serde_json::Value> = backend_keys()
        .into_iter()
        .map(|key| {
            (
                key.to_string(),
                serde_json::Value::Bool(config.local_backend(key).is_some()),
            )
        })
        .collect();
    serde_json::json!({ "cases": cases, "backends": backends })
}

#[test]
fn resolved_targets_match_golden() {
    let actual = build_corpus();
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::write(
            golden_path(),
            serde_json::to_string_pretty(&actual).expect("serialize corpus"),
        )
        .expect("write golden file");
        return;
    }
    let expected_raw = std::fs::read_to_string(golden_path()).expect(
        "golden file exists (generate once with UPDATE_GOLDEN=1 before migrating)",
    );
    let expected: serde_json::Value =
        serde_json::from_str(&expected_raw).expect("golden file parses");
    assert_eq!(
        actual, expected,
        "resolved targets drifted from the frozen golden"
    );
}

#[test]
fn golden_spot_checks_dispatch_contract() {    // The golden is opaque by design; these spot checks state its meaning:
    // every group dispatches to the single qualified fleet model with the
    // profile's sampling params, and unknown keys fail closed.
    let corpus = build_corpus();
    let by_id: std::collections::HashMap<&str, &serde_json::Value> = corpus["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| (c["id"].as_str().expect("id"), &c["target"]))
        .collect();

    for (group, route) in group_routes() {
        let target = by_id
            .get(format!("group.{group}.route.{route}.complexity.none").as_str())
            .expect("group case");
        assert_eq!(target["model"], "code:default", "group {group} serves code");
        assert_eq!(
            target["group"], group,
            "resolved target carries its group"
        );
        assert_eq!(target["params"]["temperature"], 0.6);
    }
    assert!(
        by_id["target_for_key.nope:default"].is_null(),
        "unknown base fails closed"
    );
    assert_eq!(
        by_id["target_for_key.code:latest"]["model"], "code:default",
        "latest normalizes to the default point"
    );
    let backends = &corpus["backends"];
    assert_eq!(backends["code:default"], true);
    assert_eq!(backends["nope"], false);
    assert_eq!(backends["last"], false, "sentinels are not backends");
    assert_eq!(backends["any"], false, "sentinels are not backends");
}

#[test]
fn role_vocabulary_serves_shipped_groups() {
    // Groups reference roles, and role keys build backends. The frozen
    // corpus above proves the resolved targets are byte-identical; this pins
    // the vocabulary mechanics on the live config.
    let config = load_live_config();
    assert!(!config.roles.is_empty(), "shipped config declares roles");
    let routing = config.routing_config();

    assert_eq!(
        routing.role_expanded_members("default"),
        vec!["code:default", "last", "any"],
        "role fans out, sentinels pass through"
    );
    assert_eq!(
        routing.role_expanded_members("code"),
        vec!["code:default", "last", "any"],
    );
    assert_eq!(
        routing.role_expanded_members("summarize"),
        vec!["code:default", "last", "any"],
        "role-less groups ride the fleet-default role"
    );

    assert!(
        config.local_backend("default").is_some(),
        "role keys build head-candidate backends"
    );
    assert!(
        config.local_backend("classifier").is_some(),
        "every shipped role builds"
    );
    let backends = config.target_backends();
    assert!(
        backends.contains_key("code:default"),
        "matcher backends key on expanded candidates"
    );
    assert!(
        !backends.contains_key("default") || backends.contains_key("code:default"),
        "no dangling role-keyed backend without its candidate"
    );
}
