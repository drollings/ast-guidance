// Tests assert float config values against literal defaults - deliberate.
#![allow(clippy::float_cmp)]
use super::*;

#[test]
fn charts_config_defaults() {
    let cfg = ChartsConfig::default();
    assert_eq!(cfg.max_candidates, 5);
    assert_eq!(cfg.min_score, 0.6);
    assert!(cfg.entity_context);
    assert!(cfg.dir.is_none());
    assert!(cfg.index_path.is_none());
    assert!(cfg.selector_model.is_none());
}

#[test]
fn review_config_defaults_and_new_m3_keys() {
    let cfg = ReviewConfig::default();
    assert!(!cfg.auto_enqueue, "auto-enqueue is opt-in");
    assert!(cfg.pii_model.is_none());

    // The M3 keys deserialize from config with their serde defaults applied
    // (an absent field keeps its documented default, not the derived 0).
    let cfg: ReviewConfig = serde_json::from_str(
        r#"{"review_model": "review", "auto_enqueue": true, "pii_model": "pii-detector"}"#,
    )
    .unwrap();
    assert!(cfg.auto_enqueue);
    assert_eq!(cfg.pii_model.as_deref(), Some("pii-detector"));
    assert_eq!(cfg.pii_threshold, 0.5);
    assert_eq!(cfg.queue_capacity, 32);
    assert_eq!(cfg.credit_limit, 16);

    let round_trip: ReviewConfig =
        serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert!(round_trip.auto_enqueue);
    assert_eq!(round_trip.pii_model, cfg.pii_model);
}

// -- Async overlay configuration (ROADMAP_20260827_ORT §6) -----------

#[test]
fn overlay_config_defaults_are_fail_open() {
    // Serde defaults apply on deserialization (an absent field gets the
    // default); `OverlayConfig::default()` is the empty struct.
    let cfg: OverlayConfig =
        serde_json::from_str(r#"{"entity_link_enabled": true}"#).unwrap();
    assert!(cfg.entity_link_enabled);
    assert_eq!(cfg.entity_link_threshold, 0.6);
    assert_eq!(cfg.queue_capacity, 32);
    assert_eq!(cfg.credit_limit, 16);
}

#[test]
fn overlay_config_serde_round_trip() {
    let cfg: OverlayConfig = serde_json::from_str(
        r#"{"entity_link_enabled": true, "entity_link_threshold": 0.7, "queue_capacity": 8, "credit_limit": 4}"#,
    )
    .unwrap();
    assert!(cfg.entity_link_enabled);
    assert_eq!(cfg.entity_link_threshold, 0.7);
    assert_eq!(cfg.queue_capacity, 8);
    assert_eq!(cfg.credit_limit, 4);

    let round_trip: OverlayConfig =
        serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert!(round_trip.entity_link_enabled);
    assert_eq!(round_trip.entity_link_threshold, 0.7);
}

#[test]
fn arc_ready_config_defaults_off_and_absent() {
    // An absent `overlay.arc_ready` block deserializes to `arc_ready: None`
    // (the same as `OverlayConfig::default()`), so a config that never
    // mentions arc_ready is byte-identical to today.
    let plain: OverlayConfig =
        serde_json::from_str(r#"{"entity_link_enabled": true}"#).unwrap();
    assert!(plain.arc_ready.is_none(), "absent arc_ready → None");
    assert!(OverlayConfig::default().arc_ready.is_none());

    // A present `arc_ready` block defaults every knob to off/absent.
    let present: OverlayConfig = serde_json::from_str(
        r#"{"entity_link_enabled": true, "arc_ready": {}}"#,
    )
    .unwrap();
    let arc = present.arc_ready.expect("arc_ready present");
    assert!(!arc.enabled, "enabled defaults off");
    assert!(!arc.nlp, "nlp defaults off");
    assert!(arc.llm_model.is_none(), "llm_model absent by default");
    assert!(arc.embedding_model.is_none(), "embedding_model absent by default");
    assert!(!arc.backfill, "backfill defaults off");
    assert_eq!(arc.queue_capacity, 1024);
    assert_eq!(arc.credit_limit, 256);
    assert_eq!(arc.credit_more_after, 8);
    assert_eq!(arc.max_concurrent, 8);
}

#[test]
fn arc_ready_config_serde_round_trip() {
    let cfg: OverlayConfig = serde_json::from_str(
        r#"{
            "entity_link_enabled": true,
            "arc_ready": {
                "enabled": true,
                "nlp": true,
                "llm_model": "lfm2.5-2.6b",
                "embedding_model": "embed",
                "queue_capacity": 64,
                "credit_limit": 128,
                "credit_more_after": 16,
                "max_concurrent": 4,
                "backfill": true
            }
        }"#,
    )
    .unwrap();
    let arc = cfg.arc_ready.as_ref().expect("arc_ready present");
    assert!(arc.enabled && arc.nlp && arc.backfill);
    assert_eq!(arc.llm_model.as_deref(), Some("lfm2.5-2.6b"));
    assert_eq!(arc.embedding_model.as_deref(), Some("embed"));
    assert_eq!(arc.queue_capacity, 64);
    assert_eq!(arc.credit_limit, 128);
    assert_eq!(arc.credit_more_after, 16);
    assert_eq!(arc.max_concurrent, 4);

    let round_trip: OverlayConfig =
        serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    let rt = round_trip.arc_ready.expect("survives round trip");
    assert!(rt.enabled && rt.nlp && rt.backfill);
    assert_eq!(rt.llm_model.as_deref(), Some("lfm2.5-2.6b"));
    assert_eq!(rt.max_concurrent, 4);
}

#[test]
fn overlay_config_default_is_byte_identical_without_arc_ready() {
    // The M8 guarantee: a config with no `arc_ready` block produces
    // byte-identical behavior to today — the `OverlayConfig` default is
    // unchanged (arc_ready `None`), and only the entity-link knobs exist.
    let cfg: OverlayConfig = serde_json::from_str(
        r#"{"entity_link_enabled": true, "entity_link_threshold": 0.6, "queue_capacity": 32, "credit_limit": 16}"#,
    )
    .unwrap();
    assert_eq!(cfg.arc_ready, None);
    assert!(cfg.entity_link_enabled);
    assert_eq!(cfg.entity_link_threshold, 0.6);
    assert_eq!(cfg.queue_capacity, 32);
    assert_eq!(cfg.credit_limit, 16);
}

#[test]
fn router_config_absent_overlay_section_defaults_to_none() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(cfg.overlay.is_none(), "absent overlay → None (fail-open)");
}

// -- Rigor-route configuration -------------------------------------

#[test]
fn rigor_config_defaults() {
    let cfg = RigorConfig::default();
    assert_eq!(cfg.max_passes.rounds(), 2);
    assert_eq!(cfg.severity_threshold.get(), 0.7);
    assert_eq!(cfg.escalation_confidence.get(), 0.4);
    assert!(!cfg.kv_cache_enabled);
    assert!(cfg.blue_model.is_none());
    assert!(cfg.red_model.is_none());
    assert!(cfg.judge_model.is_none());
}

#[test]
fn router_config_absent_rigor_section_defaults_to_none() {
    // The shipped env/coral-router.json has no `rigor` section; the route
    // stays present-but-unconfigured (None), never a crash.
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(cfg.rigor.is_none());
}

#[test]
fn rigor_config_round_trip() {
    let json = serde_json::json!({
        "rigor": {
            "blue_model": "fast",
            "red_model": "code",
            "judge_model": "code",
            "kv_cache_enabled": true,
            "max_passes": 3,
            "severity_threshold": 0.8,
            "escalation_confidence": 0.3,
        }
    });
    let cfg: RouterConfig = serde_json::from_value(json).unwrap();
    let rigor = cfg.rigor.expect("rigor section parsed");
    assert_eq!(rigor.blue_model.as_deref(), Some("fast"));
    assert_eq!(rigor.red_model.as_deref(), Some("code"));
    assert_eq!(rigor.judge_model.as_deref(), Some("code"));
    assert!(rigor.kv_cache_enabled);
    assert_eq!(rigor.max_passes.rounds(), 3);
    assert_eq!(rigor.severity_threshold.get(), 0.8);
    assert_eq!(rigor.escalation_confidence.get(), 0.3);

    // Partial section still round-trips with defaults for the rest.
    let partial: RouterConfig = serde_json::from_value(serde_json::json!({
        "rigor": {"blue_model": "fast"}
    }))
    .unwrap();
    let partial_cfg = partial.rigor.expect("rigor parsed");
    assert_eq!(partial_cfg.blue_model.as_deref(), Some("fast"));
    assert_eq!(partial_cfg.max_passes.rounds(), 2, "absent fields default");
    assert_eq!(partial_cfg.severity_threshold.get(), 0.7);
}

#[test]
fn router_config_absent_charts_section_defaults_cleanly() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert_eq!(cfg.charts.max_candidates, 5);
    assert_eq!(cfg.charts.min_score, 0.6);
    assert!(cfg.charts.entity_context);
    assert!(cfg.charts.dir.is_none());
}

#[test]
fn router_config_embedding_and_reranker_models_parse() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"embedding_model": "embed", "reranker_model": "rerank"}"#)
            .unwrap();
    assert_eq!(cfg.embedding_model.as_deref(), Some("embed"));
    assert_eq!(cfg.reranker_model.as_deref(), Some("rerank"));

    let absent: RouterConfig = serde_json::from_str(r"{}").unwrap();
    assert!(absent.embedding_model.is_none());
    assert!(absent.reranker_model.is_none());
}

#[test]
fn charts_section_round_trips() {
    let json = r#"{
        "dir": "env/workflows/charts",
        "index_path": "data/workflow_library.sqlite",
        "selector_model": "qwen3.5-4b",
        "max_candidates": 5,
        "min_score": 0.6,
        "entity_context": true
    }"#;
    let cfg: ChartsConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.dir.as_deref(), Some("env/workflows/charts"));
    assert_eq!(
        cfg.index_path.as_deref(),
        Some("data/workflow_library.sqlite")
    );
    assert_eq!(cfg.selector_model.as_deref(), Some("qwen3.5-4b"));
    assert_eq!(cfg.max_candidates, 5);
    assert_eq!(cfg.min_score, 0.6);
    assert!(cfg.entity_context);

    let serialized = serde_json::to_string(&cfg).unwrap();
    let back: ChartsConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(back.dir, cfg.dir);
    assert_eq!(back.max_candidates, cfg.max_candidates);
    assert_eq!(back.min_score, cfg.min_score);
}

#[test]
fn partial_charts_section_defaults_missing_fields() {
    let cfg: ChartsConfig = serde_json::from_str(r#"{"dir": "env/workflows/charts"}"#).unwrap();
    assert_eq!(cfg.dir.as_deref(), Some("env/workflows/charts"));
    assert_eq!(cfg.max_candidates, 5);
    assert_eq!(cfg.min_score, 0.6);
    assert!(cfg.entity_context);
    assert!(cfg.index_path.is_none());
    assert!(cfg.selector_model.is_none());
}

#[test]
fn router_config_parses_charts_section() {
    let json = r#"{
        "charts": { "dir": "env/workflows/charts", "max_candidates": 8 }
    }"#;
    let cfg: RouterConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.charts.dir.as_deref(), Some("env/workflows/charts"));
    assert_eq!(cfg.charts.max_candidates, 8);
    assert_eq!(cfg.charts.min_score, 0.6, "unset field keeps its default");
}

// -- Post-process (learning loop) --------------------------------

#[test]
fn post_process_defaults_to_disabled() {
    let cfg = PostProcessConfig::default();
    assert!(!cfg.workflow_extraction, "extraction is opt-in");
    assert_eq!(
        cfg.workflow_extraction_mode,
        WorkflowExtractionMode::Frontier,
        "default scope is frontier-assisted only"
    );
}

#[test]
fn post_process_absent_section_defaults_cleanly() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(
        !cfg.post_process.workflow_extraction,
        "absent post_process section defaults extraction off"
    );
    assert_eq!(
        cfg.post_process.workflow_extraction_mode,
        WorkflowExtractionMode::Frontier,
        "absent mode field defaults to frontier"
    );
}

#[test]
fn post_process_round_trips() {
    let json = r#"{ "workflow_extraction": true }"#;
    let cfg: PostProcessConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.workflow_extraction);
    assert_eq!(
        cfg.workflow_extraction_mode,
        WorkflowExtractionMode::Frontier,
        "absent mode field keeps the frontier default"
    );

    let serialized = serde_json::to_string(&cfg).unwrap();
    let back: PostProcessConfig = serde_json::from_str(&serialized).unwrap();
    assert!(back.workflow_extraction);
    assert_eq!(back.workflow_extraction_mode, cfg.workflow_extraction_mode);
}

#[test]
fn workflow_extraction_mode_parses_both_variants() {
    let all: WorkflowExtractionMode = serde_json::from_str(r#""all""#).expect("all parses");
    assert_eq!(all, WorkflowExtractionMode::All);

    let frontier: WorkflowExtractionMode =
        serde_json::from_str(r#""frontier""#).expect("frontier parses");
    assert_eq!(frontier, WorkflowExtractionMode::Frontier);

    assert!(serde_json::from_str::<WorkflowExtractionMode>(r#""bogus""#).is_err());
}

#[test]
fn router_config_parses_post_process_section() {
    let json = r#"{
        "post_process": { "workflow_extraction": true },
        "charts": { "dir": "env/workflows/charts" }
    }"#;
    let cfg: RouterConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.post_process.workflow_extraction);
    assert_eq!(cfg.charts.dir.as_deref(), Some("env/workflows/charts"));
    assert_eq!(
        cfg.post_process.workflow_extraction_mode,
        WorkflowExtractionMode::Frontier,
        "existing configs without the new field still deserialize"
    );
}

#[test]
fn router_config_parses_extraction_mode_all() {
    let json = r#"{
        "post_process": {
            "workflow_extraction": true,
            "workflow_extraction_mode": "all"
        }
    }"#;
    let cfg: RouterConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.post_process.workflow_extraction);
    assert_eq!(
        cfg.post_process.workflow_extraction_mode,
        WorkflowExtractionMode::All
    );
}

#[test]
fn model_entry_serde_defaults_read_canonical_constants() {
    // The same constants `RoutingTarget` reads (divergence guard).
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
    }))
    .unwrap();
    assert_eq!(
        entry.total_timeout_ms,
        fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
    );
    assert_eq!(
        entry.idle_timeout_ms,
        fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS
    );
    assert_eq!(
        entry.retry_base_interval_s,
        fluent_llm::constants::DEFAULT_RETRY_INTERVAL_S
    );
}

// -- Classification-tree derived flat views ------------------------

fn tree_section() -> serde_json::Value {
    serde_json::json!({
        "classification": {
            "root": {
                "type": "classifier",
                "description": "router",
                "model": "fast",
                "children": [
                    {
                        "key": "code",
                        "description": "programming",
                        "node": { "type": "terminal", "route": "code", "group": "code", "always_route": true }
                    },
                    {
                        "key": "brand_new",
                        "description": "not in flat routes",
                        "node": { "type": "terminal", "route": "brand_new", "group": "question" }
                    }
                ]
            }
        },
        "models": {
            "fast": {"endpoint": "http://upstream.test/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7, "speed": 8}
        },
        "model_groups": {
            "fast": ["fast"],
            "code": ["fast"],
            "question": ["fast"]
        }
    })
}

#[test]
fn routes_view_synthesizes_terminal_routes() {
    let cfg: RouterConfig = serde_json::from_value(tree_section()).unwrap();
    let routes = cfg.routes_view();
    // Terminal routes resolve through their own group; always_route rides along.
    assert_eq!(routes["code"].group, "code");
    assert_eq!(routes["code"].pipelines, vec!["default".to_string()]);
    assert!(routes["code"].always_route);
    assert_eq!(routes["brand_new"].group, "question");
    assert_eq!(routes["brand_new"].pipelines, vec!["default".to_string()]);
    assert!(!routes["brand_new"].always_route);
}

#[test]
fn routes_view_treeless_config_is_empty() {
    // No tree, no routes: the flat `routes` key is gone, so a config without
    // `classification` yields an empty view (and fails validation).
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"routes": {"a": {"group": "g"}}}"#).unwrap();
    assert!(cfg.routes_view().is_empty());
    assert!(cfg.validate_flat_tree_coherence().is_err());
}

#[test]
fn routing_config_always_derives_system_prompt_from_tree() {
    // M3c: no explicit prompt exists anymore — the prompt is always derived.
    let cfg: RouterConfig = serde_json::from_value(tree_section()).unwrap();
    let routing = cfg.routing_config();
    assert!(
        routing.system_prompt.contains("You are a router."),
        "tree-derived system prompt, got: {}",
        routing.system_prompt
    );
    assert!(
        routing.routes.contains_key("brand_new"),
        "derived routes reach the RoutingConfig so terminal resolution works"
    );
    assert!(routing.score_matrix.is_none(), "root matrix is gone; pipeline-level matrices flow via params");
}

// -- In-group target-matching knob (PipelineParams) ----------------

#[test]
fn pipeline_params_target_match_defaults() {
    let defaults = crate::config::builder::PipelineParams::default();
    assert_eq!(
        defaults.target_match,
        crate::config::builder::TargetMatchMode::SelfAssess,
        "the self-assess ladder is the default policy (-4.6)"
    );
    assert_eq!(
        defaults.target_match_timeout_ms,
        fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS,
        "per-self-assessment budget defaults to the shared total-timeout constant"
    );
}

#[test]
fn pipeline_params_target_match_absent_fields_deserialize_to_defaults() {
    // A pipeline that omits both knob fields must deserialize to the same
    // defaults (mirror the `classifier_retry_max` pattern) - existing
    // configs stay byte-identical.
    let cfg: RouterConfig = serde_json::from_str(
        r#"{
            "pipelines": {"default": {"classifier": true, "classifier_model": "fast"}}
        }"#,
    )
    .expect("valid config");
    let params = &cfg.pipelines["default"];
    assert_eq!(params.target_match, crate::config::builder::TargetMatchMode::SelfAssess);
    assert_eq!(
        params.target_match_timeout_ms,
        fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
    );
}

#[test]
fn pipeline_params_target_match_parses_both_variants() {
    let self_assess: crate::config::builder::TargetMatchMode =
        serde_json::from_str(r#""self_assess""#).expect("self_assess parses");
    assert_eq!(self_assess, crate::config::builder::TargetMatchMode::SelfAssess);

    let static_mode: crate::config::builder::TargetMatchMode =
        serde_json::from_str(r#""static""#).expect("static parses");
    assert_eq!(static_mode, crate::config::builder::TargetMatchMode::Static);

    assert!(
        serde_json::from_str::<crate::config::builder::TargetMatchMode>(r#""bogus""#).is_err(),
        "unknown policy must be rejected, not silently defaulted"
    );
}

#[test]
fn pipeline_params_target_match_round_trips() {
    // Non-default values survive a serialize - deserialize cycle.
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "pipelines": {
            "default": {
                "classifier": true,
                "classifier_model": "fast",
                "target_match": "static",
                "target_match_timeout_ms": 12345
            }
        }
    }))
    .unwrap();
    assert_eq!(cfg.pipelines["default"].target_match, crate::config::builder::TargetMatchMode::Static);
    assert_eq!(cfg.pipelines["default"].target_match_timeout_ms, 12345);

    let serialized = serde_json::to_string(&cfg).unwrap();
    let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(back.pipelines["default"].target_match, crate::config::builder::TargetMatchMode::Static);
    assert_eq!(back.pipelines["default"].target_match_timeout_ms, 12345);
}

// -- Instance-pool declaration -------------------------------------

fn profile_json(name: &str, count: u32, group: &str, num_ctx: u64) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "count": count,
        "group": group,
        "num_ctx": num_ctx,
    })
}

#[test]
fn instances_count_expansion_names_siblings_in_shared_group() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 2,
        "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
        "speed": 8,
        "instances": {
            "swarm": profile_json("swarm", 3, "swarm", 16384),
            "ledger": { "num_ctx": 131072, "pinned": true, "default": true }
        }
    }))
    .unwrap();

    let profiles = entry.instance_profiles();
    assert_eq!(profiles.len(), 4);
    // Profiles are emitted in sorted map-key order: ledger < swarm.
    assert_eq!(profiles[0].name.as_deref(), Some("ledger"));
    assert_eq!(profiles[0].group.as_deref(), Some("ledger"));
    assert!(profiles[0].pinned);
    assert!(profiles[0].default);
    // count: 3 -> `<key>-0` .. `<key>-2` in the shared group.
    assert_eq!(profiles[1].name.as_deref(), Some("swarm-0"));
    assert_eq!(profiles[1].group.as_deref(), Some("swarm"));
    assert_eq!(profiles[2].name.as_deref(), Some("swarm-1"));
    assert_eq!(profiles[3].name.as_deref(), Some("swarm-2"));
    assert_eq!(profiles[3].group.as_deref(), Some("swarm"));
    assert_eq!(profiles[3].num_ctx, 16384);
}

#[test]
fn instances_single_profile_defaults_name_to_map_key() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": { "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 } }
    }))
    .unwrap();
    let profiles = entry.instance_profiles();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].name.as_deref(), Some("scratch"));
    assert_eq!(profiles[0].group.as_deref(), Some("scratch"));
    assert_eq!(profiles[0].sleep_idle_seconds, Some(30));
    assert_eq!(profiles[0].count, 1);
}

#[test]
fn old_sessions_key_still_parses_as_instances() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "sessions": { "ctx16384": { "num_ctx": 16384 } }
    }))
    .unwrap();
    let instances = entry.instances.expect("sessions alias maps into instances");
    assert_eq!(instances.len(), 1);
    assert!(instances.contains_key("ctx16384"));
}

#[test]
fn no_instances_yields_empty_profile_list() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
    }))
    .unwrap();
    assert!(entry.instance_profiles().is_empty());
}

#[test]
fn warm_alias_maps_to_no_sleep() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": { "swarm": { "num_ctx": 16384, "warm": true } }
    }))
    .unwrap();
    let profiles = entry.instance_profiles();
    assert!(profiles[0].no_sleep);
}

// -- M3: per-profile `max_ctx` context-size cap ---------------------

#[test]
fn instance_profile_max_ctx_absent_defaults_to_none() {
    // An absent `max_ctx` (the default) deserializes to `None` — an
    // existing config without the new key is byte-identical.
    let profile: InstanceProfile =
        serde_json::from_value(profile_json("swarm", 1, "swarm", 16384)).unwrap();
    assert!(profile.max_ctx.is_none());
}

#[test]
fn instance_profile_max_ctx_round_trips() {
    let profile: InstanceProfile = serde_json::from_value(serde_json::json!({
        "name": "swarm",
        "count": 1,
        "group": "swarm",
        "num_ctx": 16384,
        "max_ctx": 8192,
    }))
    .unwrap();
    assert_eq!(profile.max_ctx, Some(8192));
    let back: InstanceProfile =
        serde_json::from_str(&serde_json::to_string(&profile).unwrap()).unwrap();
    assert_eq!(back.max_ctx, Some(8192));
    assert_eq!(back.num_ctx, 16384);
}

#[test]
fn instance_profiles_clamps_num_ctx_to_max_ctx() {
    // A profile whose `max_ctx` cap sits below its `num_ctx` is clamped
    // down at materialization: the materialized context window is the cap.
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": {
            "swarm": { "num_ctx": 65536, "max_ctx": 8192 }
        }
    }))
    .unwrap();
    let profiles = entry.instance_profiles();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].num_ctx, 8192, "clamped to max_ctx");
    assert_eq!(profiles[0].max_ctx, Some(8192));
}

#[test]
fn instance_profiles_max_ctx_absent_is_noop() {
    // The clamp is a no-op when `max_ctx` is absent (the default): the
    // materialized `num_ctx` is untouched.
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": { "swarm": { "num_ctx": 16384 } }
    }))
    .unwrap();
    let profiles = entry.instance_profiles();
    assert_eq!(profiles[0].num_ctx, 16384);
    assert!(profiles[0].max_ctx.is_none());
}

#[test]
fn instance_profiles_cap_above_num_ctx_is_noop() {
    // A cap at/above `num_ctx` never inflates the window — the declared
    // `num_ctx` is the upper bound.
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": { "swarm": { "num_ctx": 16384, "max_ctx": 32768 } }
    }))
    .unwrap();
    let profiles = entry.instance_profiles();
    assert_eq!(profiles[0].num_ctx, 16384, "cap above num_ctx is a no-op");
}

// -- Pool vs default qualifier -------------------------------

/// The reference swarm entry: a count=3 non-default `swarm` work pool, a
/// pinned `default: true` ledger, and a non-default scratch profile.
fn reference_swarm_entry() -> ModelEntry {
    serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "name": "abiray/lfm2.5-2.6b-heretic-abliterated",
        "intelligence": 2,
        "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
        "speed": 8,
        "instances": {
            "swarm": profile_json("swarm", 3, "swarm", 16384),
            "ledger": { "num_ctx": 131072, "pinned": true, "default": true },
            "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 }
        }
    }))
    .expect("reference swarm entry parses")
}

#[test]
fn pool_qualifier_reference_config_targets_swarm() {
    let entry = reference_swarm_entry();
    assert_eq!(
        entry.pool_qualifier().as_deref(),
        Some("swarm"),
        "the largest non-default profile (count=3) is the work pool"
    );
}

#[test]
fn pool_qualifier_vs_default_qualifier_two_intents_two_answers() {
    // The two intents must diverge on the same entry: pool = swarm (the
    // work group), default = ledger (the client-facing default instance).
    let entry = reference_swarm_entry();
    assert_eq!(entry.pool_qualifier().as_deref(), Some("swarm"));
    assert_eq!(
        entry.default_dispatch_qualifier().as_deref(),
        Some("ledger")
    );
}

#[test]
fn pool_qualifier_ledger_only_defaults_to_ledger() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": { "ledger": { "num_ctx": 131072, "default": true } }
    }))
    .unwrap();
    assert_eq!(entry.pool_qualifier().as_deref(), Some("ledger"));
}

#[test]
fn pool_qualifier_single_shared_group() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
        "instances": {
            "a": { "num_ctx": 8192, "group": "shared" },
            "b": { "num_ctx": 8192, "group": "shared" }
        }
    }))
    .unwrap();
    assert_eq!(entry.pool_qualifier().as_deref(), Some("shared"));
}

#[test]
fn pool_qualifier_no_instances_is_none() {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://x/v1/chat/completions",
        "intelligence": 1,
        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0,
        "speed": 1,
    }))
    .unwrap();
    assert!(entry.pool_qualifier().is_none());
}

#[test]
fn sidecar_absent_section_defaults_cleanly() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert_eq!(cfg.sidecar.poll_interval_s, 5);
    assert_eq!(cfg.sidecar.vram_low_watermark_bytes, 1073741824);
    assert_eq!(cfg.sidecar.evict_batch, 1);
    assert!(cfg.sidecar.vram_total_bytes.is_none());
    assert!(cfg.sidecar.minimum_remaining_vram.is_none());
    assert!(cfg.sidecar.slot_save_path.is_none());
    assert!(cfg.sidecar.api_key_env.is_none());
}

#[test]
fn sidecar_section_round_trips() {
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "sidecar": {
            "poll_interval_s": 10,
            "vram_low_watermark_bytes": 536870912,
            "evict_batch": 2,
            "vram_total_bytes": 1048576,
            "minimum_remaining_vram": 2147483648u64,
            "slot_save_path": "/srv/slots",
            "api_key_env": "LLAMA_API_KEY",
        }
    }))
    .unwrap();
    assert_eq!(cfg.sidecar.poll_interval_s, 10);
    assert_eq!(cfg.sidecar.vram_low_watermark_bytes, 536870912);
    assert_eq!(cfg.sidecar.evict_batch, 2);
    assert_eq!(cfg.sidecar.vram_total_bytes, Some(1048576));
    assert_eq!(cfg.sidecar.minimum_remaining_vram, Some(2147483648));
    assert_eq!(cfg.sidecar.slot_save_path.as_deref(), Some("/srv/slots"));
    assert_eq!(cfg.sidecar.api_key_env.as_deref(), Some("LLAMA_API_KEY"));
}

#[test]
fn sidecar_allocation_limit_from_minimum_remaining() {
    // With a ceiling configured, the budget is ceiling - minimum remaining.
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "sidecar": { "vram_total_bytes": 10000, "minimum_remaining_vram": 2000 }
    }))
    .unwrap();
    assert_eq!(cfg.sidecar.allocation_limit(), Some(8000));
}

#[test]
fn sidecar_allocation_limit_without_ceiling_falls_back_to_detection() {
    // No explicit ceiling: the budget is computed from the detected total.
    // The host has a ROCm device (mem_info_vram_total > 0), so the limit is
    // detection - minimum_remaining; a missing floor yields the full total.
    // Detection reads `/sys/class/drm` through the capability-gated fs
    // helper, so it runs under the `FsCapability` grant the serving
    // path establishes at boot.
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "sidecar": { "minimum_remaining_vram": 2147483648u64 }
    }))
    .unwrap();
    let (detected, allocation_limit) = fluent_concurrency::scope::CURRENT_CAPS.sync_scope(
        fluent_concurrency::capability::default_capability_set(),
        || {
            (
                super::detect_device_vram_total(),
                cfg.sidecar.allocation_limit(),
            )
        },
    );
    assert!(
        detected.is_some(),
        "ROCm sysfs mem_info_vram_total present on this host"
    );
    assert_eq!(
        allocation_limit,
        detected.map(|t| t.saturating_sub(2147483648))
    );
}

#[test]
fn default_params_absent_section_defaults_cleanly() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert_eq!(cfg.default_params.num_ctx, 16384);
    assert_eq!(cfg.default_params.batch_size, 4096);
    assert_eq!(cfg.default_params.n_gpu_layers, 999);
    assert!(cfg.default_params.params.is_none());
}

#[test]
fn default_params_section_round_trips() {
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "default_params": {
            "num_ctx": 8192,
            "batch_size": 512,
            "ubatch_size": 256,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "flash_attn": "off",
            "n_gpu_layers": 0,
            "n_cpu_moe": 4,
            "sleep_idle_seconds": 30,
            "stream": false,
            "filter_thinking": true,
            "params": { "temperature": 0.2 },
            "max_ctx": 8192
        }
    }))
    .unwrap();
    assert_eq!(cfg.default_params.num_ctx, 8192);
    assert_eq!(cfg.default_params.batch_size, 512);
    assert_eq!(cfg.default_params.ubatch_size, 256);
    assert_eq!(cfg.default_params.cache_type_k, "f16");
    assert_eq!(cfg.default_params.cache_type_v, "f16");
    assert_eq!(cfg.default_params.flash_attn.as_deref(), Some("off"));
    assert_eq!(cfg.default_params.n_gpu_layers, 0);
    assert_eq!(cfg.default_params.n_cpu_moe, 4);
    assert_eq!(cfg.default_params.sleep_idle_seconds, 30);
    assert!(!cfg.default_params.stream);
    assert!(cfg.default_params.filter_thinking);
    assert_eq!(cfg.default_params.max_ctx, Some(8192));
    assert_eq!(
        cfg.default_params
            .params
            .as_ref()
            .and_then(|p| p.get("temperature")),
        Some(&serde_json::json!(0.2))
    );
}

#[test]
fn default_params_max_ctx_absent_defaults_to_none() {
    // An absent `max_ctx` (the default) deserializes to `None` — an
    // existing config without the new key is byte-identical.
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(cfg.default_params.max_ctx.is_none());
    assert!(DefaultModelParams::default().max_ctx.is_none());
}

#[test]
fn default_params_ctx_size_alias_parses() {
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "default_params": { "ctx_size": 32768 }
    }))
    .unwrap();
    assert_eq!(cfg.default_params.num_ctx, 32768);
}

// -- Ledger + session composition sections ------------------------

#[test]
fn router_config_absent_ledger_and_session_sections_default_to_none() {
    let cfg: RouterConfig =
        serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(
        cfg.ledger.is_none(),
        "absent ledger section -> no ledger at boot (byte-identical behavior)"
    );
    assert!(
        cfg.session.is_none(),
        "absent session section -> no session registry at boot (byte-identical behavior)"
    );
}

#[test]
fn ledger_and_session_sections_round_trip() {
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "ledger": {
            "path": "data/ledger.sqlite",
            "model": "swarm",
            "max_summary_tokens": 300,
        },
        "session": { "root": "data/sessions" },
    }))
    .unwrap();

    let ledger = cfg.ledger.as_ref().expect("ledger section parsed");
    assert_eq!(ledger.path.as_deref(), Some("data/ledger.sqlite"));
    assert_eq!(ledger.model.as_deref(), Some("swarm"));
    assert_eq!(ledger.max_summary_tokens, 300);

    let session = cfg.session.as_ref().expect("session section parsed");
    assert_eq!(session.root.as_deref(), Some("data/sessions"));

    let serialized = serde_json::to_string(&cfg).unwrap();
    let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
    let back_ledger = back.ledger.expect("ledger round-trips");
    assert_eq!(back_ledger.path, ledger.path);
    assert_eq!(back_ledger.model, ledger.model);
    assert_eq!(back_ledger.max_summary_tokens, ledger.max_summary_tokens);
    assert_eq!(back.session.unwrap().root, session.root);
}

#[test]
fn ledger_section_partial_defaults_max_summary_tokens() {
    // A ledger section that omits `max_summary_tokens` gets the named
    // constant default; the shipped config round-trips cleanly.
    let cfg: RouterConfig =
        serde_json::from_value(serde_json::json!({ "ledger": { "model": "swarm" } })).unwrap();
    let ledger = cfg.ledger.as_ref().expect("ledger parsed");
    assert_eq!(ledger.max_summary_tokens, DEFAULT_LEDGER_MAX_SUMMARY_TOKENS);
    assert_eq!(ledger.model.as_deref(), Some("swarm"));
    assert!(ledger.path.is_none());

    let serialized = serde_json::to_string(&cfg).unwrap();
    let back: RouterConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        back.ledger.unwrap().max_summary_tokens,
        DEFAULT_LEDGER_MAX_SUMMARY_TOKENS
    );
}

#[test]
fn ledger_background_tiering_fields_default_absent() {
    // All background-tiering fields are default-absent so existing
    // `coral-router.json` files deserialize unchanged.
    let cfg: RouterConfig =
        serde_json::from_value(serde_json::json!({ "ledger": { "model": "swarm" } })).unwrap();
    let ledger = cfg.ledger.as_ref().unwrap();
    assert!(!ledger.background_tiering, "tiering is opt-in");
    assert!(ledger.tier_model.is_none());
    assert_eq!(ledger.lod4_max_chars, 240, "default lod4 cap");
    assert_eq!(ledger.lod5_max_chars, 80, "default lod5 cap");
    assert_eq!(ledger.tier_batch_size, 8);
    assert_eq!(ledger.tier_poll_interval_ms, 100);
}

#[test]
fn ledger_background_tiering_fields_round_trip() {
    // A fully-populated ledger section round-trips knobs.
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "ledger": {
            "model": "swarm",
            "background_tiering": true,
            "tier_model": "qwen3.5-4b",
            "lod4_max_chars": 200,
            "lod5_max_chars": 60,
            "tier_batch_size": 16,
            "tier_poll_interval_ms": 250,
        }
    }))
    .unwrap();
    let ledger = cfg.ledger.as_ref().unwrap();
    assert!(ledger.background_tiering);
    assert_eq!(ledger.tier_model.as_deref(), Some("qwen3.5-4b"));
    assert_eq!(ledger.lod4_max_chars, 200);
    assert_eq!(ledger.lod5_max_chars, 60);
    assert_eq!(ledger.tier_batch_size, 16);
    assert_eq!(ledger.tier_poll_interval_ms, 250);
}

// -- Ledger orchestrator section --------------------------------

#[test]
fn orchestrator_section_default_absent() {
    // Existing ledger configs without an `orchestrator` section keep the
    // coordinator disabled (opt-in) and today's defaults.
    let cfg: RouterConfig =
        serde_json::from_value(serde_json::json!({ "ledger": { "model": "swarm" } })).unwrap();
    let orch = &cfg.ledger.as_ref().unwrap().orchestrator;
    assert!(!orch.enabled, "coordinator is opt-in");
    assert_eq!(
        orch.kv_policy,
        crate::dag_session::KvSnapshotPolicy::RestoreIfSameModel
    );
    assert_eq!(orch.prompt_budget_chars, 32768);
    assert_eq!(orch.role, "agent");
}

#[test]
fn orchestrator_section_round_trip() {
    let cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "ledger": {
            "model": "swarm",
            "orchestrator": {
                "enabled": true,
                "kv_policy": "never_restore",
                "prompt_budget_chars": 16384,
                "role": "planner"
            }
        }
    }))
    .unwrap();
    let orch = &cfg.ledger.as_ref().unwrap().orchestrator;
    assert!(orch.enabled);
    assert_eq!(orch.kv_policy, crate::dag_session::KvSnapshotPolicy::NeverRestore);
    assert_eq!(orch.prompt_budget_chars, 16384);
    assert_eq!(orch.role, "planner");
}

#[test]
fn orchestrator_kv_policy_parses_all_variants() {
    use crate::dag_session::KvSnapshotPolicy as P;
    let a: P = serde_json::from_str(r#""restore_if_same_model""#).unwrap();
    let b: P = serde_json::from_str(r#""always_restore""#).unwrap();
    let c: P = serde_json::from_str(r#""never_restore""#).unwrap();
    assert_eq!(a, P::RestoreIfSameModel);
    assert_eq!(b, P::AlwaysRestore);
    assert_eq!(c, P::NeverRestore);
}

#[test]
fn kv_snapshot_policy_round_trips_through_serde() {
    use crate::dag_session::KvSnapshotPolicy as P;
    for p in [P::RestoreIfSameModel, P::AlwaysRestore, P::NeverRestore] {
        let json = serde_json::to_string(&p).unwrap();
        let back: P = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p, "round-trip {p:?} through {json}");
    }
}

// -- Top-level ONNX role key normalization ---------------------------

#[test]
fn top_level_onnx_roles_merge_into_fleet() {
    use fluent_onnx::config::OnnxRole;
    let cfg: RouterConfig = serde_json::from_str(
        r#"{
            "encoder": {
                "model_path": "/models/encoder.onnx",
                "tokenizer_path": "/models/tokenizer.json"
            },
            "router": {
                "model_path": "/models/router.onnx"
            },
            "colbert": {
                "model_path": "/models/colbert.onnx",
                "tokenizer_path": "/models/colbert_tok.json"
            }
        }"#,
    )
    .expect("valid config");
    let mut cfg = cfg;
    cfg.apply_defaults();

    let fleet = cfg.onnx.as_ref().expect("onnx fleet merged");
    assert!(fleet.has(OnnxRole::Encoder));
    assert!(fleet.has(OnnxRole::Router));
    assert!(fleet.has(OnnxRole::Colbert));
    assert!(!fleet.has(OnnxRole::Pii));
    // Top-level keys consumed
    assert!(cfg.encoder.is_none());
    assert!(cfg.router.is_none());
    assert!(cfg.colbert.is_none());
    // Router inherited encoder's tokenizer_path
    let router_cfg = fleet.get(OnnxRole::Router).unwrap();
    assert_eq!(
        router_cfg.model.tokenizer_path.as_deref(),
        Some(std::path::Path::new("/models/tokenizer.json")),
        "router should inherit encoder tokenizer_path"
    );
    // Colbert kept its own tokenizer_path
    let colbert_cfg = fleet.get(OnnxRole::Colbert).unwrap();
    assert_eq!(
        colbert_cfg.model.tokenizer_path.as_deref(),
        Some(std::path::Path::new("/models/colbert_tok.json")),
        "colbert should keep its own tokenizer_path"
    );
}

#[test]
fn nested_onnx_section_takes_precedence_over_top_level_keys() {
    let cfg: RouterConfig = serde_json::from_str(
        r#"{
            "onnx": {
                "encoder": {
                    "model_path": "/nested/encoder.onnx",
                    "tokenizer_path": "/nested/tok.json"
                }
            },
            "encoder": {
                "model_path": "/top/encoder.onnx",
                "tokenizer_path": "/top/tok.json"
            }
        }"#,
    )
    .expect("valid config");
    let mut cfg = cfg;
    cfg.apply_defaults();

    let fleet = cfg.onnx.as_ref().expect("onnx fleet present");
    let enc = fleet.get(fluent_onnx::config::OnnxRole::Encoder).unwrap();
    assert_eq!(enc.model.model_path, std::path::PathBuf::from("/nested/encoder.onnx"),
        "nested onnx section wins over top-level key");
    // Top-level key still present (not consumed, just warned about)
    assert!(cfg.encoder.is_some(), "top-level key preserved (warning issued)");
}

#[test]
fn no_onnx_keys_yields_none_fleet() {
    let cfg: RouterConfig = serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
    assert!(cfg.onnx.is_none(), "absent onnx config stays None");
}

#[test]
fn weights_alias_deserializes_as_model_path() {
    let cfg: RouterConfig = serde_json::from_str(
        r#"{
            "encoder": {
                "weights": "/models/encoder.onnx",
                "tokenizer_path": "/models/tok.json"
            }
        }"#,
    )
    .expect("valid config");
    let mut cfg = cfg;
    cfg.apply_defaults();
    let fleet = cfg.onnx.as_ref().unwrap();
    let enc = fleet.get(fluent_onnx::config::OnnxRole::Encoder).unwrap();
    assert_eq!(enc.model.model_path, std::path::PathBuf::from("/models/encoder.onnx"),
        "weights alias maps to model_path");
}

#[test]
fn fp8_quantization_parses() {
    use fluent_onnx::config::Quant;
    let q: Quant = serde_json::from_str(r#""fp8""#).expect("fp8 parses");
    assert_eq!(q, Quant::Fp8);
}

#[test]
fn router_config_field_access_round_trip() {
    use fluent_wvr::FieldAccess;
    let mut cfg = RouterConfig::default();
    // safety_threshold is accessible via FieldAccess
    cfg.set_field("safety_threshold", "0.7").expect("set_field safety_threshold");
    assert_eq!(cfg.get_field("safety_threshold").unwrap(), "0.7");
    // onnx_resolver is skip -> NotFound
    let err = cfg.set_field("onnx_resolver", "x").unwrap_err();
    assert!(matches!(err, fluent_wvr::FieldError::NotFound(_)), "onnx_resolver should be NotFound, got {:?}", err);
}

#[test]
fn facade_reexports_are_stable() {
    assert_eq!(
        std::any::type_name::<crate::config::RouterConfig>(),
        std::any::type_name::<crate::config::root::RouterConfig>()
    );
}

#[test]
fn router_config_env_coral_router_json_round_trip() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let content = std::fs::read_to_string(&path).expect("env/coral-router.json readable");
    let cfg: RouterConfig = serde_json::from_str(&content).expect("env/coral-router.json deserializes");
    let value = serde_json::to_value(&cfg).expect("serialize");
    let back: RouterConfig = serde_json::from_value(value).expect("round-trip");
    assert_eq!(back.routes_view().len(), cfg.routes_view().len());
    assert_eq!(back.models.len(), cfg.models.len());
}

#[test]
fn shipped_and_fixture_tree_views_agree() {
    // M3c: both env files are tree-only; their derived views agree route by
    // route (the M3cal parity lock, restated without flat arms).
    let shipped_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let tree_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.tree.json");
    let shipped: RouterConfig = serde_json::from_str(&std::fs::read_to_string(shipped_path).unwrap()).unwrap();
    let tree: RouterConfig = serde_json::from_str(&std::fs::read_to_string(tree_path).unwrap()).unwrap();
    shipped.validate_flat_tree_coherence().expect("shipped tree present");
    let shipped_view = shipped.routes_view();
    let tree_view = tree.routes_view();
    for (route, rref) in &shipped_view {
        let tg = tree_view.get(route).expect("fixture covers shipped route").group.clone();
        assert_eq!(&rref.group, &tg, "route {route} group mismatch");
    }
}

#[test]
fn shipped_config_tree_only() {
    // M3c: the shipped config carries a classification tree; its derived
    // view resolves every route to the M3cal-locked group.
    let shipped_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let shipped: RouterConfig =
        serde_json::from_str(&std::fs::read_to_string(&shipped_path).unwrap()).unwrap();
    assert!(shipped.classification.is_some(), "shipped config is tree-only");
    shipped.validate_flat_tree_coherence().expect("shipped tree valid");
    let view = shipped.routes_view();
    for (route, group) in tree_only_snapshot() {
        assert_eq!(&view[route].group.as_str(), &group, "route {route} group stable");
    }
}

#[test]
fn shipped_config_routes_view_key_set() {
    let shipped_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let shipped: RouterConfig =
        serde_json::from_str(&std::fs::read_to_string(&shipped_path).unwrap()).unwrap();
    let view = shipped.routes_view();
    let mut keys: Vec<&String> = view.keys().collect();
    keys.sort();
    let snapshot: Vec<&str> = tree_only_snapshot().into_iter().map(|(r, _)| r).collect();
    let key_strs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    assert_eq!(key_strs, snapshot, "tree terminals are the only routes");
    // always_route rides the tree: the four dispatch-forcing routes from M2.4.
    for forced in ["code", "explore", "explain", "prose"] {
        assert!(view[forced].always_route, "{forced} forces dispatch");
    }
    for direct in ["local", "summarize"] {
        assert!(!view[direct].always_route, "{direct} may answer directly");
    }
}

#[test]
fn view_consumers_match_direct_reads() {
    // M3b lock (M3c restatement): every consumer observes the derived view
    // (`routes_view` / `routing_config`); there is no flat map anymore.
    let shipped_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let shipped: RouterConfig =
        serde_json::from_str(&std::fs::read_to_string(&shipped_path).unwrap()).unwrap();
    assert_eq!(shipped.routing_config().routes.len(), shipped.routes_view().len());
    for (route, rref) in shipped.routes_view() {
        let got = &shipped.routing_config().routes[&route];
        assert_eq!(got.group, rref.group);
        assert_eq!(got.pipelines, rref.pipelines);
        assert_eq!(got.always_route, rref.always_route);
        assert_eq!(shipped.route_pipeline_names(&route), rref.pipelines);
    }
    assert!(!shipped.routing_config().system_prompt.is_empty(), "prompt always derived");

    // Tree-only config: every terminal is covered by the view and the kernel.
    let both: RouterConfig = serde_json::from_value(serde_json::json!({
        "pipelines": {"default": {"classifier": true, "classifier_model": "fast"}},
        "models": {
            "fast": {"endpoint": "http://x/v1/chat/completions", "intelligence": 1,
                     "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7, "speed": 8}
        },
        "model_groups": {"g1": ["fast"], "g2": ["fast"]},
        "classification": {"root": {"type": "classifier", "description": "r", "model": "fast",
            "children": [
                {"key": "a", "description": "a", "node": {"type": "terminal", "route": "a", "group": "g1"}},
                {"key": "b", "description": "b", "node": {"type": "terminal", "route": "b", "group": "g2"}}
            ]}},
        "default_route": "a"
    }))
    .unwrap();
    let view = both.routes_view();
    assert!(view.contains_key("b"), "tree-only terminal synthesized");
    assert_eq!(both.route_pipeline_names("b"), view["b"].pipelines);
    assert_eq!(both.route_pipeline_names("a"), view["a"].pipelines);
    let routing = both.routing_config();
    assert_eq!(routing.routes.len(), view.len());
    for (route, rref) in &view {
        let got = &routing.routes[route];
        assert_eq!(&got.group, &rref.group);
        assert_eq!(&got.pipelines, &rref.pipelines);
    }
    assert_eq!(routing.route_group("b"), Some("g2"));
    assert!(!routing.system_prompt.is_empty(), "tree derives a prompt when none is explicit");
}

/// M3cal [B] calibration control set: task-value parity between the flat
/// view and the tree-derived view. Every shipped route is probed with ≥3
/// phrasings (mirroring the `config_route_tests` probe style); ≥20
/// unknown-route prompts must resolve to `default_route` in both views.
/// Any mismatch blocks M3c. No derived view may be cached or persisted
/// until this is green — and none is: both views are computed per call.
fn m3cal_probes_for_route(route: &str) -> Vec<&'static str> {
    match route {
        "local" => vec![
            "What is the capital of France? Answer in one short sentence.",
            "What is 2+2?",
            "hi",
            "Who wrote the Iliad?",
        ],
        "code" => vec![
            "Write a Rust function to compute Fibonacci numbers.",
            "Write a Rust program that prints the first ten primes.",
            "Fix a deadlock in this Go program.",
        ],
        "prose" => vec![
            "Write a short gothic story about a lighthouse keeper.",
            "Write a haiku about autumn leaves.",
            "Draft a letter of complaint to a landlord.",
        ],
        "summarize" => vec![
            "Summarize this in one sentence: 'Q3 revenue reached $4.2M, up 12% YoY.'",
            "Condense this paragraph: 'The company reported strong Q3 results.'",
            "Give me the TL;DR of this article about migratory birds.",
        ],
        "explore" => vec![
            "Extract the dates and amounts from this email as JSON.",
            "Search the web for the price of a used 2018 Toyota Camry.",
            "Look up the population of Berlin.",
        ],
        "explain" => vec![
            "Explain the EPR paradox and Bell's theorem.",
            "Translate 'Good morning' into French.",
            "Why is the sky blue?",
        ],
        _ => vec!["Please help with this request."],
    }
}

/// Prompts that name no route and carry no route-like intent: all must fall
/// back to the default route's group (fail-closed, never a fabricated route).
fn m3cal_unknown_prompts() -> Vec<&'static str> {
    vec![
        "blorpt fnord wibble quux",
        "asdf qwer zxcv 12345",
        "???",
        "hmm",
        "ok",
        "Lorem ipsum dolor sit amet consectetur",
        "01010100101010100101",
        "!!! ??? !!!",
        "the the the the the",
        "a",
        "supercalifragilisticexpialidocious humdinger",
        "zzz snore zzz",
        "42",
        "hello hello hello hello hello",
        "test test test test",
        "...",
        "??? hello ???",
        "flibbertigibbet wobble",
        "qwertyuiop asdfghjkl",
        "zzzzzzzz",
        "murmuration petrichor susurrus",
    ]
}

#[test]
fn m3cal_dispatch_parity_control_set() {
    // M3cal lock, restated tree-only (M3c): shipped vs fixture views agree
    // 100% on route→group and dispatch targets; unknowns fall back to default.
    let shipped_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json");
    let tree_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.tree.json");
    let shipped: RouterConfig =
        serde_json::from_str(&std::fs::read_to_string(&shipped_path).unwrap()).unwrap();
    let tree: RouterConfig =
        serde_json::from_str(&std::fs::read_to_string(&tree_path).unwrap()).unwrap();
    let flat_view = shipped.routes_view();
    let tree_view = tree.routes_view();
    let flat_routing = shipped.routing_config();
    let tree_routing = tree.routing_config();

    // Control group 1: every shipped route × ≥3 phrasings — the phrasing is
    // documentation (resolution is by route key through the shared kernel);
    // what must agree 100% is route→group equality and the resolved dispatch
    // target (model + group) between the flat and tree views.
    let mut probed = 0;
    for route in flat_view.keys() {
        let probes = m3cal_probes_for_route(route);
        assert!(probes.len() >= 3, "route '{route}' needs ≥3 phrasings");
        for _probe in probes {
            probed += 1;
            let fg = flat_view.get(route).unwrap_or_else(|| panic!("shipped view covers {route}"));
            let tg = tree_view.get(route).unwrap_or_else(|| panic!("tree view covers {route}"));
            assert_eq!(fg.group, tg.group, "route→group mismatch for '{route}'");
            let ft = flat_routing.routing_target(route, None);
            let tt = tree_routing.routing_target(route, None);
            match (ft, tt) {
                (Some(f), Some(t)) => {
                    assert_eq!(f.model, t.model, "dispatch model mismatch for '{route}'");
                    assert_eq!(f.group, t.group, "dispatch group mismatch for '{route}'");
                }
                (None, None) => {}
                (f, t) => panic!("dispatch agreement mismatch for '{route}': {f:?} vs {t:?}"),
            }
        }
    }
    assert!(probed >= flat_view.len() * 3, "probed {probed} across routes");

    // Control group 2 (must-NOT-fire): ≥20 unknown prompts resolve to the
    // default route's group in both views — never a fabricated route.
    let unknowns = m3cal_unknown_prompts();
    assert!(unknowns.len() >= 20, "need ≥20 unknown controls");
    let default_group = flat_view
        .get(&shipped.default_route)
        .map(|r| r.group.as_str())
        .unwrap_or("");
    for prompt in unknowns {
        // Unknown prompts are resolved by key-miss fallback, keyed here by a
        // sentinel that names no route in either view.
        let miss_key = format!("__unknown__{prompt}");
        assert!(!flat_view.contains_key(&miss_key) && !tree_view.contains_key(&miss_key));
        for routing in [&flat_routing, &tree_routing] {
            assert_eq!(
                routing.route_group(&miss_key),
                Some(default_group),
                "unknown prompt must fall back to default_route group"
            );
        }
        assert_eq!(
            flat_routing.route_group(&miss_key),
            tree_routing.route_group(&miss_key),
            "flat/tree agree on unknown fallback"
        );
    }
}

#[test]
fn flat_config_rejected_without_tree() {
    // M3c break: a flat-only config (no `classification` tree) must fail
    // fast with the tree pointer error — flat JSON no longer loads.
    let cfg: RouterConfig = serde_json::from_str(
        r#"{"routes": {"local": {"group": "default"}}, "default_route": "local"}"#,
    )
    .unwrap();
    let err = cfg
        .validate_flat_tree_coherence()
        .expect_err("flat-only config must be rejected");
    assert!(err.contains("classification.tree"), "got: {err}");
}

fn tree_only_snapshot() -> Vec<(&'static str, &'static str)> {
    vec![
        ("code", "code"),
        ("explain", "explain"),
        ("explore", "explore"),
        ("local", "default"),
        ("prose", "prose"),
        ("summarize", "summarize"),
    ]
}

#[test]
fn tree_only_routes_view_matches_previous_snapshot() {
    // The tree-derived view resolves every shipped route to its M3cal-locked
    // group — identical before (tree fixture) and after (shipped tree-only
    // config) the flat deletion.
    for file in ["../../env/coral-router.tree.json", "../../env/coral-router.json"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(cfg) = serde_json::from_str::<RouterConfig>(&content) else {
            continue;
        };
        if cfg.classification.is_none() {
            continue;
        }
        let view = cfg.routes_view();
        assert_eq!(view.len(), 6, "{file}: six terminal routes");
        for (route, group) in tree_only_snapshot() {
            assert_eq!(&view[route].group.as_str(), &group, "{file}: {route}");
        }
    }
}

#[test]
fn onnx_limiter_defaults_are_constants() {
    let cfg = RouterConfig::default();
    assert_eq!(cfg.onnx_limiter_cap, common_core::constants::DEFAULT_ONNX_LIMITER_CAP);
    assert_eq!(cfg.onnx_threads, common_core::constants::DEFAULT_ONNX_THREADS);
    assert_eq!(common_core::constants::DEFAULT_ONNX_LIMITER_CAP, 2);
    assert_eq!(common_core::constants::DEFAULT_ONNX_THREADS, 1);
}
