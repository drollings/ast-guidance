use super::*;
use std::collections::HashMap;

use fluent_onnx::{
    OnnxConfig, OnnxFleetConfig, OnnxRoleConfig, OnnxTask, OrtError, OrtSessionRegistry,
    ResidencyPolicy, SessionHandle, SessionLoader,
};

use super::super::stub::StubServer;
use crate::config::SidecarConfig;

/// Stub onnx loader (no ort, no model).
#[derive(Default)]
struct StubLoader;

impl SessionLoader for StubLoader {
    fn load(
        &self,
        _config: &OnnxConfig,
        _model_key: &str,
    ) -> Result<SessionHandle, OrtError> {
        Ok(SessionHandle::new("stub"))
    }
}

fn sidecar_policy() -> SidecarConfig {
    SidecarConfig::default()
}

fn llama_info(id: &str, group: &str, pinned: bool, last_used: i64) -> InstanceInfo {
    InstanceInfo {
        id: id.into(),
        aliases: vec![],
        group: group.into(),
        n_ctx: 16384,
        parallel: 1,
        pinned,
        is_default: false,
        resume: false,
        state: "loaded".into(),
        model_bytes: 100_000_000,
        context_bytes: 262144,
        compute_bytes: 1048576,
        total_bytes: 1310720,
        vram_bytes: 1000,
        last_used,
    }
}

/// An `Always`-resident, pinned onnx `llm` role backed by a stub loader
/// (the reference config's shape, minus the real model file).
fn onnx_registry() -> Arc<OrtSessionRegistry> {
    let registry = Arc::new(OrtSessionRegistry::new(Arc::new(StubLoader::default())));
    registry
        .register_with_lifecycle(
            "onnx/llm",
            OnnxConfig::new()
                .model_path("/models/llm.onnx")
                .tokenizer_path("/models/llm/tokenizer.json")
                .task(OnnxTask::CausalLm)
                .resident(true)
                .maybe_resident_bytes(Some(4_000_000_000))
                .build(),
            ResidencyPolicy::Always,
            true,
            Some(30),
        )
        .expect("register llm");
    registry
}

/// A `RouterConfig` declaring the same onnx `llm` role (the `instances`
/// block absent — the M3 default, byte-identical behavior).
fn config_with_onnx_llm() -> RouterConfig {
    let mut config = RouterConfig::default();
    config.onnx = Some(OnnxFleetConfig {
        llm: Some(OnnxRoleConfig {
            pinned: true,
            no_sleep: false,
            sleep_idle_seconds: Some(30),
            total_timeout_ms: 0,
            idle_timeout_ms: 0,
            params: None,
            instances: None,
            model: OnnxConfig::new()
                .model_path("/models/llm.onnx")
                .tokenizer_path("/models/llm/tokenizer.json")
                .task(OnnxTask::CausalLm)
                .resident(true)
                .build(),
        }),
        ..Default::default()
    });
    config
}

fn llama_pool_with(stub: &StubServer) -> InstancePool {
    let manager = Arc::new(InstanceManager::new(
        "base",
        InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
        Vec::new(),
        sidecar_policy(),
    ));
    let mut managers = HashMap::new();
    managers.insert("base".into(), manager);
    InstancePool::from_managers(managers, None)
}

/// The reference-config facade: a llama manager whose fork reports one
/// unpinned instance, and an Always-pinned onnx `llm` role. The llama
/// envelope must be byte-identical to the llama-only pool's; the onnx role
/// contributes exactly one synthesized `onnx/llm:default` row.
#[tokio::test]
async fn fleet_aggregate_adds_onnx_synthesized_row_and_keeps_llama_identical() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "GET" && path == "/instances" {
                (
                    200,
                    serde_json::json!({
                        "instances": [ llama_info("base:ledger", "ledger", false, 5) ],
                        "snapshots": [],
                        "total": { "model": 100_000_000, "context": 262144, "compute": 1048576, "total": 102383720 },
                    })
                    .to_string(),
                )
            } else {
                (200, "{}".into())
            }
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);

    let onnx = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(onnx), &config);
    assert!(!fleet.is_empty(), "onnx weights registered -> fleet non-empty");

    let llama_only = pool.aggregate(None).await.expect("llama aggregate");
    let combined = fleet.aggregate(None).await.expect("fleet aggregate");

    // The llama envelope is byte-identical: the fleet's llama rows equal
    // the llama-only pool's rows, and the totals carry the onnx additions
    // on top of the llama totals.
    let llama_rows: Vec<&Value> = combined["instances"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["runtime"].as_str() != Some("onnx"))
        .collect();
    assert_eq!(
        serde_json::to_value(&llama_rows).unwrap(),
        llama_only["instances"],
        "llama rows byte-identical"
    );
    assert_eq!(combined["snapshots"], llama_only["snapshots"]);
    assert_eq!(
        combined["total"]["context"],
        llama_only["total"]["context"].as_u64().unwrap()
            + combined["instances"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|i| i["runtime"].as_str() == Some("onnx"))
                .map(|i| i["context_bytes"].as_u64().unwrap_or(0))
                .sum::<u64>(),
    );

    // The onnx role contributes exactly one synthesized default row.
    let onnx_rows: Vec<&Value> = combined["instances"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["runtime"].as_str() == Some("onnx"))
        .collect();
    assert_eq!(onnx_rows.len(), 1, "exactly one onnx row");
    let row = onnx_rows[0];
    assert_eq!(row["id"], "onnx/llm:default");
    assert_eq!(row["state"], "loaded");
    assert_eq!(row["model_bytes"], 4_000_000_000u64, "resident footprint");
    assert_eq!(row["vram_bytes"], 0u64, "RAM-resident context owns no VRAM");
    assert_eq!(row["pinned"], true, "pinned role row");
    assert_eq!(row["runtime"], "onnx");
    let aliases = row["aliases"].as_array().unwrap();
    assert!(aliases.iter().any(|a| a == "onnx/llm"), "bare model alias");
    assert!(
        aliases.iter().any(|a| a == "onnx/llm:latest"),
        "latest alias on the default row"
    );

    // `total.model` = llama weights + onnx resident footprint.
    assert_eq!(
        combined["total"]["model"],
        llama_only["total"]["model"].as_u64().unwrap() + 4_000_000_000u64
    );
}

#[tokio::test]
async fn fleet_list_models_appends_onnx_rows_with_aliases() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "GET" && path == "/instances" {
                (
                    200,
                    serde_json::json!({
                        "instances": [ llama_info("base:ledger", "ledger", false, 5) ],
                        "snapshots": [],
                        "total": { "model": 100_000_000, "context": 262144, "compute": 1048576, "total": 102383720 },
                    })
                    .to_string(),
                )
            } else {
                (200, "{}".into())
            }
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);
    let onnx = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(onnx), &config);

    let models = fleet.list_models().await;
    // llama entry + the onnx entry.
    let llama_entries: Vec<&Value> = models
        .iter()
        .filter(|m| m["id"].as_str().unwrap_or_default().starts_with("base:"))
        .collect();
    assert_eq!(llama_entries.len(), 1, "llama rows preserved");
    let onnx_entry = models
        .iter()
        .find(|m| m["id"].as_str() == Some("onnx/llm:default"))
        .expect("onnx entry");
    assert_eq!(onnx_entry["state"], "loaded");
    assert_eq!(onnx_entry["runtime"], "onnx");
    assert_eq!(onnx_entry["pinned"], true);
    assert!(onnx_entry["aliases"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a == "onnx/llm"));
}

#[tokio::test]
async fn fleet_scoped_aggregate_returns_only_matching_rows() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |_method, _path, _body| {
            (
                200,
                serde_json::json!({
                    "instances": [],
                    "snapshots": [],
                    "total": { "model": 0, "context": 0, "compute": 0, "total": 0 },
                })
                .to_string(),
            )
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);
    let onnx = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(onnx), &config);

    let scoped = fleet
        .aggregate(Some("onnx/llm"))
        .await
        .expect("scoped aggregate");
    let rows = scoped["instances"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["id"], "onnx/llm:default");

    let llama_scoped = fleet
        .aggregate(Some("base"))
        .await
        .expect("scoped aggregate");
    assert_eq!(
        llama_scoped["instances"].as_array().unwrap().len(),
        0,
        "empty llama fork -> no rows, onnx filtered out"
    );
}

#[tokio::test]
async fn fleet_unload_releases_unloadable_onnx_role_and_refuses_always() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |_method, _path, _body| {
            (
                200,
                serde_json::json!({
                    "instances": [],
                    "snapshots": [],
                    "total": { "model": 0, "context": 0, "compute": 0, "total": 0 },
                })
                .to_string(),
            )
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);

    // An `Unloadable`, unpinned onnx role unloads through the trait surface.
    // Registered under the config's role key (`onnx/llm` — the fleet's
    // generative role), exactly as `build_onnx_registry` would.
    let lazy_registry = Arc::new(OrtSessionRegistry::new(Arc::new(StubLoader::default())));
    lazy_registry
        .register_with_lifecycle(
            "onnx/llm",
            OnnxConfig::new()
                .model_path("/models/lazy.onnx")
                .tokenizer_path("/models/lazy/tokenizer.json")
                .task(OnnxTask::CausalLm)
                .resident(false)
                .build(),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
            false,
            Some(1),
        )
        .expect("register lazy");
    lazy_registry.ensure_loaded("onnx/llm").expect("load");

    let mut config = RouterConfig::default();
    config.onnx = Some(OnnxFleetConfig {
        llm: Some(OnnxRoleConfig {
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: Some(1),
            total_timeout_ms: 0,
            idle_timeout_ms: 0,
            params: None,
            instances: None,
            model: OnnxConfig::new()
                .model_path("/models/lazy.onnx")
                .tokenizer_path("/models/lazy/tokenizer.json")
                .task(OnnxTask::CausalLm)
                .resident(false)
                .build(),
        }),
        ..Default::default()
    });
    let fleet = LlmFleet::build(pool.clone(), None, Some(lazy_registry.clone()), &config);
    assert!(fleet.is_known_model("onnx/llm"));
    fleet.unload("onnx/llm").await.expect("unload");
    assert!(!lazy_registry
        .residency_report()
        .iter()
        .find(|r| r.key == "onnx/llm")
        .unwrap()
        .loaded, "lazy role released");

    // An `Always` (resident) onnx role refuses unload (UnloadRefused).
    let always = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(always), &config);
    let err = fleet.unload("onnx/llm").await.expect_err("Always refuses");
    assert!(matches!(err, LlmRuntimeError::UnloadRefused(_)), "got {err:?}");
}

/// ROADMAP M7 §3: `LlmFleet::resize_context` routes an onnx context through
/// `LlmContext::resize`; unknown models/contexts are loud errors, and the
/// fleet's onnx id grammar resolves `<onnx_key>:<context>`.
#[tokio::test]
async fn fleet_resize_context_and_onnx_id_grammar() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |_method, _path, _body| {
            (
                200,
                serde_json::json!({
                    "instances": [],
                    "snapshots": [],
                    "total": { "model": 0, "context": 0, "compute": 0, "total": 0 },
                })
                .to_string(),
            )
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);
    let onnx = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(onnx), &config);

    // The onnx id grammar `<onnx_key>:<context>` resolves through the fleet.
    let resolved = fleet.resolve_instance_id("onnx/llm:default");
    assert_eq!(
        resolved.as_ref().map(|(m, n)| (m.as_str(), n.as_str())),
        Some(("onnx/llm", "default"))
    );
    // A llama key is not the fleet's (the llama pool owns it), and a
    // malformed id is rejected.
    assert!(fleet.resolve_instance_id("base:ledger").is_none());
    assert!(fleet.resolve_instance_id("onnx/llm:a:b").is_none());

    // Resize on an onnx context: the context is not materialized (no pool
    // built — the stub registry never loads a session), so it is a loud
    // `NotLoaded` error (the "unknown context" shape), not a silent no-op.
    let err = fleet
        .resize_context("onnx/llm", "default", 32768)
        .await
        .expect_err("unloaded onnx context is loud");
    assert!(
        matches!(err, LlmRuntimeError::NotLoaded(_)),
        "got {err:?}"
    );

    // Unknown models are loud too.
    let err = fleet
        .resize_context("nope", "x", 1)
        .await
        .expect_err("unknown model is loud");
    assert!(matches!(err, LlmRuntimeError::NotLoaded(_)));
}

/// The `ps` render: the llama block is byte-identical (the pre-M4 layout,
/// `ctx-mem` = VRAM), and an onnx key renders a `(onnx)` block with the
/// RAM memory column (`ram-mem` from `total_bytes`).
#[test]
fn ps_render_keeps_llama_block_and_adds_onnx_block() {
    let dir = tempfile::tempdir().unwrap();
    let llama_row = serde_json::json!({
        "id": "base:ledger",
        "n_ctx": 16384,
        "parallel": 1,
        "pinned": false,
        "resume": false,
        "state": "loaded",
        "vram_bytes": 1000,
        "model_bytes": 100_000_000,
    });
    let llama_block =
        crate::cli::commands::server::render_weight_block(None, dir.path(), "base", &[llama_row]);
    assert!(llama_block.contains("ctx-mem"), "llama uses the VRAM column");
    assert!(llama_block.contains("weights"), "weights row");
    assert!(llama_block.contains("100 MB"), "weights size rendered");
    assert!(!llama_block.contains("onnx"), "no onnx marker in the llama block");
    assert!(llama_block.contains("base"));

    let onnx_row = serde_json::json!({
        "id": "onnx/llm:default",
        "n_ctx": 16384,
        "parallel": 1,
        "pinned": true,
        "resume": false,
        "state": "loaded",
        "runtime": "onnx",
        "total_bytes": 0,
        "vram_bytes": 0,
        "model_bytes": 4_000_000_000u64,
    });
    let onnx_block = crate::cli::commands::server::render_weight_block(
        None,
        dir.path(),
        "onnx/llm",
        &[onnx_row],
    );
    assert!(onnx_block.contains("(onnx)"), "runtime marker");
    assert!(onnx_block.contains("ram-mem"), "RAM memory column");
    assert!(!onnx_block.contains("ctx-mem"), "no VRAM column for onnx");
    assert!(onnx_block.contains("onnx/llm"), "display line names the model");
    assert!(onnx_block.contains("default"), "the default context row");
}

#[tokio::test]
async fn fleet_knows_llama_and_onnx_keys() {
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |_method, _path, _body| {
            (
                200,
                serde_json::json!({
                    "instances": [],
                    "snapshots": [],
                    "total": { "model": 0, "context": 0, "compute": 0, "total": 0 },
                })
                .to_string(),
            )
        },
    );
    let stub = StubServer::start(handler);
    let pool = llama_pool_with(&stub);
    let onnx = onnx_registry();
    let config = config_with_onnx_llm();
    let fleet = LlmFleet::build(pool.clone(), None, Some(onnx), &config);
    assert!(fleet.is_known_model("base"), "llama manager key known");
    assert!(fleet.is_known_model("onnx/llm"), "onnx role key known");
    assert!(!fleet.is_known_model("nope"));
}
