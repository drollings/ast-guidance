//! Aggregation-envelope goldens for the shared residency engine.
//!
//! Records the exact `GET /instances` aggregate envelope and the
//! `GET /v1/models` list over a fixed hermetic harness (one stub fork
//! reporting two instances + one stub onnx `llm` role). The llama rows flow
//! through the shared engine's `LlmWeights` surface; any drift in the
//! envelopes fails here.
//!
//! `list_models` carries a wall-clock `created` per entry; it is normalized
//! to `0` before comparison so the golden is deterministic.
//!
//! To re-record (only when the envelope shape intentionally changes):
//! `UPDATE_GOLDENS=1 cargo test -p fluent-router --lib residency_engine`.

use super::stub::StubServer;
use super::*;

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn check_golden(name: &str, observed: &Value) {
    let rendered = serde_json::to_string_pretty(observed).expect("render golden") + "\n";
    let path = golden_path(name);
    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::write(&path, &rendered).expect("write golden");
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing golden file {} — re-record with UPDATE_GOLDENS=1", path.display()));
    assert_eq!(rendered, expected, "golden drift: {name}");
}

fn fork_info(id: &str, group: &str, pinned: bool, is_default: bool, last_used: i64) -> Value {
    serde_json::json!({
        "id": id,
        "aliases": [],
        "group": group,
        "n_ctx": 16384,
        "parallel": 1,
        "pinned": pinned,
        "is_default": is_default,
        "resume": false,
        "state": "loaded",
        "model_bytes": 100_000_000,
        "context_bytes": 262144,
        "compute_bytes": 1048576,
        "total_bytes": 1310720,
        "vram_bytes": 1000,
        "last_used": last_used,
    })
}

/// A stub-fork responder: method + path + body → status + JSON body.
type ForkResponder = dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync;

fn fork_handler(
    method: &str,
    path: &str,
    _body: &str,
) -> (u16, String) {
    if method == "GET" && path == "/instances" {
        (
            200,
            serde_json::json!({
                "instances": [
                    fork_info("base:ledger", "ledger", false, false, 5),
                    fork_info("base:swarm", "swarm", true, true, 9),
                ],
                "snapshots": [],
                "total": { "model": 100_000_000, "context": 524288, "compute": 2097152, "total": 102621440 },
            })
            .to_string(),
        )
    } else {
        (200, "{}".into())
    }
}

fn llama_pool_with_fork() -> (StubServer, InstancePool) {
    let handler: Arc<ForkResponder> = Arc::new(fork_handler);
    let stub = StubServer::start(handler);
    let manager = Arc::new(InstanceManager::new(
        "base",
        InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
        Vec::new(),
        crate::config::SidecarConfig::default(),
    ));
    // Router-side resume overlay on the unpinned instance (pinned in goldens).
    manager.set_resume("ledger", true);
    let mut managers = HashMap::new();
    managers.insert("base".into(), manager);
    let pool = InstancePool::from_managers(managers, None);
    (stub, pool)
}

/// `created` is wall-clock; zero it so the golden is deterministic.
fn normalize_models(models: &mut [Value]) {
    for entry in models.iter_mut() {
        if let Value::Object(obj) = entry {
            obj.insert("created".into(), Value::Number(0.into()));
        }
        normalize_onnx_last_used(entry);
    }
}

/// The onnx `default` row's `last_used` is the registry's wall-clock
/// (milliseconds); zero it so the golden is deterministic. Llama rows keep
/// their stub-fixed `last_used` (the ordering signal is asserted
/// separately by the eviction-order tests).
fn normalize_onnx_last_used(entry: &mut Value) {
    let is_onnx = entry["runtime"].as_str() == Some("onnx");
    if is_onnx {
        if let Value::Object(obj) = entry {
            obj.insert("last_used".into(), Value::Number(0.into()));
        }
    }
}

#[cfg(feature = "onnx")]
fn normalize_aggregate(agg: &mut Value) {
    if let Some(rows) = agg["instances"].as_array_mut() {
        for row in rows.iter_mut() {
            normalize_onnx_last_used(row);
        }
    }
}

#[tokio::test]
async fn pool_aggregate_envelope_golden() {
    let (_stub, pool) = llama_pool_with_fork();
    let agg = pool.aggregate(None).await.expect("aggregate");
    check_golden("residency_engine_pool_aggregate.json", &agg);

    let mut models = pool.list_models().await;
    normalize_models(&mut models);
    check_golden(
        "residency_engine_pool_models.json",
        &Value::Array(models),
    );
}

#[cfg(feature = "onnx")]
mod onnx_golden {
    use super::*;

    use crate::config::RouterConfig;
    use fluent_llm::onnx_config::{
        OnnxConfig, OnnxFleetConfig, OnnxRoleConfig, OnnxTask, ResidencyPolicy,
    };
    use fluent_llm::onnx_session::OrtSessionRegistry;

    use fluent_llm::testutil::StubSessionLoader;

    fn onnx_registry() -> Arc<OrtSessionRegistry> {
        let registry = Arc::new(OrtSessionRegistry::new(Arc::new(StubSessionLoader)));
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

    #[tokio::test]
    async fn fleet_aggregate_envelope_golden() {
        let (_stub, pool) = llama_pool_with_fork();
        let fleet = LlmFleet::build(
            pool.clone(),
            None,
            Some(onnx_registry()),
            &config_with_onnx_llm(),
        );

        let mut combined = fleet.aggregate(None).await.expect("fleet aggregate");
        normalize_aggregate(&mut combined);
        // Llama rows stay byte-identical to the llama-only pool's envelope.
        let llama_only = pool.aggregate(None).await.expect("llama aggregate");
        let llama_rows: Vec<&Value> = combined["instances"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i["runtime"].as_str() != Some("onnx"))
            .collect();
        assert_eq!(
            serde_json::to_value(&llama_rows).unwrap(),
            llama_only["instances"],
            "llama rows byte-identical inside the fleet envelope"
        );
        check_golden("residency_engine_fleet_aggregate.json", &combined);

        let mut models = fleet.list_models().await;
        normalize_models(&mut models);
        check_golden(
            "residency_engine_fleet_models.json",
            &Value::Array(models),
        );
    }
}
