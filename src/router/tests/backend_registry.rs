//! Inference-registry routing goldens.
//!
//! The frozen corpus below records `local_backend` / `local_backend_for_instance`
//! outputs through the registry path: `Some`/`None` per key plus the stub
//! marker for onnx keys. Hermetic: stub backends only, `LlmClient`s are built
//! but never called.

use std::sync::Arc;

use crate::config::RouterConfig;
use crate::test_stubs::StubInferenceBackend;

const ONNX_LLM_KEY: &str = "onnx/llm";

/// Fixed config corpus: one llama model with a pinned pool + instances, one
/// generative onnx role, and the registry installed (as boot installs it).
fn golden_corpus_config() -> RouterConfig {
    let mut config: RouterConfig = serde_json::from_value(serde_json::json!({
        "models": {
            "swarm": {
                "endpoint": "http://x/v1/chat/completions",
                "name": "swarm", "intelligence": 2,
                "cost_input": 1.0, "cost_output": 6.0, "cost_cached_read": 0.4, "speed": 8,
                "instances": {
                    "swarm": { "count": 3, "group": "swarm", "num_ctx": 16384 },
                    "ledger": { "num_ctx": 131072, "pinned": true, "default": true },
                    "scratch": { "num_ctx": 131072, "sleep_idle_seconds": 30 }
                }
            }
        },
        "onnx": {
            "llm": {
                "model_path": "/models/llm.onnx",
                "tokenizer_path": "/models/llm/tokenizer.json",
                "resident": false,
                "quantization": "q4"
            }
        }
    }))
    .expect("valid config");
    let pool = crate::instances::InstancePool::from_managers(std::collections::HashMap::new(), None);
    let llama = crate::instances::traits::LlamaBackend::new(
        pool,
        config.models.clone(),
        config.roles.clone(),
        config.default_params.instances.clone(),
        config.onnx_role_keys(),
        config.sidecar.clone(),
    );
    let mut registry = fluent_llm::backend::InferenceRegistry::new();
    registry.register(Arc::new(llama));
    registry.register(Arc::new(StubInferenceBackend::named(
        "onnx",
        ONNX_LLM_KEY,
        "onnx-llm",
        "onnx-instance",
    )));
    config.set_inference_registry(Arc::new(std::sync::RwLock::new(registry)));
    config
}

/// Probe one key: (`is_some`, marker when the backend is a stub).
fn probe(config: &RouterConfig, key: &str, instance: Option<&str>) -> (bool, Option<String>) {
    let backend = match instance {
        Some(name) => config.local_backend_for_instance(key, name),
        None => config.local_backend(key),
    };
    match backend {
        None => (false, None),
        Some(b) => {
            // Only the stub is safe to call; an `LlmClient` would dial the
            // network, so its marker stays `None` ("some, not a stub").
            let marker = if key == ONNX_LLM_KEY {
                b.chat_complete(&[]).ok()
            } else {
                None
            };
            (true, marker)
        }
    }
}



#[test]
fn local_backend_golden_corpus() {
    let config = golden_corpus_config();
    let cases: &[(&str, Option<&str>)] = &[
        ("swarm", None),
        ("swarm:latest", None),
        ("swarm:ledger", None),
        ("missing", None),
        (ONNX_LLM_KEY, None),
        ("swarm", Some("ledger")),
        ("swarm", Some("scratch")),
        ("swarm", Some("ghost")),
        ("missing", Some("scratch")),
        (ONNX_LLM_KEY, Some("swarm")),
    ];
    let observed: Vec<serde_json::Value> = cases
        .iter()
        .map(|(key, instance)| {
            let (some, marker) = probe(&config, key, *instance);
            serde_json::json!({"key": key, "instance": instance, "some": some, "marker": marker})
        })
        .collect();
    let expected = serde_json::json!([
        {"key": "swarm", "instance": null, "some": true, "marker": null},
        {"key": "swarm:latest", "instance": null, "some": true, "marker": null},
        {"key": "swarm:ledger", "instance": null, "some": true, "marker": null},
        {"key": "missing", "instance": null, "some": false, "marker": null},
        {"key": "onnx/llm", "instance": null, "some": true, "marker": "onnx-llm"},
        {"key": "swarm", "instance": "ledger", "some": true, "marker": null},
        {"key": "swarm", "instance": "scratch", "some": true, "marker": null},
        {"key": "swarm", "instance": "ghost", "some": false, "marker": null},
        {"key": "missing", "instance": "scratch", "some": false, "marker": null},
        {"key": "onnx/llm", "instance": "swarm", "some": true, "marker": "onnx-instance"},
    ]);
    assert_eq!(
        serde_json::Value::Array(observed),
        expected,
        "local_backend outputs must match the frozen golden corpus"
    );
}

#[test]
fn registry_prefers_llama_for_llama_keys_and_onnx_for_onnx_keys() {
    use fluent_llm::backend::{InferenceBackend, InferenceRegistry};
    let pool =
        crate::instances::InstancePool::from_managers(std::collections::HashMap::new(), None);
    let corpus = golden_corpus_config();
    let llama = crate::instances::traits::LlamaBackend::new(
        pool,
        corpus.models.clone(),
        corpus.roles.clone(),
        corpus.default_params.instances.clone(),
        corpus.onnx_role_keys(),
        corpus.sidecar.clone(),
    );
    assert_eq!(llama.backend_id(), "llama");
    assert!(llama.model_keys().contains(&"swarm".to_string()));
    assert!(
        !llama.model_keys().contains(&ONNX_LLM_KEY.to_string()),
        "llama yields onnx keys to the onnx adapter"
    );
    assert!(llama.chat_backend("swarm", None).is_some());
    assert!(llama.chat_backend(ONNX_LLM_KEY, None).is_none());
    assert!(llama.chat_backend("missing", None).is_none());
    assert_eq!(
        llama.capabilities(),
        fluent_llm::backend::BackendCaps::chat_with_contexts()
    );
    assert_eq!(
        llama.readiness("swarm"),
        fluent_llm::backend::Readiness::Unloaded,
        "empty pool: known key, no running server"
    );

    let mut registry = InferenceRegistry::new();
    registry.register(Arc::new(llama));
    registry.register(Arc::new(StubInferenceBackend::named(
        "onnx",
        ONNX_LLM_KEY,
        "onnx-llm",
        "onnx-instance",
    )));
    // Llama keys route to the llama adapter (built clients, never called).
    assert!(registry.route_chat("swarm", None).is_some());
    assert!(registry.route_chat("swarm", Some("ledger")).is_some());
    // Onnx keys route to the onnx adapter (stub marker proves the backend).
    let marker = registry
        .route_chat(ONNX_LLM_KEY, None)
        .expect("onnx key routes")
        .chat_complete(&[])
        .unwrap();
    assert_eq!(marker, "onnx-llm");
    // Unknown keys consult every backend and find nothing.
    assert!(registry.route_chat("missing", None).is_none());
    assert!(registry.route_embed("missing").is_none());
}

#[test]
fn llama_backend_yields_onnx_keys_on_collision() {
    // A `models` entry that collides with an onnx role key still resolves
    // `None` from the llama adapter: the onnx branch wins, as before.
    let mut corpus = golden_corpus_config();
    let entry = corpus.models.get("swarm").unwrap().clone();
    corpus.models.insert(ONNX_LLM_KEY.to_string(), entry);
    let pool =
        crate::instances::InstancePool::from_managers(std::collections::HashMap::new(), None);
    let llama = crate::instances::traits::LlamaBackend::new(
        pool,
        corpus.models.clone(),
        corpus.roles.clone(),
        corpus.default_params.instances.clone(),
        corpus.onnx_role_keys(),
        corpus.sidecar.clone(),
    );
    use fluent_llm::backend::InferenceBackend;
    assert!(llama.chat_backend(ONNX_LLM_KEY, None).is_none());
    assert!(llama.chat_backend(ONNX_LLM_KEY, Some("ledger")).is_none());
}

#[test]
fn onnx_backend_contract_without_a_session() {
    use fluent_llm::backend::{BackendCaps, InferenceBackend};
    use fluent_llm::testutil::StubSessionLoader as StubLoader;
    let registry: crate::ort::OrtRegistry = Arc::new(fluent_llm::onnx_session::OrtSessionRegistry::new(
        Arc::new(StubLoader),
    ));
    // No generative role → no backend (fail-open).
    assert!(
        crate::ort::OnnxBackend::from_config(&RouterConfig::default(), &registry).is_none()
    );
    let corpus = golden_corpus_config();
    let backend =
        crate::ort::OnnxBackend::from_config(&corpus, &registry).expect("llm role configured");
    assert_eq!(backend.backend_id(), "onnx");
    assert_eq!(backend.model_keys(), vec![ONNX_LLM_KEY.to_string()]);
    // Unregistered session: only non-llama keys are rejected here; the llm
    // key has no single-shot without a CausalLm session.
    assert!(backend.chat_backend("swarm", None).is_none());
    assert!(backend.weights("swarm").is_none());
    let caps: BackendCaps = backend.capabilities();
    assert!(caps.named_contexts && caps.kv_snapshot);
    assert_eq!(caps.grammar_constrained, cfg!(feature = "onnx"));
    assert_eq!(
        backend.readiness(ONNX_LLM_KEY),
        fluent_llm::backend::Readiness::Unloaded
    );
}
