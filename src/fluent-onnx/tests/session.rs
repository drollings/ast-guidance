#[cfg(feature = "onnx")]
use crate::session::ort_loader::is_gpu_provider;

    /// The GPU-provider classification the loader dispatches on — the
    /// `"gpu"`/`"migraphx"` config values must map to the AMD ROCm path
    /// (case-insensitive), and CPU must never be misclassified.
    /// Loader-gated: `ort_loader` (and `is_gpu_provider`) need `ort`.
    #[cfg(feature = "onnx")]
    #[test]
    fn gpu_provider_classification() {
        for gpu in ["gpu", "GPU", "migraphx", "MIGraphX", "Migraphx"] {
            assert!(
                is_gpu_provider(gpu),
                "{gpu:?} should classify as the AMD ROCm GPU provider"
            );
        }
        for cpu in ["cpu", "CPU", "Cpu"] {
            assert!(
                !is_gpu_provider(cpu),
                "{cpu:?} must not classify as the GPU provider"
            );
        }
        assert!(!is_gpu_provider("tensorrt"));
        assert!(!is_gpu_provider(""));
    }


use super::*;
use crate::config::{OnnxConfig, OnnxTask, Quant};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A stub session that records its construction.
#[derive(Debug)]
pub struct StubSession {
    pub marker: &'static str,
}

/// Counts load calls so tests can assert lazy-vs-boot loading.
#[derive(Default)]
pub struct CountingLoader {
    loads: AtomicUsize,
}

impl CountingLoader {
    pub fn load_count(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }
}

impl SessionLoader for CountingLoader {
    fn load(&self, config: &OnnxConfig, model_key: &str) -> Result<SessionHandle, OrtError> {
        let _ = (config, model_key);
        self.loads.fetch_add(1, Ordering::SeqCst);
        Ok(SessionHandle::new(StubSession {
            marker: "stub",
        }))
    }
}

fn test_config(task: OnnxTask) -> OnnxConfig {
    OnnxConfig::new()
        .model_path("/models/test.onnx")
        .tokenizer_path("/models/tokenizer.json")
        .task(task)
        .quantization(Quant::Q8)
        .build()
}

#[test]
fn always_policy_loads_at_register() {
    let loader = Arc::new(CountingLoader::default());
    let registry = OrtSessionRegistry::new(loader.clone());
    let config = test_config(OnnxTask::FillMask);
    registry
        .register("encoder", config.clone())
        .expect("register");
    assert_eq!(loader.load_count(), 1);
    assert!(registry.refuses_unload("encoder"));
    assert_eq!(registry.config("encoder"), Some(config));
}

#[test]
fn unloadable_policy_loads_lazily() {
    let loader = Arc::new(CountingLoader::default());
    let registry = OrtSessionRegistry::new(loader.clone());
    let config = test_config(OnnxTask::FillMask);
    registry
        .register_with_policy(
            "encoder",
            config.clone(),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
        )
        .expect("register");
    assert_eq!(loader.load_count(), 0);
    assert!(!registry.refuses_unload("encoder"));

    let handle = registry.ensure_loaded("encoder").expect("lazy load");
    assert!(handle.is_some());
    assert_eq!(loader.load_count(), 1);

    // Second ensure_loaded reuses the loaded session.
    registry.ensure_loaded("encoder").expect("idempotent");
    assert_eq!(loader.load_count(), 1);
}

#[test]
fn session_handle_downcasts() {
    let handle = SessionHandle::new(StubSession { marker: "x" });
    assert_eq!(handle.downcast_ref::<StubSession>().unwrap().marker, "x");
    assert!(handle.downcast_ref::<String>().is_none());
}

#[test]
fn unknown_model_returns_none() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    assert!(registry.ensure_loaded("nope").unwrap().is_none());
    assert_eq!(registry.config("nope"), None);
    assert!(!registry.refuses_unload("nope"));
    assert!(registry.model_keys().is_empty());
}

#[test]
fn duplicate_registration_is_an_error() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register("m", test_config(OnnxTask::FillMask))
        .expect("first");
    let err = registry.register("m", test_config(OnnxTask::FillMask));
    assert!(err.is_err());
}

#[test]
fn typed_accessors_exist_per_task() {
    // M0: the registry's task-aware surface is the config; typed workers
    // (encoder/two_tower/pii/colbert) are built on top of the handle in
    // their own modules. Assert the registry keys survive a mixed set.
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register("pii", test_config(OnnxTask::TokenClassification))
        .expect("pii");
    registry
        .register("router", test_config(OnnxTask::ZeroShotRouting))
        .expect("router");
    let mut keys = registry.model_keys();
    keys.sort();
    assert_eq!(keys, vec!["pii", "router"]);
    assert_eq!(registry.len(), 2);
    assert!(!registry.is_empty());
}

// ── M0: registry residency parity ──

#[test]
fn release_drops_unloadable_handle_and_refuses_always() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register_with_policy(
            "lazy",
            test_config(OnnxTask::FillMask),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
        )
        .expect("register");
    let handle = registry.ensure_loaded("lazy").expect("load");
    let handle = handle.unwrap();
    let stub = handle.downcast_ref::<StubSession>().unwrap();
    assert_eq!(stub.marker, "stub");

    assert!(registry.release("lazy").expect("release"));
    // The handle is gone: a downcast against a re-ensure_loaded would
    // reload; instead assert the report shows it unloaded.
    assert_eq!(registry.unloadable_keys(), Vec::<String>::new());
    assert!(!registry.release("lazy").expect("already released"));

    // An `Always` entry refuses release — the same refusal `refuses_unload`
    // implies.
    registry
        .register("always", test_config(OnnxTask::FillMask))
        .expect("register");
    assert!(registry.refuses_unload("always"));
    assert!(matches!(registry.release("always"), Err(OrtError::Other(_))));

    // Unknown key: nothing was released.
    assert!(!registry.release("nope").expect("unknown"));
}

#[test]
fn release_refuses_pinned_unloadable_entry() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register_with_lifecycle(
            "pinned-lazy",
            test_config(OnnxTask::FillMask),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
            true,
            Some(30),
        )
        .expect("register");
    assert!(registry.is_pinned("pinned-lazy"));
    assert_eq!(registry.sleep_idle_seconds("pinned-lazy"), Some(30));
    registry.ensure_loaded("pinned-lazy").expect("load");
    assert!(matches!(
        registry.release("pinned-lazy"),
        Err(OrtError::Other(_))
    ));
    assert_eq!(registry.unloadable_keys(), Vec::<String>::new());
}

#[test]
fn last_used_advances_on_ensure_loaded_and_touch() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register_with_policy(
            "lazy",
            test_config(OnnxTask::FillMask),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
        )
        .expect("register");
    assert_eq!(registry.last_used_of("lazy"), Some(0), "never used");

    registry.ensure_loaded("lazy").expect("load");
    let after_load = registry.last_used_of("lazy").unwrap();
    assert!(after_load > 0, "load touches last_used");

    registry.touch("lazy");
    let after_touch = registry.last_used_of("lazy").unwrap();
    assert!(
        after_touch >= after_load,
        "touch advances last_used (got {after_touch} >= {after_load})"
    );

    assert_eq!(registry.last_used_of("nope"), None);
}

#[test]
fn resident_bytes_sums_model_and_external_data() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model_q4.onnx");
    std::fs::write(&model, vec![0u8; 10]).unwrap();
    std::fs::write(dir.path().join("model_q4.onnx_data"), vec![0u8; 20]).unwrap();
    std::fs::write(dir.path().join("model_q4.onnx_data_1"), vec![0u8; 30]).unwrap();
    // A sibling that must NOT count.
    std::fs::write(dir.path().join("model_q4.onnx_data_1.bak"), vec![0u8; 999]).unwrap();

    let cfg = OnnxConfig::new()
        .model_path(&model)
        .tokenizer_path("/models/tokenizer.json")
        .task(OnnxTask::FillMask)
        .build();
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry.register("m", cfg).expect("register");
    assert_eq!(registry.resident_bytes("m"), Some(60));

    // A declared `resident_bytes` wins over the file computation.
    let cfg = OnnxConfig::new()
        .model_path(&model)
        .tokenizer_path("/models/tokenizer.json")
        .task(OnnxTask::FillMask)
        .maybe_resident_bytes(Some(777))
        .build();
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry.register("m", cfg).expect("register");
    assert_eq!(registry.resident_bytes("m"), Some(777));

    // An unresolved fixture path yields a zero footprint (never fabricated).
    let cfg = test_config(OnnxTask::FillMask);
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry.register("fake", cfg).expect("register");
    assert_eq!(registry.resident_bytes("fake"), Some(0));
}

#[test]
fn residency_report_is_stable_and_owned() {
    let registry = OrtSessionRegistry::new(Arc::new(CountingLoader::default()));
    registry
        .register_with_lifecycle(
            "lazy",
            test_config(OnnxTask::FillMask),
            ResidencyPolicy::Unloadable {
                weights: true,
                context: true,
            },
            false,
            Some(15),
        )
        .expect("register");
    registry
        .register("always", test_config(OnnxTask::FillMask))
        .expect("register");
    registry.ensure_loaded("lazy").expect("load");

    let report = registry.residency_report();
    assert_eq!(report.len(), 2);
    let lazy = report.iter().find(|r| r.key == "lazy").unwrap();
    assert!(lazy.loaded);
    assert!(!lazy.pinned);
    assert_eq!(lazy.sleep_idle_seconds, Some(15));
    assert_eq!(lazy.policy, registry.policy("lazy").unwrap());
    assert!(!lazy.policy.is_always());
    let always = report.iter().find(|r| r.key == "always").unwrap();
    assert!(always.policy.is_always());
    assert!(always.loaded, "Always loads at register");

    let keys = registry.unloadable_keys();
    assert_eq!(keys, vec!["lazy".to_string()]);
}
