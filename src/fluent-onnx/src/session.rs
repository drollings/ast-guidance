//! `OrtSessionRegistry` — one ONNX session per declared model, keyed by model
//! key, with a `SessionLoader` DIP seam for hermeticity.
//!
//! The registry itself is ort-free: it stores type-erased `SessionHandle`s and
//! delegates actual loading to an injected `SessionLoader`. The real
//! `OrtSessionLoader` lives behind the `onnx` feature; tests inject a stub, so
//! registry behavior (Always-at-boot, Unloadable-lazy, typed access) is
//! verified without loading a model.

use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use common_core::sync::lock;

use crate::config::{OnnxConfig, ResidencyPolicy};
use crate::error::OrtError;

/// Opaque handle to a loaded ONNX session. The concrete type lives behind the
/// `onnx` feature; the registry stores this type-erased value so its logic
/// compiles and is testable without `ort`.
#[derive(Clone)]
pub struct SessionHandle {
    inner: Arc<dyn Any + Send + Sync>,
}

impl SessionHandle {
    /// Wrap a concrete session in an opaque handle.
    pub fn new<T: Send + Sync + 'static>(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Borrow the concrete session type, if this handle holds one.
    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }

    /// Take an owned `Arc<T>` to the concrete session, if this handle holds one.
    pub fn downcast_arc<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.inner.clone().downcast::<T>().ok()
    }
}

/// How sessions get loaded. The real implementation is `OrtSessionLoader`
/// (feature `onnx`); tests inject a stub that returns a canned handle.
pub trait SessionLoader: Send + Sync {
    fn load(&self, config: &OnnxConfig, model_key: &str) -> Result<SessionHandle, OrtError>;
}

/// Registry of ONNX sessions keyed by model key.
#[derive(Clone)]
pub struct OrtSessionRegistry {
    inner: Arc<Mutex<RegistryInner>>,
    loader: Arc<dyn SessionLoader>,
}

struct RegistryInner {
    entries: HashMap<String, SessionEntry>,
}

struct SessionEntry {
    config: OnnxConfig,
    policy: ResidencyPolicy,
    session: Option<SessionHandle>,
    /// Unix-milliseconds of last use (load or touch). `0` = never used.
    /// `AtomicI64` so reads (`last_used_of`, `residency_report`) never take
    /// the registry lock; the residency loop reads the report without writing.
    last_used: AtomicI64,
    /// Resident memory this model occupies once loaded (bytes): the config's
    /// `resident_bytes` when declared, else computed at register from the
    /// model file + external-data siblings. Drives working-set eviction.
    resident_bytes: u64,
    /// Whether this entry is pinned (never released even when `Unloadable`).
    pinned: bool,
    /// Per-role idle threshold (seconds) after which the session may be
    /// released. `None`/`0` inherits the residency loop's default.
    sleep_idle_seconds: Option<i32>,
}

impl SessionEntry {
    fn touch_now(&self) {
        self.last_used.store(now_unix_ms(), Ordering::Relaxed);
    }
}

/// One row of the registry's residency view — the loop's read snapshot.
#[derive(Debug, Clone)]
pub struct ResidencyReportEntry {
    /// The registry key.
    pub key: String,
    /// Whether a session is currently loaded.
    pub loaded: bool,
    /// Resident bytes this model occupies once loaded.
    pub resident_bytes: u64,
    /// Unix-milliseconds of last use (`0` = never used).
    pub last_used_ms: i64,
    /// The residency policy.
    pub policy: ResidencyPolicy,
    /// Whether the entry is pinned (never released).
    pub pinned: bool,
    /// Per-entry idle threshold (seconds); `None`/`0` inherits the loop default.
    pub sleep_idle_seconds: Option<i32>,
}

/// Current system time as unix-milliseconds.
fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// The model file's external-data sibling prefix: `<model>.onnx` →
/// `<model>.onnx_data` (`onnx_data`, `onnx_data_1`, …).
fn external_data_prefix(model_file_name: &str) -> String {
    let base = model_file_name.strip_suffix(".onnx").unwrap_or(model_file_name);
    format!("{base}.onnx_data")
}

/// Whether `name` is an external-data sibling of the model file: `<prefix>`
/// exactly, or `<prefix>_<digits>` (the ONNX external-data `_N` suffix). A
/// `.bak` or any other suffix is not external data.
fn is_external_data(name: &str, prefix: &str) -> bool {
    let Some(rest) = name.strip_prefix(prefix) else {
        return false;
    };
    rest.is_empty() || {
        let digits = rest.strip_prefix('_');
        digits.is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
    }
}

/// Compute a model's resident memory (bytes): the config's declared
/// `resident_bytes` first, else the model file + every external-data sibling
/// under the same directory. `0` when the model file does not resolve (the
/// registry never fabricates a footprint for a fixture path).
fn compute_resident_bytes(config: &OnnxConfig) -> u64 {
    if let Some(bytes) = config.resident_bytes {
        return bytes;
    }
    let Ok(model_file) = config.resolve_model_file() else {
        return 0;
    };
    let total = file_size(&model_file);
    let Some(name) = model_file.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return total;
    };
    let prefix = external_data_prefix(&name);
    let Some(dir) = model_file.parent() else {
        return total;
    };
    let Ok(read) = std::fs::read_dir(dir) else {
        return total;
    };
    read.flatten().fold(total, |acc, entry| {
        let entry_name = entry.file_name().to_string_lossy().into_owned();
        if is_external_data(&entry_name, &prefix) {
            acc.saturating_add(file_size(&entry.path()))
        } else {
            acc
        }
    })
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

impl OrtSessionRegistry {
    /// Build an empty registry backed by the given loader.
    pub fn new(loader: Arc<dyn SessionLoader>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                entries: HashMap::new(),
            })),
            loader,
        }
    }

    /// Register a model. `Always` policies load at boot; `Unloadable` stay
    /// lazy until `ensure_loaded`.
    pub fn register(&self, model_key: impl Into<String>, config: OnnxConfig) -> Result<(), OrtError> {
        let policy = config.policy();
        self.register_with_policy(model_key, config, policy)
    }

    /// Register a model with an explicit policy (used by tests to pin the
    /// residency behavior independently of the `resident` knob).
    pub fn register_with_policy(
        &self,
        model_key: impl Into<String>,
        config: OnnxConfig,
        policy: ResidencyPolicy,
    ) -> Result<(), OrtError> {
        self.register_with_lifecycle(model_key, config, policy, false, None)
    }

    /// Register a model with the full residency lifecycle: the policy plus the
    /// per-role `pinned` flag and idle threshold. The router's role registration
    /// (`build_onnx_registry`) uses this so the residency loop honors each
    /// role's `OnnxRoleConfig` (`pinned` / `sleep_idle_seconds`) without a
    /// second parallel table.
    pub fn register_with_lifecycle(
        &self,
        model_key: impl Into<String>,
        config: OnnxConfig,
        policy: ResidencyPolicy,
        pinned: bool,
        sleep_idle_seconds: Option<i32>,
    ) -> Result<(), OrtError> {
        let model_key = model_key.into();
        let mut inner = lock(&self.inner);
        if inner.entries.contains_key(&model_key) {
            return Err(OrtError::Other(format!(
                "onnx model already registered: {model_key}"
            )));
        }
        let resident_bytes = compute_resident_bytes(&config);
        let mut entry = SessionEntry {
            config,
            policy,
            session: None,
            last_used: AtomicI64::new(0),
            resident_bytes,
            pinned,
            sleep_idle_seconds,
        };
        if policy.is_always() {
            entry.session = Some(self.loader.load(&entry.config, &model_key)?);
            entry.touch_now();
        }
        inner.entries.insert(model_key, entry);
        Ok(())
    }

    /// Load (or return the already-loaded) session for a model, touching its
    /// `last_used` clock. `None` when the model is not registered.
    pub fn ensure_loaded(&self, model_key: &str) -> Result<Option<SessionHandle>, OrtError> {
        let mut inner = lock(&self.inner);
        let Some(entry) = inner.entries.get_mut(model_key) else {
            return Ok(None);
        };
        if entry.session.is_none() {
            entry.session = Some(self.loader.load(&entry.config, model_key)?);
        }
        entry.touch_now();
        Ok(entry.session.clone())
    }

    /// The loaded (or loadable-on-demand) session handle for a model.
    pub fn session(&self, model_key: &str) -> Result<Option<SessionHandle>, OrtError> {
        self.ensure_loaded(model_key)
    }

    /// The config a model was registered with.
    pub fn config(&self, model_key: &str) -> Option<OnnxConfig> {
        lock(&self.inner)
            .entries
            .get(model_key)
            .map(|e| e.config.clone())
    }

    /// The residency policy a model was registered with.
    pub fn policy(&self, model_key: &str) -> Option<ResidencyPolicy> {
        lock(&self.inner)
            .entries
            .get(model_key)
            .map(|e| e.policy)
    }

    /// Whether this entry is pinned (never released even when `Unloadable`).
    pub fn is_pinned(&self, model_key: &str) -> bool {
        lock(&self.inner)
            .entries
            .get(model_key)
            .is_some_and(|e| e.pinned)
    }

    /// This entry's per-role idle threshold (seconds); `None`/`0` inherits the
    /// residency loop's default.
    pub fn sleep_idle_seconds(&self, model_key: &str) -> Option<i32> {
        lock(&self.inner)
            .entries
            .get(model_key)
            .and_then(|e| e.sleep_idle_seconds)
    }

    /// Mark a model as just-used, advancing its `last_used` clock. The
    /// residency loop uses this to keep hot sessions resident.
    pub fn touch(&self, model_key: &str) {
        let inner = lock(&self.inner);
        if let Some(entry) = inner.entries.get(model_key) {
            entry.touch_now();
        }
    }

    /// The unix-ms of a model's last use (`0` = never used). `None` when the
    /// model is not registered.
    pub fn last_used_of(&self, model_key: &str) -> Option<i64> {
        lock(&self.inner)
            .entries
            .get(model_key)
            .map(|e| e.last_used.load(Ordering::Relaxed))
    }

    /// The resident bytes this model occupies once loaded. `None` when the
    /// model is not registered.
    pub fn resident_bytes(&self, model_key: &str) -> Option<u64> {
        lock(&self.inner)
            .entries
            .get(model_key)
            .map(|e| e.resident_bytes)
    }

    /// Release a loaded session, clearing its handle so the next `ensure_loaded`
    /// reloads it. Only `Unloadable` (and unpinned) entries are released —
    /// `Always`/pinned entries are refused with an error, and an unknown or
    /// already-released key returns `false` (nothing was released). This is the
    /// registry's half of the residency-parity contract the llama supervisor
    /// has for VRAM (`ManagedServer::unload`).
    pub fn release(&self, model_key: &str) -> Result<bool, OrtError> {
        let mut inner = lock(&self.inner);
        let Some(entry) = inner.entries.get_mut(model_key) else {
            return Ok(false);
        };
        if entry.policy.is_always() || entry.pinned {
            return Err(OrtError::Other(format!(
                "onnx model {model_key} is {} and refuses unload",
                if entry.pinned { "pinned" } else { "Always-resident" }
            )));
        }
        if entry.session.is_none() {
            return Ok(false);
        }
        entry.session = None;
        entry.last_used.store(0, Ordering::Relaxed);
        Ok(true)
    }

    /// Keys of loaded, releasable (`Unloadable`, unpinned) sessions — the
    /// residency loop's eviction candidate set.
    pub fn unloadable_keys(&self) -> Vec<String> {
        let inner = lock(&self.inner);
        let mut keys: Vec<String> = inner
            .entries
            .iter()
            .filter(|(_, e)| {
                !e.policy.is_always() && !e.pinned && e.session.is_some()
            })
            .map(|(k, _)| k.clone())
            .collect();
        keys.sort_unstable();
        keys
    }

    /// A stable, owned residency snapshot of every registered model — the read
    /// model the residency loop iterates without holding the registry lock
    /// across a pass.
    pub fn residency_report(&self) -> Vec<ResidencyReportEntry> {
        let inner = lock(&self.inner);
        let mut report: Vec<ResidencyReportEntry> = inner
            .entries
            .iter()
            .map(|(key, e)| ResidencyReportEntry {
                key: key.clone(),
                loaded: e.session.is_some(),
                resident_bytes: e.resident_bytes,
                last_used_ms: e.last_used.load(Ordering::Relaxed),
                policy: e.policy,
                pinned: e.pinned,
                sleep_idle_seconds: e.sleep_idle_seconds,
            })
            .collect();
        report.sort_by(|a, b| a.key.cmp(&b.key));
        report
    }

    /// Whether this model is registered.
    pub fn is_registered(&self, model_key: &str) -> bool {
        lock(&self.inner).entries.contains_key(model_key)
    }

    /// Whether `/models/unload` must refuse this model (`Always` residency).
    pub fn refuses_unload(&self, model_key: &str) -> bool {
        self.policy(model_key)
            .is_some_and(ResidencyPolicy::is_always)
    }

    /// All registered model keys.
    pub fn model_keys(&self) -> Vec<String> {
        let inner = lock(&self.inner);
        let mut keys: Vec<String> = inner.entries.keys().cloned().collect();
        keys.sort_unstable();
        keys
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        lock(&self.inner).entries.len()
    }

    /// Whether no model is registered.
    pub fn is_empty(&self) -> bool {
        lock(&self.inner).entries.is_empty()
    }
}

/// The real `SessionLoader`: builds ONNX Runtime sessions from `OnnxConfig`.
/// Lives behind the `onnx` feature; `#[cfg(not(feature = "onnx"))]` builds
/// expose only the registry and the stub test seam.
#[cfg(feature = "onnx")]
pub mod ort_loader {
    use super::{SessionHandle, SessionLoader};
    use crate::config::OnnxConfig;
    use crate::error::OrtError;
    use ort::ep::ExecutionProvider;
    use ort::session::builder::GraphOptimizationLevel;

    /// Real session loader backed by ONNX Runtime.
    #[derive(Default)]
    pub struct OrtSessionLoader;

    fn optimization_level(name: &str) -> GraphOptimizationLevel {
        match name {
            "disable" => GraphOptimizationLevel::Disable,
            "basic" => GraphOptimizationLevel::Level1,
            "extended" => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::All,
        }
    }

    /// Whether a config's `execution_provider` requests the AMD ROCm GPU
    /// provider. Case-insensitive. The GPU provider is `MIGraphX` — AMD's
    /// supported ONNX Runtime EP for ROCm. (The `ROCMExecutionProvider` was
    /// removed from upstream ORT in 1.23; MIGraphX is its successor.)
    pub(crate) fn is_gpu_provider(name: &str) -> bool {
        name.eq_ignore_ascii_case("gpu") || name.eq_ignore_ascii_case("migraphx")
    }

    impl SessionLoader for OrtSessionLoader {
        fn load(&self, config: &OnnxConfig, model_key: &str) -> Result<SessionHandle, OrtError> {
            let model_file = config.resolve_model_file()?;
            let mut builder = ort::session::Session::builder()
                .map_err(|e| session_load_error(model_key, &e))?
                .with_intra_threads(config.intra_threads)
                .map_err(|e| session_load_error(model_key, &e))?
                .with_optimization_level(optimization_level(&config.optimization_level))
                .map_err(|e| session_load_error(model_key, &e))?;

            // Execution-provider selection. Only the CPU and the AMD ROCm GPU
            // providers are wired:
            //
            //   "cpu"        → `CPUExecutionProvider` (deterministic, the
            //                  hermetic default).
            //   "gpu" | "migraphx"
            //                → `MIGraphXExecutionProvider` — AMD's supported
            //                  GPU EP for ROCm (successor to the removed
            //                  `ROCMExecutionProvider`). The linked runtime is
            //                  probed first: a build without MIGraphX support
            //                  (e.g. the CPU prebuilt binary) fails open to the
            //                  CPU with a loud, actionable warning — a `"gpu"`
            //                  request is never silently served.
            //
            // Anything else fails open to the CPU with a loud warning (a
            // mistyped or unsupported provider must never silently pretend to
            // accelerate).
            if config.execution_provider.eq_ignore_ascii_case("cpu") {
                builder = builder
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .map_err(|e| session_load_error(model_key, &e))?;
            } else if is_gpu_provider(&config.execution_provider) {
                let gpu = ort::ep::MIGraphX::default();
                match gpu.is_available() {
                    Ok(true) => {
                        tracing::info!(
                            target: "fluent-onnx",
                            model = model_key,
                            execution_provider = %config.execution_provider,
                            "using MIGraphX (AMD ROCm GPU) execution provider",
                        );
                        builder = builder
                            .with_execution_providers([gpu.build()])
                            .map_err(|e| session_load_error(model_key, &e))?;
                    }
                    Ok(false) => {
                        tracing::warn!(
                            target: "fluent-onnx",
                            model = model_key,
                            execution_provider = %config.execution_provider,
                            "GPU execution requested but the linked ONNX Runtime has no \
                             MIGraphX (AMD ROCm) support; falling back to the CPU. Link a \
                             MIGraphX-enabled onnxruntime (an AMD ROCm build, e.g. via \
                             ORT_LIB_PATH) to actually accelerate",
                        );
                        builder = builder
                            .with_execution_providers([ort::ep::CPU::default().build()])
                            .map_err(|e| session_load_error(model_key, &e))?;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "fluent-onnx",
                            model = model_key,
                            execution_provider = %config.execution_provider,
                            error = %e,
                            "GPU execution probe failed; falling back to the CPU",
                        );
                        builder = builder
                            .with_execution_providers([ort::ep::CPU::default().build()])
                            .map_err(|e| session_load_error(model_key, &e))?;
                    }
                }
            } else {
                tracing::warn!(
                    target: "fluent-onnx",
                    model = model_key,
                    execution_provider = %config.execution_provider,
                    "requested execution provider is not wired; falling back to the default (CPU)",
                );
                builder = builder
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .map_err(|e| session_load_error(model_key, &e))?;
            }

            let session = builder
                .commit_from_file(&model_file)
                .map_err(|e| session_load_error(model_key, &e))?;
            // Session::run needs `&mut self`; the handle stores a Mutex so
            // every worker (encoder, two-tower, PII) can serialize runs on the
            // shared session.
            Ok(SessionHandle::new(std::sync::Mutex::new(session)))
        }
    }

    fn session_load_error(model_key: &str, source: &impl ToString) -> OrtError {
        OrtError::SessionLoad {
            model: model_key.to_string(),
            detail: source.to_string(),
        }
    }
}

#[cfg(feature = "onnx")]
pub use ort_loader::OrtSessionLoader;

#[cfg(test)]
#[path = "../tests/session.rs"]
mod tests;
