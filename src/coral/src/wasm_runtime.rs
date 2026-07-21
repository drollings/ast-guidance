use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use thiserror::Error;

type CachedPlugin = Arc<Mutex<Box<dyn WasmPlugin>>>;
type PluginCache = LruCache<String, CachedPlugin>;

#[derive(Error, Debug)]
pub enum WasmError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("plugin load failed: {0}")]
    PluginLoad(String),
    #[error("plugin call failed: {0}")]
    PluginCall(String),
    #[error("invalid payload: {0}")]
    InvalidPayload(String),
}

impl From<std::io::Error> for WasmError {
    fn from(e: std::io::Error) -> Self {
        WasmError::Io(common_core::error::IoError(e))
    }
}

pub trait WasmPlugin: Send {
    fn call(&mut self, payload: &[u8]) -> Result<Vec<u8>, WasmError>;
}

/// LRU cache of loaded WASM plugins, keyed by file path.
///
/// Plugins are loaded on first access and reused for subsequent calls to the
/// same path. When the cache exceeds `max_capacity`, the least-recently-used
/// plugin is evicted.
pub struct PluginPool {
    plugins: Mutex<PluginCache>,
    runtime: Arc<dyn WasmRuntime>,
}

impl PluginPool {
    pub fn new(runtime: Arc<dyn WasmRuntime>, max_capacity: usize) -> Self {
        let cap = NonZeroUsize::new(max_capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            plugins: Mutex::new(LruCache::new(cap)),
            runtime,
        }
    }

    /// Return a cached plugin for `path`, or load + cache a new one.
    ///
    /// The caller receives an `Arc<Mutex<Box<dyn WasmPlugin>>>` and should
    /// lock it only for the duration of the `call`. On eviction the `Arc`
    /// stays valid (the caller may still hold a reference) but subsequent
    /// `get_or_load` calls for the same path will load a fresh instance.
    pub fn get_or_load(&self, path: &str) -> Result<CachedPlugin, WasmError> {
        // Fast path: plugin already cached
        {
            let mut cache = self.plugins.lock().unwrap();
            if let Some(plugin) = cache.get(path) {
                return Ok(Arc::clone(plugin));
            }
        }
        // Slow path: load from disk, insert into cache
        let plugin = self.runtime.load_plugin_from_file(Path::new(path))?;
        let shared = Arc::new(Mutex::new(plugin));
        let mut cache = self.plugins.lock().unwrap();
        cache.put(path.to_string(), Arc::clone(&shared));
        Ok(shared)
    }

    /// Number of plugins currently held in the cache.
    pub fn cache_size(&self) -> usize {
        self.plugins.lock().unwrap().len()
    }
}

pub trait WasmRuntime: Send + Sync {
    fn load_plugin(&self, wasm_bytes: &[u8]) -> Result<Box<dyn WasmPlugin>, WasmError>;
    fn load_plugin_from_file(&self, path: &Path) -> Result<Box<dyn WasmPlugin>, WasmError>;
}

pub struct ExtismWasmRuntime;

impl ExtismWasmRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExtismWasmRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmRuntime for ExtismWasmRuntime {
    fn load_plugin(&self, wasm_bytes: &[u8]) -> Result<Box<dyn WasmPlugin>, WasmError> {
        let plugin = extism::Plugin::new(wasm_bytes, [], true)
            .map_err(|e| WasmError::PluginLoad(e.to_string()))?;

        Ok(Box::new(ExtismPlugin { plugin }))
    }

    fn load_plugin_from_file(&self, path: &Path) -> Result<Box<dyn WasmPlugin>, WasmError> {
        let wasm_bytes = std::fs::read(path)?;
        self.load_plugin(&wasm_bytes)
    }
}

struct ExtismPlugin {
    plugin: extism::Plugin,
}

impl WasmPlugin for ExtismPlugin {
    fn call(&mut self, payload: &[u8]) -> Result<Vec<u8>, WasmError> {
        let result = self
            .plugin
            .call("execute", payload)
            .map_err(|e| WasmError::PluginCall(e.to_string()))?;
        Ok(result)
    }
}

use fluent_wvr::{
    impl_component, Describable, FieldAccess, FieldError, WorkContext, WorkError, WorkOutput,
    WorkUnit,
};
use internment::ArcIntern;

pub struct WasmComponent {
    name: String,
    plugin: Mutex<Box<dyn WasmPlugin>>,
    config: Mutex<std::collections::HashMap<String, String>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl WasmComponent {
    pub fn new(name: impl Into<String>, plugin: Box<dyn WasmPlugin>) -> Self {
        Self {
            name: name.into(),
            plugin: Mutex::new(plugin),
            config: Mutex::new(std::collections::HashMap::new()),
            depends: Vec::new(),
            provides: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_depends(mut self, deps: &[ArcIntern<str>]) -> Self {
        self.depends = deps.to_vec();
        self
    }

    #[must_use]
    pub fn with_provides(mut self, prov: &[ArcIntern<str>]) -> Self {
        self.provides = prov.to_vec();
        self
    }
}

impl FieldAccess for WasmComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        self.config
            .lock()
            .unwrap()
            .insert(name.to_string(), value.to_string());
        Ok(())
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        self.config
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| FieldError::NotFound(name.into()))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for WasmComponent {
    fn describe(&self) -> serde_json::Value {
        let config = self.config.lock().unwrap();
        serde_json::json!({
            "name": self.name,
            "type": "wasm",
            "config": *config,
        })
    }
}

impl_component!(WasmComponent);

impl WorkUnit for WasmComponent {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let payload = serde_json::to_vec(&*self.config.lock().unwrap())
            .map_err(|e| WorkError::Execution(format!("serialization: {e}")))?;
        let result = self
            .plugin
            .lock()
            .unwrap()
            .call(&payload)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let data: serde_json::Value = serde_json::from_slice(&result).unwrap_or_default();
        Ok(WorkOutput::ok_with_data(
            format!("{} executed", self.name),
            data,
        ))
    }
}

/// Decode a base64-encoded WASM binary using the extism-compatible simple decoder.
pub fn decode_wasm_base64(b64: &str) -> Result<Vec<u8>, WasmError> {
    // Simple base64 decode without external crate dependency
    let bytes = b64.as_bytes();
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0;
    for &c in bytes {
        let val = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => {
                return Err(WasmError::InvalidPayload(format!(
                    "invalid base64 char: {}",
                    c as char
                )))
            }
        };
        buf = (buf << 6) | u32::from(val);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPlugin;

    impl WasmPlugin for MockPlugin {
        fn call(&mut self, payload: &[u8]) -> Result<Vec<u8>, WasmError> {
            Ok(payload.to_vec())
        }
    }

    struct MockRuntime;

    impl WasmRuntime for MockRuntime {
        fn load_plugin(&self, _wasm_bytes: &[u8]) -> Result<Box<dyn WasmPlugin>, WasmError> {
            Ok(Box::new(MockPlugin))
        }
        fn load_plugin_from_file(&self, _path: &Path) -> Result<Box<dyn WasmPlugin>, WasmError> {
            Ok(Box::new(MockPlugin))
        }
    }

    #[test]
    fn test_runtime_creation() {
        let runtime = ExtismWasmRuntime::new();
        let result = runtime.load_plugin(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_plugin_file_not_found() {
        let runtime = ExtismWasmRuntime::new();
        let result = runtime.load_plugin_from_file(Path::new("/nonexistent/plugin.wasm"));
        assert!(result.is_err());
    }

    #[test]
    fn test_wasm_component_name() {
        let plugin = MockPlugin;
        let comp = WasmComponent::new("test_wasm", Box::new(plugin));
        assert_eq!(comp.name(), "test_wasm");
    }

    #[test]
    fn test_wasm_component_field_access() {
        let mut comp = WasmComponent::new("test", Box::new(MockPlugin));
        comp.set_field("key1", "value1").unwrap();
        assert_eq!(comp.get_field("key1").unwrap(), "value1");
        assert!(comp.get_field("nonexistent").is_err());
    }

    #[test]
    fn test_wasm_component_describe() {
        let mut comp = WasmComponent::new("desc_test", Box::new(MockPlugin));
        comp.set_field("port", "8080").unwrap();
        let desc = comp.describe();
        assert_eq!(desc["name"], "desc_test");
        assert_eq!(desc["config"]["port"], "8080");
    }

    #[test]
    fn test_wasm_component_execute() {
        let mut comp = WasmComponent::new("exec_test", Box::new(MockPlugin));
        comp.set_field("msg", "hello").unwrap();
        let ctx = WorkContext::default();
        let output = comp.execute(&ctx).unwrap();
        assert!(output.success);
    }

    #[test]
    fn test_wasm_component_deps() {
        let comp = WasmComponent::new("deps_test", Box::new(MockPlugin))
            .with_depends(&[ArcIntern::from("prep")])
            .with_provides(&[ArcIntern::from("result")]);
        assert_eq!(comp.depends().len(), 1);
        assert_eq!(&*comp.provides()[0], "result");
    }

    // --- PluginPool tests ---

    #[test]
    fn test_plugin_pool_returns_same_instance_for_same_path() {
        let runtime: Arc<dyn WasmRuntime> = Arc::new(MockRuntime);
        let pool = PluginPool::new(runtime, 4);
        let p1 = pool.get_or_load("/fake/tool.wasm").unwrap();
        let p2 = pool.get_or_load("/fake/tool.wasm").unwrap();
        assert!(
            Arc::ptr_eq(&p1, &p2),
            "pool should return same Arc instance"
        );
        assert_eq!(pool.cache_size(), 1);
    }

    #[test]
    fn test_plugin_pool_different_paths_return_different_instances() {
        let runtime: Arc<dyn WasmRuntime> = Arc::new(MockRuntime);
        let pool = PluginPool::new(runtime, 4);
        let p1 = pool.get_or_load("/tool_a.wasm").unwrap();
        let p2 = pool.get_or_load("/tool_b.wasm").unwrap();
        assert!(
            !Arc::ptr_eq(&p1, &p2),
            "different paths should yield different plugin instances"
        );
        assert_eq!(pool.cache_size(), 2);
    }

    #[test]
    fn test_plugin_pool_lru_eviction() {
        let runtime: Arc<dyn WasmRuntime> = Arc::new(MockRuntime);
        let pool = PluginPool::new(runtime, 2);
        let p1 = pool.get_or_load("/a.wasm").unwrap();
        let _p2 = pool.get_or_load("/b.wasm").unwrap();
        assert_eq!(pool.cache_size(), 2);
        // Adding a third entry evicts the least-recently-used (/a.wasm)
        let _p3 = pool.get_or_load("/c.wasm").unwrap();
        assert_eq!(pool.cache_size(), 2);
        // /a.wasm was evicted — get_or_load loads a fresh instance
        let p1_new = pool.get_or_load("/a.wasm").unwrap();
        let p1_old = p1;
        assert!(
            !Arc::ptr_eq(&p1_new, &p1_old),
            "evicted plugin should be reloaded as a new instance"
        );
    }
}
