//! Central memory plugin registry. Type-erased storage via `Arc<dyn MemoryPlugin>`.

use crate::traits::{MemoryOps, MemoryPlugin};
use crate::types::MemoryError;
use common_core::registry::KeyedRegistry;
use std::sync::Arc;

/// A registered plugin entry.
struct PluginEntry {
    plugin: Arc<dyn MemoryPlugin>,
    active: bool,
}

/// The central memory plugin registry.
///
/// Stores type-erased `Arc<dyn MemoryPlugin>` handles keyed by `&'static str`
/// plugin name. The orchestrator never branches on implementation type.
///
/// # Thread Safety
///
/// Mutation (register, set_active) is infallible and happens at startup.
/// Lookup (active, get, list) is read-only and can happen concurrently.
/// The registry is designed to be wrapped in `tokio::sync::RwLock` for
/// the async integration layer.
pub struct MemoryPluginRegistry {
    plugins: KeyedRegistry<&'static str, PluginEntry>,
    active_name: Option<&'static str>,
}

impl MemoryPluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: KeyedRegistry::new(),
            active_name: None,
        }
    }

    /// Register a plugin. Type erasure happens here — the concrete type
    /// is wrapped in `Arc<dyn MemoryPlugin>` and never seen again.
    ///
    /// # Panics
    ///
    /// Panics if a plugin with the same name is already registered.
    /// This is intentional: duplicate registration is a programming error,
    /// not a runtime condition.
    pub fn register(&mut self, plugin: Arc<dyn MemoryPlugin>) {
        let name = <dyn MemoryPlugin as MemoryOps>::name(&*plugin);
        assert!(
            !self.plugins.contains(name),
            "duplicate memory plugin registration: '{name}'"
        );
        self.plugins.insert(
            name,
            PluginEntry {
                plugin,
                active: false,
            },
        );
    }

    /// Set the active plugin by name. Only one can be active at a time.
    pub fn set_active(&mut self, name: &'static str) -> Result<(), MemoryError> {
        if !self.plugins.contains(name) {
            return Err(MemoryError::InitFailed(format!(
                "plugin '{name}' not registered"
            )));
        }
        // Deactivate current
        if let Some(current) = self.active_name {
            if let Some(entry) = self.plugins.get_mut(current) {
                entry.active = false;
            }
        }
        self.plugins.get_mut(name).unwrap().active = true;
        self.active_name = Some(name);
        Ok(())
    }

    /// Get the active plugin, if one is set and available.
    pub fn active(&self) -> Option<Arc<dyn MemoryPlugin>> {
        self.active_name
            .and_then(|name| self.plugins.get(name))
            .filter(|entry| entry.plugin.is_available())
            .map(|entry| Arc::clone(&entry.plugin))
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn MemoryPlugin>> {
        self.plugins
            .get(name)
            .map(|entry| Arc::clone(&entry.plugin))
    }

    /// List all registered plugins: (name, is_active, is_available).
    pub fn list(&self) -> Vec<(&'static str, bool, bool)> {
        self.plugins
            .iter()
            .map(|(name, entry)| (*name, entry.active, entry.plugin.is_available()))
            .collect()
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for MemoryPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryOps;
    use crate::types::*;
    use fluent_wvr::prelude::*;
    use internment::ArcIntern;
    use serde_json::json;
    use std::future::Future;
    use std::pin::Pin;

    // ── Test stub ──────────────────────────────────────────────

    struct StubPlugin {
        name: &'static str,
        available: bool,
    }

    impl MemoryOps for StubPlugin {
        fn name(&self) -> &'static str {
            self.name
        }
        fn is_available(&self) -> bool {
            self.available
        }
        fn initialize(&mut self, _ctx: &MemoryInitContext) -> Result<(), MemoryError> {
            Ok(())
        }
        fn shutdown(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
        fn prefetch(
            &self,
            _query: &str,
            _ctx: &MemoryQueryContext,
        ) -> Pin<Box<dyn Future<Output = String> + Send + '_>> {
            Box::pin(async { String::new() })
        }
        fn queue_prefetch(&self, _query: &str, _ctx: &MemoryQueryContext) {}
        fn search(
            &self,
            _req: &MemorySearchRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryResult>, MemoryError>> + Send + '_>>
        {
            Box::pin(async { Ok(vec![]) })
        }
        fn sync_turn(
            &self,
            _user: &str,
            _assistant: &str,
            _ctx: &MemoryQueryContext,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
        fn on_session_end(
            &self,
            _messages: &[TurnMessage],
        ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
        fn handle_tool_call(
            &self,
            _tool_name: &str,
            _args: &serde_json::Value,
        ) -> Result<String, MemoryError> {
            Ok("{}".into())
        }
        fn tool_schemas(&self) -> Vec<ToolSchema> {
            vec![]
        }
    }

    impl FieldAccess for StubPlugin {
        fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
            Ok(())
        }
        fn get_field(&self, _name: &str) -> Result<String, FieldError> {
            Err(FieldError::NotFound("stub".into()))
        }
        fn field_names(&self) -> &'static [&'static str] {
            &[]
        }
    }

    impl Describable for StubPlugin {
        fn describe(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
    }

    impl WorkUnit for StubPlugin {
        fn name(&self) -> &str {
            self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("stub"))
        }
    }

    impl_component!(StubPlugin);

    // StubPlugin satisfies MemoryPlugin via our blanket impl.

    #[test]
    fn register_and_activate() {
        let mut reg = MemoryPluginRegistry::new();
        reg.register(Arc::new(StubPlugin {
            name: "alpha",
            available: true,
        }));
        reg.register(Arc::new(StubPlugin {
            name: "beta",
            available: false,
        }));

        assert_eq!(reg.len(), 2);

        // No active plugin yet
        assert!(reg.active().is_none());

        // Set alpha as active
        reg.set_active("alpha").unwrap();
        let active = reg.active().unwrap();
        assert_eq!(MemoryOps::name(&*active), "alpha");

        // Beta is not available, so active returns None
        reg.set_active("beta").unwrap();
        assert!(reg.active().is_none());

        // Switch back to alpha
        reg.set_active("alpha").unwrap();
        assert!(reg.active().is_some());
    }

    #[test]
    fn set_active_unknown_plugin() {
        let mut reg = MemoryPluginRegistry::new();
        let result = reg.set_active("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn list_reports_status() {
        let mut reg = MemoryPluginRegistry::new();
        reg.register(Arc::new(StubPlugin {
            name: "a",
            available: true,
        }));
        reg.register(Arc::new(StubPlugin {
            name: "b",
            available: false,
        }));
        reg.set_active("a").unwrap();

        let list = reg.list();
        assert_eq!(list.len(), 2);
        assert!(list
            .iter()
            .any(|(n, active, avail)| *n == "a" && *active && *avail));
        assert!(list
            .iter()
            .any(|(n, active, avail)| *n == "b" && !*active && !*avail));
    }
}
