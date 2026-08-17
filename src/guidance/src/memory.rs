//! Memory integration for the guidance query pipeline.
//!
//! Provides the bridge between `memory-plugin` and guidance's query engine.
//! The `MemoryBridge` holds a `MemoryCapability` and provides methods for
//! prefetch injection, turn syncing, and tool dispatch.

use std::sync::Arc;

use fluent_types::SessionId;
use memory_plugin::capability::MemoryCapability;
use memory_plugin::plugins::holographic::{HolographicConfig, HolographicMemory};
use memory_plugin::registry::MemoryPluginRegistry;
use memory_plugin::types::{
    MemoryError, MemoryQueryContext, MemoryResult, MemorySearchRequest, ToolSchema, TurnMessage,
};

/// Bridge between guidance's query pipeline and the memory plugin system.
///
/// Holds a `MemoryCapability` and provides high-level methods that guidance
/// components can call without directly depending on memory-plugin internals.
pub struct MemoryBridge {
    capability: MemoryCapability,
    session_id: SessionId,
}

impl MemoryBridge {
    /// Create a new memory bridge.
    pub fn new(capability: MemoryCapability, session_id: SessionId) -> Self {
        Self {
            capability,
            session_id,
        }
    }

    /// Create a query context for memory operations.
    fn query_ctx(&self) -> MemoryQueryContext {
        MemoryQueryContext {
            session_id: self.session_id.clone(),
            caps: fluent_wvr::CapabilitySet::default(),
            rt: Arc::new(fluent_wvr::NoopRuntime),
        }
    }

    /// Pre-fetch memory context for injection into the system prompt.
    ///
    /// Returns formatted text that should be prepended to the LLM system
    /// prompt. Returns empty string if no memory plugin is active or
    /// no relevant context is found.
    pub async fn prefetch_context(&self, query: &str) -> String {
        let ctx = self.query_ctx();
        self.capability.prefetch(query, &ctx).await
    }

    /// Sync a completed turn with the active memory plugin.
    ///
    /// Call this after LLM synthesis completes to persist the interaction.
    pub async fn sync_turn(&self, user_content: &str, assistant_content: &str) {
        let ctx = self.query_ctx();
        self.capability
            .sync_turn(user_content, assistant_content, &ctx)
            .await;
    }

    /// Notify the memory plugin of session end.
    pub async fn session_end(&self, messages: &[TurnMessage]) {
        self.capability.on_session_end(messages).await;
    }

    /// Dispatch a tool call to the active memory plugin.
    pub async fn handle_tool_call(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, MemoryError> {
        self.capability.handle_tool_call(tool_name, args).await
    }

    /// Get tool schemas from the active memory plugin.
    pub async fn tool_schemas(&self) -> Vec<ToolSchema> {
        self.capability.tool_schemas().await
    }

    /// Search the active memory plugin.
    pub async fn search(
        &self,
        req: &MemorySearchRequest,
    ) -> Result<Vec<MemoryResult>, MemoryError> {
        self.capability.search(req).await
    }

    /// Initialize the memory plugin system.
    ///
    /// Note: `initialize` takes `&mut self` on the plugin, which means it
    /// must be called before the plugin is wrapped in `Arc`. This method
    /// is provided for completeness but in practice initialization should
    /// happen during startup before the MemoryBridge is created.
    pub fn initialize(&self) -> Result<(), MemoryError> {
        // Plugin initialization must happen before Arc wrapping.
        // This method is a no-op placeholder; actual initialization
        // should be done in the binary's startup code before creating
        // the MemoryBridge.
        Ok(())
    }
}

/// Initialize the memory plugin system and return a bridge for the query pipeline.
///
/// Creates a registry, registers the holographic memory plugin (the primary
/// deterministic-first memory backend), sets it as active, and wraps the
/// capability in a `MemoryBridge` for guidance's query engine.
///
/// Returns `None` if initialization fails.
pub fn init_memory_bridge() -> Option<MemoryBridge> {
    let mut registry = MemoryPluginRegistry::new();

    // Register the holographic memory plugin (deterministic-first, SQLite-backed)
    let config = HolographicConfig {
        db_path: dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".guidance")
            .join("memory.db"),
        ..HolographicConfig::default()
    };
    let plugin = std::sync::Arc::new(HolographicMemory::new(config));
    registry.register(plugin);

    // Set holographic as the active memory plugin
    if registry.set_active("holographic").is_err() {
        return None;
    }

    let registry = std::sync::Arc::new(tokio::sync::RwLock::new(registry));
    let capability = MemoryCapability::new(std::sync::Arc::clone(&registry));
    let session_id = SessionId::new(format!("guidance-{}", std::process::id()));

    Some(MemoryBridge::new(capability, session_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_plugin::capability::MemoryCapability;

    // A bridge over an empty registry (no active plugin). This exercises the
    // deterministic "no plugin configured" happy/error paths that a
    // production deployment without the memory plugin always lands on.
    fn empty_bridge() -> MemoryBridge {
        let registry = Arc::new(tokio::sync::RwLock::new(MemoryPluginRegistry::new()));
        MemoryBridge::new(
            MemoryCapability::new(registry),
            SessionId::new("test-session"),
        )
    }

    #[tokio::test]
    async fn prefetch_with_no_active_plugin_returns_empty() {
        let bridge = empty_bridge();
        assert_eq!(bridge.prefetch_context("what is x").await, "");
    }

    #[tokio::test]
    async fn tool_schemas_with_no_active_plugin_is_empty() {
        let bridge = empty_bridge();
        assert!(bridge.tool_schemas().await.is_empty());
    }

    #[tokio::test]
    async fn handle_tool_call_errors_without_active_plugin() {
        let bridge = empty_bridge();
        let err = bridge
            .handle_tool_call("tool", &serde_json::json!({}))
            .await
            .expect_err("no plugin -> error");
        assert!(matches!(err, MemoryError::NotAvailable(_)));
    }

    #[tokio::test]
    async fn search_errors_without_active_plugin() {
        let bridge = empty_bridge();
        let req = MemorySearchRequest {
            query: "x".into(),
            category: None,
            min_trust: 0.0,
            limit: 5,
            strategy: memory_plugin::types::SearchStrategy::FtsKeyword,
        };
        assert!(bridge.search(&req).await.is_err());
    }

    #[test]
    fn initialize_is_noop_ok() {
        assert!(empty_bridge().initialize().is_ok());
    }

    #[test]
    fn session_id_is_preserved() {
        assert_eq!(empty_bridge().session_id.as_str(), "test-session");
    }
}
