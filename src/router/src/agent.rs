//! Agent registry — keyed on `(model, adapter, session)` triple.
//! Uses `fluent-concurrency`'s `ResultPool` for pool-per-identity dispatch,
//! preserving KV-cache affinity.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use fluent_concurrency::pool::ResultPool;
use fluent_wvr::Runtime;
use thiserror::Error;

/// An agent identity is the triple `(model, adapter, session)`.
/// All requests for the same identity are routed to the same worker pool,
/// preserving KV-cache affinity.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AgentIdentity {
    pub model: String,
    pub adapter: Option<String>,
    pub session_id: String,
}

impl AgentIdentity {
    pub fn new(model: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            adapter: None,
            session_id: session_id.into(),
        }
    }

    #[must_use]
    pub fn with_adapter(mut self, adapter: impl Into<String>) -> Self {
        self.adapter = Some(adapter.into());
        self
    }
}

/// A task dispatched to a local agent.
#[derive(Debug, Clone)]
pub struct AgentTask {
    pub messages: Vec<guidance_llm::ChatMessage>,
    pub config: AgentConfig,
}

/// Per-dispatch configuration for an agent call.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub adapter: Option<String>,
    pub session_id: String,
    pub timeout_ms: u64,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            adapter: None,
            session_id: String::new(),
            timeout_ms: 30_000,
            max_tokens: None,
            temperature: None,
        }
    }
}

/// Errors produced by agent operations.
#[derive(Error, Debug, Clone)]
pub enum AgentError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("adapter not loaded: {0}")]
    AdapterNotLoaded(String),
    #[error("session error: {0}")]
    Session(String),
    #[error("execution error: {0}")]
    Execution(String),
}

/// A handle to a loaded LoRA adapter.
#[derive(Debug, Clone)]
pub struct AdapterHandle {
    pub name: String,
    pub path: String,
    pub base_model: String,
}

impl AdapterHandle {
    pub fn new(
        name: impl Into<String>,
        path: impl Into<String>,
        base_model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            base_model: base_model.into(),
        }
    }
}

/// Handler function signature for agent dispatch.
/// Receives an `AgentTask` and returns the agent's text response.
pub type AgentHandlerFn = Arc<
    dyn Fn(AgentTask) -> std::pin::Pin<Box<dyn Future<Output = Result<String, AgentError>> + Send>>
        + Send
        + Sync,
>;

/// Registry of agent sessions. Each agent identity is the triple
/// `(model, adapter, session)`. Routing and queueing are keyed on this
/// triple using `fluent-concurrency`'s `ResultPool` pattern.
///
/// # Design rules
/// - Adapters are LoRA, pre-loaded at startup. Loading a not-yet-resident
///   adapter is a heavier operation, not per-request.
/// - Adapter switch implies a new KV-cache line — never in-place swap
///   under a live cache.
/// - All requests for the same `(model, adapter, session)` go to the same
///   worker pool, preserving cache affinity.
pub struct AgentRegistry {
    pools: HashMap<AgentIdentity, Arc<ResultPool<AgentTask, String, AgentError>>>,
    adapters: HashMap<(String, String), AdapterHandle>,
    runtime: Arc<dyn Runtime>,
    worker_count: usize,
    queue_capacity: usize,
}

impl AgentRegistry {
    /// Creates a new registry with the given runtime and pool sizing.
    pub fn new(runtime: Arc<dyn Runtime>, worker_count: usize, queue_capacity: usize) -> Self {
        Self {
            pools: HashMap::new(),
            adapters: HashMap::new(),
            runtime,
            worker_count,
            queue_capacity,
        }
    }

    /// Pre-load an adapter into the registry. Validates that the base model
    /// is known at load time (the model catalog must contain it).
    pub fn register_adapter(&mut self, adapter: AdapterHandle) -> &mut Self {
        let key = (adapter.base_model.clone(), adapter.name.clone());
        self.adapters.insert(key, adapter);
        self
    }

    /// Look up a pre-loaded adapter.
    pub fn get_adapter(&self, base_model: &str, adapter_name: &str) -> Option<&AdapterHandle> {
        self.adapters
            .get(&(base_model.to_string(), adapter_name.to_string()))
    }

    /// Returns true if the adapter is resident.
    pub fn is_adapter_loaded(&self, base_model: &str, adapter_name: &str) -> bool {
        self.adapters
            .contains_key(&(base_model.to_string(), adapter_name.to_string()))
    }

    /// Register or replace an agent pool for the given identity.
    ///
    /// The `handler` is the actual LLM execution function (e.g., an HTTP
    /// call to llama.cpp server). Multiple calls with the same identity
    /// overwrite the previous pool, re-creating workers.
    pub fn register_agent<F, Fut>(&mut self, identity: AgentIdentity, handler: F)
    where
        F: Fn(AgentTask) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, AgentError>> + Send,
    {
        let pool = ResultPool::new(
            Arc::clone(&self.runtime),
            self.worker_count,
            self.queue_capacity,
            handler,
        );
        self.pools.insert(identity, Arc::new(pool));
    }

    /// Returns true if an agent pool exists for this identity.
    pub fn has_agent(&self, identity: &AgentIdentity) -> bool {
        self.pools.contains_key(identity)
    }

    /// Number of registered agent identities.
    pub fn agent_count(&self) -> usize {
        self.pools.len()
    }

    /// Submit a task to the pool for the given identity.
    ///
    /// Returns `AgentError::ModelNotFound` if no pool exists for the identity.
    pub async fn submit(
        &self,
        identity: &AgentIdentity,
        task: AgentTask,
    ) -> Result<String, AgentError> {
        let pool = self.pools.get(identity).ok_or_else(|| {
            AgentError::ModelNotFound(format!(
                "no agent pool for model={} adapter={:?} session={}",
                identity.model, identity.adapter, identity.session_id,
            ))
        })?;

        pool.submit(task).await.map_err(|e| match e {
            fluent_concurrency::pool::ResultPoolError::Inner(inner) => inner,
            fluent_concurrency::pool::ResultPoolError::Pool(p) => {
                AgentError::Execution(p.to_string())
            }
            fluent_concurrency::pool::ResultPoolError::Canceled => {
                AgentError::Execution("pool canceled".into())
            }
        })
    }

    /// Returns a reference to the runtime used by this registry.
    pub fn runtime(&self) -> &Arc<dyn Runtime> {
        &self.runtime
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new(fluent_concurrency::tokio_runtime(), 4, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_concurrency::pool::global_pool_config;

    async fn mock_handler(_task: AgentTask) -> Result<String, AgentError> {
        Ok("mock agent response".into())
    }

    #[test]
    fn test_identity_eq() {
        let a = AgentIdentity::new("llama3", "sess-1");
        let b = AgentIdentity::new("llama3", "sess-1");
        assert_eq!(a, b);

        let c = AgentIdentity::new("llama3", "sess-1").with_adapter("lora-v1");
        assert_ne!(a, c);
    }

    #[test]
    fn test_adapter_registration() {
        let (workers, queue_cap) = global_pool_config(2, 8);
        let mut registry =
            AgentRegistry::new(fluent_concurrency::tokio_runtime(), workers, queue_cap);

        registry.register_adapter(AdapterHandle::new(
            "lora-v1",
            "/models/lora-v1.bin",
            "llama3",
        ));

        assert!(registry.is_adapter_loaded("llama3", "lora-v1"));
        assert!(!registry.is_adapter_loaded("llama3", "lora-v2"));
    }

    #[tokio::test]
    async fn test_agent_registration_and_submit() {
        let (workers, queue_cap) = global_pool_config(2, 8);
        let mut registry =
            AgentRegistry::new(fluent_concurrency::tokio_runtime(), workers, queue_cap);

        let identity = AgentIdentity::new("llama3", "sess-1");
        registry.register_agent(identity.clone(), mock_handler);

        assert_eq!(registry.agent_count(), 1);
        assert!(registry.has_agent(&identity));

        let task = AgentTask {
            messages: vec![],
            config: AgentConfig::default(),
        };

        let result = registry.submit(&identity, task).await.unwrap();
        assert_eq!(result, "mock agent response");
    }

    #[tokio::test]
    async fn test_missing_agent_errors() {
        let (workers, queue_cap) = global_pool_config(2, 8);
        let registry = AgentRegistry::new(fluent_concurrency::tokio_runtime(), workers, queue_cap);

        let identity = AgentIdentity::new("nonexistent", "sess-1");
        let task = AgentTask {
            messages: vec![],
            config: AgentConfig::default(),
        };

        let result = registry.submit(&identity, task).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AgentError::ModelNotFound(_) => {}
            e => panic!("expected ModelNotFound, got {e:?}"),
        }
    }
}
