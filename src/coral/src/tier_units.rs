use std::sync::Arc;

use fluent_llm::client::{is_malformed_response, ChatBackend, LlmClient};
use fluent_llm::LlmConfig;
use fluent_llm::LlmRequestQueue;
use fluent_wvr::prelude::*;
use internment::ArcIntern;

use crate::cache_l1::{CacheTier, RoutingResult};
use crate::cache_router::ParallelRouter;
use crate::db::Library;
use crate::wasm_runtime::PluginPool;

// ---------------------------------------------------------------------------
// Query extraction helper
// ---------------------------------------------------------------------------

const QUERY_KEY: &str = "query";

fn query_deps() -> Vec<ArcIntern<str>> {
    vec![ArcIntern::from("coral.query")]
}

fn extract_query(ctx: &WorkContext) -> Result<String, WorkError> {
    ctx.metadata
        .get(QUERY_KEY)
        .and_then(|v| match v {
            fluent_wvr::MetadataValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| WorkError::Dependency("missing 'query' in WorkContext.metadata".into()))
}

fn make_output(result: &RoutingResult) -> Result<WorkOutput, WorkError> {
    WorkOutput::typed(result.tier.to_string(), result)
}

fn routing_to_work_result(
    result: Result<RoutingResult, crate::error::CacheError>,
) -> Result<WorkOutput, WorkError> {
    match result {
        Ok(r) => make_output(&r),
        Err(e) => Err(WorkError::Execution(e.to_string())),
    }
}

// ---------------------------------------------------------------------------
// L2 — WASM workflow tier
// ---------------------------------------------------------------------------

pub struct L2WasmUnit {
    pub runtime: Arc<dyn crate::wasm_runtime::WasmRuntime>,
    pub tool: fluent_types::WasmTool,
    pub library: Arc<Library>,
    pub pool: Arc<PluginPool>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
    schema: std::sync::Mutex<Option<serde_json::Value>>,
}

impl L2WasmUnit {
    pub fn new(
        runtime: Arc<dyn crate::wasm_runtime::WasmRuntime>,
        tool: fluent_types::WasmTool,
        library: Arc<Library>,
        pool: Arc<PluginPool>,
    ) -> Self {
        Self {
            runtime,
            tool,
            library,
            pool,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l2")],
            schema: std::sync::Mutex::new(None),
        }
    }

    fn load_schema(&self) -> Result<serde_json::Value, WorkError> {
        let mut guard = self.schema.lock().unwrap();
        if let Some(ref val) = *guard {
            return Ok(val.clone());
        }
        let plugin = self
            .pool
            .get_or_load(&self.tool.path)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let mut plugin_guard = plugin.lock().unwrap();
        let result = plugin_guard
            .call(b"get_schema")
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let value: serde_json::Value =
            serde_json::from_slice(&result).unwrap_or(serde_json::Value::Null);
        *guard = Some(value.clone());
        Ok(value)
    }
}

impl WorkUnit for L2WasmUnit {
    fn name(&self) -> &str {
        "coral.l2.wasm"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_query(ctx)?;
        let plugin = self
            .pool
            .get_or_load(&self.tool.path)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let result_bytes = {
            let mut guard = plugin.lock().unwrap();
            guard
                .call(query.as_bytes())
                .map_err(|e| WorkError::Execution(e.to_string()))?
        };
        let result_str = String::from_utf8_lossy(&result_bytes);
        make_output(&RoutingResult {
            query,
            result: result_str.into_owned(),
            tier: CacheTier::L2WasmWorkflow,
        })
    }
}

impl FieldAccess for L2WasmUnit {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "L2WasmUnit has no configurable fields".into(),
        ))
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "schema" => {
                let val = self
                    .load_schema()
                    .map_err(|e| FieldError::NotFound(e.to_string()))?;
                Ok(val.to_string())
            }
            "tool" => Ok(self.tool.name.to_string()),
            _ => Err(FieldError::NotFound(format!(
                "L2WasmUnit has no field '{name}'"
            ))),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["schema", "tool"]
    }
}

impl Describable for L2WasmUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "tool": { "type": "string", "description": "WASM tool name" }
            },
            "required": ["tool"]
        })
    }
}

impl_component!(L2WasmUnit);

// ---------------------------------------------------------------------------
// L3 — Graph traversal / keyword search tier
// ---------------------------------------------------------------------------

pub struct L3GraphUnit {
    pub router: Arc<ParallelRouter>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L3GraphUnit {
    pub fn new(router: Arc<ParallelRouter>) -> Self {
        Self {
            router,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l3")],
        }
    }
}

impl WorkUnit for L3GraphUnit {
    fn name(&self) -> &str {
        "coral.l3.graph"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_query(ctx)?;
        routing_to_work_result(self.router.route(&query))
    }
}

impl FieldAccess for L3GraphUnit {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "L3GraphUnit has no configurable fields".into(),
        ))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "L3GraphUnit has no configurable fields".into(),
        ))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for L3GraphUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(L3GraphUnit);

// ---------------------------------------------------------------------------
// L4 — Semantic (embedding-based) search tier
// ---------------------------------------------------------------------------

pub struct L4SemanticUnit {
    pub router: Arc<ParallelRouter>,
    pub embedder: Arc<dyn fluent_llm::EmbeddingProvider>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L4SemanticUnit {
    pub fn new(
        router: Arc<ParallelRouter>,
        embedder: Arc<dyn fluent_llm::EmbeddingProvider>,
    ) -> Self {
        Self {
            router,
            embedder,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l4")],
        }
    }
}

impl WorkUnit for L4SemanticUnit {
    fn name(&self) -> &str {
        "coral.l4.semantic"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_query(ctx)?;
        let emb = self
            .embedder
            .embed(&query)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        if emb.is_empty() {
            return Err(WorkError::Execution("empty embedding".into()));
        }
        routing_to_work_result(self.router.route_with_embedding(&query, &emb))
    }
}

impl FieldAccess for L4SemanticUnit {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "L4SemanticUnit has no configurable fields".into(),
        ))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "L4SemanticUnit has no configurable fields".into(),
        ))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for L4SemanticUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(L4SemanticUnit);

// ---------------------------------------------------------------------------
// L5 — Frontier LLM fallback tier
// ---------------------------------------------------------------------------

pub struct L5FrontierUnit {
    pub config: LlmConfig,
    pub chat_backend: Option<Box<dyn ChatBackend>>,
    pub queue: Option<Arc<LlmRequestQueue>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L5FrontierUnit {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            chat_backend: None,
            queue: None,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l5")],
        }
    }

    pub fn with_chat_backend(config: LlmConfig, backend: Box<dyn ChatBackend>) -> Self {
        Self {
            config,
            chat_backend: Some(backend),
            queue: None,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l5")],
        }
    }

    /// Attach a shared `LlmRequestQueue` (worker pool) so frontier calls flow
    /// through bounded workers instead of unbounded direct HTTP. The queue's
    /// handler is `fluent_llm::llm_queue::default_handler`, which preserves
    /// the `LlmConfig` fields (`timeout_ms`, `think`, `extra_body_params`,
    /// `debug`, `show_prompts`) exactly.
    pub fn with_queue(config: LlmConfig, queue: Arc<LlmRequestQueue>) -> Self {
        Self {
            config,
            chat_backend: None,
            queue: Some(queue),
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l5")],
        }
    }
}

impl WorkUnit for L5FrontierUnit {
    fn name(&self) -> &str {
        "coral.l5.frontier"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_query(ctx)?;
        let anonymized = fluent_llm::anonymize::anonymize(&query);
        let messages = vec![
            fluent_llm::ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant. Answer concisely.".into(),
            },
            fluent_llm::ChatMessage {
                role: "user".into(),
                content: anonymized,
            },
        ];
        let response = if let Some(ref backend) = self.chat_backend {
            backend
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("frontier error: {e}")))?
        } else {
            // When a shared `LlmRequestQueue` is attached, route through its
            // worker pool (bounded concurrency) via the queued `LlmClient`;
            // otherwise call the HTTP transport directly as before.
            let client = match &self.queue {
                Some(q) => LlmClient::with_queue_and_config(Arc::clone(q), self.config.clone()),
                None => LlmClient::with_config(self.config.clone()),
            };
            client
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("frontier error: {e}")))?
        };
        if is_malformed_response(&response) {
            return Err(WorkError::Execution("malformed response".into()));
        }
        make_output(&RoutingResult {
            query,
            result: response,
            tier: CacheTier::L5Frontier,
        })
    }
}

impl FieldAccess for L5FrontierUnit {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "L5FrontierUnit has no configurable fields".into(),
        ))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "L5FrontierUnit has no configurable fields".into(),
        ))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for L5FrontierUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(L5FrontierUnit);

// ---------------------------------------------------------------------------
// TierRegistry — sequential tier cascade
//
// No separate `TierRegistry` trait is needed. Each tier unit implements
// `WorkUnit` (and `Component`), and its `provides()` returns a unique
// identifier (`"coral.tier.l2"`, `"coral.tier.l3"`, etc.) that serves as
// the tier-identification mechanism.  `TierRegistry` is simply a
// `Vec<Arc<dyn Component>>` iterated in registration order — the first
// tier that returns `Ok` wins.  Adding a new tier means implementing
// `WorkUnit` + `Component` and pushing it into the registry; no additional
// trait or dispatch table is needed because `fluent_wvr::Component` +
// `TierRegistry` already provides the registry/execution abstraction.
// ---------------------------------------------------------------------------

pub struct TierRegistry {
    cascade: ComponentCascade,
}

impl TierRegistry {
    pub fn new(tiers: Vec<Arc<dyn Component>>) -> Self {
        Self {
            cascade: ComponentCascade::with_units(tiers),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cascade.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cascade.len()
    }

    pub fn execute(&self, query: &str, depth: u8) -> Result<RoutingResult, WorkError> {
        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert(QUERY_KEY.into(), query.to_string().into());
        if depth > 0 {
            ctx.metadata.insert("depth".into(), i64::from(depth).into());
        }

        if self.cascade.is_empty() {
            return Err(WorkError::Execution("no tiers configured".into()));
        }

        let output = self.cascade.execute_first_ok(&ctx)?;
        // Exactly one deserialize hop — the tier serialized once via
        // `WorkOutput::typed`, this consumes it back.
        output
            .data_take::<RoutingResult>()
            .map_err(|e| WorkError::Execution(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_registry_empty_returns_error() {
        let reg = TierRegistry::new(vec![]);
        assert!(reg.execute("test", 0).is_err());
    }

    #[test]
    fn test_l5_frontier_unit_name() {
        let unit = L5FrontierUnit::new(
            LlmConfig::new()
                .api_url("http://localhost:11434/v1".into())
                .model("llama3".into())
                .build(),
        );
        assert_eq!(unit.name(), "coral.l5.frontier");
        assert_eq!(unit.depends().len(), 1);
        assert_eq!(unit.provides().len(), 1);
        assert!(unit.queue.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_l5_frontier_unit_with_queue_routes_through_pool() {
        // When a shared `LlmRequestQueue` is attached, frontier calls flow
        // through the worker pool (`default_handler` HTTP transport) instead of
        // unbounded direct HTTP. An unreachable endpoint surfaces an error from
        // the queue, proving the wiring exists end-to-end.
        let queue = fluent_llm::llm_queue::build_default_queue(
            fluent_concurrency::tokio_runtime(),
            &fluent_concurrency::llm_queue::LlmQueueConfig {
                worker_count: 2,
                queue_capacity: 20,
            },
        );
        let unit = L5FrontierUnit::with_queue(
            LlmConfig::new()
                .api_url("http://127.0.0.1:1/v1".into())
                .model("test".into())
                .timeout_ms(100)
                .build(),
            queue,
        );
        assert_eq!(
            unit.queue.as_ref().expect("queue attached").worker_count(),
            2
        );

        let mut ctx = WorkContext::default();
        ctx.metadata.insert("query".into(), "hello".into());
        let output = unit.execute(&ctx);
        assert!(
            output.is_err(),
            "unreachable endpoint must surface an error from the queued path"
        );
    }

    #[test]
    fn test_l3_graph_unit_describe() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let router = Arc::new(ParallelRouter::new(lib, 10, 0.7, 4));
        let unit = L3GraphUnit::new(router);
        let desc = unit.describe();
        assert_eq!(desc["type"], "object");
    }

    #[test]
    fn test_extract_query_missing() {
        let ctx = WorkContext::default();
        assert!(extract_query(&ctx).is_err());
    }

    #[test]
    fn test_extract_query_present() {
        let mut ctx = WorkContext::default();
        ctx.metadata.insert("query".into(), "hello".into());
        assert_eq!(extract_query(&ctx).unwrap(), "hello");
    }
}
