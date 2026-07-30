use std::sync::{Arc, Weak};

use fluent_wvr::prelude::*;
use guidance_llm::client::{is_malformed_response, ChatBackend, LlmClient};
use guidance_llm::decomposer::Decomposer;
use guidance_llm::LlmConfig;
use internment::ArcIntern;

use crate::cache::reactor::QueueReactor;
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

fn make_output(result: &RoutingResult) -> WorkOutput {
    WorkOutput::ok_with_data(
        result.tier.to_string(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn routing_to_work_result(
    result: Result<RoutingResult, crate::error::CacheError>,
) -> Result<WorkOutput, WorkError> {
    match result {
        Ok(r) => Ok(make_output(&r)),
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
        let result_str = String::from_utf8_lossy(&result_bytes).to_string();
        Ok(make_output(&RoutingResult {
            query,
            result: result_str,
            tier: CacheTier::L2WasmWorkflow,
        }))
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
    pub embedder: Arc<dyn guidance_llm::EmbeddingProvider>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L4SemanticUnit {
    pub fn new(
        router: Arc<ParallelRouter>,
        embedder: Arc<dyn guidance_llm::EmbeddingProvider>,
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
// L4.5 — Local decomposition tier
// ---------------------------------------------------------------------------

pub struct L4_5DecomposeUnit {
    pub decomposer: Box<dyn Decomposer>,
    pub reactor: Weak<QueueReactor>,
    pub max_depth: u8,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L4_5DecomposeUnit {
    pub fn new(
        decomposer: Box<dyn Decomposer>,
        reactor: Weak<QueueReactor>,
        max_depth: u8,
    ) -> Self {
        Self {
            decomposer,
            reactor,
            max_depth,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l4_5")],
        }
    }
}

impl WorkUnit for L4_5DecomposeUnit {
    fn name(&self) -> &str {
        "coral.l4_5.decompose"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_query(ctx)?;
        let depth: u8 = ctx
            .metadata
            .get("depth")
            .and_then(|v| match v {
                fluent_wvr::MetadataValue::String(s) => s.parse().ok(),
                fluent_wvr::MetadataValue::Number(n) => Some(*n as u8),
                _ => None,
            })
            .unwrap_or(0);

        if depth >= self.max_depth {
            return Err(WorkError::Execution(format!(
                "max depth {} reached",
                self.max_depth
            )));
        }

        let reactor = self
            .reactor
            .upgrade()
            .ok_or_else(|| WorkError::Execution("reactor dropped".into()))?;

        let subtasks = self.decomposer.decompose(&query);
        let mut merged_result = String::new();
        for subtask in &subtasks {
            if let Ok(sub_result) = reactor.route_with_depth(subtask, depth + 1) {
                if !merged_result.is_empty() {
                    merged_result.push_str("\n---\n");
                }
                merged_result.push_str(&sub_result.result);
            }
        }
        if merged_result.is_empty() {
            return Err(WorkError::Execution("all subtasks returned empty".into()));
        }
        Ok(make_output(&RoutingResult {
            query,
            result: merged_result,
            tier: CacheTier::L4_5Decompose,
        }))
    }
}

impl FieldAccess for L4_5DecomposeUnit {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "L4_5DecomposeUnit has no configurable fields".into(),
        ))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "L4_5DecomposeUnit has no configurable fields".into(),
        ))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for L4_5DecomposeUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(L4_5DecomposeUnit);

// ---------------------------------------------------------------------------
// L5 — Frontier LLM fallback tier
// ---------------------------------------------------------------------------

pub struct L5FrontierUnit {
    pub config: LlmConfig,
    pub chat_backend: Option<Box<dyn ChatBackend>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl L5FrontierUnit {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            chat_backend: None,
            depends: query_deps(),
            provides: vec![ArcIntern::from("coral.tier.l5")],
        }
    }

    pub fn with_chat_backend(config: LlmConfig, backend: Box<dyn ChatBackend>) -> Self {
        Self {
            config,
            chat_backend: Some(backend),
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
        let anonymized = guidance_llm::anonymize::anonymize(&query);
        let messages = vec![
            guidance_llm::ChatMessage {
                role: "system".into(),
                content: "You are a helpful assistant. Answer concisely.".into(),
            },
            guidance_llm::ChatMessage {
                role: "user".into(),
                content: anonymized,
            },
        ];
        let response = if let Some(ref backend) = self.chat_backend {
            backend
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("frontier error: {e}")))?
        } else {
            let client = LlmClient::with_config(self.config.clone());
            client
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("frontier error: {e}")))?
        };
        if is_malformed_response(&response) {
            return Err(WorkError::Execution("malformed response".into()));
        }
        Ok(make_output(&RoutingResult {
            query,
            result: response,
            tier: CacheTier::L5Frontier,
        }))
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
    tiers: Vec<Arc<dyn Component>>,
}

impl TierRegistry {
    pub fn new(tiers: Vec<Arc<dyn Component>>) -> Self {
        Self { tiers }
    }

    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tiers.len()
    }

    pub fn execute(&self, query: &str, depth: u8) -> Result<RoutingResult, WorkError> {
        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert(QUERY_KEY.into(), query.to_string().into());
        if depth > 0 {
            ctx.metadata.insert("depth".into(), i64::from(depth).into());
        }

        let mut last_err = None;
        for tier in &self.tiers {
            match tier.execute(&ctx) {
                Ok(output) => {
                    return serde_json::from_value::<RoutingResult>(output.data)
                        .map_err(|e| WorkError::Execution(e.to_string()));
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or(WorkError::Execution("no tiers configured".into())))
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
