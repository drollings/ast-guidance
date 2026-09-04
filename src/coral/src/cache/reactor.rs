use std::sync::Arc;

use crate::error::CacheError;
use crate::wasm_runtime::WasmRuntime;
use bon::Builder;
use common_core::hash::content_hash_with_model;
use common_core::metrics::LatencyHistogram;
use fluent_llm::decomposer::Decomposer;
use fluent_llm::LlmConfig;
use fluent_types::{ContentNode, NodeId, WasmTool};

use crate::cache_l1::{CacheTier, L1Cache, RoutingResult};
use crate::cache_router::ParallelRouter;
use crate::db::Library;
use crate::tier_units::{L2WasmUnit, L3GraphUnit, L4SemanticUnit, L5FrontierUnit, TierRegistry};
use crate::wasm_runtime::PluginPool;
use fluent_wvr::prelude::*;

use super::stats::CoralStats;

/// Wrap a tier unit in the standard name-override + metrics adapter stack.
fn wrap_tier(
    unit: Arc<dyn Component>,
    name: &'static str,
    hist: Arc<LatencyHistogram>,
) -> Arc<dyn Component> {
    let adapted = ComponentAdapter::new(unit).with_name_override(name);
    Arc::new(Instrumented::with_metrics(adapted, name, hist))
}

#[derive(Builder)]
pub struct QueueReactorCreateArgs {
    pub library: Arc<Library>,

    #[builder(default)]
    pub l1_cache: L1Cache,

    #[builder(default = 10)]
    pub knn_k: usize,

    #[builder(default = 0.7)]
    pub l4_threshold: f32,

    #[builder(default = 4)]
    pub l3_max_depth: u8,

    pub decomposer: Option<Box<dyn Decomposer>>,

    pub frontier_config: Option<LlmConfig>,

    /// Opt-in worker-pool sizing for the L5 frontier tier. When set, frontier
    /// calls flow through a shared `LlmRequestQueue` with this many workers
    /// (bounded concurrency) instead of unbounded direct HTTP. `None`
    /// (default) preserves the direct-HTTP path.
    pub frontier_workers: Option<usize>,

    #[builder(default = 3)]
    pub max_depth: u8,

    pub embedder: Option<Arc<dyn fluent_llm::EmbeddingProvider>>,

    pub wasm_runtime: Option<Arc<dyn WasmRuntime>>,

    #[builder(default = 16)]
    pub max_plugins: usize,
}

pub struct QueueReactor {
    pub library: Arc<Library>,
    pub l1_cache: L1Cache,
    tier_registry: TierRegistry,
    histograms: Vec<Arc<LatencyHistogram>>,
    pub knn_k: usize,
    pub l4_threshold: f32,
    pub l3_max_depth: u8,
    pub decomposer: Option<Box<dyn Decomposer>>,
    pub frontier_config: Option<LlmConfig>,
    pub max_depth: u8,
    pub embedder: Option<Arc<dyn fluent_llm::EmbeddingProvider>>,
    pub wasm_runtime: Option<Arc<dyn WasmRuntime>>,
    pub plugin_pool: Option<Arc<PluginPool>>,
}

impl QueueReactor {
    pub fn new(args: QueueReactorCreateArgs) -> Self {
        let router = Arc::new(ParallelRouter::new(
            Arc::clone(&args.library),
            args.knn_k,
            args.l4_threshold,
            args.l3_max_depth,
        ));

        // Build the tier registry from configured tiers
        let mut tiers: Vec<Arc<dyn Component>> = Vec::new();
        let mut histograms: Vec<Arc<LatencyHistogram>> = Vec::new();

        // L2: WASM — create pool and register if runtime is available
        let plugin_pool: Option<Arc<PluginPool>> = args
            .wasm_runtime
            .as_ref()
            .map(|rt| Arc::new(PluginPool::new(Arc::clone(rt), args.max_plugins)));

        if let (Some(ref rt), Some(ref pool)) = (&args.wasm_runtime, &plugin_pool) {
            if let Ok(tools) = args.library.find_wasm_tools_by_capability("query") {
                for tool in tools.into_iter().take(1) {
                    let hist = Arc::new(LatencyHistogram::new());
                    let unit = L2WasmUnit::new(
                        Arc::clone(rt),
                        tool,
                        Arc::clone(&args.library),
                        Arc::clone(pool),
                    );
                    tiers.push(wrap_tier(
                        Arc::new(unit),
                        "coral.l2.wasm",
                        Arc::clone(&hist),
                    ));
                    histograms.push(hist);
                }
            }
        }

        // L3: Graph traversal (always available) — wrapped with metrics
        {
            let hist = Arc::new(LatencyHistogram::new());
            let unit = L3GraphUnit::new(Arc::clone(&router));
            tiers.push(wrap_tier(
                Arc::new(unit),
                "coral.l3.graph",
                Arc::clone(&hist),
            ));
            histograms.push(hist);
        }

        // L4: Semantic search (only if embedder is configured) — wrapped with metrics
        if let Some(ref embedder) = args.embedder {
            let hist = Arc::new(LatencyHistogram::new());
            let unit = L4SemanticUnit::new(Arc::clone(&router), Arc::clone(embedder));
            tiers.push(wrap_tier(
                Arc::new(unit),
                "coral.l4.semantic",
                Arc::clone(&hist),
            ));
            histograms.push(hist);
        }

        // L5: Frontier LLM (only if frontier config is provided) — wrapped with metrics
        if let Some(ref frontier) = args.frontier_config {
            let hist = Arc::new(LatencyHistogram::new());
            let unit = match args.frontier_workers {
                // Route frontier calls through a shared worker pool
                // (`LlmRequestQueue` with the `default_handler` transport)
                // sized from the reactor config — opt-in, non-regressing.
                Some(workers) => {
                    let workers = workers.max(1);
                    let queue = fluent_llm::llm_queue::build_default_queue(
                        fluent_concurrency::tokio_runtime(),
                        &fluent_llm::protocol::LlmQueueConfig {
                            worker_count: workers,
                            queue_capacity: workers * 10,
                        },
                    );
                    L5FrontierUnit::with_queue(frontier.clone(), queue)
                }
                None => L5FrontierUnit::new(frontier.clone()),
            };
            tiers.push(wrap_tier(
                Arc::new(unit),
                "coral.l5.frontier",
                Arc::clone(&hist),
            ));
            histograms.push(hist);
        }

        let tier_registry = TierRegistry::new(tiers);

        Self {
            library: args.library,
            l1_cache: args.l1_cache,
            tier_registry,
            histograms,
            knn_k: args.knn_k,
            l4_threshold: args.l4_threshold,
            l3_max_depth: args.l3_max_depth,
            decomposer: args.decomposer,
            frontier_config: args.frontier_config,
            max_depth: args.max_depth,
            embedder: args.embedder,
            wasm_runtime: args.wasm_runtime,
            plugin_pool,
        }
    }

    fn set_l1(&self, query: &str, result: &Arc<RoutingResult>) {
        self.l1_cache.set(query.to_string(), Arc::clone(result));
    }

    pub fn route(&self, query: &str) -> Result<Arc<RoutingResult>, CacheError> {
        self.route_with_depth(query, 0)
    }

    pub(crate) fn route_with_depth(
        &self,
        query: &str,
        depth: u8,
    ) -> Result<Arc<RoutingResult>, CacheError> {
        // L1: Memory cache — check at any depth (subtask results are cached too)
        if let Some(cached) = self.l1_cache.get(query) {
            return Ok(cached);
        }

        // Execute tiers via the registry (L3 → L4 → L5 cascade)
        match self.tier_registry.execute(query, depth) {
            Ok(result) => {
                let result = Arc::new(result);
                if let Err(e) = self.persist_solution(query, &result, depth) {
                    tracing::warn!(query = %query, error = %e, "persist_solution failed");
                }
                self.set_l1(query, &result);
                return Ok(result);
            }
            Err(_work_err) => {
                // All non-L4.5 tiers missed or failed; try L4.5 decomposition if configured
            }
        }

        // L4.5: Local decomposition (recursive — cannot go through registry)
        if let Some(ref decomposer) = self.decomposer {
            if depth < self.max_depth {
                let subtasks = decomposer.decompose(query);
                let mut merged_result = String::new();
                for subtask in &subtasks {
                    if let Ok(sub_result) = self.route_with_depth(subtask, depth + 1) {
                        if !merged_result.is_empty() {
                            merged_result.push_str("\n---\n");
                        }
                        merged_result.push_str(&sub_result.result);
                    }
                }
                if !merged_result.is_empty() {
                    let result = Arc::new(RoutingResult {
                        query: query.to_string(),
                        result: merged_result,
                        tier: CacheTier::L4_5Decompose,
                    });
                    if let Err(e) = self.persist_solution(query, &result, depth) {
                        tracing::warn!(query = %query, error = %e, "persist_solution failed");
                    }
                    self.set_l1(query, &result);
                    return Ok(result);
                }
            }
        }

        Err(CacheError::Miss)
    }

    pub fn persist_solution(
        &self,
        query: &str,
        result: &RoutingResult,
        depth: u8,
    ) -> Result<NodeId, CacheError> {
        let hash_bytes = content_hash_with_model(query, "solution");
        let hash_id = i64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        // An embedder failure is now observable via `CacheError::Embedding`
        // instead of being silently swallowed; a missing embedder (feature not
        // configured) still persists the node without an embedding.
        let embedding = match self.embedder.as_ref() {
            Some(e) => match e.embed(&result.result) {
                Ok(v) if !v.is_empty() => Some(v),
                Ok(_) => None,
                Err(e) => {
                    return Err(CacheError::Embedding(format!("query={query}: {e}")));
                }
            },
            None => None,
        };
        let name = if depth > 0 {
            format!("solution:subtask:{query}")
        } else {
            format!("solution:{query}")
        };
        let node = ContentNode {
            id: Some(NodeId::from_int(hash_id)),
            name: name.into(),
            source: query.to_string(),
            lod: vec![result.result.clone(), query.to_string()],
            embedding,
            capabilities: None,
            ..Default::default()
        };
        self.library
            .insert_node(&node)
            .map_err(|e| CacheError::PersistFailed(format!("query={query}: {e}")))
    }

    pub fn find_wasm_tool(&self, capability: &str) -> Option<WasmTool> {
        self.library
            .find_wasm_tools_by_capability(capability)
            .ok()?
            .into_iter()
            .next()
    }

    #[must_use]
    pub fn coral_stats(&self) -> CoralStats {
        let mut total_count = 0u64;
        let mut total_sum_ms = 0u64;
        for h in &self.histograms {
            total_count += h.count();
            total_sum_ms += h.sum_ms();
        }
        // p50/p99 are aggregated across ALL tier histograms (weighted
        // by count) instead of reflecting only the first tier.
        let refs: Vec<&LatencyHistogram> = self.histograms.iter().map(Arc::as_ref).collect();
        let p50 = LatencyHistogram::aggregate(&refs, 50.0);
        let p99 = LatencyHistogram::aggregate(&refs, 99.0);
        CoralStats {
            tier_count: self.histograms.len(),
            total_count,
            total_sum_ms,
            p50_ms: p50,
            p99_ms: p99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_stubs::{StubChatBackend, StubDecomposer, StubEmbedder};
    use crate::tests::common::make_node;
    use fluent_types::ContentNode;
    use std::collections::HashMap;

    fn make_reactor(lib: Arc<Library>) -> QueueReactor {
        let args = QueueReactorCreateArgs::builder().library(lib).build();
        QueueReactor::new(args)
    }

    fn insert_node_with_embedding(lib: &Library, name: &str, embedding: Vec<f32>) {
        let node = ContentNode {
            lod: vec![format!("lod for {name}")],
            embedding: Some(embedding),
            ..make_node(name, &format!("source for {name}"))
        };
        lib.insert_node(&node).expect("insert node");
    }

    #[test]
    fn test_reactor_l1_hit() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);

        let cached = Arc::new(RoutingResult {
            query: "cached_q".into(),
            result: "cached_result".into(),
            tier: CacheTier::L1Memory,
        });
        reactor.l1_cache.set("cached_q".into(), cached);

        let result = reactor.route("cached_q").expect("should hit L1");
        assert_eq!(result.tier, CacheTier::L1Memory);
        assert_eq!(result.result, "cached_result");
    }

    #[test]
    fn test_reactor_l1_miss_falls_through_to_tiers() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let node = make_node("test_node", "source");
        lib.insert_node(&node).expect("insert");

        let reactor = make_reactor(lib);
        let result = reactor.route("test_node").expect("should route via L3");
        assert_eq!(result.tier, CacheTier::L3Graph);
        assert!(result.result.contains("test_node"));
    }

    #[test]
    fn test_reactor_route_uses_l3_when_data_exists() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let node = make_node("zig_compiler", "Zig compiler documentation");
        lib.insert_node(&node).expect("insert");

        let reactor = make_reactor(Arc::clone(&lib));
        let results = lib.keyword_search("zig").expect("keyword_search");
        assert!(!results.is_empty(), "keyword_search should find the node");
        let result = reactor.route("zig").expect("should find via L3");
        assert_eq!(result.tier, CacheTier::L3Graph);
        assert!(result.result.contains("zig_compiler"));
    }

    #[test]
    fn test_reactor_l4_semantic_routing() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let node_emb = vec![1.0, 0.0, 0.0, 0.0];
        insert_node_with_embedding(&lib, "embedded_node", node_emb);

        let embedder: Arc<dyn fluent_llm::EmbeddingProvider> = Arc::new(StubEmbedder::new(4));
        let router = Arc::new(ParallelRouter::new(lib, 10, 0.7, 4));
        let unit = crate::tier_units::L4SemanticUnit::new(router, embedder);

        let mut ctx = fluent_wvr::WorkContext::default();
        ctx.metadata.insert("query".into(), "aaaa".into());
        let output = unit.execute(&ctx).expect("L4 should succeed");
        assert_eq!(output.message, "L4");
    }

    #[test]
    fn test_reactor_l5_frontier_not_configured() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);
        assert!(reactor.frontier_config.is_none());
    }

    #[test]
    fn test_reactor_l5_frontier_error_propagation() {
        let config = LlmConfig::new()
            .api_url("http://localhost:11434/v1".into())
            .model("test".into())
            .build();
        let backend: Box<dyn fluent_llm::client::ChatBackend> =
            Box::new(StubChatBackend::always_err("simulated HTTP failure"));

        let unit = crate::tier_units::L5FrontierUnit::with_chat_backend(config, backend);
        let hist = Arc::new(LatencyHistogram::new());
        let wrapped = Instrumented::with_metrics(unit, "coral.l5.frontier", Arc::clone(&hist));
        let tier_registry = TierRegistry::new(vec![Arc::new(wrapped)]);

        let output = tier_registry.execute("unfindable_query_xyz", 0);
        match output {
            Ok(_) => panic!("expected error from L5 stub"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("frontier error"),
                    "expected frontier error, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn test_reactor_l4_5_decompose_not_configured() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);
        assert!(reactor.decomposer.is_none());
    }

    #[test]
    fn test_reactor_l4_5_decomposition() {
        // Production decompose-merge path: an empty library makes the L3
        // tier miss (keyword search + traverse_all both empty), so the L4.5
        // block in `route_with_depth` fires; subtasks resolve from the L1
        // cache, and the merged result is returned as `L4_5Decompose`.
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let sub_alpha = Arc::new(RoutingResult {
            query: "subtask_alpha".into(),
            result: "result from subtask_alpha".into(),
            tier: CacheTier::L3Graph,
        });
        let sub_beta = Arc::new(RoutingResult {
            query: "subtask_beta".into(),
            result: "result from subtask_beta".into(),
            tier: CacheTier::L3Graph,
        });

        let mut responses = HashMap::new();
        responses.insert(
            "complex_parent_query".to_string(),
            vec!["subtask_alpha".to_string(), "subtask_beta".to_string()],
        );
        let decomposer: Box<dyn fluent_llm::decomposer::Decomposer> =
            Box::new(StubDecomposer::new(responses));

        let args = QueueReactorCreateArgs::builder()
            .library(lib)
            .decomposer(decomposer)
            .max_depth(3)
            .build();
        let reactor = Arc::new(QueueReactor::new(args));
        reactor.set_l1("subtask_alpha", &sub_alpha);
        reactor.set_l1("subtask_beta", &sub_beta);

        let result = reactor
            .route_with_depth("complex_parent_query", 0)
            .expect("L4.5 should succeed");
        assert_eq!(result.tier, CacheTier::L4_5Decompose);
        assert!(
            result.result.contains("subtask_alpha"),
            "result should contain subtask_alpha, got: {}",
            result.result
        );
        assert!(
            result.result.contains("subtask_beta"),
            "result should contain subtask_beta, got: {}",
            result.result
        );
    }

    #[test]
    fn test_reactor_subtask_recursion_l1_backfill() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);

        let sub_x_result = Arc::new(RoutingResult {
            query: "sub_x".into(),
            result: "result_x".into(),
            tier: CacheTier::L3Graph,
        });
        let sub_y_result = Arc::new(RoutingResult {
            query: "sub_y".into(),
            result: "result_y".into(),
            tier: CacheTier::L3Graph,
        });
        reactor.set_l1("sub_x", &sub_x_result);
        reactor.set_l1("sub_y", &sub_y_result);

        let from_x = reactor.l1_cache.get("sub_x").expect("sub_x in L1");
        assert_eq!(from_x.result, "result_x");
        let from_y = reactor.l1_cache.get("sub_y").expect("sub_y in L1");
        assert_eq!(from_y.result, "result_y");

        let parent_result = Arc::new(RoutingResult {
            query: "parent_q".into(),
            result: "result_x\n---\nresult_y".into(),
            tier: CacheTier::L4_5Decompose,
        });
        reactor.set_l1("parent_q", &parent_result);

        let r = reactor.route("parent_q").expect("should hit L1");
        assert!(r.result.contains("result_x"));
        assert!(r.result.contains("result_y"));
    }

    #[test]
    fn test_reactor_persist_solution() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let embedder: Arc<dyn fluent_llm::EmbeddingProvider> = Arc::new(StubEmbedder::new(4));
        let args = QueueReactorCreateArgs::builder()
            .library(lib.clone())
            .embedder(embedder)
            .build();
        let reactor = QueueReactor::new(args);

        let result = RoutingResult {
            query: "test".into(),
            result: "test_result".into(),
            tier: CacheTier::L3Graph,
        };
        let node_id = reactor
            .persist_solution("test", &result, 0)
            .expect("persist should succeed");

        let found_id = lib
            .find_node_by_name("solution:test")
            .expect("find")
            .expect("node exists");
        assert_eq!(found_id, node_id);

        let node = lib.get_node(found_id).expect("get node").expect("node");
        let emb = node.embedding.expect("node should have embedding");
        assert_eq!(emb.len(), 4, "embedding should have 4 dimensions");
        let expected = reactor
            .embedder
            .as_ref()
            .unwrap()
            .embed("test_result")
            .unwrap();
        assert_eq!(emb, expected, "embedding should match stub embedder output");
    }

    #[test]
    fn test_persist_solution_subtask_naming() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib.clone());

        let result = RoutingResult {
            query: "subtask_q".into(),
            result: "subtask_result".into(),
            tier: CacheTier::L3Graph,
        };

        reactor
            .persist_solution("subtask_q", &result, 0)
            .expect("persist");
        let node = lib
            .find_node_by_name("solution:subtask_q")
            .expect("find")
            .expect("node exists at depth 0");

        let subtask_result = RoutingResult {
            query: "child_q".into(),
            result: "child_result".into(),
            tier: CacheTier::L3Graph,
        };
        reactor
            .persist_solution("child_q", &subtask_result, 1)
            .expect("persist subtask");
        let subtask_node = lib
            .find_node_by_name("solution:subtask:child_q")
            .expect("find")
            .expect("node exists at depth 1");
        assert_ne!(node, subtask_node);
    }

    #[test]
    fn test_coral_stats_zero_before_any_route() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);
        let stats = reactor.coral_stats();
        assert_eq!(stats.total_count, 0);
        assert!(
            stats.tier_count > 0,
            "should have at least one tier histogram"
        );
    }

    #[test]
    fn test_coral_stats_records_l3_hit() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let node = make_node("test_node", "source");
        lib.insert_node(&node).expect("insert");

        let reactor = make_reactor(Arc::clone(&lib));
        let _ = reactor.route("test_node");

        let stats = reactor.coral_stats();
        assert!(
            stats.total_count > 0,
            "after a route, total_count should be > 0, got {}",
            stats.total_count
        );
    }

    #[test]
    fn test_subtask_results_backfill_l1() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);

        let cached = Arc::new(RoutingResult {
            query: "cached_subtask".into(),
            result: "subtask answer".into(),
            tier: CacheTier::L3Graph,
        });
        reactor
            .l1_cache
            .set("cached_subtask".into(), Arc::clone(&cached));

        let result = reactor
            .route_with_depth("cached_subtask", 1)
            .expect("should hit L1 at depth 1");
        assert_eq!(result.result, "subtask answer");

        let result2 = reactor.route("cached_subtask").expect("should hit L1");
        assert_eq!(result2.result, "subtask answer");

        let new_result = Arc::new(RoutingResult {
            query: "new_query".into(),
            result: "new_result".into(),
            tier: CacheTier::L3Graph,
        });
        reactor.set_l1("new_query", &new_result);
        let from_cache = reactor.l1_cache.get("new_query").expect("should be cached");
        assert_eq!(from_cache.result, "new_result");
    }

    #[test]
    fn test_cache_tier_display() {
        assert_eq!(CacheTier::L1Memory.to_string(), "L1");
        assert_eq!(CacheTier::L2WasmWorkflow.to_string(), "L2");
        assert_eq!(CacheTier::L3Graph.to_string(), "L3");
        assert_eq!(CacheTier::L4Semantic.to_string(), "L4");
        assert_eq!(CacheTier::L4_5Decompose.to_string(), "L4.5");
        assert_eq!(CacheTier::L5Frontier.to_string(), "L5");
    }

    #[test]
    fn test_parallel_router_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParallelRouter>();
    }

    #[test]
    fn test_reactor_tier_registry_has_l3() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let reactor = make_reactor(lib);
        assert!(!reactor.tier_registry.is_empty());
    }

    #[test]
    fn component_adapter_name_override_reports_new_name() {
        use fluent_wvr_testutil::StubComponent;

        let stub = StubComponent::ok("inner_unit").with_handler(|_| {
            Ok(fluent_wvr::WorkOutput::ok("from_stub"))
        });
        let adapted = ComponentAdapter::new(Arc::new(stub)).with_name_override("coral.l3.graph");
        assert_eq!(adapted.name(), "coral.l3.graph");

        let result = adapted
            .execute(&fluent_wvr::WorkContext::default())
            .unwrap();
        assert_eq!(result.message, "from_stub");
    }
}
