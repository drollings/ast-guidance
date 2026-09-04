use std::sync::Arc;

use fluent_concurrency::ladder::first_accept_in_order_sync;

use crate::error::CacheError;
use fluent_types::GraphNode;

use crate::cache_l1::{CacheTier, RoutingResult};
use crate::db::Library;

pub struct ParallelRouter {
    library: Arc<Library>,
    knn_k: usize,
    l4_threshold: f32,
    l3_max_depth: u8,
}

impl ParallelRouter {
    pub fn new(library: Arc<Library>, knn_k: usize, l4_threshold: f32, l3_max_depth: u8) -> Self {
        Self {
            library,
            knn_k,
            l4_threshold,
            l3_max_depth,
        }
    }

    pub fn route(&self, query: &str) -> Result<RoutingResult, CacheError> {
        enum Attempt { Keyword, Traverse }

        let rungs = [Attempt::Keyword, Attempt::Traverse];
        let out: Result<Option<RoutingResult>, CacheError> = first_accept_in_order_sync(
            rungs,
            |attempt| match attempt {
                Attempt::Keyword if !query.is_empty() => match self.library.keyword_search(query) {
                    Ok(results) if !results.is_empty() => Ok(Some(RoutingResult {
                        query: query.to_string(),
                        result: serde_json::to_string(&results).unwrap_or_default(),
                        tier: CacheTier::L3Graph,
                    })),
                    _ => Ok(None),
                },
                Attempt::Keyword => Ok(None),
                Attempt::Traverse => match self.traverse_all(self.l3_max_depth) {
                    Ok(nodes) if !nodes.is_empty() => Ok(Some(RoutingResult {
                        query: query.to_string(),
                        result: format!(
                            "Graph traversal: {} nodes at depth {}",
                            nodes.len(),
                            nodes.iter().map(|n| n.depth).max().unwrap_or(0)
                        ),
                        tier: CacheTier::L3Graph,
                    })),
                    Ok(_) => Ok(None),
                    Err(e) => Err(e),
                },
            },
            |_| false,
        );
        out.and_then(|opt| opt.ok_or(CacheError::Miss))
    }

    pub fn route_with_embedding(
        &self,
        query: &str,
        query_emb: &[f32],
    ) -> Result<RoutingResult, CacheError> {
        if query_emb.is_empty() {
            return Err(CacheError::Miss);
        }
        enum Attempt { KnnOrHybrid }
        let rungs = [Attempt::KnnOrHybrid];
        let out: Result<Option<RoutingResult>, CacheError> = first_accept_in_order_sync(
            rungs,
            |_| {
                let hits = if query.is_empty() {
                    self.library.knn_search(query_emb, self.knn_k, None)
                } else {
                    self.library.hybrid_search(query, Some(query_emb), self.knn_k)
                };
                match hits {
                    Ok(ref h) if !h.is_empty() && h[0].distance < self.l4_threshold => Ok(Some(RoutingResult {
                        query: query.to_string(),
                        result: format!("KNN hit: {}", h[0].name.as_str()),
                        tier: CacheTier::L4Semantic,
                    })),
                    _ => Ok(None),
                }
            },
            |_| false,
        );
        out.and_then(|opt| opt.ok_or(CacheError::Miss))
    }

    fn traverse_all(&self, max_depth: u8) -> Result<Vec<GraphNode>, CacheError> {
        let node_count = self.library.node_count().map_err(|_| CacheError::Miss)?;
        if node_count == 0 {
            return Ok(vec![]);
        }
        self.library
            .traverse_all_nodes(max_depth)
            .map_err(|_| CacheError::Miss)
    }
}

#[cfg(test)]
mod tests {
    use fluent_types::ContentNode;

    use super::*;
    use crate::cache_l1::CacheTier;
    use crate::tests::common::make_node;

    fn make_router() -> ParallelRouter {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        ParallelRouter::new(lib, 10, 0.7, 4)
    }

    fn insert_test_node(lib: &Arc<Library>, name: &str, source: &str) {
        let node = make_node(name, source);
        lib.insert_node(&node).expect("insert node");
    }

    fn insert_edge(lib: &Arc<Library>, from: &str, to: &str) {
        let from_id = lib
            .find_node_by_name(from)
            .expect("find")
            .expect("from node");
        let to_id = lib.find_node_by_name(to).expect("find").expect("to node");
        lib.insert_edge(from_id, to_id, "depends", 1.0)
            .expect("insert edge");
    }

    #[test]
    fn test_router_empty_db_returns_miss() {
        let router = make_router();
        let result = router.route("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_router_keyword_search_hit() {
        let router = make_router();
        insert_test_node(
            &router.library,
            "zig_compiler",
            "Zig compiler documentation",
        );
        let result = router.route("zig");
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.result.contains("zig_compiler"));
    }

    #[test]
    fn test_traverse_all_works() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        insert_test_node(&lib, "root", "root source");
        insert_test_node(&lib, "child", "child source");
        insert_test_node(&lib, "grandchild", "grandchild source");
        insert_edge(&lib, "root", "child");
        insert_edge(&lib, "child", "grandchild");

        let router = ParallelRouter::new(lib, 10, 0.7, 4);
        let nodes = router.traverse_all(3).expect("traverse_all");
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_route_with_embedding_hit() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let emb = vec![0.1, 0.2, 0.3, 0.4];
        let node = ContentNode {
            embedding: Some(emb.clone()),
            ..make_node("target_node", "source")
        };
        lib.insert_node(&node).expect("insert");

        let router = ParallelRouter::new(Arc::clone(&lib), 10, 0.7, 4);
        let query_emb = vec![0.1, 0.2, 0.3, 0.4];
        let result = router.route_with_embedding("target", &query_emb);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tier, CacheTier::L4Semantic);
    }

    // ── M0 characterization: tier-order harness ──────────────────────────────

    #[test]
    fn test_route_keyword_precedes_traverse() {
        let router = make_router();
        insert_test_node(&router.library, "zig_compiler", "Zig compiler docs");
        let result = router.route("zig").expect("route zig");
        assert!(result.result.contains("zig_compiler"), "keyword hit should contain node name");
        assert!(!result.result.contains("Graph traversal"), "keyword should precede traverse");
        assert_eq!(result.tier, CacheTier::L3Graph);
    }

    #[test]
    fn test_route_empty_query_goes_to_traverse() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let n = make_node("root", "src");
        lib.insert_node(&n).expect("insert");
        let router = ParallelRouter::new(Arc::clone(&lib), 10, 0.7, 4);
        let result = router.route("").expect("empty query should hit traverse");
        assert!(result.result.contains("Graph traversal"));
        // Empty DB → Miss
        let empty_router = make_router();
        let miss = empty_router.route("");
        assert!(matches!(miss, Err(crate::error::CacheError::Miss)));
    }

    #[test]
    fn test_route_traverse_fallback_after_keyword_miss() {
        let router = make_router();
        insert_test_node(&router.library, "alpha", "source alpha");
        let result = router.route("nonexistent-xyz").expect("should fallback to traverse");
        assert!(result.result.contains("Graph traversal"));
        assert_eq!(result.tier, CacheTier::L3Graph);
    }

    #[test]
    fn test_route_all_miss_returns_miss() {
        let router = make_router();
        assert!(matches!(router.route("test"), Err(crate::error::CacheError::Miss)));
        assert!(matches!(router.route_with_embedding("x", &[0.1, 0.2]), Err(crate::error::CacheError::Miss)));
    }

    #[test]
    fn test_route_with_embedding_empty_emb_returns_miss() {
        let router = make_router();
        let result = router.route_with_embedding("x", &[]);
        assert!(matches!(result, Err(crate::error::CacheError::Miss)));
    }

    #[test]
    fn test_route_with_embedding_threshold_boundary() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let emb = vec![1.0, 0.0];
        let node = ContentNode { embedding: Some(emb.clone()), ..make_node("threshold_node", "src") };
        lib.insert_node(&node).expect("insert");
        // distance 0 < 0.7 => hit
        let router_hit = ParallelRouter::new(Arc::clone(&lib), 10, 0.7, 4);
        let hit = router_hit.route_with_embedding("q", &emb).expect("should hit at 0.7");
        assert_eq!(hit.tier, CacheTier::L4Semantic);
        // distance 0 < 0.0 is false => miss (strict <)
        let router_miss = ParallelRouter::new(Arc::clone(&lib), 10, 0.0, 4);
        let miss = router_miss.route_with_embedding("q", &emb);
        assert!(matches!(miss, Err(crate::error::CacheError::Miss)), "threshold strict < : 0 < 0 must be Miss");
    }

    #[test]
    fn test_route_with_embedding_empty_query_uses_knn_not_hybrid() {
        // Both paths should succeed when embedding matches, regardless of query string
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let emb = vec![0.5, 0.5];
        let node = ContentNode { embedding: Some(emb.clone()), ..make_node("knn_node", "source knn") };
        lib.insert_node(&node).expect("insert");
        let router = ParallelRouter::new(Arc::clone(&lib), 10, 0.7, 4);
        // empty query → knn_search branch
        let r1 = router.route_with_embedding("", &emb).expect("empty query knn");
        assert_eq!(r1.tier, CacheTier::L4Semantic);
        // non-empty query → hybrid_search branch
        let r2 = router.route_with_embedding("knn", &emb).expect("non-empty hybrid");
        assert_eq!(r2.tier, CacheTier::L4Semantic);
    }

    #[test]
    fn test_l4_sweep_fpr_bound_at_0_7() {
        // Synthetic near-neighbor positives (distance ~0) and far negatives (large L2).
        // For calibration we map score = -distance so score >= -threshold => distance <= threshold.
        struct Case { distance: f64, label: bool }
        let cases: Vec<Case> = vec![
            // Positives: near neighbors (should fire at 0.7)
            Case { distance: 0.0, label: true },
            Case { distance: 0.05, label: true },
            Case { distance: 0.10, label: true },
            Case { distance: 0.20, label: true },
            Case { distance: 0.30, label: true },
            Case { distance: 0.50, label: true },
            Case { distance: 0.60, label: true },
            Case { distance: 0.65, label: true },
            // Negatives: far (must NOT fire at 0.7)
            Case { distance: 1.5, label: false },
            Case { distance: 2.0, label: false },
            Case { distance: 1.2, label: false },
            Case { distance: 1.0, label: false },
            Case { distance: 0.85, label: false },
            Case { distance: 0.90, label: false },
            Case { distance: 1.8, label: false },
            Case { distance: 1.1, label: false },
            Case { distance: 0.80, label: false },
            Case { distance: 2.5, label: false },
        ];
        // Sweep -threshold (negated distance) to use calibration's score >= threshold semantics.
        let thresholds_pos: Vec<f64> = (0..=8).map(|i| 0.5 + i as f64 * 0.05).collect(); // 0.5..0.9
        let thresholds_neg: Vec<f64> = thresholds_pos.iter().map(|t| -t).collect();
        let reports = common_core::calibration::sweep_thresholds(&cases, |c| -c.distance, |c| c.label, &thresholds_neg);
        common_core::calibration::emit_markdown_artifact("coral_l4_threshold", &reports);
        let threshold_neg = -0.7;
        let report = common_core::calibration::calibrate_threshold(&cases, |c| -c.distance, |c| c.label, threshold_neg);
        assert!(report.passes_gate(), "L4 0.7 must pass gate: precision {} FPR {} \n{}", report.precision, report.fpr, common_core::calibration::render_markdown_table(&reports));
    }
}
