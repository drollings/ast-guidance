use std::sync::Arc;

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
        if !query.is_empty() {
            if let Ok(results) = self.library.keyword_search(query) {
                if !results.is_empty() {
                    return Ok(RoutingResult {
                        query: query.to_string(),
                        result: serde_json::to_string(&results).unwrap_or_default(),
                        tier: CacheTier::L3Graph,
                    });
                }
            }
        }

        if let Ok(nodes) = self.traverse_all(self.l3_max_depth) {
            if !nodes.is_empty() {
                return Ok(RoutingResult {
                    query: query.to_string(),
                    result: format!(
                        "Graph traversal: {} nodes at depth {}",
                        nodes.len(),
                        nodes.iter().map(|n| n.depth).max().unwrap_or(0)
                    ),
                    tier: CacheTier::L3Graph,
                });
            }
        }

        Err(CacheError::Miss)
    }

    pub fn route_with_embedding(
        &self,
        query: &str,
        query_emb: &[f32],
    ) -> Result<RoutingResult, CacheError> {
        if !query_emb.is_empty() {
            let hits = if query.is_empty() {
                self.library.knn_search(query_emb, self.knn_k, None)
            } else {
                self.library
                    .hybrid_search(query, Some(query_emb), self.knn_k)
            };
            if let Ok(ref hits) = hits {
                if !hits.is_empty() && hits[0].distance < self.l4_threshold {
                    return Ok(RoutingResult {
                        query: query.to_string(),
                        result: format!("KNN hit: {}", hits[0].name.as_str()),
                        tier: CacheTier::L4Semantic,
                    });
                }
            }
        }
        Err(CacheError::Miss)
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
}
