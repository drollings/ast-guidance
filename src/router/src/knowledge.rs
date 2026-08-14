//! `KnowledgeCapability` implementation for the router's shared `ContentNodeStore`.
//!
//! The router server calls `ContentNodeStore` **directly** on the hot path — no
//! gating, no trait indirection. This trait impl is the boundary for
//! embedded/cross-crate consumers that reach the store through
//! `fluent_types::KnowledgeCapability`; each method asserts a router-owned
//! capability token (`RouterKnowledgeCapability`) in the current task-local,
//! mirroring `fluent_db::capability::check_db_capability`.

use fluent_types::{ContentNode, KnnHit, KnowledgeCapability, KnowledgeError, NodeId};
use fluent_wvr::capability::{check_capability, Capability};

use crate::ledger::LedgerError;
use crate::node_store::ContentNodeStore;

/// Router-owned capability token gating the `KnowledgeCapability` surface.
pub struct RouterKnowledgeCapability;

impl Capability for RouterKnowledgeCapability {
    fn name(&self) -> &'static str {
        "router.knowledge"
    }
}

fn denied() -> KnowledgeError {
    KnowledgeError::Other("missing capability: router.knowledge".into())
}

fn ledger_to_knowledge(e: &LedgerError) -> KnowledgeError {
    match e {
        LedgerError::NotFound(id) => KnowledgeError::NotFound(*id),
        other => KnowledgeError::Other(other.to_string()),
    }
}

impl KnowledgeCapability for ContentNodeStore {
    fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
        if check_capability(&RouterKnowledgeCapability).is_err() {
            return None;
        }
        self.snapshot(node_id)
    }

    fn get_session_nodes(&self, session_id: &str, limit: usize) -> Vec<ContentNode> {
        if check_capability(&RouterKnowledgeCapability).is_err() {
            return Vec::new();
        }
        self.get_session_nodes(session_id, limit)
            .unwrap_or_default()
    }

    fn insert_node(&self, node: &ContentNode) -> Result<NodeId, KnowledgeError> {
        if check_capability(&RouterKnowledgeCapability).is_err() {
            return Err(denied());
        }
        self.insert_node(node).map_err(|e| ledger_to_knowledge(&e))
    }

    fn knn_search(&self, embedding: &[f32], k: usize) -> Result<Vec<KnnHit>, KnowledgeError> {
        if check_capability(&RouterKnowledgeCapability).is_err() {
            return Err(denied());
        }
        Ok(self.knn_search(embedding, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr::capability::CURRENT_CAPS;
    use fluent_wvr::CapabilitySet;
    use std::sync::Arc;

    fn caps() -> CapabilitySet {
        CapabilitySet::new().with(RouterKnowledgeCapability)
    }

    fn temp_store() -> ContentNodeStore {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-knowledge-{}",
            common_core::hash::uuid_v4()
        ));
        let store = ContentNodeStore::open(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        store
    }

    #[test]
    fn deny_without_token() {
        let store = temp_store();
        // No CURRENT_CAPS task-local at all (sync, outside a runtime): every
        // trait method must deny.
        assert!(KnowledgeCapability::get_node(&store, NodeId::from_int(1)).is_none());
        assert!(
            KnowledgeCapability::get_session_nodes(&store, "sess", 10).is_empty(),
            "session nodes denied without token"
        );
        let node = fluent_types::ContentNode::default();
        assert!(matches!(
            KnowledgeCapability::insert_node(&store, &node),
            Err(KnowledgeError::Other(_))
        ));
        assert!(matches!(
            KnowledgeCapability::knn_search(&store, &[0.0; 4], 3),
            Err(KnowledgeError::Other(_))
        ));
    }

    #[tokio::test]
    async fn allow_with_token() {
        let store = Arc::new(temp_store());
        CURRENT_CAPS
            .scope(caps(), async {
                let id = store
                    .record_request("sess", "req-1", "hello knowledge")
                    .unwrap();
                // get_node: shared snapshot path.
                let node = KnowledgeCapability::get_node(&*store, id).unwrap();
                assert_eq!(node.lod[0], "hello knowledge");
                // get_session_nodes: interned index.
                let nodes = KnowledgeCapability::get_session_nodes(&*store, "sess", 10);
                assert_eq!(nodes.len(), 1);
                // insert_node: allocates + persists (session set so the
                // interned index picks it up).
                let mut new_node = fluent_types::ContentNode::default();
                new_node.lod = vec!["second".into()];
                new_node.session_id = Some("sess".into());
                let new_id = KnowledgeCapability::insert_node(&*store, &new_node).unwrap();
                assert!(new_id.as_int() > id.as_int());
                assert_eq!(
                    KnowledgeCapability::get_session_nodes(&*store, "sess", 10).len(),
                    2
                );
                // knn_search: brute force over embeddings.
                let hits = KnowledgeCapability::knn_search(&*store, &[0.0; 4], 3).unwrap();
                assert!(hits.is_empty());
            })
            .await;
    }

    #[tokio::test]
    async fn deny_when_token_absent_inside_runtime() {
        // Inside a runtime but with an empty capability set: still denied.
        let store = Arc::new(temp_store());
        CURRENT_CAPS
            .scope(CapabilitySet::new(), async {
                assert!(
                    KnowledgeCapability::get_session_nodes(&*store, "sess", 10).is_empty(),
                    "must deny without the token even inside a runtime"
                );
            })
            .await;
    }
}
