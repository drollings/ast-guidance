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
#[path = "../tests/knowledge.rs"]
mod tests;
