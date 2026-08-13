//! KnowledgeCapability — the cross-crate read/write vocabulary for a
//! ContentNode store.
//!
//! This is **pure vocabulary**: it must stay leaf-clean (no `fluent-wvr`
//! dependency, no `fluent-db`, no `fluent-router`). Its signatures use only
//! `fluent-types` types — `ContentNode`, `NodeId`, `KnnHit` — plus
//! `Option`/`Result`. Capability *mediation* (gating behind a token) is the
//! implementor's job, not the trait's: each implementor (coral's `Library`,
//! the router's `ContentNodeStore`) defines its own marker `Capability` token and
//! asserts it via `fluent_wvr::capability::check_capability` in its own crate
//! (the trait never references fluent-wvr).

use crate::{ContentNode, KnnHit, NodeId};

/// Errors from a `KnowledgeCapability` implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeError {
    /// A node with the given id was not found.
    NotFound(NodeId),
    /// The node has no embedding to search against.
    NoEmbedding,
    /// Serialization of a node to its durable form failed.
    Serialization(String),
    /// Any other implementation-specific failure.
    Other(String),
}

impl std::fmt::Display for KnowledgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "node not found: {id:?}"),
            Self::NoEmbedding => write!(f, "node has no embedding"),
            Self::Serialization(e) => write!(f, "serialization error: {e}"),
            Self::Other(e) => write!(f, "knowledge error: {e}"),
        }
    }
}

impl std::error::Error for KnowledgeError {}

/// The cross-crate boundary for reading and writing a `ContentNode` store.
///
/// Concrete implementations are composed at the binary (coral-context's
/// `Library`, the router's `ContentNodeStore`); a consumer never imports a concrete
/// store crate through this trait.
pub trait KnowledgeCapability: Send + Sync {
    /// Fetch a node by id (`None` when absent).
    fn get_node(&self, node_id: NodeId) -> Option<ContentNode>;
    /// All nodes for a session, most recent first, capped at `limit`.
    fn get_session_nodes(&self, session_id: &str, limit: usize) -> Vec<ContentNode>;
    /// Persist a node, allocating an id when none is set. Returns the node id.
    fn insert_node(&self, node: &ContentNode) -> Result<NodeId, KnowledgeError>;
    /// Cosine KNN search over node embeddings; `k` nearest hits.
    fn knn_search(&self, embedding: &[f32], k: usize) -> Result<Vec<KnnHit>, KnowledgeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial in-crate fake so the trait is exercised without depending on
    /// any store implementation.
    struct FakeKnowledge;

    impl KnowledgeCapability for FakeKnowledge {
        fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
            (node_id == NodeId::from_int(1)).then(|| ContentNode {
                id: Some(node_id),
                ..Default::default()
            })
        }
        fn get_session_nodes(&self, session_id: &str, _limit: usize) -> Vec<ContentNode> {
            if session_id == "s" {
                vec![ContentNode {
                    id: Some(NodeId::from_int(1)),
                    ..Default::default()
                }]
            } else {
                vec![]
            }
        }
        fn insert_node(&self, _node: &ContentNode) -> Result<NodeId, KnowledgeError> {
            Ok(NodeId::from_int(42))
        }
        fn knn_search(&self, _embedding: &[f32], _k: usize) -> Result<Vec<KnnHit>, KnowledgeError> {
            Ok(vec![])
        }
    }

    #[test]
    fn trait_is_dispatchable_through_dyn() {
        let k: &dyn KnowledgeCapability = &FakeKnowledge;
        assert!(k.get_node(NodeId::from_int(1)).is_some());
        assert!(k.get_node(NodeId::from_int(2)).is_none());
        assert_eq!(k.get_session_nodes("s", 10).len(), 1);
        assert!(k.get_session_nodes("other", 10).is_empty());
    }

    #[test]
    fn insert_and_knn_round_trip_through_dyn() {
        let k: &dyn KnowledgeCapability = &FakeKnowledge;
        let node = ContentNode::default();
        assert_eq!(k.insert_node(&node).unwrap(), NodeId::from_int(42));
        assert_eq!(k.knn_search(&[0.0; 4], 3).unwrap().len(), 0);
    }

    #[test]
    fn knowledge_error_variants_display() {
        assert!(KnowledgeError::NotFound(NodeId::from_int(7))
            .to_string()
            .contains("7"));
        assert_eq!(
            KnowledgeError::NoEmbedding.to_string(),
            "node has no embedding"
        );
    }
}
