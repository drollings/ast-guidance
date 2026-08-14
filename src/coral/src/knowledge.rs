//! `KnowledgeCapability` implementation for coral-context's `Library`.
//!
//! This is the agreed cross-crate boundary: the router can read coral's Context
//! through `fluent_types::KnowledgeCapability` without importing coral (and
//! coral never imports the router). It is a thin adapter over the existing
//! `insert_node` / store queries / `knn_brute_force` — no new storage logic.
//!
//! The two processes are separate today; this surface is the boundary made
//! testable in-crate. Each method asserts a coral-owned capability token
//! (`CoralKnowledgeCapability`) in the current task-local, mirroring
//! `fluent_db::capability::check_db_capability`.

use fluent_types::{ContentNode, KnnHit, KnowledgeCapability, KnowledgeError, NodeId};
use fluent_wvr::capability::{check_capability, Capability};

use crate::db::{Library, LibraryError};

/// Coral-owned capability token gating the `KnowledgeCapability` surface.
pub struct CoralKnowledgeCapability;

impl Capability for CoralKnowledgeCapability {
    fn name(&self) -> &'static str {
        "coral.knowledge"
    }
}

fn denied() -> KnowledgeError {
    KnowledgeError::Other("missing capability: coral.knowledge".into())
}

fn library_to_knowledge(e: LibraryError) -> KnowledgeError {
    match e {
        LibraryError::NodeNotFound(msg) | LibraryError::DuplicateNode(msg) => {
            KnowledgeError::Other(msg)
        }
        LibraryError::Db(e) => KnowledgeError::Other(e.to_string()),
    }
}

impl KnowledgeCapability for Library {
    fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
        if check_capability(&CoralKnowledgeCapability).is_err() {
            return None;
        }
        self.get_node(node_id).ok().flatten()
    }

    fn get_session_nodes(&self, _session_id: &str, _limit: usize) -> Vec<ContentNode> {
        if check_capability(&CoralKnowledgeCapability).is_err() {
            return Vec::new();
        }
        // coral's `context_nodes` table has no session concept — the store is
        // cross-session. The trait's session read maps to the empty set here;
        // the router's ledger-backed store is the session-scoped reader.
        Vec::new()
    }

    fn insert_node(&self, node: &ContentNode) -> Result<NodeId, KnowledgeError> {
        if check_capability(&CoralKnowledgeCapability).is_err() {
            return Err(denied());
        }
        self.insert_node(node).map_err(library_to_knowledge)
    }

    fn knn_search(&self, embedding: &[f32], k: usize) -> Result<Vec<KnnHit>, KnowledgeError> {
        if check_capability(&CoralKnowledgeCapability).is_err() {
            return Err(denied());
        }
        self.knn_search(embedding, k, None)
            .map_err(library_to_knowledge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr::capability::CURRENT_CAPS;
    use fluent_wvr::CapabilitySet;

    fn caps() -> CapabilitySet {
        CapabilitySet::new().with(CoralKnowledgeCapability)
    }

    fn lib() -> Library {
        Library::open_in_memory().expect("in-memory library")
    }

    #[test]
    fn deny_without_token() {
        let lib = lib();
        assert!(KnowledgeCapability::get_node(&lib, NodeId::from_int(1)).is_none());
        assert!(KnowledgeCapability::get_session_nodes(&lib, "s", 10).is_empty());
        let node = ContentNode {
            name: "n".into(),
            source: "test".into(),
            ..Default::default()
        };
        assert!(matches!(
            KnowledgeCapability::insert_node(&lib, &node),
            Err(KnowledgeError::Other(_))
        ));
        assert!(matches!(
            KnowledgeCapability::knn_search(&lib, &[0.0; 4], 3),
            Err(KnowledgeError::Other(_))
        ));
    }

    #[tokio::test]
    async fn allow_with_token() {
        let lib = lib();
        CURRENT_CAPS
            .scope(caps(), async {
                let mut node = ContentNode {
                    name: "test-node".into(),
                    source: "test".into(),
                    lod: vec!["coral content".into()],
                    embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
                    ..Default::default()
                };
                let id = KnowledgeCapability::insert_node(&lib, &mut node).unwrap();
                let fetched = KnowledgeCapability::get_node(&lib, id).unwrap();
                assert_eq!(fetched.name.as_str(), "test-node");
                // Reuses the existing Library::knn_search (brute force / HNSW).
                let hits = KnowledgeCapability::knn_search(&lib, &[1.0, 0.0, 0.0, 0.0], 3).unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].node_id, id);
            })
            .await;
    }

    #[tokio::test]
    async fn deny_when_token_absent_inside_runtime() {
        let lib = lib();
        CURRENT_CAPS
            .scope(CapabilitySet::new(), async {
                assert!(
                    KnowledgeCapability::get_node(&lib, NodeId::from_int(1)).is_none(),
                    "must deny without the coral token even inside a runtime"
                );
            })
            .await;
    }
}
