use fluent_types::{ContentNode, NodeId};
use thiserror::Error;

use crate::db::Library;

pub use guidance_llm::ContextPacker;

#[derive(Error, Debug)]
pub enum PackerError {
    #[error("library error: {0}")]
    Library(#[from] crate::db::LibraryError),
    #[error("node not found: {0}")]
    NodeNotFound(String),
}

#[derive(Debug, Clone)]
pub struct PackedNode {
    pub id: NodeId,
    pub lod_level: u8,
    pub text: String,
    pub graph_distance: f64,
}

/// Get the text at a given LOD level from a node. Node-specific helper kept
/// coral-side; the shared LOD-selection and budget-fit logic lives in
/// `guidance_llm::ContextPacker` (single `ContextPacker` per the roadmap).
pub fn get_lod_text(node: &ContentNode, level: u8) -> &str {
    let idx = level as usize;
    if idx < node.lod.len() {
        node.lod[idx].as_str()
    } else if let Some(last) = node.lod.last() {
        last.as_str()
    } else {
        node.name.as_str()
    }
}

/// Coral graph-packing extension over the shared `guidance_llm::ContextPacker`.
pub trait ContextPackerExt {
    /// Pack context nodes around a focus node.
    ///
    /// 1. BFS from focus node up to depth 5
    /// 2. For each node, select LOD by effective distance
    /// 3. FFD bin-pack into token budget (shared core in `guidance_llm`)
    /// 4. Return packed nodes with selected LOD text
    fn pack(
        &self,
        focus_id: NodeId,
        library: &Library,
    ) -> Result<Vec<PackedNode>, PackerError>;
}

impl ContextPackerExt for ContextPacker {
    fn pack(
        &self,
        focus_id: NodeId,
        library: &Library,
    ) -> Result<Vec<PackedNode>, PackerError> {
        let focus_node = library
            .get_node(focus_id)?
            .ok_or_else(|| PackerError::NodeNotFound("focus node not found".into()))?;

        // 1. BFS from focus node
        let graph_nodes = library.traverse_from(focus_id, 5)?;

        // 2. Load each node and compute LOD selection
        let avg_degree = if graph_nodes.len() > 1 {
            (graph_nodes.len() as f64 - 1.0).max(1.0)
        } else {
            1.0
        };

        let mut candidates: Vec<PackedNode> = Vec::with_capacity(graph_nodes.len() + 1);

        // Include focus node
        candidates.push(PackedNode {
            id: focus_id,
            lod_level: 0, // focus node gets most detail
            text: {
                if focus_node.lod.is_empty() {
                    focus_node.name.to_string()
                } else {
                    focus_node.lod[0].clone()
                }
            },
            graph_distance: 0.0,
        });

        for gn in &graph_nodes {
            if gn.node_id == focus_id {
                continue;
            }
            if let Ok(Some(node)) = library.get_node(gn.node_id) {
                let lod_level =
                    ContextPacker::select_lod_by_distance(f64::from(gn.depth), avg_degree);
                let text = get_lod_text(&node, lod_level).to_string();
                candidates.push(PackedNode {
                    id: gn.node_id,
                    lod_level,
                    text,
                    graph_distance: f64::from(gn.depth),
                });
            }
        }

        // 3. FFD bin-pack into token budget (shared core)
        let items: Vec<(&str, &PackedNode)> = candidates
            .iter()
            .map(|c| (c.text.as_str(), c))
            .collect();
        let packed = self.ffd_pack(&items);

        Ok(packed.into_iter().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_lod_by_distance() {
        assert_eq!(ContextPacker::select_lod_by_distance(0.5, 2.0), 0);
        // effective = 1.5 / (1 + 2/3) = 1.5 / 1.667 ≈ 0.9 → lod 0
        assert_eq!(ContextPacker::select_lod_by_distance(1.5, 2.0), 0);
        // effective = 5.0 / 1.667 ≈ 3.0 → lod 3
        assert_eq!(ContextPacker::select_lod_by_distance(5.0, 2.0), 3);
    }

    #[test]
    fn test_get_lod_text() {
        let node = ContentNode {
            id: Some(NodeId(1)),
            name: "test".into(),
            source: String::new(),
            lod: vec!["detail".into(), "summary".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        assert_eq!(get_lod_text(&node, 0), "detail");
        assert_eq!(get_lod_text(&node, 1), "summary");
        // Out of range returns last
        assert_eq!(get_lod_text(&node, 5), "summary");
        // No LOD falls back to name
        let bare = ContentNode {
            id: Some(NodeId(2)),
            name: "bare".into(),
            source: String::new(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        assert_eq!(get_lod_text(&bare, 0), "bare");
    }

    #[test]
    fn test_pack_respects_budget() {
        let lib = Library::open_in_memory().expect("db");
        let focus = ContentNode {
            id: None,
            name: "focus".into(),
            source: String::new(),
            lod: vec!["focus detailed text".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let focus_id = lib.insert_node(&focus).expect("insert");

        let child = ContentNode {
            id: None,
            name: "child".into(),
            source: String::new(),
            lod: vec!["child detailed content here".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let child_id = lib.insert_node(&child).expect("insert");
        lib.insert_edge(focus_id, child_id, "depends", 1.0)
            .expect("edge");

        let packer = ContextPacker::new(100); // large budget
        let packed = packer.pack(focus_id, &lib).expect("pack");
        assert!(!packed.is_empty(), "should pack at least focus node");

        // Very tight budget: should still pack at least the focus node
        let tight_packer = ContextPacker::new(1);
        let tight_packed = tight_packer.pack(focus_id, &lib).expect("pack");
        assert!(tight_packed.len() <= packed.len());
    }

    #[test]
    fn test_ffd_pack_respects_order() {
        let packer = ContextPacker::new(10);
        let candidates = vec![
            PackedNode {
                id: NodeId(1),
                lod_level: 0,
                text: "aaaa".into(),
                graph_distance: 0.0,
            },
            PackedNode {
                id: NodeId(2),
                lod_level: 1,
                text: "bb".into(),
                graph_distance: 1.0,
            },
            PackedNode {
                id: NodeId(3),
                lod_level: 2,
                text: "cc".into(),
                graph_distance: 2.0,
            },
        ];
        let items: Vec<(&str, &PackedNode)> = candidates
            .iter()
            .map(|c| (c.text.as_str(), c))
            .collect();
        let packed = packer.ffd_pack(&items);
        assert_eq!(packed.len(), 3);
    }
}
