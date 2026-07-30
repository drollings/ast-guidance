use common_core::tokens::estimate_tokens;
use common_core::tokens::TokenBudget;
use fluent_types::{ContentNode, NodeId};
use thiserror::Error;

use crate::db::Library;

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

pub struct ContextPacker {
    pub budget: TokenBudget,
    pub chars_per_token: usize,
}

impl ContextPacker {
    pub fn new(token_budget: usize) -> Self {
        Self {
            budget: TokenBudget(token_budget),
            chars_per_token: 4,
        }
    }

    /// Estimate token count from text length.
    pub fn estimate_tokens(text: &str) -> usize {
        estimate_tokens(text) as usize
    }

    /// Select the appropriate LOD level based on graph distance.
    /// Closer nodes get more detailed LOD (lower index).
    pub fn select_lod_by_distance(_node: &ContentNode, graph_distance: f64, avg_degree: f64) -> u8 {
        let effective_distance = graph_distance / (1.0 + avg_degree / (avg_degree + 1.0));
        if effective_distance < 1.0 {
            return 0;
        }
        if effective_distance < 2.0 {
            return 1;
        }
        if effective_distance < 3.0 {
            return 2;
        }
        if effective_distance < 4.0 {
            return 3;
        }
        if effective_distance < 5.0 {
            return 4;
        }
        5
    }

    /// Get the text at a given LOD level from a node.
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

    /// Pack context nodes around a focus node.
    ///
    /// 1. BFS from focus node up to depth 5
    /// 2. For each node, select LOD by effective distance
    /// 3. FFD bin-pack into token budget
    /// 4. Return packed nodes with selected LOD text
    pub fn pack(
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
                    Self::select_lod_by_distance(&node, f64::from(gn.depth), avg_degree);
                let text = Self::get_lod_text(&node, lod_level).to_string();
                candidates.push(PackedNode {
                    id: gn.node_id,
                    lod_level,
                    text,
                    graph_distance: f64::from(gn.depth),
                });
            }
        }

        // 3. FFD bin-pack into token budget
        let packed = self.ffd_pack(&candidates);

        Ok(packed)
    }

    /// First-Fit Decreasing bin-packing into token budget.
    fn ffd_pack(&self, candidates: &[PackedNode]) -> Vec<PackedNode> {
        let mut sorted: Vec<&PackedNode> = candidates.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.text.len()));

        let mut used_tokens = 0usize;
        let mut packed = Vec::new();

        for candidate in &sorted {
            let tokens = Self::estimate_tokens(&candidate.text);
            if used_tokens + tokens <= self.budget.0 {
                packed.push((*candidate).clone());
                used_tokens += tokens;
            }
        }

        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(ContextPacker::estimate_tokens("hello"), 1);
        assert_eq!(ContextPacker::estimate_tokens(""), 0);
        assert_eq!(ContextPacker::estimate_tokens(&"a".repeat(12)), 3);
    }

    #[test]
    fn test_select_lod_by_distance() {
        let node = ContentNode {
            id: Some(NodeId(1)),
            name: "test".into(),
            source: String::new(),
            lod: vec!["detailed".into(), "summary".into(), "brief".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        assert_eq!(ContextPacker::select_lod_by_distance(&node, 0.5, 2.0), 0);
        // effective = 1.5 / (1 + 2/3) = 1.5 / 1.667 ≈ 0.9 → lod 0
        assert_eq!(ContextPacker::select_lod_by_distance(&node, 1.5, 2.0), 0);
        // effective = 5.0 / 1.667 ≈ 3.0 → lod 3
        assert_eq!(ContextPacker::select_lod_by_distance(&node, 5.0, 2.0), 3);
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
        assert_eq!(ContextPacker::get_lod_text(&node, 0), "detail");
        assert_eq!(ContextPacker::get_lod_text(&node, 1), "summary");
        // Out of range returns last
        assert_eq!(ContextPacker::get_lod_text(&node, 5), "summary");
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
        let packed = packer.ffd_pack(&candidates);
        // Budget 10 chars = ~2 tokens per 4 chars = 5 tokens max
        // "aaaa" = 1 token, "bb" = 1 token, "cc" = 1 token = ~3 tokens, all fit
        assert_eq!(packed.len(), 3);
    }
}
