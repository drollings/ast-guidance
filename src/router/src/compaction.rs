//! LOD compaction policy — shrink older session nodes to lower detail levels
//! to stay within context budget. The interface is a trait so smarter policies
//! can be plugged in later.

use fluent_types::ContentNode;

/// Given a session's nodes, return the LOD level each node should be at.
/// Lower LOD = less detail retained. Higher LOD = more compacted.
/// LOD 0 = full detail, LOD N = progressively more compressed.
pub trait CompactionStrategy: Send + Sync {
    /// Returns LOD levels for each node. The returned `Vec` must have
    /// the same length as `nodes`.
    fn select_lod(&self, nodes: &[ContentNode], max_nodes: usize) -> Vec<u8>;
}

/// Simple recency-based compaction: recent nodes keep high detail,
/// older nodes are progressively demoted.
pub struct RecencyCompaction;

impl CompactionStrategy for RecencyCompaction {
    fn select_lod(&self, nodes: &[ContentNode], max_nodes: usize) -> Vec<u8> {
        let n = nodes.len();
        let mut lods = vec![0u8; n];

        // If we're under the max, everything stays at full detail.
        if n <= max_nodes {
            return lods;
        }

        // Recent nodes keep high detail (LOD 0-1), older nodes drop.
        for (i, lod) in lods.iter_mut().enumerate() {
            let position_from_end = n - i;
            if position_from_end <= max_nodes / 4 {
                *lod = 0; // most recent: full detail
            } else if position_from_end <= max_nodes / 2 {
                *lod = 1; // recent: ~800 chars
            } else if position_from_end <= 3 * max_nodes / 4 {
                *lod = 2; // moderate: ~240 chars
            } else {
                *lod = 3; // older: ~80 chars
            }
        }

        lods
    }
}

/// No-op compaction: leaves all nodes at full detail.
/// Useful for testing or when compaction is disabled.
pub struct NoopCompaction;

impl CompactionStrategy for NoopCompaction {
    fn select_lod(&self, nodes: &[ContentNode], _max_nodes: usize) -> Vec<u8> {
        vec![0u8; nodes.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::NodeId;

    fn make_nodes(count: usize) -> Vec<ContentNode> {
        (0..count)
            .map(|i| ContentNode {
                id: Some(NodeId::from_int(i as i64)),
                name: format!("node-{i}").into(),
                source: "test".into(),
                lod: vec![],
                embedding: None,
                capabilities: None,
                session_id: None,
                request_id: None,
                role: Some("user".into()),
                turn_index: Some(i as u64),
                accepted: Some(true),
                acceptance_score: None,
                active_lod: Some(0),
                parent_id: None,
                step_id: None,
                step_status: None,
                metadata: None,
                created_at: None,
            })
            .collect()
    }

    #[test]
    fn test_recency_compaction_under_max() {
        let nodes = make_nodes(3);
        let lods = RecencyCompaction.select_lod(&nodes, 10);
        assert_eq!(lods, vec![0, 0, 0]);
    }

    #[test]
    fn test_recency_compaction_over_max() {
        let nodes = make_nodes(8);
        let lods = RecencyCompaction.select_lod(&nodes, 4);
        assert_eq!(lods[0], 3);
        assert_eq!(lods[1], 3);
        assert_eq!(lods[7], 0);
    }

    #[test]
    fn test_noop_compaction() {
        let nodes = make_nodes(100);
        let lods = NoopCompaction.select_lod(&nodes, 10);
        assert!(lods.iter().all(|&l| l == 0));
    }
}
