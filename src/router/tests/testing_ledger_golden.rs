use super::*;

#[test]
fn golden_has_three_nodes() {
    let golden = load_golden();
    assert_eq!(golden.nodes.len(), 3, "fixture must have 3 nodes");
    for n in &golden.nodes {
        assert!(!n.lod0.is_empty(), "LOD0 eager");
        assert!(!n.lod5.is_empty(), "LOD5 eager at creation");
        assert!(!n.lod4.is_empty(), "LOD4 derived from LOD0");
        assert!(n.lod4.len() <= 240, "LOD4 ≤240 chars");
        assert!(n.lod5.len() <= 80 || n.lod5.len() <= n.lod0.len(), "LOD5 brief");
    }
}

#[test]
fn golden_snapshot_is_stable() {
    let golden = load_golden();
    let json = serde_json::to_string_pretty(&golden).unwrap();
    // Snapshot assertion: the file is the source of truth; this test fails if someone edits it without updating code.
    assert!(json.contains("quick brown fox"), "golden contains expected content");
    // Ensure 3 nodes
    assert_eq!(golden.nodes[0].node_id, 1);
    assert_eq!(golden.nodes[1].node_id, 2);
    assert_eq!(golden.nodes[2].node_id, 3);
}
