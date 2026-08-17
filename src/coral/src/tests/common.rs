//! Crate-typed test fixtures shared by coral-context Tier-1 suites.
//!
//! `make_node` is the single home for the bare `ContentNode` shape that was
//! hand-written ~30 times across `db/`, `cache/`, `ingest.rs`, `packer.rs`,
//! `cache_router.rs`, and `knowledge.rs` (see `ROADMAP_20260816_TESTS.md`
//! §1.3). Variants that add lod/embedding/capabilities use struct-update
//! syntax on top of it:
//!
//! ```ignore
//! let embedded = ContentNode { embedding: Some(vec![0.1, 0.2]), ..make_node("n", "s") };
//! ```
//!
//! Never copy a node literal into a new test module — use `make_node`.

use fluent_types::ContentNode;

/// A bare node: no id, no embedding, no capabilities, empty LOD.
pub fn make_node(name: &str, source: &str) -> ContentNode {
    ContentNode {
        id: None,
        name: name.into(),
        source: source.into(),
        lod: vec![],
        embedding: None,
        capabilities: None,
        ..Default::default()
    }
}