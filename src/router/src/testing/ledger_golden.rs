//! Golden ledger LOD snapshot (M4).
//! Fixture `testing/fixtures/ledger_lod.json` holds 3 nodes with LOD0/LOD5 eager + derived LOD4 (from LOD0 only).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenNode {
    pub node_id: i64,
    pub session_id: String,
    pub request_id: String,
    pub lod0: String,
    pub lod5: String,
    pub lod4: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenLedger {
    pub nodes: Vec<GoldenNode>,
}

pub fn load_golden() -> GoldenLedger {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/testing/fixtures/ledger_lod.json");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read ledger_lod.json {}: {}", path.display(), e));
    serde_json::from_str(&content).expect("ledger_lod.json parses")
}
