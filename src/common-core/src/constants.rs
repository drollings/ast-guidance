//! Cross-crate magic numbers (size caps, dimension limits, HNSW defaults).

pub const MAX_VALUE_LEN: usize = 128;
pub const MAX_FILE_SIZE: usize = 100 * 1024 * 1024;
pub const MAX_JSON_DEPTH: usize = 100;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 4_096;
/// Max characters of a log message body to include in tracing/error records.
pub const MAX_LOG_MESSAGE_LEN: usize = 120;

/// Canonical per-dispatch wall-clock budget for a full LLM request (ms).
/// `RoutingTarget` (serde) and `ModelEntry` (serde) both read this constant
/// (ROADMAP_20260804_DRY M7.2 — the values were divergent at 120s vs 300s).
pub const DEFAULT_TOTAL_TIMEOUT_MS: u64 = 300_000;
/// Canonical per-chunk idle budget for a streaming LLM response (ms).
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 30_000;
/// Canonical base interval between retries (seconds).
pub const DEFAULT_RETRY_INTERVAL_S: u64 = 1;

/// Upper bound on KNN candidates before routing through the HNSW index
/// (promoted from coral's `db/mod.rs` — single-consumer magic constant that
/// reached the AGENTS.md promotion rule when a second consumer appeared).
pub const MAX_KNN_CANDIDATES: usize = 100_000;
/// Maximum accepted MCP request payload size, in bytes (promoted from coral's
/// `mcp.rs` alongside `MAX_KNN_CANDIDATES`).
pub const MAX_MCP_REQUEST_SIZE: usize = 10 * 1024 * 1024;

/// Serde default helper that returns `true`.
pub const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    pub max_nb_connection: usize,
    pub max_layer: usize,
    pub ef_construction: usize,
    pub initial_capacity: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_layer: 16,
            ef_construction: 200,
            initial_capacity: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_expected() {
        assert_eq!(MAX_VALUE_LEN, 128);
        assert_eq!(MAX_FILE_SIZE, 100 * 1024 * 1024);
        assert_eq!(MAX_JSON_DEPTH, 100);
        assert_eq!(MAX_EMBEDDING_DIMENSIONS, 4_096);
    }

    #[test]
    fn promoted_constants_match_coral_originals() {
        assert_eq!(MAX_KNN_CANDIDATES, 100_000);
        assert_eq!(MAX_MCP_REQUEST_SIZE, 10 * 1024 * 1024);
    }

    #[test]
    fn hnsw_params_default_matches_previous_inline_values() {
        let p = HnswParams::default();
        assert_eq!(p.max_nb_connection, 16);
        assert_eq!(p.max_layer, 16);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.initial_capacity, 1024);
    }
}
