//! LLM-domain constants — the single owner (ROADMAP_20260903_LLM M6).
//!
//! Moved verbatim from `common_core::constants`: `MAX_EMBEDDING_DIMENSIONS`
//! (the embedding width every provider and HNSW store agrees on) and the
//! three request-budget defaults (`DEFAULT_TOTAL_TIMEOUT_MS`,
//! `DEFAULT_IDLE_TIMEOUT_MS`, `DEFAULT_RETRY_INTERVAL_S`) that every
//! router serde default and dispatch call site reads. Generic caps
//! (`MAX_VALUE_LEN`, `MAX_FILE_SIZE`, `MAX_JSON_DEPTH`, `MAX_LOG_MESSAGE_LEN`,
//! `MAX_KNN_CANDIDATES`, `MAX_MCP_REQUEST_SIZE`, `DEFAULT_HNSW_THRESHOLD`,
//! `DEFAULT_ONNX_*`, `default_true`, `HnswParams`) stay in `common-core`.
//!
//! M11 deleted the `common-core::constants` duplicate shim definitions
//! (kept through M10 under `#[deprecated]`); the locked values in
//! `tests/constants.rs` are the lasting contract.
//!
//! Calibration (roadmap §1, M10): these are task-value budgets (context
//! width, wall-clock budgets, retry cadence), not producer confidence.
//! The values move unchanged; retuning them is M10.

/// Maximum embedding dimensions every provider and HNSW store agrees on.
pub const MAX_EMBEDDING_DIMENSIONS: usize = 4_096;

/// Canonical per-dispatch wall-clock budget for a full LLM request (ms).
/// `RoutingTarget` (serde) and `ModelEntry` (serde) both read this constant.
pub const DEFAULT_TOTAL_TIMEOUT_MS: u64 = 300_000;
/// Canonical per-chunk idle budget for a streaming LLM response (ms).
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 30_000;
/// Canonical base interval between retries (seconds).
pub const DEFAULT_RETRY_INTERVAL_S: u64 = 1;
