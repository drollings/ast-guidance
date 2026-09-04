//! The common-core prelude — import this in every consumer to get the
//! 80% case for I/O, hashing, formatting, metrics, JSON-RPC, and tokens.
//!
//! ```rust
//! use common_core::prelude::*;
//! ```

// NOTE (ROADMAP_20260903_LLM M11): `estimate_tokens` / `TokenBudget` were
// re-exported here through M10 (canonical owner `fluent_llm::tokens`);
// M11 removed them from the prelude. Import token budgets from
// `fluent_llm::tokens` instead.
pub use crate::{
    blake3_hex, ensure_dir, ensure_dir_or_panic, fnv1a64, format_json,
    format_size, hex_encode, method_not_found, read_to_string_err, sha256_hex, write_atomic,
    IoError, JsonRpcError, JsonRpcHandler, JsonRpcRequest, JsonRpcResponse, LatencyHistogram,
};
