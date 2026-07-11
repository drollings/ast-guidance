//! The common-core prelude — import this in every consumer to get the
//! 80% case for I/O, hashing, formatting, metrics, JSON-RPC, and tokens.
//!
//! ```rust
//! use common_core::prelude::*;
//! ```

pub use crate::{
    blake3_hex, ensure_dir, ensure_dir_or_panic, estimate_tokens, fnv1a64, format_json,
    format_size, hex_encode, method_not_found, read_to_string_err, sha256_hex, write_atomic,
    IoError, JsonRpcError, JsonRpcHandler, JsonRpcRequest, JsonRpcResponse, LatencyHistogram,
    TokenBudget,
};
