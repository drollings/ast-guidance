#![forbid(unsafe_code)]
#![allow(clippy::pedantic, clippy::all)]

//! common-core: Zero-domain generic utility crate (hashing, formatting, I/O,
//! shell, metrics, string utilities, watchdogs, time helpers, sync→async
//! runtime bridging).
//!
//! # Consolidation contract
//!
//! `common-core` is the only permitted **zero-domain** crate in the workspace.
//! It must NOT import any `guidance-*` / `coral-*` / `fluent-*` / `dag` crate.
//! Domain logic — anything that knows what a "node", "session", "target",
//! "embedding", or "WASM plugin" is — belongs in its respective domain crate,
//! not here. The sole exceptions are generic storage backends (`rusqlite`
//! behind the `sqlite` feature) and generic data utilities (hashing, I/O,
//! strings, formatting, metrics, drift, interner, watchdogs).
pub mod blob_spec;
pub mod cache;
pub mod calibration;
pub mod cite;
pub mod score;
pub mod telemetry;

pub use cache::LoadCache;

pub mod config;
pub mod constants;
pub mod drift;
pub mod error;
pub mod error_context;
pub mod format;
pub mod git;
pub mod hash;
#[cfg(feature = "http")]
pub mod http;
pub mod interner;
pub mod io;
pub mod jsonrpc;
pub mod metrics;
pub mod prelude;
pub mod registry;
pub mod retry;
pub mod runtime;
pub mod shell;
pub mod shell_parser;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod string;
pub mod sync;
pub mod time;
pub mod vector_math;
pub mod walk;
pub mod watchdog;
pub mod yago_normalize;
pub mod yago_taxonomy;

pub use config::{load_json, load_json_or_default};
pub use constants::{
    default_true, HnswParams, MAX_FILE_SIZE, MAX_JSON_DEPTH, MAX_VALUE_LEN,
};
pub use drift::BitSetDrift;
#[cfg(feature = "sqlite")]
pub use error::SqliteError;
pub use error::{IoError, ResolverError};
pub use error_context::{ErrorContext, HeapErrorContext};
pub use format::{format_csv, format_json, format_size, parse_size, Column, Table};
pub use hash::{
    blake3_hash, blake3_hex, content_hash_with_model, fnv1a64, hash_batch, hash_file, hex_encode,
    sha256_digest, sha256_hex, uuid_v4, BatchHashResult, HashAlgorithm, HashState,
};
pub use interner::CapabilityRegistry;
pub use io::{
    ensure_dir, ensure_dir_or_panic, make_path_absolute, mtime, read_file_alloc,
    read_file_alloc_err, read_to_string_err, resolve_path, strip_path_prefix, write_atomic,
};
pub use jsonrpc::{
    method_not_found, serve_stdio, JsonRpcError, JsonRpcHandler, JsonRpcRequest, JsonRpcResponse,
    METHOD_NOT_FOUND,
};
pub use metrics::LatencyHistogram;
pub use registry::{ConcurrentRegistry, KeyedRegistry};
pub use retry::{backoff_ms, capped_backoff_ms, retry_async, PollResult, PollWithBackoff};
pub use runtime::block_on;
pub use shell::{run_capture, run_command, run_shell_capture, shell_cmd, CommandOutput};
#[cfg(feature = "sqlite")]
pub use sqlite::{
    make_hnsw, open_in_memory, open_shared_in_memory, open_wal, run_batch,
};
pub use string::{
    contains_any, contains_any_word, contains_ident_word, contains_ignore_case, contains_word,
    detect_identifier_kind, filter_unsafe_chars, find_subseq, first_comment_line,
    first_sentence, has_extension, is_noisy_comment, is_path_token, is_test_path,
    looks_like_identifier, lower_into, skill_name_from_ref, slugify, strip_boilerplate,
    strip_nl_prefix, trim_doc_prefix, trim_left, trim_right, truncate_at_sentence, AnsiStripper,
    IdentifierKind, STOP_WORDS,
};
pub use time::now_secs;
pub use walk::{collect_extensions, should_skip_dir, walk_files, SOURCE_EXTENSIONS};
pub use watchdog::{
    BudgetWatchdog, RepetitionWatchdog, WallClockWatchdog, WatchdogEvent, WatchdogEventType,
    WatchdogSet,
};

