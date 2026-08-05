# common-core — Zero-Domain Utility Crate

**Context**: `common-core` (`src/common-core/`) is the **only permitted zero-domain**
crate in the workspace. It must NOT import any `guidance-*`, `coral-*`,
`fluent-*`, or `dag` crate. Generic utilities live here; domain logic does not.

## Reusable primitives — always use, never reimplement

| Concern | Module | Path |
|---------|--------|------|
| Hashing (blake3, sha256, fnv1a64, hex) | `hash` | `src/common-core/src/hash.rs` |
| Text utilities (contains_ignore_case, truncate_at_sentence, strip_html, strip_thinking_blocks, StreamingThinkFilter, AnsiStripper, filter_unsafe_chars, trim_doc_prefix, detect_identifier_kind, slugify, …) | `string` | `src/common-core/src/string.rs` |
| Path/fs helpers (mtime, read_file_alloc_err, write_atomic, ensure_dir, …) | `io` | `src/common-core/src/io.rs` |
| Shared error leaf types (IoError, SqliteError, ResolverError) + `impl_from_io_error!` macro | `error` | `src/common-core/src/error.rs` |
| Cross-crate magic constants (MAX_FILE_SIZE, HnswParams, MAX_KNN_CANDIDATES, MAX_MCP_REQUEST_SIZE, …) | `constants` | `src/common-core/src/constants.rs` |
| Bitset / capability registry | `interner` | `src/common-core/src/interner.rs` |
| BitSetDrift | `drift` | `src/common-core/src/drift.rs` |
| Latency histograms / metrics (LatencyHistogram, bucket_counts, aggregate) | `metrics` | `src/common-core/src/metrics.rs` |
| Poison-safe mutex locking (lock, lock_read, lock_write — PoisonError::into_inner) | `sync` | `src/common-core/src/sync.rs` |
| SQLite open helpers + schemas (open_wal, make_hnsw, init_embedding_cache, is_unique_violation, in_clause) | `sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| JSON-RPC / MCP stdio loop | `jsonrpc` | `src/common-core/src/jsonrpc.rs` |
| Token budget helpers | `tokens` | `src/common-core/src/tokens.rs` |
| Directory walk / file scan | `walk` | `src/common-core/src/walk.rs` |
| Shell / subprocess helpers | `shell` | `src/common-core/src/shell.rs` |
| JSON config load-or-default | `config` | `src/common-core/src/config.rs` |
| ReadThroughCache<K, V>, LoadCache<K, V, E> (bounded get-or-load LRU) | `cache` | `src/common-core/src/cache.rs` |
| Generic keyed registry (insert/get/keys/remove/len) | `registry` | `src/common-core/src/registry.rs` |

## Import pattern

```rust
use common_core::prelude::*;        // 80% case — brings in hash, io, string, tokens
use common_core::metrics::LatencyHistogram;  // explicit for metrics
use common_core::sqlite::open_wal;           // feature-gated
```

## What NOT to put here

- Anything that knows what a "node", "session", "target", "embedding",
  "WASM plugin", "Component", "WorkUnit", or "FieldAccess" is.
- Domain logic for guidance, coral, job-copilot, or fluent-router.

## Consolidation rule

When a cross-crate constant or utility has **two or more consumers**, it
MUST be promoted to `common-core`. Single-consumer items stay in their
domain crate temporarily — see `ROADMAP_20260625_CONSOLIDATE.md` for the
active promotion plan.

## Promotion candidates (canonical homes — compose, don't copy)

These consumer-side patterns are genuinely reusable but have **one** live
consumer today. Per the no-speculative-promotion rule, they stay in place
until a second consumer appears — at which point the *second* consumer must
move to the shared home below rather than copy the consumer copy.

| Pattern (current home) | Canonical home when promoted |
|---|---|
| `TelemetryEvent` + `TelemetrySink` PII-free observability contract (`src/router/src/telemetry.rs`) | `common-core::telemetry` (or `fluent-wvr::metrics` if it needs trait integration) |
| `ScoreMatrix` weighted ranked-candidate scoring (`src/router/src/score_matrix.rs`) | `common-core` (pure math — score-normalization + band matching) |
| `file:line` citation scanner (`src/guidance/src/grounding.rs:42-94` — `extract_citations`/`extract_citation_at`) | `common-core::string` (or a new `common-core::cite` module) |
