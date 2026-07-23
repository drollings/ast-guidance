# common-core — Zero-Domain Utility Crate

**Context**: `common-core` (`src/common-core/`) is the **only permitted zero-domain**
crate in the workspace. It must NOT import any `guidance-*`, `coral-*`,
`fluent-*`, or `dag` crate. Generic utilities live here; domain logic does not.

## Reusable primitives — always use, never reimplement

| Concern | Module | Path |
|---------|--------|------|
| Hashing (blake3, sha256, fnv1a64, hex) | `hash` | `src/common-core/src/hash.rs` |
| Text utilities (contains_ignore_case, truncate_at_sentence, strip_html, slugify, …) | `string` | `src/common-core/src/string.rs` |
| Path/fs helpers (mtime, read_file_alloc_err, write_atomic, ensure_dir, …) | `io` | `src/common-core/src/io.rs` |
| Shared error leaf types (IoError, SqliteError, ResolverError) | `error` | `src/common-core/src/error.rs` |
| Cross-crate magic constants (MAX_FILE_SIZE, HnswParams, …) | `constants` | `src/common-core/src/constants.rs` |
| Bitset / capability registry | `interner` | `src/common-core/src/interner.rs` |
| BitSetDrift | `drift` | `src/common-core/src/drift.rs` |
| Latency histograms / metrics | `metrics` | `src/common-core/src/metrics.rs` |
| SQLite open helpers + schemas (open_wal, make_hnsw, init_embedding_cache) | `sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| JSON-RPC / MCP stdio loop | `jsonrpc` | `src/common-core/src/jsonrpc.rs` |
| Token budget helpers | `tokens` | `src/common-core/src/tokens.rs` |
| Directory walk / file scan | `walk` | `src/common-core/src/walk.rs` |
| Shell / subprocess helpers | `shell` | `src/common-core/src/shell.rs` |
| JSON config load-or-default | `config` | `src/common-core/src/config.rs` |
| ReadThroughCache<K, V> | `cache` | `src/common-core/src/cache.rs` |

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
