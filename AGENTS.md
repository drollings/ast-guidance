# Agent Bootloader — guidance

**Context**: guidance is a Rust-native, deterministic-first AST-guided vector search
database generator with local AI enhancement.  When used to search the
codebase's capabilities and code, it can save over 90% of the tokens and tool
calls compared to the orchestrating AI coder using other tools.

## Prime Directive

1. **Never guess**: use `guidance explain "<query text>"` for guidance, and
follow instructions for any queries of interest

---

## Quick Start: RALPH Loop (Discovery → Implementation)

```
1. DISCOVER (guidance):  guidance explain "<keywords or a short question>"
                         Prefer keywords: "cmdExplain"
                         Or, prefer a short question: "How do we sync guidance?"
                         Scan: module purpose, pattern type, skill list

2. UNDERSTAND (MCP):     Read the primary source file(s) from step 1
                         Grep callers: who imports this file?
                         Ask: do the listed skills actually apply?

3. DECIDE:               If skills match → read them
                         If not → proceed to implementation

4. IMPLEMENT:            Write to src/guidance/ or src/bin/ (for binary targets)
                         Follow source patterns and applicable skills only

5. VERIFY (cargo):       cargo build --workspace && cargo test --workspace
                         && cargo clippy --workspace -- -D warnings
                         && cargo run --bin guidance -- structure .guidance

6. HEALTH (optional):    job-copilot-daemon doctor  (check daemon health)
```

---

## Source Layout

```
src/
  bin/
    guidance/          guidance binary (16-subcommand CLI + MCP server)
    coral/             coral binary (MCP server + ingest CLI)
  guidance/            guidance-core: AST parser, sync engine, query engine, config
  coral/               coral-context: graph DB, cache router, MCP server, WASM runtime
  dag/                 guidance-dag: executor, resolver, work_unit, adapter, middleware,
                       drift, type_inference, target, capability registry, error types
  fluent-wvr/          Fluent WVR: Component, WorkUnit, FieldAccess, Describable traits
  fluent-wvr-macros/   Proc macros for FieldAccess derive
  fluent-concurrency/  WorkerPool, Scope, Zone, Limiter, PriorityQueue, CreditFlow
  llm/                 LLM HTTP client + embeddings (CachedEmbeddingProvider, LlmRequestQueue,
                       LlmClient, url, error)
  types/               guidance-types (FileType, MemberType, Param, Member, etc.)
  common-core/         General-purpose utility crate (fluent-wvr-common)
                       Note: common-core contains no domain-specific logic;
                       no imports from dag/, coral/, or guidance/
  content-node/        guidance-content-node (lod slicing, file content annotation)
  search-vector/       guidance-search-vector (SQLite hybrid search + HNSW index)
  project-knowledge/   guidance-project-knowledge (WordIndex, TrigramIndex, CsrGraph, QueryCache)
  ontology/            guidance-ontology (entity extraction, YAGO taxonomy, capability inference)
  rdf/                 guidance-rdf (Turtle/N-Quads parser, normalization)
  wasm_ipc/            guidance-wasm-ipc (WASM IPC binary types)
  memory-plugin/       Pluggable memory tier (holographic, hindsight, honcho backends)
  bin/
    job-copilot-daemon/ job-copilot binary (serve, validate-profile, install-native-messaging, doctor)
  job-copilot/          job-copilot-core: config, schema, sanitize, profile, dispatcher, server, components
extension/             Chromium MV3 extension (JS/HTML/CSS — not a Cargo crate)
.guidance/
  guidance-config.json   Model / provider configuration
  .skills/          Structured skill documents (GoF, zig-current, domain-patterns)
  .doc/             Capabilities, diary, inbox
  src/              Generated guidance JSON (mirrors src/ tree)
.guidance.db        SQLite vector search database consumed by guidance explain
env/
  mk/               Shared Makefile helpers and per-language target overrides
  mise/             Language-specific mise.toml fragments
doc/
  DESIGN.md         System design reference
```

---

## Job Copilot — import boundary

`src/job-copilot` and `src/bin/job-copilot-daemon` may import from
`common-core`, `fluent-wvr`, `fluent-concurrency`, `guidance-llm`,
`guidance-types`, `dag`, `search-vector`, `memory-plugin`, `content-node`,
and the standard library / `tokio` / `reqwest` (the latter two via the
workspace deps). They **must not** import from
`guidance`, `coral`, `wasm_ipc`,
`project-knowledge`, `ontology`, or `rdf`. Domain logic for the copilot
lives in `src/job-copilot`; do not add it to any shared crate.

Shared crates **may** be improved when doing so produces more generic,
reusable, composable code. For example: adding a `with_histogram`
constructor to `TimingMiddleware`, extending `WorkContext` to accept
typed metadata, or adding a `FieldAccess` derive helper for complex
field types. These improvements benefit all consumers, not just the
copilot.

---

**DO:**
- Run `guidance explain "<query>"` and read the results
- Ask: "What capability is used here?" before consulting skills

**DON'T:**
- Assume skills apply without validating against source code
- Import from `src/guidance/` or `src/coral/` — those are consumers, not producers

---

## Consolidation Contract

`src/common-core` is the **only permitted zero-domain crate** in the workspace.
It must NOT import any `guidance-*` / `coral-*` / `fluent-*` / `dag` crate
(see `src/common-core/src/lib.rs` module doc). Generic storage backends
(`rusqlite` behind the `sqlite` feature) and generic data utilities
(hashing, I/O, strings, formatting, metrics, drift, interner) belong here;
anything that knows what a "node", "session", "target", "embedding", or
"WASM plugin" is belongs in its respective domain crate.

The active consolidation plan lives in
`ROADMAP_20260625_CONSOLIDATE.md` (checklist:
`ROADMAP_20260625_CONSOLIDATE_CHECKLIST.md`). Add new cross-crate limit or
helper there before re-implementing it locally.

### Canonical Locations (single source of truth)

| Concept | Canonical location | Notes |
|---------|-------------------|-------|
| Hashing (blake3, sha256, fnv1a64, hex) | `common-core::hash` | `src/common-core/src/hash.rs` |
| Text utilities (`contains_ignore_case`, `truncate_at_sentence`, …) | `common-core::string` | `src/common-core/src/string.rs` |
| Path / fs helpers (`mtime`, `read_file_alloc_err`, `write_atomic`, …) | `common-core::io` | `src/common-core/src/io.rs` |
| Shared error leaf types (`IoError`, `SqliteError`, `ResolverError`) | `common-core::error` | `src/common-core/src/error.rs` |
| Cross-crate magic constants (`MAX_FILE_SIZE`, `HnswParams`, …) | `common-core::constants` | `src/common-core/src/constants.rs` |
| Bitset / capability registry | `common-core::interner` | `src/common-core/src/interner.rs` |
| BitSetDrift | `common-core::drift` | `src/common-core/src/drift.rs` |
| Latency histograms / metrics | `common-core::metrics` | `src/common-core/src/metrics.rs` |
| Fluent WVR newtype wrappers (`Instrumented`, `WithRetry`, `ComponentAdapter`, `Pipeline`, `retry_call`) | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| Shared domain newtypes (`NodeId`, `SessionId`, `TargetId`, `LOD_COUNT`) | `guidance-types` | `src/types/src/lib.rs` |
| Cosine similarity / brute-force KNN | `search-vector::math` | `src/search-vector/src/math.rs` |
| SQLite open helpers + schemas | `common-core::sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| JSON-RPC / MCP stdio loop | `common-core::jsonrpc` | `src/common-core/src/jsonrpc.rs` |
| Token budget helpers | `common-core::tokens` | `src/common-core/src/tokens.rs` |
| Directory walk / file scan | `common-core::walk` | `src/common-core/src/walk.rs` |
| Shell / subprocess helpers | `common-core::shell` | `src/common-core/src/shell.rs` |
| JSON config load-or-default | `common-core::config` | `src/common-core/src/config.rs` |
| Test utilities (`impl_component_for_test!`, `PassthroughUnit`, `tempdir()`) | `fluent-wvr-testutil` | `src/fluent-wvr-testutil/src/lib.rs` |

Cross-crate limits that currently have a single consumer stay in their
domain crate but **must** be moved to `common-core::constants` if a second
consumer appears. Current single-consumer limits (candidates for future
promotion): `MAX_KNN_CANDIDATES` in `src/coral/src/db.rs:15`,
`MAX_MCP_REQUEST_SIZE` in `src/coral/src/mcp.rs:11`, `MAX_WASM_HOST_CALLS`
in `src/wasm_ipc/src/lib.rs:17`.

### HNSW instances

`coral` and `search-vector` each maintain **separate** HNSW indices backed by
the same `HnswParams::default()` constants (Milestone 2). They remain separate
because no shared vector store exists between the two crates today. If a shared
store appears in the future, host the HNSW index in `search-vector` and have
`coral` delegate. Both crates use `knn_brute_force` from `search-vector::math`
for brute-force fallback. Each crate owns a private `make_hnsw(&HnswParams,
initial_capacity)` helper that centralizes the positional-argument unpacking
of `Hnsw::new(...)` so the `(max_nb_connection, initial_capacity, max_layer,
ef_construction, DistCosine)` ordering is decided exactly once per crate
(`ROADMAP_20260625_CONSOLIDATE.md` Priority 6). The rebuild paths pass
`count.max(p.initial_capacity)` to grow the index with the loaded row count.

### Metrics / Instrumented wiring (M12)

`common_core::metrics::LatencyHistogram` is the canonical latency surface,
and `fluent_wvr::wrapper::Instrumented::with_metrics(inner, label, histogram)`
is the future-ready API for recording per-unit execution durations. The
in-tree consumer today is the CLI-level `cmd_histogram` in
`src/bin/guidance/src/main.rs` (total command timing). The `with_metrics`
constructor itself has **no production consumer** yet — its only in-tree
exercise is the `instrumented_with_metrics_records_duration` unit test in
`src/fluent-wvr/src/wrapper.rs`. Candidate adoption sites are documented in
the `with_metrics` doc comment (the L4 Semantic KNN dispatch in
`coral::cache_reactor`, and the top-level dispatch in `dag::executor`).
Adoption at any of those moves M12 from "test-only" to a real consumer
wiring; see `ROADMAP_20260625_CONSOLIDATE_CHECKLIST.md` M12 notes.

---

## Refinement contract

The active refinement plan lives in `ROADMAP_REFINE.md` (checklist:
`ROADMAP_REFINE_CHECKLIST.md`). Three non-negotiables for new coral code:

1. **Cache tiers MUST be `WorkUnit` implementations** — the orchestrator
   never branches on implementation type; every tier is an `Arc<dyn Component>`
   behind a uniform registry (M3).
2. **`RoutingResult` MUST be shared via `Arc`, never cloned under a lock** —
   `L1Cache::get` returns `Option<Arc<RoutingResult>>`; callers hold the `Arc`
   clone, not a deep copy (M1).
3. **`Handle::block_on(tokio::time::sleep)` MUST NOT appear inside a
   `WorkUnit::execute`** — sync `WorkUnit::execute` is a real boundary;
   `WithRetry` uses `std::thread::sleep` unconditionally (M5).

---

## Debugging and LLM Usage

### Command-Line Flags

**`--debug` / `--verbose`**:
- Shows LLM metadata: `[enhancer] generating file doc for X`, `[enhancer] received response`
- Hides raw prompt text (use `--show-prompts` for prompts)
- Use for general debugging and progress tracking

**`--show-prompts`**:
- Shows complete raw prompt text sent to LLM
- Use when debugging prompt engineering or LLM responses
- Independent of `--debug` (can combine both)

Example:
```bash
# View metadata only
guidance sync --debug --file src/example.rs

# View metadata + prompts
guidance sync --debug --show-prompts --file src/example.rs

# View prompts only (no metadata)
guidance sync --show-prompts --file src/example.rs
```

### Comment Management

**Source Files** (`.rs`, `.zig`, `.py`):
- Member comments (`///`) are the source of truth
- File/module comments (`//!`) also stored in JSON

**JSON Files** (`.guidance/src/**/*.json`):
- Store metadata: signatures, line numbers, match_hash
- File/module comments stored for backward compatibility
- Member comments NOT stored (smaller files, cleaner diffs)

**Database** (`.guidance.db`):
- Synced from both JSON and source files
- Member comments extracted from source during sync
- Used for semantic search via `guidance explain`

**Workflow**:
```bash
# Generate JSON without member comments
guidance sync --file src/example.rs

# View what changed (only metadata, no comment diffs)
git diff .guidance/src/example.rs.json

# Database sync extracts comments from source
guidance sync --file src/example.rs --db .guidance.db
```

### Staleness Detection

Files are processed when:
1. **JSON absent** → needs initial generation
2. **JSON newer than source** → needs processing (e.g., imported)
3. **JSON older than source by >1 second** → genuinely stale
4. **JSON = src_mtime - 1 second** → validated, skipped (no changes)

The `--force` flag bypasses staleness checks for full regeneration.
