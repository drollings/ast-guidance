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
                         Use: use common_core::prelude::*; for the 80% case

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
                       drift, type_inference, target, capability registry, error types,
                       dep_graph (DependencyGraph<K> — canonical dependency-tracking primitive)
  fluent-wvr/          Fluent WVR: Component, WorkUnit, FieldAccess, Describable traits
  fluent-wvr-macros/   Proc macros for FieldAccess derive
  fluent-concurrency/  WorkerPool, Scope, Limiter, PriorityQueue, CreditFlow,
                       Zone (supervision + dependency cancellation via DependencyGraph)
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
  router/              fluent-router: LLM Router & Agent Orchestration Framework,
                       DependencySession (composes DependencyGraph), pipeline,
                       config, stages, transforms, dispatch, watchdog, server
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

## Composability guide

For day-to-day patterns (Component, Scope, Limiter, JSON-RPC, prelude)
see `doc/COMPOSABILITY.md`. The full API inventory and roadmap lives in
`ROADMAP_20260709_COMPOSABILITY.md` (checklist:
`ROADMAP_20260709_COMPOSABILITY_CHECKLIST.md`).

---

## Production adoption sites

Which consumers use which `fluent-concurrency` primitives:

| Primitive | Production consumer | Location |
|---|---|---|
| `Scope::defer` | `job-copilot` handler | `src/job-copilot/src/server/handler.rs` |
| `ResultPool` | `guidance-llm` request queue | `src/llm/src/client.rs` |
| `PriorityResultPool` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `WorkerPool` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `Queue` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `Instrumented::with_metrics` | `bin/guidance` histogram | `src/bin/guidance/src/main.rs` |
| `ComponentAdapter` | `coral` cache reactor | `src/coral/src/cache_reactor.rs` |
| `PartitionedRouter` | `job-copilot` dispatcher | `src/job-copilot/src/dispatcher/llm.rs` |
| `Zone` | `fluent-concurrency` supervision | `src/fluent-concurrency/src/zone.rs` |
| `DependencyGraph` | `Zone` cancellation, `DependencySession` | `src/dag/src/dep_graph.rs`, `src/router/src/dag_session.rs` |

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
| 80% import line for component work (`Component`, `WorkUnit`, `FieldAccess`, `prelude::*`) | `fluent-wvr::prelude` | `src/fluent-wvr/src/prelude.rs` |
| HTML stripping (`strip_html`) | `common-core::string` | `src/common-core/src/string.rs` |
| `impl_component!` macro (eliminates as_any boilerplate) | `fluent-wvr::impl_component!` | `src/fluent-wvr/src/macros.rs` |
| `ComponentArcExt::try_as_any_mut` (safe mutable access to shared Arc) | `fluent-wvr::ComponentArcExt` | `src/fluent-wvr/src/traits.rs` |
| `WorkOutput::typed` (Result-returning) and `WorkOutput::data_take` (zero-copy) | `fluent-wvr::WorkOutput` | `src/fluent-wvr/src/work.rs` |
| `make_hnsw()` | `common-core::sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| `Middleware` trait | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `MiddlewareChain` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `SuffixedComponent` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `Pipeline<T, E>` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `global_pool_config()` | `fluent-concurrency::pool` | `src/fluent-concurrency/src/pool.rs` |
| `thread_local_resource!` / `with_tlr` | `fluent-concurrency::thread_resource` | `src/fluent-concurrency/src/thread_resource.rs` |
| `ReadThroughCache<K, V>` | `common-core::cache` | `src/common-core/src/cache.rs` |
| Generic dependency graph (`DependencyGraph<K>`, `GraphError`) | `fluent-dag::dep_graph` | `src/dag/src/dep_graph.rs` |

Cross-crate limits that currently have a single consumer stay in their
domain crate but **must** be moved to `common-core::constants` if a second
consumer appears. Current single-consumer limits (candidates for future
promotion): `MAX_KNN_CANDIDATES` in `src/coral/src/db.rs:15`,
`MAX_MCP_REQUEST_SIZE` in `src/coral/src/mcp.rs:11`, `MAX_WASM_HOST_CALLS`
in `src/wasm_ipc/src/lib.rs:17`.

### HNSW instances

`coral` and `search-vector` each maintain **separate** HNSW indices backed by
the same `HnswParams::default()` constants. They remain separate because no
shared vector store exists between the two crates today. If a shared store
appears in the future, host the HNSW index in `search-vector` and have `coral`
delegate. Both crates use `knn_brute_force` from `search-vector::math` for
brute-force fallback and `common_core::sqlite::make_hnsw()` for index
construction — the positional-argument unpacking of `Hnsw::new(...)` is now
centralized in exactly one place (`ROADMAP_20260721_WVR_DEDUPE.md` M1).

### Metrics / Instrumented wiring (M12)

`common_core::metrics::LatencyHistogram` is the canonical latency surface,
and `fluent_wvr::wrapper::Instrumented::with_metrics(inner, label, histogram)`
is the future-ready API for recording per-unit execution durations. The
in-tree consumer today is the CLI-level `cmd_histogram` in
`src/bin/guidance/src/main.rs` (total command timing). Candidate adoption sites
are documented in the `with_metrics` doc comment (the L4 Semantic KNN dispatch
in `coral::cache_reactor`, and the top-level dispatch in `dag::executor`).
Adoption at any of those moves M12 from "test-only" to a real consumer
wiring; see `ROADMAP_20260625_CONSOLIDATE_CHECKLIST.md` M12 notes.

### Dependency tracking

`fluent_dag::dep_graph::DependencyGraph<K>` (`src/dag/src/dep_graph.rs`)
is the **canonical dependency-graph primitive** for the workspace. It
provides `register`, `dependents_of` (transitive dependent set via
cycle-resilient DFS), `topo_sort` / `topo_sort_from` (Kahn's algorithm),
`is_ready` / `ready_nodes`, and `unresolved_deps` (unsatisfiable
dependency detection). Any new dependency-tracking workflow — session
step DAGs, build-target graphs, task-supervision cancellation trees —
MUST compose `DependencyGraph<K>` rather than re-implementing graph
algorithms. The reference integration is `fluent-concurrency`'s `Zone`
(`src/fluent-concurrency/src/zone.rs`), which replaced three hand-rolled
`HashMap`s with a single `DependencyGraph<ArcIntern<str>>`. Future
consumers (e.g. the coral router's `DependencySession`, M5.2 of
`ROADMAP_20260722_CORAL_ROUTER.md`) must follow the same pattern.

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
