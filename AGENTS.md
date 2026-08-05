# Agent Bootloader — coral-router

**Context**: coral-router is a Rust-native LLM request router with a 3-stage
pipeline (`DeterministicPreFilter → Classifier → Router`,
see `src/router/src/pipeline_types.rs:48-53`) exposed as an OpenAI-compatible
HTTP API on :8079. The classifier is a single LLM call that subsumes the
former quality-gate, planning-refinement, and guardrail stages (see its doc
comment in `stages/classifier.rs`). It dispatches to a configurable frontier
LLM after the pipeline completes.

This is a rust monorepo with multiple projects and shared infrastructure.  coral-router the priority.

## Build-Test Loop

1. BUILD:       make router          # builds coral-router binary
                make router-start    # builds + starts server on :8079

2. TEST:        make router-test     # fluent-router unit/golden/e2e tests (kills server)

3. SMOKE:       make router-mock     # 29 curl smoke tests against live server
                curl -s http://127.0.0.1:8079/health
                curl -s -X POST http://127.0.0.1:8079/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -d '{"model":"fast","messages":{"role":"user","content":"What is 2+2?"}}'

4. VERIFY:      cargo build --workspace && cargo test --workspace
                && cargo clippy --workspace -- -D warnings


### Quick Reference

| Target         | Purpose |
|---|---|
| `make router`       | Build coral-router |
| `make router-start` | Build + start server (kills old first) |
| `make router-test`  | Kill server + run fluent-router unit/golden/e2e tests + --help dry-run |
| `make router-mock`  | Depends on router-start, runs 29 curl smoke tests, leaves server running |


### Import Boundaries

`fluent-router` may import from `common-core`, `fluent-wvr`, `fluent-concurrency`,
`guidance-llm`, `guidance-types`, `dag`, `search-vector`, and standard library /
`tokio` / `reqwest`.
It must NOT import from `guidance`, `coral`, `wasm_ipc`, `knowledge`,
`ontology`, or `rdf`.


## Source Layout

```
src/
  bin/
    guidance/          guidance binary (16-subcommand CLI + MCP server)
    coral/             coral binary (MCP server + ingest CLI)
    coral-router/      coral-router binary (config loading, main)
    job-copilot-daemon/ job-copilot binary (serve, validate-profile, install-native-messaging, doctor)
  guidance/            guidance-core: AST parser, sync engine, query engine, config
  coral/               coral-context: graph DB, cache router, MCP server, WASM runtime
  dag/                 guidance-dag: resolver, target_work_unit (Target → WorkUnit
                       bridge, runnable under Zone supervision), work_unit (CommandUnit),
                       adapter, middleware, drift, type_inference, target, capability
                       registry, error types, dep_graph (DependencyGraph<K> — canonical
                       dependency-tracking primitive)
  fluent-wvr/          Fluent WVR: Component, WorkUnit, FieldAccess, Describable traits
  fluent-wvr-macros/   Proc macros for FieldAccess derive
  fluent-concurrency/  WorkerPool, Scope, Limiter, PriorityQueue, CreditFlow,
                       Zone (supervision + dependency cancellation via DependencyGraph)
  llm/                 LLM HTTP client + embeddings (CachedEmbeddingProvider, LlmRequestQueue,
                       LlmClient, openai, url, http_class, llm_queue, error)
  types/               guidance-types (FileType, MemberType, Param, Member, etc.)
  common-core/         General-purpose utility crate (fluent-wvr-common)
                       Note: common-core contains no domain-specific logic;
                       no imports from dag/, coral/, or guidance/
  content-node/        guidance-content-node (lod slicing, file content annotation)
  search-vector/       guidance-search-vector (SQLite hybrid search + HNSW index)
  knowledge/           fluent-knowledge (WordIndex, TrigramIndex, CsrGraph, QueryCache)
  ontology/            guidance-ontology (entity extraction, YAGO taxonomy, capability inference)
  rdf/                 guidance-rdf (Turtle/N-Quads parser, normalization)
  wasm_ipc/            fluent-wasm-ipc (WASM IPC binary types)
  memory-plugin/       Pluggable memory tier (holographic, hindsight, honcho backends)
  job-copilot/          job-copilot-core: config, schema, sanitize, profile, dispatcher, server, components
  router/              fluent-router: LLM Router & Agent Orchestration Framework,
                       DependencySession (composes DependencyGraph), pipeline,
                       config, stages, transforms, dispatch, server
    src/
      pipeline.rs              PipelineOrchestrator — 3-stage sequential pipeline
      server.rs                HTTP server (+ frontier dispatch after pipeline)
      config.rs                RouterConfig deserialization + pipeline builder
      normalize.rs             Request/response normalization to OpenAI format
      ledger.rs                ContentNodeLedger — canonical ContentNode store;
                               LOD0/LOD5 eager, LOD1–4 lazy from LOD0 via
                               Summarizer; CompactionStrategy/RecencyCompaction
                               (folded in from deleted compaction.rs)
      stages/
        deterministic.rs       Stage 1: command dispatch, PII detection
        classifier.rs          Stage 2: single LLM call that subsumes the former
                               quality-gate / planning-refinement / guardrail
                               stages; emits direct response, routing target, or
                               rejection
        common.rs              Shared stage helpers (extract_user_message, …)
        retry_classifier.rs    Retry-with-backoff wrapper over the classifier
        pipeline_ref.rs        Re-usable named pipeline stage
      frontier/
        modes.rs               EscalationMode ladder taxonomy (filter → question →
                               team → turnover) + FrontierResult/AuditEntry;
                               execute_frontier_mode stub (forward track)
      dispatch/
        backend.rs             ChatBackend + OpenAiChatBackend/RetryChatBackend/
                               FallbackChatBackend (production dispatch trait)
        frontier.rs            OpenAI/Anthropic wire-format build/parse helpers,
                               reserved for the escalation ladder (forward track)
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

## Production adoption sites

Which consumers use which `fluent-concurrency` primitives:

| Primitive | Production consumer | Location |
|---|---|---|
| `Scope::defer` | `job-copilot` handler | `src/job-copilot/src/server/handler.rs` |
| `ResultPool` | `guidance-llm` request queue | `src/llm/src/client.rs` |
| `PriorityResultPool` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `WorkerPool` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `Queue` | `fluent-concurrency` tests only | `src/fluent-concurrency/src/pool.rs` |
| `Instrumented::with_metrics` | `bin/guidance` histogram, guidance `Enhancer` | `src/bin/guidance/src/main.rs`, `src/guidance/src/enhancer.rs` |
| `ComponentAdapter` | `coral` cache reactor | `src/coral/src/cache/reactor.rs` |
| `PartitionedRouter` | `job-copilot` dispatcher | `src/job-copilot/src/dispatcher/llm.rs` |
| `Zone` | `fluent-concurrency` supervision | `src/fluent-concurrency/src/zone.rs` |
| `DependencyGraph` | `Zone` cancellation, `DependencySession` | `src/dag/src/dep_graph.rs`, `src/router/src/dag_session.rs` |

---

## Consolidation Contract

`src/common-core` is the **only permitted zero-domain crate** in the workspace.
It must NOT import any `guidance-*` / `coral-*` / `fluent-*` / `dag` crate
(see `src/common-core/src/lib.rs` module doc). Generic storage backends
(`rusqlite` behind the `sqlite` feature) and generic data utilities
(hashing, I/O, strings, formatting, metrics, drift, interner) belong here;
anything that knows what a "node", "session", "target", "embedding", or
"WASM plugin" is belongs in its respective domain crate.

### Canonical Locations (single source of truth)

| Concept | Canonical location | Notes |
|---------|-------------------|-------|
| Hashing (blake3, sha256, fnv1a64, hex) | `common-core::hash` | `src/common-core/src/hash.rs` |
| Text utilities (`contains_ignore_case`, `truncate_at_sentence`, `strip_thinking_blocks`, `StreamingThinkFilter`, `AnsiStripper`, `filter_unsafe_chars`, `trim_doc_prefix`, `detect_identifier_kind`, …) | `common-core::string` | `src/common-core/src/string.rs` |
| Path / fs helpers (`mtime`, `read_file_alloc_err`, `write_atomic`, …) | `common-core::io` | `src/common-core/src/io.rs` |
| Shared error leaf types (`IoError`, `SqliteError`, `ResolverError`) + the `impl_from_io_error!` boilerplate macro | `common-core::error` | `src/common-core/src/error.rs` |
| Cross-crate magic constants (`MAX_FILE_SIZE`, `HnswParams`, …) and the canonical timeout/retry defaults (`DEFAULT_TOTAL_TIMEOUT_MS`=300_000, `DEFAULT_IDLE_TIMEOUT_MS`=30_000, `DEFAULT_RETRY_INTERVAL_S`=1) | `common-core::constants` | `src/common-core/src/constants.rs` |
| Bitset / capability registry | `common-core::interner` | `src/common-core/src/interner.rs` |
| BitSetDrift | `common-core::drift` | `src/common-core/src/drift.rs` |
| Latency histograms / metrics (`LatencyHistogram`, `bucket_counts`, `aggregate`) | `common-core::metrics` | `src/common-core/src/metrics.rs` |
| Poison-safe mutex locking (`lock`, `lock_read`, `lock_write` — `PoisonError::into_inner`) | `common-core::sync` | `src/common-core/src/sync.rs` |
| Fluent WVR newtype wrappers (`Instrumented`, `ComponentAdapter`, `Pipeline`, `retry_call`, `ComponentCascade`) | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| Jittered-exponential retry (`backoff_ms`, `retry_async`) | `common-core::retry` | `src/common-core/src/retry.rs` |
| Shared domain newtypes (`NodeId`, `SessionId`, `TargetId`, `LOD_COUNT`) | `guidance-types` | `src/types/src/lib.rs` |
| Cosine similarity / brute-force KNN | `search-vector::math` | `src/search-vector/src/math.rs` |
| SQLite open helpers + schemas (`open_wal`, `is_unique_violation`, `in_clause`) | `common-core::sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| JSON-RPC / MCP stdio loop | `common-core::jsonrpc` | `src/common-core/src/jsonrpc.rs` |
| Token budget helpers | `common-core::tokens` | `src/common-core/src/tokens.rs` |
| Directory walk / file scan | `common-core::walk` | `src/common-core/src/walk.rs` |
| Shell / subprocess helpers | `common-core::shell` | `src/common-core/src/shell.rs` |
| JSON config load-or-default | `common-core::config` | `src/common-core/src/config.rs` |
| Test utilities (`impl_component_for_test!`, `PassthroughUnit`, `tempdir()`) | `fluent-wvr-testutil` | `src/fluent-wvr-testutil/src/lib.rs` |
| 80% import line for component work (`Component`, `WorkUnit`, `FieldAccess`, `prelude::*`) | `fluent-wvr::prelude` | `src/fluent-wvr/src/prelude.rs` |
| HTML stripping (`strip_html`) | `common-core::string` | `src/common-core/src/string.rs` |
| `impl_component!` / `impl_fieldless!` macros (eliminate `as_any`/fieldless-`FieldAccess` boilerplate) | `fluent-wvr::impl_component!` / `fluent-wvr::impl_fieldless!` | `src/fluent-wvr/src/macros.rs` |
| Tolerant LLM-JSON parse (`parse_json_response` — fence-strip → parse → extract) | `fluent_llm::parse` | `src/llm/src/parse.rs` |
| OpenAI-compatible request body (`build_openai_chat_body` — carries the `chat_template_kwargs: {"enable_thinking": false}` default), stream-delta parser (`parse_openai_stream_delta`/`OpenAiDelta`), and OpenAI-format normalization (`normalize_request`/`normalize_response`/`error_response`/`messages_to_json`, parameterized on `serde_json::Value`) | `guidance-llm::openai` | `src/llm/src/openai.rs` |
| LLM endpoint derivation (`chat_completions_url`, `derive_embeddings_url`) + URL validation | `guidance-llm::url` | `src/llm/src/url.rs` |
| HTTP-status → failure taxonomy (`HttpClass`, `FailureClass`, `classify_http_status`) | `guidance-llm::http_class` | `src/llm/src/http_class.rs` |
| Typed LLM request queue (`LlmRequestQueue`, `build_default_queue`, `default_handler`) — worker-pool dispatch for chat completions | `guidance-llm::llm_queue` | `src/llm/src/llm_queue.rs` |
| `ComponentArcExt::try_as_any_mut` (safe mutable access to shared Arc) | `fluent-wvr::ComponentArcExt` | `src/fluent-wvr/src/traits.rs` |
| `WorkOutput::typed` (Result-returning) and `WorkOutput::data_take` (zero-copy) | `fluent-wvr::WorkOutput` | `src/fluent-wvr/src/work.rs` |
| `make_hnsw()` | `common-core::sqlite` | `src/common-core/src/sqlite.rs` (feature `sqlite`) |
| `Middleware` trait | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `MiddlewareChain` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `SuffixedComponent` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `Pipeline<T, E>` | `fluent-wvr::wrapper` | `src/fluent-wvr/src/wrapper.rs` |
| `global_pool_config()` | `fluent-concurrency::pool` | `src/fluent-concurrency/src/pool.rs` |
| `thread_local_resource!` / `with_tlr` | `fluent-concurrency::thread_resource` | `src/fluent-concurrency/src/thread_resource.rs` |
| `ReadThroughCache<K, V>`, `LoadCache<K, V, E>` (bounded get-or-load LRU) | `common-core::cache` | `src/common-core/src/cache.rs` |
| Generic keyed registry (`KeyedRegistry<K, V>` — insert/get/get_mut/keys/values/iter/remove/len; register-by-key lookup for plugin/provider registries) | `common-core::registry` | `src/common-core/src/registry.rs` |
| Generic dependency graph (`DependencyGraph<K>`, `GraphError`) | `fluent-dag::dep_graph` | `src/dag/src/dep_graph.rs` |

Cross-crate limits that currently have a single consumer stay in their
domain crate but **must** be moved to `common-core::constants` if a second
consumer appears. `MAX_KNN_CANDIDATES` and `MAX_MCP_REQUEST_SIZE` were
promoted in ROADMAP_20260804_SHARED_CORE M4.3 (coral re-exports them).
Current single-consumer limits (candidates for future promotion):
`MAX_WASM_HOST_CALLS` in `src/wasm_ipc/src/lib.rs:17`.

### HNSW instances

`coral` and `search-vector` each maintain **separate** HNSW indices backed by
the same `HnswParams::default()` constants. They remain separate because no
shared vector store exists between the two crates today. If a shared store
appears in the future, host the HNSW index in `search-vector` and have `coral`
delegate. Both crates use `knn_brute_force` from `search-vector::math` for
brute-force fallback and `common_core::sqlite::make_hnsw()` for index
construction — the positional-argument unpacking of `Hnsw::new(...)` is now
centralized in exactly one place.

### Metrics / Instrumented wiring (M12)

`common_core::metrics::LatencyHistogram` is the canonical latency surface,
and `fluent_wvr::wrapper::Instrumented::with_metrics(inner, label,
histogram)` is the future-ready API for recording per-unit execution
durations.  The in-tree consumers today are the CLI-level `cmd_histogram` in
`src/bin/guidance/src/main.rs` (total command timing) and the guidance
`Enhancer::with_metrics` (`src/guidance/src/enhancer.rs`, per-LLM-call
latency, wired by ROADMAP_20260804_SHARED_CORE M9).  Coral's `coral_stats`
aggregates across units via `LatencyHistogram::aggregate` (M4).  Candidate
adoption sites are documented in the `with_metrics` doc comment (the L4
Semantic KNN dispatch in `coral::cache::reactor`, and the top-level dispatch
on the `TargetWorkUnit` bridge under `Zone` supervision).  Adoption at any
of those moves M12 from "test-only" to a
real consumer wiring.

### Dependency tracking

`fluent_dag::dep_graph::DependencyGraph<K>` (`src/dag/src/dep_graph.rs`) is
the **canonical dependency-graph primitive** for the workspace.  It provides
`register`, `dependents_of` (transitive dependent set via cycle-resilient
DFS), `topo_sort` / `topo_sort_from` (Kahn's algorithm), `is_ready` /
`ready_nodes`, and `unresolved_deps` (unsatisfiable dependency detection). 
Any new dependency-tracking workflow — session step DAGs, build-target
graphs, task-supervision cancellation trees — MUST compose
`DependencyGraph<K>` rather than re-implementing graph algorithms.  The
reference integration is `fluent-concurrency`'s `Zone`
(`src/fluent-concurrency/src/zone.rs`), which replaced three hand-rolled
`HashMap`s with a single `DependencyGraph<ArcIntern<str>>`.

---

### `filter_thinking` — Think-block filtering contract

Every LLM call (classifier, frontier dispatch buffered, frontier dispatch
streaming) MUST respect `filter_thinking: bool` from the model config in
`env/coral-router.json`.  The contract has two parts:

1. **Request body**: The classifier HTTP request body (built by
   `chat_complete_http_inner_async` in `src/llm/src/client.rs`) MUST include
   `chat_template_kwargs: {"enable_thinking": false}` as a default so the model
   does not emit `<think>...</think>` blocks in its response.  Model-level
   `params` from `ModelEntry` are merged via `extra_body_params` so they can
   override this default.  The `LlmConfig.think` flag (boolean) is the final
   override — when `Some(true)` it enables thinking regardless of defaults.

2. **Response filtering**:
   - **Buffered dispatch** (`dispatch_to_llm_buffered` in `server.rs`): apply
     `strip_thinking_blocks()` after receiving the full response when
     `filter_thinking` is true.
   - **Streaming dispatch** (`dispatch_to_llm_streaming` / `stream_dispatch_inner`
     in `server.rs`): pass `filter_thinking` to `StreamingHandler` via
     `with_filter_thinking()`.  The handler tracks think-block open/close tags
     across chunks so partial tags are never leaked to the client.

When modifying any LLM call path, verify that `ModelEntry.params` from the
config are forwarded through `LlmConfig.extra_body_params` (for the classifier
path) or `RoutingTarget.params` (for the frontier dispatch path), and that
`filter_thinking` is propagated to the response handler.
