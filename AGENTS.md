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
  db/                  fluent-db: canonical database-access layer (pooled/typed/async,
                       feature-gated on `sqlite`). Owns SqliteStore, SqlitePool,
                       DbError, HnswIndex, TtlCache, vector math, migrations, query
                       helpers, DbCapability, DbWorkUnit.
                       Raw SQLite mechanics stay in common-core::sqlite.
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
                               (folded in from deleted compaction.rs); facade
                               owns the M1 write-path guard (scrub before write)
      ledger_guard.rs          Irreversible write-path scrubber (M1): evaluates
                               the builtin filter engine with the
                               ContentNodeWrite scope; also the `transform` hook
                               impl for the PII frontier view (M2)
      views.rs                 Reference-only view layer over NodeStore (M2):
                               Lod newtype (0..=5), LedgerView trait (render is
                               the single text-exit), ParallelLedger,
                               FilteredLedger<V>, pii_redacted helper
      node_store.rs            Shared Arc<RwLock<ContentNode>> store + interned
                               session/role indices + durable content_json;
                               ensure_tier/lod_text/session_node_ids render
                               primitives
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
| Sync→async runtime bridge (`block_on`, `OnceLock` fallback runtime — multi-thread → `block_in_place` + `handle.block_on`; current-thread → `handle.block_on`; no handle → fallback) | `common-core::runtime` | `src/common-core/src/runtime.rs` |
| Shared domain newtypes (`NodeId`, `SessionId`, `TargetId`, `LOD_COUNT`) | `guidance-types` | `src/types/src/lib.rs` |
| Cosine similarity / brute-force KNN / RRF fusion (`cosine_similarity`, `knn_brute_force`, `vec_to_bytes`/`bytes_to_vec`/`try_bytes_to_vec`, `QuantizedEmbedding`, `cosine_similarity_q8`, `rrf_merge`) | `fluent-db::vector` (`search-vector::math` re-exports it) | `src/db/src/vector.rs` |
| Database error taxonomy (`DbError` — `Sqlite|NotFound|DuplicateEntry|Busy|PoolExhausted|InvalidSchemaVersion|Other`; the single `From<rusqlite::Error>` centralizing `is_unique_violation` → `DuplicateEntry` and `SQLITE_BUSY` → `Busy`) | `fluent-db::error` | `src/db/src/error.rs` |
| Single-connection store (open/WAL/schema-init/migrations/typed helpers, poison-safe lock) | `fluent-db::store` (`SqliteStore`) | `src/db/src/store.rs` |
| Pooled async store (`Semaphore` + `spawn_blocking` + RAII checkout, `PoolConfig { size, busy_timeout_ms }`; `acquire` capability-gated, `transaction` helper) | `fluent-db::pool` (`SqlitePool`) | `src/db/src/pool.rs` |
| Typed statement helpers (`query_row`/`query_rows`/`execute`/`execute_batch`/`query_rows_from_iter`/`last_insert_rowid`/`transaction`) | `fluent-db::query` | `src/db/src/query.rs` |
| Idempotent schema migrations (`Migration` trait, `migrate` via `PRAGMA user_version`, `ensure_column`, `schema_version`) | `fluent-db::migrate` | `src/db/src/migrate.rs` |
| TTL/LRU key-value cache store (`TtlCache` — get-with-expiry/put/evict_expired/evict_lru/clear/stats) | `fluent-db::cache` | `src/db/src/cache.rs` |
| HNSW-backed index store (`HnswIndex` — insert/rebuild_from/search/id_map; lock order `hnsw → id_map` never inverted, R9) | `fluent-db::hnsw` | `src/db/src/hnsw.rs` |
| Capability token over the pool (`DbCapability` — gated via `fluent_wvr::capability::check_capability`; re-exported from `fluent-concurrency::io::db` behind its `db` feature) | `fluent-db::capability` | `src/db/src/capability.rs` |
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
| LLM client sync→async bridge (`client::block_on` — delegates to `common_core::runtime::block_on`; module path preserved for callers) | `guidance-llm::client` | `src/llm/src/client.rs` |
| LLM endpoint derivation (`chat_completions_url`, `derive_embeddings_url`) + URL validation | `guidance-llm::url` | `src/llm/src/url.rs` |
| HTTP-status → failure taxonomy (`HttpClass`, `FailureClass`, `classify_http_status`) | `guidance-llm::http_class` | `src/llm/src/http_class.rs` |
| Typed LLM request queue (`LlmRequestQueue`, `build_default_queue`, `default_handler`) — worker-pool dispatch for chat completions | `guidance-llm::llm_queue` | `src/llm/src/llm_queue.rs` |
| `ComponentArcExt::try_as_any_mut` (safe mutable access to shared Arc) | `fluent-wvr::ComponentArcExt` | `src/fluent-wvr/src/traits.rs` |
| Database `Component`/`WorkUnit` adapters (`DbStore` — blocking-connection abstraction; `DbWorkUnit<F>` — `execute` offloads via `block_in_place`/`spawn_blocking`, scoping `ctx.caps` on **both** offload paths so pool-backed units are capability-correct; `store_unit(Arc<SqliteStore>, name, op)` factory; the pool-backed `DbStore` bridges sync→async via `common_core::runtime::block_on`) | `fluent-db::wvr` | `src/db/src/wvr.rs` |
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
consumer appears.
