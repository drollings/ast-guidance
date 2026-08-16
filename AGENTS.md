# Coral Router — Development Guide

**What it is**: a Rust-native LLM request router (crate `fluent-router`, binary
`coral-router`) exposing an OpenAI-compatible HTTP API on `:8079`. Every
request runs a **two-stage pipeline** — `DeterministicPreFilter → Classifier`
(see `src/router/src/pipeline_types.rs`) — that resolves to a direct response,
a routing target, or a rejection. Coral Router is also the **process owner** of
the local inference fleet: it spawns and supervises one `llama-server` process
per model weights file, serves the `/instances` management contract at its own
address, and is the single routing element between those llama-server tasks and
every other OpenAI-compatible endpoint. The llama.cpp router mode is never used.

This is a Rust monorepo with many crates and shared infrastructure; coral-router
is the priority.

## Build-Test Loop

1. **BUILD** — `make router` builds the binary; `make router-start` builds and
   (re)starts the server on `:8079`, waiting for `/health`.

2. **TEST** — `make router-test` runs the fluent-router unit/golden/e2e suites
   and a `--help` dry-run of the built binary.

3. **SMOKE** — `make router-mock` runs the config-synced routing integration
   tests in `fluent-router` (`src/router/src/config_route_tests.rs`): every
   intent/route declared in `env/coral-router.json` is probed against an
   in-process mock server and must dispatch through the `model_group` its
   `routes` entry maps to. Expectations are derived from the config at runtime,
   so the suite cannot drift from it. Spot-check a live server:
   ```
   curl -s http://127.0.0.1:8079/health
   curl -s -X POST http://127.0.0.1:8079/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"local","messages":[{"role":"user","content":"What is 2+2?"}]}'
   ```

4. **VERIFY** — the full gate before landing:
   ```
   cargo build --workspace && cargo test --workspace \
     && cargo clippy --workspace -- -D warnings
   ```

### Serving-layer notes

- A real (non-mock) boot spawns one `llama-server` per model that declares a
  weights source (`weights`/`hf_repo`) or `instances`, on a free localhost
  port, and rewrites that model's `endpoint` to it. Override the binary with
  `LLAMA_SERVER`; the supervisor also resolves it from `$PATH`.
- **On-demand residency**: only models declaring at least one **pinned**
  instance are spawned at boot. Other managed models are loaded lazily — the
  dispatch path calls `supervisor.ensure_running` on first use — and unloaded
  again when the sidecar evicts their last context. Within a booted model,
  only pinned instances are declared at spawn; unpinned instances (e.g.
  `scratch`) are created on demand via `POST /instances`.
- **VRAM budget**: the sidecar residency loop (`InstancePool::run_residency`)
  aggregates every manager's `/instances` and enforces an allocation budget of
  `device_total - sidecar.minimum_remaining_vram` (device total from
  `sidecar.vram_total_bytes` or auto-detected from ROCm `mem_info_vram_total`).
  Over budget it evicts the least-recently-used **largest** unpinned contexts
  and unloads any model left with zero contexts (freeing its weights). Pinned
  instances are never evicted.
- `default_params` in the config supplies run defaults for every managed model
  (`--batch-size`, `--ubatch-size`, `--cache-type-k/v`, `--flash-attn`,
  `--n-gpu-layers`, `--n-cpu-moe`, `--ctx-size`, `--sleep-idle-seconds`) and
  sampling params merged into dispatch bodies (per-model values win).
- `router-start` stops the old process tree first: SIGTERM triggers graceful
  shutdown in the router, which stops its llama-servers before exiting, so a
  restart never orphans serving processes. It then waits up to
  `ROUTER_START_TIMEOUT_S` (default 300s) for `/health` and fails loudly with
  the log tail on timeout.
- `--mock` mode skips supervision entirely (canned dispatch needs no real
  model), so the config-synced routing suite boots fast in-process and is
  independent of `router-start`.

## Make targets

| Target | Purpose |
|---|---|
| `make router` | Build coral-router |
| `make router-start` | Build + (re)start on `:8079`, waiting for `/health` (kills old tree first) |
| `make router-test` | Kill server + fluent-router unit/golden/e2e tests + `--help` dry-run |
| `make router-test-all` | `router-test` + coral-context HNSW benchmarks (slow) |
| `make router-mock` | Config-synced routing integration tests (intent → model_group, derived from `env/coral-router.json`) |

## Import boundaries

Shared library crates may NOT import from `guidance`, `coral`, or
`wasm_ipc`, as those are reserved for building compiled tools.

## Router crate map

```
src/
  router/src/                 fluent-router (the Coral Router crate)
    pipeline.rs               PipelineOrchestrator — two-stage pipeline
    pipeline_types.rs         StageDecision/Verdict, PipelineStage taxonomy, RoutingTarget
    config.rs                 RouterConfig, ModelEntry (weights/hf_repo/instances), sidecar
    stages/deterministic.rs   Stage 1: deterministic filters, commands, PII
    stages/classifier.rs      Stage 2: single LLM call / classification tree →
                              direct response | routing target | rejection
    server.rs                 RouterServer — hyper accept loop; runs sidecar tasks
    server/handler.rs         HTTP routing, ServerDeps; query routing fields, model-id grammar
    server/dispatch.rs        handle_dispatch — ChatBackend chain, allocate-on-503
    server/instances_api.rs   public /instances facade, /v1/models, /props, model-less proxies
    instances.rs              InstanceClient / InstanceManager / InstancePool + grammar + sidecar
    supervisor.rs             LlamaServerSupervisor — spawn/supervise llama-server per model
    dispatch/backend.rs       ChatBackend trait + OpenAi/Retry/Fallback backends
    dispatch/escalation.rs    Ladder (filter → question → team → turnover)
    ledger.rs, node_store.rs  ContentNode ledger, shared ContentNodeStore, views, scrub
    views.rs, ledger_guard.rs
    dag_session.rs            DependencySession + SessionRegistry (checkpoint/rewind)
    kv_cache.rs               Hot/Cold KV snapshot metadata + fork round-trip
    routes/plan.rs, routes/rigor.rs   plan & rigor HTTP routes
    normalize.rs              OpenAI JSON ↔ RouterRequest (preserves routing fields)
  bin/coral-router/           binary: config load, supervisor boot, endpoint rewrite, serve
  common-core/                shared utilities (hash/string/retry/runtime/cache/constants)
  llm/                        guidance-llm: LLM client, OpenAI wire format, HttpClass
  fluent-wvr/                 Component/WorkUnit/FieldAccess; wrapper::*; impl_component!
  fluent-concurrency/         Limiter, Zone, ResultPool, affinity scheduler
  dag/                        fluent-dag: DependencyGraph<K>, Target/WorkUnit bridge
  db/                         fluent-db: vector math, sqlite stores (feature `sqlite`)
  types/                      guidance-types: NodeId, SessionId, TargetId, LOD_COUNT
```

### Prohibited AI Use Cases or Actions

* Any git write operations (checkout, commit, stash, etc.)

* Any implementation without review of relevant documents including, but not limited to:

  - ./doc/skills/common-core/SKILL.md
  - ./doc/skills/fluent-wvr/SKILL.md
  - ./doc/skills/fluent-concurrency/SKILL.md

* Any implementation without reviewing shared primitives in shared libraries, and any removal of code from without human approval from these libraries:

  - ./src/common-core
  - ./src/fluent-wvr
  - ./src/fluent-concurrency

* Removing suspected unused primitives from shared crates without specific user approval.  Do not ask for such approval without extensive code review.

* Referencing transient roadmaps or milestones in documentation or comments
