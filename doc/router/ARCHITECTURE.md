# Coral Router — Architecture

## Source code location

The source code may be referenced at `./src/router/src/`.

## Overview

Coral Router exposes an OpenAI-compatible HTTP endpoint (`POST /v1/chat/completions` on `:8079`) that runs every incoming request through a two-stage pipeline before dispatching to a model. The pipeline is built from `Arc<dyn Component>` units (the Fluent WVR uniform interface) and the server itself is also a `WorkUnit` — everything is composable.

The architecture follows the MOA Router Specification (`MOA_ROUTER_SPEC.md`) and the design principles in `VISION.md`: deterministic before probabilistic, cheap before expensive, condensed context via a ledger, and frontier as a bounded, audited exception.

```
┌─ Request ──────────────────────────────────────────────────────────────┐
│  POST /v1/chat/completions  { model, messages, temperature, ... }      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ HTTP Server (RouterServer) ────────────────────────────────────────────┐
│  hyper HTTP/1.1 server on tokio (server.rs + server/handler.rs)         │
│  normalizes request body → RouterRequest via serde                      │
│  records initial request in ContentNodeLedger (LOD0)                    │
│  calls PipelineOrchestrator::execute() → WorkOutput::typed(PipelineResult) │
│  on classifier_response: respond directly                               │
│  on routing_target: handle_dispatch() via ChatBackend + HttpClass       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ PipelineOrchestrator ──────────────────────────────────────────────────┐
│  Vec<Arc<dyn Component>> executed sequentially                          │
│  each stage reads/writes via WorkContext.metadata + WorkOutput.data     │
│  decisions accumulate as Vec<StageDecision>                              │
│  short-circuits on StageVerdict::Rejected / Error                       │
│                                                                         │
│  Stage 1: DeterministicPreFilter  — deterministic filter engine        │
│    (no model call; Filter trait chain, PII detection, commands)        │
│  Stage 2: ClassifierStage         — single LLM call + routing          │
│    returns structured JSON → resolves route → builds routing target    │
│    resolves score matrix for multi-dimensional route scoring            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                   ┌─────────────────┼──────────────────┐
                   ▼                 ▼                   ▼
             Local Agent      Frontier API          Watchdogs
           (AgentRegistry)  (ChatBackend chain)  (Token/WallClock/Repeat)
```

## Pipeline: Two-stage design

The pipeline has two stages (formerly three). The classifier stage (`stages/classifier.rs`) absorbed routing logic — the LLM call returns a `ClassifierOutput` that carries both the classification verdict and the route target. The `PipelineStage::Router` enum variant is retained for forward compatibility; today it is only emitted by `PipelineOrchestrator` when a stage's `execute` returns an `Err` (the orchestrator records the failure as a `Router`-stage error decision before propagating).

| Stage | File | Model call? | Produces |
|-------|------|-------------|----------|
| 1. DeterministicPreFilter | `stages/deterministic.rs` | No | Command result, PII flag, or pass-through |
| 2. ClassifierStage | `stages/classifier.rs` | Yes | Direct response, rejection, or routing target |

## Design Contract

Every pipeline stage implements `Component` (the Fluent WVR supertrait). The orchestrator never branches on concrete type.

```rust
// Every stage: same trait, same dispatch, zero branching in hot path.
impl WorkUnit for DeterministicPreFilter { … }
impl FieldAccess for DeterministicPreFilter { … }
impl Describable for DeterministicPreFilter { … }
impl_component!(DeterministicPreFilter);
```

Data flows between stages as `StageDecision` (serde JSON) serialized into `WorkOutput.data` via `WorkOutput::typed()`. The orchestrator pulls it back out with `output.data_take::<StageDecision>()`.

## Source Layout (`src/router/src/`)

### Core types

| File | Role |
|------|------|
| `types.rs` | `RouterRequest`, `RouterResponse`, `RouterMessage`, `RouterChoice`, `Usage` — serde-serializable OpenAI protocol |
| `pipeline_types.rs` | `StageDecision`, `PipelineStage`, `StageVerdict`, `RoutingDestination` — the structured record every stage emits |
| `config.rs` | `RouterConfig`, `ModelEntry`, `SessionProfile`, `AuditLogConfig`, `ChartsConfig`, `PostProcessConfig` — deserialized from `env/coral-router.json`. Split into submodules re-exported from `config`: `PipelineParams` in `config/builder.rs`, `RoutingConfig` in `config/routing.rs`, `RejectPatterns`/`PatternEntry`/`FilterAction`/`FilterScope`/`ConfidenceGate` in `config/filters.rs` |

### Pipeline & Stages

| File | Role |
|------|------|
| `pipeline.rs` | `PipelineOrchestrator` — `WorkUnit` that iterates stages, collects decisions, short-circuits on reject/error |
| `pipeline_graph.rs` | DAG-based pipeline topology with dynamic stage routing via `fluent_dag::dep_graph::DependencyGraph` |
| `normalize.rs` | OpenAI JSON ↔ `RouterRequest`/`RouterResponse` conversion; `error_response()` for OpenAI-format errors |
| `stages/deterministic.rs` | `DeterministicPreFilter` — delegates to `DeterministicFilterEngine` for regex PII detection; slash-command dispatch (`/help`, `/stats`, `/checkpoint`) |
| `stages/classifier.rs` | `ClassifierStage` — single LLM call returning structured JSON; resolves route via `RoutingConfig::resolve_route`; resolves score matrix; builds routing target |
| `stages/common.rs` | Shared helpers — `extract_user_message()`, `get_metadata_string()` |
| `stages/retry_classifier.rs` | `RetryClassifier` — wraps the classifier stage with retry-with-backoff for LLM call resilience |
| `stages/switch.rs` | `SwitchStage` — branching pipeline logic for conditional stage dispatch |
| `stages/pipeline_ref.rs` | `PipelineRef` — re-usable pipeline stage from named config |

### Filters (MOA_ROUTER_SPEC §2)

| File | Role |
|------|------|
| `filters/mod.rs` | `Filter` trait — `kind() → FilterKind` + `evaluate(ctx) → Option<FilterDecision>`; `FilterKind::{Regex, Whitelist, HnswSimilarity, ModelClassification}`; `DeterministicFilterEngine` — ordered chain-of-responsibility over `Vec<Box<dyn Filter>>`; `FilterDecision::HardReject`, `SoftRedirect`, `OutputFilter` with codeword substitution support |
| `filters/regex_filter.rs` | `RegexFilter` — compiles regexes from `PatternEntry` config; respects `FilterScope` (Any, FrontierBound), `ConfidenceGate` (LuhnValid, None), and `FilterAction` (Redact, Anonymize, Omit) |
| `filters/injection_detect.rs` | `InjectionDetectFilter` — heuristic prompt-injection / system-prompt-exfiltration detection |
| `filters/luhn.rs` | Luhn algorithm validation — secondary check gate for credit-card-number patterns |

### Dispatch

| File | Role |
|------|------|
| `dispatch/backend.rs` | `ChatBackend` trait (`complete` / `stream_complete`); `OpenAiChatBackend` (single-attempt HTTP), `RetryChatBackend` (exponential-backoff retry), `FallbackChatBackend` (ordered backend chain) |
| `dispatch/frontier.rs` | `DispatchBackend` trait (`provider_name`, `build_request`, `parse_response`, `parse_stream_event`) + `DispatchError`; backends `OpenAiBackend`, `AnthropicBackend`, `OpenAiCompatBackend`; `LlmDispatcher` with concurrency `Limiter` |
| `dispatch/retry.rs` | `retry_http_request` — bare HTTP POST retry with exponential backoff and `HttpClass` classification |
| `dispatch/agent.rs` | `AgentDispatcher` — bridges `RoutingDestination::LocalAgent` to `AgentRegistry` with KV-cache restore/save |
| `agent.rs` | `AgentRegistry` — maps `(model, adapter, session)` triple to `ResultPool<AgentTask, String, AgentError>` |

### Session & Orchestration

| File | Role |
|------|------|
| `session.rs` | Thin shim — re-exports `StepStatus` from `fluent-types` (the canonical session node schema is `fluent_types::ContentNode`; `SessionStep` in `dag_session.rs` carries a `status: StepStatus`) |
| `orchestrator.rs` | `OrchestratorSession` — linear session with recency compaction, checkpoint/rewind, LLM client |
| `dag_session.rs` | `DependencySession` — DAG-based session composing `fluent_dag::dep_graph::DependencyGraph<String>` for step dependency tracking, checkpoint/rewind, KV-cache snapshot restore |
| `compaction.rs` | `CompactionStrategy` trait; `RecencyCompaction` demotes older nodes to higher LOD (less detail) |
| `ledger.rs` | `ContentNodeLedger` — SQLite-backed LOD0 request/response recording via `common_core::sqlite::open_wal()`; records requests before pipeline, updates with results afterward; supports `collapse_node()` for context condensation |

### Routes (MOA_ROUTER_SPEC §6-7)

| File | Role |
|------|------|
| `routes/plan.rs` | `PlanRoute` — retrieves similar prior workflows via HNSW index; template workflow application; gap analysis; targeted interview generation |
| `routes/rigor.rs` | `RigorRoute` — blue/red/judge 3-pass protocol; KV-cache checkpoint/rewind support; frontier escalation threshold |
| `workflow_config.rs` | `WorkflowConfig` — serializable workflow templates with range switches and branching |
| `charts/` | Chart (DAG workflow) library — `ChartStore`, binding, compile, execute, render, rubric, select, extract (`WorkflowExtractor`) — the M6–M10 workflow engine consumed by `PlanRoute` and the dispatch learning loop |

### Infrastructure

| File | Role |
|------|------|
| `server.rs` | `RouterServer` (`WorkUnit`) — hyper HTTP/1.1 accept loop on tokio; entry point that fans out to the `server/` submodule |
| `server/handler.rs` | HTTP routing — `/v1/chat/completions`, `/v1/plan`, `/health`, `/stats`, `/admin/cache/*`; request→pipeline→ledger orchestration |
| `server/dispatch.rs` | `handle_dispatch` / `dispatch_real` — primary + `fallbacks` chain dispatch through `ChatBackend`, response cache read/write, M10 workflow extraction |
| `server/responses.rs` | OpenAI-completion response builders, SSE/CORS headers, `ServerStats` counters |
| `kv_cache.rs` | Two-tier: `HotKvCache` (RAM LRU) + `ColdKvCache` (disk tree `model/adapter/session`); `KvCacheManager` composes both |
| `scheduler.rs` | `AffinityScheduler` re-exporting `fluent_concurrency::affinity::AffinityScheduler` — affinity bonus + aging for KV-cache affinity |
| `watchdog.rs` | `WatchdogSet` composing `MaxTokenWatchdog`, `WallClockWatchdog`, `RepetitionWatchdog` |
| `streaming.rs` | `StreamingHandler` — SSE delta formatting for OpenAI-compatible streaming chunks; think-block filtering across chunks |
| `summarization.rs` | `ResultScorer` + `Summarizer` — both `WorkUnit` impls that call an LLM to score/condense responses |
| `score_matrix.rs` | `ScoreMatrix` — multi-dimensional weighted scoring (coherence/complexity/completeness/risk) with per-route dimension bands; resolved in classifier stage |
| `transforms/` | `TransformStrategy` trait: `NoTransform`, `PiiAnonymize`, `DecomposeToAnonymizedHypothetical`, `DecomposeToSubtasks`, `CodewordAnonymizer` (session-scoped codeword substitution), `Sanitize`, `SecretMask` |
| `indexer.rs` | `AdapterIndexer` — eager validation of LoRA adapter files at startup |
| `metrics.rs` | `RouterMetrics` — per-model/agent latency histograms via `common_core::metrics::LatencyHistogram`, stage verdict counters, watchdog/error counters |
| `logging.rs` | Two-stream `tracing` subscriber: operational JSON/console rolling file + optional audit log stream (separate retention, always JSON, gated on `router.audit=info`) |
| `frontier/modes.rs` | `FrontierMode` enum — `AnonymizedHypothetical`, `AuthorizedCodeReview`, `WorkflowComposition`, `CopilotJudge`; `FrontierResult` and `AuditEntry` for the frontier audit trail (mode execution is a placeholder today — `execute_frontier_mode` returns `ServerError::FrontierNotImplemented` until agent wiring lands) |
| `hnsw.rs` | `HnswIndices` — three named HNSW index handles (`workflow_library`, `rubric_cache`, `blacklist_similarity`) for future coral embedding integration |
| `testing/` | `TranscriptProvider`, `MockTranscriptEntry`, `MockDispatchContext` — transcript-driven integration-test harness for E2E and golden tests |

### Adapter architecture (`dispatch/`)

The provider adapters follow the **Strategy** pattern in `dispatch/frontier.rs`:

```rust
pub trait DispatchBackend: Send + Sync {
    fn provider_name(&self) -> &str;
    fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError>;
    fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError>;
    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError>;
}
```

Three backends ship: `OpenAiBackend`, `AnthropicBackend`, `OpenAiCompatBackend`. `LlmDispatcher` holds a `HashMap<String, ProviderConfig>` and a `Limiter` for concurrency capping, dispatching through `reqwest`.

The **production dispatch path** does not use `LlmDispatcher`; it runs through `dispatch/backend.rs`, which defines the `ChatBackend` trait that every server dispatch site depends on:

```rust
pub trait ChatBackend: Send + Sync {
    fn complete(&self, request, model, params, idle_timeout_ms, total_timeout_ms)
        -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>>;
    fn stream_complete(&self, request, model, params, idle_timeout_ms, total_timeout_ms, filter_thinking)
        -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>>;
}
```

Concrete backends: `OpenAiChatBackend` (single-attempt HTTP; non-2xx status classified via `HttpClass` into `DispatchError::RateLimited` vs `DispatchError::Http`), `RetryChatBackend` (exponential backoff `retry_base * 1000 * 2^attempt`), and `FallbackChatBackend` (ordered backend chain). `server/dispatch.rs::dispatch_real` iterates the primary `RoutingTarget` plus its `fallbacks` list, wrapping each target in a retry backend and short-circuiting on non-retryable errors. Streaming flows through `StreamingHandler` (see below) over an `http_body_util` channel.

### Agent architecture (`agent.rs`, `dispatch/agent.rs`)

Agents are keyed on the triple `(model, adapter_option, session_id)`. Each identity gets its own `ResultPool` — this is deliberate: all requests for the same identity route to the same worker pool, preserving KV-cache affinity in the underlying llama.cpp server.

```rust
pub struct AgentRegistry {
    pools: HashMap<AgentIdentity, Arc<ResultPool<AgentTask, String, AgentError>>>,
    adapters: HashMap<(String, String), AdapterHandle>,  // (base_model, name)
    runtime: Arc<dyn Runtime>,
    …
}
```

The `AgentDispatcher` wraps registry access with pre-dispatch KV-cache restore and post-dispatch KV-cache save, plus per-token watchdog checking.

### Filter engine architecture (`filters/`)

Filters follow the **Chain of Responsibility** pattern (GoF). The `Filter` trait declares two methods — `kind()` (one of `FilterKind::{Regex, Whitelist, HnswSimilarity, ModelClassification}`) and `evaluate`:

```rust
trait Filter: Send + Sync {
    fn kind(&self) -> FilterKind;
    fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision>;
}
```

`DeterministicFilterEngine` holds `Vec<Box<dyn Filter>>` and evaluates filters in order, returning the first non-`None` decision. Built-in filters: `RegexFilter` (compiled from `PatternEntry` config) and `InjectionDetectFilter` (heuristic prompt-injection detection). Filters can gate on `ConfidenceGate` (Luhn validation) and scope themselves to `FilterScope::FrontierBound` (only apply when traffic is head-to-frontier). `FilterDecision::OutputFilter` carries `RegexMatch` structs with position data so the `CodewordAnonymizer` can do consistent, position-aware substitution.

## Key Compositions & Reusable Primitives

| Primitive | Source | Used by router at |
|-----------|--------|-------------------|
| `Component` / `WorkUnit` | `fluent-wvr` | Every pipeline stage, `PipelineOrchestrator`, `RouterServer` |
| `DependencyGraph<K>` | `fluent-dag::dep_graph` | `DependencySession` for step DAG tracking; `pipeline_graph` for dynamic pipeline topology |
| `ResultPool` | `fluent-concurrency::pool` | `AgentRegistry` — one pool per `(model, adapter, session)` |
| `PriorityResultPool` | `fluent-concurrency::pool` | `AffinityScheduler` — priority dispatch with aging |
| `Limiter` | `fluent-concurrency::pool` | `ClassifierStage` — concurrent classifier call cap; `charts/compile.rs` + `charts/execute.rs` — chart-DAG execution cap |
| `WorkContext` | `fluent-wvr` | Carries request, caps, runtime through every stage |
| `Runtime` trait | `fluent-wvr` | Plugged via `fluent_concurrency::tokio_runtime()` everywhere |
| `LatencyHistogram` | `common-core::metrics` | `RouterMetrics` — per-model and per-agent latency |
| `open_wal()` | `common-core::sqlite` | `ContentNodeLedger` — WAL-mode SQLite with busy timeout |
| `HttpClass` | `guidance-llm` | `dispatch/backend.rs` — status classification in `OpenAiChatBackend` (streaming + buffered) |
| `DispatchError::is_retryable()` | `fluent-router` | `dispatch/frontier.rs` — retry/fallback decisions in `dispatch/backend.rs` and `server/dispatch.rs` |
| `LlmError::is_retryable()` | `fluent-concurrency` | `guidance-llm` client error classification |

## HttpClass: where it lives and why

`HttpClass` (`HardReject`, `TransientFailure`, `EscalationRequired`, `UpstreamFailure`) is defined in `guidance-llm/src/http_class.rs` and re-exported via `guidance_llm::HttpClass`. It is consumed in two layers:

1. **`LlmClient::chat_complete_http_inner_async`** (in `guidance-llm`) — checks HTTP status before parsing the response body. On a non-2xx status, short-circuits with `LlmError::RateLimited` (retryable) or `LlmError::Api` (permanent). This fixes a latent bug where 503 HTML bodies were parsed as chat-completion JSON.

2. **Router dispatch backends** (in `dispatch/backend.rs`) — the router dispatches with a raw `reqwest::Client` through `OpenAiChatBackend` (not `LlmClient`), so it applies `HttpClass` directly. Both `OpenAiChatBackend::complete` (buffered) and `OpenAiChatBackend::stream_complete` (streaming) use the identical pattern: `HttpClass::from_status(status)` → `is_retryable()` → `Err(DispatchError::RateLimited)` (retry) vs `Err(DispatchError::Http)` (permanent). Retries are applied by `RetryChatBackend` (per-target exponential backoff), and the primary-plus-`fallbacks` chain is walked by `server/dispatch.rs::dispatch_real`.

The router's own error taxonomy mirrors this at a higher level: `DispatchError::is_retryable()` (in `dispatch/frontier.rs`) returns `true` for `Http(_)` and `RateLimited`. Separately, `LlmError::is_retryable()` (in `fluent-concurrency::llm_queue`) classifies `guidance-llm`/queue errors: `Http(_)` and `RateLimited` are retryable; `Api(_)` and `NoResponse` are permanent. Both are error-level classifications independent of how the error was produced.

## Import Boundaries (enforced)

Following AGENTS.md: `fluent-router` imports from `common-core`, `fluent-wvr`, `fluent-concurrency`, `guidance-llm`, `fluent-types`, `fluent-dag`, and standard library / `tokio` / `reqwest`. It does NOT import from `guidance`, `coral`, `wasm_ipc`, `project-knowledge`, `ontology`, or `rdf`.

## Pipeline data flow detail

1. **Server**: hyper reads the HTTP request; `server/handler.rs` collects the body (enforcing `max_payload`), deserializes JSON → `normalize::normalize_request` → `RouterRequest`
2. **Ledger** (pre-pipeline): `ContentNodeLedger::record_request()` writes the full request at LOD0 before any filter runs
3. **Pipeline**: `WorkContext.metadata["request"]` = serialized `RouterRequest`; calls `PipelineOrchestrator::execute`
4. **Stage 1** (`DeterministicPreFilter`): extracts user message from metadata, runs `DeterministicFilterEngine` (chain of `Filter` implementations). Returns `StageDecision` with one of: command result (`/help`, `/stats`, `/checkpoint`), hard reject (API key pattern), output filter flag (PII detected), or pass-through
5. **Stage 2** (`ClassifierStage`): extracts user message, calls LLM via `ChatBackend`, parses `ClassifierOutput` (structured JSON with action, target, coherence/safety scores, complexity, completeness, risk, intent). Checks coherence and safety thresholds. Resolves route via `RoutingConfig::resolve_route()` with complexity-gated model selection. Resolves score matrix for multi-dimensional route ranking. Returns `StageDecision` with `metadata.response` (direct answer), `metadata.routing_target` (dispatch instructions), or rejection verdict
6. **Server** (post-pipeline): `server/handler.rs` reads `PipelineResult` — if `classifier_response` exists, responds directly; if `routing_target` exists, calls `server/dispatch.rs::handle_dispatch`, which walks the primary target plus its `fallbacks` list through `ChatBackend`s (each wrapped in `RetryChatBackend` with exponential backoff `retry_base_interval_s * 1000 * 2^attempt`), short-circuiting on non-retryable errors; if no target, dispatches to the classifier fallback model or a canned fallback response
7. **Ledger** (post-pipeline): `ContentNodeLedger::record_result()` updates the ledger entry with acceptance score and metadata

## Config-driven pipeline assembly

Pipelines are defined in `env/coral-router.json` under the `pipelines` key. Each pipeline entry controls:

```json
{
    "pipelines": {
        "default": {
            "deterministic_prefilter": true,
            "classifier": true,
            "classifier_model": "fast",
            "coherence_threshold": 0.70,
            "blacklist": "env/pii-patterns.json",
            "score_matrix": { … }
        }
    }
}
```

`RouterConfig::build_named_pipeline_with_backend()` constructs the pipeline from config, optionally injecting a mock `ChatBackend` for testing. The deterministic pre-filter uses `DeterministicPreFilter::from_config()` when a blacklist path is present, or `DeterministicPreFilter::new()` (which includes built-in PII patterns) when no blacklist is configured.

## Session profiles on models

`ModelEntry` supports a `sessions` field: a map of named session profiles with per-profile `num_ctx`, `sleep_idle_seconds`, and `params` overrides. Example: `qwythos-9b` exposes three profiles — `orchestrator` (large context, never sleep), `code` (medium context), `compact` (small context, short idle timeout). Callers select a profile by constructing the appropriate model config for the session type.

## Ledger: condensed context architecture

The `ContentNodeLedger` stores every request at full detail (LOD0) before the pipeline runs, and records results afterward. This separates durable storage from live working context:

```
User message → Ledger (durable, full detail)
                ↓
         Pipeline stages (read from WorkContext, not from ledger)
                ↓
         Orchestrator/Session (reads condensed summary, not raw history)
```

`collapse_node()` compresses a ledger entry to a higher LOD (less detail), so abandoned approaches and resolved subtasks condense to a single line rather than vanishing. The orchestrator works against this condensed ledger, not raw session history, keeping its own context high-signal.

## Logging: two-stream architecture

Operational logs and audit logs are separate streams with independent retention policies:

| Stream | Format | Retention | Filter | Writer |
|--------|--------|-----------|--------|--------|
| Operational | JSON or text (configurable) | Configurable rolling files | Standard `EnvFilter` | File + optional stderr |
| Audit | Always JSON | Longer retention (90-day default) | `router.audit=info` | Separate file appender |

Configured via `env/coral-router.json` → `logging.audit_log`. The implementation uses `tracing_subscriber::fmt::Layer::boxed()` to erase concrete types per layer, with a 4-arm match (console yes/no × audit yes/no) rather than the previous 8-arm combinatoric explosion.
