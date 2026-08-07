# Coral Router — Architecture

*This document describes the **current** implementation of Coral Router and
which pieces are load-bearing. The aspirational goals and ideal finished
design live in [`VISION.md`](./VISION.md).*

## Source code location

The source code may be referenced at `./src/router/src/` (crate
`fluent-router`), with the binary entry point in `./src/bin/coral-router/`.

## Overview

Coral Router exposes an OpenAI-compatible HTTP endpoint (`POST
/v1/chat/completions` on `:8079`) that runs every incoming request through a
two-stage pipeline before dispatching to a model. The pipeline is built from
`Arc<dyn Component>` units (the Fluent WVR uniform interface) and the server
itself is also a `WorkUnit` — everything is composable.

The architecture follows the design principles in `VISION.md`:
deterministic before probabilistic, cheap before expensive, condensed context
via a ledger, and frontier as a bounded, audited exception.

```
┌─ Request ──────────────────────────────────────────────────────────────┐
│  POST /v1/chat/completions  { model, messages, temperature, ... }      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ HTTP Server (RouterServer) ────────────────────────────────────────────┐
│  hyper HTTP/1.1 server on tokio (server.rs + server/handler.rs)         │
│  normalizes request body → RouterRequest via serde (normalize.rs)       │
│  records initial request in ContentNodeLedger (LOD0)                    │
│  calls PipelineOrchestrator::execute() → WorkOutput::typed(PipelineResult) │
│  on classifier_response: respond directly                               │
│  on routing_target: server/dispatch.rs (ChatBackend chain)              │
│  routes: /health /stats /v1/chat/completions /v1/plan /v1/rigor         │
│          /admin/cache/invalidate, DELETE /admin/cache/{key}             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ PipelineOrchestrator ──────────────────────────────────────────────────┐
│  Vec<Arc<dyn Component>> executed sequentially                          │
│  known stages call StageDecisionProducer::evaluate (typed handoff,      │
│  STAGE_DECISION_KEY); arbitrary components via WorkOutput.data          │
│  decisions accumulate as Vec<StageDecision>                              │
│  short-circuits on StageVerdict::Rejected / Error                       │
│                                                                         │
│  Stage 1: DeterministicPreFilter — deterministic filter engine          │
│    (no model call; Filter trait chain, PII detection, commands)         │
│  Stage 2: ClassifierStage         — single LLM call (or Classification  │
│    Tree engine) → direct response / routing target / rejection          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼──────────────────┐
                    ▼               ▼                  ▼
             Local dispatch   Escalation ladder   plan / rigor routes
             (ChatBackend      (dispatch/escalation  (routes/plan.rs,
              chain)            → frontier modes)      routes/rigor.rs)
```

## Pipeline: two-stage design

The pipeline executes exactly two stages. The `PipelineStage::Router` enum
variant (in `pipeline_types.rs`) is retained as a taxonomy slot; today it is
only emitted by `PipelineOrchestrator` when a stage's `execute` returns an
`Err` (the orchestrator records the failure as a `Router`-stage error
decision before propagating).

| Stage | File | Model call? | Produces |
|-------|------|-------------|----------|
| 1. DeterministicPreFilter | `stages/deterministic.rs` | No | Command result, PII flag, or pass-through |
| 2. ClassifierStage | `stages/classifier.rs` | Yes | Direct response, rejection, or routing target |

The classifier stage has two modes. Flat mode performs a single LLM call that
returns structured JSON — a direct response, a rejection, or a `RoutingTarget`.
Tree mode wraps the M4 `ClassificationEngine` (`stages/tree.rs`): when a
classification tree is configured (`config/classification.rs`), the engine
evaluates the nested tree recursively — filter nodes (hard_reject /
soft_redirect / output_filter), classifier nodes (auto-built prompt, three-axis
JSON verdict), terminal nodes (resolve a `RoutingTarget`), and fallback
children (evaluated when a classifier picks no named child or its LLM call
fails — always resolving to a fallback dispatch *target*, never a classifier
backup) — emitting a per-node `StageDecision` into `metadata.tree_path` and a
`kind = "tree_node"` audit record per visited node. Route-name guessing is
gone; route selection is the tree's job.

## Design Contract

Every pipeline stage implements `Component` (the Fluent WVR supertrait). The
orchestrator never branches on concrete type.

```rust
// Every stage: same trait, same dispatch, zero branching in hot path.
impl WorkUnit for DeterministicPreFilter { … }
impl FieldAccess for DeterministicPreFilter { … }
impl Describable for DeterministicPreFilter { … }
impl_component!(DeterministicPreFilter);
```

**Typed handoff (M5.4).** The two known stages implement
`StageDecisionProducer` (`pipeline_types.rs`); the orchestrator downcasts via
`component_downcast_ref` and calls `evaluate(ctx, prior)` directly — a typed
call that removes the per-stage `StageDecision` serialize→deserialize through
`WorkOutput.data`. The decision is published to the in-process typed store
under `STAGE_DECISION_KEY` (`pipeline.rs`), where `handle_stage_verdict` and
any downstream stage read it by reference. Arbitrary components (test stubs,
pipeline refs) still flow through the `WorkOutput` channel, which remains the
genuine serialization boundary; their serialized decision is deserialized
exactly once and published to the same typed store.

## Source Layout (`src/router/src/`)

### Core types

| File | Role |
|------|------|
| `types.rs` | `RouterRequest`, `RouterResponse`, `RouterMessage`, `RouterChoice`, `Usage` — serde-serializable OpenAI protocol |
| `pipeline_types.rs` | `StageDecision`, `PipelineStage`, `StageVerdict`, `StageDecisionProducer`, `StageMetadata` (typed metadata handoff keys), `PiiVerdict` |
| `pipeline.rs` | `PipelineOrchestrator`, `PipelineResult`, `RoutingTarget` (url/model/group/params/filter_thinking/retry/stream/timeouts/fallbacks), `STAGE_DECISION_KEY` |
| `error.rs` | `ServerError` — the single typed server error (Bind / Http / Addr / transparent `DispatchError`) |
| `config.rs` | `RouterConfig` + sub-config types, split into re-exported submodules: `addr`, `builder` (`PipelineParams`), `classification` (`ClassificationTree`/`ClassificationNode`/`ClassificationChild`), `escalation` (`EscalationLadderConfig`, `FrontierConfig`), `filters` (`RejectPatterns`/`PatternEntry`/`FilterAction`/`FilterScope`/`ConfidenceGate`), `routing` (`RoutingConfig`, `RouteRef`) |
| `normalize.rs` | Thin adapter over `fluent_llm::openai`: OpenAI JSON ↔ `RouterRequest`/`RouterResponse`, `error_response()`, `messages_to_json()`, `parse_openai_stream_delta` |

### Pipeline & Stages

| File | Role |
|------|------|
| `stages/deterministic.rs` | `DeterministicPreFilter` — delegates to `DeterministicFilterEngine`; slash-command dispatch (`/help`, `/stats`, `/checkpoint`) |
| `stages/classifier.rs` | `ClassifierStage` — single LLM call (flat) or the M4 `ClassificationEngine` (tree); emits direct response / routing target / rejection; builds the `RoutingTarget` |
| `stages/tree.rs` | `ClassificationEngine` — recursive nested-tree evaluation; filter / classifier / terminal / fallback nodes; `tree_path` audit trail; `kind = "tree_node"` records |
| `stages/common.rs` | Shared stage helpers — `extract_user_message()`, `get_metadata_string()`, JSON-field ensure helpers |
| `stages/retry_classifier.rs` | `RetryClassifier` — retry-with-backoff decorator over the classifier stage (opt-in behind `classifier_retry_max`) |
| `stages/pipeline_ref.rs` | `PipelineRefStage` — re-usable pipeline stage from named config |

### Filters (MOA_ROUTER_SPEC §2)

| File | Role |
|------|------|
| `filters/mod.rs` | `Filter` trait — `kind() → FilterKind` + `evaluate(ctx) → Option<FilterDecision>`; `FilterKind::{Regex, Whitelist, HnswSimilarity, ModelClassification}`; `FilterDecision::HardReject` / `SoftRedirect` / `OutputFilter`; `FilterContext` with scopes (`Any`, `FrontierBound`, `ContentNodeWrite`); `DeterministicFilterEngine` — ordered chain-of-responsibility, first non-`None` wins |
| `filters/regex_filter.rs` | `RegexFilter` — compiles regexes from `PatternEntry` config; respects `FilterScope`, `ConfidenceGate` (LuhnValid), and `FilterAction` (Redact / Anonymize / Omit) |
| `filters/injection_detect.rs` | `InjectionDetectFilter` — heuristic prompt-injection / system-prompt-exfiltration detection |
| `filters/luhn.rs` | Luhn algorithm validation — secondary check gate for credit-card-number patterns |

### Dispatch

| File | Role |
|------|------|
| `dispatch/backend.rs` | `ChatBackend` trait (`complete` / `stream_complete` — object-safe, per-request params passed as args); `OpenAiChatBackend` (single-attempt HTTP via raw `reqwest`), `RetryChatBackend` (jittered-exponential retry via `common_core::retry`), `FallbackChatBackend` (ordered backend chain) — the single production dispatch path (D4) |
| `dispatch/escalation.rs` | `EscalationLadder` — the load-bearing ladder runtime: local-chain exhaustion → `filter → question → team → turnover` modes, deterministic-first `ContextCache` short-circuit, `ResultPool`-backed parallel slots, `kind = "escalation"` audit records; `EscalationBackends` / `LocalBackend` / `FrontierBackend` role wiring; `dispatch_frontier` bypass for frontier-owned sessions |
| `dispatch/frontier.rs` | `DispatchError` + `is_retryable` (the public error type of `ChatBackend`); wire-format build/parse helpers reserved for the ladder — `OpenAiBackend` (`parse_response` reused by `OpenAiChatBackend`), `Anthropic` Messages-API helpers, `StreamEvent` |

### Session & Orchestration

| File | Role |
|------|------|
| `session.rs` | Thin shim — re-exports `StepStatus` from `fluent-types` (the canonical session node schema is `fluent_types::ContentNode`) |
| `dag_session.rs` | `DependencySession` — DAG-based session composing `fluent_dag::dep_graph::DependencyGraph<String>` for step tracking, checkpoint/rewind, real KV-cache snapshot restore (model/adapter/session keyed), frontier-ownership flag; `SessionRegistry` — the canonical server-side session home (D6), per-`session_id`, shared `KvCacheManager`, retained for process lifetime |
| `ledger.rs` | `ContentNodeLedger` — **thin facade** over the shared `NodeStore` (M4); owns the LOD lifecycle (LOD0/LOD5 eager, LOD1–4 lazy from LOD0 via `Summarizer`, at most once); `CompactionStrategy` / `RecencyCompaction` (folded in from the deleted `compaction.rs`); routes all writes through the M1 write-path scrub |
| `node_store.rs` | `NodeStore` (M4) — the shared store: nodes behind `Arc<RwLock<ContentNode>>`, interned `ArcIntern<str>` session/role index keys, durable `content_json` hydration (seeded `next_id` from `MAX(node_id)`), `ensure_tier` / `lod_text` / `session_node_ids` render primitives, `knn_brute_force` |
| `ledger_guard.rs` | `ScrubGuard` (M1) — the irreversible write-path scrubber (`scrub_for_ledger`), decision D1; Redact/Anonymize collapse to `[REDACTED:<pattern>]`, no codeword map retained; uses the builtin filter engine with the `ContentNodeWrite` scope |
| `views.rs` | `LedgerView` (M2) — the reference-only view layer over `NodeStore`; `Lod` (0..=5), `ParallelLedger` (one store, N views), `FilteredLedger<V>` (exclusion set + render transform); `render()` is the single text-exit; rendering degrades to LOD0 when a lazy tier is un-derivable |
| `knowledge.rs` | `KnowledgeCapability` impl on `NodeStore` (M4D) behind the `RouterKnowledgeCapability` token — the cross-crate read path for embedded consumers |

### Routes

| File | Role |
|------|------|
| `routes/plan.rs` | `PlanRoute` (M7/M8) — boot-loaded `ChartStore` + `ChartSelector`; Exact → server-side chart compile+execute under `Zone` supervision; Partial → one-round targeted interview (≤ `CHART_MAX_INTERVIEW_QUESTIONS`); Mismatch → fresh draft; `workflow_extractor` hook for the dispatch learning loop |
| `routes/rigor.rs` | `RigorRoute` (M3) — fixed-pass blue/red/judge protocol; real `DependencySession` checkpoint (`rigor.blue`) + `rewind_to_checkpoint` on a material rejection; red team reads through `FilteredLedger` at `Lod::LOD0` (dead ends excluded); final rejection resolves to a targeted interview (≤ 3 questions), frontier escalation only on low judge confidence; `/v1/rigor` is present-but-unconfigured when no backends are attached (explicit error, never a crash) |
| `charts/` | Chart (DAG workflow) library — `store` (`ChartStore`), `binding` (`Entity`, `ENTITIES_META_KEY`), `compile`, `execute` (under `Limiter` + Zone), `render`, `rubric`, `select` (`ChartSelector`, `ChartFit`), `extract` (`WorkflowExtractor`) — the M6–M10 workflow engine consumed by `PlanRoute` and the dispatch learning loop |

### Infrastructure

| File | Role |
|------|------|
| `server.rs` | `RouterServer` (`WorkUnit`) — hyper HTTP/1.1 accept loop on tokio; assembles `ServerDeps` and fans out to the `server/` submodule; `serve_http` is `pub(crate)` for integration tests |
| `server/handler.rs` | HTTP routing + request orchestration; `ServerDeps` (the collapsed former 12-Option dependency bundle): pipelines, routes, models, stats, cache, ledger, plan/rigor routes, sessions, ladders, context_cache, mock_dispatch, http_client |
| `server/dispatch.rs` | `handle_dispatch` / `dispatch_real` — primary + `fallbacks` chain through `ChatBackend` (each wrapped in `RetryChatBackend`), short-circuit on non-retryable errors, response cache read/write, M10 workflow extraction |
| `server/responses.rs` | OpenAI-completion response builders, SSE/CORS headers, `ServerStats` counters |
| `streaming.rs` | `StreamingHandler` — SSE delta formatting for OpenAI-compatible streaming chunks; cross-chunk think-block filtering via `StreamingThinkFilter` |
| `kv_cache.rs` | Two-tier: `HotKvCache` (RAM LRU over `common_core::cache::LoadCache`, metadata only) + `ColdKvCache` (disk tree `model/adapter/session`); `KvCacheManager` composes both; the router never reads/writes raw KV bytes — it manages filesystem layout + sidecar metadata for llama.cpp slot save/restore |
| `instances.rs` | Instance-pool grammar generation + validation (`instance_grammar_string`, `validate_instances`) and the M4 sidecar: `InstanceClient` (fork management API over raw `reqwest`, `HttpClass`-classified) + `InstanceManager` (boot reconcile, `/memory` residency loop with LRU eviction, allocate-on-503) |
| `scheduler.rs` | Re-exports `AffinityScheduler` / `ScheduledTask` / `AgingConfig` from `fluent_concurrency::affinity` |
| `summarization.rs` | `ResultScorer` + `Summarizer` — `WorkUnit` impls that call an LLM (via `Arc<dyn ChatBackend>`) to score/condense responses; feeds the ledger's lazy LOD tiers |
| `score_matrix.rs` | `ScoreMatrix` — multi-dimensional weighted scoring (coherence/complexity/completeness/risk) with per-route dimension bands |
| `metrics.rs` | `FailureClass` + `classify_error` — typed-first error classification with a string-regex fallback for opaque shell/command output (D10) |
| `audit.rs` | The canonical durable-audit surface — a single `tracing` target `router.audit`; `AuditRecord` + `emit(kind, detail)`; audit kinds are distinguished by the `kind` field (`route`, `filter`, `tree_node`, `escalation`, `rigor`, `chart_target`, …) |
| `logging.rs` | Two-stream `tracing` subscriber: operational JSON/console rolling file + the durable audit stream (separate retention, always JSON, gated on `router.audit=info`) |
| `frontier/modes.rs` | `EscalationMode` ladder taxonomy — `Filter`, `Question`, `Team`, `Turnover` (D8; the old `FrontierMode` enum is gone), serde snake_case; `FrontierResult` and `AuditEntry`. Taxonomy and audit types only — the runtime lives in `dispatch/escalation.rs` |
| `hnsw.rs` | `HnswIndexHandle` — the single HNSW index handle type for the chart store's brute-force / `knn_brute_force` fallback |
| `transforms/` | `TransformStrategy` trait + `rewrite_text_messages` shared helper (M7.5/D9): `NoTransform`, `PiiAnonymize`, `DecomposeToAnonymizedHypothetical`, `DecomposeToSubtasks`, `CodewordAnonymizer`, `Sanitize`, `SecretMask` |
| `testing/` | `TranscriptProvider`, `MockTranscriptEntry`, `MockDispatchContext` — transcript-driven integration-test harness for E2E and golden tests |
| `test_stubs.rs` | `StubChatBackend`, `HashEmbedder` — test-only backends (cfg(test)) |

### Adapter architecture (`dispatch/`)

The **production dispatch path** runs through `dispatch/backend.rs`, which
defines the object-safe `ChatBackend` trait that every server dispatch site
depends on (the single dispatch trait, D4):

```rust
pub trait ChatBackend: Send + Sync {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>>;
    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>>;
}
```

Concrete backends: `OpenAiChatBackend` (single-attempt HTTP through a raw
`reqwest::Client`; non-2xx status classified via `HttpClass` into
`DispatchError::RateLimited` vs `DispatchError::Http`), `RetryChatBackend`
(jittered-exponential retry via `common_core::retry::retry_async` — the single
backoff helper), and `FallbackChatBackend` (ordered backend chain that
short-circuits on terminal 4xx). `server/dispatch.rs::dispatch_real` iterates
the primary `RoutingTarget` plus its `fallbacks` list, wrapping each target in
a retry backend. The `fallbacks` are *target* candidates — populated at
route-resolution time by `RoutingConfig::all_dispatch_targets` from the
route's group plus cross-group models, ordered by intelligence proximity to
the request complexity (primary group first, cost as tie-break) — not backups
for the classifier. Streaming flows through `StreamingHandler` over an
`http_body_util` channel. Request bodies are built by the canonical
`fluent_llm::openai::build_openai_chat_body` (which carries the
`chat_template_kwargs: {"enable_thinking": false}` default).

`dispatch/escalation.rs` owns the ladder runtime. After every local model in a
`model_group` chain fails, `try_escalate` consults the deterministic
`ContextCache` first (short-circuit before any frontier call), then runs the
configured modes in order. Each mode's frontier transport reuses
`dispatch/backend.rs` (`ChatBackend`) — no third HTTP path — and every
interaction emits a `kind = "escalation"` audit record (`mode`/`accepted`/
`payload`/`raw_response`/`trigger`/`timestamp`). Turnover marks the session
frontier-owned (`DependencySession::set_frontier_owned`); subsequent requests
in that session bypass the pipeline via `dispatch_frontier`.

`dispatch/frontier.rs` owns the wire-format build/parse logic reserved for the
ladder: `DispatchError` + `is_retryable`, `OpenAiBackend` (whose
`parse_response` is reused by `OpenAiChatBackend`), the `Anthropic`
Messages-API helpers, and `StreamEvent`. The old `DispatchBackend` trait,
`LlmDispatcher`, `ProviderConfig`, and `OpenAiCompatBackend` were deleted by
the M3 dispatch collapse (D4).

### Filter engine architecture (`filters/`)

Filters follow the **Chain of Responsibility** pattern (GoF). The `Filter`
trait declares two methods — `kind()` (one of `FilterKind::{Regex, Whitelist,
HnswSimilarity, ModelClassification}`) and `evaluate`:

```rust
trait Filter: Send + Sync {
    fn kind(&self) -> FilterKind;
    fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision>;
}
```

`DeterministicFilterEngine` holds `Vec<Box<dyn Filter>>` and evaluates filters
in order, returning the first non-`None` decision. Built-in filters:
`RegexFilter` (compiled from `PatternEntry` config) and `InjectionDetectFilter`
(heuristic prompt-injection detection). Filters gate on `ConfidenceGate`
(Luhn validation) and scope themselves via `FilterContext` — `FrontierBound`
(only apply to traffic heading to frontier) and `ContentNodeWrite` (always
apply on the ledger write path, decision D1). `FilterDecision::OutputFilter`
carries `RegexMatch` structs with position data so the `CodewordAnonymizer`
can do consistent, position-aware substitution. The same engine backs both the
pipeline pre-filter and the M1 write-path scrubber (`ledger_guard.rs`).

## Key Compositions & Reusable Primitives

| Primitive | Source | Used by router at |
|-----------|--------|-------------------|
| `Component` / `WorkUnit` | `fluent-wvr` | Every pipeline stage, `PipelineOrchestrator`, `RouterServer`, `ResultScorer`/`Summarizer` |
| `DependencyGraph<K>` | `fluent-dag::dep_graph` | `DependencySession` for step DAG tracking |
| `ResultPool` | `fluent-concurrency::pool` | `dispatch/escalation.rs` — parallel classifier slots (team mode) |
| `PriorityResultPool` | `fluent-concurrency::pool` | `AffinityScheduler` — priority dispatch with aging |
| `Limiter` | `fluent-concurrency::pool` | `ClassifierStage` — concurrent classifier call cap; `charts/compile.rs` + `charts/execute.rs` — chart-DAG execution cap; `PlanRoute` |
| `WorkContext` | `fluent-wvr` | Carries request, caps, runtime through every stage |
| `Runtime` trait | `fluent-wvr` | Plugged via `fluent_concurrency::tokio_runtime()` everywhere |
| `LoadCache<K,V,E>` | `common-core::cache` | `HotKvCache` — bounded get-or-load LRU |
| `ArcIntern<str>` | `internment` | `NodeStore` session/role index keys; work-unit and graph asset names |
| `LatencyHistogram` | `common-core::metrics` | `Instrumented::with_metrics` wiring |
| `retry_async` | `common-core::retry` | `RetryChatBackend`, `RetryClassifier`, `Zone` retries |
| `make_hnsw()` / `knn_brute_force` | `common-core::sqlite` / `fluent-db::vector` | `NodeStore` KNN; `hnsw.rs` chart-store fallback |
| `HttpClass` | `guidance-llm` | `dispatch/backend.rs` — status classification in `OpenAiChatBackend` (streaming + buffered) |
| `DispatchError::is_retryable()` | `fluent-router` (`dispatch/frontier.rs`) | retry/fallback decisions in `dispatch/backend.rs` and `server/dispatch.rs` |
| `LlmError::is_retryable()` | `fluent-concurrency::llm_queue` | `guidance-llm` client error classification |
| `parse_json_response` | `fluent_llm::parse` | `routes/rigor.rs` (red/judge parses), `dispatch/escalation.rs` |
| `Decomposer` | `fluent_llm` | `dispatch/escalation.rs` — question-mode hypothetical decomposition |

## HttpClass: where it lives and why

`HttpClass` (`HardReject`, `TransientFailure`, `EscalationRequired`,
`UpstreamFailure`) is defined in `guidance-llm/src/http_class.rs` and
re-exported via `fluent_llm::HttpClass`. It is consumed in two layers:

1. **`LlmClient`** (in `guidance-llm`) — checks HTTP status before parsing
   the response body; a non-2xx status short-circuits with `LlmError::RateLimited`
   (retryable) or `LlmError::Api` (permanent).

2. **Router dispatch backends** (in `dispatch/backend.rs`) — the router
   dispatches with a raw `reqwest::Client` through `OpenAiChatBackend` (not
   `LlmClient`), so it applies `HttpClass` directly. Both `complete` (buffered)
   and `stream_complete` (streaming) use the identical pattern:
   `HttpClass::from_status(status)` → `is_retryable()` →
   `Err(DispatchError::RateLimited)` (retry) vs `Err(DispatchError::Http)`
   (permanent). Retries are applied by `RetryChatBackend`, and the
   primary-plus-`fallbacks` chain is walked by `server/dispatch.rs::dispatch_real`.

The router's own error taxonomy mirrors this at a higher level:
`DispatchError::is_retryable()` returns `true` for `Http(_)` and `RateLimited`.
Separately, `LlmError::is_retryable()` (in `fluent-concurrency::llm_queue`)
classifies `guidance-llm`/queue errors: `Http(_)` and `RateLimited` are
retryable; `Api(_)` and `NoResponse` are permanent. Both are error-level
classifications independent of how the error was produced.

## Import Boundaries (enforced)

Following AGENTS.md: `fluent-router` imports from `common-core`, `fluent-wvr`,
`fluent-concurrency`, `guidance-llm`, `fluent-types`, `fluent-dag`, and
standard library / `tokio` / `reqwest`. It does NOT import from `guidance`,
`coral`, `wasm_ipc`, `knowledge`, `ontology`, or `rdf`. `knowledge.rs` gives
coral's Context a reachable read path without the router importing coral.

## Pipeline data flow detail

1. **Server**: hyper reads the HTTP request; `server/handler.rs` collects the
   body (enforcing `max_payload`), deserializes JSON →
   `normalize::normalize_request` → `RouterRequest`.
2. **Ledger** (pre-pipeline): `ContentNodeLedger::record_request()` writes the
   full request at LOD0 before any filter runs (through the M1 write-path
   scrub).
3. **Pipeline**: `WorkContext.structured["request"]` = serialized
   `RouterRequest`; the orchestrator calls each stage via `StageDecisionProducer`
   (typed handoff) or the `WorkOutput` channel.
4. **Stage 1** (`DeterministicPreFilter`): extracts the user message, runs
   `DeterministicFilterEngine` (chain of `Filter` implementations). Emits a
   `StageDecision`: command result (`/help`, `/stats`, `/checkpoint`), hard
   reject, output-filter flag (PII detected), or pass-through.
5. **Stage 2** (`ClassifierStage`): extracts the user message, calls the LLM
   via `ChatBackend` (or the classification-tree engine in tree mode), parses
   the structured JSON verdict (action, target, coherence/safety/complexity
   scores, reason). Checks coherence and safety thresholds. Resolves the route
   via `RoutingConfig::resolve_route()` with complexity-gated model selection
   and optional score-matrix ranking — or, when the pipeline opts in
   (`target_match: "self_assess"`), via the shared `TargetMatcher`, which runs
   the in-group target-matching ladder (each candidate self-assesses the
   prompt; the first whose `intelligence` meets its assessed complexity — or
   the last member — becomes the primary target). Emits a `StageDecision`
   carrying `metadata.response` (direct answer), `metadata.routing_target`
   (dispatch instructions), or a rejection verdict.
6. **Server** (post-pipeline): `server/handler.rs` reads `PipelineResult` — if
   `classifier_response` exists, responds directly; if `routing_target` exists,
   calls `server/dispatch.rs::handle_dispatch`, which walks the primary target
   plus its `fallbacks` list through `ChatBackend`s (each wrapped in
   `RetryChatBackend`), short-circuiting on non-retryable errors; if no target,
   dispatches to the classifier's model as a fallback *target* (the model the
   classifier ran on now answers the request), or a canned fallback response.
   Fallback models are target models — never a backup for the classifier. When
   dispatch and escalation fail, the per-group `EscalationLadder`
   (`try_escalate`) runs its configured modes, short-circuiting on a
   `ContextCache` hit.
7. **Ledger** (post-pipeline): `ContentNodeLedger::record_result()` updates the
   ledger entry with acceptance score and metadata, and — on the routed and
   classifier-fallback dispatch branches — the matched target's answer text is
   recorded into the ledger node (LOD0) via `record_ledger_result` and into the
   session step via `SessionStepHandle::complete` (best-effort; streaming
   records whatever content is available at stream finalization). If a
   `session_id` is present, the request is tracked as a step in the session
   registry's `DependencySession`.

## Config-driven pipeline assembly

Pipelines are defined in `env/coral-router.json` under the `pipelines` key.
Each pipeline entry controls:

```json
{
    "pipelines": {
        "default": {
            "deterministic_prefilter": true,
            "classifier": true,
            "classifier_model": "fast",
            "coherence_threshold": 0.70,
            "blacklist": "env/pii-patterns.json",
            "score_matrix": { … },
            "target_match": "self_assess",
            "target_match_timeout_ms": 300000
        }
    }
}
```

`target_match` (`"self_assess"` default | `"static"`) selects the in-group
target-matching policy (§"Model-group target selection"); `target_match_timeout_ms`
(default `DEFAULT_TOTAL_TIMEOUT_MS`) bounds each self-assessment call.

`RouterConfig::build_named_pipeline_with_backend()` constructs the pipeline
from config, optionally injecting a mock `ChatBackend` for testing. The
deterministic pre-filter uses `DeterministicPreFilter::from_config()` when a
blacklist path is present, or `DeterministicPreFilter::new()` (which includes
built-in PII patterns) when no blacklist is configured.

## Model-group target selection: an in-group target-matching ladder

`env/coral-router.json` gives every model an `intelligence` score (0–10) and
every `model_group` an ordered list of model keys (e.g. `"default":
["swarm", "qwen3.6-27b"]`). Selection within a group is complexity-gated, in
one of two modes controlled per pipeline by `pipelines.<name>.target_match`:

- **`target_match: "self_assess"`** (default) — the VISION ladder. At
  route-resolution time inside the classifier stage, `TargetMatcher`
  (`target_match.rs`) climbs the group: each candidate target self-assesses
  the request's complexity via its own `ChatBackend` call (the same shape as a
  classifier call, bounded by `target_match_timeout_ms` under the shared
  `Limiter`). The first candidate whose assessed complexity does not exceed its
  `intelligence` — or the last member of the group — is the matched target.
  The classifier's own complexity estimate only seeds the *start* index (§4.1
  of the roadmap): the cheapest candidate whose `intelligence` meets the
  estimate self-assesses first, so the climb never skips a candidate the
  classifier already ruled out as too weak. The ladder is DRY-shared between
  the flat classifier path and the classification-tree engine, and runs only
  for 2+ member groups (single-member groups and `"static"` resolve
  byte-identically to today). Every self-assessment and the final match emit a
  `kind = "target_match"` audit record.
- **`target_match: "static"`** — today's behavior. `RoutingConfig::resolve_route`
  picks the cheapest model in the route's group whose `intelligence` meets the
  classifier's `complexity` score; if none qualifies, it picks the cheapest in
  the group.

In both modes, `RoutingConfig::routing_target` populates `RoutingTarget.fallbacks`
via `all_dispatch_targets` — every model across the group, ordered by
intelligence proximity to the request complexity (primary group first, cost as
tie-break). The ladder reorders the primary/first fallbacks: the matched
target becomes the primary and its more-intelligent group tail `G[i+1..=n]`
leads the fallback list (mechanical-failure walk, in order), followed by any
cross-group models from `all_dispatch_targets` not already included. These are
*target* candidates, and a `fallback` tree child resolves through the same
path. `dispatch_real` (`server/dispatch.rs`) walks the primary target plus its
`fallbacks` in order when a target fails (rate limit, timeout, parse error);
non-retryable 4xx errors short-circuit the chain. Only after the whole local
chain is exhausted does the per-group `EscalationLadder` engage
(`dispatch/escalation.rs`).

Every model in the chain is a candidate to answer the request — a fallback
*target*. None of them backs up the classifier: the classifier stage runs on
its own `classifier_model`, and when the pipeline produces no target the
handler dispatches to that classifier model as a fallback target
(`server/handler.rs`) rather than to a classifier backup. The matched
target's answer is recorded in the session ledger and session step after
dispatch (§"Pipeline data flow detail", step 7).

## Instance pools and the sidecar

`ModelEntry` supports an `instances` field (the old `sessions` key is a serde
alias): a map of `InstanceProfile`s declaring the fork's shared-weight
instances for that model. The fork's `llama-server` (see
`LLAMA_CPP_SERVER_INSTANCES.md` at the repo root) loads a `(base, variant)`
weight pool once and serves many named instances — separate KV + compute
buffers sharing those weights. Requests route to an instance by the model-id
grammar (`<base>`, `<base>:<group>`, `<base>:<instance>`, ...). The fork no
longer reads `num_ctx`/`parallel`/`sleep_idle_seconds` from the request body;
those are declaration-only and coral-router strips them from dispatched bodies.

A profile declares `count` sibling instances (reinterpreted from the old
`parallel`), each with its own KV. For `count > 1` the profile expands to
`<name>0..<name>{count-1}` sharing the profile's `group`. Example: the `swarm`
model declares three 16384-ctx instances in group `swarm`, plus a pinned
131072-ctx `ledger` instance (the default dispatch point, targeted by a bare
`<base>` request) and a 131072-ctx `scratch` instance that auto-sleeps after
30s idle. The dispatch grammar generator
(`instances::instance_grammar_string`) emits the exact `--instance` flags the
operator hands to `llama-server`, matching the fork's
`common_instances_to_string` byte-for-byte:

```
--instance "swarm0:group=swarm:ctx=16384:sleep=0" \
--instance "swarm1:group=swarm:ctx=16384:sleep=0" \
--instance "swarm2:group=swarm:ctx=16384:sleep=0" \
--instance "ledger:ctx=131072:pinned:default" \
--instance "scratch:ctx=131072:sleep=30"
```

Dispatch is encoded through the model id: `RoutingTarget::from_model_entry`
qualifies `model` to `<base>:<qualifier>` (the `default` profile's group, else
the single shared group, else bare `<base>`). `RoutingTarget::from_model_entry_instance`
targets a named point (`<base>:ledger`, `<base>:scratch`) for callers like the
ledger summarizer and any on-demand scratch route. `snapshot`/`id_slot`/
`instance` travel as explicit top-level request fields (`build_chat_body` adds
them only when set).

**Sidecar.** coral-router acts as the sidecar the fork's docs describe
(`config.sidecar`). At boot each model endpoint that declares an instance pool
gets an `InstanceManager` (`instances.rs`) that reconciles configured
instances against `GET /instances`, creating missing ones (`POST /instances`),
resizing `n_ctx` drift, and tolerating a 409 duplicate. A residency loop polls
`GET /memory` and, when free VRAM drops below `vram_low_watermark_bytes`,
evicts up to `evict_batch` least-recently-used unpinned instances
(`DELETE /instances/:name`; `pinned` instances are never evicted). On a 503
`"no free instance in group"` group-miss, dispatch calls
`InstanceManager::ensure_group` to allocate a fresh `<group>-<uuid>` instance
before retrying once. `config.sidecar.slot_save_path` feeds the `KvSnapshot`
`file_path` derivation so the router's snapshot metadata and the server's
`--slot-save-path` layout agree. The management client reuses the raw-reqwest
pattern of `OpenAiChatBackend` with `HttpClass`-classified errors.

## Ledger: condensed context architecture

`ContentNodeLedger` is a thin facade over the shared `NodeStore`. Every request
is stored at full detail (LOD0) before the pipeline runs, and results are
recorded afterward. This separates durable storage from live working context:

```
User message → ContentNodeLedger → NodeStore (durable, full detail)
                ↓                         ↓
         Pipeline stages          ParallelLedger / FilteredLedger
         (read from WorkContext,   (render-only views; single text-exit
          not from ledger)          LedgerView::render → lod_text)
                ↓
         Orchestrator/Session (reads condensed summary, not raw history)
```

Key load-bearing properties:

- **Write path is checked (M1, D1).** Every write reaches `NodeStore` only
  after passing through `ledger_guard::scrub_for_ledger` — the builtin filter
  engine with the `ContentNodeWrite` scope active. PII-matching text is
  irreversibly replaced (`[REDACTED:<pattern>]`), no codeword map retained.
  Direct `NodeStore` writes are the documented bypass (production writes route
  through the facade).
- **LOD lifecycle.** LOD0 (full text) + LOD5 (label) are eager; LOD1–LOD4 are
  derived lazily, always from LOD0 only (never chained), via the `Summarizer`
  WorkUnit, and cached on the node at most once. `CompactionStrategy`/
  `RecencyCompaction` demote older nodes to a higher LOD (setting `active_lod`).
- **Views never own text (M2, D4).** `LedgerView::render` is the single
  text-exit from the store; `ParallelLedger` gives independent default-LOD
  views over one shared `Arc<NodeStore>`; `FilteredLedger<V>` is a reference
  overlay (exclusion set + optional render transform) used by both the PII
  frontier view and the rigor red-team view. Rendering degrades to LOD0 when a
  lazy tier is un-derivable rather than erroring.
- **Shared store (M4).** Nodes live once behind `Arc<RwLock<ContentNode>>`
  with interned `ArcIntern<str>` session/role index keys and durable
  `content_json` hydration (seeded `next_id` from `MAX(node_id)` so restarts
  never re-issue colliding ids).

## Logging: two-stream architecture

Operational logs and audit logs are separate streams with independent
retention policies:

| Stream | Format | Retention | Filter | Writer |
|--------|--------|-----------|--------|--------|
| Operational | JSON or text (configurable) | Configurable rolling files | Standard `EnvFilter` | File + optional stderr |
| Audit | Always JSON | Longer retention (90-day default) | `router.audit=info` | Separate file appender |

Every audit producer emits through `audit::emit(kind, detail)` into the single
`router.audit` `tracing` target; audit kinds are distinguished by the `kind`
field, never by a second dot-namespace. Configured via `env/coral-router.json`
→ `logging.audit_log`. The implementation uses `tracing_subscriber::fmt::Layer::boxed()`
to erase concrete types per layer, with a 4-arm match (console yes/no × audit
yes/no).
