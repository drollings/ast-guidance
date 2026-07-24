# Coral Router — Architecture

## Overview

Coral Router exposes a single OpenAI-compatible HTTP endpoint (`POST /v1/chat/completions` on `:8081`) that runs every incoming request through a three-stage pipeline before dispatching to a model. The pipeline is built from `Arc<dyn Component>` units (the Fluent WVR uniform interface) and the server itself is also a `WorkUnit` — everything is composable.

```
┌─ Request ──────────────────────────────────────────────────────────────┐
│  POST /v1/chat/completions  { model, messages, temperature, ... }      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ HTTP Server (RouterServer) ────────────────────────────────────────────┐
│  raw tokio::TcpListener, hand-parsed HTTP/1.1, CORS                     │
│  normalizes request body → RouterRequest via serde                      │
│  calls PipelineOrchestrator::execute() → WorkOutput::typed(PipelineResult) │
│  on classifier_response: respond directly                               │
│  on routing_target: dispatch_to_frontier() via reqwest                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ PipelineOrchestrator ──────────────────────────────────────────────────┐
│  Vec<Arc<dyn Component>> executed sequentially                          │
│  each stage reads/writes via WorkContext.metadata + WorkOutput.data     │
│  decisions accumulate as Vec<StageDecision>                              │
│  short-circuits on StageVerdict::Rejected / Error                       │
│                                                                         │
│  Stage 1: DeterministicPreFilter  — regex blacklist + /commands        │
│    (no model call; pure regex, PII detection)                           │
│  Stage 2: ClassifierStage         — single LLM call                    │
│    returns structured JSON: { action, response, target, coherence, ... }│
│    replaces QualityGate + PlanningRefinement + GuardrailCheck           │
│  Stage 3: RouterStage             — selects destination                │
│    emits RoutingDecision: LocalAgent or Frontier                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                   ▼
            Local Agent      Frontier API          Watchdogs
          (AgentRegistry)  (FrontierDispatcher)  (Token/WallClock/Repeat)
```

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
| `pipeline_types.rs` | `StageDecision`, `PipelineStage`, `StageVerdict` — the structured record every stage emits |
| `config.rs` | `RouterConfig`, `PipelineConfig`, `ModelEntry`, `RoutingConfig`, all sub-configs — deserialized from JSON |

### Pipeline

| File | Role |
|------|------|
| `pipeline.rs` | `PipelineOrchestrator` — `WorkUnit` that iterates stages, collects decisions, short-circuits on reject/error |
| `normalize.rs` | OpenAI JSON ↔ `RouterRequest`/`RouterResponse` conversion; `error_response()` for OpenAI-format errors |
| `stages/deterministic.rs` | Regex-based PII detection and slash-command dispatch (`/help`, `/stats`, `/checkpoint`) |
| `stages/classifier.rs` | Single LLM call; returns structured `ClassifierOutput`; resolves route via `RoutingConfig::resolve_route` |
| `stages/router.rs` | `RoutingPolicy` enum (LocalFirst, FrontierOnly, CostMinimizing, AutoRouting); emits `RoutingDecision` |

### Dispatch

| File | Role |
|------|------|
| `dispatch/frontier.rs` | `FrontierBackend` trait (OpenAI, Anthropic, OpenAiCompatible); `FrontierDispatcher` with concurrency `Limiter` |
| `dispatch/agent.rs` | `AgentDispatcher` — bridges `RoutingDestination::LocalAgent` to `AgentRegistry` with KV-cache restore/save |
| `agent.rs` | `AgentRegistry` — maps `(model, adapter, session)` triple to `ResultPool<AgentTask, String, AgentError>` |

### Session & Orchestration

| File | Role |
|------|------|
| `session.rs` | `SessionNode` — graph DB node with role, turn_index, accepted, LOD level, step_id, step_status |
| `orchestrator.rs` | `OrchestratorSession` — linear session with recency compaction, checkpoint/rewind, LLM client |
| `dag_session.rs` | `DependencySession` — DAG-based session composing `fluent_dag::dep_graph::DependencyGraph<String>` for step dependency tracking, checkpoint/rewind, KV-cache snapshot restore |
| `compaction.rs` | `CompactionStrategy` trait; `RecencyCompaction` demotes older nodes to higher LOD (less detail) |

### Infrastructure

| File | Role |
|------|------|
| `server.rs` | `RouterServer` (`WorkUnit`) — raw TCP listener, HTTP/1.1 parser, CORS, pipeline invoke, frontier dispatch |
| `kv_cache.rs` | Two-tier: `HotKvCache` (RAM LRU) + `ColdKvCache` (disk tree `model/adapter/session`); `KvCacheManager` composes both |
| `scheduler.rs` | `AffinityScheduler` wrapping `PriorityResultPool` — affinity bonus + aging for KV-cache affinity |
| `watchdog.rs` | `WatchdogSet` composing `MaxTokenWatchdog`, `WallClockWatchdog`, `RepetitionWatchdog` |
| `streaming.rs` | `StreamingHandler` — SSE delta formatting for OpenAI-compatible streaming chunks |
| `summarization.rs` | `ResultScorer` + `Summarizer` — both `WorkUnit` impls that call an LLM to score/condense responses |
| `transforms/` | `TransformStrategy` trait: `NoTransform`, `PiiAnonymize`, `DecomposeToAnonymizedHypothetical`, `DecomposeToSubtasks` |
| `indexer.rs` | `AdapterIndexer` — eager validation of LoRA adapter files at startup |
| `metrics.rs` | `RouterMetrics` — per-model/agent latency histograms, stage verdict counters, watchdog/error counters |
| `logging.rs` | `tracing` subscriber with JSON rolling file output + stderr |

### Adapter architecture (`dispatch/frontier.rs`)

The frontier adapter follows the **Strategy** pattern:

```rust
pub trait FrontierBackend: Send + Sync {
    fn provider_name(&self) -> &str;
    fn build_request(&self, request: &RouterRequest) -> Result<Value, FrontierError>;
    fn parse_response(&self, body: &Value) -> Result<RouterResponse, FrontierError>;
    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, FrontierError>;
}
```

Three backends ship: `OpenAiBackend`, `AnthropicBackend`, `OpenAiCompatibleBackend`. The `FrontierDispatcher` holds a `HashMap<String, ProviderConfig>` and a `Limiter` for concurrency capping.

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

## Key Compositions & Reusable Primitives

| Primitive | Source | Used by router at |
|-----------|--------|-------------------|
| `Component` / `WorkUnit` | `fluent-wvr` | Every pipeline stage, `PipelineOrchestrator`, `RouterServer` |
| `DependencyGraph<K>` | `fluent-dag::dep_graph` | `DependencySession` for step DAG tracking |
| `ResultPool` | `fluent-concurrency::pool` | `AgentRegistry` — one pool per `(model, adapter, session)` |
| `PriorityResultPool` | `fluent-concurrency::pool` | `AffinityScheduler` — priority dispatch with aging |
| `Limiter` | `fluent-concurrency::pool` | `FrontierDispatcher` — concurrent frontier call cap |
| `WorkContext` | `fluent-wvr` | Carries request, caps, runtime through every stage |
| `Runtime` trait | `fluent-wvr` | Plugged via `fluent_concurrency::tokio_runtime()` everywhere |
| `LatencyHistogram` | `common-core::metrics` | `RouterMetrics` — per-model and per-agent latency |

## Import Boundaries (enforced)

Following AGENTS.md: `fluent-router` imports from `common-core`, `fluent-wvr`, `fluent-concurrency`, `guidance-llm`, `fluent-types`, `fluent-dag`, and standard library / `tokio` / `reqwest`. It does NOT import from `guidance`, `coral`, `wasm_ipc`, `project-knowledge`, `ontology`, or `rdf`.

## Pipeline data flow detail

1. **Server**: reads raw TCP bytes, parses headers, deserializes JSON body → `normalize::normalize_request` → `RouterRequest`
2. **Pipeline**: `WorkContext.metadata["request"]` = serialized `RouterRequest`; calls `PipelineOrchestrator::execute`
3. **Stage 1** (`DeterministicPreFilter`): extracts user message from metadata, runs regexes for PII + command dispatch. Returns `StageDecision(PipelineStage::DeterministicPreFilter, verdict)` via `WorkOutput::typed`
4. **Stage 2** (`ClassifierStage`): extracts user message, calls LLM, parses `ClassifierOutput`. Returns `StageDecision` with `metadata.response` or `metadata.routing_target`
5. **Stage 3** (`RouterStage`): returns `StageDecision` with `metadata.routing_decision` (local agent or frontier)
6. **Server** (post-pipeline): reads `PipelineResult` — if `classifier_response` exists, responds directly; if `routing_target` exists, dispatches to frontier via `reqwest`; if no target, dispatches to fallback frontier URL
