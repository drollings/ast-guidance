# Coral Router — Architecture

*This document describes the **current** implementation of Coral Router and
which pieces are load-bearing. The aspirational goals and ideal finished
design live in [`VISION.md`](./VISION.md).*

## Source code location

The source code may be referenced at `./src/router/src/` (crate
`fluent-router`), with the binary entry point in `./src/bin/coral-router/`.

## Overview

Coral Router exposes an OpenAI-compatible HTTP endpoint (`POST
/v1/chat/completions` on `:8079`) that runs every incoming request through an
ordered stage pipeline (deterministic pre-filter → optional NLP/overlay →
classifier) before dispatching to a model. The pipeline is built from
`Arc<dyn Component>` units (the Fluent WVR uniform interface) and the server
itself is also a `WorkUnit` — everything is composable.

The architecture follows the design principles in `VISION.md`:
deterministic before probabilistic, cheap before expensive, condensed context
via a ledger, and frontier as a bounded, audited exception.

Local serving is owned, not proxied. Coral Router spawns and supervises one
`llama-server` process per model weights file (`supervisor.rs`), serves the
`/instances` management contract at its own address (`server/instances_api.rs`),
and is the single routing element between those llama-server tasks and every
other OpenAI-compatible endpoint. The llama.cpp router mode is never used; a
local dispatch is a direct HTTP call to the owning server.

```
┌─ Request ──────────────────────────────────────────────────────────────┐
│  POST /v1/chat/completions  { model, messages, temperature, ... }      │
│  routing fields (model/instance/snapshot/id_slot) also from query      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ HTTP Server (RouterServer) ────────────────────────────────────────────┐
│  hyper HTTP/1.1 server on tokio (server.rs + server/handler.rs)         │
│  merges query routing fields into the body (body wins), normalizes →    │
│  RouterRequest via serde (normalize.rs)                                 │
│  records initial request in ContentNodeLedger (LOD0)                    │
│  calls PipelineOrchestrator::execute() → WorkOutput::typed(PipelineResult) │
│  on classifier_response: respond directly                               │
│  on routing_target: server/dispatch.rs (ChatBackend chain)              │
│  management: /instances /v1/models /models /memory /props + model-less  │
│  proxies; /health /stats /v1/chat/completions /v1/plan /v1/rigor        │
│  /admin/cache/invalidate, DELETE /admin/cache/{key}; admin /models/unload │
│  + /metrics                                                              │
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
│  ordered stage list (deterministic_first, no model on the cheap path):  │
│   1. DeterministicPreFilter — deterministic filter engine (no model     │
│      call; Filter trait chain, PII detection, slash commands)           │
│   2. NlpStage (opt-in, `nlp: true`) — spacy-rs deterministic tokenize / │
│      parse / interlingua; publishes RoutingSignals under "nlp_parse"    │
│   3. OverlayStage (opt-in, deterministic-first only) — scores the       │
│      parser's residuals with onnx overlays → route hints                │
│   4. ClassifierStage — single LLM call (or Classification Tree engine)  │
│      → direct response / routing target / rejection (optionally wrapped │
│      in RetryClassifier)                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼──────────────────┐
                    ▼               ▼                  ▼
             Local dispatch   Escalation ladder   plan / rigor routes
             (ChatBackend      (dispatch/escalation/  (routes/plan.rs,
              chain)            → frontier modes)      routes/rigor/)
                    │
                    ▼
        ┌─ Serving layer (supervisor.rs + instances/) ───────────────────┐
        │  one llama-server per weights file on a free localhost port    │
        │  direct /instances + generation calls, no llama.cpp router     │
        │  sidecar: footprint-weighted eviction (weights included) when  │
        │  over the VRAM budget; admission control before cold loads;    │
        │  resume-marked contexts KV-snapshotted before they drop        │
        └─────────────────────────────────────────────────────────────────┘
```

## Pipeline: ordered stage list

The pipeline is an **ordered list** of `Arc<dyn Component>` stages built from
config (`config/builder.rs::build_named_pipeline_*`). The two decision stages
that resolve a request — the deterministic pre-filter and the classifier —
are the load-bearing core; the NLP and overlay stages are opt-in **enrichment**
between them (they publish parse/route context, never a verdict). The
`PipelineStage::Router` enum variant (in `pipeline_types.rs`) is retained as a
taxonomy slot; today it is only emitted by `PipelineOrchestrator` when a
stage's `execute` returns an `Err` (the orchestrator records the failure as a
`Router`-stage error decision before propagating).

| Stage | File | Model call? | Produces |
|-------|------|-------------|----------|
| 1. DeterministicPreFilter | `stages/deterministic.rs` | No | Command result, PII flag, or pass-through |
| 2. NlpStage *(opt-in)* | `stages/nlp.rs` | *Deterministic base `[ArcEager, Rule]` always; gated refine `[Encoder, Llm]` on confidence OR task-value via `RefinePolicy` (fallback keeps base)* | `RoutingSignal`s under `"nlp_parse"` (predicate/subject/object + interlingua ids) |
| 3. OverlayStage *(opt-in)* | `stages/overlay.rs` | Yes — onnx overlay models (Prompt-Router) | Route hints the classifier merges as deterministic context |
| 4. ClassifierStage | `stages/classifier.rs` | Yes | Direct response, rejection, or routing target |

The NLP and overlay stages are gated on config (`pipelines.<name>.nlp`,
`pipelines.<name>.overlay` + `nlp_ordering: "deterministic_first"`), so the
default pipeline — filter → classifier — is byte-identical to the historical
two-stage shape until a deployment opts in. See §"Natural-language processing
spine" below for the NLP/overlay integration in full.

The classifier stage has two modes. Flat mode performs a single LLM call that
returns structured JSON — a direct response, a rejection, or a `RoutingTarget`.
Tree mode wraps the `ClassificationEngine` (`stages/tree/`): when a
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

**Typed handoff.** The two known stages implement
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
| `types.rs` | `RouterRequest`, `RouterResponse`, `RouterMessage`, `RouterChoice`, `Usage` — serde-serializable OpenAI protocol. `RouterRequest` also carries the routing fields the owning llama-server reads: `instance`, `snapshot`, `id_slot` |
| `pipeline_types.rs` | `StageDecision`, `PipelineStage`, `StageVerdict`, `StageDecisionProducer`, `StageMetadata` (typed metadata handoff keys, incl. the `nlp_parse` handoff key), `PiiVerdict` |
| `pipeline.rs` | `PipelineOrchestrator`, `PipelineResult`, `RoutingTarget` (url/model/group/params/filter_thinking/retry/stream/timeouts/fallbacks, plus `instance`/`snapshot`/`id_slot` request fields), `STAGE_DECISION_KEY` |
| `error.rs` | `ServerError` — the single typed server error (Bind / Http / Addr / transparent `DispatchError`) |
| `config.rs` | `RouterConfig` + sub-config types, split into re-exported submodules: `addr`, `builder` (`PipelineParams`), `classification` (`ClassificationTree`/`ClassificationNode`/`ClassificationChild`), `escalation` (`EscalationLadderConfig`, `FrontierConfig`), `filters` (`RejectPatterns`/`PatternEntry`/`FilterAction`/`FilterScope`/`ConfidenceGate`), `routing` (`RoutingConfig`, `RouteRef` with `always_route`). Top-level `ledger`/`session`/`review`/`overlay` sections (`LedgerConfig`, `ReviewConfig`, `OverlayConfig`). `ModelEntry` adds the weights source for managed models (`weights`, `hf_repo`, `hf_file`), an `onnx` block (`fluent_onnx::config::OnnxConfig` — makes the model router-served from the ort registry instead of a llama-server), and `is_managed()`. Top-level `gguf_dir` is the admin-CLI GGUF root; `sidecar.resume_ttl_s` bounds resume snapshot lifetime |
| `normalize.rs` | Thin adapter over `fluent_llm::openai`: OpenAI JSON ↔ `RouterRequest`/`RouterResponse`, `error_response()`, `messages_to_json()`, `parse_openai_stream_delta`. Re-attaches the routing fields the shared normalizer strips |
| `target_match.rs` | `TargetMatcher` — the in-group complexity ladder: `build_self_assessment_prompt`, `parse_self_assessment`, `candidates_for_group`, `start_index`/`is_match` selection core. DRY-shared by the flat classifier path (`stages/classifier.rs`) and the classification-tree engine (`stages/tree/engine.rs`) |

### Pipeline & Stages

| File | Role |
|------|------|
| `stages/deterministic.rs` | `DeterministicPreFilter` — delegates to `DeterministicFilterEngine`; slash-command dispatch (`/help`, `/stats`, `/checkpoint`) |
| `stages/nlp.rs` | `NlpStage` — runs `spacy-rs` on the inbound text and publishes per-sentence `RoutingSignal`s (predicate/subject/object + interlingua ids) under the `"nlp_parse"` handoff key. Enrichment, never a gate. Deterministic-first ladder: base `[ArcEager, Rule]` always runs; gated refine `[Encoder, Llm]` (in that order) on confidence OR task-value via `RefinePolicy` (fallback keeps base); `annotation_fetch` bridges a `ChatBackend` to `spacy_rs::LlmFetchSync` |
| `stages/overlay.rs` | `OverlayStage` — under `nlp_ordering: "deterministic_first"`, scores the parser's residuals with configured onnx overlays (`PromptRouterOverlay`, from two-tower `zero_shot_routing` sessions via `ort::disambiguation_overlays`) and publishes route hints the classifier merges as deterministic context. Enrichment, never a gate |
| `stages/classifier.rs` | `ClassifierStage` — single LLM call (flat) or the `ClassificationEngine` (tree); emits direct response / routing target / rejection; builds the `RoutingTarget`; enforces route-level `always_route` (override `action=respond` → `route`) and auto-generates the "Dispatch rules" section of the system prompt from routes marked `always_route` |
| `stages/tree/` | `ClassificationEngine` — recursive nested-tree evaluation; filter / classifier / terminal / fallback nodes; `tree_path` audit trail; `kind = "tree_node"` records. Split into `mod.rs` (re-exports), `engine.rs` (the recursive walk + `cost` + the `match_interlingua` filter that dispatches on the NLP stage's parsed interlingua ids), `verdict.rs` (`TreeClassifierVerdict` + `parse_tree_verdict`), `decisions.rs` (`TreeOutcome`/`TreeEvaluation` + the `StageDecision` builders); the classifier-node prompt builders live in `config/classification.rs` (`ClassificationNode::build_prompt`) |
| `stages/common.rs` | Shared stage helpers — `extract_user_message()`, `get_metadata_string()`, JSON-field ensure helpers |
| `stages/prompt_parse.rs` | `chat_json` + `PromptParseError` — the router-local LLM-JSON round-trip codec over `fluent_llm::parse::parse_typed` (call + tolerant parse/coerce in one envelope) |
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
| `dispatch/backend.rs` | `DispatchBackend` trait (`complete` / `stream_complete` — object-safe, per-request params passed as args); `OpenAiChatBackend` (single-attempt HTTP via raw `reqwest`), `RetryBackend` (jittered-exponential retry via `common_core::retry`), `BackendChain` (ordered backend chain) — the single production dispatch path (D4). LLM client trait remains `fluent_llm::client::ChatBackend` |
| `dispatch/escalation/` | `Ladder` — the load-bearing ladder runtime: local-chain exhaustion → `filter → question → team → turnover` modes, deterministic-first `ContextCache` short-circuit, `ResultPool`-backed parallel slots, `kind = "escalation"` audit records; `EscalationBackends` / `LocalBackend` / `FrontierBackend` role wiring; `dispatch_frontier` bypass for frontier-owned sessions. Split into `mod.rs` (`Ladder` + `try_escalate`), `modes.rs` (the four mode implementations + the single shared `frontier_complete` transport), `assemble.rs` (parse/assemble/scorer helpers), `audit.rs` (`emit_audit` record builder) |
| `dispatch/frontier.rs` | `DispatchError` + `is_retryable` (the public error type of `ChatBackend`); wire-format build/parse helpers reserved for the ladder — `OpenAiBackend` (`parse_response` reused by `OpenAiChatBackend`), `Anthropic` Messages-API helpers, `StreamEvent` |

### Session & Orchestration

| File | Role |
|------|------|
| `session.rs` | Thin shim — re-exports `StepStatus` from `fluent-types` (the canonical session node schema is `fluent_types::ContentNode`) |
| `dag_session.rs` | `DependencySession` — DAG-based session composing `fluent_dag::dep_graph::DependencyGraph<String>` for step tracking, checkpoint/rewind, real KV-cache snapshot restore (model/adapter/session keyed), frontier-ownership flag; `SessionRegistry` — the canonical server-side session home (D6), per-`session_id`, shared `SnapshotStore`, retained for process lifetime |
| `ledger.rs` | `ContentNodeLedger` — **thin facade** over the shared `ContentNodeStore`; owns the LOD lifecycle (LOD0/LOD5 eager, LOD1–4 lazy from LOD0 via `Summarizer`, at most once); `CompactionStrategy` / `RecencyCompaction` (folded in from the deleted `compaction.rs`); routes all writes through the write-path scrub; `write_annotation` is the public tiered-annotation facade over `ContentNodeStore::write_annotation` (provenance tiers + `content_hash` keying) |
| `ledger/nlp.rs` | Ledger writer for the NLP parse — `record_parse_node_with_confidence` writes the parse `ContentNode` **and** the per-sentence × role `interlingua_index` rows (id, role, confidence, `review_status='unreviewed'`) in one transaction, so "requests whose `direct_object_id` is X" is a real SQL query. `record_parse_node` is a thin wrapper over it |
| `ledger/overlay.rs` | `OverlayCandidateStore` over `overlay_candidates` — the durable candidate plane for async overlay outputs (entity links, PII-shaped spans, parse corrections, concept summaries). The **only** surface overlays write: nothing ever writes `TokenRecord.interlingua_entity_id`/`concept_ids` at runtime (boot-only registration is an invariant) |
| `ledger/overlay_worker.rs` | `OverlayWorker` — the background arc_ready annotation worker. Composes the shared `CreditedFeedWorker<NodeId>` primitive (the **same** feed shape as `LedgerTierWorker` — it deliberately does **not** reimplement that worker's shape): per node it fans out three independent derivation jobs (`annotation_for` / `embedding_for` / `llm_overlay_for`, each from LOD0, at-most-once, fail-open) **in parallel** on the injected `Runtime`, so a slow LLM never delays the spacy/embedding overlays. Emits a `kind = "overlay"` audit record per derived overlay; boot backfill via `node_ids_needing_overlays()`. Driven by `OverlayKind`/`OverlayState` (`fluent-types`) |
| `ledger/correction_index.rs` | `SqliteCorrectionIndex` — the router's `spacy_rs::CorrectionIndex` over `interlingua_index`. Pattern-cache rows use the sentinel `PATTERN_NODE` node id and `role = "correction"` so they never collide with real parse-node audit rows |
| `ledger/annotations.rs` | `AnnotationStore` — the tiered annotation ledger over `nr_annotations`, keyed `(content_hash, claim_key)`. A higher `ProvenanceTier` upsert supersedes (never deletes) the prior claim; equal/lower marks it `Provisional`. Written via `ContentNodeStore::write_annotation` / `ContentNodeLedger::write_annotation` |
| `ledger/frame_index.rs` | `SqlitePreferredSenseIndex` (the router's `spacy_rs::PreferredSenseIndex` over the `interlingua_index` pattern-cache rows, `role = "sense"`) + the wave-batched `FrameResolutionWorker` — one grammar-constrained call per `SupervisedBatch` wave returning N resolutions, applied + promoted so a repeating `(predicate, ambiguity_kind)` pattern replays deterministically |
| `node_store.rs` | `ContentNodeStore` — the shared store: nodes behind `Arc<RwLock<ContentNode>>`, interned `ArcIntern<str>` session/role index keys, durable `content_json` hydration (seeded `next_id` from `MAX(node_id)`), `ensure_tier` / `lod_text` / `session_node_ids` render primitives, `knn_brute_force`, and the M4 `content_hash` stamping (`insert_node`/`record_result` recompute LOD0's hash in the single write funnel). `write_annotation` routes a tiered `AnnotationClaim` to the `AnnotationStore` under the node's current hash. **ArcReady overlays:** `annotation_for` / `embedding_for` / `llm_overlay_for` derive the three overlays lazily and **at-most-once** from LOD0 (via a shared `derive_overlay` that re-checks under the write lock and marks `ready`/`failed`); fail-open seams (`overlay_pipeline` / `overlay_embedder` / `overlay_llm`) injected by `set_overlay_pipeline` / `set_overlay_embedder` / `set_overlay_llm`; the `overlay_events` feed (`set_overlay_events`, drained by `OverlayWorker`) enqueues nodes missing an overlay on write (`enqueue_if_needs_overlay`, `node_ids_needing_overlays` boot backfill) |
| `ledger_guard.rs` | `scrub_for_ledger` — the irreversible write-path scrubber, decision D1; Redact/Anonymize collapse to `[REDACTED:<pattern>]`, no codeword map retained; uses the builtin filter engine with the `ContentNodeWrite` scope |
| `views.rs` | `LedgerView` — the reference-only view layer over `ContentNodeStore`; `Lod` (0..=5), `ParallelLedger` (one store, N views), `FilteredLedger<V>` (exclusion set + render transform); `render()` is the single text-exit; rendering degrades to LOD0 when a lazy tier is un-derivable |
| `knowledge.rs` | `KnowledgeCapability` impl on `ContentNodeStore` behind the `RouterKnowledgeCapability` token — the cross-crate read path for embedded consumers |

### Routes

| File | Role |
|------|------|
| `routes/plan.rs` | `PlanRoute` — boot-loaded `ChartStore` + `ChartSelector`; Exact → server-side chart compile+execute under `SupervisedBatch` supervision; Partial → one-round targeted interview (≤ `CHART_MAX_INTERVIEW_QUESTIONS`); Mismatch → fresh draft; `workflow_extractor` hook for the dispatch learning loop |
| `routes/rigor/` | `RigorRoute` — fixed-pass blue/red/judge protocol; real `DependencySession` checkpoint (`rigor.blue`) + `rewind_to_checkpoint` on a material rejection; red team reads through `FilteredLedger` at `Lod::LOD0` (dead ends excluded); final rejection resolves to a targeted interview (≤ 3 questions), frontier escalation only on low judge confidence; `/v1/rigor` is present-but-unconfigured when no backends are attached (explicit error, never a crash). Prompt constants / message builders / tolerant parses live in the `prompts` submodule |
| `charts/` | Chart (DAG workflow) library — `store` (`ChartStore`), `binding` (`Entity`, `ENTITIES_META_KEY`), `compile`, `execute` (under `Limiter` + SupervisedBatch), `render`, `rubric`, `select` (`ChartSelector`, `ChartFit`), `extract` (`WorkflowExtractor`), `stage` (`ChartPromptStage` — a `ClassifierStage`-shaped component that renders one target's template and makes one LLM call) — the workflow engine consumed by `PlanRoute` and the dispatch learning loop |

### Infrastructure

| File | Role |
|------|------|
| `server.rs` | `RouterServer` (`WorkUnit`) — hyper HTTP/1.1 accept loop on tokio; assembles `ServerDeps` and fans out to the `server/` submodule; `serve_http` is `pub(crate)` for integration tests; runs each `InstanceManager`'s boot reconcile + residency task from the attached `InstancePool` (in a drained `JoinSet`, never detached) |
| `server/handler.rs` | HTTP routing + request orchestration; `ServerDeps` (the collapsed former 12-`Option` dependency bundle): pipelines, routes, models, stats, cache, ledger, plan/rigor routes, sessions, ladders, context_cache, mock_dispatch, http_client, `instance_pool`, `api_key_env_name`, `supervisor`, `coordinator`. Merges query-string routing fields into the body, resolves the model-id grammar (`<model_id>[:<instance|group|latest>]`) in `resolve_pipeline`, and routes the management/model-less endpoints |
| `server/instances_api.rs` | The public `/instances` management facade: aggregate envelope across models, `POST /instances`, per-instance ops (delete/pin/unpin/resume/no-resume/resize/snapshot), `/memory` compat reshape, `/v1/models`, `/props`, model-less proxies (`/tokenize`, `/detokenize`, `/apply-template`, `/control`); management API-key enforcement; query parse/percent-decode helpers |
| `server/dispatch.rs` | `handle_dispatch` / `dispatch_real` — primary + `fallbacks` chain through `ChatBackend` (each wrapped in `RetryBackend`), short-circuit on non-retryable errors, response cache read/write, workflow extraction, allocate-on-503 via the `InstancePool` |
| `server/responses.rs` | OpenAI-completion response builders, SSE/CORS headers, `ServerStats` counters |
| `server/admin.rs` | Admin endpoints for the CLI — `POST /models/unload` (stop a managed model's `llama-server`; spec stays registered so dispatch reloads it) and `GET /metrics` (aggregates the managed llama-servers' Prometheus expositions, `?model=`-filterable, `# HELP`/`# TYPE` lines deduplicated) |
| `server/review.rs` | `ReviewWorker` — the async parse-review boundary. A credit-gated `WorkerPool<ReviewJob>` via the shared `CreditGatedPool` primitive (the ledger-tier-worker pattern) that runs `spacy_rs::review_prompt` (taxonomy-grounded) + `apply_corrections` against the `SqliteCorrectionIndex` and writes corrected annotations back with `HumanReview` provenance. The review call is served by the onnx LLM with the `ParseReview` grammar (constrained at generation). Served at `POST`/`GET /v1/sessions/{id}/review-parse`; an optional PII pre-filter (`fluent_onnx::pii::PiiSpanDetector` or the `RegexPiiDetector` baseline) flags review candidates when `review.auto_enqueue` is on |
| `server/entity_link.rs` | `EntityLinkWorker` — the async entity-link overlay. A credit-gated `WorkerPool<EntityLinkJob>` via the shared `CreditGatedPool` primitive that scores unresolved PROPN spans against the boot-baked ColBERT `EntitySimilarityIndex` (via the injected `EntityLinkScorer` seam) and writes **candidates only** to the `overlay_candidates` ledger table — never a doc-level `interlingua_entity_id` write at parse time |
| `streaming.rs` | `StreamingHandler` — SSE delta formatting for OpenAI-compatible streaming chunks; cross-chunk think-block filtering via `StreamingThinkFilter` |
| `kv_cache.rs` | Two-tier: `HotSnapshotIndex` (RAM LRU over `common_core::cache::LoadCache`, metadata only) + `ColdSnapshotIndex` (disk tree `model/adapter/session`); `SnapshotStore` composes both; the router never reads/writes raw KV bytes — it manages filesystem layout + sidecar metadata for llama.cpp slot save/restore |
| `instances/` | Instance-pool grammar generation + validation (`instance_grammar_string`, `validate_instances`, `is_valid_instance_name`) and the sidecar. Split into `mod.rs` (grammar/validation helpers + test stub), `client.rs` (`InstanceClient` — one server's `/instances` API over raw `reqwest`, `HttpClass`-classified; `InstanceError`/`InstanceInfo`/`InstanceTotals`/`SnapshotInfo`), `manager.rs` (`InstanceManager` — boot reconcile; per-instance `resume` map; `is_sleeping` residency probe; `list_with_fallback` synthesizing a resident footprint for plain — weights-only, no-instance-grammar — models; `weights_bytes`; `ensure_instance` on-demand creation; `ensure_group` allocate-on-503), and `pool.rs` (`InstancePool` — the router's aggregate facade: `<model_id>:<name>` ids, 64-bit-summed `total`, `/v1/models`, op proxies; footprint-weighted eviction `Evictable::{Context, Model}` + `evict_to_fit`; load-time admission control `make_room_for`; `resume` snapshot/expiry and control ops; the residency engine `run_residency`). The public `/instances` surface lives in `instances/api.rs` |
| `supervisor.rs` | `LlamaServerSupervisor` + `ManagedServer` — resolves `llama-server` from `$PATH` (or `LLAMA_SERVER`), spawns one process per managed model on a free localhost port (`--alias`, `-m`/`-hf`, `--instance` grammar, `--slot-save-path`, `--api-key`), waits for `/health`, and supervises each child (logs its output, restarts with capped backoff; post-boot **liveness** supervision probes `/health` and kills+restarts a hung server past `liveness_failures_before_restart`). Boots only models with a pinned instance (`start_all`); lazy models load on demand via `ensure_running` (spawn-locked, no double-spawn) and unload via `unload` (spec stays registered). `free_port`, `build_server_args`, `shutdown`. **Onnx-declared models (`ModelEntry.onnx`) are excluded** — they are served by the ort registry (`ort.rs`), never by a llama-server |
| `ort.rs` | The onnx composition root (crate `fluent-onnx`). `build_onnx_registry(config)` turns every `ModelEntry.onnx` block into a boot-built `OrtSessionRegistry` (validates task-vs-model-file sanity and `weights`/`hf_repo`/`instances` exclusivity; `resident: true` loads at boot, else lazily; fail-open to an empty registry or to plain-HTTP entries when the `onnx` feature is off). **`onnx_chat_backend(key)`** wraps a `CausalLm` session as the generative `fluent_llm::client::ChatBackend` (`OnnxRole::Llm`, `onnx/llm`) — the single wiring point for the LLM rung, review, enrichment, tiering, summarization, ranking, and retrieval; `grammar_from_json_schema` (in `fluent-onnx`) constrains structured decodes. **`OrtResidencyLoop`** (in `fluent-onnx::residency`) is the CPU-RAM residency parity sibling of the llama sidecar. The encoder-role builders — `disambiguation_overlays` (`PromptRouterOverlay` per `overlay_models` key), `nlp_encoder_fetch` (the legacy trained-encoder `EncoderFetchSync`), `pii_prefilter` (PII-Detector or `RegexPiiDetector` baseline), `colbert_entity_scorer` (entity-link `EntitySimilarityIndex`), `onnx_colbert_reranker`/`ColbertChartReranker` (chart MaxSim re-rank) — remain as feature-gated legacy seams behind `cfg(feature = "onnx")`; no encoder-role model is configured in the reference deployment |
| `concept_store_sqlite.rs` | `SqliteConceptStore` — the router's durable `spacy_rs::ConceptStore` over the shared ledger SQLite connection (`interlingua_concepts` table, keyed `(namespace, canonical_name)` with the content-addressed `id` + coral `node_id` cross-reference). The **second home** of the one-loader/two-homes YaGO design; boot reconciliation locks it equal to coral's content-addressed graph |
| `scheduler.rs` | Re-exports `AffinityScheduler` / `ScheduledTask` / `AgingConfig` from `fluent_concurrency::affinity` |
| `summarization.rs` | `ResultScorer` + `Summarizer` — `WorkUnit` impls that call an LLM (via `Arc<dyn ChatBackend>`) to score/condense responses; feeds the ledger's lazy LOD tiers |
| `score_matrix.rs` | `ScoreMatrix` — multi-dimensional weighted scoring (coherence/complexity/completeness/risk) with per-route dimension bands |
| `metrics.rs` | `FailureClass` + `classify_error` — typed-first error classification with a string-regex fallback for opaque shell/command output (D10) |
| `audit.rs` | The canonical durable-audit surface — a single `tracing` target `router.audit`; `AuditRecord` + `emit(kind, detail)`; audit kinds are distinguished by the `kind` field (`route`, `filter`, `tree_node`, `escalation`, `rigor`, `chart_target`, …) |
| `logging.rs` | Two-stream `tracing` subscriber: operational JSON/console rolling file + the durable audit stream (separate retention, always JSON, gated on `router.audit=info`) |
| `frontier/modes.rs` | `EscalationMode` ladder taxonomy — `Filter`, `Question`, `Team`, `Turnover` (D8; the old `FrontierMode` enum is gone), serde snake_case; `FrontierResult` and `AuditEntry`. Taxonomy and audit types only — the runtime lives in `dispatch/escalation/` (mode implementations in `dispatch/escalation/modes.rs`) |
| `hnsw.rs` | `HnswIndexHandle` — the single HNSW index handle type for the chart store's brute-force / `knn_brute_force` fallback |
| `telemetry.rs` | Structured telemetry events with a controlled vocabulary (`ToolName`/`ProviderCategory`/`FeatureName`), the `TelemetryEvent` enum, and `TelemetrySink` (`TracingSink`/`NoopSink`) — no free strings, no PII |
| `transforms/` | `TransformStrategy` trait + the `rewrite_text_messages` and `insert_string_map` shared helpers: `NoTransform`, `PiiAnonymize`, `DecomposeToAnonymizedHypothetical`, `DecomposeToSubtasks` (metadata intentionally bespoke — its `subtasks` payload is a structured object, not a string map), `CodewordAnonymizer`, `Sanitize`, `SecretMask` |
| `ranking.rs` | The M6 ranking pipeline — `SalienceSignal`/`SalienceScorer` (weighted, pure), `pagerank` (minimal power-iteration over the `interlingua_index` co-reference + `parent_id` graph), `SalienceSource`/`LedgerSalienceProvider` (read-only snapshot), `rank_candidates` (salience prefilter to top-K then **one** grammar-constrained call over the shortlist; degrades to salience order), and `SalienceRanker` (the composable step the retrieval service composes) |
| `retrieval.rs` | The M5/M6 live dispatch seam — `NodeRetrievalService` parses a candidate node's LOD0, runs `spacy-rs::retrieval`'s lemma-grep + fuzzy + `cross_check`, and pre-filters the pool through the `SalienceRanker`; exposed on `LedgerAgentCoordinator` via `with_retrieval`/`retrieve_nodes` |
| `cli/` | Admin CLI support backing the `coral-router` binary's `list`/`ps`/`pull`/`scan`/`rm`/`show`/`stop`/`speedtest` subcommands (ported from `gguf_tool.py`): `gguf.rs` (GGUF-directory scanning + `models.json` cache), `preset.rs` (`models-preset.ini` rendering), `commands/` (`filesystem.rs` for the GGUF commands, `server.rs` for the HTTP-driven `ps`/`stop`/`speedtest`), and `CliContext`/`CliError` in `mod.rs` |
| `testing/` | `TranscriptProvider`, `MockTranscriptEntry`, `MockDispatchContext` — transcript-driven integration-test harness for E2E and golden tests |
| `test_stubs.rs` | `StubChatBackend`, `HashEmbedder` — test-only backends (cfg(test)) |

### Adapter architecture (`dispatch/`)

The **production dispatch path** runs through `dispatch/backend.rs`, which
defines the object-safe `DispatchBackend` trait that every server dispatch site
depends on (the single dispatch trait, D4). The LLM client trait remains
`fluent_llm::client::ChatBackend` — router dispatch trait is `DispatchBackend`:

```rust
pub trait DispatchBackend: Send + Sync {
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
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        // No abort signal: delegates to stream_complete_with_abort with None.
        self.stream_complete_with_abort(request, model, params, idle_timeout_ms,
            total_timeout_ms, filter_thinking, None)
    }
    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<fluent_concurrency::stream::StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>>;
}
```

Concrete backends: `OpenAiChatBackend` (single-attempt HTTP through a raw
`reqwest::Client`; non-2xx status classified via `HttpClass` into
`DispatchError::RateLimited` vs `DispatchError::Http`), `RetryBackend`
(jittered-exponential retry via `common_core::retry::retry_async` — the single
backoff helper), and `BackendChain` (ordered backend chain that
short-circuits on terminal 4xx). `server/dispatch.rs::dispatch_real` iterates
the primary `RoutingTarget` plus its `fallbacks` list, wrapping each target in
a retry backend. The `fallbacks` are *target* candidates — populated at
route-resolution time by `RoutingConfig::all_dispatch_targets` from the
route's group plus cross-group models, ordered by intelligence proximity to
the request complexity (primary group first, cost as tie-break) — not backups
for the classifier. Streaming flows through `StreamingHandler` over an
`http_body_util` channel, wrapped in a `StreamBody` whose `Drop` fires a
`StreamAbort` (see `dispatch/backend.rs`). `RetryBackend` and `BackendChain`
**forward** the abort token unchanged — the wrapper-delegation rule — so a
downstream abort reaches the transport through any retry/fallback layer.
Request bodies are built by the canonical
`fluent_llm::openai::build_openai_chat_body` (which carries the
`chat_template_kwargs: {"enable_thinking": false}` default).

**Abort propagation (streaming).** The body the router hands to the client is
a `StreamBody` (a `Channel` receiver + `StreamAbort`); when the client stops
consuming, hyper drops that body and the `StreamAbort` fires. The forwarding
task `select!`s its upstream read *and* its `send_data` against the signal, so
a downstream disconnect (a) drops the upstream `reqwest::Response` — closing
the connection, which the llama-server fork reads as a slot interrupt — and
(b) finalizes the `StreamAnswer` with whatever content streamed so far, so the
ledger records the partial answer rather than a stub label. Because Coral
Router is the process owner of the fleet, `dispatch_to_single_target` also
arms a watcher on the same signal that issues an explicit `POST /abort`
(`InstanceClient::abort`, `id_slot` from the target) to the owning server —
belt-and-suspenders on top of the transport close, best-effort when the slot
is no longer running.

`dispatch/escalation/` owns the ladder runtime. After every local model in a
`model_group` chain fails, `try_escalate` consults the deterministic
`ContextCache` first (short-circuit before any frontier call), then runs the
configured modes in order (`modes.rs`). Each mode's frontier transport reuses
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
the dispatch collapse (D4).

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
pipeline pre-filter and the write-path scrubber (`ledger_guard.rs`).

## Key Compositions & Reusable Primitives

| Primitive | Source | Used by router at |
|-----------|--------|-------------------|
| `Component` / `WorkUnit` | `fluent-wvr` | Every pipeline stage, `PipelineOrchestrator`, `RouterServer`, `ResultScorer`/`Summarizer` |
| `DependencyGraph<K>` | `fluent-dag::dep_graph` | `DependencySession` for step DAG tracking |
| `ResultPool` | `fluent-concurrency::pool` | `dispatch/escalation/modes.rs` — parallel classifier slots (team mode) and parallel hypotheticals (question mode) |
| `PriorityResultPool` | `fluent-concurrency::pool` | `AffinityScheduler` — priority dispatch with aging |
| `CreditGatedPool<J>` | `fluent-concurrency::credit_pool` | `ReviewWorker`, `EntityLinkWorker` — the two credit-gated bounded workers (a `WorkerPool` whose `submit` path consumes a credit token and whose handler releases it per processed job) |
| `CreditedFeedWorker<Item>` | `fluent-concurrency::feed_worker` | `LedgerTierWorker` (background LOD4/LOD5 derivation — the feed it was extracted from), and the `arc_ready` overlay worker (`ledger/overlay_worker.rs`) — a credit-gated bounded `mpsc` feed whose drain loop runs a shared handler under a `Limiter` and releases credit only after each item's handler completes |
| `Limiter` | `fluent-concurrency::pool` | `ClassifierStage` — concurrent classifier call cap; `charts/compile.rs` + `charts/execute.rs` — chart-DAG execution cap; `PlanRoute` |
| `WorkContext` | `fluent-wvr` | Carries request, caps, runtime through every stage |
| `Runtime` trait | `fluent-wvr` | Plugged via `fluent_concurrency::tokio_runtime()` everywhere |
| `LoadCache<K,V,E>` | `common-core::cache` | `HotSnapshotIndex` — bounded get-or-load LRU |
| `ArcIntern<str>` | `internment` | `ContentNodeStore` session/role index keys; work-unit and graph asset names |
| `LatencyHistogram` | `common-core::metrics` | `Instrumented::with_metrics` wiring |
| `retry_async` | `common-core::retry` | `RetryBackend`, `RetryClassifier`, `SupervisedBatch` retries |
| `make_hnsw()` / `knn_brute_force` | `common-core::sqlite` / `fluent-db::vector` | `ContentNodeStore` KNN; `hnsw.rs` chart-store fallback |
| `HttpClass` | `guidance-llm` | `dispatch/backend.rs` — status classification in `OpenAiChatBackend` (streaming + buffered) |
| `DispatchError::is_retryable()` | `fluent-router` (`dispatch/frontier.rs`) | retry/fallback decisions in `dispatch/backend.rs` and `server/dispatch.rs` |
| `LlmError::is_retryable()` | `fluent-concurrency::llm_queue` | `guidance-llm` client error classification |
| `parse_json_response` | `fluent_llm::parse` | `routes/rigor/` (red/judge parses), `dispatch/escalation/assemble.rs` |
| `Decomposer` | `fluent_llm` | `dispatch/escalation/modes.rs` — question-mode hypothetical decomposition |
| `first_accept_in_order` / `first_accept_in_order_sync` | `fluent-concurrency::ladder` | The spacy-rs annotation ladder (async + the sync `run_ladder_sync`), the escalation `Ladder::try_escalate`, the `BackendChain` complete/stream-complete fallback, and the `dispatch_real` fallback walk. **`TargetMatcher::match_target` is the documented exception** — a different ladder shape (start-index skip + last-member-always-wins + per-rung audit emission) |
| `NlpPipeline` / `AnnotationRung` / `ConceptStore` / `CorrectionIndex` / `PreferredSenseIndex` / `retrieval` | `spacy-rs` | `NlpStage` runs the pipeline; `SqliteConceptStore` / `SqliteCorrectionIndex` / `SqlitePreferredSenseIndex` implement the traits; `match_interlingua` tree filters dispatch on the ids; `FrameResolutionWorker` resolves + promotes frames; `retrieval` backs `NodeRetrievalService` |
| `OrtSessionRegistry` / `OrtLlmSession` / `OrtResidencyLoop` / `grammar` | `fluent-onnx` | `ort.rs` builds the registry; `onnx_chat_backend` wraps a `CausalLm` session as the generative `ChatBackend` (LLM rung / review / enrichment / tiering / ranking / retrieval); `OrtResidencyLoop` is the CPU-RAM residency parity sibling of the llama sidecar; `JsonObjectGrammar`/`JsonArrayGrammar` constrain structured decodes. Legacy encoder seams: `nlp_encoder_fetch`, `pii_prefilter`, `colbert_entity_scorer`, `onnx_colbert_reranker`, `disambiguation_overlays` |

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
   (permanent). Retries are applied by `RetryBackend`, and the
   primary-plus-`fallbacks` chain is walked by `server/dispatch.rs::dispatch_real`.

The router's own error taxonomy mirrors this at a higher level:
`DispatchError::is_retryable()` returns `true` for `Http(_)` and `RateLimited`.
Separately, `LlmError::is_retryable()` (in `fluent-concurrency::llm_queue`)
classifies `guidance-llm`/queue errors: `Http(_)` and `RateLimited` are
retryable; `Api(_)` and `NoResponse` are permanent. Both are error-level
classifications independent of how the error was produced.

## Import Boundaries (enforced)

Following AGENTS.md: `fluent-router` imports from `common-core`, `fluent-wvr`,
`fluent-concurrency`, `guidance-llm`, `fluent-types`, `fluent-dag`, `spacy-rs`,
`fluent-onnx`, and standard library / `tokio` / `reqwest`. It does NOT import
from `guidance`, `coral`, `wasm_ipc`, `knowledge`, `ontology`, or `rdf`.
`knowledge.rs` gives coral's Context a reachable read path without the router
importing coral. Dependency direction is preserved: `fluent-onnx` and `spacy-rs`
never import the router (or each other's domain types); the router implements
their `ConceptStore`/`CorrectionIndex`/`EmbeddingProvider`/`SessionLoader`/
`LlmFetch` seams. The **binary** crate (`src/bin/coral-router/`, package
`coral-router`) is the composition root and additionally depends on
`coral-context` and `guidance-ontology` — its `boot.rs` runs the one-loader/
two-homes YaGO load into both the coral graph and the router's
`SqliteConceptStore`.

## Pipeline data flow detail

1. **Server**: hyper reads the HTTP request; `server/handler.rs` collects the
   body (enforcing `max_payload`), merges the query-string routing fields
   (`model`/`instance`/`snapshot`/`id_slot`) into the body when the body does
   not define them (body wins), and deserializes JSON →
   `normalize::normalize_request` → `RouterRequest` (the normalizer re-attaches
   those routing fields after the shared `fluent_llm` normalizer strips
   non-OpenAI keys).
2. **Ledger** (pre-pipeline): `ContentNodeLedger::record_request()` writes the
   full request at LOD0 before any filter runs (through the write-path
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
   scores, reason). Checks coherence and safety thresholds. **Route-level
   `always_route`**: when the requested route is configured `always_route:
   true` (`RouteRef.always_route`, e.g. prose, code, translation, science,
   legal, medical), a classifier `action=respond` is overridden to
   `action=route` toward that route — the classifier never answers those
   domains directly, compensating for a small model's overconfidence; the same
   routes are advertised to the LLM as "Dispatch rules" in the generated
   system prompt. Resolves the route via `RoutingConfig::resolve_route()` with
   complexity-gated model selection and optional score-matrix ranking — or,
   when the pipeline opts in (`target_match: "self_assess"`), via the shared
   `TargetMatcher`, which runs the in-group target-matching ladder (each
   candidate self-assesses the prompt; the first whose `intelligence` meets its
   assessed complexity — or the last member — becomes the primary target).
   Emits a `StageDecision` carrying `metadata.response` (direct answer),
   `metadata.routing_target` (dispatch instructions), or a rejection verdict.
6. **Server** (post-pipeline): `server/handler.rs` reads `PipelineResult` — if
   `classifier_response` exists, responds directly; if `routing_target` exists,
   calls    `server/dispatch.rs::handle_dispatch`, which walks the primary target
   plus its `fallbacks` list through `ChatBackend`s (each wrapped in
   `RetryBackend`), short-circuiting on non-retryable errors. The client's
   explicit `instance`/`snapshot`/`id_slot` fields are overlaid onto the target
   so they reach the outgoing body; a 503 group-miss asks the `InstancePool`
   to allocate fresh KV before one retry. If no target, the handler dispatches
   to the classifier's model as a fallback *target* (the model the
   classifier ran on now answers the request), or a canned fallback response.
   Fallback models are target models — never a backup for the classifier. When
   dispatch and escalation fail, the per-group `Ladder`
   (`try_escalate`) runs its configured modes, short-circuiting on a
   `ContextCache` hit. Every local dispatch lands as a direct HTTP call on the
   owning spawned `llama-server` carrying the translated model id.
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
            "classifier_model": "swarm",
            "coherence_threshold": 0.70,
            "blacklist": "env/pii-patterns.json",
            "score_matrix": { … },
            "target_match": "self_assess",
            "target_match_timeout_ms": 300000
        }
    },
    "routes": {
        "prose": { "group": "prose", "description": "Write a story, novel, poem…", "always_route": true },
        "science": { "group": "science", "description": "Physics, chemistry, biology…", "always_route": true },
        "local": { "group": "default", "description": "General Q&A…", "always_route": false }
    },
    "gguf_dir": "/app/ai/models/gguf"
}
```

The two decision stages and their knobs are above. The opt-in NLP/ONNX
stages are enabled the same way (`build_named_pipeline_with_backend_onnx_and_nlp`):
`nlp: true` inserts the `NlpStage` (requires a ledger for the concept store;
without one it fails open) which runs the deterministic-first ladder — base
`[ArcEager, Rule]` unconditional, then gated refine `[Encoder, Llm]` on
confidence OR task-value via `RefinePolicy` (fallback keeps base). `overlay:
true` + `nlp_ordering: "deterministic_first"` inserts the `OverlayStage` for
residual scoring, and `classifier_retry_max` wraps the classifier in a
`RetryClassifier`. `nlp_ordering` (`"llm_first"` default ⇒ `RefinePolicy::Always`,
`"deterministic_first"` ⇒ `RefinePolicy::OnUncertain` with default thresholds,
unless `pipelines.<name>.refine_policy` overrides) and the explicit
`refine_policy` knob (`mode`/`min_overall`/`min_role_coverage`/`refine_on_ties`/
`min_token_score`/`refine_on_unresolved_critical_role`/
`refine_on_unresolved_propn`/`refine_on_collision_note` — each independently
tunable) control when the refiner is consulted. The generative ONNX LLM
(`models.<key>.onnx.task = "causal_lm"`, the `onnx/llm` role) is wired once and
reached everywhere through `onnx_llm_backend()` — the LLM rung, review
(`ParseReview` grammar), enrichment, tiering, summarization, ranking, and
retrieval — the single `ChatBackend` wiring point. `encoder_model` (a legacy
trained-encoder `fill_mask` + `annotation_heads` rung) and `overlay_models`
(two-tower `zero_shot_routing` Prompt-Router) remain feature-gated but are not
configured in the reference deployment. The top-level sections that back these —
`models.<key>.onnx`, `review` (async parse-review worker), and `overlay`
(arc_ready annotation overlays) — are documented in §"Natural-language
processing spine".

`target_match` (`"self_assess"` default | `"static"`) selects the in-group
target-matching policy (§"Model-group target selection"); `target_match_timeout_ms`
(default `DEFAULT_TOTAL_TIMEOUT_MS`) bounds each self-assessment call.
`routes.<name>.always_route` forbids direct classifier answers on that route
(§"Pipeline data flow detail", step 5) — prose, code, translation, science,
legal, and medical all route unconditionally to their group's model, while
`local` keeps the classifier's direct-answer path for simple prompts. `gguf_dir`
feeds the admin CLI's weights resolution (`list`/`scan`/`show`/`pull`/`ps`),
overridden by an explicit `--gguf-dir`.

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
chain is exhausted does the per-group `Ladder` engage
(`dispatch/escalation/`).

Every model in the chain is a candidate to answer the request — a fallback
*target*. None of them backs up the classifier: the classifier stage runs on
its own `classifier_model`, and when the pipeline produces no target the
handler dispatches to that classifier model as a fallback target
(`server/handler.rs`) rather than to a classifier backup. The matched
target's answer is recorded in the session ledger and session step after
dispatch (§"Pipeline data flow detail", step 7).

**Always-route domains.** Routes configured `always_route: true` (the reference
deployment: `prose`, `code`, `translation`, `science`, `legal`, `medical` —
every single-member group backed by a stronger model) never let the classifier
answer directly; step 5 of the pipeline data flow overrides `action=respond`
into `action=route` toward the requested route, deterministically. This is what
keeps a small overconfident classifier from "writing" prose or "answering"
legal/medical questions itself: a request on those routes always reaches the
route's group model. Routes without the flag (`local`, `extract`, `summarize`)
keep the direct-answer path, so simple prompts, prompt formulation, and direct
classification still happen on the cheap model. All of it is config — the
classifier prompt's "Dispatch rules" section is generated from the same
`always_route` flags.

## Instance pools, the serving layer, and the sidecar

**Coral Router owns the serving processes.** A model entry is *managed* when it
declares a weights source or an instance pool: `ModelEntry.weights`
(a local GGUF path), `hf_repo`/`hf_file` (on-demand Hugging Face load), or
`instances`. An `onnx` block is a separate serving path — it is *not* served
by a llama-server at all (see §"Natural-language processing spine"). At boot `main.rs` builds a `LlamaServerSupervisor`
(`supervisor.rs`), finds `llama-server` on `$PATH` (or `LLAMA_SERVER`), and
**on-demand residency**: only models declaring at least one pinned instance are
spawned at boot (each on a free localhost port, awaited for `/health`); the rest
are loaded lazily by the dispatch path (`supervisor.ensure_running` — spawn-locked
so concurrent dispatches never double-spawn) and unloaded again when the sidecar
evicts their last context (`supervisor.unload` — the spec stays registered).
Each spawned model's `endpoint` is rewritten to
`http://127.0.0.1:<port>/v1/chat/completions`, so every classifier, dispatch
target, and backend points at the owned server. The supervisor supervises each
child for the life of the process — logging its output and restarting it with
capped backoff on an unexpected exit, and probing `/health` post-boot to
kill+restart a server that stops answering. In `--mock` mode supervision is
skipped: canned dispatch needs no real model.

The llama.cpp router mode is never used. Coral Router talks to each spawned
server directly, and each server loads its model's weights exactly once and
allocates many named contexts ("instances") from them — separate KV + compute
buffers sharing those weights. A `count: N` profile expands to N sibling
instances named `<key>-0` .. `<key>-{N-1}` in the shared `group` (each its own
full-size window; `parallel` slots share one window and never multiply it).
Requests route to an instance by the model-id grammar. The minimal-branch
server no longer reads `num_ctx`/`parallel`/`sleep_idle_seconds` from the
request body; those are declaration-only, and coral-router strips them from
dispatched bodies. `sleep_idle_seconds` survives only as a sidecar
eviction-priority hint.

The dispatch grammar generator (`instances::instance_grammar_string`) emits the
exact `--instance` flags the supervisor hands to `llama-server` — the minimal
grammar `name[:group=G][:ctx=N][:parallel=M][:pinned][:default]` (no sleep
component). For the reference `swarm` pool (`count: 2` at 8192 ctx, a pinned
65536-ctx `ledger` default, a 131072-ctx `scratch`):

```
--instance "swarm-0:group=swarm:ctx=8192:pinned" \
--instance "swarm-1:group=swarm:ctx=8192:pinned" \
--instance "ledger:ctx=65536:pinned:default" \
--instance "scratch:ctx=131072"
```

**All-lazy pools register `/instances` too.** Only pinned instances are
declared at spawn (unpinned ones are created on demand by the sidecar, saving
their VRAM until used). Exception: a pool with **no pinned instance** — an
all-lazy model — declares its *full* grammar at spawn. The fork registers its
`/instances` management endpoint only when `--instance` grammar is present, so
an all-unpinned pool that declared nothing would be a plain server whose
`/instances` 404s, making it invisible to `/v1/models`, `/instances`, and
`coral-router ps` even while loaded. Declaring the full grammar makes such a
model trackable the moment the dispatch path loads it on demand; there is no
resident anchor whose VRAM a declaration would waste. It stays lazy (`boot` is
false — no pinned instance), so it is still absent until first use.

**Model id translation.** Config keys are the public model ids; llama.cpp model
names are internal. `ModelEntry::llama_model_name` resolves the server's
`--alias`, and dispatch always sends the translated id (`<llama-name>[:<instance>]`).
Two distinct qualifier intents stay separate:

- `ModelEntry::default_dispatch_qualifier()` - the pool's **default instance**
  (`:ledger`), used by `RoutingTarget::from_model_entry` for client-facing
  bare-`<base>` dispatch.
- `ModelEntry::pool_qualifier()` - the router's **internal work group** (the
  pool, `:swarm`): the largest non-default `count` profile's group, else the
  default profile's group, else the single shared group, else `None`.
  `local_backend` (the single DIP `LlmClient` factory behind the classifier,
  chart selector/adjudicator/reranker, target-matching ladder, and rigor roles)
  routes internal work through this pool so those calls spread across the
  shared-weight instances instead of pinning to the default.

`RoutingTarget::from_model_entry_instance` targets a named point
(`<base>:ledger`, `<base>:scratch`); `local_backend_for_instance` merges the
named instance profile's `params` over the entry `params` (profile wins) and
strips declaration-only keys, so instance-level sampling knobs actually reach
the body. On the client-facing surface the handler also resolves the model-id
grammar directly (`resolve_pipeline`): a request for `model: "<id>:<instance>"`
(or `:<group>`, `:latest`) bypasses the route table and targets the owning
server. The routing fields `instance`/`snapshot`/`id_slot` are read from the
JSON body and the query string (body wins), merged in `server/handler.rs`,
preserved through `normalize`, and overlaid onto the dispatch target in
`server/dispatch.rs`.

**Public `/instances` API.** `InstancePool` (`instances/pool.rs`) is the router's
aggregate facade over every managed server. It is served at Coral Router's own
address (`server/instances_api.rs`) as the single sidecar entry point —
`GET/POST /instances`, `DELETE /instances/:name`, pin/unpin, and the snapshot
endpoints, plus the resume control ops `POST /instances/<model>:<name>/resume`
and `/no-resume` (set/clear the preserve-on-evict flag; `no-resume` also
deletes the `<name>-resume` snapshot) — mirroring the llama-server contract
under `<model_id>:<name>` ids with 64-bit-summed `total` memory. The aggregate
envelope carries the router-side `resume` flag per instance and the
synthesized plain-model footprints. `GET /v1/models` / `/models` lists one
entry per instance plus aliases; `/props` proxies the default server and adds
`total_slots` + an `instances` array; `/memory` is a compat reshape of the same
envelope; the model-less endpoints (`/tokenize`, `/detokenize`,
`/apply-template`, `/control`) proxy to the pool's default server. All
management endpoints require the API key when `sidecar.api_key_env` names a
variable. The managed servers bind to `127.0.0.1` only and are never exposed
directly.

**Sidecar residency.** Each manager (`InstanceManager`) talks directly to its
own server's `/instances`. At boot the pool validates every grammar
(`validate_instances`, fail-fast on duplicate names / group-name collisions)
and the server runs each manager's reconcile (create missing instances,
resize `n_ctx` drift, tolerate a 409 duplicate) and residency loop. The
residency loop **always** polls the aggregate `GET /instances` envelope (the
`total` is the VRAM signal) and logs free/used; eviction is gated on the
allocation budget `device_total - minimum_remaining_vram`.

**The onnx context pool.** The generative onnx role is the second row of the
same "one weights load, N named context windows" table: `fluent-onnx`'s
`OnnxContextPool` wraps one `OrtLlmSession` (a single `Arc<Mutex<Session>>` +
tokenizer) with a `ConcurrentRegistry` of named `OnnxContext`s, each owning its
own KV. Contexts are created on demand (`ensure_context`, mirroring the llama
fork's `ensure_instance`), interleave on the shared session (their KV lives
per-context), and honor the same lifecycle knobs the llama fork has — `n_ctx`,
`max_ctx`, `pinned`, `resume`, `last_used`. Both fleets present through the
shared `fluent-llm::runtime` traits, so the aggregate `/instances` facade,
`/v1/models`, `coral-router ps`, `POST /models/unload`, and `POST
/instances/:name/resize` drive llama and onnx uniformly: llama rows are
byte-identical, and onnx rows appear only when the onnx fleet is configured.

**Plain models are resident too.** A managed model that declares only
`weights` (no `instances`) is served by a plain `llama-server` whose
`/instances` returns 404 — it has no instance grammar. The manager's
`list_with_fallback` detects that 404 and synthesizes a single envelope entry
(`<model>:default`) from `/props` (`n_ctx`, `is_sleeping`) plus the configured
weights file size, so `/instances`, `/v1/models`, the residency budget, and
`coral-router ps` all account for a plain model's resident weights — 0 bytes
while the fork reports `is_sleeping` (its weights slept out of VRAM by
`--sleep-idle-seconds`), the file size once loaded. `coral-router ps` reports
resident `model_bytes`, never the on-disk file size of a sleeping model.

**Footprint-weighted eviction.** Over budget, `evict_to_fit` evicts units
built by `gather_residency` — each is either a single unpinned context
(`Evictable::Context`, frees its KV + compute) or a whole model with no pinned
instances (`Evictable::Model`, frees its weights *and* every context). A model
with any pinned context keeps its weights resident; `pinned` instances are
never candidates. Units are ordered by a `freed_bytes × idle-time` score — the
largest coldest resident footprint goes first, so a 10.5 GB weight pool is a
real eviction target while a just-used model scores near zero and stays. This
is the OOM-avoidance priority: context-only trimming cannot keep the device
under budget when a big model's weights dominate. After the pass, models left
with zero contexts are unloaded (`unload_empty_models`).

**Load-time admission control.** `ensure_target_ready` (called on every
dispatch) never spawns or wakes a cold model without first making room:
`make_room_for(model_key, weights_bytes)` runs the same eviction until
`used + required ≤ budget`. Residency is judged by the *actual* resident
state, not the process flag — a plain model whose fork has slept its weights
out of VRAM (`/props is_sleeping = true`) is treated as not resident, so
waking it reloads its weights only after room is freed. This prevents the
OOM abort a naive wake causes when a second big model has since loaded.

**Resume (preserve-on-evict).** An instance can be marked `resume` (config
profile, `POST /instances` body, or `POST /instances/<model>:<name>/resume`).
When a `resume` context is evicted it is first KV-snapshotted under the
deterministic name `<name>-resume` (best-effort; a failed save logs and the
eviction still proceeds); the session transcript is already durable in the
ledger, so KV + text log together preserve the workload. A later dispatch to
the same instance with `snapshot=<name>-resume` and the same `session_id`
restores it. Coral Router concludes the work is done — clearing the flag and
deleting the snapshot — explicitly via `POST /instances/<model>:<name>/no-resume`
or automatically once the context is idle past `sidecar.resume_ttl_s` (checked
each residency pass). `resume` is moot on `pinned` instances, which are never
evicted. `sleep_idle_seconds` survives as an eviction-priority hint: a model
idle past it is a better candidate, and once actually sleeping its weights
contribute 0 to the budget. On a 503 `"no free instance in group"` group-miss,
dispatch calls `InstanceManager::ensure_group` to allocate a fresh
`<group>-<uuid>` instance before retrying once. `config.sidecar.slot_save_path`
is created at boot and feeds the server's `--slot-save-path`; it also drives
the `KvSnapshot` `file_path` derivation so the router's snapshot metadata and
the server's layout agree. The management client reuses the raw-reqwest
pattern of `OpenAiChatBackend` with `HttpClass`-classified errors.

**Admin CLI (`coral-router ps` / `list` / `show` / `pull` / `rm`).** The CLI
reads the aggregate `/instances` + `/v1/models` envelopes and the config to
report the live residency picture: per model, the resident weights bytes (the
envelope's `model_bytes` — `0` while the fork has them slept out of VRAM,
never the on-disk file size), per-instance context memory, `resume` flags, and
the total `weights + contexts` resident. The router overlays the **weights
identity** (`short_id`/`arch`/`quant`, derived from the GGUF file it loaded)
on each `/instances` row, so `ps` shows `id=…, arch=…, quant=…` without needing
the weights file on the CLI's host; `render_weight_block` falls back to local
GGUF resolution only when the router didn't report it. The GGUF root for
file-size fallbacks resolves from `--gguf-dir`, else the config's `gguf_dir`,
else the built-in default, so paths are configurable and never recompiled in.

**KV snapshot round-trip.** `SnapshotStore` may hold an optional
`InstanceClient` handle (`with_fork_io`). `save_snapshot` then POSTs the
snapshot to the owning server (`POST /instances/:name/snapshot`) via the shared
`common_core::runtime::block_on` bridge and records the metadata locally;
`list_snapshots`/`delete_snapshot` delegate to that server. Rigor's blue-pass
completion triggers the save, so the subsequent blue->rewind->red flow sends a
real `snapshot`/`instance`/`id_slot` on the next dispatch. Without the handle
these degrade to metadata-only no-ops.

**Boot composition (`ledger`/`session` sections).** When `config.ledger` is
present, `main.rs` opens a `ContentNodeLedger` (path, or in-memory with a
`warn!`) and attaches the DIP `Summarizer` backend via
`RouterConfig::summarizer_for_ledger()` (targeting `<base>:ledger`). When
`config.session` is present it builds a `SessionRegistry` mapped to the
`session.root` KV root. Both attach to the server (`with_ledger` /
`with_sessions`), so rigor rewind and ledger LOD derivation exist at runtime.
Both sections are default-absent: existing deployments are byte-identical until
they opt in.

## Ledger: condensed context architecture

`ContentNodeLedger` is a thin facade over the shared `ContentNodeStore`. Every request
is stored at full detail (LOD0) before the pipeline runs, and results are
recorded afterward. This separates durable storage from live working context:

```
User message → ContentNodeLedger → ContentNodeStore (durable, full detail)
                ↓                         ↓
         Pipeline stages         ParallelLedger / FilteredLedger
         (read from WorkContext,   (render-only views; single text-exit
          not from ledger)          LedgerView::render → lod_text)
                ↓
         Orchestrator/Session (reads condensed summary, not raw history)
```

Key load-bearing properties:

- **Write path is checked.** Every write reaches `ContentNodeStore` only
  after passing through `ledger_guard::scrub_for_ledger` — the builtin filter
  engine with the `ContentNodeWrite` scope active. PII-matching text is
  irreversibly replaced (`[REDACTED:<pattern>]`), no codeword map retained.
  Direct `ContentNodeStore` writes are the documented bypass (production writes route
  through the facade).
- **LOD lifecycle.** LOD0 (full text) + LOD5 (label) are eager; LOD1–LOD4 are
  derived lazily, always from LOD0 only (never chained), via the `Summarizer`
  WorkUnit, and cached on the node at most once. `CompactionStrategy`/
  `RecencyCompaction` demote older nodes to a higher LOD (setting `active_lod`).
- **Views never own text.** `LedgerView::render` is the single
  text-exit from the store; `ParallelLedger` gives independent default-LOD
  views over one shared `Arc<ContentNodeStore>`; `FilteredLedger<V>` is a reference
  overlay (exclusion set + optional render transform) used by both the PII
  frontier view and the rigor red-team view. Rendering degrades to LOD0 when a
  lazy tier is un-derivable rather than erroring.
- **Shared store.** Nodes live once behind `Arc<RwLock<ContentNode>>`
  with interned `ArcIntern<str>` session/role index keys and durable
  `content_json` hydration (seeded `next_id` from `MAX(node_id)` so restarts
  never re-issue colliding ids).
- **Tiered annotations keyed to content.** Every annotation carries a
  `ProvenanceTier` (`Deterministic < LocalModel < Frontier < HumanReview`).
  `write_annotation` routes an `AnnotationClaim` to the `AnnotationStore` under
  the node's current `content_hash`; a higher-tier claim supersedes (never
  deletes) a lower-tier one for the same node version, and mutating LOD0
  changes the hash and thereby invalidates the node's cached claims by keying —
  no staleness scheduler.

### Background tier derivation (`ledger/tiering.rs`) — opt-in

`LedgerTierWorker` derives LOD1–LOD4 (and upgrades LOD5 labels) in the
background instead of on the read path. Boot backfills underivable nodes,
then drains newly-recorded nodes from the `ContentNodeStore` tier-event feed.
Concurrency is bounded by a shared `Limiter`; writes are at-most-once
(re-checked under the node write lock); summarizer failures degrade the node
to a higher LOD rather than erroring; a successful higher-tier derivation
downgrades it back (never repeats work it already did). Enabled only when
`ledger.background_tiering = true` — otherwise the lazy on-read derivation
described above is unchanged.

The feed mechanics are composed from the shared
`CreditedFeedWorker<NodeId>` primitive (`fluent-concurrency::feed_worker`)
— the bounded feed, the `CreditFlow` gate, the `Limiter`, the drain loop,
and the boot backfill — rather than hand-rolled; the worker supplies only
the per-node derivation handler and its retry policy.

The tier feed is bounded: it is a `tokio::sync::mpsc` channel
(`queue_capacity`, default 1024) plus a `CreditFlow` gate. The async producer
path (`run_agent` step 5 → `enqueue_with_credit`) acquires a credit token
before forwarding a `NodeId` — blocking while exhausted — and the worker
releases a token after each processed node (`credit_receiver.recv()`), so a
burst of agent turns cannot grow the feed without bound. Knobs:
`ledger.tier_credit_limit` (default 256) and `ledger.tier_credit_more_after`
(default 8). The store's synchronous write paths enqueue via the bounded
channel's non-blocking `try_send` (skipping on a full feed; agent nodes are
still covered by the credit-gated enqueue, and boot backfill catches
stragglers).

### Background arc_ready annotation overlays (`ledger/overlay_worker.rs`) — opt-in

`OverlayWorker` derives the three **arc_ready** overlays on the background —
the spacy `ArcReadyAnnotation` (`annotation`), the LLM enrichment summary
(`metadata["llm_overlay"]`), and the dense `embedding` — each lazily and
**at-most-once** from LOD0 only (an overlay never reads another overlay's
output). It is the **second consumer** of the shared
`CreditedFeedWorker<NodeId>` primitive (`fluent-concurrency::feed_worker`)
and deliberately **composes that primitive rather than reimplementing
`LedgerTierWorker`'s shape** — the feed, the `CreditFlow` gate, the `Limiter`,
the drain loop, and the boot backfill all come from the primitive; the worker
supplies only the per-node handler.

Per node the handler **fans out the three derivations in parallel** on the
injected `Runtime` (no ambient `tokio::spawn`) and joins them, so a slow or
down LLM backend never delays the spacy or embedding overlays for the same
node. Each derivation goes through the store's at-most-once, fail-open
`annotation_for` / `embedding_for` / `llm_overlay_for`, so a missing or failing
seam leaves that one overlay off (`status: failed`, never retried) without
affecting the others. The worker emits a `kind = "overlay"` audit record per
derived overlay (success or permanent failure).

Opt-in via `overlay.arc_ready.enabled` (config-absent is byte-identical to a
pre-roadmap boot). Requires a ledger; each seam is attached only when its
model/pipeline resolves. Drained on graceful shutdown so in-flight derivations
complete before the router exits.

### Prompt assembly (`ledger/prompt.rs`) — pure function object

`LedgerPromptAssembler` builds a fidelity-budget-fit prompt from a session's
cached LOD tiers: the first and last LOD0 anchors are guaranteed, intermediate
nodes are relevance-ranked (lexical score, recency) and demoted to the highest
LOD that fits the token budget; only *cached* tiers are rendered — it never
triggers a summarization cost. Pure (no I/O), unit-tested in isolation, and
shared by the coordinator (`orchestrator.rs`) and the plan route.

### Agent coordinator (`ledger/orchestrator.rs`) — opt-in

`LedgerAgentCoordinator` runs the agent loop over a session's shared ledger:
restore a KV snapshot (`KvSnapshotPolicy`: never / always / restore-if-same-model)
or assemble the prompt → execute the LLM call (on a blocking task, session
guard dropped — never held across the call) → record the exchange → snapshot
the KV state (`advance_and_snapshot` → `SnapshotStore`) → enqueue a tier
derivation (credit-gated, see above) → complete the step. `last_resident_model`
is session-scoped (walks the session's own checkpoint-node index, not all
sessions' nodes). Enabled only when `ledger.orchestrator.enabled = true`;
config supplies the backend, adapter, prompt-budget, and snapshot policy.
`node_plan` metadata on recorded nodes feeds a future workflow-learning replay.

KV-affinity scheduling is an opt-in layer on the coordinator: setting
`ledger.orchestrator.affinity_cap = <cap>` attaches an `AffinityScheduler`
(bounded by `cap` concurrent agent turns) via
`LedgerAgentCoordinator::build_affinity_scheduler(cap)` in
`build_ledger_coordinator`. The scheduler marks each `run_agent` session as
KV-affine (`affinity_session()`), giving its turns a priority bonus (minimize
context switches) while starved sessions age up. `None` (default) leaves
affinity bookkeeping off.

Live retrieval is another opt-in layer on the coordinator: composing a
`router::retrieval::NodeRetrievalService` via `with_retrieval` exposes
`retrieve_nodes(query, nodes)` — the M5 lemma-grep/fuzzy/cross-check tools
pre-filtered through the M6 `SalienceRanker` — as the entry point a future
tool-calling agent loop calls. `None` (default) → `retrieve_nodes` returns
empty (fail-open).

Both opt-in layers reuse the shared primitives (`ContentNodeStore`,
`SnapshotStore`, `DependencySession`/`CheckpointedStepGraph`, `Limiter`,
`AffinityScheduler`, `CreditFlow`) rather than adding a parallel store.

## Natural-language processing spine (spacy-rs + fluent-onnx)

The router's deterministic NLP layer is composed from the `spacy-rs` crate
(`src/spacy-rs/`, the native spaCy core) and the `fluent-onnx` crate
(`src/fluent-onnx/`, ONNX/`ort` workers).  Together they give the router a
unit-testable, model-free understanding of text *before* any model is
consulted, and a content-addressed **interlingua id** on every token that is
a pure function of the text and the YaGO 4.5 taxonomy.  The detail design
lives in `doc/spacy-rs/{VISION,ARCHITECTURE}.md` and
`doc/skills/interlingua/SKILL.md`; this section maps how the router consumes
them.

**Annotation ladder (`spacy-rs`).** `NlpPipeline` runs a **deterministic-first,
two-phase** ladder (ROADMAP_20260831_ARCEAGER): base phase `first_accept_in_order`
over `[ArcEagerRung, RuleRung]` (always produces a validated parse), then — when
`RefinePolicy` says so — refine phase `first_accept_in_order` over
`[EncoderRung, LlmRung]` (in that order; `LlmRung` via `stages/nlp.rs::annotation_fetch`
and the general-purpose onnx LLM default, `EncoderRung` feature-gated and not
configured by default). Refinement is **gated** on confidence (low
`overall`/`role_coverage`/near-tie `oracle_margins`) **OR** task-value (a
routing-critical role present but interlingua-unresolved, a PROPN with no entity
id, a surfaced `CollisionNote` — each behind its own `refine_on_*` flag) and
adopted only when it passes the 7-check gate **and** does not regress
`frame_coverage`; otherwise the base is kept. `RefinePolicy::Always` preserves
today's `LlmFirst` behavior; `RefinePolicy::OnUncertain` (the
`DeterministicFirst` wiring) consults models only on uncertainty or routing
incompleteness, with every `refine_on_*` threshold independently tunable via
`pipelines.<name>.refine_policy`. The deterministic rule rung guarantees a valid
parse, so a request is never left unparsed because a model was unavailable. The
router builds the pipeline through the builder with a resolver threaded in
(`NlpDeps`); when `nlp: true` and a concept store is present it builds
`en_default_with_strings` + a live `InterlinguaResolver` (backed by
`SqliteConceptStore`), else it `warn!`s and uses `en_default()` (fail-open).
The `NlpStage` carries the effective `RefinePolicy` (derived from
`nlp_ordering` unless `refine_policy` overrides, via `RouterRefinePolicy` DTO
`config/refine_policy.rs` field-for-field mirror with `From` both ways —
`spacy-rs` never deserializes router config directly, F5) and threads it to
`process_sync_with_confidence`; `NlpStage::new` defaults to `Off`
unconditionally, the builder is the single source of `Always`/`OnUncertain`
(F1/R6); `confidence_summary` treats refined products
(`Llm`/`Encoder` sources that are *not* confidence-bearing) as `1.0` by
convention, so a refined parse is never double-gated. Span-cache keys are
stored as fixed-width hex `span_key TEXT` (`{:016x}`), not `i64` cast (F7).

**The NLP stage.** `NlpStage` runs that pipeline on the inbound user text and
publishes per-sentence `spacy_rs::routing::RoutingSignal`s under the
`"nlp_parse"` handoff key. It is an **enrichment** stage, never a gate — a
parse problem must not reject the request. The classification tree's
`match_interlingua` filter nodes (`stages/tree/engine.rs`) dispatch on the
parsed interlingua ids directly (same phrasing → same ids → same route, zero
tokens) and gate on the signal's `confidence_min`.

**Frame resolution.** `spacy-rs::frame::FrameStage` sits between `attach` and
`resolve` in the pipeline, deriving per-predicate `Frame`s + a typed
`AmbiguityEntry` list and minting a **permanent** interlingua key only for an
ambiguity-free frame (a provisional key otherwise). `ledger/frame_index.rs`
implements the `PreferredSenseIndex` seam over the `interlingua_index`
pattern-cache rows (`role='sense'`) and runs the wave-batched
`FrameResolutionWorker` — one grammar-constrained call per `SupervisedBatch`
wave returning N resolutions, applied and promoted so a repeating
`(predicate, ambiguity_kind)` pattern replays deterministically (zero LLM cost
next time).

**Interlingua on the ledger.** `ledger/nlp.rs::record_parse_node_with_confidence`
writes the parse `ContentNode` and the per-sentence × role `interlingua_index`
rows in one transaction; `SqliteConceptStore` (`interlingua_concepts`) is the
durable home for the YaGO taxonomy, and `boot.rs` reconciles it id-membership
equal to coral's content-addressed graph at boot (one loader, two homes). Every
annotation is tiered (`fluent-types::ProvenanceTier { Deterministic < LocalModel
< Frontier < HumanReview }`) and keyed to the node's `content_hash`:
`ledger/annotations.rs::AnnotationStore` (written via
`ContentNodeStore::write_annotation`) supersedes — never deletes — a lower-tier
claim for the same node version, and a content mutation invalidates claims by
keying, never a scheduler.

**Async review.** `ReviewWorker` (`server/review.rs`) runs
`spacy_rs::review_prompt` + `apply_corrections` against the
`SqliteCorrectionIndex`, served at `POST`/`GET /v1/sessions/{id}/review-parse`.
The review call is served by the onnx LLM with the `ParseReview` grammar
(constrained at generation); an optional PII pre-filter (`fluent_onnx::pii::PiiSpanDetector`
or the `RegexPiiDetector` baseline) flags review candidates when
`review.auto_enqueue` is on. Corrected annotations write back with `HumanReview`
provenance (the highest tier).

**Entity-link overlay.** `EntityLinkWorker` (`server/entity_link.rs`) scores
unresolved PROPN spans against the boot-baked ColBERT `EntitySimilarityIndex`
(`ort.rs::colbert_entity_scorer`) and writes **candidates only** to
`overlay_candidates` (`ledger/overlay.rs`) — parse-time `interlingua_entity_id`/
`concept_ids` stamping remains boot-only-registration-invariant and is future
work (the VISION's Partial surface).

**Ranking pipeline (`ranking.rs`).** The local model ranks only a
**deterministic shortlist**, never the full session context. `SalienceScorer`
weights four ledger-derived signals (`frame_frequency`, `interlingua_centrality`
— a minimal `pagerank` power-iteration over the `interlingua_index`
co-reference + `parent_id` graph — `recency`, and `reference_count`, a derived
in-degree); `LedgerSalienceProvider` loads them from a read-only snapshot.
`rank_candidates` pre-filters to a top-K shortlist, then makes **one**
grammar-constrained onnx call over the shortlist only (degrading to the
salience order on failure). This is the model-ranks-shortlist invariant.

**Live retrieval seam (`retrieval.rs`).** `NodeRetrievalService` is the live
dispatch seam for the M5 subagent tools: it parses a candidate node's LOD0,
runs `spacy_rs::retrieval`'s lemma-grep (every hit carries `ParseConfidence`)
+ the fuzzy paraphrase axis (over an injected `EmbeddingProvider`) + the
`cross_check` combiner (material disagreement surfaces both axes), and
pre-filters the pool through the `SalienceRanker`. `LedgerAgentCoordinator`
exposes it via `with_retrieval`/`retrieve_nodes` (fail-open without it) — the
entry point a future tool-calling agent loop calls.

**Serving.** The generative ONNX LLM is a first-class serving surface alongside
the llama fleet. `main.rs` builds `onnx_llm_backend` via
`ort.rs::onnx_chat_backend(registry, OnnxRole::Llm.registry_key())` — a
`fluent_onnx::OrtLlmSession` (a `CausalLm` decoder) wrapped as a
`fluent_llm::client::ChatBackend` — and `RouterConfig::onnx_llm_backend()` is
the **single wiring point** for the LLM annotation rung, the review worker, and
ledger enrichment/tiering/summarization. The checkpoint is the general-purpose
LFM2.5-2.6B export at `/ai/models/lfm2/2.6b/LiquidAI/LFM2.5-2.6B-ONNX` (q4),
served **in-process on the CPU** (`execution_provider: cpu`,
`intra_threads: 1` for deterministic decode) and bounded by a `Limiter`, so it
runs concurrently with the GPU-resident llama fleet. Structured calls request
`response_format.schema`; the `ort` backend constrains decode with a structural
JSON automaton (`JsonObjectGrammar`/`JsonArrayGrammar`) so invalid output
shapes are prevented at generation time.

**Shared runtime base (`fluent-llm::runtime`, ROADMAP_20260830_LLMS).** Both
fleets present through the same control-plane traits — `LlmWeights` (one loaded
weights instance), `LlmContext` (one named context window with its own KV),
`LlmKVCache` (snapshot/restore/list/delete), and `LlmResidencyRow` — so the
aggregate `/instances` facade, `/v1/models`, `coral-router ps`, and
`POST /models/unload` drive llama and onnx uniformly. The **single
`LlmResidencyEngine`** (`fluent-llm::runtime`) replaced the two sibling loops:
one pass over every weights instance with injected per-fleet budgets (VRAM for
llama, CPU RAM for onnx) and eviction orderings (`FootprintColdness` for llama,
`LruLargest` for onnx — never unified). The onnx generative role now serves
**named contexts** from an in-process `OnnxContextPool` (one weights load, N
context windows, per-context KV) — the defining capability of the llama fork —
and `POST /instances/:name/resize` routes onnx contexts through
`LlmContext::resize` (bounded by `max_ctx`). The old
`OrtResidencyLoop` is a delegating compatibility shim over the shared engine.

**ONNX residency parity.** The shared `LlmResidencyEngine` (spawned once, over
the fleet of llama adapters + onnx implementors) applies the same
unload-from-memory policy the VRAM fleet has: idle release past a per-role
`sleep_idle_seconds`, LRU-largest working-set eviction against
`sidecar.onnx_working_set_budget_bytes`, and `pinned`/`Always` entries never
released — CPU-RAM parity for `ort` sessions in the same loop, not a sibling
task.

The **encoder-role world** (the two-tower `zero_shot_routing` Prompt-Router
overlays, the `fill_mask`+`annotation_heads` encoder rung, the `token_classification`
PII-Detector, the Policy-Linter, the `late_interaction` ColBERT retriever) is a
feature-gated legacy seam: those functions remain behind `cfg(feature = "onnx")`
in `ort.rs`, but no encoder-role model is configured in the reference deployment —
the general-purpose LLM does enrichment. The opt-in PII pre-filter
(`fluent_onnx::pii::PiiSpanDetector` or the `RegexPiiDetector` baseline) and the
entity-link ColBERT `EntitySimilarityIndex` remain live seams where enabled.

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
