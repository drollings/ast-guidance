# Coral Router — Vision

*This document is the aspirational brief: the goals of Coral Router and its
ideal finished design. It deliberately does not track what is landed today —
that lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md), which describes the
current implementation and which pieces are load-bearing.*

## Mission

Coral Router is a local-first control plane for LLM traffic: a single
OpenAI-compatible endpoint that decides, for every request, the cheapest and
safest way to answer it — deterministic logic where possible, a small local
model where sufficient, larger local models where warranted, and frontier
providers only when genuinely necessary. To anything calling it, it behaves
like one coherent, capable model. Underneath, it is a disciplined mixture of
deterministic filters, small classifiers, local reasoning models, and
occasional frontier calls, none of which are consulted unless a cheaper stage
has already failed to resolve the request.

Coral Router is also the process owner of the local inference fleet and the
routing element between it and everything else: it spawns and supervises one
`llama-server` process per model weights file, serves the `/instances`
management contract at its own address, and is the single router between those
llama-server tasks and every other OpenAI-compatible endpoint. A dispatch is a
direct call to the owning server; a frontier or remote call is the same request
routed onward after the local ladder has failed.

## Design principles

- **Deterministic before probabilistic.** Anything decidable by a regex or a
  fixed rule should never reach a model call. This is a cost and latency
  floor, not an optimization — it also gives the system a layer that is fully
  unit-testable with no model in the loop.

- **Cheap before expensive.** Every model carries its own cost and speed
  profile. Routing is an economic decision as much as a capability one: the
  ladder runs deterministic filter → fast classifier/score-matrix → local
  orchestrator or agent → frontier, and a request only reaches a given rung
  after the previous one has genuinely failed to resolve it — never by
  default.

- **Condensed context, not accumulated context.** Sessions compact rather
  than grow without bound. The ledger is the mechanism: it stands between raw
  session history and the orchestrator's live KV cache, so the orchestrator
  never has to reason over noise, dead ends, or superseded exploration — that
  material stays in durable storage, retrievable if needed, but off the
  model's working context.

- **LOD0 is authoritative; nothing else is derived from anything but LOD0.**
  Every level of detail below full text is a summary, and summaries drift.
  The failure mode to avoid is a summary-of-a-summary: if LOD3 is computed
  from LOD2 and LOD2 from LOD1, an error introduced at any tier becomes
  unfalsifiable a few tiers down. Every LOD tier is therefore computed
  directly from LOD0, never chained from a lower-fidelity tier, and any route
  doing adversarial or high-stakes reasoning (`rigor`, in particular) is
  entitled to dereference LOD0 rather than trust a cached summary.

- **Local-first, frontier as a bounded, audited exception.** Frontier calls
  are for genuine difficulty, privacy-sensitive decomposition, or a real
  capability gap — never a default path. Every frontier interaction, in any
  of its modes, writes back to either the durable audit log or a reusable
  local artifact (a stored workflow, a validated rubric/answer pair). The
  metric that tells you this design is working is frontier-call frequency
  trending *down* over the life of an installation as those local libraries
  fill in — not staying flat.

- **Terminate, don't loop.** Anywhere the system reaches for more than one
  model pass on a single request — the `rigor` route's blue/red/judge
  sequence, the `plan` route's clarifying interview — the round count is
  fixed in advance, never open-ended. This is a deliberate rejection of the
  failure mode seen in academic multi-agent ensembles and debate systems,
  which run every available model on every query with no adaptive gating and
  burn tokens accordingly. Escalation past the fixed structure (e.g., to
  frontier) happens only on a specific, named trigger — low judge confidence,
  not "red team scored a point" — never as a default resolution.

- **Structural separation by origin, not just by prompt discipline.**
  Content entering the system carries a role — user, system, tool result,
  subagent, self — and that role should be visible in the ledger's structure,
  not just implied by prompt formatting. This is cheap instruction-hierarchy
  hardening: it does not require a stream-native model to pay off, only
  consistent typing of Content Nodes by origin at write time.

- **Auditable by construction.** Every filter, classification, route, and
  frontier decision produces a legible reason alongside its verdict, written
  to a durably-retained audit stream distinct from routine operational logs.
  A rejected, redirected, or escalated request should be explainable after
  the fact without guesswork.

- **Own the serving processes, route everything else.** Coral Router spawns
  and supervises one `llama-server` per model weights file — found on `$PATH`
  — on a free localhost port, and talks to each directly. The llama.cpp router
  mode is never used; Coral Router is the only orchestrator and the single
  routing element between its own llama-server tasks and every other
  OpenAI-compatible endpoint. It owns all VRAM policy through the `/instances`
  management API it serves at its own address, so the servers bind to
  `127.0.0.1` and are never exposed directly.

- **Reuse infrastructure, extend it, don't parallel-build it.** Enforced by
  explicit import-boundary rules and a documented DRY convention, not just
  stated as a preference. A change that reimplements something the shared
  crates already provide — graph algorithms, hashing, config loading, error
  types, shared-string interning — is treated as a defect to fix, not a
  style choice to debate. This applies to new ledger/Content Node work as
  much as to anything already shipped.

## The Classification Tree: a self-updating routing config

The central configuration structure for Coral Router is a **nested
classification tree** — not a flat list of routes, a separate score matrix,
and a hardcoded system prompt that drift apart as the deployment evolves.

### Node types

Every node in the tree is one of four types:

| Type | Role | LLM call? |
|------|------|-----------|
| **`classifier`** | An LLM call that picks one child branch. The prompt is auto-generated from the children's descriptions. | Yes (small local model) |
| **`terminal`** | A dispatch target. Resolves to a model from a named `model_group`, optionally with a specific `session` profile. | No — terminal is where the routed model takes over |
| **`filter`** | A deterministic check (regex, prefix match, PII pattern). Produces `hard_reject`, `soft_redirect`, or `output_filter`. | No |
| **`fallback`** | A child of a classifier node that resolves to a fallback dispatch *target* — used when the classifier picks no named child, its LLM call fails, or the chosen branch's target cannot resolve. A fallback always lands on a model that will answer the request; it is never a backup for the classifier's own classification job. | Only if the wrapped node is itself a classifier (in which case it routes onward, still to a target) |

### Prompt auto-construction

A classifier node carries a `description` and a map of named `children`.
From those children the system constructs the prompt body:

```
You are a {node.description}.

Available routes:
- {child_key}: {child.description}
- {child_key}: {child.description}
...

You must output exactly one JSON object with:
  "route": "<exactly one of: {comma-joined child keys}>"
  "coherence": 0.0–1.0 (how well-formed and coherent the query is)
  "safety": 0.0–1.0 (1.0 = completely safe, 0.0 = policy violation)
  "complexity": 0–10 (0 = trivial, 10 = requires most capable model)
  "reason": "brief explanation for the routing decision"
```

If a child key is added, removed, or its description changes, the prompt
updates automatically. No manual prompt maintenance. No stale route names.

### Three axes of routing

1. **Domain** — the classifier's primary output: `"code"` vs `"prose"` vs
   `"translation"`, etc. This is the first branch, matching the human
   category the query falls into.

2. **Coherence / Safety** — every classifier node enforces configurable
   thresholds. A query below the coherence threshold is rejected
   (nonsensical / adversarial input). A query below the safety threshold
   is rejected (policy violation). These are the gating checks that
   protect downstream models from garbage or harmful input.

3. **Complexity** — each model carries an `intelligence` field (0–10),
   configured in `env/coral-router.json`. A `model_group` is an ordered
   list of target models, and the classifier's job is to choose the
   best-matching *target* within that group. The current candidate target
   is prompted to classify the prompt, and if the complexity exceeds that
   model's configured `intelligence` — while a more intelligent model
   exists in the group — the request falls back to the next model in the
   group, which re-classifies. The first target whose `intelligence` meets
   or exceeds the complexity is the one that actually answers, and its
   answer is recorded in the session's ledger. The group's chain is a
   target-matching ladder, not a failure fallback and never a backup for
   the classifier. This is automatic for every terminal node, not a
   separate config section.

### Pre-filters: deterministic before probabilistic

Before any node in the tree is evaluated, a `pre_filters` list runs. These
are pure regex / prefix-match checks with no model in the loop:

- **`hard_reject`** — ends the request immediately with an HTTP error code
- **`soft_redirect`** — sends the request directly to a named branch,
  skipping the classifier

Pre-filters are the cheapest possible decision and protect the classifier
from work it should never see (PII-bearing content, known-bad patterns).

### Complexity-based branching (optional)

Classifier nodes can also branch on complexity directly, for deployments
that want explicit complexity bands rather than dispatch-time filtering:

```
root (classifier, model=fast)
├── low_complexity → (terminal, group=fast)
├── high_complexity → (terminal, group=code)
```

When a classifier has children, it asks the LLM to pick one. The children
can represent any axis, including complexity. This gives the operator full
control: domain-only, complexity-only, or both in a single tree.

### Tree replaces flattened config

The classification tree replaces four previously-separate config sections:

| Old section | Replaced by |
|-------------|-------------|
| `pipelines` | Tree structure IS the pipeline — each classifier node is a stage |
| `routes` | The children of each classifier node |
| `system_prompt` | Auto-generated from the tree children + descriptions |
| `score_matrix` | Coherence/safety thresholds at each classifier node + complexity-based model selection |

`models`, `model_groups`, `server`, and `logging` are unchanged.

## The Escalation Ladder: progressive frontier engagement

When a terminal node dispatches to a `model_group` and every local model in
that group's chain fails or times out, the system does not fail outright.
Instead it escalates through a configurable **escalation ladder** — a fixed
sequence of increasingly permissive frontier-engagement modes. Each mode is
a discrete policy governing how much context, data, and agency the frontier
model receives.

### Why a ladder, not a single fallback

Frontier models are expensive, external, and outside the local trust
boundary. Straight turnover is the **most permissive** option — it gives the
frontier everything. By ordering less-permissive modes first, the system
only pays the cost and takes the risk of full context exposure when genuinely
necessary. The ladder makes frontier calls progressively more expensive, not
all-or-nothing.

### The four modes

| Stage | Mode | What the local system does | What the frontier sees | Frontier risk |
|-------|------|---------------------------|----------------------|---------------|
| 1 | **filter** | Deterministic PII/anonymization rules strip sensitive content from the query. The filtered query is sent as a one-shot prompt to frontier. | Filtered/de-identified text only | Low — no raw data crosses the boundary |
| 2 | **question** | A `decomposer_model` (fast local LLM) breaks the problem into generic hypothetical questions. The frontier answers each independently. An `assembler_model` synthesizes the responses into the final answer. | Abstract hypotheticals with no personal data, no session context | Low — frontier sees constructed questions, not user data |
| 3 | **team** | `classifier_parallel` instances of a `classifier_model` run in parallel slots and vote on approach. A `draft_model` attempts the easier sub-steps locally. A `judge_model` reviews the draft, identifies gaps, and crafts a precise frontier prompt containing only the unsolved sub-problem and the successful partial work. | A focused prompt with the unsolved gap and verified partial work | Medium — frontier sees partial solution structure |
| 4 | **turnover** | Full context handoff. The frontier model receives the entire session ledger, all tool access, and continues autonomously. All subsequent messages in the session go through frontier. | The entire session — all context, tools, history | High — frontier has full agency and raw data |

Each stage is tried in order. If the frontier rejects/errors, or the local
assembler/judge rejects the frontier's output, the system escalates to the
next stage. If all stages are exhausted without a successful response, the
request fails with an escalation-exhausted error.

### Parallel classifiers (team mode)

The `team` mode uses `classifier_parallel` slots of the same `classifier_model`
running in parallel via `ResultPool` — the same primitive used for
continuous-batching LLM fan-out. Each slot receives the same query with a
slightly varied temperature/seed, producing a set of votes. The vote
distribution (e.g., "3/3 say decompose into sub-tasks X, Y, Z") feeds into
the draft model's prompt as a structured signal. This avoids the config
complexity of managing N different classifier models while still getting
diversity through stochastic variation.

### Target selection within a group (per group)

Before escalation even begins, a `model_group` is an ordered `local` chain
of target models:

```
default:
  1. swarm (intelligence 2)          ← cheapest target; self-assesses the prompt
  2. qwen3.6-27b (intelligence 6)    ← next target if complexity exceeds swarm's
```

The chain is a complexity-matching ladder for *targets*. The classifier
resolves a route to a group; within the group, the current target model is
prompted to classify the prompt, and if the complexity exceeds that model's
configured `intelligence` — and a more intelligent model exists in the
group — the request falls back to the next model in the group, which
re-classifies. The target that matches the complexity actually answers the
request, and its answer is added to the session's ledger. Only if every
local target fails for a mechanical reason (unreachable, timeout, incoherent
output) does the escalation ladder engage. The frontier is never consulted
when a local target can handle the query — even the smallest local target
gets its self-assessment shot first. Fallback models are always target
models; none of them backs up the classifier's own classification job.

### Post-processing: audit + workflow extraction

Every frontier interaction, in any escalation mode, writes a structured
entry to the durable audit log recording:
- Which escalation stage fired
- What the local system sent to frontier
- What the frontier returned
- Whether the local assembler/judge accepted the result
- Total cost incurred

Per-group `post_process.workflow_extraction` controls whether successful
frontier-aided solutions that are **not** already in the context cache get
processed into reusable DAG workflows:

1. The full `query → local_attempts → escalation_stage → frontier_call →
   assembly` chain is decomposed into discrete steps.
2. Each step becomes a `Target` node in a DAG, with `depends` / `provides`
   edges capturing the dependency structure (e.g., "the frontier response
   depends on the judge's crafted prompt").
3. The workflow DAG is stored as `ContentNode` entries in the ledger's graph
   database, keyed by an embedding of the original query.
4. When a future query has a near-neighbor embedding, the cache reactor
   can replay the DAG steps — skipping the frontier call entirely when
   the same decomposition structure applies.

This is the "neurosymbolic learning loop": the frontier path becomes a
one-time cost that amortizes across similar queries.

## The Serving Layer: owned llama-server processes

Local models are not reached through a third-party gateway: Coral Router owns
the serving processes. It spawns and supervises one `llama-server` process per
model weights file, assigns each a free localhost port, and keeps it under
supervision — restarting with backoff if it dies. It is the routing element
between those llama-server tasks and every other OpenAI-compatible endpoint,
and it never reaches a llama.cpp router: a dispatch is a direct HTTP call to
the owning server.

- **One process, one pool.** A model's weights are loaded exactly once per
  `llama-server`. From those weights the server allocates named context
  windows ("instances"), each an independent window of exactly its own
  `ctx_size`; `parallel` slots share that one window and never multiply or
  divide it. To run N full-size contexts the operator declares N instances
  (`count: N`), never `parallel: N`.

- **Declarative pools.** `models.<id>.instances` is the instance spec: a
  `count: N` profile expands to N sibling instances named `<key>-0` ..
  `<key>-{N-1}` in a shared group. Coral Router materializes the pool at boot
  and reconciles drift through the management API.

- **Model id translation.** Config keys are the public model ids; llama.cpp
  model names are internal. Coral Router translates between them when it
  proxies a request or a management call, so a client addresses an instance as
  `<model_id>:<instance>` while the server sees `<llama-name>:<instance>`.

- **Direct management.** The `/instances` contract — create, destroy, pin,
  resize, snapshot — is served at Coral Router's own address as the single
  sidecar entry point, aggregated across every managed model under
  `<model_id>:<name>` ids with 64-bit-summed memory. The managed servers bind
  to `127.0.0.1` only and are never exposed directly.

- **VRAM policy lives in the router.** The sidecar polls the aggregate
  `/instances` envelope, evicts least-recently-used unpinned instances when
  free device memory drops below a watermark, and allocates fresh KV on a 503
  group miss. It may retire a server process after its last instance is
  evicted; the weights are freed either way.

### VRAM residency: load on demand, evict by LRU and size

The fleet will not fit in device memory all at once, so residency is an
explicit, first-class goal rather than an accident of whatever fits:

- **Only pinned instances boot.** A model is spawned at boot only when it
  declares at least one pinned instance profile (e.g. `swarm:ledger`,
  `swarm:swarm`). Everything else — whole models with no pinned instance, and
  unpinned context windows within a booted model — is loaded on demand at first
  dispatch and is a candidate for eviction.

- **Weights load on demand and are unloaded again.** The supervisor spawns a
  lazy model's `llama-server` on its first routed request (`ensure_running`)
  and waits for `/health` before dispatching. When the sidecar evicts the
  model's last context, the server is stopped and its weights freed
  (`unload`); the next request reloads it. No model stays resident merely
  because it was once used.

- **Contexts load on demand too.** Unpinned instances (e.g. `swarm:scratch`)
  are not declared at spawn; a request targeting one creates it via
  `POST /instances` and it becomes an eviction candidate like any other.

- **Eviction is LRU, weighted by size.** The residency loop enforces an
  allocation budget of `device_total - minimum_remaining_vram` (device total
  from `sidecar.vram_total_bytes` or auto-detected from ROCm
  `mem_info_vram_total`; floor from `sidecar.minimum_remaining_vram`). When
  the budget is exceeded it evicts the least-recently-used **largest** unpinned
  context first — freeing the most VRAM from the coldest window — across every
  managed server on the device, not per server. Pinned instances are never
  evicted.

- **A model with no contexts is a model that leaves.** After evicting a
  model's last context window, the router unloads that model's weights rather
  than letting them sit idle; residency is measured by how few weights and
  contexts stay resident when they are not earning their VRAM.

This is what makes the fleet coherent: the router keeps the fastest, most
relevant model resident for the traffic it actually serves, and spends the
rest of device memory the moment a request earns it — never before.

- **Routing fields.** Every generation request may carry `model`, `instance`,
  `snapshot`, and `id_slot` from the JSON body or the query string (the body
  wins). Coral Router resolves the model id to its owning server and forwards
  the remainder, so a conversation can switch KV snapshots in and out of a
  slot without re-prefilling.

## The Ledger: Content Nodes and levels of detail

Every paragraph, prompt, tool result, or intermediate artifact is stored as a
**Content Node** — the game-engine concept of level-of-detail and scene graph
applied to semantic text. A `ContentNode` is the canonical type (defined in
`fluent_types`) that unifies durable storage fields with session-scoped
metadata — no separate `ContextNode` / `SessionNode` split. The 6-tier LOD
scheme is defined here (as routing policy); storage and rendering of
individual tiers is delegated to the `ContentNode` store:

| Tier | Description | Bound |
|------|-------------|-------|
| LOD0 | Full text | — (authoritative source) |
| LOD1 | Compressed but complete | no fixed bound, but lossless-in-substance |
| LOD2 | Short summary | ≤ 1000 characters |
| LOD3 | Compact summary | ≤ 280 characters |
| LOD4 | Single line | ≤ 80 characters |
| LOD5 | Name / label | brief, for listings and identification |

**Computation and caching.** LOD0 and LOD5 are guaranteed filled at node
creation — LOD0 because it is the authoritative anchor everything else
derives from, LOD5 because cheap identification and listing (directory-style
browsing of the ledger, dependency-graph node names, audit-log references)
needs a label to exist unconditionally. LOD1–LOD4 are computed lazily, on
first access, directly from LOD0 (never from each other), by a small local
model, and cached on the node thereafter. "At most once" is a property of the
node, not of any one caller: the first ledger or agent that requests a given
node's LOD2, say, pays the summarization cost; every subsequent request for
that node's LOD2, from any ledger, hits the cache.

**Metadata.** Each node carries what it needs to be more than isolated text:
related filesystem paths, database lookup keys, embeddings, and KV-cache
snapshot references where applicable. This is what lets a node be rendered
either as bare text or as an anchor into richer context (a file on disk, a
prior session's KV state, a knowledge-graph entity).

## Ledger HNSW: conceptual-distance scene graphs

Each LOD tier maintains its own lazily-computed HNSW index over that tier's
node embeddings — a "scene," in the game-engine sense, recomputed only for
the dirty subset when the nodes it covers change. This is the mechanism that
lets the ledger behave like a stable, bounded context window instead of a
monotonically growing one: conceptual distance from the current focus
determines how a node renders. Near neighbors render toward full detail
(down toward LOD0); distant nodes collapse toward LOD4/LOD5. A specialized
agent can request a scene rendered with a different subjective focus, and get
a different fidelity distribution over the same underlying nodes without any
of them being duplicated or recomputed.

This is deliberately a **separate concern** from the three library-scale
HNSW indices — the prior-workflow library, the rubric/validated-answer
cache, and the blacklist-similarity index. Those operate cross-session,
over durable artifacts, and are kept as separate indices from each other
because a false positive means something different, and costs something
different, in each case (a workflow-library miss just falls back to planning
from scratch; a blacklist-similarity false positive is a false accusation).
The ledger's per-level scene graphs are session-scoped, operate at a
different granularity, and are not merged into the library-scale indices —
five index concerns, kept apart, each with its own acceptable error rate and
its own dirty/rebuild cadence.

All five HNSW index instances (one per LOD tier, three library-scale) are
built using the same shared HNSW factory (`fluent-db::hnsw` /
`common_core::sqlite::make_hnsw`) and stored in the router's SQLite store.
Coral Router owns the index scoping and the decision of which index to query
for a given routing or cache operation.

## Shared Content Nodes and parallel ledgers

A **ledger** is a nested-list view — directory/file-tree-like — of pointers
into a shared, reference-counted Content Node store. Ledgers do not own
nodes; they reference them. This makes parallel ledgers cheap: an
orchestrator's ledger, a subagent's ledger, and a rigor-route judge's ledger
can all hold reference-counted pointers to the same underlying nodes while
each maintains its own **default level of detail** — the orchestrator might
default to LOD1 for breadth, a narrow specialist to LOD3 for focus — without
duplicating any text.

Because LOD1–LOD4 are cached on the shared node itself rather than per-ledger,
the "computed at most once" guarantee holds globally: whichever ledger first
triggers computation of a tier pays the cost once, and every other ledger
referencing that node — present or future — gets the cached result for free.

This requires cheap, shared string storage to actually pay off. Node
identifiers, tags, and cached LOD strings should be backed by interned,
reference-counted strings — the same `ArcIntern<str>` pattern
`fluent-concurrency` already uses for work-unit names and dependency-graph
asset names — so that sharing a node across N ledgers costs a refcount bump,
not a copy, and identical strings across nodes (a recurring entity name, a
common tool-result shape) are deduplicated automatically rather than stored
redundantly per node.

## Filtered ledgers

A **filtered ledger** is a lightweight overlay over an existing ledger: the
same reference-counted pointers, minus an exclusion set, rather than a copy
of any content. Building one is cheap — construct a filtered reference list
— and discarding one is cheap — drop the list; the underlying nodes are
untouched and remain owned by the shared store.

This is the natural mechanism for several cases that would otherwise need
bespoke copying logic:

- A PII-anonymized view of a ledger handed to a frontier call.
- A red-team ledger in the `rigor` route that excludes blue-team's already-
  rejected dead ends, without needing to physically prune anything from the
  underlying session.
- A specialist agent's narrowed view that excludes nodes outside its concern
  — the multi-stream-inspired "give each role only what it needs" principle,
  realized as a reference filter rather than a context-assembly rewrite.

Because a filtered ledger only manipulates references, filtering never forces
recomputation of any LOD tier and never duplicates cached content. The cost
of constructing a filter is proportional to the size of the exclusion set,
not to the size of the underlying node population.

## Lessons from parallel-stream architectures, applied without retraining

A separate line of work on multi-stream language models — instruction-tuning
a model to read from and write to several causally-dependent token streams
in a single forward pass, one stream per role — motivates several pieces of
this design, without requiring Coral Router to depend on a stream-native
model:

- **Adopted now, structurally:** Content Nodes are typed by origin (user,
  system, tool result, subagent, self), and that typing is preserved through
  rendering rather than flattened into an undifferentiated prompt. This gets
  most of the instruction-hierarchy hardening that true stream separation
  provides — a cleaner structural signal of where content came from —
  without needing a purpose-trained checkpoint.

- **Adopted now, as a node convention:** a dedicated `audit`/`concern` node
  type, populated by agents alongside their normal output, gives a legible,
  separately-stored channel for considerations that shouldn't necessarily
  surface in the user-facing answer — the same shape of benefit as the
  parallel-stream architecture's auxiliary thinking streams, materialized
  here as ledger content rather than a causally-entangled model output. It's
  a weaker guarantee (it depends on agents actually populating it honestly,
  rather than being architecturally inescapable), but it's a real, cheap
  approximation that plugs directly into the existing audit-trail principle.

- **Translated, not adopted literally, for efficiency:** the throughput gain
  parallel streams get from one memory-bound forward pass serving many
  streams at once translates, for an off-the-shelf llama.cpp deployment,
  into parallel-slot / continuous-batching support on shared resident
  weights — many classifier or agent calls sharing one loaded model's memory
  bandwidth, not literally one forward pass emitting many roles. This is the
  correct reading of "small local models run in parallel across many
  requests" for this stack.

- **Deliberately deferred:** a genuinely stream-native local model —
  instruction-tuned so an agent can, say, keep composing a user-facing answer
  while a search result arrives mid-generation and gets incorporated without
  restarting the turn — is a real, scoped option for later, requiring its
  own fine-tune on stream-formatted data. It is not a prerequisite for
  anything above and is treated the same way the four frontier-involvement
  modes and the adapter registry are treated: a named longer-term direction,
  not something the near-term ledger and routing work waits on.

## The fully realized system

A request arrives and passes through a strict escalation ladder, spending as
little as possible at each rung before the next is even considered.

**The serving layer is already there.** At boot Coral Router finds
`llama-server` on `$PATH` and spawns one process per model weights file on a
free localhost port, loading each model's weights exactly once. It materializes
the configured instance pool, then supervises every process for the life of
the installation. A routed local dispatch is a direct HTTP call to the owning
server carrying the translated model id and any `instance`/`snapshot`/`id_slot`
fields; a management call goes through Coral Router's own `/instances` API,
which aggregates every pool into one memory surface. No llama.cpp router is
involved at any point.

**Deterministic filters** run first, with no model in the loop, resolving to
one of three outcomes: a hard rejection that ends the request outright, a
soft redirect that sends it down a different path, or an output filter that
redacts, anonymizes, or omits specific content before anything continues.
These filters are scoped (some apply only to frontier-bound traffic, some to
the Content Node write path where local summarization could otherwise cache
unfiltered sensitive content) and can be gated behind a secondary check, so a
rule never fires on a bare pattern match alone when a cheap confirmation is
available.

**A fast classifier** — small, fast, running across parallel slots on shared
resident weights — evaluates intent, coherence, safety, and complexity, and
resolves the result through a weighted score matrix rather than nested
thresholds. Most requests are fully decided by this point: answered
trivially, rejected, or routed to a specific local model, all without
touching the system's larger models. Routing lands on the best-matching
*target* within a target model group: the current target self-classifies
the prompt, and when the complexity exceeds that model's `intelligence` the
request falls to the next, more intelligent model in the group; the target
that matches the complexity answers, and its answer is recorded to the
session's ledger.

**The Ledger** — nested Content Nodes, HNSW-scened per level, shared and
reference-counted across parallel and filtered views — replaces a large
accumulated context window. Conceptual distance between nodes determines
whether they render in full detail or collapse toward a summary or label,
giving any session a stable, boundedly-sized context regardless of its raw
length, renderable at whatever fidelity and whatever subjective focus a
given agent needs.

**Two purpose-built routes** handle requests that don't fit the standard
path. A vague or underspecified request goes through **planning**: matched
against a library of prior workflows where possible, or built fresh by
identifying exactly what's missing and asking the user a short, targeted set
of questions to fill the gap — never an open-ended back-and-forth. A
complete but high-stakes request can go through **rigor**: a fixed
blue-team/red-team/judge sequence, checkpointing the reasoning model's KV
cache first so a red-team-identified dead end can be rewound rather than
argued out of in place, with red team and judge dereferencing LOD0 rather
than a cached summary when the material under review is high-stakes. When
red team raises something material, the default resolution is a targeted
interview with the user — not silent escalation.

**Local reasoning models handle the bulk of real work** — an orchestrator
handles the largest context window as rendered from the Ledger, and
specialist agents are reached via adapter switching on shared base models
rather than one model per role, scheduled with awareness of KV-cache affinity
so context switches are minimized rather than incidental.

**Frontier models are the last, narrowest rung**, used in one of a small set
of deliberate modes: a pure fallback for problems genuinely beyond local
capability; a PII-anonymized fallback for sensitive content (served via a
filtered ledger, not a redacted copy); a decomposed, anonymized hypothetical
question with a validation rubric, for when only a narrow piece of frontier
reasoning is needed; or a copilot/judge role reviewing the local model's
in-progress reasoning at checkpoints. Every mode is logged to a durable,
separate audit trail, and every frontier answer that proves out feeds back
into a stored workflow or a validated rubric — so the same class of question
never has to pay frontier cost twice.

The system as a whole should feel, from the outside, like a single capable
assistant. From the inside, it should be legible at every step: which rung
handled a given request, why, and what it cost.

## What this project deliberately is not

- Not a general-purpose LLM gateway or multi-tenant API product — it's built
  for one local workstation's traffic.
- Not a wrapper around a third-party gateway crate's, or reference project's
  (litellm-rs, aichat), routing, auth, or caching logic — those are mined for
  patterns only, never imported as dependencies. Routing, scheduling, and
  caching are purpose-built around KV-cache affinity, which generic LLM
  gateways have no concept of.
- Not reliant on frontier models for anything a well-scaffolded local model
  can be made to handle credibly — frontier usage is a deliberate, bounded,
  audited exception, not the default path.
- Not reliant on a stream-native, purpose-trained model as a prerequisite for
  any of the above. The structural and monitorability benefits of parallel-
  stream architectures are adopted now as ledger and node-typing
  conventions; literal multi-stream fine-tuning remains an optional,
  deferred track evaluated on its own merits.
- Not an ensemble-by-default system. Unlike academic mixture-of-agents or
  multi-agent-debate designs, which improve output quality by running every
  available model on every query with no cost constraint, Coral Router treats
  every additional model call — local or frontier — as something a prior,
  cheaper stage must have failed to resolve first. Quality comes from
  routing and verification discipline, not from brute-force ensembling.

- Not a re-implementation of llama.cpp's router mode. Coral Router never
  reaches a llama.cpp router; it spawns and supervises the `llama-server`
  processes itself, talks to each directly, and owns the `/instances`
  management contract at its own address. Building a second router on top of
  the llama.cpp router would parallel-build exactly the orchestration this
  project exists to provide.
