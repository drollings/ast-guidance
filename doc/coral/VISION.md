# Coral Context: Architectural Vision

**A Deterministic-First Context Graph Library with MCP Server and Multi-Tier Cache**

---

## Executive Summary

Coral Context is a **Rust-native context graph library** that provides a 6-tier intelligent cache, SQLite-backed graph database, MCP server interface, and WASM plugin runtime. It serves as the knowledge backbone for guidance, separating deterministic lookups from probabilistic inference.

### The Core Goal

Traditional AI systems invoke probabilistic models for every query, incurring latency, cost, and unpredictability. Coral Context inverts this relationship:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC-FIRST EXECUTION MODEL                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Traditional AI:       Query → LLM → Response (slow, expensive, variable)  │
│                                                                             │
│   Coral Context:       Query → Cache Tier Check → Result                    │
│                            ↓ (miss at each tier)                            │
│                       L1 Memory → L3 Graph → L4 Semantic →                  │
│                       L4.5 Decompose → L5 Frontier                          │
│                                                                             │
│   Result: Sub-100ms for cached patterns, zero marginal cost, auditability  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Outcomes**:
- **Sub-100ms latency** for deterministic execution paths
- **Zero marginal cost** for cached patterns (no LLM API calls)
- **Full auditability** through deterministic replay
- **Continuous improvement** as solutions become permanent cached nodes
- **Edge-native** design for resource-constrained environments

---

## Design Philosophy: Goals Over Implementation

### Goal 1: Replace LLM Reasoning with Cache-Tier Resolution

The fundamental innovation: **cascading cache tiers** instead of prompt-based reasoning. Each tier is progressively more expensive, and the system stops at the first hit:

- **L1 Memory**: LRU in-memory cache for hot queries (<1ms)
- **L3 Graph**: SQLite keyword search + recursive CTE graph traversal (<10ms)
- **L4 Semantic**: Brute-force KNN cosine similarity over embeddings (<50ms)
- **L4.5 Decompose**: Delegated to Coral Router — a local LLM decomposes complex
  queries into subtasks. The cache reactor checks for cached decompositions
  but the decision to decompose is a routing decision made by the Router's
  escalation ladder. (200ms when no cached decomposition exists)
- **L5 Frontier**: Delegated to Coral Router — external LLM dispatch
  controlled by the Router's four-mode escalation ladder (filter →
  question → team → turnover). The cache reactor records frontier results
  as cached ContentNodes but never initiates a frontier call on its own.
  (500ms+)

### Goal 2: Edge-First Efficiency

Every component is designed for resource-constrained environments:

- **Memory Safety**: Rust's ownership model prevents leaks
- **Single-Process Embedding**: SQLite runs in-process, no separate database server
- **Token Optimization**: LOD packing reduces context window requirements by 80%+
- **No Runtime Overhead**: Zero-cost abstractions via Rust's trait system

### Goal 3: Neurosymbolic Learning Loop

When deterministic paths fail, the system learns from the solution:

```
Novel Query → Router Escalation Ladder → Router decomposes or engages frontier
                                ↓
                    Solution Cached as New Node
                                ↓
                Next Time: Deterministic Execution (< 50ms)
```

The expensive probabilistic step becomes a **one-time cost**, not a recurring one.

### Goal 4: Security Through Sandboxing

Dynamic tools (LLM-generated or user-provided) run in isolation:

- **WASM Sandboxing**: Extism provides memory-safe execution
- **No Host Access**: Filesystem and network access blocked by default
- **SSRF Protection**: URL validation blocks private IPs and remote HTTP
- **PII Anonymization**: Regex-based redaction for sensitive data

---

## Architectural Components

### Component 1: SQLite Graph Database (`db.rs`)

The core storage layer — a single SQLite database replacing dual-engine architectures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SQLite GRAPH DATABASE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Tables:                                                                   │
│   ├── context_nodes    (id, name, source, lod, embedding, capabilities)    │
│   ├── edges            (source_id, target_id, edge_type, weight)           │
│   ├── wasm_tools       (name, path, capabilities)                          │
│   ├── targets          (name, bit_index, depends, provides, command)       │
│   ├── embedding_cache  (query_hash, query_text, embedding)                 │
│   ├── entity_types     (node_id, type_iri)                                 │
│   └── entity_hierarchy (subclass_iri, superclass_iri)                      │
│                                                                             │
│   Query Modes:                                                              │
│   ├── SQL + recursive CTE   (topological traversal, BFS/DFS)               │
│   ├── Brute-force KNN       (cosine similarity over float32 BLOBs)         │
│   ├── Hybrid search         (keyword + vector with RRF merge)              │
│   └── Duck typing           (recursive CTE is_a hierarchy traversal)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Capabilities**:
- Thread-safe via `Mutex<rusqlite::Connection>`
- KNN search capped at 100K candidates
- Recursive CTE for graph traversal with depth limit
- Batch insert with transactional flush
- Embedding cache for repeated queries
- Ontology type hierarchy with transitive `is_a` queries

### Component 2: 6-Tier Cache Cascade (`cache/reactor.rs`, `cache_l1.rs`, `cache_router.rs`)

The intelligence layer that routes queries through progressively more expensive tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CACHE TIER CASCADE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query                                                                     │
│     │                                                                       │
│     ▼                                                                       │
│   ┌─────────┐  hit  ┌─────────┐  hit  ┌─────────┐  hit  ┌─────────┐      │
│   │ L1:     │──────▶│ L2:     │──────▶│ L3:     │──────▶│ L4:     │      │
│   │ Memory  │       │ WASM    │       │ Graph   │       │Semantic │      │
│   │ (LRU)   │       │ Tool    │       │ (SQLite)│       │ (KNN)   │      │
│   └─────────┘       └─────────┘       └─────────┘       └─────────┘      │
│                                                       │                   │
│                                                       ▼                   │
│                                                 ┌─────────┐  hit         │
│                                                 │ L4.5:   │─────────▶    │
│                                                 │Decompose│              │
│                                                 │ (local) │              │
│                                                 └─────────┘              │
│                                                       │ miss             │
│                                                       ▼                   │
│                                                 ┌─────────┐              │
│                                                 │ L5:     │              │
│                                                 │Frontier │              │
│                                                 │ (LLM)   │              │
│                                                 └─────────┘              │
│                                                                             │
│   Every non-L1 hit is persisted as a solution node for future queries.     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Tier | Name | Implementation | Latency | When |
|------|------|---------------|---------|------|
| L1 | Memory | LRU cache (10K entries) | <1ms | Hot queries |
| L2 | WASM Workflow | WASM tool matching via `wasm_tools` table | <5ms | Tool-capable queries |
| L3 | Graph | SQLite LIKE + recursive CTE traversal | <10ms | Structural queries |
| L4 | Semantic | Brute-force KNN cosine similarity | <50ms | Semantic queries |
| L4.5 | Decompose | Delegated to Coral Router's decomposer; cache checks for existing decompositions | 200ms | Complex multi-step |
| L5 | Frontier | Delegated to Coral Router's escalation ladder; cache records results | 500ms+ | Novel problems |

**Uniform dispatch** — each tier is an `Arc<dyn Component>` stored in a
`TierRegistry`. The orchestrator never branches on tier type:

```rust
// cache/reactor.rs — route_with_depth
match self.tier_registry.execute(query, depth) {
    Ok(result) => { /* persist + cache */ }
    Err(_) => { /* fall through to L4.5 decomposition */ }
}
```

Each tier's `execute()` receives the prior tier's miss as a signal via
`WorkContext.metadata`, and returns a `RoutingResult` indicating which tier
satisfied the query. Tiers L3–L5 are wrapped in `Instrumented::with_metrics`
before type erasure, providing per-tier latency histograms exposed via
`coral_stats`.

### Component 3: MCP Server (`mcp.rs`)

JSON-RPC 2.0 server implementing the Model Context Protocol for AI agent integration:

| Method | Parameters | Behavior |
|--------|-----------|----------|
| `coral_query` | `{ "name": "..." }` | Node lookup by name |
| `coral_insert` | Full `ContentNode` JSON | Insert a node, return `node_id` |
| `coral_traverse` | `{ "node_id": N, "max_depth": N }` | Graph traversal |

**Transport**: STDIO (line-delimited JSON), max request size 10MB.

### Component 4: WASM Plugin Runtime (`wasm_runtime.rs`)

Dynamic tool execution via Extism WASM SDK, bridged to the fluent-wvr trait system:

```rust
// WasmComponent implements all fluent-wvr traits:
impl WorkUnit for WasmComponent { ... }
impl FieldAccess for WasmComponent { ... }
impl Describable for WasmComponent { ... }
// Automatic Component via blanket impl
```

**Security Model**:
- No filesystem access (unless explicitly granted)
- No network access from sandboxed tools
- Memory-safe execution via Extism
- Host functions: whitelisted only

### Component 5: Context Packing with LOD (`packer.rs`)

Token-budget-aware context packing using Level of Detail, storing and
rendering the 6-tier LOD scheme defined by Coral Router:

| LOD Level | Size | Use Case |
|-----------|------|----------|
| LOD0 | Complete | Authoritative source text — never derived, always preserved |
| LOD1 | Compressed | Lossless-in-substance, no fixed bound |
| LOD2 | ≤ 1000 chars | Short summary |
| LOD3 | ≤ 280 chars | Compact summary |
| LOD4 | ≤ 80 chars | Single line |
| LOD5 | Brief label | Name / identifier, for listings and identification |

**Storage**: LOD tiers are stored as `Vec<String>` on a `ContentNode`.
LOD0 and LOD5 are guaranteed filled at node creation; LOD1–LOD4 are
lazily computed, directly from LOD0 (never chained from a lower tier),
cached on the node thereafter, and computed at most once globally.

**Algorithm** (context packing for prompt assembly, distinct from the
LOD computation policy which is owned by the Router):
1. BFS from focus node up to depth 5
2. Select LOD by effective graph distance (normalized by avg degree)
3. First-Fit Decreasing bin-pack into token budget

### Component 6: Batch Ingestion (`ingest.rs`)

RDF/Turtle/N-Quads ingestion pipeline with transactional flush:

```
File → Lexer → Parser → TripleMapper → PendingNode/PendingEdge
                                              ↓
                                    BatchIngestor (10K batch)
                                              ↓
                                    Transactional flush to SQLite
```

**Features**:
- Turtle and N-Quads format support
- YAGO ontology whitelist filtering
- Auto-discovery of neighbor edges via KNN (distance < 0.3)
- Embedding computation during ingestion

---

## Data Model

### ContentNode (canonical storage type)

The canonical type is `ContentNode` (defined in `guidance-types` as
`fluent_types::ContentNode`), which unifies the durable storage fields
formerly in `ContentNode` with session-scoped metadata from Coral Router's
ledger:

```rust
pub struct ContentNode {
    // ── Core fields (durable, used by coral) ──
    pub id: Option<NodeId>,
    pub name: SmolStr,
    pub source: String,
    pub lod: Vec<String>,              // 6-tier LOD pyramid (LOD0–LOD5)
    pub embedding: Option<Vec<f32>>,
    pub capabilities: Option<Vec<u8>>,
    // ── Session fields (optional, used by router) ──
    pub session_id: Option<String>,
    pub role: Option<String>,
    pub turn_index: Option<u64>,
    pub accepted: Option<bool>,
    pub acceptance_score: Option<f64>,
    pub active_lod: Option<u8>,
    pub parent_id: Option<NodeId>,
    pub step_id: Option<String>,
    pub step_status: Option<StepStatus>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: Option<u64>,
    // ... remaining session fields
}
```

Coral crate uses only the core fields; session fields are `None`. The
SQLite table retains the name `context_nodes` for backward compatibility;
the Rust type is `fluent_types::ContentNode`. This single type is the
storage entity for both the SQLite graph database (coral) and the session
ledger (router), replacing the former `ContentNode` / `SessionNode` split.

### Edge

| Field | Type | Description |
|-------|------|-------------|
| source_node_id | i64 | Source node |
| target_node_id | i64 | Target node |
| edge_type | String | Relationship type |
| weight | f64 | Edge weight |

### Target (DAG Node)

| Field | Type | Description |
|-------|------|-------------|
| name | String | Human-readable name |
| bit_index | usize | Capability bit position |
| depends | BitVec | Required capabilities |
| provides | BitVec | Provided capabilities |
| essential | bool | Must succeed |
| command | String | Shell command |

---

## Integration with Coral Router: local model dispatch

Coral Context's 6-tier cache cascade and Coral Router's classification tree
are complementary, not overlapping. The cache cascade answers "has this exact
query, or a near neighbor, been seen before?" The classification tree answers
"which model is best suited to handle this query?" They compose as follows:

```
Request → Coral Router (classification tree)
                │
                ├─ pre_filters (regex/PII rejection)
                ├─ classifier nodes (LLM picks domain → complexity → target)
                └─ terminal node dispatch
                        │
                        ▼
                Model resolution (cheapest model in group with sufficient intelligence)
                        │
                        ▼
                Coral Context cache check (optional, before model inference)
                        │
                        ├─ L1 hit  → return cached result (< 1ms)
                        ├─ L3 hit  → return graph-traversal result (< 10ms)
                        ├─ L4 hit  → return semantic-match result (< 50ms)
                        └─ miss    → invoke the routed model
                                        │
                                        ▼
                                 Result cached as new ContentNode
```

The division of labor is clear:

- **Coral Router owns the routing decision**: which model, which session
  profile, whether to escalate to frontier or reject.
- **Coral Context owns the cache decision**: whether the query (or a near
  neighbor) has already been answered, and whether the prior answer is
  still valid.

They share `fluent_types::ContentNode` as the canonical type for both durable
storage and session-scoped metadata — the former `ContentNode` / `ContentNode`
split is eliminated.

### Why the cache sits after routing, not before

The classification tree can redirect or reject a query based on policy
(PII, coherence, safety) regardless of whether a cached answer exists.
A cached answer to "how do I make a bomb?" is still a policy rejection.
Routing and caching are independent decisions, and routing must fire first
because it enforces safety and domain gating that caching does not.

The cache check is an optional optimization layer at dispatch time: a
terminal node may reach for the cache before invoking the model, but it
may not skip the classification tree to do so.

### Workflow extraction: frontier-assisted solutions become DAG workflows

When a `model_group` has `post_process.workflow_extraction: true` and a
frontier-assisted solution succeeds (via any escalation mode), Coral Router
decomposes the solution into a reusable DAG workflow stored in Coral Context:

1. **Decomposition**: The full chain — `classifier decision → local model
   attempts → escalation stage → frontier prompt → frontier response →
   local assembly → final answer` — is split into discrete steps. Each
   step that involved a specific model call or deterministic transform
   becomes a `Target` node.

2. **Dependency edges**: `depends` / `provides` edges capture the data-flow
   structure. For example, a `team` mode escalation produces:
   ```
   classifier_vote_1 ─┐
   classifier_vote_2 ─┼─→ draft_attempt ─→ judge_review ─→ frontier_prompt
   classifier_vote_3 ─┘
   ```
   where `classifier_vote_N` nodes are parallel-slot classifier results that
   feed into the draft model, whose output feeds the judge, whose output is
   the frontier prompt.

3. **Storage**: The `Target` DAG is persisted as `ContentNode` entries in
   SQLite (`context_nodes` + `edges` tables), keyed by an embedding of the
   original user query. The extraction process is idempotent — if a
   sufficiently similar query already has a cached workflow, the new
   solution updates or subsumes the existing one rather than creating a
   duplicate.

4. **Replay**: On a future query with a near-neighbor embedding (L4 semantic
   cache hit), the cache reactor can replay the DAG steps deterministically.
   Steps that involved frontier calls are flagged — if the local models have
   improved since the workflow was stored (new local model versions, new
   tools), the system can attempt local re-execution of those steps before
   falling back to the stored frontier response.

This closes the neurosymbolic learning loop described in Goal 3: the one-time
expensive frontier call becomes a permanent, replayable local DAG. The cost
of novel problems trends down over the life of the installation as the DAG
store fills in.

## Integration with guidance

### How Coral Serves guidance

```
guidance explain "query"
        │
        ▼
   QueryEngine.classify()
        │
        ▼
   WordIndex / GuidanceDb hybrid search
        │ (if vector search needed)
        ▼
   coral::db::Library::knn_search()
        │ (if graph traversal needed)
        ▼
   coral::db::Library::traverse_from()
        │ (if context packing needed)
        ▼
   coral::packer::ContextPacker::pack()
```

### Shared Types

| Type | Defined In | Used By |
|------|-----------|---------|
| `ContentNode` | `guidance-types` | coral, router, guidance |
| `NodeId` | `guidance-types` | coral, guidance |
| `KnnHit` | `guidance-types` | coral, guidance |
| `WasmTool` | `guidance-types` | coral |
| `Component` | `fluent-wvr` | coral, dag |
| `WorkUnit` | `fluent-wvr` | coral, dag |

---

## Implementation Status

### Completed

1. **SQLite Graph Database**: Full schema with 7 tables, KNN search, recursive CTE traversal, duck typing
2. **6-Tier Cache Cascade**: L1 (LRU) through L5 (Frontier), with solution persistence
3. **MCP Server**: JSON-RPC 2.0 over STDIO with 3 methods
4. **WASM Plugin Runtime**: Extism integration with fluent-wvr trait bridge
5. **Context Packing**: Token-budget-aware LOD selection with FFD bin-packing
6. **Batch Ingestion**: Turtle/N-Quads parsing with YAGO whitelist filtering
7. **Embedding Support**: Ollama and OpenAI embedding providers with caching
8. **SSRF Protection**: URL validation blocking private IPs and remote HTTP
9. **PII Anonymization**: Regex-based redaction for emails, credit cards, SSN, etc.
10. **Hybrid Search**: Reciprocal Rank Fusion (k=60) for keyword + vector

1. **SOLID Refactoring**: `db.rs` decomposed into `db/` (schema, nodes, edges, hnsw, embeddings, kv_cache);
   `cache_reactor.rs` decomposed into `cache/` (reactor, stats) — Single Responsibility.
4. **DRY Consolidation**: PII regexes centralized in `fluent_llm::pii_patterns`;
   `strip_html` canonical in `common_core::string`;
   think-block stripping canonical in `common_core::string`.
5. **DIP Architecture**: `OrchestratorSession`, `ResultScorer`, `Summarizer`, `ClassifierStage`
   all accept `Arc<dyn ChatBackend>` rather than constructing concrete `LlmClient`.

### In Progress

1. **Async I/O**: Replacing synchronous SQLite calls with async-friendly patterns

### Wired (documented, not separately listed)

1. **Fluent WVR Pattern Adoption**: All 6 cache tiers are `WorkUnit` implementations dispatched uniformly through `TierRegistry` (completed by M3 of `ROADMAP_REFINE.md`)

### Planned

1. **HNSW Index**: The canonical HNSW implementation lives here, backed by
   `common_core::sqlite::make_hnsw()`. Currently wired — `insert_node`
   (`db.rs:152`) calls `hnsw_insert` (`db.rs:167`) on every node with an
   embedding; brute-force KNN remains the primary query path. Upgrade to
   HNSW query when candidate count exceeds 100K. Coral Router's indexed
   collections (prior-workflow library, rubric/validated-answer cache,
   blacklist-similarity index, and per-LOD-tier scene graphs) are scoped
   views over this HNSW layer — they use the same `make_hnsw()` factory
   but maintain separate index instances with distinct scope and error
   tolerances.
2. **Persistent L1 Cache**: Disk-backed LRU for warm starts
3. **Graph Analytics**: PageRank, community detection for node importance

---

## Deployment Profiles

### Edge Profile (Raspberry Pi, Mobile)

- **Memory**: 4-8 GB RAM
- **Storage**: SQLite (WAL mode)
- **Max Nodes**: ~100K ContentNodes
- **Target Latency**: < 50ms deterministic

### Server Profile (Linux x86_64)

- **Memory**: 16-64 GB RAM
- **Storage**: SQLite (WAL mode, high concurrency)
- **Max Nodes**: ~1M+ ContentNodes
- **Target Latency**: < 20ms deterministic

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Deterministic resolution rate | > 40% |
| Query latency (cached) | < 50ms |
| Memory footprint (edge) | < 500MB |
| Binary size (edge) | < 50MB |
| Frontier consultation rate | < 15% |

---

## Conclusion

Coral Context represents a shift in AI architecture: **deterministic execution first, probabilistic inference only when necessary**. By replacing prompt-based LLM reasoning with a cascading cache tier system, the system achieves:

- **Predictable performance**: Sub-100ms latency for known patterns
- **Zero marginal cost**: No API calls for cached solutions
- **Full auditability**: Every decision traceable through the cache tiers
- **Continuous improvement**: Each novel solution becomes a permanent cached node
- **Edge deployment**: Full functionality on Raspberry Pi-class hardware

The result is an AI system that grows more capable with every use, while remaining fast, cheap, and deterministic.

---

*Vision Document v3.0 — June 2026 (Rust codebase)*
