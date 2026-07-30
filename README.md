# Fluent Monorepo - foundations for Coral Context, Fluent Concurrency

This is a Rust monorepo, built as an integrated incubator of unified projects sharing a
common infrastructure and enforced design patterns that build a dynamic, efficient runtime 
and extreme agentic efficiency.  

Coral Context and Coral Router are meant to make a deterministic-first backbone of agentic
"mixture of agents" orchestration, continuous context management, a plugin-driven system
for agentic memory, and an managed WASM sandbox for a large index of tools.

Its concurrent operations are based on Fluent Concurrency, a lightweight layer of
guardrails over tokio, hyper, and reqwest meant to make async I/O and inference for
agentic LLM applications blazing fast and battle-tested.

Its foundation is the Fluent WVR (WASM, vtables, reflection) set of design patterns,
meant to maximize code reuse, composable primitives, deterministic-first design, and
a uniform source of metadata and sanitized input constraints.  Every crate in `src/`
builds on the same foundation of proven design patterns, capability-gated concurrency,
and type-safe runtime composition.

## Quick start

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings
```

## Projects

- **Coral Context** — Deterministic-first context graph library with a 6-tier
  cache cascade (L1 memory → L5 frontier LLM), SQLite graph database, MCP server,
  and WASM plugin runtime.  Separates deterministic lookups from probabilistic
  inference.  → `doc/coral/VISION.md`

- **Fluent Concurrency** — Structured concurrency primitives: `WorkerPool`,
  `Scope`, `Zone` (supervision + dependency cancellation), `Limiter`, `PriorityQueue`,
  `CreditFlow` backpressure, and `PartitionedRouter`.  Forms the execution fabric for
  all pipeline and session orchestration in the workspace.
  → `doc/skills/fluent-concurrency/SKILL.md`

- **Coral Router** — LLM request router with a 5-stage pipeline (deterministic
  pre-filter → quality gate → planning refinement → guardrail check → router).
  Decomposes and dispatches complex queries through an escalation ladder, exposing
  an OpenAI-compatible HTTP API.  → `doc/router/VISION.md`

- **Guidance** — AST-guided code navigation subagent producing
  `.guidance/src/**/*.json` metadata mirrors and `.guidance.db` SQLite vector
  search databases.  Sub-100ms deterministic queries for AI-assisted development.
  → `doc/guidance/VISION.md`

- **Fluent WVR** — The unifying component model (12 composable patterns):
  `WorkUnit`, `FieldAccess`, `Describable`, `Component` — every orchestratable
  task presents `Arc<dyn Component>`, whether native, WASM, or DB-driven.
  → `doc/skills/fluent-wvr/SKILL.md`

## Library infrastructure

### Workspace crates

```
src/
  bin/
    guidance/            guidance CLI binary (14+ subcommands)
    coral/               coral binary (MCP server + ingest CLI)
    coral-router/        coral-router binary (HTTP API server)
    yamake-coral/        yamake-coral binary
  guidance/              guidance-core: AST parser, sync engine, query engine
  coral/                 coral-context: graph DB, cache cascade, MCP server, WASM runtime
  router/                fluent-router: pipeline orchestration, dispatch, agent runtime
  dag/                   DAG executor: resolver, work_unit, adapter, middleware
  fluent-wvr/            Component, WorkUnit, FieldAccess, Describable traits
  fluent-wvr-macros/     proc macros for FieldAccess derive
  fluent-concurrency/    WorkerPool, Scope, Zone, Limiter, PriorityQueue, CreditFlow
  llm/                   LLM HTTP client + embeddings (Ollama, OpenAI)
  types/                 Shared domain types (ContentNode, NodeId, etc.)
  common-core/           General utilities (hashing, formatting, shell, string ops)
  search-vector/         SQLite hybrid search (vector + keyword + RRF fusion)
  project-knowledge/     WordIndex, TrigramIndex, CsrGraph, QueryCache
  content-node/          LOD slicing and file content annotation
  ontology/              Entity extraction, YAGO taxonomy, capability inference
  rdf/                   Turtle/N-Quads parser and normalization
  wasm_ipc/              WASM IPC binary types (#[repr(C, packed)])
  memory-plugin/         Pluggable persistent memory backends
```

Cross-crate conventions are enforced through `common-core` (the zero-domain crate)
and a `fluent-wvr-testutil` crate for shared test infrastructure.

### Key capabilities

| Capability | Crate | Description |
|-----------|-------|-------------|
| AST indexing | `guidance` | Tree-sitter parsing for Zig, Python, Rust with incremental `match_hash` sync |
| Vector search | `search-vector` | Cosine similarity + keyword + RRF hybrid over SQLite, quantized embeddings |
| Concurrency | `fluent-concurrency` | Bounded pools, structured scopes, capability-gated I/O, credit flow backpressure |
| DAG execution | `dag` | Dependency-driven workflow with adapters, middleware, type inference |
| LLM client | `llm` | Ollama/OpenAI chat + embeddings with context packing and request queueing |
| Context packing | `coral` | Token-budget-aware LOD selection with BFS distance weighting |
| WASM runtime | `coral` + `wasm_ipc` | Extism plugin execution with binary IPC across the sandbox boundary |
| MCP server | `coral` | JSON-RPC 2.0 over STDIO for IDE integration |
| Graph database | `coral` | SQLite graph store with KNN search, recursive CTE traversal, duck typing |
| Ontology | `ontology` | YAGO taxonomy with transitive `is_a` inference for duck-typed capabilities |
| RDF ingestion | `rdf` | Turtle/N-Quads parsing with transactional batch flush |
| Content nodes | `content-node` | 6-level LOD pyramid (full text → keywords) for context window packing |
| Project knowledge | `project-knowledge` | Word/trigram inverted indexes, CSR graph, frequency tables |

## Design philosophy

1. **Deterministic-first**: Prefer local computation over probabilistic inference;
   LLM enhancement is additive, never authoritative
2. **Cache over compute**: Every novel solution becomes a permanent cached node
3. **Edge-deployable**: Single-process SQLite, no external services, targets
   Raspberry Pi class hardware (<50MB binary, <500MB RAM)
4. **Capability-gated I/O**: All file/network/DB access requires explicit
   capability tokens — no ambient authority
5. **Structured concurrency**: Every spawned task belongs to a Scope whose close
   must be awaited; panics are contained within Zones
6. **Uniform interface**: Native Rust, WASM plugins, and DB-driven configs all
   present `Arc<dyn Component>` — the orchestrator never branches on origin

## Design patterns

The codebase implements twelve composable patterns documented in
`doc/skills/fluent-wvr/SKILL.md` — Fluent Builder, Trait-Based Reflection,
Trait Composition (newtype wrappers), Trait Objects, Binary IPC, Scoped
Ownership, Newtype Handles, Unit of Work, Middleware Chain, Component Adapter,
Structured Logging Context, and Runtime Composition.

## Source layout

```
doc/
  skills/               Fluent WVR and Fluent Concurrency skill docs
  guidance/VISION.md    Guidance vision document
  coral/VISION.md       Coral Context vision document
  router/VISION.md      Coral Router vision document
```

## Authorship

Authored by Daniel Rollings, February 2026, based on conceptual transfer from
projects in Python, C++, and Zig, ported to Rust.

## License

Dual-licensed under GNU LGPL v3.0 and a Commercial License.

**GPLv3**: Free for open-source, hobby, and individual use. If you distribute
software including this code, you must open-source it under GPLv3.

**Commercial License** required for:
- Proprietary, closed-source products
- Organizations with gross annual revenue exceeding $1,000,000 USD
- More than one developer seat
- Technical support, indemnification, or liability waivers

See `LICENSE`, `LICENSE-Contributor-Agreement`, and `LICENSE-Commercial-Requirement`.
