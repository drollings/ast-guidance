# Fluent Monorepo - a high-speed agentic backbone

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

Note:  This project's support for parallel inference is built around a branch of 
llama.cpp that allows parallel context windows and window sizes upon requests via 
HTTP parameters.  You can find that at:

https://github.com/drollings/llama.cpp/tree/_gguf_tool_ctx

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
  knowledge/             WordIndex, TrigramIndex, CsrGraph, QueryCache
  content-node/          LOD slicing and file content annotation
  ontology/              Entity extraction, YAGO taxonomy, capability inference
  rdf/                   Turtle/N-Quads parser and normalization
  wasm_ipc/              WASM IPC binary types (#[repr(C, packed)])
  memory-plugin/         Pluggable persistent memory backends
```

Cross-crate conventions are enforced through `common-core` (the zero-domain crate)
and Fluent WVR patterns that supplement Traits with run-time polymorphism and reflection, 
single sources of truth, and support for run-time IPC with WASM sandboxes.  Where you have
configurable components and control panes, this allows object-oriented and Entity Component 
System behaviors.

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
| Project knowledge | `knowledge` | Word/trigram inverted indexes, CSR graph, frequency tables |

## Design philosophy

1. **Deterministic-first**: Prefer local computation over probabilistic inference;
   LLM enhancement is additive, never authoritative
2. **Cache over compute**: Every novel solution becomes a permanent cached node
3. **Edge-deployable**: Single-process SQLite, no external services, targets
   Raspberry Pi class hardware
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

`fluent-monorepo` is **dual-licensed** under the terms of either:

1. **GNU Lesser General Public License v3.0 or later** (`LGPL-3.0-or-later`), OR
2. **Commercial License**

You may select the license terms that best fit your project's compliance requirements.

---

### Option 1: Open Source Use (LGPLv3)

You are free to use, modify, and distribute this software under the terms of the **GNU Lesser General Public License v3.0** (`LICENSE-LGPLv3`).

* **Internal / Cloud SaaS Use:** You can freely use `fluent-monorepo` inside your organization or behind a network/SaaS boundary without triggering copyleft obligations.
* **Open Source Projects:** You may freely include or link against `fluent-monorepo` in open-source applications.
* **Rust Static Linking Notice:** Because Cargo compiles Rust dependencies directly into static application binaries, distributing a proprietary closed-source application that embeds `fluent-monorepo` under LGPLv3 requires you to either:
  1. Open-source your application under a compatible license, **OR**
  2. Provide object files (`.rlib`/`.o`) or source code sufficient to allow end users to re-link your application against modified versions of `fluent-monorepo` (per LGPLv3 Section 4).

If your project cannot comply with LGPLv3 static-linking requirements or you do not wish to distribute object files for proprietary code, you must obtain a **Commercial License**.

---

### Option 2: Commercial License

The **Commercial License** removes all LGPLv3 copyleft and re-linking obligations, allowing you to freely embed, statically link, and distribute `fluent-monorepo` within closed-source, proprietary products.

A Commercial License is recommended for:
* Closed-source commercial software products distributed to end users.
* Enterprise deployments requiring formal SLA guarantees, dedicated technical support, indemnification, or liability waivers.
* Teams requiring custom contributor agreements or tailored integration support.

---

### Third-Party Dependencies & Acknowledgments

`fluent-monorepo` is built on top of the Rust open-source ecosystem and relies on third-party crates, including:

* **Tokio** runtime and async primitives — licensed under the permissive [MIT License](https://github.com/tokio-rs/tokio/blob/master/LICENSE)
* Additional ecosystem dependencies — licensed under permissive standard licenses (MIT, Apache-2.0, or BSD)

Under the terms of these permissive upstream licenses, you remain fully compliant when linking them alongside `fluent-monorepo`. Complete license notices for all transitive dependencies are included in the source distribution and generated dependency manifests (`cargo-deny` audit reports).
