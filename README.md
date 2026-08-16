# Fluent Monorepo - deterministic-first agentic backbone

This Rust monorepo is an incubator of projects meant to use and
preserve uncompromisingly fast and flexible design patterns among its
members.  Every crate in `src/` is built on the same foundation - the same
design patterns, the same tested guardrails over asynchronous concurrency,
the same dependency charts using Directed Acyclic Graphs, all written to be
highly reusable, maximizing the chance that new work is composed from
extant, battle-tested code instead of reinventing it.

It is written to seamlessly allow polymorphism to a degree that C++ and Python
programmers take for granted, keep guardrails on fast WASM IPC, and sanitize
input, because it is built from the start to have single sources of truth
for its types and their metadata, which can be reflected at runtime.

The flagship project is **Coral Router**: a deterministic-first LLM request
router and process owner of a local inference fleet. Coral Context, Guidance,
and the supporting libraries extend the same backbone.

Its concurrent operations are built on **Fluent Concurrency**, a lightweight
layer of guardrails over Tokio, hyper, and reqwest that keeps async I/O and
inference for agentic LLM applications fast, guardrailed, and rooted in the
battle-tested Rust ecosystem.

Its foundation is the **Fluent WVR** set of design patterns:

`Fluent, Wrapped, Verified, Reflected`

These patterns allow for single sources of truth for of metadata and validated
input constraints.  In turn, this enhances composable, reliable code, made
polymorphic and flexible at control pane-facing objects where it counts,
with zero cost to the hot paths.

## Coral Router - an agentic swarm orchestrator

Coral Router exposes an OpenAI-compatible HTTP API on `:8079` and runs
route-dispatched requests through a **two-stage pipeline** - a deterministic
pre-filter, then a classifier (see `src/router/src/pipeline_types.rs`) - that
resolves to a direct response, a routing target, or a rejection. Requests that
address a managed model directly (`model:instance`) dispatch to that model
without running the pipeline.

It is built for use with a branch of llama.cpp that allows
parallel inference across context windows with their own sizes, parameters,
and lifetimes per request via HTTP parameters:

https://github.com/drollings/llama.cpp/tree/_multi_context

It is also the **process owner of the local inference fleet**: it spawns and
supervises one `llama-server` per model weights file, serves the `/instances`
management contract, and is the single routing element between those local
tasks and every other OpenAI-compatible endpoint. A dispatch to a local model
is a direct call to the owning server; a frontier or remote call is the same
request routed onward after the local ladder has genuinely failed to resolve
it - never by default.

- **Deterministic before probabilistic.** Anything decidable by a regex or a
  fixed rule is handled without a model call wherever the pipeline runs - a
  cost floor and a fully unit-testable layer with no model in the loop.
- **Cheap before expensive.** Routing is an economic decision: the ladder runs
  deterministic filter → fast classifier → local model → frontier, and a
  request only reaches a rung after the previous one failed.
- **Condensed context, not accumulated context.** The ledger compacts sessions
  rather than growing them; the orchestrator never reasons over noise, dead
  ends, or superseded exploration.

→ `doc/router/VISION.md` · source in `src/router/`

## Fluent WVR - the design patterns

Fluent WVR (`Fluent, Wrapped, Verified, Reflected`) gives this codebase a
consistent control plane, reflection layer, and baseline for data validation.
It does this with a collection of interlocking design patterns for consistent
metadata on composable units of work, polymorphism where needed, schemas for
datatypes internally and over IPC, and other single sources of truth.

Every orchestratable task presents the same `Arc<dyn Component>` interface -
whether it is a native Rust struct, a WASM plugin, or a database-driven
config - so the orchestrator iterates uniform handles and never branches on
origin. Twelve composable patterns are documented in
`doc/skills/fluent-wvr/SKILL.md` (Fluent Builder, Trait-Based Reflection,
Trait Composition, Trait Objects, Binary IPC, Scoped Ownership, Newtype
Handles, Unit of Work, Middleware Chain, Component Adapter, Structured
Logging Context, Runtime Composition). These patterns are for the control
plane, not for hot-path inner loops: the data plane uses concrete types and
flat enums, and `dyn` lives at the request boundary, not in the tight loop.

## Fluent Concurrency - the execution fabric

`fluent-concurrency` is a thin, 100% safe extension layer over Tokio:
bounded worker pools, structured `Scope`s, the `SupervisedBatch`
(supervision + dependency cancellation + panic/fail/cancel tracking),
`Limiter`, `PriorityQueue`, `CreditFlow`, and the `first_accept_in_order`
ladder.  Tokio is the workhorse - the crate composes its primitives rather
than rebuilding the scheduler.

Tasks spawned inside a `Scope` are awaited when the scope closes, and
server-owned background/connection tasks are awaited when the server drains
them at graceful shutdown. Capability tokens gate DB and knowledge access on
the serving path - operator CLI tooling is capability-exempt by design.

→ `doc/skills/fluent-concurrency/SKILL.md`

## DAG - the dependency fabric

`fluent-dag` provides the `DependencyGraph` and `CheckpointedStepGraph`
primitives that drive the chart executor, session orchestration, and workflow
execution: dependency validation, ready-node selection, dependency-aware
cancellation, and checkpoint/rewind - shared by every graph consumer in the
workspace rather than re-implemented per crate. → `doc/skills/dag/SKILL.md`

## Safe Rust

The workspace is near-total safe Rust: `fluent-wvr`, `fluent-concurrency`,
`fluent-dag`, and `common-core` are `#![forbid(unsafe_code)]`, and the entire
monorepo contains just three `unsafe` blocks, all of them `read_unaligned` for
packed WASM IPC structs in `wasm_ipc`.

## Quick start

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

Coral Router's own loop: `make router` (build), `make router-start` (build and
restart on `:8079`), `make router-test` (tests), `make router-mock`
(config-synced routing integration tests, derived from `env/coral-router.json`).

## Projects

- **Coral Router** - LLM request router; two-stage deterministic-first
  pipeline, escalation ladder, OpenAI-compatible API, owns and supervises the
  local llama-server fleet. → `doc/router/VISION.md`
- **Coral Context** - deterministic-first context graph library: 6-tier LOD
  pyramid, SQLite graph database, MCP server, WASM plugin runtime. Separates
  deterministic lookups from probabilistic inference. → `doc/coral/VISION.md`
- **Guidance** - AST-guided code navigation subagent producing metadata
  mirrors and SQLite vector search databases; sub-100ms deterministic queries
  for AI-assisted development. → `doc/guidance/VISION.md`
- **Fluent WVR** - the unifying component model. → `doc/skills/fluent-wvr/SKILL.md`
- **Fluent Concurrency** - structured concurrency primitives. → `doc/skills/fluent-concurrency/SKILL.md`
- **Fluent DAG** - dependency graph and checkpointed step graph. → `doc/skills/dag/SKILL.md`

## Design philosophy

1. **Deterministic-first**: prefer local computation over probabilistic
   inference; LLM enhancement is additive, never authoritative
2. **Cache over compute**: every novel solution becomes a permanent cached node
3. **Edge-deployable**: single-process SQLite, no external services, targets
   Raspberry Pi class hardware
4. **Capability-gated I/O**: DB and knowledge effects require explicit
   capability tokens; operator CLI tooling is capability-exempt by design
5. **Structured concurrency**: `Scope`-spawned tasks are awaited when the
   scope closes and server-owned background tasks are awaited at graceful
   shutdown; panics are contained within `SupervisedBatch`
6. **Uniform interface**: native Rust, WASM plugins, and DB-driven configs all
   present `Arc<dyn Component>` - the orchestrator never branches on origin
7. **Safe by default**: the shared foundation crates carry
   `#![forbid(unsafe_code)]`; the only `unsafe` in the workspace is boundary
   IPC `read_unaligned`

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

* **Tokio** runtime and async primitives - licensed under the permissive [MIT License](https://github.com/tokio-rs/tokio/blob/master/LICENSE)
* Additional ecosystem dependencies - licensed under permissive standard licenses (MIT, Apache-2.0, or BSD)

Under the terms of these permissive upstream licenses, you remain fully compliant when linking them alongside `fluent-monorepo`. Complete license notices for all transitive dependencies are included in the source distribution and generated dependency manifests (`cargo-deny` audit reports).
