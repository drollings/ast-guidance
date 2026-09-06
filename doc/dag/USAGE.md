# fluent-dag — Dependency Graph & DAG-driven Workflows

**Context**: `fluent-dag` (`src/dag/`) provides the canonical dependency-tracking
primitive for the workspace, plus a resolver, work-unit dispatcher, the
type-inference engine, target registry, and capability-based middleware.

**Prime directive**: Never re-implement graph algorithms. Always compose
`DependencyGraph<K>`.

## Core primitive: `DependencyGraph<K>`

**Path**: `src/dag/src/dep_graph.rs`  
**Export**: `fluent_dag::dep_graph::{DependencyGraph, GraphError}`

A generic, cycle-resilient directed graph where `K: Eq + Hash + Clone + Display`.
Every edge is `(dependent ← dependency)` — the node provides an asset, and
other nodes depend on that asset.

### API surface

```rust
impl<K: Eq + Hash + Clone> DependencyGraph<K> {
    pub fn new() -> Self;
    pub fn register(&mut self, node: &K, deps: &[K], provides: &[K])
        -> Result<(), GraphError>
    where K: Debug;
    pub fn contains(&self, node: &K) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn nodes(&self) -> &[K];

    // Graph query — node-specific
    pub fn deps_of(&self, node: &K) -> Option<&[K]>;
    pub fn provides_of(&self, node: &K) -> Option<&[K]>;

    // Readiness — satisfied is the set of assets already provided
    pub fn is_ready(&self, node: &K, satisfied: &HashSet<K>) -> bool;
    pub fn ready_nodes(&self, satisfied: &HashSet<K>) -> Vec<K>;

    // Transitive dependents — DFS with cycle detection
    pub fn dependents_of(&self, node: &K) -> Vec<K>
    where K: Debug;

    // Assets depended on but provided by no registered node
    pub fn unresolved_deps(&self) -> Vec<K>;

    // Ordering
    pub fn topo_sort(&self) -> Result<Vec<K>, GraphError>
    where K: Ord + Debug;
    pub fn topo_sort_from(&self, roots: &[K]) -> Result<Vec<K>, GraphError>
    where K: Ord + Debug;
}
```

### Cycle handling

`dependents_of` uses cycle-resilient DFS: back-edges into the active path
emit a `tracing::warn!` and return the partial result rather than looping
indefinitely. `topo_sort` returns `Err(GraphError::Cycle)` on cycles.

## Production consumers

| Consumer | What it tracks | File |
|----------|---------------|------|
| `SupervisedBatch` (fluent-concurrency) | Task supervision cancellation tree | `src/fluent-concurrency/src/batch.rs` |
| `DependencySession` (fluent-router) | Session step DAG with checkpoint/rewind | `src/router/src/dag_session.rs` |
| `ChartExecutionPlan` (fluent-router) | Compiled chart stage order + ready-set | `src/router/src/charts/execute.rs` |
| `compile_chart_stages` (fluent-router) | Chart target dependency validation/topo order | `src/router/src/charts/compile.rs` |

## `CheckpointedStepGraph<K, S>` — checkpointed step tracking

**Path**: `src/dag/src/checkpointed.rs`
**Export**: `fluent_dag::checkpointed::CheckpointedStepGraph`

**Status: production.** The canonical step-DAG primitive with checkpoint/rewind
bookkeeping, carved out of the router's `DependencySession`. Any session
step graph that needs to record checkpoints and rewind to them MUST compose
this rather than re-implementing `steps`/`completed`/`checkpoints`/`step_order`
bookkeeping by hand.

`CheckpointedStepGraph` composes `DependencyGraph<K>` (it delegates `is_ready`,
`ready_nodes`, `dependents_of`, `topo_sort`) and adds the checkpoint state that
`DependencyGraph` deliberately does not track: the insertion-order step list,
each step's owned `S` state, a single rewind marker, and which steps have
completed.

### API surface

```rust
impl<K: Eq + Hash + Clone + Display, S: Send + Sync + 'static> CheckpointedStepGraph<K, S> {
    pub fn new() -> Self;
    pub fn add_step(&mut self, key: K, deps: &[K], state: S) -> Result<(), GraphError>;  // GraphError::DuplicateStep
    pub fn checkpoint(&mut self, name: K) -> Result<(), GraphError>;   // names a rewind point
    pub fn complete(&mut self, key: &K);                               // marks a step finished
    pub fn is_ready(&self, key: &K) -> bool;
    pub fn ready_steps(&self) -> Vec<K>;                               // canonical ready_nodes
    pub fn cancel_dependents(&self, key: &K) -> Vec<K>;                // dependents_of
    pub fn rewind_to(&mut self, name: &K) -> Result<Vec<K>, GraphError>; // suffix to re-run
    pub fn status(&self, key: &K) -> Option<&S>;                       // owned per-step state
    pub fn state_mut(&mut self, key: &K) -> Option<&mut S>;
    pub fn is_completed(&self, key: &K) -> bool;
    pub fn is_checkpoint(&self, key: &K) -> bool;
    pub fn step_ids(&self) -> &[K];                                    // insertion order
    pub fn step_count(&self) -> usize;
    pub fn completed_count(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn graph(&self) -> &DependencyGraph<K>;                        // delegate for topo_sort etc.
}
```

`rewind_to(name)` returns the steps after the named checkpoint (the suffix to
re-run) and un-completes them; the consumer owns any status resets (e.g.
resetting per-step result state). The router's `DependencySession` composes it
— see `dag_session.rs` — keeping the public session API unchanged.

## When to use

Any new dependency-tracking workflow — session step DAGs, build-target
graphs, pipeline stage ordering, task-supervision cancellation trees —
MUST compose `DependencyGraph<K>` rather than re-implementing graph
algorithms (topo sort, transitive DFS, cycle detection).

```rust
use fluent_dag::dep_graph::DependencyGraph;
use std::collections::HashSet;

let mut graph = DependencyGraph::new();
graph.register(&"parse", &["input.file"], &["parse.output"])?;
graph.register(&"validate", &["parse.output"], &["validate.output"])?;

let satisfied: HashSet<&str> = ["input.file"].into_iter().collect();
assert!(graph.is_ready(&"parse", &satisfied));       // deps satisfied
assert!(!graph.is_ready(&"validate", &satisfied));    // parse.output not yet satisfied

for node in graph.topo_sort()? {
    println!("{node}");
}
```

## Other modules in fluent-dag

| Module | Purpose | Path |
|--------|---------|------|
| `executor` | ~~Sequential target executor~~ **pruned** — run plans under `SupervisedBatch` via the `TargetWorkUnit` bridge instead | (`deleted`; see `target_work_unit`) |
| `resolver` | Resolves abstract dependencies to concrete targets; `ProviderSelection::{All, NarrowOne}` policy | `src/dag/src/resolver.rs` |
| `target_work_unit` | `TargetWorkUnit` — `Target` → `WorkUnit` bridge (`from_target`), runnable under `SupervisedBatch` supervision | `src/dag/src/target_work_unit.rs` |
| `closure` | Shared transitive-closure primitive (DFS over depends edges) — `pub(crate)` | `src/dag/src/closure.rs` |
| `narrowing` | Narrowing pipeline (essential/strict-sat/locality/no-dep) + error constructors — `pub(crate)` | `src/dag/src/narrowing.rs` |
| `work_unit` | CommandUnit — shell command wrapper implementing `Component` | `src/dag/src/work_unit.rs` |
| `target` | Target + TargetRegistry + re-export of `CapabilityRegistry` from `common_core::interner` | `src/dag/src/target.rs` |
| `middleware` | TimingMiddleware + MiddlewareChain (RetryMiddleware deleted — retry composes `common_core::retry`/`SupervisedBatch`) | `src/dag/src/middleware.rs` |
| `type_inference` | Class hierarchy / subtyping for target resolution | `src/dag/src/type_inference.rs` |
| `wvr` | Re-exports `fluent_wvr` core types for DAG integration | `src/dag/src/wvr.rs` |
| `yamake_loader` | JSON loader for yamake-compatible target definitions with self-provision for File/Phony targets | `src/dag/src/yamake_loader.rs` |
| `error` | RegistryError (DuplicateTarget, TargetNotFound, etc.) | `src/dag/src/error.rs` |
| `adapter` | Re-exports `ComponentAdapter` / `ExecuteFn` from `fluent_wvr::wrapper` (moved from this crate to fluent-wvr) | `src/dag/src/adapter.rs` |

## Capability Resolution (for LLM workflow orchestration)

The resolver (`src/dag/src/resolver.rs`) extends classic build-graph semantics
with **provider-selection policies** so the same algorithm handles both
compilation DAGs and LLM workflow capabilities.

### Abstract vs. Concrete Targets

- **Concrete** (`TargetType::File` | `Phony`) — an executable target. It
  **self-provides**: a target named `"spell_check"` automatically provides
  the `"spell_check"` capability, letting name-based deps resolve without
  explicit registration.
- **Abstract** (`TargetType::Abstract`) — a pure capability descriptor with
  no executable body. It does **not** self-provide (doing so would create cycles).
  Abstract targets act as semantic glue between "what I need" and "who provides it."

### Provider Selection Policy

| `ProviderSelection` | Semantics | Use case |
|---|---|---|
| `All` (default) | Include every provider of every capability | Build graphs, data pipelines where all outputs are needed |
| `NarrowOne` | Pick exactly one provider per contested capability | LLM tool selection, agent orchestration, capability graphs |

`NarrowOne` requires a `CapabilityRegistry` (from `common_core::interner`) that
interns capability name strings to stable bit indices. Targets store `depends`
and `provides` as `BitVec` sets for O(1) membership checks.

### Narrowing Pipeline (4 stages)

When multiple targets claim the same capability, `NarrowOne` applies these
stages in order on the candidate set. The pipeline stops as soon as one
remains:

| Priority | Stage | Rule |
|---|---|---|
| 1 | **Essential** | If any candidate has `essential: true`, drop all non-essential |
| 2 | **Strict satisfaction** | Keep only candidates whose **every** dep is in `full_provides` |
| 3 | **Locality** | Keep only candidates with **at least one** dep in `full_provides` (excludes no-dep targets) |
| 4 | **No-dep** | Strict-reduction: replace with only no-dep candidates, but only when that strictly reduces the set |

If all four stages run and more than one remains, the resolver returns
`ResolverError::AmbiguousDependency { name, candidates }` with the full
candidate list for re-prompting or human-in-the-loop resolution.

### Durability Invariant

Once a provider loses narrowing for a capability, it enters a `rejected`
bitset. The final transitive closure expansion skips rejected targets, so a
narrowing loser cannot sneak back in through a different dependency path
that leads to the same capability. Rejection is **capability-scoped** — a
target rejected for capability X can still enter the plan through a
different uncontested capability Y.

### Execution Plan

`resolver::resolve(target_names)` returns `ExecutionPlan { order, target_names }`
— a topologically-sorted list of target bit-indices and their human-readable
names. Run the plan by mapping each target to a `TargetWorkUnit` via
`TargetWorkUnit::from_target` and executing it (sequentially in `plan.order`,
or as a supervision `SupervisedBatch` that runs independent waves concurrently).

### Error Types (from `common_core::error::ResolverError`)

| Error | When |
|---|---|
| `MissingDependency(String)` | A needed capability has no registered provider |
| `AmbiguousDependency { name, candidates }` | Narrowing cannot reduce to one provider |
| `TargetNotFound(String)` | Requested target name not in registry |
| `CircularDependency` | Kahn's sort detects a cycle |

### Performance

Benchmarked at three scales:
- **Linear chain (200 nodes)**: <10ms, ~0.05µs/node
- **Breadth-first fan-out (100 leaves)**: ~1400µs total, ~14µs/node
- **Deep diamond (height=80)**: ~3400µs total, 240 nodes resolved

The fast-path check (no contested capabilities + no abstract seeds) delegates
directly to Kahn's sort with zero narrowing overhead — ~4µs for simple graphs
on both `All` and `NarrowOne` paths. See `src/dag/src/resolver.rs` for
benchmarking tests.
