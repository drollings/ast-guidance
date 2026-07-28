# fluent-dag — Dependency Graph & DAG-driven Workflows

**Context**: `fluent-dag` (`src/dag/`) provides the canonical dependency-tracking
primitive for the workspace, plus an executor, resolver, work-unit dispatcher,
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
impl<K> DependencyGraph<K> {
    pub fn new() -> Self;
    pub fn register(&mut self, deps: &[K], provides: &[K]) -> Result<(), GraphError>;
    pub fn contains(&self, node: &K) -> bool;
    pub fn nodes(&self) -> Vec<K>;

    // Query
    pub fn is_ready(&self, node: &K) -> bool;
    pub fn ready_nodes(&self) -> Vec<K>;
    pub fn dependents_of(&self, node: &K) -> Vec<K>;  // transitive DFS, cycle-resilient
    pub fn deps_of(&self, node: &K) -> Vec<K>;          // direct dependencies
    pub fn provides_of(&self, node: &K) -> Vec<K>;      // direct provides
    pub fn unresolved_deps(&self, node: &K) -> Vec<K>;  // unsatisfied deps

    // Ordering
    pub fn topo_sort(&self) -> Result<Vec<K>, GraphError>;
    pub fn topo_sort_from(&self, roots: &[K]) -> Result<Vec<K>, GraphError>;
}
```

### Cycle handling

`dependents_of` uses cycle-resilient DFS: back-edges into the active path
emit a `tracing::warn!` and return the partial result rather than looping
indefinitely. `topo_sort` returns `Err(GraphError::Cycle)` on cycles.

## Production consumers

| Consumer | What it tracks | File |
|----------|---------------|------|
| `Zone` (fluent-concurrency) | Task supervision cancellation tree | `src/fluent-concurrency/src/zone.rs` |
| `DependencySession` (fluent-router) | Session step DAG with checkpoint/rewind | `src/router/src/dag_session.rs` |

## When to use

Any new dependency-tracking workflow — session step DAGs, build-target
graphs, pipeline stage ordering, task-supervision cancellation trees —
MUST compose `DependencyGraph<K>` rather than re-implementing graph
algorithms (topo sort, transitive DFS, cycle detection).

```rust
use fluent_dag::dep_graph::DependencyGraph;

let mut graph = DependencyGraph::new();
graph.register(&["input.file"], &["parse.output"])?;
graph.register(&["parse.output"], &["validate.output"])?;

assert!(graph.is_ready(&"parse.output"));    // no transitive deps
assert!(!graph.is_ready(&"validate.output")); // depends on parse.output

for node in graph.topo_sort()? {
    println!("{node}");
}
```

## Other modules in fluent-dag

| Module | Purpose | Path |
|--------|---------|------|
| `executor` | Executes a sequence of targets through the graph | `src/dag/src/executor.rs` |
| `resolver` | Resolves abstract dependencies to concrete targets; `ProviderSelection::{All, NarrowOne}` policy | `src/dag/src/resolver.rs` |
| `closure` | Shared transitive-closure primitive (DFS over depends edges) — `pub(crate)` | `src/dag/src/closure.rs` |
| `narrowing` | Narrowing rules + canonical error constructors — `pub(crate)` | `src/dag/src/narrowing.rs` |
| `work_unit` | CommandUnit — shell command wrapper implementing `Component` | `src/dag/src/work_unit.rs` |
| `target` | Target registry — register and retrieve typed build targets | `src/dag/src/target.rs` |
| `middleware` | TimingMiddleware, RetryMiddleware, MiddlewareChain | `src/dag/src/middleware.rs` |
| `type_inference` | Class hierarchy / subtyping for target resolution | `src/dag/src/type_inference.rs` |
| `drift` | UnitDrift — bit-set based version tracking | `src/dag/src/drift.rs` |
| `capability` | CapabilityRegistry — what each target can do | `src/dag/src/capability.rs` |
| `adapter` | Adapter middleware for runtime adaptation | `src/dag/src/adapter.rs` |
