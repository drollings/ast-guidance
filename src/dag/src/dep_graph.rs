//! Pure dependency-graph algorithms shared by `Zone`, `DependencySession`,
//! and `DependencyResolver`. No side effects, no execution, no state
//! machine — callers apply the effect (abort a task, mark a step
//! cancelled, schedule a build target).
//!
//! # Design
//!
//! The graph tracks four indices over the same set of registered nodes:
//!
//! - `deps`: node → assets it depends on
//! - `provides`: node → assets it provides
//! - `provides_to_dependents`: asset → nodes that depend on it (inverted
//!   index for O(1) lookup in `dependents_of`)
//! - `asset_to_providers`: asset → nodes that provide it (inverted index
//!   for `topo_sort` edge construction)
//!
//! All methods are side-effect-free. `dependents_of` returns the
//! transitive dependent set; the caller decides what to do with it
//! (abort a task, mark a step cancelled, etc.).

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Reserved for a future strict-query mode (e.g. a `register` or
    /// query method that rejects unregistered nodes). No method in this
    /// file currently returns it — query methods return empty/`false`/
    /// `None` for unregistered nodes by design (see
    /// `test_is_ready_unregistered_node`).
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("dependency cycle detected involving: {0}")]
    CircularDependency(String),
    #[error("duplicate node: {0}")]
    DuplicateNode(String),
}

/// A pure dependency graph. Tracks which nodes depend on which assets,
/// which nodes provide which assets, and an inverted index from assets
/// to the nodes that depend on them.
///
/// Generic over the node/asset key type `K`. All methods are
/// side-effect-free — `dependents_of` returns the transitive dependent
/// set, and the caller decides what to do with it (abort a task, mark
/// a step cancelled, etc.).
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use fluent_dag::dep_graph::DependencyGraph;
///
/// let mut g: DependencyGraph<&str> = DependencyGraph::new();
/// g.register(&"compile", &[], &["object"]).unwrap();
/// g.register(&"link", &["object"], &["binary"]).unwrap();
///
/// // Who depends on "compile" (transitively)?
/// let deps = g.dependents_of(&"compile");
/// assert!(deps.contains(&"link"));
///
/// // Is "link" ready given that "object" is satisfied?
/// let satisfied: HashSet<&str> = ["object"].into_iter().collect();
/// assert!(g.is_ready(&"link", &satisfied));
///
/// // Topological order
/// let order = g.topo_sort().unwrap();
/// assert_eq!(order[0], "compile");
/// ```
pub struct DependencyGraph<K: Eq + Hash + Clone> {
    /// All registered nodes, in insertion order.
    nodes: Vec<K>,
    /// node → assets it depends on.
    deps: HashMap<K, Vec<K>>,
    /// node → assets it provides.
    provides: HashMap<K, Vec<K>>,
    /// Inverted index: asset → nodes that depend on it.
    /// Built during `register` for O(1) lookup in `dependents_of`.
    provides_to_dependents: HashMap<K, Vec<K>>,
    /// Inverted index: asset → nodes that provide it.
    /// Built during `register` for edge construction in `topo_sort`.
    asset_to_providers: HashMap<K, Vec<K>>,
}

impl<K: Eq + Hash + Clone> DependencyGraph<K> {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            deps: HashMap::new(),
            provides: HashMap::new(),
            provides_to_dependents: HashMap::new(),
            asset_to_providers: HashMap::new(),
        }
    }

    /// Register a node with its dependencies and provided assets.
    /// Builds the inverted `provides_to_dependents` and
    /// `asset_to_providers` indices.
    ///
    /// Returns `Err(GraphError::DuplicateNode)` if `node` is already
    /// registered.
    pub fn register(&mut self, node: &K, deps: &[K], provides: &[K]) -> Result<(), GraphError>
    where
        K: std::fmt::Debug,
    {
        if self.deps.contains_key(node) {
            return Err(GraphError::DuplicateNode(format!("{node:?}")));
        }
        self.nodes.push(node.clone());
        self.deps.insert(node.clone(), deps.to_vec());
        self.provides.insert(node.clone(), provides.to_vec());

        for dep in deps {
            self.provides_to_dependents
                .entry(dep.clone())
                .or_default()
                .push(node.clone());
        }
        for prov in provides {
            self.asset_to_providers
                .entry(prov.clone())
                .or_default()
                .push(node.clone());
        }

        Ok(())
    }

    /// Returns `true` if `node` has been registered.
    pub fn contains(&self, node: &K) -> bool {
        self.deps.contains_key(node)
    }

    /// Number of registered nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes are registered.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// All registered nodes, in insertion order.
    pub fn nodes(&self) -> &[K] {
        &self.nodes
    }

    /// Assets that `node` depends on. Empty if `node` has no deps.
    /// Returns `None` if `node` is not registered.
    pub fn deps_of(&self, node: &K) -> Option<&[K]> {
        self.deps.get(node).map(Vec::as_slice)
    }

    /// Assets that `node` provides. Empty if `node` provides nothing.
    /// Returns `None` if `node` is not registered.
    pub fn provides_of(&self, node: &K) -> Option<&[K]> {
        self.provides.get(node).map(Vec::as_slice)
    }

    /// Assets depended on by some registered node but provided by none.
    ///
    /// These are unsatisfiable dependencies — a node depending on an
    /// unresolved asset will never become ready (see [`is_ready`](Self::is_ready)),
    /// and [`topo_sort`](Self::topo_sort) will silently skip the edge.
    /// Returns assets in the order they were first registered as a
    /// dependency, deduplicated.
    pub fn unresolved_deps(&self) -> Vec<K> {
        let mut seen: HashSet<&K> = HashSet::new();
        let mut out: Vec<K> = Vec::new();
        for node in &self.nodes {
            if let Some(deps) = self.deps.get(node) {
                for dep in deps {
                    if !self.asset_to_providers.contains_key(dep) && seen.insert(dep) {
                        out.push(dep.clone());
                    }
                }
            }
        }
        out
    }

    /// Transitive set of nodes that depend on `node` (directly or
    /// indirectly), via the inverted `provides_to_dependents` index.
    ///
    /// Performs a DFS with cycle detection. Three sets are used:
    /// - `visited`: fully-expanded nodes (prevents re-expansion).
    /// - `active_path`: nodes on the current DFS ancestor stack. A
    ///   back-edge into this set indicates a **real** cycle — the node
    ///   transitively depends on itself. This is checked at expansion
    ///   time, so a node that provides multiple assets consumed by the
    ///   same downstream node does **not** trigger a false positive.
    /// - `enqueued`: nodes already pushed onto the stack (prevents
    ///   duplicate result/stack entries when a node is reachable via
    ///   multiple provided assets).
    ///
    /// A back-edge into `active_path` emits a `tracing::warn!`
    /// rather than panicking — the cycle is left in place but the
    /// offending dependents are not double-returned.
    ///
    /// **No side effect.** The caller decides what to do with the
    /// returned nodes (abort a task, mark a step cancelled, etc.).
    ///
    /// Algorithm ported verbatim from the original `Zone::cancel_dependents_of`
    /// (now retired) at `src/fluent-concurrency/src/zone.rs`.
    pub fn dependents_of(&self, node: &K) -> Vec<K>
    where
        K: std::fmt::Debug,
    {
        let mut result: Vec<K> = Vec::new();
        let mut visited: HashSet<K> = HashSet::new();
        let mut active_path: HashSet<K> = HashSet::new();
        let mut enqueued: HashSet<K> = HashSet::new();
        let mut stack: Vec<(K, bool)> = vec![(node.clone(), false)];
        enqueued.insert(node.clone());

        while let Some((current, expanded)) = stack.last_mut() {
            if *expanded {
                active_path.remove(current);
                stack.pop();
                continue;
            }
            *expanded = true;

            if !visited.insert(current.clone()) {
                stack.pop();
                continue;
            }
            active_path.insert(current.clone());

            // Look up what this node provides, then which nodes depend on
            // each provided asset via the inverted index.
            if let Some(provides) = self.provides.get(current) {
                for provided in provides {
                    if let Some(dependents) = self.provides_to_dependents.get(provided) {
                        for dep_name in dependents {
                            if active_path.contains(dep_name) {
                                tracing::warn!(
                                    "Dependency cycle detected: '{:?}' transitively \
                                     depends on itself",
                                    dep_name,
                                );
                                continue;
                            }
                            if !enqueued.insert(dep_name.clone()) {
                                continue;
                            }
                            result.push(dep_name.clone());
                            stack.push((dep_name.clone(), false));
                        }
                    }
                }
            }
        }

        result
    }

    /// Nodes whose dependencies are all in `satisfied`.
    ///
    /// A node is "ready" when every asset in its `deps` appears in
    /// `satisfied`. The node itself need not be in `satisfied`.
    /// Returns nodes in insertion order (deterministic).
    pub fn ready_nodes(&self, satisfied: &HashSet<K>) -> Vec<K> {
        self.nodes
            .iter()
            .filter(|node| self.is_ready(node, satisfied))
            .cloned()
            .collect()
    }

    /// `true` if every asset in `node`'s `deps` is in `satisfied`.
    /// Returns `false` if `node` is not registered.
    pub fn is_ready(&self, node: &K, satisfied: &HashSet<K>) -> bool {
        match self.deps.get(node) {
            Some(deps) => deps.iter().all(|d| satisfied.contains(d)),
            None => false,
        }
    }

    /// Kahn's topological sort of all registered nodes.
    ///
    /// Returns nodes in dependency order (a node appears after all
    /// nodes it depends on). Returns `Err(GraphError::CircularDependency)`
    /// if a cycle is detected.
    ///
    /// Edges are derived as `provider → consumer`: for each node N,
    /// for each asset D in N's deps, every node P that provides D gets
    /// an edge P → N (P must come before N). Self-loops (a node that
    /// provides an asset it also depends on) are skipped to avoid
    /// deadlocks.
    ///
    /// Output is deterministic: the Kahn queue is sorted on each
    /// insertion so that ties break by `Ord` order.
    pub fn topo_sort(&self) -> Result<Vec<K>, GraphError>
    where
        K: Ord + std::fmt::Debug,
    {
        self.topo_sort_inner(&self.nodes.iter().collect::<Vec<_>>())
    }

    /// Kahn's topological sort of the subgraph reachable from `roots`.
    ///
    /// Only includes nodes transitively needed by `roots` (i.e. the
    /// transitive closure of `roots`' dependencies). This is what
    /// `DependencyResolver::resolve(&["build"])` needs — it wants the
    /// order for just the build subgraph, not every registered target.
    pub fn topo_sort_from(&self, roots: &[K]) -> Result<Vec<K>, GraphError>
    where
        K: Ord + std::fmt::Debug,
    {
        if roots.is_empty() {
            return Ok(Vec::new());
        }

        // Compute transitive closure: all nodes transitively needed by
        // `roots`. Walk from each root through its deps, resolving each
        // dep asset to its providers via `asset_to_providers`.
        let mut closure: HashSet<K> = HashSet::new();
        let mut to_visit: Vec<K> = roots.to_vec();
        while let Some(node) = to_visit.pop() {
            if !closure.insert(node.clone()) {
                continue;
            }
            if let Some(deps) = self.deps.get(&node) {
                for dep_asset in deps {
                    if let Some(providers) = self.asset_to_providers.get(dep_asset) {
                        for provider in providers {
                            if !closure.contains(provider) {
                                to_visit.push(provider.clone());
                            }
                        }
                    }
                }
            }
        }

        // Run Kahn on just the closure subset, preserving insertion order
        // from `self.nodes` for determinism.
        let subset: Vec<&K> = self.nodes.iter().filter(|n| closure.contains(n)).collect();
        self.topo_sort_inner(&subset)
    }

    /// Kahn's topological sort of the subgraph over exactly `subset`
    /// (references into `self.nodes`). Shared edge derivation for
    /// `topo_sort` and `topo_sort_from`; the Kahn loop itself is
    /// delegated to the crate-wide [`kahn_sort`] core (shared with
    /// `DependencyResolver::plan_from_set`).
    fn topo_sort_inner(&self, subset: &[&K]) -> Result<Vec<K>, GraphError>
    where
        K: Ord + std::fmt::Debug,
    {
        let subset_set: HashSet<&K> = subset.iter().copied().collect();

        // Build in-degree map and adjacency list from the graph.
        // Edge P → N means "P must come before N" (N depends on an
        // asset that P provides).
        let mut in_degree: HashMap<&K, usize> = subset.iter().map(|&n| (n, 0)).collect();
        let mut adj: HashMap<&K, Vec<&K>> = HashMap::new();

        for &node in subset {
            if let Some(deps) = self.deps.get(node) {
                for dep_asset in deps {
                    if let Some(providers) = self.asset_to_providers.get(dep_asset) {
                        for provider in providers {
                            // Skip self-loops and providers outside the subset.
                            if provider == node || !subset_set.contains(provider) {
                                continue;
                            }
                            adj.entry(provider).or_default().push(node);
                            *in_degree.get_mut(node).unwrap() += 1;
                        }
                    }
                }
            }
        }

        let order = match kahn_sort(&mut in_degree, &adj, subset.len()) {
            Ok(order) => order.into_iter().cloned().collect(),
            Err(partial) => {
                let ordered_set: HashSet<&K> = partial.iter().copied().collect();
                let cycle_nodes: Vec<String> = subset
                    .iter()
                    .filter(|n| !ordered_set.contains(*n))
                    .map(|n| format!("{n:?}"))
                    .collect();
                return Err(GraphError::CircularDependency(cycle_nodes.join(", ")));
            }
        };

        Ok(order)
    }
}

/// The Kahn's-algorithm core shared by `DependencyGraph::topo_sort_inner`
/// and `DependencyResolver::plan_from_set`. Both callers derive edges
/// differently (graph `asset_to_providers` vs. registry `get_providers`),
/// but once `in_degree` and `adj` are built the ordering loop is
/// identical.
///
/// `in_degree` is consumed (mutated in place); `total` is the number of
/// nodes that must appear in the output. Returns `Ok(order)` on success or
/// `Err(partial)` when a cycle leaves some node unvisited — `partial` is
/// the deterministic order computed so far, so callers can identify the
/// cycle nodes (all nodes not present in `partial`).
pub(crate) fn kahn_sort<K: Ord + Clone + Hash>(
    in_degree: &mut HashMap<K, usize>,
    adj: &HashMap<K, Vec<K>>,
    total: usize,
) -> Result<Vec<K>, Vec<K>> {
    // Initial queue: all nodes with in_degree 0, sorted for
    // deterministic output.
    let mut queue: Vec<K> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(k, _)| k.clone())
        .collect();
    queue.sort_unstable();

    let mut order = Vec::with_capacity(total);
    let mut head = 0;
    while head < queue.len() {
        let current = queue[head].clone();
        head += 1;
        order.push(current.clone());
        if let Some(dependents) = adj.get(&current) {
            for dep in dependents {
                if let Some(deg) = in_degree.get_mut(dep) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(dep.clone());
                        queue[head..].sort_unstable();
                    }
                }
            }
        }
    }

    if order.len() != total {
        return Err(order);
    }

    Ok(order)
}

impl<K: Eq + Hash + Clone> Default for DependencyGraph<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_contains() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &["y"]).unwrap();
        g.register(&"c", &[], &[]).unwrap();

        assert!(g.contains(&"a"));
        assert!(g.contains(&"b"));
        assert!(g.contains(&"c"));
        assert!(!g.contains(&"z"));
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_duplicate_register_errors() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &[]).unwrap();
        let err = g.register(&"a", &[], &[]).unwrap_err();
        assert!(matches!(err, GraphError::DuplicateNode(_)));
    }

    #[test]
    fn test_deps_and_provides_of() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"compile", &["header"], &["object"]).unwrap();

        assert_eq!(g.deps_of(&"compile"), Some(&["header"][..]));
        assert_eq!(g.provides_of(&"compile"), Some(&["object"][..]));
        assert_eq!(g.deps_of(&"missing"), None);
        assert_eq!(g.provides_of(&"missing"), None);

        // Node with no deps/provides
        g.register(&"noop", &[], &[]).unwrap();
        assert_eq!(g.deps_of(&"noop"), Some(&[][..]));
        assert_eq!(g.provides_of(&"noop"), Some(&[][..]));
    }

    #[test]
    fn test_dependents_of_linear_chain() {
        // A provides "x", B depends on "x" and provides "y", C depends on "y"
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &["y"]).unwrap();
        g.register(&"c", &["y"], &[]).unwrap();

        let deps = g.dependents_of(&"a");
        assert!(deps.contains(&"b"));
        assert!(deps.contains(&"c"));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_dependents_of_diamond() {
        // A → B, A → C, B → D, C → D
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &["y"]).unwrap();
        g.register(&"c", &["x"], &["z"]).unwrap();
        g.register(&"d", &["y", "z"], &[]).unwrap();

        let deps = g.dependents_of(&"a");
        assert!(deps.contains(&"b"));
        assert!(deps.contains(&"c"));
        assert!(deps.contains(&"d"));
        assert_eq!(deps.len(), 3);
    }

    #[test]
    fn test_dependents_of_no_dependents() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &[], &[]).unwrap();

        let deps = g.dependents_of(&"a");
        assert!(deps.is_empty());
    }

    #[test]
    fn test_dependents_of_cycle_logs_warn_returns_partial() {
        // A ↔ B cycle: A provides "a_provides", depends on "b_provides";
        // B provides "b_provides", depends on "a_provides".
        // dependents_of("A") should not panic or hang; it should return
        // [B] (the direct dependent) and emit a warn for the back-edge.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &["b_provides"], &["a_provides"]).unwrap();
        g.register(&"b", &["a_provides"], &["b_provides"]).unwrap();

        let deps = g.dependents_of(&"a");
        assert!(deps.contains(&"b"));
        // The back-edge to "a" is detected and warned, not returned.
        assert!(!deps.contains(&"a"));
    }

    #[test]
    fn test_dependents_of_multi_asset_no_false_cycle() {
        // Regression: a single provider node provides two assets that the
        // same downstream node depends on. This is NOT a cycle — the
        // dependent should appear exactly once, and no spurious cycle
        // warning should fire.
        //
        // Before the fix, `active_path` was updated at push time, so the
        // second edge to "link" found it already in `active_path` and
        // logged a false "dependency cycle detected" warning.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"compile", &[], &["object", "debug_symbols"])
            .unwrap();
        g.register(&"link", &["object", "debug_symbols"], &[])
            .unwrap();

        let deps = g.dependents_of(&"compile");
        assert_eq!(
            deps.len(),
            1,
            "link should appear exactly once, got: {:?}",
            deps
        );
        assert!(deps.contains(&"link"));
    }

    #[test]
    fn test_unresolved_deps_finds_missing_asset() {
        // "debug_symbols" is depended on by "link" but provided by no
        // registered node — it's an unsatisfiable dependency.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"compile", &[], &["object"]).unwrap();
        g.register(&"link", &["object", "debug_symbols"], &[])
            .unwrap();

        let unresolved = g.unresolved_deps();
        assert_eq!(unresolved, vec!["debug_symbols"]);
    }

    #[test]
    fn test_ready_nodes_all_satisfied() {
        // A provides "x", B depends on "x". satisfied = {"x"}.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &[]).unwrap();

        let satisfied: HashSet<&str> = ["x"].into_iter().collect();
        let ready = g.ready_nodes(&satisfied);
        // A has no deps → always ready. B's deps satisfied → ready.
        assert!(ready.contains(&"a"));
        assert!(ready.contains(&"b"));
    }

    #[test]
    fn test_ready_nodes_none_satisfied() {
        // B depends on "x", no one provides "x" yet. satisfied = {}.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &[]).unwrap();
        g.register(&"b", &["x"], &[]).unwrap();

        let satisfied: HashSet<&str> = HashSet::new();
        let ready = g.ready_nodes(&satisfied);
        // Only A (no deps) is ready. B's dep "x" is not satisfied.
        assert!(ready.contains(&"a"));
        assert!(!ready.contains(&"b"));
    }

    #[test]
    fn test_is_ready_partial() {
        // B depends on ["x", "y"]. satisfied = {"x"}.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"b", &["x", "y"], &[]).unwrap();

        let partial: HashSet<&str> = ["x"].into_iter().collect();
        assert!(!g.is_ready(&"b", &partial));

        let full: HashSet<&str> = ["x", "y"].into_iter().collect();
        assert!(g.is_ready(&"b", &full));
    }

    #[test]
    fn test_is_ready_unregistered_node() {
        let g: DependencyGraph<&str> = DependencyGraph::new();
        let satisfied: HashSet<&str> = HashSet::new();
        assert!(!g.is_ready(&"missing", &satisfied));
    }

    #[test]
    fn test_topo_sort_linear() {
        // A → B → C
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &["y"]).unwrap();
        g.register(&"c", &["y"], &[]).unwrap();

        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], "a");
        assert_eq!(order[1], "b");
        assert_eq!(order[2], "c");
    }

    #[test]
    fn test_topo_sort_diamond() {
        // A → B, A → C, B → D, C → D
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();
        g.register(&"b", &["x"], &["y"]).unwrap();
        g.register(&"c", &["x"], &["z"]).unwrap();
        g.register(&"d", &["y", "z"], &[]).unwrap();

        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], "a");
        assert_eq!(order[3], "d");
        // B and C are in positions 1 and 2 (in some order).
        assert!(order.contains(&"b"));
        assert!(order.contains(&"c"));
    }

    #[test]
    fn test_topo_sort_cycle_errors() {
        // A ↔ B
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &["b_provides"], &["a_provides"]).unwrap();
        g.register(&"b", &["a_provides"], &["b_provides"]).unwrap();

        let result = g.topo_sort();
        assert!(matches!(result, Err(GraphError::CircularDependency(_))));
    }

    #[test]
    fn test_topo_sort_from_roots() {
        // A → B → C, plus unrelated X → Y.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["a_out"]).unwrap();
        g.register(&"b", &["a_out"], &["b_out"]).unwrap();
        g.register(&"c", &["b_out"], &[]).unwrap();
        g.register(&"x", &[], &["x_out"]).unwrap();
        g.register(&"y", &["x_out"], &[]).unwrap();

        let order = g.topo_sort_from(&["c"]).unwrap();
        assert_eq!(order.len(), 3);
        assert!(order.contains(&"a"));
        assert!(order.contains(&"b"));
        assert!(order.contains(&"c"));
        assert!(!order.contains(&"x"));
        assert!(!order.contains(&"y"));
        // a must come before b, b before c
        let pos_a = order.iter().position(|n| n == &"a").unwrap();
        let pos_b = order.iter().position(|n| n == &"b").unwrap();
        let pos_c = order.iter().position(|n| n == &"c").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_topo_sort_from_empty_roots() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &[], &["x"]).unwrap();

        let order = g.topo_sort_from(&[]).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_topo_sort_self_loop_skipped() {
        // A node that depends on an asset it also provides must not
        // create a self-edge (which would deadlock the sort).
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"a", &["x"], &["x"]).unwrap();
        g.register(&"b", &[], &["x"]).unwrap();

        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_topo_sort_empty_graph() {
        let g: DependencyGraph<&str> = DependencyGraph::new();
        let order = g.topo_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_topo_sort_deterministic() {
        // Multiple nodes with no deps: output must be sorted.
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"c", &[], &[]).unwrap();
        g.register(&"a", &[], &[]).unwrap();
        g.register(&"b", &[], &[]).unwrap();

        let order = g.topo_sort().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_default_is_empty() {
        let g: DependencyGraph<&str> = DependencyGraph::default();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_nodes_preserves_insertion_order() {
        let mut g: DependencyGraph<&str> = DependencyGraph::new();
        g.register(&"c", &[], &[]).unwrap();
        g.register(&"a", &[], &[]).unwrap();
        g.register(&"b", &[], &[]).unwrap();

        assert_eq!(g.nodes(), &["c", "a", "b"]);
    }
}
