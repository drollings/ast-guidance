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
