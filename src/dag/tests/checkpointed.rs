use super::*;

fn graph() -> CheckpointedStepGraph<&'static str, &'static str> {
    let mut g = CheckpointedStepGraph::new();
    g.add_step("a", &[], "A").unwrap();
    g.add_step("b", &["a"], "B").unwrap();
    g.add_step("c", &["b"], "C").unwrap();
    g
}

#[test]
fn add_step_duplicate_is_rejected() {
    let mut g = graph();
    assert!(matches!(
        g.add_step("a", &[], "again"),
        Err(GraphError::DuplicateNode(_))
    ));
    assert_eq!(g.step_count(), 3);
}

#[test]
fn ready_and_complete_flow() {
    let mut g = graph();
    assert!(g.is_ready(&"a"));
    assert!(!g.is_ready(&"b"));
    assert_eq!(g.ready_steps(), vec!["a"]);
    g.complete(&"a");
    assert!(g.is_ready(&"b"));
    assert!(g.is_completed(&"a"));
    assert_eq!(g.completed_count(), 1);
}

#[test]
fn cancel_dependents_delegates_to_graph() {
    let g = graph();
    let mut deps = g.cancel_dependents(&"a");
    deps.sort();
    assert_eq!(deps, vec!["b", "c"]);
}

#[test]
fn rewind_returns_suffix_clears_checkpoint_and_uncompletes() {
    let mut g = graph();
    g.checkpoint("b").unwrap();
    g.complete(&"a");
    g.complete(&"b");
    g.complete(&"c");
    assert_eq!(g.completed_count(), 3);

    let reset = g.rewind_to(&"b").unwrap();
    assert_eq!(reset, vec!["b", "c"]);
    assert!(!g.is_checkpoint(&"b"));
    assert!(g.is_completed(&"a"), "steps before the checkpoint stay completed");
    assert!(!g.is_completed(&"b"));
    assert!(!g.is_completed(&"c"));
}

#[test]
fn rewind_unknown_checkpoint_is_error() {
    let mut g = graph();
    assert!(matches!(
        g.rewind_to(&"nope"),
        Err(GraphError::NodeNotFound(_))
    ));
}

#[test]
fn rewind_preserves_state_for_consumer_reset() {
    let mut g = graph();
    g.checkpoint("b").unwrap();
    g.complete(&"b");
    let reset = g.rewind_to(&"b").unwrap();
    // The consumer decides what "reset to Pending" means for S; the
    // primitive leaves the state intact.
    assert_eq!(g.status(&"b"), Some(&"B"));
    assert_eq!(reset, vec!["b", "c"]);
}

#[test]
fn state_mut_updates_in_place() {
    let mut g = graph();
    *g.state_mut(&"b").unwrap() = "B2";
    assert_eq!(g.status(&"b"), Some(&"B2"));
    assert!(g.state_mut(&"missing").is_none());
}

#[test]
fn step_ids_and_count() {
    let g = graph();
    assert_eq!(g.step_ids(), &["a", "b", "c"]);
    assert_eq!(g.step_count(), 3);
    assert!(!g.is_empty());
}

#[test]
fn default_is_empty() {
    let g: CheckpointedStepGraph<&'static str, &'static str> = Default::default();
    assert!(g.is_empty());
}
