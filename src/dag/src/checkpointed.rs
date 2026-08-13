//! Checkpoint/rewind over an ordered dependency-graph step list.
//!
//! The "session-graph + checkpoint-restore" primitive that `DependencySession`
//! and the rigor/plan routes need: an ordered set of steps whose readiness is
//! answered by a `DependencyGraph<K>` and whose checkpoint/rewind bookkeeping
//! (insertion order, checkpoint markers, the completed set) lives here.
//!
//! The primitive is deliberately free of any step-status/result vocabulary:
//! `S` is an opaque per-step state the consumer supplies on `add_step` and
//! resets on rewind. `rewind_to` returns the ordered keys to reset and clears
//! the checkpoint; the consumer decides what "reset to Pending" means for `S`
//! (preserving result data for audit).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::dep_graph::{DependencyGraph, GraphError};

/// An ordered step graph with checkpoint/rewind.
///
/// `K` is the step key, `S` the opaque per-step state. Owns the
/// `DependencyGraph<K>` (readiness/dependents), the insertion order, the
/// per-step state, the completed set, and the checkpoint markers.
pub struct CheckpointedStepGraph<K, S>
where
    K: Eq + Hash + Clone,
{
    graph: DependencyGraph<K>,
    /// Insertion order (rewind walks this).
    order: Vec<K>,
    /// Current per-step state.
    state: HashMap<K, S>,
    /// Satisfied (completed) keys.
    completed: HashSet<K>,
    /// Checkpoint name -> position in `order`.
    checkpoints: HashMap<K, usize>,
}

impl<K, S> CheckpointedStepGraph<K, S>
where
    K: Eq + Hash + Clone + Debug,
{
    /// Create an empty step graph.
    pub fn new() -> Self {
        Self {
            graph: DependencyGraph::new(),
            order: Vec::new(),
            state: HashMap::new(),
            completed: HashSet::new(),
            checkpoints: HashMap::new(),
        }
    }

    /// Register a step with its dependency keys, supplying its initial state.
    ///
    /// Each step provides its own key as an asset (the same convention
    /// `DependencySession` uses: step IDs double as provided assets), so
    /// dependents of `key` are satisfied once `key` is [`Self::complete`]d.
    /// Returns `Err(GraphError::DuplicateNode)` if `key` is already present.
    pub fn add_step(&mut self, key: K, deps: &[K], state: S) -> Result<(), GraphError> {
        self.graph.register(&key, deps, std::slice::from_ref(&key))?;
        self.order.push(key.clone());
        self.state.insert(key, state);
        Ok(())
    }

    /// Mark `name` as a checkpoint at its current position in the order. A
    /// [`Self::rewind_to`] resets the suffix starting at this step (inclusive).
    /// Fails if `name` is not a registered step.
    pub fn checkpoint(&mut self, name: K) -> Result<(), GraphError> {
        let idx = self
            .order
            .iter()
            .position(|k| k == &name)
            .ok_or_else(|| GraphError::NodeNotFound(format!("{name:?}")))?;
        self.checkpoints.insert(name, idx);
        Ok(())
    }

    /// Record `key` as satisfied (its dependents become ready).
    pub fn complete(&mut self, key: &K) {
        self.completed.insert(key.clone());
    }

    /// Whether every dependency of `key` is satisfied.
    pub fn is_ready(&self, key: &K) -> bool {
        self.graph.is_ready(key, &self.completed)
    }

    /// Steps whose dependencies are all satisfied, in insertion order.
    pub fn ready_steps(&self) -> Vec<K> {
        self.graph.ready_nodes(&self.completed)
    }

    /// The transitive dependents of `key` (cycle-resilient).
    pub fn cancel_dependents(&self, key: &K) -> Vec<K> {
        self.graph.dependents_of(key)
    }

    /// Rewind to the checkpoint `name`: clears the checkpoint and returns the
    /// ordered steps from `name` onward (inclusive), removing them from the
    /// completed set so readiness is correct after the reset.
    ///
    /// The consumer resets each returned key's state to "pending" — result
    /// data is preserved for audit. Returns `Err(GraphError::NodeNotFound)`
    /// when `name` is not a recorded checkpoint.
    pub fn rewind_to(&mut self, name: &K) -> Result<Vec<K>, GraphError> {
        let idx = *self
            .checkpoints
            .get(name)
            .ok_or_else(|| GraphError::NodeNotFound(format!("{name:?}")))?;
        self.checkpoints.remove(name);
        let reset: Vec<K> = self.order[idx..].to_vec();
        for key in &reset {
            self.completed.remove(key);
        }
        Ok(reset)
    }

    /// Borrow a step's state.
    pub fn status(&self, key: &K) -> Option<&S> {
        self.state.get(key)
    }

    /// Mutably borrow a step's state (the consumer mutates status/result).
    pub fn state_mut(&mut self, key: &K) -> Option<&mut S> {
        self.state.get_mut(key)
    }

    /// Whether `key` is in the completed set.
    pub fn is_completed(&self, key: &K) -> bool {
        self.completed.contains(key)
    }

    /// Whether `name` is a recorded checkpoint.
    pub fn is_checkpoint(&self, key: &K) -> bool {
        self.checkpoints.contains_key(key)
    }

    /// All step keys in insertion order.
    pub fn step_ids(&self) -> &[K] {
        &self.order
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.order.len()
    }

    /// Number of completed steps.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// `true` when no steps are registered.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Borrow the underlying dependency graph (for inspection).
    pub fn graph(&self) -> &DependencyGraph<K> {
        &self.graph
    }

    /// Steps that depend on assets provided by no step.
    pub fn unresolved_deps(&self) -> Vec<K> {
        self.graph.unresolved_deps()
    }
}

impl<K, S> Default for CheckpointedStepGraph<K, S>
where
    K: Eq + Hash + Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
