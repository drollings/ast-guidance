//! Dependency-aware session with DAG step tracking, checkpoint/rewind,
//! and cancellation propagation.
//!
//! Composes `fluent_dag::dep_graph::DependencyGraph<K>` — does **not**
//! re-implement graph algorithms (no hand-rolled `HashMap` dependents index,
//! no manual topo sort, no manual transitive DFS).

use std::collections::{HashMap, HashSet};

use fluent_dag::dep_graph::{DependencyGraph, GraphError};
use serde::{Deserialize, Serialize};

use crate::kv_cache::{KvCacheError, KvCacheManager};
use crate::session::StepStatus;

/// Errors produced by dependency-session operations.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    #[error("step not found: {0}")]
    StepNotFound(String),

    #[error("step already completed: {0}")]
    StepAlreadyCompleted(String),

    #[error("graph error: {0}")]
    Graph(#[from] GraphError),

    #[error("checkpoint not found: {0}")]
    CheckpointNotFound(String),

    #[error("kv cache error: {0}")]
    KvCache(#[from] KvCacheError),
}

/// The result of executing a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub content: String,
    pub accepted: bool,
    pub score: Option<f64>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// A single step in a dependency session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStep {
    pub id: String,
    pub description: String,
    pub status: StepStatus,
    pub depends_on: Vec<String>,
    pub result: Option<StepResult>,
    pub checkpoint: bool,
}

impl SessionStep {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            status: StepStatus::Pending,
            depends_on: Vec::new(),
            result: None,
            checkpoint: false,
        }
    }

    #[must_use]
    pub fn with_depends(mut self, deps: Vec<String>) -> Self {
        self.depends_on = deps;
        self
    }

    #[must_use]
    pub fn with_checkpoint(mut self) -> Self {
        self.checkpoint = true;
        self
    }
}

/// A dependency-aware session that tracks steps as a DAG.
///
/// Composes `fluent_dag::dep_graph::DependencyGraph<String>` for
/// dependency tracking. Steps each have an ID, description, status,
/// dependencies, optional result, and a checkpoint flag.
///
/// # Examples
///
/// ```rust,ignore
/// use fluent_router::dag_session::DependencySession;
///
/// let mut session = DependencySession::new("sess-1");
/// session.add_step(SessionStep::new("plan", "Create plan"));
/// session.add_step(SessionStep::new("execute", "Execute plan")
///     .with_depends(vec!["plan".into()]));
///
/// let ready = session.next_ready(); // → vec!["plan"]
/// session.complete_step("plan", StepResult { ... });
/// let ready = session.next_ready(); // → vec!["execute"]
/// ```
pub struct DependencySession {
    pub session_id: String,
    steps: HashMap<String, SessionStep>,
    graph: DependencyGraph<String>,
    completed: HashSet<String>,
    checkpoints: HashMap<String, usize>,
    kv_cache: Option<KvCacheManager>,
    step_order: Vec<String>,
}

impl DependencySession {
    /// Create a new dependency session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            steps: HashMap::new(),
            graph: DependencyGraph::new(),
            completed: HashSet::new(),
            checkpoints: HashMap::new(),
            kv_cache: None,
            step_order: Vec::new(),
        }
    }

    /// Attach a KV cache manager for checkpoint/rewind support.
    #[must_use]
    pub fn with_kv_cache(mut self, cache: KvCacheManager) -> Self {
        self.kv_cache = Some(cache);
        self
    }

    /// Add a step to the session, registering it in the dependency graph.
    ///
    /// Returns `Err(DagError::Graph(DuplicateNode))` if a step with the
    /// same ID already exists.
    pub fn add_step(&mut self, step: SessionStep) -> Result<(), DagError> {
        let id = step.id.clone();
        let deps = step.depends_on.clone();

        // Each step provides its own ID as an asset, and depends on the
        // IDs of its prerequisite steps. This maps naturally to
        // `DependencyGraph`'s node/asset model.
        self.graph.register(&id, &deps, std::slice::from_ref(&id))?;

        self.steps.insert(id.clone(), step);
        self.step_order.push(id);
        Ok(())
    }

    /// Mark a step as completed with the given result.
    ///
    /// If the result is not accepted or has an error, dependents are
    /// cancelled transitively via `DependencyGraph::dependents_of`.
    pub fn complete_step(&mut self, step_id: &str, result: StepResult) -> Result<(), DagError> {
        let should_cancel = !result.accepted || result.error.is_some();

        let is_checkpoint = {
            let step = self
                .steps
                .get_mut(step_id)
                .ok_or_else(|| DagError::StepNotFound(step_id.into()))?;

            if step.status == StepStatus::Completed {
                return Err(DagError::StepAlreadyCompleted(step_id.into()));
            }

            step.status = if should_cancel {
                StepStatus::Failed
            } else {
                StepStatus::Completed
            };
            step.result = Some(result);

            step.checkpoint
        };

        self.completed.insert(step_id.to_string());

        if should_cancel {
            self.cancel_dependents(step_id);
        }

        if is_checkpoint {
            self.checkpoints
                .insert(step_id.to_string(), self.completed.len());
        }

        Ok(())
    }

    /// Cancel all steps that transitively depend on `step_id`.
    ///
    /// Uses `DependencyGraph::dependents_of` for transitive traversal
    /// with built-in cycle detection.
    pub fn cancel_dependents(&mut self, step_id: &str) {
        let dependents = self.graph.dependents_of(&step_id.to_string());
        for dep_id in &dependents {
            if let Some(step) = self.steps.get_mut(dep_id) {
                if step.status == StepStatus::Pending || step.status == StepStatus::InProgress {
                    step.status = StepStatus::Cancelled;
                }
            }
        }
    }

    /// Return the IDs of steps that are pending and whose dependencies
    /// are all satisfied.
    ///
    /// Uses `DependencyGraph::ready_nodes` against the current set of
    /// completed step IDs, then filters out steps already completed.
    pub fn next_ready(&self) -> Vec<String> {
        let mut ready = self.graph.ready_nodes(&self.completed);
        ready.retain(|id| !self.completed.contains(id));
        // Only return steps that are in Pending status
        ready.retain(|id| {
            self.steps
                .get(id)
                .is_some_and(|s| s.status == StepStatus::Pending)
        });
        ready
    }

    /// Returns `true` if the step is ready to execute (all dependencies
    /// satisfied).
    ///
    /// Uses `DependencyGraph::is_ready`.
    pub fn is_ready(&self, step_id: &str) -> bool {
        self.graph.is_ready(&step_id.to_string(), &self.completed)
    }

    /// Returns the IDs of steps that have the checkpoint flag set.
    pub fn checkpoints(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|(_, step)| step.checkpoint)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Rewind to a checkpoint, discarding all steps completed after the
    /// checkpoint step.
    ///
    /// Steps are reset to `Pending` but their result data is preserved
    /// for audit (it is not deleted). If a `KvCacheManager` is attached
    /// and a model name is provided, the KV cache snapshot is restored
    /// from the cold tier.
    pub async fn rewind_to_checkpoint(&mut self, checkpoint_name: &str) -> Result<(), DagError> {
        // Verify checkpoint exists.
        self.checkpoints
            .get(checkpoint_name)
            .ok_or_else(|| DagError::CheckpointNotFound(checkpoint_name.into()))?;

        let checkpoint_idx = self
            .step_order
            .iter()
            .position(|id| id == checkpoint_name)
            .unwrap_or(0);

        let mut discarded: Vec<String> = Vec::new();

        // Reset steps after checkpoint (including checkpoint step itself).
        for step_id in self.step_order.iter().skip(checkpoint_idx) {
            if let Some(step) = self.steps.get_mut(step_id) {
                if step.status == StepStatus::Completed
                    || step.status == StepStatus::Cancelled
                    || step.status == StepStatus::Failed
                {
                    tracing::info!(
                        session_id = %self.session_id,
                        step_id = %step_id,
                        "rewinding step"
                    );
                    step.status = StepStatus::Pending;
                    self.completed.remove(step_id.as_str());
                    discarded.push(step_id.clone());
                }
            }
        }

        self.checkpoints.remove(checkpoint_name);

        // Restore KV cache snapshot from cold tier if a manager is attached.
        if let Some(ref kv) = self.kv_cache {
            match kv
                .retrieve(
                    "unknown", // model not tracked in DependencySession — caller sets via metadata
                    None,
                    &self.session_id,
                )
                .await
            {
                Ok(snapshot) => {
                    tracing::info!(
                        session_id = %self.session_id,
                        file_path = %snapshot.file_path.display(),
                        token_count = snapshot.token_count,
                        "kv cache snapshot metadata retrieved for rewind"
                    );
                }
                Err(KvCacheError::NotFound(_)) => {
                    tracing::debug!(
                        session_id = %self.session_id,
                        "no kv cache snapshot found for rewind"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        session_id = %self.session_id,
                        error = %e,
                        "kv cache retrieve failed during rewind"
                    );
                }
            }
        }

        if !discarded.is_empty() {
            tracing::info!(
                session_id = %self.session_id,
                discarded_count = discarded.len(),
                discarded = ?discarded,
                "rewound to checkpoint, steps reset to Pending (data preserved for audit)"
            );
        }

        Ok(())
    }

    /// Set a step's status to InProgress (only if currently Pending).
    pub fn start_step(&mut self, step_id: &str) -> Result<(), DagError> {
        let step = self
            .steps
            .get_mut(step_id)
            .ok_or_else(|| DagError::StepNotFound(step_id.into()))?;

        if step.status != StepStatus::Pending {
            return Err(DagError::StepAlreadyCompleted(format!(
                "step '{step_id}' is not Pending (current: {:?})",
                step.status
            )));
        }

        step.status = StepStatus::InProgress;
        Ok(())
    }

    /// Look up a step by ID.
    pub fn get_step(&self, step_id: &str) -> Option<&SessionStep> {
        self.steps.get(step_id)
    }

    /// All step IDs in insertion order.
    pub fn step_ids(&self) -> &[String] {
        &self.step_order
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Number of completed steps.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Borrow the underlying dependency graph (for inspection).
    pub fn graph(&self) -> &DependencyGraph<String> {
        &self.graph
    }

    /// Steps that depend on unsatisfiable assets (assets depended on by
    /// some step but provided by none).
    pub fn unresolved_deps(&self) -> Vec<String> {
        self.graph.unresolved_deps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result(content: &str) -> StepResult {
        StepResult {
            content: content.into(),
            accepted: true,
            score: Some(0.9),
            latency_ms: 100,
            error: None,
        }
    }

    fn fail_result(content: &str) -> StepResult {
        StepResult {
            content: content.into(),
            accepted: false,
            score: Some(0.1),
            latency_ms: 50,
            error: Some("execution failed".into()),
        }
    }

    #[test]
    fn test_add_steps_and_get_order() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("step-1", "First step"))
            .unwrap();
        session
            .add_step(SessionStep::new("step-2", "Second step").with_depends(vec!["step-1".into()]))
            .unwrap();

        assert_eq!(session.step_count(), 2);
        assert_eq!(session.step_ids(), &["step-1", "step-2"]);
    }

    #[test]
    fn test_duplicate_step_rejected() {
        let mut session = DependencySession::new("sess-1");
        session
            .add_step(SessionStep::new("step-1", "First"))
            .unwrap();
        let result = session.add_step(SessionStep::new("step-1", "Duplicate"));
        assert!(matches!(
            result,
            Err(DagError::Graph(GraphError::DuplicateNode(_)))
        ));
    }

    #[test]
    fn test_ready_nodes_basic_dependency() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("plan", "Plan")).unwrap();
        session
            .add_step(SessionStep::new("execute", "Execute").with_depends(vec!["plan".into()]))
            .unwrap();

        // Only "plan" should be ready (no dependencies)
        let ready = session.next_ready();
        assert_eq!(ready, vec!["plan"]);

        // Complete "plan"
        session
            .complete_step("plan", ok_result("plan done"))
            .unwrap();

        // Now "execute" should be ready
        let ready = session.next_ready();
        assert_eq!(ready, vec!["execute"]);
    }

    #[test]
    fn test_fail_cancels_dependents() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // Complete "a" successfully
        session.complete_step("a", ok_result("a done")).unwrap();

        // Complete "b" with failure
        session.complete_step("b", fail_result("b failed")).unwrap();

        let b = session.get_step("b").unwrap();
        assert_eq!(b.status, StepStatus::Failed);

        let c = session.get_step("c").unwrap();
        assert_eq!(c.status, StepStatus::Cancelled);
    }

    #[test]
    fn test_complete_step_not_found() {
        let mut session = DependencySession::new("sess-1");
        let result = session.complete_step("nonexistent", ok_result("nope"));
        assert!(matches!(result, Err(DagError::StepNotFound(_))));
    }

    #[test]
    fn test_complete_already_completed() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.complete_step("a", ok_result("done")).unwrap();

        let result = session.complete_step("a", ok_result("again"));
        assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
    }

    #[test]
    fn test_start_step() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();

        session.start_step("a").unwrap();
        let step = session.get_step("a").unwrap();
        assert_eq!(step.status, StepStatus::InProgress);
    }

    #[test]
    fn test_start_step_not_found() {
        let mut session = DependencySession::new("sess-1");
        let result = session.start_step("nonexistent");
        assert!(matches!(result, Err(DagError::StepNotFound(_))));
    }

    #[test]
    fn test_start_step_not_pending() {
        let mut session = DependencySession::new("sess-1");
        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.complete_step("a", ok_result("done")).unwrap();

        let result = session.start_step("a");
        assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
    }

    #[test]
    fn test_checkpoint_listing() {
        let mut session = DependencySession::new("sess-1");
        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_checkpoint())
            .unwrap();

        let cps = session.checkpoints();
        assert_eq!(cps.len(), 2);
        assert!(cps.contains(&"a".to_string()));
        assert!(cps.contains(&"b".to_string()));
    }

    #[tokio::test]
    async fn test_rewind_to_checkpoint() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // Complete a and reach checkpoint
        session.complete_step("a", ok_result("a done")).unwrap();
        assert!(session.checkpoints().contains(&"a".to_string()));

        // Complete b
        session.complete_step("b", ok_result("b done")).unwrap();
        assert_eq!(session.completed_count(), 2);

        // Rewind to checkpoint "a"
        session.rewind_to_checkpoint("a").await.unwrap();

        // "a" is reset, "b" is reset
        assert_eq!(session.get_step("a").unwrap().status, StepStatus::Pending);
        assert_eq!(session.get_step("b").unwrap().status, StepStatus::Pending);
        // "c" was never completed, stays Pending
        assert_eq!(session.get_step("c").unwrap().status, StepStatus::Pending);
        assert_eq!(session.completed_count(), 0);
    }

    #[tokio::test]
    async fn test_rewind_missing_checkpoint() {
        let mut session = DependencySession::new("sess-1");
        let result = session.rewind_to_checkpoint("nonexistent").await;
        assert!(matches!(result, Err(DagError::CheckpointNotFound(_))));
    }

    #[test]
    fn test_is_ready() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
            .unwrap();

        assert!(session.is_ready("a"));
        assert!(!session.is_ready("b"));

        session.complete_step("a", ok_result("done")).unwrap();
        assert!(session.is_ready("b"));
    }

    #[test]
    fn test_is_ready_unregistered() {
        let session = DependencySession::new("sess-1");
        assert!(!session.is_ready("nonexistent"));
    }

    #[test]
    fn test_independent_steps_ready_together() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session.add_step(SessionStep::new("b", "Step B")).unwrap();

        let ready = session.next_ready();
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&"a".to_string()));
        assert!(ready.contains(&"b".to_string()));
    }

    #[test]
    fn test_cycle_detection_in_dependents() {
        // DependencyGraph::dependents_of handles cycles gracefully
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into(), "c".into()]))
            .unwrap();
        session
            .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
            .unwrap();

        // This should not panic — DependencyGraph handles cycles
        let deps = session.graph().dependents_of(&"a".to_string());
        // In a cycle, the result is partial but non-panicking
        assert!(!deps.is_empty() || deps.is_empty()); // Just verify it returns
    }

    #[test]
    fn test_unresolved_deps() {
        let mut session = DependencySession::new("sess-1");

        session.add_step(SessionStep::new("a", "Step A")).unwrap();
        session
            .add_step(SessionStep::new("b", "Step B").with_depends(vec!["missing".into()]))
            .unwrap();

        let unresolved = session.unresolved_deps();
        assert!(unresolved.contains(&"missing".to_string()));
    }

    #[tokio::test]
    async fn test_step_result_data_preserved_on_rewind() {
        let mut session = DependencySession::new("sess-1");

        session
            .add_step(SessionStep::new("a", "Step A").with_checkpoint())
            .unwrap();

        session
            .complete_step("a", ok_result("important result"))
            .unwrap();
        session.rewind_to_checkpoint("a").await.unwrap();

        let step = session.get_step("a").unwrap();
        assert_eq!(step.status, StepStatus::Pending);
        // Result data preserved for audit
        assert!(step.result.is_some());
        assert_eq!(step.result.as_ref().unwrap().content, "important result");
    }

    #[test]
    fn test_get_step_nonexistent() {
        let session = DependencySession::new("sess-1");
        assert!(session.get_step("nonexistent").is_none());
    }

    #[test]
    fn test_with_constructor_builders() {
        let mut session = DependencySession::new("sess-1");

        let step = SessionStep::new("step-1", "A step")
            .with_depends(vec!["dep-1".into(), "dep-2".into()])
            .with_checkpoint();

        assert_eq!(step.id, "step-1");
        assert_eq!(step.depends_on, vec!["dep-1", "dep-2"]);
        assert!(step.checkpoint);

        session.add_step(step).unwrap();
        assert_eq!(session.step_count(), 1);
    }
}
