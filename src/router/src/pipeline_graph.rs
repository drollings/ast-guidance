//! PipelineGraph — DAG-based pipeline executor.
//!
//! Composes `fluent_dag::dep_graph::DependencyGraph<String>` for topological
//! ordering of stages. Unlike `PipelineOrchestrator` (which executes stages
//! in insertion order), `PipelineGraph` resolves the execution order from
//! each stage's `depends()` / `provides()` declarations.
//!
//! # Execution flow
//!
//! 1. **Build graph** — register every stage in a `DependencyGraph` using
//!    its declared deps/provides, then topo-sort.
//! 2. **Walk topo order** — execute each stage sequentially.  Branching
//!    (via `SwitchStage`) is transparent because the switch itself is a
//!    `WorkUnit` that internally delegates to the selected sub-pipeline.
//! 3. **Promote metadata** — after each stage, scalar fields from the
//!    stage's `StageDecision.metadata` are copied into `WorkContext.metadata`
//!    so downstream `SwitchStage` instances can read them.
//! 4. **Return** — aggregate decisions into a `PipelineResult` (same type
//!    used by `PipelineOrchestrator`).
//!
//! # Cycle detection
//!
//! `DependencyGraph::topo_sort()` returns `Err(GraphError::CircularDependency)`
//! if the stage dependency graph contains a cycle.  `PipelineGraph::new()`
//! surfaces this as a `WorkError` rather than panicking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use fluent_dag::dep_graph::{DependencyGraph, GraphError};
use fluent_wvr::prelude::*;

use crate::pipeline::{PipelineResult, RoutingTarget};
use crate::pipeline_types::{PipelineStage, StageDecision, StageMetadata, StageVerdict};
use crate::stages::common::get_metadata_string;
use crate::stages::switch::promote_decision_metadata;

/// A DAG-based pipeline executor.  Stages are registered with their
/// dependency declarations; execution order is derived from topological
/// sort of the resulting dependency graph.
pub struct PipelineGraph {
    name: ArcIntern<str>,
    /// All stages in the graph, in insertion order.
    stages: Vec<Arc<dyn Component>>,
    /// Indices into `stages` in topological execution order.
    stage_order: Vec<usize>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl PipelineGraph {
    /// Build a pipeline graph from a list of stages.  Stages are registered
    /// in a `DependencyGraph` using their `depends()` and `provides()`
    /// declarations; the execution order is derived from a topological sort.
    ///
    /// Returns an error if the dependency graph contains a cycle or if any
    /// required dependency cannot be satisfied (unresolved deps are treated
    /// as unsatisfiable — stages depending on them will never become ready
    /// and won't appear in the execution order).
    pub fn new(stages: Vec<Arc<dyn Component>>) -> Result<Self, GraphError> {
        if stages.is_empty() {
            // Empty graph: no stages, no deps, no execution.
            return Ok(Self {
                name: ArcIntern::from("pipeline.graph"),
                stages: Vec::new(),
                stage_order: Vec::new(),
                depends: vec![],
                provides: vec![ArcIntern::from("pipeline.result")],
            });
        }

        let mut dep_graph: DependencyGraph<String> = DependencyGraph::new();

        for stage in &stages {
            let stage_name = stage.name().to_string();
            let deps: Vec<String> = stage.depends().iter().map(ToString::to_string).collect();
            let provides: Vec<String> = stage.provides().iter().map(ToString::to_string).collect();

            dep_graph
                .register(&stage_name, &deps, &provides)
                .map_err(|e| match e {
                    GraphError::DuplicateNode(_) => GraphError::DuplicateNode(format!(
                        "pipeline graph: duplicate stage name: {stage_name}"
                    )),
                    other => other,
                })?;
        }

        // Verify no unsatisfiable dependencies exist (stages depending on
        // assets provided by no one).
        let unresolved = dep_graph.unresolved_deps();
        if !unresolved.is_empty() {
            tracing::warn!(
                target: "router.pipeline_graph",
                unresolved = ?unresolved,
                "unresolved dependencies detected; stages depending on \
                 these assets will never become ready in this graph"
            );
        }

        let order = dep_graph.topo_sort()?;

        // Map sorted names back to stage indices.
        let name_to_idx: HashMap<&str, usize> = stages
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name(), i))
            .collect();

        let mut stage_order: Vec<usize> = Vec::with_capacity(order.len());
        for name in &order {
            if let Some(&idx) = name_to_idx.get(name.as_str()) {
                stage_order.push(idx);
            } else {
                tracing::warn!(
                    target: "router.pipeline_graph",
                    stage_name = %name,
                    "stage in topo sort not found in stage list"
                );
            }
        }

        tracing::info!(
            target: "router.pipeline_graph",
            total_stages = stages.len(),
            execution_order = ?order,
            "pipeline graph built"
        );

        Ok(Self {
            name: ArcIntern::from("pipeline.graph"),
            stages,
            stage_order,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.result")],
        })
    }

    /// Convenience: build a pipeline graph from a `PipelineOrchestrator`
    /// (which is itself a `Component`).  This wraps the existing linear
    /// orchestrator in a single-node DAG for compatibility.
    pub fn from_orchestrator(
        orchestrator: Arc<crate::pipeline::PipelineOrchestrator>,
    ) -> Result<Self, GraphError> {
        Self::new(vec![orchestrator as Arc<dyn Component>])
    }

    /// Build the per-stage `WorkContext`, populating metadata with the
    /// current request string and accumulated decision fields.
    fn build_stage_context(
        base: &WorkContext,
        current_request: &str,
        accumulated_metadata: &HashMap<String, MetadataValue>,
    ) -> WorkContext {
        let mut ctx = base.clone();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(current_request.to_string()),
        );
        // Copy accumulated decision fields so SwitchStage can read them.
        for (k, v) in accumulated_metadata {
            ctx.metadata.insert(k.clone(), v.clone());
        }
        ctx
    }
}

impl WorkUnit for PipelineGraph {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        if self.stages.is_empty() {
            return WorkOutput::typed(
                "graph_empty",
                &PipelineResult {
                    decisions: Vec::new(),
                    final_response: None,
                    rejected: false,
                    reject_reason: None,
                    routing_target: None,
                    classifier_response: None,
                },
            );
        }

        let mut decisions: Vec<StageDecision> = Vec::new();
        let mut accumulated_metadata: HashMap<String, MetadataValue> = HashMap::new();
        let mut current_request = get_metadata_string(ctx, "request").unwrap_or_default();
        let mut routing_target: Option<RoutingTarget> = None;
        let mut classifier_response: Option<String> = None;

        for &stage_idx in &self.stage_order {
            let stage = &self.stages[stage_idx];
            let stage_ctx = Self::build_stage_context(ctx, &current_request, &accumulated_metadata);
            let start = Instant::now();
            let stage_name_human = stage.name().to_string();

            tracing::debug!(
                target: "router.pipeline_graph",
                stage = %stage_name_human,
                "stage entering"
            );

            match stage.execute(&stage_ctx) {
                Ok(output) => {
                    let mut decision: StageDecision = output
                        .data_take()
                        .map_err(|e| WorkError::Execution(e.to_string()))?;

                    let latency_ms = start.elapsed().as_millis() as u64;
                    decision.latency_ms = latency_ms;
                    let verdict = decision.verdict.clone();
                    let stage_name = decision.stage;

                    let fallback = stage_name == PipelineStage::Classifier
                        && decision
                            .metadata
                            .get("fallback")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false);

                    tracing::info!(
                        target: "router.pipeline_graph",
                        stage = ?stage_name,
                        verdict = ?verdict,
                        latency_ms = latency_ms,
                        score = ?decision.score,
                        reason = %decision.reason,
                        fallback = fallback,
                        "stage complete"
                    );

                    decisions.push(decision.clone());

                    // Promote decision metadata so SwitchStage can read
                    // fields like "intent", "complexity", "action".
                    let prefix = match stage_name {
                        PipelineStage::DeterministicPreFilter => "prefilter",
                        PipelineStage::Classifier => "classifier",
                        PipelineStage::Router => "router",
                    };
                    promote_decision_metadata(
                        &mut accumulated_metadata,
                        prefix,
                        &decision.metadata,
                    );

                    let metadata = StageMetadata::from(decision.metadata.clone());
                    match verdict {
                        StageVerdict::Passed | StageVerdict::Skipped => {
                            if stage_name == PipelineStage::Classifier {
                                if let Some(resp) = metadata.response() {
                                    tracing::info!(
                                        target: "router.pipeline_graph",
                                        response_len = resp.len(),
                                        "classifier direct response"
                                    );
                                    classifier_response = Some(resp.to_string());
                                }
                                if let Some(rt) = metadata.routing_target() {
                                    routing_target = Some(rt);
                                }
                            }
                        }
                        StageVerdict::Rerouted => {
                            if let Some(rewritten) = metadata.rewritten_request() {
                                tracing::info!(
                                    target: "router.pipeline_graph",
                                    new_request_len = rewritten.len(),
                                    "request rerouted"
                                );
                                current_request = rewritten.to_string();
                            }
                        }
                        StageVerdict::Rejected => {
                            tracing::info!(
                                target: "router.pipeline_graph",
                                stage = ?stage_name,
                                reason = %decision.reason,
                                "pipeline rejected request"
                            );
                            return WorkOutput::typed(
                                "rejected",
                                &PipelineResult {
                                    decisions,
                                    final_response: None,
                                    rejected: true,
                                    reject_reason: Some(decision.reason),
                                    routing_target: None,
                                    classifier_response: None,
                                },
                            );
                        }
                        StageVerdict::Error => {
                            tracing::error!(
                                target: "router.pipeline_graph",
                                stage = ?stage_name,
                                reason = %decision.reason,
                                "stage error"
                            );
                            return WorkOutput::typed(
                                "pipeline_error",
                                &PipelineResult {
                                    decisions,
                                    final_response: None,
                                    rejected: true,
                                    reject_reason: Some(format!(
                                        "stage error: {}",
                                        decision.reason
                                    )),
                                    routing_target: None,
                                    classifier_response: None,
                                },
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        target: "router.pipeline_graph",
                        stage = %stage_name_human,
                        error = %e,
                        latency_ms = %start.elapsed().as_millis(),
                        "stage execution error"
                    );
                    decisions.push(StageDecision {
                        stage: PipelineStage::Router,
                        verdict: StageVerdict::Error,
                        score: None,
                        reason: e.to_string(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        metadata: serde_json::json!({}),
                    });
                    return Err(e);
                }
            }
        }

        let has_routing = routing_target.is_some();
        let has_classifier_resp = classifier_response.is_some();
        tracing::info!(
            target: "router.pipeline_graph",
            stages = decisions.len(),
            has_routing_target = has_routing,
            has_classifier_response = has_classifier_resp,
            routing_model = ?routing_target.as_ref().map(|rt| &rt.model),
            "graph complete"
        );

        WorkOutput::typed(
            "graph_complete",
            &PipelineResult {
                decisions,
                final_response: None,
                rejected: false,
                reject_reason: None,
                routing_target,
                classifier_response,
            },
        )
    }
}

impl FieldAccess for PipelineGraph {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "PipelineGraph has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "PipelineGraph has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for PipelineGraph {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "stage_count": self.stages.len(),
                "execution_order": self.stage_order.iter().map(|&i| {
                    self.stages[i].name()
                }).collect::<Vec<&str>>(),
            },
            "required": []
        })
    }
}

impl_component!(PipelineGraph);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_stubs::SimplePassStage;

    /// Create a stage with custom dependency declarations.
    fn make_dep_stage(name: &str, dep: Option<&str>, prov: Option<&str>) -> Arc<dyn Component> {
        let mut stub = fluent_wvr_testutil::StubComponent::new(name);
        if let Some(d) = dep {
            stub = stub.with_dep(d);
        }
        if let Some(p) = prov {
            stub = stub.with_provides(p);
        }
        Arc::new(stub)
    }

    #[test]
    fn graph_toposorts_linear_chain() {
        let a = make_dep_stage("a", None, Some("x"));
        let b = make_dep_stage("b", Some("x"), Some("y"));
        let c = make_dep_stage("c", Some("y"), None);

        let graph = PipelineGraph::new(vec![b.clone(), c.clone(), a.clone()]).expect("build graph");

        // a must execute before b, b before c
        let names: Vec<&str> = graph
            .stage_order
            .iter()
            .map(|&i| graph.stages[i].name())
            .collect();

        let pos_a = names.iter().position(|&n| n == "a").unwrap();
        let pos_b = names.iter().position(|&n| n == "b").unwrap();
        let pos_c = names.iter().position(|&n| n == "c").unwrap();
        assert!(pos_a < pos_b, "a must execute before b");
        assert!(pos_b < pos_c, "b must execute before c");
    }

    #[test]
    fn graph_detects_cycle() {
        let a = make_dep_stage("a", Some("b_provides"), Some("a_provides"));
        let b = make_dep_stage("b", Some("a_provides"), Some("b_provides"));

        let result = PipelineGraph::new(vec![a, b]);
        assert!(result.is_err(), "cycle should be detected and rejected");
    }

    #[test]
    fn graph_duplicate_stage_name_errors() {
        let a = make_dep_stage("dup", None, None);
        let b = make_dep_stage("dup", None, None);

        let result = PipelineGraph::new(vec![a, b]);
        assert!(result.is_err(), "duplicate stage names should error");
    }

    #[test]
    fn empty_graph_produces_empty_result() {
        let graph = PipelineGraph::new(vec![]).unwrap();
        let ctx = WorkContext::default();
        let output = graph.execute(&ctx).unwrap();
        let result: PipelineResult = output.data_as().unwrap();
        assert!(result.decisions.is_empty());
        assert!(!result.rejected);
    }

    #[test]
    fn graph_executes_stages_in_topo_order() {
        let a = Arc::new(SimplePassStage::new("stage_a", "a completed"));
        let b = Arc::new(SimplePassStage::new("stage_b", "b completed"));
        let graph = PipelineGraph::new(vec![a, b]).unwrap();

        let ctx = WorkContext::default();
        let output = graph.execute(&ctx).unwrap();
        let result: PipelineResult = output.data_as().unwrap();
        assert_eq!(result.decisions.len(), 2);
    }

    #[test]
    fn graph_stage_rejection_short_circuits() {
        let failing = Arc::new(fluent_wvr_testutil::StubComponent::fail("failing_stage"));
        let _after = Arc::new(SimplePassStage::new("after_stage", "would run after"));
        let graph = PipelineGraph::new(vec![failing, _after]).unwrap();

        let ctx = WorkContext::default();
        let err = graph.execute(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("stub fail"),
            "graph should propagate stage error: {err}"
        );
    }
}
