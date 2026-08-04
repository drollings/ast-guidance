//! Chart compiler — turns a validated, fully-bound `ChartDef` into a
//! `WorkflowConfig` whose stages are `chart_prompt` (`WorkflowStage::ChartPrompt`),
//! ready for `PipelineGraph` execution.
//!
//! Compilation is deterministic and fail-fast: a chart whose deps are not
//! fully satisfiable (unmatched required dep, or ambiguous binding) returns
//! `ChartError::Compile` instead of producing a partially-runnable graph.
//! The compiled stage graph is verified with `DependencyGraph` (cycle +
//! unresolved-deps check) before being returned. See `ROADMAP_20260802_DAG_WORKFLOW.md` M5.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_dag::dep_graph::DependencyGraph;
use fluent_wvr::prelude::Component;
use guidance_llm::client::ChatBackend;

use crate::charts::stage::ChartPromptStage;
use crate::workflow_config::{WorkflowConfig, WorkflowDef, WorkflowStage};

use super::binding::{bind_entities, Entity};
use super::{ChartDef, ChartError, ChartRubric, DepSpec};

/// A compiled chart target: the executable stage plus the metadata the
/// supervisor needs — the `essential` flag, the acceptance `rubric` (M9),
/// the upstream stage ids it reads from, and the concrete assets it provides.
///
/// Produced by [`compile_chart_stages`]; consumed by `RouterConfig`
/// (`build_chart_pipeline` → `PipelineGraph`) and by the M9 Zone supervisor
/// (`ChartExecutionPlan`).
#[derive(Clone)]
pub struct CompiledTarget {
    /// The executable `ChartPromptStage` (also a `Component`).
    pub stage: Arc<dyn Component>,
    /// Target name == stage id == self-provided asset.
    pub name: String,
    /// `essential` flag — a failed essential target fails the whole chart.
    pub essential: bool,
    /// Acceptance rubric gating this target's output (M9). `None` = no gate.
    pub rubric: Option<ChartRubric>,
    /// Upstream chart-target stage ids this target's output depends on.
    pub upstream_ids: Vec<String>,
    /// Concrete asset names this target provides (explicit `provides` +
    /// self-name, deduplicated).
    pub provides: Vec<String>,
}

/// Compile a chart into a `WorkflowConfig` for the given bound entities.
///
/// Resolution rules per `DepSpec` (mirror the runtime semantics in
/// `ChartPromptStage::execute`):
///
/// - `Capability { name }` provided by another chart target → a `depends_on`
///   edge to that target's stage id (its self-provided asset name).
/// - `Capability { name }` satisfied only by a bound entity → no stage edge
///   (the entity preamble is injected by the renderer via `deps`).
/// - `EntityMatch` satisfied by a bound entity → no stage edge.
/// - Anything else (unmatched required dep, or ambiguous binding) →
///   `ChartError::Compile` — the chart is not executable until the interview
///   loop (M8) resolves the gap.
///
/// The returned `WorkflowConfig` has a single workflow keyed by the chart
/// name; its `system_prompt` is unset (chart targets carry inline templates).
pub fn compile_chart(chart: &ChartDef, entities: &[Entity]) -> Result<WorkflowConfig, ChartError> {
    // Asset → provider target map (DependencySession convention: every target
    // self-provides its own name in addition to its explicit provides list).
    let mut provider_of: HashMap<&str, &str> = HashMap::new();
    for target in &chart.targets {
        provider_of.insert(target.name.as_str(), target.name.as_str());
        for provides in &target.provides {
            provider_of.insert(provides.as_str(), target.name.as_str());
        }
    }

    let mut stages: Vec<WorkflowStage> = Vec::with_capacity(chart.targets.len());
    for target in &chart.targets {
        let bindings = bind_entities(&target.depends, entities);
        let mut depends_on: Vec<String> = Vec::new();

        for dep in &target.depends {
            match dep {
                DepSpec::Capability { name } => {
                    if let Some(provider) = provider_of.get(name.as_str()) {
                        if *provider != target.name {
                            depends_on.push((*provider).to_string());
                        }
                    } else if bindings.entity_map.contains_key(name) {
                        // Satisfied by a bound entity at runtime.
                    } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: capability '{}' for target '{}' \
                                 matches multiple entities (ambiguous)",
                                chart.name, name, target.name
                            ),
                        });
                    } else {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: capability '{}' for target '{}' \
                                 is not provided by any target or bound entity",
                                chart.name, name, target.name
                            ),
                        });
                    }
                }
                DepSpec::EntityMatch {
                    name,
                    required,
                    ..
                } => {
                    if bindings.entity_map.contains_key(name) {
                        // Satisfied by a bound entity; preamble injected at render.
                    } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: entity dep '{}' for target '{}' \
                                 matches multiple entities (ambiguous)",
                                chart.name, name, target.name
                            ),
                        });
                    } else if *required {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: required entity dep '{}' for \
                                 target '{}' matched no entity; run selection/interview first",
                                chart.name, name, target.name
                            ),
                        });
                    }
                    // Optional entity dep with no match → render without it.
                }
            }
        }

        stages.push(WorkflowStage::ChartPrompt {
            id: target.name.clone(),
            template: target.template.clone(),
            depends_on,
            essential: target.essential,
        });
    }

    verify_stage_graph(&stages)?;

    let mut workflows = HashMap::new();
    workflows.insert(
        chart.name.clone(),
        WorkflowDef {
            system_prompt: None,
            stages,
        },
    );
    Ok(WorkflowConfig { workflows })
}

/// Compile a chart into a list of executable stage components, one per
/// target, carrying the supervisor metadata (essential, rubric, edges).
///
/// This is the shared stage-building logic behind both the `PipelineGraph`
/// path (`RouterConfig::build_chart_pipeline`) and the M9 Zone-supervised
/// path (`ChartExecutionPlan::compile`) — DRY: the dependency-resolution
/// rules live here once.
///
/// The returned stages are *not* yet topologically ordered; use
/// [`topo_order`] (canonical `DependencyGraph`) for the execution order.
pub fn compile_chart_stages(
    chart: &ChartDef,
    entities: &[Entity],
    backend: &Arc<dyn ChatBackend>,
    limiter: &Arc<Limiter>,
) -> Result<Vec<CompiledTarget>, ChartError> {
    // Asset → provider target map (DependencySession convention: every target
    // self-provides its own name in addition to its explicit provides list).
    let mut provider_of: HashMap<&str, &str> = HashMap::new();
    for target in &chart.targets {
        provider_of.insert(target.name.as_str(), target.name.as_str());
        for provides in &target.provides {
            provider_of.insert(provides.as_str(), target.name.as_str());
        }
    }

    let mut out: Vec<CompiledTarget> = Vec::with_capacity(chart.targets.len());
    for target in &chart.targets {
        let bindings = bind_entities(&target.depends, entities);
        let mut upstream_ids: Vec<String> = Vec::new();
        let mut topo_depends: Vec<fluent_wvr::ArcIntern<str>> = Vec::new();

        for dep in &target.depends {
            match dep {
                DepSpec::Capability { name } => {
                    if let Some(provider) = provider_of.get(name.as_str()) {
                        if *provider != target.name {
                            upstream_ids.push((*provider).to_string());
                            topo_depends.push(fluent_wvr::ArcIntern::from(*provider));
                        }
                    } else if bindings.entity_map.contains_key(name) {
                        // Bound at runtime; preamble injected by renderer.
                    } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: capability '{}' for target '{}' \
                                 matches multiple entities (ambiguous)",
                                chart.name, name, target.name
                            ),
                        });
                    } else {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: capability '{}' for target '{}' \
                                 is not provided by any target or bound entity",
                                chart.name, name, target.name
                            ),
                        });
                    }
                }
                DepSpec::EntityMatch {
                    name,
                    required,
                    ..
                } => {
                    if bindings.entity_map.contains_key(name) {
                        // Bound at runtime.
                    } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: entity dep '{}' for target '{}' \
                                 matches multiple entities (ambiguous)",
                                chart.name, name, target.name
                            ),
                        });
                    } else if *required {
                        return Err(ChartError::Compile {
                            reason: format!(
                                "chart '{}' not fully bound: required entity dep '{}' for \
                                 target '{}' matched no entity; run selection/interview first",
                                chart.name, name, target.name
                            ),
                        });
                    }
                }
            }
        }

        // Target provides its explicit asset list + its own name.
        let mut topo_provides: Vec<fluent_wvr::ArcIntern<str>> = target
            .provides
            .iter()
            .map(|p| fluent_wvr::ArcIntern::from(p.as_str()))
            .collect();
        if !topo_provides.iter().any(|p| p.as_ref() == target.name) {
            topo_provides.push(fluent_wvr::ArcIntern::from(target.name.as_str()));
        }

        let stage: Arc<dyn Component> = Arc::new(ChartPromptStage::new(
            backend.clone(),
            limiter.clone(),
            target.name.clone(),
            chart.name.clone(),
            target.template.clone(),
            target.depends.clone(),
            upstream_ids.clone(),
            topo_depends,
            topo_provides,
        ));
        out.push(CompiledTarget {
            stage,
            name: target.name.clone(),
            essential: target.essential,
            rubric: target.rubric.clone(),
            upstream_ids,
            provides: target.provides.clone(),
        });
    }

    // Fail-fast verification before anything runs: the stage graph must be
    // acyclic and fully resolved (every upstream id is a self-provided node).
    let order = topo_order(&out)?;
    let _ = order;

    Ok(out)
}

/// Topological execution order of a compiled target list, via the canonical
/// `DependencyGraph::topo_sort` — the supervisor never re-implements graph
/// algorithms. Each node self-provides its own id (the DependencySession
/// convention), so upstream edges are the only thing that matters.
pub fn topo_order(targets: &[CompiledTarget]) -> Result<Vec<String>, ChartError> {
    let mut graph: DependencyGraph<String> = DependencyGraph::new();
    for t in targets {
        graph
            .register(&t.name, &t.upstream_ids, std::slice::from_ref(&t.name))
            .map_err(|e| ChartError::Compile {
                reason: format!("stage graph invalid: {e}"),
            })?;
    }
    let unresolved = graph.unresolved_deps();
    if !unresolved.is_empty() {
        return Err(ChartError::Compile {
            reason: format!("unresolved stage deps: {unresolved:?}"),
        });
    }
    graph.topo_sort().map_err(|e| ChartError::Compile {
        reason: format!("compiled stage graph has a cycle: {e}"),
    })
}

/// Verify the compiled stage graph: register every `chart_prompt` stage as a
/// node that self-provides its id (the DependencySession convention), check
/// that no dep is unresolved, and that a topological order exists (no cycle).
fn verify_stage_graph(stages: &[WorkflowStage]) -> Result<(), ChartError> {
    let mut graph: DependencyGraph<String> = DependencyGraph::new();
    for stage in stages {
        let WorkflowStage::ChartPrompt {
            id,
            depends_on,
            ..
        } = stage
        else {
            continue;
        };
        graph
            .register(id, depends_on, std::slice::from_ref(id))
            .map_err(|e| ChartError::Compile {
                reason: e.to_string(),
            })?;
    }

    let unresolved = graph.unresolved_deps();
    if !unresolved.is_empty() {
        return Err(ChartError::Compile {
            reason: format!("unresolved stage deps: {unresolved:?}"),
        });
    }
    graph.topo_sort().map_err(|e| ChartError::Compile {
        reason: format!("compiled stage graph has a cycle: {e}"),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_chart() -> ChartDef {
        serde_json::from_str(
            r#"{
                "name": "bug_triage",
                "description": "triage",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    {
                        "name": "reproduce",
                        "provides": ["repro_plan"],
                        "depends": [],
                        "template": "reproduce {{ request }}",
                        "essential": true
                    },
                    {
                        "name": "root_cause",
                        "provides": ["root_cause"],
                        "depends": [
                            { "kind": "capability", "name": "repro_plan" }
                        ],
                        "template": "cause {{ upstream.reproduce.output }}",
                        "essential": true
                    },
                    {
                        "name": "fix_plan",
                        "provides": ["fix_plan"],
                        "depends": [
                            { "kind": "capability", "name": "root_cause" },
                            { "kind": "entity_match", "name": "report",
                              "description": "the report",
                              "predicate": {
                                "fields": [
                                    { "path": "title", "ty": "string", "required": true }
                                ]
                              },
                              "required": true }
                        ],
                        "template": "fix {{ upstream.root_cause.output }}",
                        "essential": true
                    }
                ]
            }"#,
        )
        .expect("linear chart JSON")
    }

    fn report_entity() -> Entity {
        Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on startup"}),
        }
    }

    fn only_chart_stages(cfg: &WorkflowConfig) -> Vec<&WorkflowStage> {
        cfg.workflows
            .values()
            .next()
            .expect("one workflow")
            .stages
            .iter()
            .filter(|s| matches!(s, WorkflowStage::ChartPrompt { .. }))
            .collect()
    }

    #[test]
    fn linear_chart_compiles_with_dep_edges() {
        let cfg = compile_chart(&linear_chart(), &[report_entity()]).expect("compiles");
        let stages = only_chart_stages(&cfg);
        assert_eq!(stages.len(), 3);

        match stages[0] {
            WorkflowStage::ChartPrompt {
                id,
                depends_on,
                essential,
                ..
            } => {
                assert_eq!(id, "reproduce");
                assert!(depends_on.is_empty());
                assert!(*essential);
            }
            _ => unreachable!(),
        }
        match stages[1] {
            WorkflowStage::ChartPrompt { depends_on, .. } => {
                assert_eq!(depends_on, &vec!["reproduce".to_string()]);
            }
            _ => unreachable!(),
        }
        match stages[2] {
            WorkflowStage::ChartPrompt {
                depends_on,
                essential,
                ..
            } => {
                // Capability root_cause → edge to root_cause; entity dep → no edge.
                assert_eq!(depends_on, &vec!["root_cause".to_string()]);
                assert!(*essential);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn diamond_chart_compiles() {
        let chart: ChartDef = serde_json::from_str(
            r#"{
                "name": "diamond",
                "description": "diamond",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    { "name": "base", "provides": ["base_out"], "depends": [],
                      "template": "base", "essential": true },
                    { "name": "left", "provides": ["left_out"], "depends": [
                        { "kind": "capability", "name": "base_out" }
                      ], "template": "left", "essential": true },
                    { "name": "right", "provides": ["right_out"], "depends": [
                        { "kind": "capability", "name": "base_out" }
                      ], "template": "right", "essential": true },
                    { "name": "join", "provides": ["join_out"], "depends": [
                        { "kind": "capability", "name": "left_out" },
                        { "kind": "capability", "name": "right_out" }
                      ], "template": "join", "essential": true }
                ]
            }"#,
        )
        .expect("diamond chart JSON");

        let cfg = compile_chart(&chart, &[]).expect("compiles");
        let stages = only_chart_stages(&cfg);
        assert_eq!(stages.len(), 4);
        match stages[3] {
            WorkflowStage::ChartPrompt { depends_on, .. } => {
                assert_eq!(depends_on, &vec!["left".to_string(), "right".to_string()]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn unmatched_required_entity_dep_is_compile_error() {
        // fix_plan requires a "report" entity; none provided.
        let err = compile_chart(&linear_chart(), &[]).unwrap_err();
        assert!(
            matches!(&err, ChartError::Compile { reason } if reason.contains("not fully bound")),
            "expected compile error, got: {err}"
        );
    }

    #[test]
    fn ambiguous_binding_is_compile_error() {
        let chart = linear_chart();
        let two = vec![report_entity(), report_entity()];
        let err = compile_chart(&chart, &two).unwrap_err();
        assert!(
            matches!(&err, ChartError::Compile { reason } if reason.contains("ambiguous")),
            "expected ambiguous compile error, got: {err}"
        );
    }

    #[test]
    fn cycle_in_capability_deps_is_compile_error() {
        let chart: ChartDef = serde_json::from_str(
            r#"{
                "name": "cycle",
                "description": "cycle",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    { "name": "a", "provides": ["a_out"], "depends": [
                        { "kind": "capability", "name": "b_out" }
                      ], "template": "a", "essential": true },
                    { "name": "b", "provides": ["b_out"], "depends": [
                        { "kind": "capability", "name": "a_out" }
                      ], "template": "b", "essential": true }
                ]
            }"#,
        )
        .expect("cycle chart JSON");

        let err = compile_chart(&chart, &[]).unwrap_err();
        assert!(
            matches!(&err, ChartError::Compile { reason } if reason.contains("cycle")),
            "expected cycle compile error, got: {err}"
        );
    }
}
