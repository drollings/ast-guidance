//! Chart compiler — turns a validated, fully-bound `ChartDef` into a list of
//! executable stage components (`CompiledTarget`s) ready for the SupervisedBatch-supervised
//! `ChartExecutionPlan` (the single chart executor).
//!
//! Compilation is deterministic and fail-fast: a chart whose deps are not
//! fully satisfiable (unmatched required dep, or ambiguous binding) returns
//! `ChartError::Compile` instead of producing a partially-runnable graph.
//! The compiled stage graph is verified with `DependencyGraph` (cycle +
//! unresolved-deps check, via [`topo_order`]) before being returned.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_dag::dep_graph::DependencyGraph;
use fluent_llm::client::ChatBackend;
use fluent_wvr::prelude::Component;

use crate::charts::stage::ChartPromptStage;

use super::binding::{bind_entities, Bindings, Entity};
use super::{ChartDef, ChartError, ChartRubric, ChartTarget, DepSpec};

/// A compiled chart target: the executable stage plus the metadata the
/// supervisor needs — the `essential` flag, the acceptance `rubric`,
/// and the upstream stage ids it reads from.
///
/// Produced by [`compile_chart_stages`]; consumed by the SupervisedBatch supervisor
/// (`ChartExecutionPlan`, the single chart executor).
#[derive(Clone)]
pub struct CompiledTarget {
    /// The executable `ChartPromptStage` (also a `Component`).
    pub stage: Arc<dyn Component>,
    /// Target name == stage id == self-provided asset.
    pub name: String,
    /// `essential` flag — a failed essential target fails the whole chart.
    pub essential: bool,
    /// Acceptance rubric gating this target's output. `None` = no gate.
    pub rubric: Option<ChartRubric>,
    /// Upstream chart-target stage ids this target's output depends on.
    pub upstream_ids: Vec<String>,
}

/// Asset → provider target name map (DependencySession convention: every
/// target self-provides its own name in addition to its explicit provides
/// list).
fn provider_of(chart: &ChartDef) -> HashMap<&str, &str> {
    let mut provider_of: HashMap<&str, &str> = HashMap::new();
    for target in &chart.targets {
        provider_of.insert(target.name.as_str(), target.name.as_str());
        for provides in &target.provides {
            provider_of.insert(provides.as_str(), target.name.as_str());
        }
    }
    provider_of
}

/// Resolve one target's `DepSpec`s against its bindings: returns the provider
/// stage ids this target depends on (an empty vec when every dep is bound by
/// a context entity), or a `ChartError::Compile` for an unsatisfiable dep.
///
/// The error branches are byte-for-byte the historical compile messages — the
/// behavior contract is locked by the `compile_chart_stages` tests.
fn resolve_target_deps(
    target: &ChartTarget,
    bindings: &Bindings,
    provider_of: &HashMap<&str, &str>,
    chart_name: &str,
) -> Result<Vec<String>, ChartError> {
    let mut provider_ids: Vec<String> = Vec::new();

    for dep in &target.depends {
        match dep {
            DepSpec::Capability { name } => {
                if let Some(provider) = provider_of.get(name.as_str()) {
                    if *provider != target.name {
                        provider_ids.push((*provider).to_string());
                    }
                } else if bindings.entity_map.contains_key(name) {
                    // Satisfied by a bound entity at runtime.
                } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                    return Err(ChartError::Compile {
                        reason: format!(
                            "chart '{}' not fully bound: capability '{}' for target '{}' \
                             matches multiple entities (ambiguous)",
                            chart_name, name, target.name
                        ),
                    });
                } else {
                    return Err(ChartError::Compile {
                        reason: format!(
                            "chart '{}' not fully bound: capability '{}' for target '{}' \
                             is not provided by any target or bound entity",
                            chart_name, name, target.name
                        ),
                    });
                }
            }
            DepSpec::EntityMatch { name, required, .. } => {
                if bindings.entity_map.contains_key(name) {
                    // Satisfied by a bound entity; preamble injected at render.
                } else if bindings.ambiguous.iter().any(|a| a.dep == *name) {
                    return Err(ChartError::Compile {
                        reason: format!(
                            "chart '{}' not fully bound: entity dep '{}' for target '{}' \
                             matches multiple entities (ambiguous)",
                            chart_name, name, target.name
                        ),
                    });
                } else if *required {
                    return Err(ChartError::Compile {
                        reason: format!(
                            "chart '{}' not fully bound: required entity dep '{}' for \
                             target '{}' matched no entity; run selection/interview first",
                            chart_name, name, target.name
                        ),
                    });
                }
                // Optional entity dep with no match → render without it.
            }
        }
    }

    Ok(provider_ids)
}

/// Compile a chart into a list of executable stage components, one per
/// target, carrying the supervisor metadata (essential, rubric, edges).
///
/// The dependency-resolution rules live here once, and are consumed by the
/// SupervisedBatch-supervised `ChartExecutionPlan::compile` — DRY.
///
/// The second return value is the topological execution order (canonical
/// `DependencyGraph` via [`topo_order`]), computed and validated here so
/// callers never re-derive it — a broken graph fails before anything runs.
pub fn compile_chart_stages(
    chart: &ChartDef,
    entities: &[Entity],
    backend: &Arc<dyn ChatBackend>,
    limiter: &Arc<Limiter>,
) -> Result<(Vec<CompiledTarget>, Vec<String>), ChartError> {
    let provider_of = provider_of(chart);

    let mut out: Vec<CompiledTarget> = Vec::with_capacity(chart.targets.len());
    for target in &chart.targets {
        let bindings = bind_entities(&target.depends, entities);
        let upstream_ids = resolve_target_deps(target, &bindings, &provider_of, &chart.name)?;
        let topo_depends: Vec<fluent_wvr::ArcIntern<str>> = upstream_ids
            .iter()
            .map(|id| fluent_wvr::ArcIntern::from(id.as_str()))
            .collect();

        // Target provides its explicit asset list + its own name.
        let mut topo_provides: Vec<fluent_wvr::ArcIntern<str>> = target
            .provides
            .iter()
            .map(|p| fluent_wvr::ArcIntern::from(p.as_str()))
            .collect();
        if !topo_provides.iter().any(|p| p.as_ref() == target.name) {
            topo_provides.push(fluent_wvr::ArcIntern::from(target.name.as_str()));
        }

        // Capability asset names this target consumes that the chart's own
        // targets provide in-graph: the runtime re-bind in the stage
        // must not fail-closed on these — their input is the upstream
        // target's `stage.{id}.output`, not a context entity.
        let graph_satisfied: Vec<String> = target
            .depends
            .iter()
            .filter_map(|dep| match dep {
                DepSpec::Capability { name }
                    if {
                        provider_of
                            .get(name.as_str())
                            .is_some_and(|p| *p != target.name)
                    } =>
                {
                    Some(name.clone())
                }
                _ => None,
            })
            .collect();

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
            graph_satisfied,
        ));
        out.push(CompiledTarget {
            stage,
            name: target.name.clone(),
            essential: target.essential,
            rubric: target.rubric.clone(),
            upstream_ids,
        });
    }

    // Fail-fast verification before anything runs: the stage graph must be
    // acyclic and fully resolved (every upstream id is a self-provided node).
    // The single topo pass both validates and yields the execution order —
    // callers reuse it rather than re-sorting.
    let order = topo_order(&out)?;

    Ok((out, order))
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

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr::prelude::WorkContext;

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

    fn test_backend() -> Arc<dyn ChatBackend> {
        Arc::new(crate::test_stubs::StubChatBackend::always("{}"))
    }

    fn test_limiter() -> Arc<Limiter> {
        Arc::new(Limiter::new(4))
    }

    #[test]
    fn linear_chart_compiles_with_dep_edges() {
        let (targets, order) = compile_chart_stages(
            &linear_chart(),
            &[report_entity()],
            &test_backend(),
            &test_limiter(),
        )
        .expect("compiles");
        assert_eq!(targets.len(), 3);
        assert_eq!(order, vec!["reproduce", "root_cause", "fix_plan"]);

        assert!(targets[0].upstream_ids.is_empty());
        assert_eq!(targets[1].upstream_ids, vec!["reproduce".to_string()]);
        // Capability root_cause → edge to root_cause; entity dep → no edge.
        assert_eq!(targets[2].upstream_ids, vec!["root_cause".to_string()]);
        assert!(targets.iter().all(|t| t.essential));
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

        let (targets, order) =
            compile_chart_stages(&chart, &[], &test_backend(), &test_limiter()).expect("compiles");
        assert_eq!(targets.len(), 4);
        assert_eq!(
            order,
            vec![
                "base".to_string(),
                "left".to_string(),
                "right".to_string(),
                "join".to_string()
            ]
        );
        assert_eq!(
            targets[3].upstream_ids,
            vec!["left".to_string(), "right".to_string()]
        );
    }

    #[test]
    fn unmatched_required_entity_dep_is_compile_error() {
        // fix_plan requires a "report" entity; none provided.
        let result = compile_chart_stages(&linear_chart(), &[], &test_backend(), &test_limiter());
        assert!(
            matches!(result, Err(ChartError::Compile { reason }) if reason.contains("not fully bound")),
            "expected compile error for unbound chart"
        );
    }

    #[test]
    fn ambiguous_binding_is_compile_error() {
        let chart = linear_chart();
        let two = vec![report_entity(), report_entity()];
        let result = compile_chart_stages(&chart, &two, &test_backend(), &test_limiter());
        assert!(
            matches!(result, Err(ChartError::Compile { reason }) if reason.contains("ambiguous")),
            "expected ambiguous compile error"
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

        let result = compile_chart_stages(&chart, &[], &test_backend(), &test_limiter());
        assert!(
            matches!(result, Err(ChartError::Compile { reason }) if reason.contains("cycle")),
            "expected cycle compile error"
        );
    }

    #[test]
    fn optional_entity_dep_unbound_compiles_without_edge() {
        // A non-required entity dep with no match renders without it — no
        // compile error, no stage edge.
        let chart: ChartDef = serde_json::from_str(
            r#"{
                "name": "opt",
                "description": "optional entity dep",
                "schema_version": 1,
                "author_model": "human",
                "targets": [
                    { "name": "t", "provides": ["t_out"], "depends": [
                        { "kind": "entity_match", "name": "note",
                          "description": "the note",
                          "predicate": {
                            "fields": [
                              { "path": "note_title", "ty": "string", "required": true }
                            ]
                          },
                          "required": false }
                    ], "template": "t", "essential": true }
                ]
            }"#,
        )
        .expect("optional chart JSON");

        let (targets, order) =
            compile_chart_stages(&chart, &[], &test_backend(), &test_limiter()).expect("compiles");
        assert_eq!(targets.len(), 1);
        assert!(targets[0].upstream_ids.is_empty());
        assert_eq!(order, vec!["t".to_string()]);
    }

    /// End-to-end (single chart path): `compile_chart_stages` output
    /// feeds `ChartExecutionPlan`, which executes under SupervisedBatch supervision and
    /// yields the compiled order + a completed summary.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn compiled_stages_feed_chart_execution_plan() {
        use crate::charts::execute::ChartExecOptions;
        use crate::charts::store::chart_from_str;

        let chart_json = r#"{
            "name": "linear",
            "description": "linear chain",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "a", "provides": ["a_out"], "depends": [],
                  "template": "a {{ request }}", "essential": true },
                { "name": "b", "provides": ["b_out"], "depends": [
                    { "kind": "capability", "name": "a_out" }
                  ], "template": "b {{ upstream.a.output }}", "essential": true }
            ]
        }"#;
        let chart = chart_from_str(chart_json).expect("chart parses");
        let backend: Arc<dyn ChatBackend> =
            Arc::new(crate::test_stubs::StubChatBackend::new(vec![
                r#"{"out": "a"}"#.into(),
                r#"{"out": "b"}"#.into(),
            ]));

        let (targets, order) =
            compile_chart_stages(&chart, &[], &backend, &test_limiter()).expect("compiles");
        assert_eq!(order, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(targets.len(), 2);

        let plan = crate::charts::execute::ChartExecutionPlan::compile(
            &chart,
            &[],
            &backend,
            &test_limiter(),
        )
        .expect("plan compiles");
        assert_eq!(plan.order(), &order);

        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({"model": "test", "messages": [{"role": "user", "content": "run"}]}),
        );
        let opts = ChartExecOptions {
            runtime: fluent_concurrency::tokio_runtime(),
            ..ChartExecOptions::default()
        };
        let summary = plan.execute(&ctx, &opts).await.expect("executes");
        assert_eq!(summary.completed.len(), 2);
        assert!(summary.failed.is_empty());
        assert!(summary.accepted);
        assert_eq!(summary.final_output, Some(serde_json::json!({"out": "b"})));
    }
}
