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
        if !topo_provides.iter().any(|p| {
            let pstr: &str = p.as_ref();
            pstr == target.name.as_str()
        }) {
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
#[path = "../../tests/charts_compile.rs"]
mod tests;
