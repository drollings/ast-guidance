//! SupervisedBatch-supervised execution of a compiled chart.
//!
//! The supervisor runs a compiled chart's targets through a `SupervisedBatch` in
//! topo-order waves:
//!
//! - **Ordering authority**: `compile_chart_stages` + `topo_order` (the
//!   canonical `DependencyGraph::topo_sort`). The executor never re-implements
//!   graph algorithms — it only walks the already-computed order in waves.
//! - **Supervision**: each target runs under the SupervisedBatch's timeout/retry
//!   contract (`WorkContext::max_retries` / `timeout_ms`). A failed or
//!   rubric-rejected target cancels its transitive dependents — independent
//!   branches keep running (VISION: contain, don't restart; local-first).
//! - **Rubric gate**: each target's output is gated by its `rubric`
//!   (deterministic field-presence rule first; optional LLM judge only when
//!   the rubric says so) before promotion to `provides`.
//! - **Observability**: every target is wrapped in `Instrumented::with_metrics`
//!   (the canonical latency surface) and each run emits structured audit
//!   entries (chart, target, fit, score, verdict) through
//!   `crate::audit::emit` (`kind = "chart_target"` / `"chart_summary"`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use common_core::metrics::LatencyHistogram;
use fluent_concurrency::pool::Limiter;
use fluent_concurrency::batch::{SupervisedBatch, SupervisedBatchConfig, SupervisedBatchEvent};
use fluent_dag::dep_graph::DependencyGraph;
use fluent_llm::client::ChatBackend;
use fluent_wvr::prelude::*;
use fluent_wvr::Runtime;
use serde::Serialize;

use crate::charts::compile::{compile_chart_stages, CompiledTarget};
use crate::charts::rubric::{check_rubric, RubricCache};
use crate::charts::stage::{CHART_OUTPUT_META_KEY, CHART_TARGET_META_KEY};
use crate::charts::store::ChartStore;
use crate::charts::{ChartDef, ChartError, ChartRubric};
use crate::pipeline_types::StageDecision;

use super::binding::Entity;

/// Per-target verdict recorded in the audit trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChartTargetVerdict {
    Completed,
    Failed,
    RubricRejected,
    Cancelled,
}

/// One audit-trail entry for a chart target run (observability).
#[derive(Debug, Clone, Serialize)]
pub struct ChartAuditEntry {
    pub chart: String,
    pub target: String,
    pub fit: Option<String>,
    pub score: Option<f64>,
    pub verdict: ChartTargetVerdict,
    pub reason: String,
}

impl ChartAuditEntry {
    fn emit(&self) {
        crate::audit::emit(
            "chart_target",
            serde_json::json!({
                "chart": self.chart,
                "chart_target": self.target,
                "fit": self.fit,
                "score": self.score,
                "verdict": self.verdict,
                "reason": self.reason,
            }),
        );
    }
}

/// Chart targets wrap their LLM-call failures in `WorkError::Execution`
/// (`"chart target '…' LLM call failed: …"`, `charts/stage.rs`). Those are
/// genuinely transient (rate limits, upstream hiccups) and recoverable when
/// the caller opts in with `ChartExecOptions.max_retries`. Render/binding
/// failures (`unmatched deps`, `ambiguous deps`, template errors) are
/// permanent. So: retry the LLM-call class plus the standard transient
/// `WorkError`s, but never a permanent `Execution` failure.
fn chart_retry_predicate(err: &WorkError) -> bool {
    match err {
        WorkError::Execution(msg) => msg.contains("LLM call failed"),
        other => other.is_retryable(),
    }
}

/// Options for a supervised chart execution.
#[derive(Clone)]
pub struct ChartExecOptions {
    /// Async runtime driving the SupervisedBatch (production: `fluent_concurrency::tokio_runtime()`).
    pub runtime: Arc<dyn Runtime>,
    /// LLM judge backend for rubrics that declare `judge_model`. `None`
    /// degrades a judge-declaring rubric to the deterministic gate.
    pub judge: Option<Arc<dyn ChatBackend>>,
    /// Validated rubric/answer cache (deduplicates judge calls across runs).
    pub cache: Option<Arc<RubricCache>>,
    /// Optional latency histogram recorded per target via
    /// `Instrumented::with_metrics` (the canonical latency surface).
    pub metrics: Option<Arc<LatencyHistogram>>,
    /// Per-attempt retries (0 = none) — the SupervisedBatch's `WorkContext.max_retries`.
    pub max_retries: u32,
    /// Per-attempt wall-clock budget in ms (0 = none) — `WorkContext.timeout_ms`.
    pub timeout_ms: u64,
    /// Selection provenance for the audit trail (e.g. `"exact"` / `"partial"`).
    pub fit: Option<String>,
    /// Selection confidence for the audit trail.
    pub score: Option<f64>,
    /// When set, the rubric-gate result is recorded against the store —
    /// consecutive failures demote the chart, a pass promotes a draft. 
    /// `None` disables the recording (executor stays decoupled).
    pub health: Option<Arc<ChartStore>>,
}

impl Default for ChartExecOptions {
    fn default() -> Self {
        Self {
            runtime: Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime),
            judge: None,
            cache: None,
            metrics: None,
            max_retries: 0,
            timeout_ms: 0,
            fit: None,
            score: None,
            health: None,
        }
    }
}

/// Summary of a supervised chart execution.
#[derive(Debug, Default, Clone)]
pub struct ChartExecutionSummary {
    /// Decisions of completed targets, in completion order.
    pub completed: Vec<StageDecision>,
    /// Names of targets that failed (execution error, timeout, or rubric reject).
    pub failed: Vec<String>,
    /// Names of targets cancelled because a dependency failed.
    pub cancelled: Vec<String>,
    /// The last completed target's `output` (the chart result), if any.
    pub final_output: Option<serde_json::Value>,
    /// Per-target audit entries.
    pub audit: Vec<ChartAuditEntry>,
    /// Whether the whole chart is accepted: no essential target failed and
    /// the chart-level rubric (if any) passed.
    pub accepted: bool,
}

impl ChartExecutionSummary {
    /// Whether this run was rejected by a rubric gate (target- or chart-level).
    /// The staleness policy counts only rubric rejections as "stale
    /// failures" — an execution error is a different failure class.
    pub fn rubric_rejected(&self) -> bool {
        self.audit
            .iter()
            .any(|e| e.verdict == ChartTargetVerdict::RubricRejected)
    }
}

/// A compiled, runnable chart under SupervisedBatch supervision.
pub struct ChartExecutionPlan {
    chart_name: String,
    chart_rubric: Option<ChartRubric>,
    targets: Vec<CompiledTarget>,
    order: Vec<String>,
}

impl ChartExecutionPlan {
    /// Compile a bound chart into a supervised execution plan.
    ///
    /// Uses the same stage construction + dependency resolution as the
    /// `PipelineGraph` path (`compile_chart_stages`) and verifies the topo
    /// order up front (fail fast — a broken graph never runs). The topo
    /// order is computed once by `compile_chart_stages` and reused here.
    pub fn compile(
        chart: &ChartDef,
        entities: &[Entity],
        backend: &Arc<dyn ChatBackend>,
        limiter: &Arc<Limiter>,
    ) -> Result<Self, ChartError> {
        let (targets, order) = compile_chart_stages(chart, entities, backend, limiter)?;
        Ok(Self {
            chart_name: chart.name.clone(),
            chart_rubric: chart.rubric.clone(),
            targets,
            order,
        })
    }

    /// The chart name (provenance for audit).
    pub fn chart_name(&self) -> &str {
        &self.chart_name
    }

    /// The planned execution order (stage ids, canonical topo order).
    pub fn order(&self) -> &[String] {
        &self.order
    }

    /// Run the chart under SupervisedBatch supervision, in topo-order waves.
    ///
    /// Each wave registers every ready (deps satisfied) target into a fresh
    /// `SupervisedBatch` with its context carrying the accumulated `stage.{id}.*`
    /// metadata; the SupervisedBatch enforces timeout/retry per target. Completed
    /// outputs are rubric-gated before promotion; a failed/rubric-rejected
    /// target's dependents never become ready (they land in `cancelled`).
    ///
    /// `base_ctx` must carry the request (`request`) and, when the chart has
    /// entity deps, the bound entities (structured `entities`).
    pub async fn execute(
        &self,
        base_ctx: &WorkContext,
        opts: &ChartExecOptions,
    ) -> Result<ChartExecutionSummary, ChartError> {
        let mut summary = ChartExecutionSummary::default();
        let mut completed: HashMap<String, StageDecision> = HashMap::new();
        let mut failed: HashSet<String> = HashSet::new();

        // The dependency graph is the single source of readiness and ordering —
        // it mirrors the topo graph built at compile time (`compile::topo_order`:
        // each target registers its name, its upstream ids as deps, and
        // provides its own name (the DependencySession convention). The executor
        // never re-implements a ready scan by hand.
        let mut graph: DependencyGraph<String> = DependencyGraph::new();
        for t in &self.targets {
            graph
                .register(&t.name, &t.upstream_ids, std::slice::from_ref(&t.name))
                .map_err(|e| ChartError::Compile {
                    reason: format!("stage graph invalid: {e}"),
                })?;
        }
        // name → target lookup for resolving ready node names back to stages.
        let by_name: HashMap<&str, &CompiledTarget> =
            self.targets.iter().map(|t| (t.name.as_str(), t)).collect();

        // The topo order guarantees every upstream id precedes its dependents,
        // so wave iteration terminates: each wave completes ≥1 target.
        loop {
            // Ready = registered nodes whose deps are all satisfied (the
            // canonical `ready_nodes` — one inverted-index scan per wave, the
            // same cost as the per-target `.all()` filter it replaces),
            // excluding targets already completed or failed.
            let satisfied: HashSet<String> = completed.keys().cloned().collect();
            let ready: Vec<&CompiledTarget> = graph
                .ready_nodes(&satisfied)
                .into_iter()
                .filter(|name| !completed.contains_key(name) && !failed.contains(name))
                .filter_map(|name| by_name.get(name.as_str()).copied())
                .collect();
            if ready.is_empty() {
                break;
            }

            let mut batch = SupervisedBatch::new_with_config(
                opts.runtime.clone(),
                CapabilitySet::default(),
                SupervisedBatchConfig {
                    is_retryable: chart_retry_predicate,
                    ..SupervisedBatchConfig::default()
                },
            );
            for target in &ready {
                let mut ctx = build_target_context(base_ctx, target, &completed);
                ctx.max_retries = opts.max_retries;
                ctx.timeout_ms = opts.timeout_ms;
                ctx.rt = opts.runtime.clone();
                ctx.caps = CapabilitySet::default();

                let wrapped: Arc<dyn Component> = match &opts.metrics {
                    Some(hist) => Arc::new(Instrumented::with_metrics(
                        target.stage.clone(),
                        format!("chart.{}.{}", self.chart_name, target.name),
                        hist.clone(),
                    )),
                    None => Arc::new(Instrumented::new(
                        target.stage.clone(),
                        format!("chart.{}.{}", self.chart_name, target.name),
                    )),
                };
                batch.register_with_context(wrapped, ctx)
                    .map_err(|e| ChartError::Compile {
                        reason: format!("register chart target '{}': {e}", target.name),
                    })?;
            }

            let batch_summary = batch.await;

            // Each batch event: promote or fail, gated by the target's rubric.
            for event in batch_summary
                .completed
                .into_iter()
                .chain(batch_summary.panicked)
                .chain(batch_summary.failed)
                .chain(batch_summary.cancelled)
            {
                self.process_event(event, opts, &mut completed, &mut failed, &mut summary)?;
            }

            // A failed *essential* target fails the whole chart — stop
            // scheduling further waves.
            let any_essential_failed = failed
                .iter()
                .any(|name| self.targets.iter().any(|t| t.name == *name && t.essential));
            if any_essential_failed {
                break;
            }
        }

        // Targets neither completed nor failed were cancelled: their
        // dependency chain never completed (a dependency failed), or an
        // essential-failure abort left them ready-but-unexecuted. A single
        // "not completed and not failed" pass classifies every target — the
        // plain sweep. Note: this is intentionally NOT
        // `graph.dependents_of(failed)` alone — an abort can strand a ready
        // target in an *independent* branch (its deps all completed, so it is
        // not a transitive dependent of the failed target) that the sweep
        // still cancels; `dependents_of` is a proper subset.
        let mut cancelled: Vec<String> = Vec::new();
        for t in &self.targets {
            if completed.contains_key(&t.name) {
                summary.completed.push(completed[&t.name].clone());
                continue;
            }
            if failed.contains(&t.name) {
                summary.failed.push(t.name.clone());
            } else {
                cancelled.push(t.name.clone());
                summary.cancelled.push(t.name.clone());
                let entry = ChartAuditEntry {
                    chart: self.chart_name.clone(),
                    target: t.name.clone(),
                    fit: opts.fit.clone(),
                    score: opts.score,
                    verdict: ChartTargetVerdict::Cancelled,
                    reason: "dependency failed".into(),
                };
                entry.emit();
                summary.audit.push(entry);
            }
        }

        // Chart-level rubric gates the final output.
        summary.final_output = summary
            .completed
            .last()
            .and_then(|d| d.metadata.get(CHART_OUTPUT_META_KEY).cloned());
        let mut accepted = summary.failed.is_empty() && summary.cancelled.is_empty();
        if let Some(rubric) = &self.chart_rubric {
            if let Some(final_output) = summary.final_output.clone() {
                let verdict = check_rubric(
                    rubric,
                    &final_output,
                    opts.judge.as_ref(),
                    opts.cache.as_deref(),
                    &self.chart_name,
                )
                .map_err(|e| ChartError::Selection {
                    reason: format!("chart rubric gate failed: {e}"),
                })?;
                accepted = accepted && verdict.accepted;
                if !verdict.accepted {
                    let entry = ChartAuditEntry {
                        chart: self.chart_name.clone(),
                        target: "<chart>".into(),
                        fit: opts.fit.clone(),
                        score: opts.score,
                        verdict: ChartTargetVerdict::RubricRejected,
                        reason: verdict.reason,
                    };
                    entry.emit();
                    summary.audit.push(entry);
                }
            }
        }
        summary.accepted = accepted;

        // Staleness: feed the rubric-gate result to the store so
        // consecutive failures demote the chart and a pass promotes a draft.
        if let Some(store) = &opts.health {
            if store.get(&self.chart_name).is_some() {
                store.record_rubric_result(&self.chart_name, !summary.rubric_rejected());
            }
        }

        crate::audit::emit(
            "chart_summary",
            serde_json::json!({
                "chart": self.chart_name,
                "fit": opts.fit,
                "score": opts.score,
                "completed": summary.completed.len(),
                "failed": summary.failed.len(),
                "cancelled": summary.cancelled.len(),
                "accepted": accepted,
            }),
        );

        Ok(summary)
    }

    /// Fold a single `BatchEvent` into the execution state.
    fn process_event(
        &self,
        event: SupervisedBatchEvent,
        opts: &ChartExecOptions,
        completed: &mut HashMap<String, StageDecision>,
        failed: &mut HashSet<String>,
        summary: &mut ChartExecutionSummary,
    ) -> Result<(), ChartError> {
        match event {
            SupervisedBatchEvent::Completed { name, output } => {
                let decision: StageDecision =
                    output.data_as().map_err(|e| ChartError::Compile {
                        reason: format!("chart target '{name}' output unreadable: {e}"),
                    })?;
                let Some(target) = self.targets.iter().find(|t| t.name == *name) else {
                    // Unknown task — nothing to gate or promote.
                    return Ok(());
                };
                let output_val = decision
                    .metadata
                    .get(CHART_OUTPUT_META_KEY)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                if let Some(rubric) = &target.rubric {
                    let verdict = check_rubric(
                        rubric,
                        &output_val,
                        opts.judge.as_ref(),
                        opts.cache.as_deref(),
                        &name,
                    )
                    .map_err(|e| ChartError::Selection {
                        reason: format!("rubric gate for target '{name}' failed: {e}"),
                    })?;
                    if !verdict.accepted {
                        failed.insert(name.to_string());
                        let entry = ChartAuditEntry {
                            chart: self.chart_name.clone(),
                            target: name.to_string(),
                            fit: opts.fit.clone(),
                            score: opts.score,
                            verdict: ChartTargetVerdict::RubricRejected,
                            reason: verdict.reason,
                        };
                        entry.emit();
                        summary.audit.push(entry);
                        return Ok(());
                    }
                }

                crate::audit::emit(
                    "chart_target",
                    serde_json::json!({
                        "chart": self.chart_name,
                        "chart_target": name,
                        "fit": opts.fit,
                        "score": opts.score,
                        "verdict": ChartTargetVerdict::Completed,
                        "reason": decision.reason,
                    }),
                );
                summary.audit.push(ChartAuditEntry {
                    chart: self.chart_name.clone(),
                    target: name.to_string(),
                    fit: opts.fit.clone(),
                    score: opts.score,
                    verdict: ChartTargetVerdict::Completed,
                    reason: decision.reason.clone(),
                });
                completed.insert(name.to_string(), decision);
                Ok(())
            }
            SupervisedBatchEvent::Failed { name, error } => {
                failed.insert(name.to_string());
                let entry = ChartAuditEntry {
                    chart: self.chart_name.clone(),
                    target: name.to_string(),
                    fit: opts.fit.clone(),
                    score: opts.score,
                    verdict: ChartTargetVerdict::Failed,
                    reason: error.to_string(),
                };
                entry.emit();
                summary.audit.push(entry);
                Ok(())
            }
            SupervisedBatchEvent::Panicked { name, info } => {
                failed.insert(name.to_string());
                let entry = ChartAuditEntry {
                    chart: self.chart_name.clone(),
                    target: name.to_string(),
                    fit: opts.fit.clone(),
                    score: opts.score,
                    verdict: ChartTargetVerdict::Failed,
                    reason: info,
                };
                entry.emit();
                summary.audit.push(entry);
                Ok(())
            }
            SupervisedBatchEvent::Cancelled { name, .. } => {
                // Cancelled within a wave (e.g. timeout) → treated as failed
                // so dependents never become ready.
                failed.insert(name.to_string());
                let entry = ChartAuditEntry {
                    chart: self.chart_name.clone(),
                    target: name.to_string(),
                    fit: opts.fit.clone(),
                    score: opts.score,
                    verdict: ChartTargetVerdict::Cancelled,
                    reason: "cancelled by supervisor".into(),
                };
                entry.emit();
                summary.audit.push(entry);
                Ok(())
            }
        }
    }
}

/// Build a target's execution context: the base request/entities plus the
/// accumulated `stage.{id}` structured metadata of all completed upstream
/// targets (raw `serde_json::Value`, one entry per upstream).
fn build_target_context(
    base: &WorkContext,
    target: &CompiledTarget,
    completed: &HashMap<String, StageDecision>,
) -> WorkContext {
    let mut ctx = base.clone();
    for (id, decision) in completed {
        ctx.structured
            .insert(format!("stage.{id}"), decision.metadata.clone());
    }
    // Make the target's own name visible as metadata for rendering/audit.
    ctx.metadata.insert(
        CHART_TARGET_META_KEY.into(),
        MetadataValue::String(target.name.clone()),
    );
    ctx
}
#[cfg(test)]
#[path = "../../tests/charts_execute.rs"]
mod tests;
