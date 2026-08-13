//! SupervisedBatch-supervised execution of a compiled chart.
//!
//! The M9 supervisor runs a compiled chart's targets through a `SupervisedBatch` in
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

/// One audit-trail entry for a chart target run (M9 observability).
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

/// SupervisedBatch retry predicate for chart zones (M5.1).
///
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
    /// M10 staleness: when set, the rubric-gate result is recorded against
    /// the store — consecutive failures demote the chart, a pass promotes a
    /// draft. `None` disables the recording (executor stays decoupled).
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
    /// The M10 staleness policy counts only rubric rejections as "stale
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
        // it mirrors the topo graph built at compile time (`compile::topo_order`,
        // F9): each target registers its name, its upstream ids as deps, and
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
        // plain sweep (M8.2). Note: this is intentionally NOT
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

        // M10 staleness: feed the rubric-gate result to the store so
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

    /// Fold a single `ZoneEvent` into the execution state.
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
mod tests {
    use super::*;
    use crate::charts::binding::Entity;
    use crate::charts::store::chart_from_str;
    use crate::test_stubs::StubChatBackend;
    use common_core::sync::lock;
    use fluent_llm::{ChatMessage, LlmError};
    use std::sync::Mutex;

    /// A deterministic test backend that keys responses on a substring of the
    /// rendered system prompt. `"__error__"` maps to `LlmError::NoResponse`.
    /// Unmatched prompts error — so a stage that never should run fails loudly.
    struct KeyedBackend {
        map: HashMap<String, String>,
    }

    impl KeyedBackend {
        fn new(entries: Vec<(String, String)>) -> Self {
            Self {
                map: entries.into_iter().collect(),
            }
        }
    }

    impl fluent_llm::client::ChatBackend for KeyedBackend {
        fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
            let system = messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            for (key, resp) in &self.map {
                if system.contains(key) {
                    if resp == "__error__" {
                        return Err(LlmError::NoResponse);
                    }
                    return Ok(resp.clone());
                }
            }
            Err(LlmError::NoResponse)
        }
    }

    /// Fails exactly the first `chat_complete` call, then succeeds forever.
    /// Exercises the SupervisedBatch's retry-with-backoff over a transient target failure.
    struct RetryOnceBackend {
        failures_left: Mutex<usize>,
        response: String,
    }

    impl RetryOnceBackend {
        fn new(response: String) -> Self {
            Self {
                failures_left: Mutex::new(1),
                response,
            }
        }
    }

    impl fluent_llm::client::ChatBackend for RetryOnceBackend {
        fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
            let mut left = lock(&self.failures_left);
            if *left > 0 {
                *left -= 1;
                return Err(LlmError::NoResponse);
            }
            Ok(self.response.clone())
        }
    }

    /// A 2-target linear chart: `a` (no deps) → `b` (depends on `a_out`).
    fn linear_chart_json() -> String {
        r#"{
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
        }"#
        .to_string()
    }

    /// A diamond chart: base → {left, right} → join. `left` is expected to fail.
    fn diamond_chart_json() -> String {
        r#"{
            "name": "diamond",
            "description": "diamond",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "base", "provides": ["base_out"], "depends": [],
                  "template": "base {{ request }}", "essential": true },
                { "name": "left", "provides": ["left_out"], "depends": [
                    { "kind": "capability", "name": "base_out" }
                  ], "template": "left {{ upstream.base.output }}", "essential": false },
                { "name": "right", "provides": ["right_out"], "depends": [
                    { "kind": "capability", "name": "base_out" }
                  ], "template": "right {{ upstream.base.output }}", "essential": false },
                { "name": "join", "provides": ["join_out"], "depends": [
                    { "kind": "capability", "name": "left_out" },
                    { "kind": "capability", "name": "right_out" }
                  ], "template": "join {{ upstream.left.output }} {{ upstream.right.output }}",
                  "essential": true }
            ]
        }"#
        .to_string()
    }

    fn make_ctx(text: &str) -> WorkContext {
        let request_json = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": text}]
        });
        let mut ctx = WorkContext::default();
        ctx.set_structured("request", &request_json);
        ctx
    }

    /// Default execution options: tokio runtime, no judge, no retries.
    fn default_opts() -> ChartExecOptions {
        ChartExecOptions {
            runtime: fluent_concurrency::tokio_runtime(),
            ..ChartExecOptions::default()
        }
    }

    fn build_plan(
        chart_json: &str,
        backend: &Arc<dyn ChatBackend>,
        entities: &[Entity],
    ) -> ChartExecutionPlan {
        let chart = chart_from_str(chart_json).expect("chart parses");
        let limiter = Arc::new(Limiter::new(4));
        ChartExecutionPlan::compile(&chart, entities, backend, &limiter).expect("compiles")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn linear_chart_completes_in_order() {
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"out": "a-done"}"#.into(),
            r#"{"out": "b-done"}"#.into(),
        ]));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        assert_eq!(plan.order(), &["a".to_string(), "b".to_string()]);

        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert_eq!(summary.completed.len(), 2);
        assert_eq!(summary.completed[0].reason, "chart target 'a' completed");
        assert_eq!(summary.completed[1].reason, "chart target 'b' completed");
        assert!(summary.failed.is_empty());
        assert!(summary.cancelled.is_empty());
        assert!(summary.accepted);
        assert_eq!(
            summary.final_output,
            Some(serde_json::json!({"out": "b-done"}))
        );
        // Audit trail has 2 completed entries (a, b).
        let completed_audits: Vec<_> = summary
            .audit
            .iter()
            .filter(|e| e.verdict == ChartTargetVerdict::Completed)
            .collect();
        assert_eq!(completed_audits.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failing_mid_chart_target_cancels_dependents_and_keeps_independent_branch() {
        // Deterministic per-stage script: base ok, left errors, right ok.
        // Keys are prefix-tagged ("base ", "left ", ...) so the rendered
        // prompts never collide (left's prompt embeds base's output, but
        // "base " with the trailing space does not match inside a JSON value).
        let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
            ("base ".to_string(), r#"{"out": "base"}"#.to_string()),
            ("left ".to_string(), "__error__".to_string()),
            ("right ".to_string(), r#"{"out": "right-done"}"#.to_string()),
            ("join ".to_string(), r#"{"out": "join"}"#.to_string()),
        ]));
        let plan = build_plan(&diamond_chart_json(), &backend, &[]);

        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");

        // base + right completed; left failed (LLM error); join cancelled.
        let names: Vec<&str> = summary
            .completed
            .iter()
            .map(|d| d.metadata["chart_target"].as_str().unwrap_or("?"))
            .collect();
        assert!(names.contains(&"base"), "base completed, got {names:?}");
        assert!(
            names.contains(&"right"),
            "independent branch survives, got {names:?}"
        );
        assert!(!names.contains(&"left"));
        assert!(!names.contains(&"join"));
        assert!(summary.failed.contains(&"left".to_string()), "left failed");
        assert!(
            summary.cancelled.contains(&"join".to_string()),
            "join cancelled: {:?}",
            summary.cancelled
        );
        // join is essential → whole chart not accepted.
        assert!(!summary.accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn essential_failure_aborts_chart() {
        // base fails (NoResponse) → essential → nothing else runs.
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert!(summary.failed.contains(&"a".to_string()));
        assert!(summary.cancelled.contains(&"b".to_string()));
        assert!(!summary.accepted);
    }

    /// An essential failure aborts the chart even when an *independent* branch
    /// has a ready-but-unexecuted target: that target is still cancelled (its
    /// deps all completed but the chart stopped), so the cancelled set is a
    /// "not completed and not failed" sweep — not merely the transitive
    /// `dependents_of` the failed essential target. Locks M8.2's classifier.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn essential_failure_cancels_independent_ready_branch() {
        let chart_json = r#"{
            "name": "split",
            "description": "two independent branches",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "a1", "provides": ["a1_out"], "depends": [],
                  "template": "a1 {{ request }}", "essential": true },
                { "name": "a2", "provides": ["a2_out"], "depends": [
                    { "kind": "capability", "name": "a1_out" }
                  ], "template": "a2 {{ upstream.a1.output }}", "essential": true },
                { "name": "b1", "provides": ["b1_out"], "depends": [],
                  "template": "b1 {{ request }}", "essential": false },
                { "name": "b2", "provides": ["b2_out"], "depends": [
                    { "kind": "capability", "name": "b1_out" }
                  ], "template": "b2 {{ upstream.b1.output }}", "essential": false }
            ]
        }"#;
        // Wave 1 schedules {a1, b1}; a1 fails (essential), b1 completes. The
        // abort leaves a2 (dependent of a1) *and* b2 (ready via b1, independent
        // of a1) both cancelled.
        let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
            ("a1 ".to_string(), "__error__".to_string()),
            ("b1 ".to_string(), r#"{"out": "b1"}"#.to_string()),
        ]));
        let plan = build_plan(chart_json, &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");

        let completed: Vec<&str> = summary
            .completed
            .iter()
            .map(|d| d.metadata["chart_target"].as_str().unwrap_or("?"))
            .collect();
        assert!(completed.contains(&"b1"), "b1 completed, got {completed:?}");
        assert!(summary.failed.contains(&"a1".to_string()), "a1 failed");
        // Both the dependent-of-the-failure (a2) and the ready-but-independent
        // (b2) land in cancelled — `dependents_of(a1)` alone would miss b2.
        assert!(
            summary.cancelled.contains(&"a2".to_string()),
            "a2 cancelled: {:?}",
            summary.cancelled
        );
        assert!(
            summary.cancelled.contains(&"b2".to_string()),
            "b2 cancelled: {:?}",
            summary.cancelled
        );
        assert!(!summary.accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zone_retry_recovers_a_transient_target_failure() {
        // `a`'s LLM call errors on the first attempt, then succeeds. With
        // max_retries = 1 the SupervisedBatch retries and the whole chain completes.
        let backend: Arc<dyn ChatBackend> =
            Arc::new(RetryOnceBackend::new(r#"{"out": "a-retried"}"#.to_string()));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        let mut opts = default_opts();
        opts.max_retries = 1;
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert_eq!(
            summary.completed.len(),
            2,
            "a recovers via SupervisedBatch retry, then b runs"
        );
        assert!(summary.failed.is_empty());
        assert!(summary.cancelled.is_empty());
        assert!(summary.accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_retries_means_transient_failure_fails_target() {
        let backend: Arc<dyn ChatBackend> =
            Arc::new(RetryOnceBackend::new(r#"{"out": "a"}"#.to_string()));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        // max_retries = 0 (default): a's first error is fatal.
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert!(summary.failed.contains(&"a".to_string()));
        assert!(summary.cancelled.contains(&"b".to_string()));
        assert!(!summary.accepted);
    }

    // ── Rubric gate ──────────────────────────────────────────────────────

    fn rubric_chart_json() -> String {
        r#"{
            "name": "gated",
            "description": "rubric-gated chart",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "probe", "provides": ["probe_out"], "depends": [],
                  "template": "probe {{ request }}", "essential": true,
                  "rubric": { "require_fields": ["answer"], "min_score": 0.7 } },
                { "name": "after", "provides": ["after_out"], "depends": [
                    { "kind": "capability", "name": "probe_out" }
                  ], "template": "after {{ upstream.probe.output }}", "essential": true }
            ]
        }"#
        .to_string()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rubric_pass_promotes_output() {
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"answer": 42}"#.into(),
            r#"{"done": true}"#.into(),
        ]));
        let plan = build_plan(&rubric_chart_json(), &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert_eq!(summary.completed.len(), 2, "probe + after both promote");
        assert!(summary.failed.is_empty());
        assert!(summary.accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rubric_fail_cancels_dependents() {
        // probe's output lacks the required `answer` field → rubric reject →
        // after never becomes ready.
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"no_answer": true}"#.into(),
            r#"{"done": true}"#.into(),
        ]));
        let plan = build_plan(&rubric_chart_json(), &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert!(summary.failed.contains(&"probe".to_string()));
        assert!(summary.cancelled.contains(&"after".to_string()));
        assert!(!summary.accepted);
        let rejected: Vec<_> = summary
            .audit
            .iter()
            .filter(|e| e.verdict == ChartTargetVerdict::RubricRejected)
            .collect();
        assert_eq!(rejected.len(), 1, "probe is rubric-rejected");
        assert_eq!(rejected[0].target, "probe");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn absent_rubric_skips_gate() {
        // No rubric on the chart → output promoted on successful execution
        // even though the field is missing.
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"only_unexpected": true}"#.into(),
            r#"{"done": true}"#.into(),
        ]));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert_eq!(summary.completed.len(), 2);
        assert!(summary.accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chart_level_rubric_gates_final_output() {
        let chart_json = r#"{
            "name": "charted",
            "description": "chart-level rubric",
            "schema_version": 1,
            "author_model": "human",
            "rubric": { "require_fields": ["final_answer"] },
            "targets": [
                { "name": "t", "provides": ["t_out"], "depends": [],
                  "template": "t {{ request }}", "essential": true }
            ]
        }"#;
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::always(r#"{"not_the_final_answer": true}"#));
        let plan = build_plan(chart_json, &backend, &[]);
        let summary = plan
            .execute(&make_ctx("run"), &default_opts())
            .await
            .expect("runs");
        assert!(!summary.accepted, "chart-level rubric rejects");
        assert!(summary.failed.is_empty(), "target itself did not fail");
        let rejected: Vec<_> = summary
            .audit
            .iter()
            .filter(|e| e.verdict == ChartTargetVerdict::RubricRejected)
            .collect();
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].target, "<chart>");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn audit_trail_records_fit_and_score() {
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"out": "a"}"#.into(),
            r#"{"out": "b"}"#.into(),
        ]));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        let mut opts = default_opts();
        opts.fit = Some("exact".into());
        opts.score = Some(0.93);
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert_eq!(summary.audit.len(), 2);
        for entry in &summary.audit {
            assert_eq!(entry.chart, "linear");
            assert_eq!(entry.fit.as_deref(), Some("exact"));
            assert_eq!(entry.score, Some(0.93));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_histogram_records_per_target() {
        let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![
            r#"{"out": "a"}"#.into(),
            r#"{"out": "b"}"#.into(),
        ]));
        let plan = build_plan(&linear_chart_json(), &backend, &[]);
        let hist = Arc::new(LatencyHistogram::new());
        let mut opts = default_opts();
        opts.metrics = Some(hist.clone());
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert_eq!(summary.completed.len(), 2);
        assert!(
            hist.count() >= 2,
            "per-target latency recorded, got {}",
            hist.count()
        );
    }

    // ── Golden e2e: real seed chart + rubric through the SupervisedBatch ─────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn golden_rubric_gated_seed_chart_runs_through_zone() {
        // Load the real Appendix A seed chart, add a target rubric, and run it
        // through the SupervisedBatch supervisor with a mock backend. The audit trail
        // must record chart/fit/score/targets.
        let seed_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/workflows/charts");
        let path = seed_dir.join("bug_triage.md.json");
        let json = std::fs::read_to_string(&path).expect("seed chart file exists");
        let mut chart = crate::charts::store::chart_from_str(&json).expect("seed chart parses");

        // Gate `root_cause` on a `cause` field being present.
        chart.targets[1].rubric = Some(crate::charts::ChartRubric {
            require_fields: vec!["cause".into()],
            judge_model: None,
            min_score: 0.7,
        });
        chart.validate().expect("rubric-gated chart validates");

        let entity = Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on startup"}),
        };
        let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![
            (
                "write a minimal reproduction plan".to_string(),
                r#"{"plan": "minimal repro"}"#.to_string(),
            ),
            (
                "identify the root cause".to_string(),
                r#"{"cause": "null pointer deref in async task"}"#.to_string(),
            ),
            (
                "Produce a fix plan".to_string(),
                r#"{"fix": "check for null before deref"}"#.to_string(),
            ),
        ]));
        let limiter = Arc::new(Limiter::new(4));
        let plan =
            ChartExecutionPlan::compile(&chart, std::slice::from_ref(&entity), &backend, &limiter)
                .expect("seed chart compiles");
        assert_eq!(plan.order().len(), 3);

        let mut opts = default_opts();
        opts.fit = Some("exact".into());
        opts.score = Some(0.99);

        // The ctx must carry both the request and the bound entities — the
        // stages re-bind from the structured `entities` at execution time.
        let mut ctx = make_ctx("app crashes on startup");
        ctx.set_structured(
            crate::charts::binding::ENTITIES_META_KEY,
            &std::slice::from_ref(&entity),
        );

        let summary = plan
            .execute(&ctx, &opts)
            .await
            .expect("seed chart executes under SupervisedBatch supervision");

        if summary.completed.len() != 3 {
            eprintln!("FAILED summary: {summary:#?}");
            panic!("seed chart did not complete 3 targets");
        }
        assert!(summary.failed.is_empty());
        assert!(summary.cancelled.is_empty());
        assert!(summary.accepted, "rubric-gated chart accepted");
        assert_eq!(summary.audit.len(), 3);
        for entry in &summary.audit {
            assert_eq!(entry.chart, "bug_triage");
            assert_eq!(entry.fit.as_deref(), Some("exact"));
            assert_eq!(entry.score, Some(0.99));
            assert!(matches!(entry.verdict, ChartTargetVerdict::Completed));
        }
        let target_names: Vec<&str> = summary.audit.iter().map(|e| e.target.as_str()).collect();
        assert_eq!(target_names, vec!["reproduce", "root_cause", "fix_plan"]);
    }

    // ── M10: staleness / demotion fed by rubric-gate results ─────────────

    /// A 1-target chart whose target rubric requires an `out` field.
    fn rubric_failing_chart_json() -> String {
        r#"{
            "name": "gated",
            "description": "rubric-gated single target",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "g", "provides": ["g_out"], "depends": [],
                  "template": "g {{ request }}", "essential": true,
                  "rubric": { "require_fields": ["out"] } }
            ]
        }"#
        .to_string()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_rubric_failures_demote_chart_in_store() {
        let store = Arc::new(ChartStore::new(None));
        store
            .upsert(chart_from_str(&rubric_failing_chart_json()).unwrap())
            .unwrap();

        // Every run returns output missing `out` → rubric-rejected → a stale
        // failure recorded against the store. `KeyedBackend` repeats the
        // response (its key "g " matches the rendered prompt each run).
        let backend: Arc<dyn ChatBackend> = Arc::new(KeyedBackend::new(vec![(
            "g ".to_string(),
            r#"{"wrong": true}"#.to_string(),
        )]));
        let plan = build_plan(&rubric_failing_chart_json(), &backend, &[]);

        for i in 0..crate::charts::CHART_STALE_FAILS {
            let mut opts = default_opts();
            opts.health = Some(store.clone());
            let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
            assert!(
                summary.rubric_rejected(),
                "run {} rejected by rubric",
                i + 1
            );
            if i + 1 < crate::charts::CHART_STALE_FAILS {
                assert!(!store.is_demoted("gated"));
            }
        }
        assert!(
            store.is_demoted("gated"),
            "crossing CHART_STALE_FAILS demotes the chart"
        );
        assert_eq!(store.demoted_charts(), vec!["gated".to_string()]);
        assert!(
            !store.charts_sorted().iter().any(|c| c.name == "gated"),
            "demoted chart is no longer selected"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn passing_run_promotes_draft_and_resets_streak() {
        // Extract the chart through the idempotent path so it is a draft.
        let store = Arc::new(ChartStore::new(None));
        let chart = chart_from_str(&rubric_failing_chart_json()).unwrap();
        store
            .upsert_idempotent(chart, crate::charts::store::CHART_SUBSUME_THRESHOLD)
            .unwrap();
        assert!(store.is_draft("gated"));

        // One stale failure, then a passing run: the streak resets and the
        // draft is promoted to selectable.
        let failing: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::always(r#"{"no_out": true}"#));
        let plan = build_plan(&rubric_failing_chart_json(), &failing, &[]);
        let mut opts = default_opts();
        opts.health = Some(store.clone());
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert!(summary.rubric_rejected());
        assert!(store.is_draft("gated"), "still a draft after a failure");

        let passing: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(r#"{"out": "g"}"#));
        let plan = build_plan(&rubric_failing_chart_json(), &passing, &[]);
        let summary = plan.execute(&make_ctx("run"), &opts).await.expect("runs");
        assert!(!summary.rubric_rejected());
        assert!(!store.is_draft("gated"), "a passing run promotes the draft");
        assert!(!store.is_demoted("gated"));
        assert!(
            store.charts_sorted().iter().any(|c| c.name == "gated"),
            "promoted chart is selectable"
        );
    }
}
