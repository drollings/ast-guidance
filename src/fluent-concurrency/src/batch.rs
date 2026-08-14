//! Supervised batch runner with async retry, dependency cancellation, and timeout.
//! A `SupervisedBatch` manages a group of `WorkUnit` tasks and propagates cancellation
//! across dependent tasks when a prerequisite fails.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use fluent_dag::dep_graph::DependencyGraph;
use fluent_wvr::prelude::*;
use fluent_wvr::Runtime;
use internment::ArcIntern;
use thiserror::Error;
use tokio::task::JoinSet;

/// Events emitted by tasks running inside a `SupervisedBatch`.
#[derive(Debug, Clone)]
pub enum SupervisedBatchEvent {
    Completed {
        name: ArcIntern<str>,
        output: WorkOutput,
    },
    Panicked {
        name: ArcIntern<str>,
        info: String,
    },
    Failed {
        name: ArcIntern<str>,
        error: WorkError,
    },
    Cancelled {
        name: ArcIntern<str>,
        reason: CancelReason,
    },
}

/// Reasons why a supervised-batch task was cancelled.
#[derive(Debug, Clone)]
pub enum CancelReason {
    Timeout,
    DependencyFailed,
    Aborted,
}

/// Configuration for a `SupervisedBatch`.
#[derive(Debug, Clone, Copy)]
pub struct SupervisedBatchConfig {
    /// Maximum number of tasks to poll per `SupervisedBatch::poll` invocation.
    /// Prevents a single SupervisedBatch from starving the executor.
    pub poll_budget: usize,
    /// Whether a `WorkError` returned by a registered unit is worth a retry.
    /// Defaults to `WorkError::is_retryable` — permanent `Execution` failures
    /// short-circuit without burning backoff budget. Override when a
    /// SupervisedBatch units wrap genuinely transient failures in
    /// `WorkError::Execution` (e.g. chart targets whose LLM call failed).
    pub is_retryable: fn(&WorkError) -> bool,
}

impl PartialEq for SupervisedBatchConfig {
    fn eq(&self, other: &Self) -> bool {
        // The retry predicate is a policy function, not data — fn-pointer
        // addresses are not guaranteed unique, so equality compares the
        // numeric tuning only.
        self.poll_budget == other.poll_budget
    }
}

impl Eq for SupervisedBatchConfig {}

impl Default for SupervisedBatchConfig {
    fn default() -> Self {
        Self {
            poll_budget: 64,
            is_retryable: WorkError::is_retryable,
        }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SupervisedBatchError {
    #[error("task already registered: {0}")]
    DuplicateName(ArcIntern<str>),
}

/// Summary of a SupervisedBatch.s execution result.
#[derive(Debug, Default)]
pub struct SupervisedBatchSummary {
    pub completed: Vec<SupervisedBatchEvent>,
    pub panicked: Vec<SupervisedBatchEvent>,
    pub failed: Vec<SupervisedBatchEvent>,
    pub cancelled: Vec<SupervisedBatchEvent>,
}

/// A SupervisedBatch that manages a group of `WorkUnit` tasks with retry, timeout,
/// and dependency-based cancellation. Implements `Future` to drive task completion.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_concurrency::batch::SupervisedBatch;
/// use fluent_wvr::{CapabilitySet, WorkContext};
/// use fluent_wvr::wrapper::ComponentAdapter;
///
/// # async fn example() {
/// let rt: Arc<dyn fluent_wvr::Runtime> = Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime);
/// let mut batch = SupervisedBatch::new(rt, CapabilitySet::new());
/// // batch.register(component_a);
/// // batch.register(component_b);
/// let summary = batch.await;
/// assert!(summary.panicked.is_empty() && summary.failed.is_empty());
/// # }
/// ```
#[must_use = "SupervisedBatch must be awaited to completion to get a SupervisedBatchSummary"]
pub struct SupervisedBatch {
    runtime: Arc<dyn Runtime>,
    caps: CapabilitySet,
    config: SupervisedBatchConfig,
    /// Dependency graph: tracks which tasks depend on which assets,
    /// which tasks provide which assets, and an inverted index for
    /// O(1) dependent lookup. Composed from `fluent_dag::dep_graph`.
    graph: DependencyGraph<ArcIntern<str>>,
    task_names: HashMap<tokio::task::Id, ArcIntern<str>>,
    abort_handles: HashMap<ArcIntern<str>, tokio::task::AbortHandle>,
    cancelled_tasks: HashSet<ArcIntern<str>>,
    join_set: JoinSet<Result<WorkOutput, WorkError>>,
    active_count: usize,
    summary: SupervisedBatchSummary,
    done: bool,
}

impl SupervisedBatch {
    /// Creates a new SupervisedBatch with the given runtime and capabilities.
    pub fn new(runtime: Arc<dyn Runtime>, caps: CapabilitySet) -> Self {
        Self::new_with_config(runtime, caps, SupervisedBatchConfig::default())
    }

    /// Creates a new SupervisedBatch with the given runtime, capabilities, and configuration.
    pub fn new_with_config(
        runtime: Arc<dyn Runtime>,
        caps: CapabilitySet,
        config: SupervisedBatchConfig,
    ) -> Self {
        Self {
            runtime,
            caps,
            config,
            graph: DependencyGraph::new(),
            task_names: HashMap::new(),
            abort_handles: HashMap::new(),
            cancelled_tasks: HashSet::new(),
            join_set: JoinSet::new(),
            active_count: 0,
            summary: SupervisedBatchSummary::default(),
            done: false,
        }
    }

    /// Registers a `Component` in the SupervisedBatch. Returns `&mut Self` for builder chaining.
    ///
    /// Returns `Err(BatchError::DuplicateName)` if a unit with the same `name()`
    /// has already been registered. If you intentionally want to replace an
    /// existing task, use `register_or_replace` (not yet implemented — file
    /// a feature request if you need it).
    pub fn register(&mut self, unit: Arc<dyn Component>) -> Result<&mut Self, SupervisedBatchError> {
        let ctx = WorkContext::for_unit_in_batch(&self.runtime, &self.caps, |_| {});
        self.register_with_context(unit, ctx)
    }

    /// Registers a `Component` with a custom `WorkContext`.
    ///
    /// Returns `Err(BatchError::DuplicateName)` if a unit with the same `name()`
    /// has already been registered.
    pub fn register_with_context(
        &mut self,
        unit: Arc<dyn Component>,
        ctx: WorkContext,
    ) -> Result<&mut Self, SupervisedBatchError> {
        let name: ArcIntern<str> = ArcIntern::from(unit.name());
        let depends: Vec<ArcIntern<str>> = unit.depends().to_vec();
        let provides: Vec<ArcIntern<str>> = unit.provides().to_vec();

        self.graph
            .register(&name, &depends, &provides)
            .map_err(|_| SupervisedBatchError::DuplicateName(name))?;

        self.spawn_unit(unit, ctx);
        Ok(self)
    }

    fn spawn_unit(&mut self, unit: Arc<dyn Component>, ctx: WorkContext) {
        let name: ArcIntern<str> = ArcIntern::from(unit.name());
        let max_retries = ctx.max_retries;
        let timeout_ms = ctx.timeout_ms;
        let is_retryable = self.config.is_retryable;

        let abort = self.join_set.spawn(async move {
            execute_with_timeout_and_retry(unit, ctx, max_retries, timeout_ms, is_retryable).await
        });

        let id = abort.id();
        self.task_names.insert(id, name.clone());
        self.abort_handles.insert(name, abort);
        self.active_count += 1;
    }

    fn cancel_dependents_of(&mut self, name: &ArcIntern<str>) {
        // Delegate the dependency traversal to `DependencyGraph::dependents_of`,
        // which performs the DFS with cycle detection (ported verbatim from
        // the original hand-rolled implementation). The graph returns the
        // transitive set of nodes that depend on `name`; SupervisedBatch applies the
        // abort side effect to each.
        let to_cancel = self.graph.dependents_of(name);
        for task_name in &to_cancel {
            if let Some(handle) = self.abort_handles.get(task_name) {
                if !handle.is_finished() {
                    handle.abort();
                    self.cancelled_tasks.insert(task_name.clone());
                }
            }
        }
    }
}

impl Future for SupervisedBatch {
    type Output = SupervisedBatchSummary;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(std::mem::take(&mut this.summary));
        }

        let mut budget = this.config.poll_budget;
        loop {
            let mut join_set = std::pin::Pin::new(&mut this.join_set);
            match join_set.as_mut().poll_join_next_with_id(cx) {
                Poll::Ready(Some(Ok((id, Ok(output))))) => {
                    let name = this
                        .task_names
                        .remove(&id)
                        .unwrap_or_else(|| ArcIntern::from("unknown"));
                    this.summary
                        .completed
                        .push(SupervisedBatchEvent::Completed { name, output });
                    this.active_count -= 1;
                    budget -= 1;
                }
                Poll::Ready(Some(Ok((id, Err(WorkError::Timeout { .. }))))) => {
                    let name = this
                        .task_names
                        .remove(&id)
                        .unwrap_or_else(|| ArcIntern::from("unknown"));
                    this.cancel_dependents_of(&name);
                    this.summary.cancelled.push(SupervisedBatchEvent::Cancelled {
                        name,
                        reason: CancelReason::Timeout,
                    });
                    this.active_count -= 1;
                    budget -= 1;
                }
                Poll::Ready(Some(Ok((id, Err(e))))) => {
                    let name = this
                        .task_names
                        .remove(&id)
                        .unwrap_or_else(|| ArcIntern::from("unknown"));
                    this.cancel_dependents_of(&name);
                    this.summary
                        .failed
                        .push(SupervisedBatchEvent::Failed { name, error: e });
                    this.active_count -= 1;
                    budget -= 1;
                }
                Poll::Ready(Some(Err(e))) => {
                    let name = this
                        .task_names
                        .remove(&e.id())
                        .unwrap_or_else(|| ArcIntern::from("unknown"));
                    if e.is_cancelled() {
                        let reason = if this.cancelled_tasks.remove(&name) {
                            CancelReason::DependencyFailed
                        } else {
                            CancelReason::Aborted
                        };
                        this.summary
                            .cancelled
                            .push(SupervisedBatchEvent::Cancelled { name, reason });
                    } else if e.is_panic() {
                        this.cancel_dependents_of(&name);
                        this.summary.panicked.push(SupervisedBatchEvent::Panicked {
                            name,
                            info: "task panicked".into(),
                        });
                    } else {
                        this.cancel_dependents_of(&name);
                        this.summary.panicked.push(SupervisedBatchEvent::Panicked {
                            name,
                            info: "task terminated abnormally".into(),
                        });
                    }
                    this.active_count -= 1;
                    budget -= 1;
                }
                Poll::Ready(None) => {
                    this.done = true;
                    return Poll::Ready(std::mem::take(&mut this.summary));
                }
                Poll::Pending => {
                    if this.active_count == 0 {
                        this.done = true;
                        return Poll::Ready(std::mem::take(&mut this.summary));
                    }
                    return Poll::Pending;
                }
            }

            if budget == 0 {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }

            if this.active_count == 0 {
                this.done = true;
                return Poll::Ready(std::mem::take(&mut this.summary));
            }
        }
    }
}

impl Drop for SupervisedBatch {
    fn drop(&mut self) {
        if !self.done {
            self.join_set.abort_all();
        }
    }
}

async fn execute_with_timeout_and_retry(
    unit: Arc<dyn Component>,
    ctx: WorkContext,
    max_retries: u32,
    timeout_ms: u64,
    is_retryable: fn(&WorkError) -> bool,
) -> Result<WorkOutput, WorkError> {
    // Yield to allow pending abort signals to be processed before
    // executing the synchronous work unit body.
    tokio::task::yield_now().await;
    let fut = async {
        common_core::retry::retry_async(
            max_retries.saturating_add(1).max(1),
            100,
            50,
            is_retryable,
            || async {
                // Allow abort signals to be processed before each attempt.
                tokio::task::yield_now().await;
                // Intentionally NOT wrapped in catch_unwind so that panics
                // propagate through JoinSet as JoinError::Panic. This ensures
                // SupervisedBatch::poll intercepts them and triggers the dependency-aware
                // cancellation graph via cancel_dependents_of.
                unit.execute(&ctx)
            },
        )
        .await
    };

    if timeout_ms > 0 {
        match tokio::time::timeout(Duration::from_millis(timeout_ms), fut).await {
            Ok(result) => result,
            Err(_) => Err(WorkError::Timeout {
                duration_ms: timeout_ms,
                unit: unit.name().to_string(),
            }),
        }
    } else {
        fut.await
    }
}
