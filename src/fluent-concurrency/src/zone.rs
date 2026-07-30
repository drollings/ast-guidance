//! Supervision zone with async retry, dependency cancellation, and timeout.
//! A `Zone` manages a group of `WorkUnit` tasks and propagates cancellation
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

/// Events emitted by tasks running inside a `Zone`.
#[derive(Debug, Clone)]
pub enum ZoneEvent {
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

/// Reasons why a zone task was cancelled.
#[derive(Debug, Clone)]
pub enum CancelReason {
    Timeout,
    DependencyFailed,
    Aborted,
}

/// Configuration for a `Zone`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneConfig {
    /// Maximum number of tasks to poll per `Zone::poll` invocation.
    /// Prevents a single zone from starving the executor.
    pub poll_budget: usize,
}

impl Default for ZoneConfig {
    fn default() -> Self {
        Self { poll_budget: 64 }
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ZoneError {
    #[error("task already registered: {0}")]
    DuplicateName(ArcIntern<str>),
}

/// Summary of a zone's execution result.
#[derive(Debug, Default)]
pub struct ZoneSummary {
    pub completed: Vec<ZoneEvent>,
    pub panicked: Vec<ZoneEvent>,
    pub failed: Vec<ZoneEvent>,
    pub cancelled: Vec<ZoneEvent>,
}

/// A supervision zone that manages a group of `WorkUnit` tasks with retry, timeout,
/// and dependency-based cancellation. Implements `Future` to drive task completion.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_concurrency::zone::Zone;
/// use fluent_wvr::{CapabilitySet, WorkContext};
/// use fluent_wvr::wrapper::ComponentAdapter;
///
/// # async fn example() {
/// let rt: Arc<dyn fluent_wvr::Runtime> = Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime);
/// let mut zone = Zone::new(rt, CapabilitySet::new());
/// // zone.register(component_a);
/// // zone.register(component_b);
/// let summary = zone.await;
/// assert!(summary.panicked.is_empty() && summary.failed.is_empty());
/// # }
/// ```
#[must_use = "Zone must be awaited to completion to get a ZoneSummary"]
pub struct Zone {
    runtime: Arc<dyn Runtime>,
    caps: CapabilitySet,
    config: ZoneConfig,
    /// Dependency graph: tracks which tasks depend on which assets,
    /// which tasks provide which assets, and an inverted index for
    /// O(1) dependent lookup. Composed from `fluent_dag::dep_graph`.
    graph: DependencyGraph<ArcIntern<str>>,
    task_names: HashMap<tokio::task::Id, ArcIntern<str>>,
    abort_handles: HashMap<ArcIntern<str>, tokio::task::AbortHandle>,
    cancelled_tasks: HashSet<ArcIntern<str>>,
    join_set: JoinSet<Result<WorkOutput, WorkError>>,
    active_count: usize,
    summary: ZoneSummary,
    done: bool,
}

impl Zone {
    /// Creates a new zone with the given runtime and capabilities.
    pub fn new(runtime: Arc<dyn Runtime>, caps: CapabilitySet) -> Self {
        Self::new_with_config(runtime, caps, ZoneConfig::default())
    }

    /// Creates a new zone with the given runtime, capabilities, and configuration.
    pub fn new_with_config(
        runtime: Arc<dyn Runtime>,
        caps: CapabilitySet,
        config: ZoneConfig,
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
            summary: ZoneSummary::default(),
            done: false,
        }
    }

    /// Registers a `Component` in the zone. Returns `&mut Self` for builder chaining.
    ///
    /// Returns `Err(ZoneError::DuplicateName)` if a unit with the same `name()`
    /// has already been registered. If you intentionally want to replace an
    /// existing task, use `register_or_replace` (not yet implemented — file
    /// a feature request if you need it).
    pub fn register(&mut self, unit: Arc<dyn Component>) -> Result<&mut Self, ZoneError> {
        let ctx = WorkContext::for_unit_in_zone(&self.runtime, &self.caps, |_| {});
        self.register_with_context(unit, ctx)
    }

    /// Registers a `Component` with a custom `WorkContext`.
    ///
    /// Returns `Err(ZoneError::DuplicateName)` if a unit with the same `name()`
    /// has already been registered.
    pub fn register_with_context(
        &mut self,
        unit: Arc<dyn Component>,
        ctx: WorkContext,
    ) -> Result<&mut Self, ZoneError> {
        let name: ArcIntern<str> = ArcIntern::from(unit.name());
        let depends: Vec<ArcIntern<str>> = unit.depends().to_vec();
        let provides: Vec<ArcIntern<str>> = unit.provides().to_vec();

        self.graph
            .register(&name, &depends, &provides)
            .map_err(|_| ZoneError::DuplicateName(name))?;

        self.spawn_unit(unit, ctx);
        Ok(self)
    }

    fn spawn_unit(&mut self, unit: Arc<dyn Component>, ctx: WorkContext) {
        let name: ArcIntern<str> = ArcIntern::from(unit.name());
        let max_retries = ctx.max_retries;
        let timeout_ms = ctx.timeout_ms;

        let abort = self.join_set.spawn(async move {
            execute_with_timeout_and_retry(unit, ctx, max_retries, timeout_ms).await
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
        // transitive set of nodes that depend on `name`; Zone applies the
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

impl Future for Zone {
    type Output = ZoneSummary;

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
                        .push(ZoneEvent::Completed { name, output });
                    this.active_count -= 1;
                    budget -= 1;
                }
                Poll::Ready(Some(Ok((id, Err(WorkError::Timeout { .. }))))) => {
                    let name = this
                        .task_names
                        .remove(&id)
                        .unwrap_or_else(|| ArcIntern::from("unknown"));
                    this.cancel_dependents_of(&name);
                    this.summary.cancelled.push(ZoneEvent::Cancelled {
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
                        .push(ZoneEvent::Failed { name, error: e });
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
                            .push(ZoneEvent::Cancelled { name, reason });
                    } else if e.is_panic() {
                        this.cancel_dependents_of(&name);
                        this.summary.panicked.push(ZoneEvent::Panicked {
                            name,
                            info: "task panicked".into(),
                        });
                    } else {
                        this.cancel_dependents_of(&name);
                        this.summary.panicked.push(ZoneEvent::Panicked {
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

impl Drop for Zone {
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
) -> Result<WorkOutput, WorkError> {
    // Yield to allow pending abort signals to be processed before
    // executing the synchronous work unit body.
    tokio::task::yield_now().await;
    let fut = async {
        let mut attempts = 0u32;
        loop {
            // Allow abort signals to be processed before each attempt.
            tokio::task::yield_now().await;
            attempts += 1;
            // Intentionally NOT wrapped in catch_unwind so that panics
            // propagate through JoinSet as JoinError::Panic. This ensures
            // Zone::poll intercepts them and triggers the dependency-aware
            // cancellation graph via cancel_dependents_of.
            match unit.execute(&ctx) {
                Ok(output) => return Ok(output),
                Err(e) => {
                    if attempts > max_retries {
                        return Err(e);
                    }
                    tokio::time::sleep(Duration::from_millis(100 * u64::from(attempts))).await;
                }
            }
        }
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
