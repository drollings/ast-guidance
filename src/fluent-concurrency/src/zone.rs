//! Supervision zone with async retry, dependency cancellation, and timeout.
//! A `Zone` manages a group of `WorkUnit` tasks and propagates cancellation
//! across dependent tasks when a prerequisite fails.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use fluent_wvr::{CapabilitySet, Component, Runtime, WorkContext, WorkError, WorkOutput, WorkUnit};
use internment::ArcIntern;
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

/// Summary of a zone's execution result.
#[derive(Debug, Default)]
pub struct ZoneSummary {
    pub completed: Vec<ZoneEvent>,
    pub panicked: Vec<ZoneEvent>,
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
/// assert!(summary.panicked.is_empty());
/// # }
/// ```
#[must_use = "Zone must be awaited to completion to get a ZoneSummary"]
pub struct Zone {
    runtime: Arc<dyn Runtime>,
    caps: CapabilitySet,
    config: ZoneConfig,
    /// Maps task_name → assets that task depends on.
    deps: HashMap<ArcIntern<str>, Vec<ArcIntern<str>>>,
    /// Maps task_name → assets that task provides.
    task_provides: HashMap<ArcIntern<str>, Vec<ArcIntern<str>>>,
    /// Inverted index: asset → task_names that depend on it.
    /// Built during registration for O(1) lookup in cancel_dependents_of.
    provides_to_dependents: HashMap<ArcIntern<str>, Vec<ArcIntern<str>>>,
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
            deps: HashMap::new(),
            task_provides: HashMap::new(),
            provides_to_dependents: HashMap::new(),
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
    pub fn register(&mut self, unit: Arc<dyn Component>) -> &mut Self {
        let ctx = WorkContext {
            rt: Arc::clone(&self.runtime),
            caps: self.caps.clone(),
            ..WorkContext::default()
        };
        self.register_with_context(unit, ctx)
    }

    /// Registers a `Component` with a custom `WorkContext`.
    pub fn register_with_context(
        &mut self,
        unit: Arc<dyn Component>,
        ctx: WorkContext,
    ) -> &mut Self {
        let name: ArcIntern<str> = ArcIntern::from(unit.name());
        let depends: Vec<ArcIntern<str>> = unit.depends().to_vec();
        let provides: Vec<ArcIntern<str>> = unit.provides().to_vec();

        if !depends.is_empty() {
            self.deps.insert(name.clone(), depends.clone());
            // Build inverted index: for each asset this task depends on,
            // record that this task is a dependent.
            for dep in &depends {
                self.provides_to_dependents
                    .entry(dep.clone())
                    .or_default()
                    .push(name.clone());
            }
        }
        if !provides.is_empty() {
            self.task_provides.insert(name.clone(), provides);
        }

        self.spawn_unit(unit, ctx);
        self
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
        let mut to_cancel: Vec<ArcIntern<str>> = Vec::new();
        // visited: all nodes ever processed (to cancel each at most once).
        let mut visited: HashSet<ArcIntern<str>> = HashSet::new();
        // active_path: nodes currently on the DFS recursion stack — used for
        // cycle detection (a back-edge into active_path indicates a cycle).
        let mut active_path: HashSet<ArcIntern<str>> = HashSet::new();
        // Stack entries: (node, whether children have been expanded).
        let mut stack: Vec<(ArcIntern<str>, bool)> = vec![(name.clone(), false)];
        active_path.insert(name.clone());

        while let Some((current, expanded)) = stack.last_mut() {
            if *expanded {
                // Backtrack: remove from active path.
                active_path.remove(current);
                stack.pop();
                continue;
            }
            *expanded = true;

            if !visited.insert(current.clone()) {
                // Already processed this node from another path; still
                // need to remove it from active_path before backtracking.
                active_path.remove(current);
                stack.pop();
                continue;
            }

            // O(1): look up what this task provides, then look up which
            // tasks depend on each provided asset via the inverted index.
            if let Some(provides) = self.task_provides.get(current) {
                for provided in provides {
                    if let Some(dependents) = self.provides_to_dependents.get(provided) {
                        for dep_name in dependents {
                            if active_path.contains(dep_name) {
                                tracing::warn!(
                                    "Dependency cycle detected: '{}' transitively depends on itself",
                                    dep_name,
                                );
                                continue;
                            }
                            if !visited.contains(dep_name) {
                                to_cancel.push(dep_name.clone());
                                active_path.insert(dep_name.clone());
                                stack.push((dep_name.clone(), false));
                            }
                        }
                    }
                }
            }
        }

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
                    this.summary.panicked.push(ZoneEvent::Panicked {
                        name,
                        info: e.to_string(),
                    });
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
