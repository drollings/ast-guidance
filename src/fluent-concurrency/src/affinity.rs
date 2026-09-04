//! Affinity-aware priority scheduler.
//!
//! Wraps `PriorityResultPool` and adds:
//! - **Affinity bonus** — tasks for the currently-active session get a priority
//!   boost, preventing excessive context switching.
//! - **Aging** — starved tasks periodically increase in priority to prevent
//!   indefinite starvation of other sessions.
//!
//! The scheduler is not a `WorkUnit` — it is a separate component that sits
//! between the pipeline and dispatch. It is designed to be owned by the
//! orchestrator and called directly in the async dispatch path.

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lru::LruCache;

use crate::pool::{PriorityResultPool, ResultPoolError};
use common_core::sync as sync_lock;

/// Configuration for aging behaviour.
#[derive(Debug, Clone)]
pub struct AgingConfig {
    /// Priority bonus added when task is for the currently-affine session.
    pub affinity_bonus: i32,
    /// How often aging runs (wall-clock).
    pub aging_interval: Duration,
    /// Priority increase per aging tick for starved tasks.
    pub aging_rate: i32,
    /// Maximum priority a starved task can reach through aging.
    pub max_priority: i32,
}

impl Default for AgingConfig {
    fn default() -> Self {
        Self {
            affinity_bonus: 10,
            aging_interval: Duration::from_secs(5),
            aging_rate: 2,
            max_priority: 100,
        }
    }
}

/// A task enqueued in the scheduler with its metadata.
#[derive(Debug, Clone)]
pub struct ScheduledTask<T> {
    pub identity: String,
    pub task: T,
    pub enqueued_at: Instant,
}

/// Affinity-aware scheduler that wraps a `PriorityResultPool`.
///
/// Tracks which session is currently "affine" and gives tasks for that
/// session a priority bonus. Starved tasks are periodically aged (their
/// base priority increases) to prevent indefinite starvation.
///
/// The scheduler is not a `WorkUnit` — it is a separate component that
/// sits between the pipeline and agent dispatch.
pub struct AffinityScheduler<T: Send + 'static, R: Send + 'static, E: Send + 'static> {
    pool: Arc<PriorityResultPool<T, R, E>>,
    current_affinity: Mutex<Option<String>>,
    aging: AgingConfig,
    base_priorities: Mutex<LruCache<String, i32>>,
    last_aging: Mutex<Instant>,
    runtime: Arc<dyn fluent_wvr::Runtime>,
}

const DEFAULT_MAX_IDENTITIES: usize = 1024;

impl<T, R, E> AffinityScheduler<T, R, E>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    E: Send + std::fmt::Debug + 'static,
{
    /// Create a new affinity scheduler wrapping the given pool.
    pub fn new(pool: Arc<PriorityResultPool<T, R, E>>) -> Self {
        Self::new_with_runtime(pool, crate::tokio_runtime())
    }

    /// Create a new affinity scheduler with an explicit runtime (for virtual-time tests).
    pub fn new_with_runtime(pool: Arc<PriorityResultPool<T, R, E>>, runtime: Arc<dyn fluent_wvr::Runtime>) -> Self {
        let cap = NonZeroUsize::new(DEFAULT_MAX_IDENTITIES).unwrap();
        Self {
            pool,
            current_affinity: Mutex::new(None),
            aging: AgingConfig::default(),
            base_priorities: Mutex::new(LruCache::new(cap)),
            last_aging: Mutex::new(runtime.now()),
            runtime,
        }
    }

    /// Create with explicit LRU cap (for testing eviction).
    pub fn new_with_cap(
        pool: Arc<PriorityResultPool<T, R, E>>,
        runtime: Arc<dyn fluent_wvr::Runtime>,
        cap: usize,
        aging: AgingConfig,
    ) -> Self {
        let cap = NonZeroUsize::new(cap.max(1)).unwrap();
        Self {
            pool,
            current_affinity: Mutex::new(None),
            aging,
            base_priorities: Mutex::new(LruCache::new(cap)),
            last_aging: Mutex::new(runtime.now()),
            runtime,
        }
    }

    /// Set the aging configuration.
    #[must_use]
    pub fn with_aging(mut self, aging: AgingConfig) -> Self {
        self.aging = aging;
        self
    }

    /// Submit a task with a base priority. If the task's identity matches
    /// the current affinity session, an affinity bonus is added.
    ///
    /// Also runs aging if the configured interval has elapsed.
    pub async fn submit(
        &self,
        task: ScheduledTask<T>,
        base_priority: i32,
    ) -> Result<R, ResultPoolError<E>> {
        self.maybe_age();
        let priority = self.compute_priority_with_base(&task, base_priority);
        self.pool.submit(task.task, priority).await
    }

    fn compute_priority_with_base(&self, task: &ScheduledTask<T>, base_priority: i32) -> i32 {
        // Canonical lock order via helper
        let (affinity_guard, mut prio_guard, _last_guard) = self.lock_in_order_for_compute();
        let affinity = affinity_guard.clone();
        // drop last_guard early? Keep order but we hold it.
        let effective_base = prio_guard.get(&task.identity).copied().unwrap_or(base_priority);
        // Blend enqueued_at age: per-task starvation, capped at aging_rate
        let now = self.runtime.now();
        let age_secs = now
            .checked_duration_since(task.enqueued_at)
            .unwrap_or(Duration::ZERO)
            .as_secs() as i32;
        let age_bonus = age_secs.min(self.aging.aging_rate).max(0);
        let with_age = effective_base.saturating_add(age_bonus);
        // Insert/update LRU
        prio_guard.put(task.identity.clone(), with_age);
        if affinity.as_deref() == Some(task.identity.as_str()) {
            with_age.saturating_add(self.aging.affinity_bonus)
        } else {
            with_age
        }
    }

    /// Public helper for tests: compute effective priority without submitting.
    pub fn effective_priority(&self, task: &ScheduledTask<T>, base_priority: i32) -> i32 {
        self.compute_priority_with_base(task, base_priority)
    }

    /// Helper enforcing canonical lock order: current_affinity -> base_priorities -> last_aging
    #[allow(clippy::type_complexity)]
    fn lock_in_order_for_compute(
        &self,
    ) -> (
        std::sync::MutexGuard<'_, Option<String>>,
        std::sync::MutexGuard<'_, LruCache<String, i32>>,
        std::sync::MutexGuard<'_, Instant>,
    ) {
        let a = sync_lock::lock(&self.current_affinity);
        let b = sync_lock::lock(&self.base_priorities);
        let c = sync_lock::lock(&self.last_aging);
        (a, b, c)
    }

    #[allow(clippy::type_complexity)]
    fn lock_all_for_age(
        &self,
    ) -> (
        std::sync::MutexGuard<'_, Option<String>>,
        std::sync::MutexGuard<'_, LruCache<String, i32>>,
        std::sync::MutexGuard<'_, Instant>,
    ) {
        let a = sync_lock::lock(&self.current_affinity);
        let b = sync_lock::lock(&self.base_priorities);
        let c = sync_lock::lock(&self.last_aging);
        (a, b, c)
    }

    /// Set the current affinity session.
    pub fn set_affinity(&self, identity: Option<String>) {
        let mut aff = sync_lock::lock(&self.current_affinity);
        *aff = identity;
    }

    /// Get the current affinity session.
    pub fn current_affinity(&self) -> Option<String> {
        sync_lock::lock(&self.current_affinity).clone()
    }

    /// Remove an identity from the bounded map (session close).
    pub fn remove_identity(&self, identity: &str) {
        let mut prios = sync_lock::lock(&self.base_priorities);
        prios.pop(identity);
    }

    /// For testing: current map length.
    pub fn base_priorities_len(&self) -> usize {
        sync_lock::lock(&self.base_priorities).len()
    }

    fn maybe_age(&self) {
        let should_age = {
            let last = sync_lock::lock(&self.last_aging);
            self.runtime.now().duration_since(*last) >= self.aging.aging_interval
        };

        if should_age {
            self.age_priorities();
        }
    }

    fn age_priorities(&self) {
        // Canonical order: current_affinity -> base_priorities -> last_aging
        let (affinity, mut priorities, mut last) = self.lock_all_for_age();
        let affinity = affinity.clone();
        for (identity, prio) in priorities.iter_mut() {
            if Some(identity.as_str()) != affinity.as_deref() {
                *prio = (*prio + self.aging.aging_rate).min(self.aging.max_priority);
            }
        }
        *last = self.runtime.now();
    }

    /// Run aging immediately (useful for testing).
    pub fn age_now(&self) {
        self.age_priorities();
    }

    /// Access the underlying pool.
    pub fn pool(&self) -> &Arc<PriorityResultPool<T, R, E>> {
        &self.pool
    }
}

#[cfg(test)]
#[path = "../tests/affinity.rs"]
mod tests;
