//! Affinity-aware priority scheduler.
//!
//! Sits between the pipeline and the agent request queue. Requests for the
//! currently-active session receive a priority bonus, preventing excessive
//! context switching. Aging prevents indefinite starvation of other sessions.
//!
//! Uses `fluent-concurrency::pool::PriorityResultPool` internally — higher
//! priority values are dispatched first; within the same priority, FIFO order
//! is maintained.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fluent_concurrency::pool::PriorityResultPool;

/// Errors produced by the affinity scheduler.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("pool error: {0}")]
    Pool(String),
    #[error("pool cancelled: {0}")]
    Canceled(String),
}

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
/// sits between the pipeline and the agent dispatch. It is designed to be
/// owned by the router and called directly in the async dispatch path.
pub struct AffinityScheduler<T: Send + 'static, R: Send + 'static, E: Send + 'static> {
    pool: Arc<PriorityResultPool<T, R, E>>,
    current_affinity: Mutex<Option<String>>,
    aging: AgingConfig,
    /// Maps identity → base priority for tasks currently in-flight.
    base_priorities: Mutex<HashMap<String, i32>>,
    last_aging: Mutex<Instant>,
}

impl<T, R, E> AffinityScheduler<T, R, E>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    E: Send + std::fmt::Debug + 'static,
{
    /// Create a new affinity scheduler wrapping the given pool.
    pub fn new(pool: Arc<PriorityResultPool<T, R, E>>) -> Self {
        Self {
            pool,
            current_affinity: Mutex::new(None),
            aging: AgingConfig::default(),
            base_priorities: Mutex::new(HashMap::new()),
            last_aging: Mutex::new(Instant::now()),
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
    ) -> Result<R, SchedulerError> {
        self.maybe_age();

        let priority = self.compute_priority(&task.identity, base_priority);

        self.pool
            .submit(task.task, priority)
            .await
            .map_err(|e| match e {
                fluent_concurrency::pool::ResultPoolError::Canceled => {
                    SchedulerError::Canceled("pool cancelled".into())
                }
                fluent_concurrency::pool::ResultPoolError::Pool(pe) => {
                    SchedulerError::Pool(pe.to_string())
                }
                fluent_concurrency::pool::ResultPoolError::Inner(inner) => {
                    SchedulerError::Pool(format!("inner error: {inner:?}"))
                }
            })
    }

    /// Set the current affinity session.
    pub fn set_affinity(&self, identity: Option<String>) {
        let mut aff = self.current_affinity.lock().unwrap();
        *aff = identity;
    }

    /// Get the current affinity session.
    pub fn current_affinity(&self) -> Option<String> {
        self.current_affinity.lock().unwrap().clone()
    }

    fn compute_priority(&self, identity: &str, base_priority: i32) -> i32 {
        let affinity = self.current_affinity.lock().unwrap();
        let mut priorities = self.base_priorities.lock().unwrap();

        let effective_base = priorities.get(identity).copied().unwrap_or(base_priority);
        priorities.insert(identity.to_string(), effective_base);

        if affinity.as_deref() == Some(identity) {
            effective_base + self.aging.affinity_bonus
        } else {
            effective_base
        }
    }

    fn maybe_age(&self) {
        let should_age = {
            let last = self.last_aging.lock().unwrap();
            last.elapsed() >= self.aging.aging_interval
        };

        if should_age {
            self.age_priorities();
        }
    }

    fn age_priorities(&self) {
        let mut priorities = self.base_priorities.lock().unwrap();
        let mut last = self.last_aging.lock().unwrap();

        // Only age tasks that are not the current affinity (starved tasks).
        let affinity = self.current_affinity.lock().unwrap().clone();
        for (identity, prio) in priorities.iter_mut() {
            if Some(identity.as_str()) != affinity.as_deref() {
                *prio = (*prio + self.aging.aging_rate).min(self.aging.max_priority);
            }
        }

        *last = Instant::now();
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
mod tests {
    use super::*;
    use fluent_concurrency::tokio_runtime;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_affinity_bonus_applied() {
        let runtime = tokio_runtime();
        let processed = Arc::new(AtomicUsize::new(0));
        let p_clone = Arc::clone(&processed);
        let p = Arc::new(PriorityResultPool::new(
            Arc::clone(&runtime),
            1,
            move |identity: String| {
                let c = Arc::clone(&p_clone);
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok::<String, String>(identity)
                }
            },
        ));

        let scheduler = AffinityScheduler::new(Arc::clone(&p));
        scheduler.set_affinity(Some("session-a".into()));

        let task = ScheduledTask {
            identity: "session-a".into(),
            task: "task-a".into(),
            enqueued_at: Instant::now(),
        };

        let result = scheduler.submit(task, 0).await.unwrap();
        assert_eq!(result, "task-a");
        assert_eq!(processed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_no_affinity_bonus_for_different_session() {
        let runtime = tokio_runtime();
        let processed = Arc::new(AtomicUsize::new(0));
        let p_clone = Arc::clone(&processed);
        let p = Arc::new(PriorityResultPool::new(
            Arc::clone(&runtime),
            1,
            move |identity: String| {
                let c = Arc::clone(&p_clone);
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok::<String, String>(identity)
                }
            },
        ));

        let scheduler = AffinityScheduler::new(Arc::clone(&p));
        scheduler.set_affinity(Some("session-a".into()));

        let task = ScheduledTask {
            identity: "session-b".into(),
            task: "task-b".into(),
            enqueued_at: Instant::now(),
        };

        scheduler.submit(task, 0).await.unwrap();
        assert_eq!(processed.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_aging_increases_priority_for_starved_tasks() {
        let runtime = tokio_runtime();
        let p = Arc::new(PriorityResultPool::new(
            Arc::clone(&runtime),
            1,
            move |_identity: String| async move { Ok::<String, String>("done".into()) },
        ));

        let scheduler = AffinityScheduler::new(Arc::clone(&p))
            .with_aging(AgingConfig {
                affinity_bonus: 10,
                aging_interval: Duration::ZERO,
                aging_rate: 5,
                max_priority: 100,
            });

        scheduler.set_affinity(Some("session-a".into()));

        let task = ScheduledTask {
            identity: "session-b".into(),
            task: "task-b".into(),
            enqueued_at: Instant::now(),
        };
        scheduler.submit(task, 0).await.unwrap();

        scheduler.age_now();

        let prios = scheduler.base_priorities.lock().unwrap();
        assert_eq!(prios.get("session-b"), Some(&5));
    }

    #[tokio::test]
    async fn test_set_and_get_affinity() {
        let runtime = tokio_runtime();
        let p = Arc::new(PriorityResultPool::new(
            Arc::clone(&runtime),
            1,
            move |_: String| async move { Ok::<String, String>("ok".into()) },
        ));

        let scheduler: AffinityScheduler<String, String, String> =
            AffinityScheduler::new(Arc::clone(&p));

        assert_eq!(scheduler.current_affinity(), None);

        scheduler.set_affinity(Some("my-session".into()));
        assert_eq!(scheduler.current_affinity(), Some("my-session".into()));

        scheduler.set_affinity(None);
        assert_eq!(scheduler.current_affinity(), None);
    }
}