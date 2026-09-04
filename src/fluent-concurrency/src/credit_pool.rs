//! Credit-gated bounded worker pool: a [`WorkerPool`] whose submissions are
//! additionally bounded by a [`CreditFlow`] pair (chain backpressure).
//!
//! The shape was extracted from the router's two structurally-identical
//! workers — the parse-review worker and the entity-link overlay worker
//! (`server/review.rs`, `server/entity_link.rs`). Each worker is a
//! `WorkerPool` whose `submit` path acquires a credit token before enqueuing
//! and whose handler releases the token via [`CreditReceiver::recv`] after
//! each processed job. Centralizing it removes the drift risk that a third
//! copy-pasted worker gets the `CreditSpec` formulas or the worker cap wrong.
//!
//! **Load-bearing constants.** The worker cap (`2`) and the `CreditSpec`
//! formulas (`initial = credit_limit.max(1)`, `more_after =
//! (credit_limit / 2).max(1)`) match both originating workers byte-for-byte —
//! do not "improve" them without changing every consumer's behavior.
//!
//! [`CreditFlow`]: crate::flow

use std::future::Future;
use std::sync::Arc;

use fluent_wvr::Runtime;
use tokio::sync::Mutex;

use crate::flow::{self, CreditSender, CreditSpec};
use crate::pool::{PoolError, SinkPool, WorkerPool};

/// A bounded worker pool whose submissions are gated by a credit flow.
///
/// [`submit`](Self::submit) acquires a credit token before enqueuing (blocking
/// only when credit is exhausted — the hot path returns immediately
/// otherwise) and the handler releases the token once per processed job. A
/// drained pool returns [`PoolError::Closed`]; the inner queue's full state is
/// handled by backpressure (`submit` waits for space — `PoolError::Full` is
/// the synchronous fast-path error and is never returned on the submit path).
pub struct CreditGatedPool<J: Send + 'static> {
    pool: Mutex<Option<WorkerPool<J>>>,
    credit: CreditSender,
    worker_count: usize,
}

impl<J: Send + Sync + 'static> CreditGatedPool<J> {
    /// Construct the pool. `credit_limit` bounds in-flight jobs (chain
    /// backpressure); `queue_capacity` bounds the pool queue. The worker cap
    /// is fixed at `2` and the `CreditSpec` is derived from `credit_limit`
    /// exactly as the originating review/entity-link workers did — these are
    /// load-bearing and must not drift.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(
        runtime: Arc<dyn Runtime>,
        credit_limit: usize,
        queue_capacity: usize,
        handler: F,
    ) -> Self
    where
        F: Fn(J) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send,
    {
        let (credit, credit_receiver) = flow::new(CreditSpec {
            initial: credit_limit.max(1),
            more_after: (credit_limit / 2).max(1),
        });
        let receiver = Arc::new(credit_receiver);
        // Load-bearing: the worker cap matches both originating workers.
        const WORKER_CAP: usize = 2;
        let handler = Arc::new(handler);
        let pool = WorkerPool::new(
            runtime,
            WORKER_CAP,
            queue_capacity,
            move |job: J| {
                let receiver = Arc::clone(&receiver);
                let handler = Arc::clone(&handler);
                async move {
                    handler(job).await;
                    // Release the credit token so the next submit can proceed.
                    receiver.recv();
                }
            },
        );

        Self {
            pool: Mutex::new(Some(pool)),
            credit,
            worker_count: WORKER_CAP,
        }
    }

    /// Enqueue a job, bounded by the credit gate (blocks only when credit is
    /// exhausted — the hot path returns immediately otherwise).
    pub async fn submit(&self, job: J) -> Result<(), PoolError> {
        let pool = self.pool.lock().await;
        let Some(pool) = pool.as_ref() else {
            return Err(PoolError::Closed);
        };
        self.credit.send(|| pool.submit(job)).await
    }

    /// Whether the credit gate is currently blocking producers.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.credit.is_blocked()
    }

    /// Drain in-flight jobs and shut the pool down (graceful shutdown). The
    /// pool is unusable afterward.
    pub async fn drain(&self) {
        let pool = self.pool.lock().await.take();
        if let Some(pool) = pool {
            pool.shutdown().await;
        }
    }

    /// The number of worker tasks processing jobs.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }
}

impl<J: Send + Sync + 'static> SinkPool for CreditGatedPool<J> {
    type Job = J;
    async fn submit(&self, job: Self::Job) -> Result<(), PoolError> {
        CreditGatedPool::submit(self, job).await
    }
    fn worker_count(&self) -> usize {
        CreditGatedPool::worker_count(self)
    }
    async fn shutdown(self) {
        CreditGatedPool::drain(&self).await;
    }
}


#[cfg(test)]
#[path = "../tests/credit_pool.rs"]
mod tests;
