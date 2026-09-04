//! `CreditedFeedWorker<Item>` — a credit-gated, bounded background feed worker.
//!
//! The shape was extracted from the router's `LedgerTierWorker`
//! (`src/router/src/ledger/tiering.rs`): a bounded `mpsc` feed of items, a
//! [`CreditFlow`] pair gating the async producer path, a `Limiter` bounding
//! concurrent handler invocations, and a drain loop that batches incoming
//! items and runs the shared handler per item. The consumer supplies only the
//! per-item `handler`; the primitive owns the feed, the credit gate, the
//! concurrency bound, and the drain-loop mechanics.
//!
//! **Load-bearing constants.** The default [`FeedConfig`] values and the
//! credit-release timing — a [`CreditReceiver::recv`] fires only *after* the
//! handler completes, never after enqueue — match `LedgerTierWorker`'s current
//! behavior exactly. Do not adjust them without changing every consumer's
//! behavior.
//!
//! Consumers:
//! - `LedgerTierWorker` (the original shape this primitive was extracted from,
//!   migrated onto it as the first consumer).
//! - the router's `overlay_worker` (the `arc_ready` annotation overlays).
//!
//! [`CreditFlow`]: crate::flow

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fluent_wvr::Runtime;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::flow::{self, CreditReceiver, CreditSender, CreditSpec};
use crate::pool::Limiter;

/// Configuration for a [`CreditedFeedWorker`]. Defaults mirror
/// `LedgerTierWorker`'s `TierConfig` (load-bearing — do not drift).
#[derive(Debug, Clone)]
pub struct FeedConfig {
    /// Capacity of the pending-item feed (`mpsc` bound). Default 1024.
    pub queue_capacity: usize,
    /// Credit granted to the feed's producer up front: the max outstanding
    /// items the async (`enqueue_with_credit`) path may have in flight before
    /// it blocks. Default 256.
    pub credit_limit: usize,
    /// How many processed items the consumer waits for before bumping credit
    /// back to the producer (`CreditSpec.more_after`). Default 8.
    pub credit_more_after: usize,
    /// Max concurrent handler invocations (the `Limiter` cap). Default 8.
    pub max_concurrent: usize,
    /// Max items drained per batch. Default 8.
    pub batch_size: usize,
    /// Poll interval (ms) before giving up on filling a batch. Default 100.
    pub poll_interval_ms: u64,
}

impl Default for FeedConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1024,
            credit_limit: 256,
            credit_more_after: 8,
            max_concurrent: 8,
            batch_size: 8,
            poll_interval_ms: 100,
        }
    }
}

/// Errors produced by a [`CreditedFeedWorker`].
#[derive(Debug, thiserror::Error)]
pub enum FeedError {
    /// The feed is closed (drained) and no longer accepts items.
    #[error("feed closed")]
    FeedClosed,
}

/// Type-erased per-item handler: an owned boxed future per item, never a
/// borrow of the worker.
type Handler<Item> =
    Arc<dyn Fn(Item) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// A credit-gated, bounded background feed worker.
///
/// A bounded `mpsc` feed accepts items from producers; the background loop
/// spawned by [`Self::start`] drains it in batches and runs the shared handler
/// under a concurrency [`Limiter`]. The async producer path
/// ([`Self::enqueue_with_credit`]) is gated by a credit flow so a burst of
/// producers cannot grow the feed without bound; the sync path ([`Self::enqueue`])
/// uses the bounded channel's non-blocking `try_send` and skips when the feed
/// is full. [`Self::drain`] stops accepting new items and lets queued items
/// complete.
pub struct CreditedFeedWorker<Item: Send + 'static> {
    runtime: Arc<dyn Runtime>,
    config: FeedConfig,
    handler: Handler<Item>,
    sender: Mutex<Option<mpsc::Sender<Item>>>,
    receiver: Mutex<Option<mpsc::Receiver<Item>>>,
    credit: CreditSender,
    credit_receiver: CreditReceiver,
    limiter: Arc<Limiter>,
    draining: AtomicBool,
}

impl<Item: Send + Sync + 'static> CreditedFeedWorker<Item> {
    /// Construct a feed worker. The `handler` is invoked for every item
    /// drained from the feed, under the concurrency `Limiter`; its completion
    /// releases the producer's credit token (never before).
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(runtime: Arc<dyn Runtime>, config: FeedConfig, handler: F) -> Self
    where
        F: Fn(Item) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let handler: Handler<Item> = Arc::new(move |item| Box::pin(handler(item)));
        let (sender, receiver) = mpsc::channel(config.queue_capacity);
        let (credit, credit_receiver) = flow::new(CreditSpec {
            initial: config.credit_limit,
            more_after: config.credit_more_after,
        });
        let limiter = Arc::new(Limiter::new(config.max_concurrent.max(1)));
        Self {
            runtime,
            config,
            handler,
            sender: Mutex::new(Some(sender)),
            receiver: Mutex::new(Some(receiver)),
            credit,
            credit_receiver,
            limiter,
            draining: AtomicBool::new(false),
        }
    }

    /// The feed's sender; attach it to a producer (e.g. a store's write path)
    /// with the store's event-setter. Cloned, so the caller owns an
    /// independent send handle.
    pub fn sender(&self) -> mpsc::Sender<Item> {
        self.sender
            .lock()
            .unwrap()
            .clone()
            .expect("CreditedFeedWorker::sender called after drain")
    }

    /// Enqueue an item non-blocking. On a full feed it skips with a debug log
    /// — the credit-gated `enqueue_with_credit` path and the next boot
    /// backfill cover the stragglers. No-op after [`Self::drain`].
    pub fn enqueue(&self, item: Item) {
        let sender = self.sender.lock().unwrap();
        let Some(sender) = sender.as_ref() else {
            return;
        };
        match sender.try_send(item) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::debug!(
                    target: "fluent_concurrency.feed",
                    "feed full - skipping enqueue",
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!(
                    target: "fluent_concurrency.feed",
                    "feed closed - skipping enqueue",
                );
            }
        }
    }

    /// Enqueue an item with chain backpressure: acquires a credit token
    /// (blocking while exhausted) before forwarding the item, so a burst of
    /// producers cannot grow the feed without bound. The consumer releases the
    /// token via the receiver's `recv()` after processing each item.
    ///
    /// `Err(FeedError::FeedClosed)` when the feed is closed (drained).
    pub async fn enqueue_with_credit(&self, item: Item) -> Result<(), FeedError> {
        let sender = self
            .sender
            .lock()
            .unwrap()
            .clone()
            .ok_or(FeedError::FeedClosed)?;
        self.credit
            .send(move || async move {
                sender
                    .send(item)
                    .await
                    .map_err(|_| FeedError::FeedClosed)
            })
            .await
    }

    /// Whether the credit-gated producer is currently blocked waiting for a
    /// credit bump (i.e. the feed is saturated and the consumer has not yet
    /// processed enough items).
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.credit.is_blocked()
    }

    /// Enqueue every item produced by `source` (e.g. a boot backfill scan),
    /// each through the same non-blocking [`Self::enqueue`] path — so a large
    /// source is bounded by the feed capacity, never unbounded growth.
    pub fn backfill(&self, mut source: impl FnMut() -> Vec<Item>) {
        for item in source() {
            self.enqueue(item);
        }
    }

    /// Start the background drain loop, returning its join handle (the caller
    /// holds it so the worker lives for the process lifetime). Spawned through
    /// the injected `Runtime` — no ambient `tokio::spawn`. Must be called at
    /// most once.
    pub fn start(self: &Arc<Self>) -> JoinHandle<()> {
        let receiver = self
            .receiver
            .lock()
            .unwrap()
            .take()
            .expect("CreditedFeedWorker::start must be called at most once");
        let this = Arc::clone(self);
        let runtime = Arc::clone(&this.runtime);
        runtime.spawn(Box::pin(async move {
            this.run(receiver).await;
        }))
    }

    /// Graceful shutdown: stop accepting new items and let queued items
    /// complete. After this the background loop processes whatever remains in
    /// the feed and exits; the caller awaits the `JoinHandle` from
    /// [`Self::start`] for completion.
    pub fn drain(&self) {
        self.sender.lock().unwrap().take();
        self.draining.store(true, Ordering::SeqCst);
    }

    async fn run(self: Arc<Self>, mut receiver: mpsc::Receiver<Item>) {
        loop {
            // Draining: process whatever remains in the feed, then exit.
            if self.draining.load(Ordering::SeqCst) {
                let mut batch = Vec::new();
                while let Ok(item) = receiver.try_recv() {
                    batch.push(item);
                }
                if batch.is_empty() {
                    break;
                }
                self.process_batch(batch).await;
                continue;
            }

            // Wait for the first item, then drain up to batch_size.
            let Some(first) = receiver.recv().await else {
                break; // channel closed
            };
            let mut batch = vec![first];
            let mut timer = self
                .runtime
                .sleep(Duration::from_millis(self.config.poll_interval_ms));
            while batch.len() < self.config.batch_size {
                tokio::select! {
                    item = receiver.recv() => match item {
                        Some(id) => batch.push(id),
                        None => break,
                    },
                    () = &mut timer => break,
                }
            }
            self.process_batch(batch).await;
        }
    }

    async fn process_batch(&self, batch: Vec<Item>) {
        // Concurrent bounded by Limiter (max_concurrent). Each completion releases
        // exactly one credit token; panics also release via the join driver.
        let mut set = tokio::task::JoinSet::new();
        for item in batch {
            let limiter = Arc::clone(&self.limiter);
            let handler = Arc::clone(&self.handler);
            set.spawn(async move {
                limiter.run(|| handler(item)).await;
            });
            while set.len() >= self.config.max_concurrent {
                let _ = set.join_next().await;
                self.credit_receiver.recv();
            }
        }
        while set.join_next().await.is_some() {
            self.credit_receiver.recv();
        }
    }
}

#[cfg(test)]
#[path = "../tests/feed_worker.rs"]
mod tests;
