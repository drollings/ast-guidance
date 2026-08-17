//! Credit-based backpressure flow control with deferred credit support.
//! Mirrors RabbitMQ's `credit_flow` semantics: a sender has a credit budget,
//! the receiver periodically sends bumps when its counter reaches `more_after`.
//!
//! Note on locking: `CreditSender` guards its bump receiver with a
//! `tokio::sync::Mutex` (not `std::sync::Mutex` like `Queue`/`PriorityJobQueue`
//! in `pool.rs`) because the guard is held across `rx.recv().await` in
//! `send`. Do not "fix" it to `std::sync::Mutex` — the guard crossing an await
//! would make the future non-`Send` and the waker bookkeeping is justified
//! here.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

use tokio::sync::mpsc;
use tokio::sync::Mutex;

/// Configuration for a credit flow pair.
#[derive(Debug, Clone)]
pub struct CreditSpec {
    pub initial: usize,
    pub more_after: usize,
}

/// Sends work items, consuming one credit per send.
/// Blocks when credit is exhausted until the receiver sends a bump.
pub struct CreditSender {
    credit: AtomicIsize,
    bump_rx: Mutex<mpsc::UnboundedReceiver<usize>>,
    blocked: AtomicBool,
}

impl CreditSender {
    pub async fn send<F, Fut, T>(&self, op: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        loop {
            let current = self.credit.load(Ordering::SeqCst);
            if current > 0 {
                if self
                    .credit
                    .compare_exchange(current, current - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    return op().await;
                }
            } else {
                self.blocked.store(true, Ordering::SeqCst);
                // Wait for a bump from the receiver
                let mut rx = self.bump_rx.lock().await;
                if let Some(amount) = rx.recv().await {
                    self.credit.fetch_add(amount as isize, Ordering::SeqCst);
                    self.blocked.store(false, Ordering::SeqCst);
                }
            }
        }
    }

    /// Returns whether the sender is currently blocked waiting for credit.
    pub fn is_blocked(&self) -> bool {
        self.blocked.load(Ordering::SeqCst)
    }

    /// Returns the current credit balance (may be negative if over-drafted).
    pub fn current_credit(&self) -> isize {
        self.credit.load(Ordering::SeqCst)
    }
}

/// Receives work notifications and sends credit bumps upstream.
pub struct CreditReceiver {
    spec: CreditSpec,
    counter: AtomicUsize,
    bump_tx: mpsc::UnboundedSender<usize>,
}

impl CreditReceiver {
    pub fn recv(&self) {
        let prev = self.counter.fetch_add(1, Ordering::SeqCst);
        if prev + 1 >= self.spec.more_after {
            self.counter.store(0, Ordering::SeqCst);
            let _ = self.bump_tx.send(self.spec.more_after);
        }
    }
}

/// Creates a new credit flow pair from a `CreditSpec`.
/// Returns `(sender, receiver)`.
pub fn new(spec: CreditSpec) -> (CreditSender, CreditReceiver) {
    let (bump_tx, bump_rx) = mpsc::unbounded_channel();
    (
        CreditSender {
            credit: AtomicIsize::new(spec.initial as isize),
            bump_rx: Mutex::new(bump_rx),
            blocked: AtomicBool::new(false),
        },
        CreditReceiver {
            spec,
            counter: AtomicUsize::new(0),
            bump_tx,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn send_returns_the_closure_result() {
        let (sender, _receiver) = new(CreditSpec {
            initial: 1,
            more_after: 1,
        });
        let result = sender.send(|| async { 42 }).await;
        assert_eq!(result, 42);
    }

    #[tokio::test(start_paused = true)]
    async fn credit_decrements_per_send_until_zero() {
        let (sender, _receiver) = new(CreditSpec {
            initial: 3,
            more_after: 10,
        });
        assert_eq!(sender.current_credit(), 3);
        assert!(!sender.is_blocked());
        sender.send(|| async {}).await;
        assert_eq!(sender.current_credit(), 2);
        sender.send(|| async {}).await;
        assert_eq!(sender.current_credit(), 1);
        sender.send(|| async {}).await;
        assert_eq!(sender.current_credit(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn receiver_counter_accumulates_below_more_after_without_bump() {
        // more_after=5: the first four recv()s must NOT produce a bump, so a
        // blocked send stays blocked (credit stays 0).
        let (sender, receiver) = new(CreditSpec {
            initial: 0,
            more_after: 5,
        });
        let sender = Arc::new(sender);
        let s = Arc::clone(&sender);
        let blocked = tokio::spawn(async move {
            s.send(|| async {}).await;
        });
        tokio::time::sleep(Duration::from_millis(10)).await; // let the send reach the block
        assert!(sender.is_blocked());
        for _ in 0..4 {
            receiver.recv();
        }
        assert!(sender.is_blocked(), "no bump until more_after is reached");
        receiver.recv(); // 5th recv -> bump(5)
        blocked.await.expect("sender completes");
        assert!(!sender.is_blocked());
        // The bump restored `more_after` (5) and the send consumed one.
        assert_eq!(sender.current_credit(), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn more_after_one_bumps_every_recv() {
        // more_after=1: every recv restores a single credit, so a producer and
        // consumer can ping-pong indefinitely without deadlock.
        let (sender, receiver) = new(CreditSpec {
            initial: 1,
            more_after: 1,
        });
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..5 {
            let c = Arc::clone(&counter);
            sender
                .send(|| async move {
                    c.fetch_add(1, Ordering::SeqCst);
                })
                .await;
            receiver.recv();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
        assert_eq!(sender.current_credit(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn zero_initial_credit_blocks_first_send_until_bump() {
        let (sender, receiver) = new(CreditSpec {
            initial: 0,
            more_after: 2,
        });
        let sender = Arc::new(sender);
        let s = Arc::clone(&sender);
        let handle = tokio::spawn(async move {
            s.send(|| async {}).await;
            assert!(!s.is_blocked());
        });
        // Give the blocked sender time to mark itself blocked.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(sender.is_blocked());
        assert_eq!(sender.current_credit(), 0);
        receiver.recv();
        receiver.recv();
        handle.await.expect("sender completes");
        assert!(!sender.is_blocked());
    }

    #[tokio::test(start_paused = true)]
    async fn is_blocked_flag_tracks_block_state() {
        let (sender, receiver) = new(CreditSpec {
            initial: 1,
            more_after: 1,
        });
        assert!(!sender.is_blocked());
        sender.send(|| async {}).await;
        let sender = Arc::new(sender);
        let s = Arc::clone(&sender);
        let handle = tokio::spawn(async move {
            s.send(|| async {}).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(sender.is_blocked(), "blocked after credit exhausted");
        receiver.recv();
        handle.await.expect("sender completes");
        assert!(!sender.is_blocked(), "unblocked after the bump");
    }

    #[tokio::test(start_paused = true)]
    async fn queued_bumps_are_consumed_on_demand() {
        // recv() bumps are delivered through the channel, not applied to the
        // atomic eagerly: a send that needs credit drains the queued bumps.
        let (sender, receiver) = new(CreditSpec {
            initial: 0,
            more_after: 1,
        });
        receiver.recv(); // bump(1) queued
        receiver.recv(); // bump(1) queued
        // Each send consumes exactly one queued bump; both complete without
        // needing another recv.
        for _ in 0..2 {
            sender.send(|| async {}).await;
        }
        assert_eq!(sender.current_credit(), 0);
    }
}
