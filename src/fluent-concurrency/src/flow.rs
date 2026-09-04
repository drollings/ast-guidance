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
            // Acquire to observe the Release from receiver's fetch_add bump
            let current = self.credit.load(Ordering::Acquire);
            if current > 0 {
                // AcqRel on success pairs with Acquire load; Acquire on failure observes concurrent bump
                if self
                    .credit
                    .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return op().await;
                }
            } else {
                // Release pairs with Acquire in is_blocked/current_credit observers
                self.blocked.store(true, Ordering::Release);
                // Wait for a bump from the receiver
                let mut rx = self.bump_rx.lock().await;
                if let Some(amount) = rx.recv().await {
                    // AcqRel: Release pairs with Acquire load above; Acquire would miss the bump
                    self.credit.fetch_add(amount as isize, Ordering::AcqRel);
                    self.blocked.store(false, Ordering::Release);
                }
            }
        }
    }

    /// Returns whether the sender is currently blocked waiting for credit.
    pub fn is_blocked(&self) -> bool {
        self.blocked.load(Ordering::Acquire)
    }

    /// Returns the current credit balance (may be negative if over-drafted).
    pub fn current_credit(&self) -> isize {
        self.credit.load(Ordering::Acquire)
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
        // Relaxed for pure counter; bump threshold is local, no happens-before needed besides bump send
        let prev = self.counter.fetch_add(1, Ordering::Relaxed);
        if prev + 1 >= self.spec.more_after {
            self.counter.store(0, Ordering::Relaxed);
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
#[path = "../tests/flow.rs"]
mod tests;
