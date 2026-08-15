//! Cooperative cancellation for long-lived streams.
//!
//! [`StreamAbort`] is a sticky, `Clone` cancellation signal: it fires once,
//! later waiters observe it immediately, and every waiter is woken. It carries
//! no I/O — transports drop their connections and management planes issue
//! explicit aborts as a *reaction* to it. `Clone` is cheap (an `Arc` bump), so
//! every task that touches a stream — the forwarding task, the downstream
//! body-drop guard, the management-abort watcher — holds its own clone and
//! observes the same state.
//!
//! The canonical wiring: the body the HTTP server hands to the client is
//! wrapped so its `Drop` fires `cancel()` — the body's lifetime is the single
//! source of truth for "the consumer is gone." The forwarding task
//! `select!`s between its upstream read and `cancelled()`, so a downstream
//! disconnect drops the upstream connection (and may trigger a management-plane
//! abort) instead of draining the upstream to the end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

/// A sticky, `Clone` cancellation signal for a long-lived stream.
///
/// Fire it with [`StreamAbort::cancel`] (idempotent); wait on it with
/// [`StreamAbort::cancelled`] (resolves immediately if already fired); inspect
/// it with [`StreamAbort::is_cancelled`].
#[derive(Clone, Default)]
pub struct StreamAbort {
    state: Arc<AbortState>,
}

#[derive(Default)]
struct AbortState {
    fired: AtomicBool,
    notify: Notify,
}

impl StreamAbort {
    /// A fresh, unfired cancellation signal.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether cancellation has fired.
    pub fn is_cancelled(&self) -> bool {
        self.state.fired.load(Ordering::Acquire)
    }

    /// Fire the signal (idempotent). Wakes every currently- and future-waiting
    /// task; a waiter that registers after this resolves immediately.
    pub fn cancel(&self) {
        if self.state.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        self.state.notify.notify_waiters();
    }

    /// Resolve once the signal has fired. Sticky: a caller that registers
    /// after `cancel()` resolves immediately, never blocking.
    ///
    /// The `notified()` future is created *before* the second `is_cancelled`
    /// check and awaited after it, closing the lost-wakeup race: `Notify` does
    /// not store a permit, so a waiter must register before the notifier fires.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let notified = self.state.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_unfired() {
        let abort = StreamAbort::new();
        assert!(!abort.is_cancelled());
    }

    #[test]
    fn cancel_fires_and_is_idempotent() {
        let abort = StreamAbort::new();
        abort.cancel();
        abort.cancel();
        assert!(abort.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_resolves_after_cancel() {
        let abort = StreamAbort::new();
        let waiter = abort.clone();
        let task = tokio::spawn(async move {
            waiter.cancelled().await;
            true
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        abort.cancel();
        assert!(task.await.expect("waiter completes"));
    }

    #[tokio::test]
    async fn cancelled_is_sticky_for_late_waiters() {
        let abort = StreamAbort::new();
        abort.cancel();
        // A waiter that registers after `cancel` must not block.
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            abort.cancelled(),
        )
        .await
        .expect("late waiter resolves immediately");
    }

    #[tokio::test]
    async fn cancel_wakes_every_waiter() {
        let abort = StreamAbort::new();
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let waiter = abort.clone();
            tasks.push(tokio::spawn(async move {
                waiter.cancelled().await;
            }));
        }
        abort.cancel();
        for t in tasks {
            t.await.expect("all waiters woken");
        }
    }

    #[tokio::test]
    async fn uncancelled_waiter_blocks_until_timeout() {
        let abort = StreamAbort::new();
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(20),
            abort.cancelled(),
        )
        .await;
        assert!(result.is_err(), "uncancelled waiter must not resolve");
    }
}
