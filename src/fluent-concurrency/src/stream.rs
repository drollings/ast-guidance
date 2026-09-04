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
#[path = "../tests/stream.rs"]
mod tests;
