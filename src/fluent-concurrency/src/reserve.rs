//! Available primitive: RAII permit on a shared `AtomicUsize`.
//!
//! Compatibility surface (scaffold) — see ROADMAP_20260901_FIXES_4.md M0
//! Currently not on the critical path — no in-tree consumer outside its own
//! tests. Use `Limiter` for "run this with a permit" patterns and `Reserve`
//! for "acquire now, release later" patterns.

use std::sync::Arc;

/// Compatibility surface (scaffold) — see ROADMAP_20260901_FIXES_4.md M0
/// A RAII permit acquired from a shared `AtomicUsize` counter.
///
/// When dropped without calling `commit()`, the permit is returned to the
/// counter. When `commit()` is called, the permit is consumed permanently.
#[doc(hidden)]
#[allow(dead_code)]
pub struct Reserve {
    counter: Arc<std::sync::atomic::AtomicUsize>,
    committed: bool,
}

impl Reserve {
    /// Attempt to acquire a permit from the counter.
    ///
    /// Returns `None` if the counter is already at zero (no permits available).
    /// Does NOT underflow — this is the safe alternative to `new()`.
    pub fn try_acquire(counter: Arc<std::sync::atomic::AtomicUsize>) -> Option<Self> {
        use std::sync::atomic::Ordering;
        // Linearizable decrement-only-when-nonzero. AcqRel on success pairs
        // with Acquire loads / AcqRel releases in Drop and other acquire
        // attempts; Acquire on failure only needs to observe the latest value.
        let mut cur = counter.load(Ordering::Acquire);
        loop {
            if cur == 0 {
                return None;
            }
            match counter.compare_exchange_weak(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Some(Self {
                        counter,
                        committed: false,
                    })
                }
                Err(v) => cur = v,
            }
        }
    }

    /// Consume the permit permanently. After calling `commit()`, dropping
    /// the `Reserve` will NOT return the permit to the counter.
    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for Reserve {
    fn drop(&mut self) {
        if !self.committed {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
#[path = "../tests/reserve.rs"]
mod tests;
