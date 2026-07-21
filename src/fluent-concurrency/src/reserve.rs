use std::sync::Arc;

/// A RAII permit acquired from a shared `AtomicUsize` counter.
///
/// When dropped without calling `commit()`, the permit is returned to the
/// counter. When `commit()` is called, the permit is consumed permanently.
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
        let prev = counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        if prev == 0 {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            None
        } else {
            Some(Self {
                counter,
                committed: false,
            })
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
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_reserve_acquire_and_drop_releases() {
        let counter = Arc::new(AtomicUsize::new(2));
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        {
            let _reserve = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_reserve_commit_does_not_release() {
        let counter = Arc::new(AtomicUsize::new(2));
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        let reserve = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        reserve.commit();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_reserve_multiple_acquires() {
        let counter = Arc::new(AtomicUsize::new(2));
        let r1 = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        let r2 = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        r1.commit();
        drop(r2);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_reserve_exhaustion_returns_none() {
        let counter = Arc::new(AtomicUsize::new(0));
        let result = Reserve::try_acquire(Arc::clone(&counter));
        assert!(result.is_none());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
