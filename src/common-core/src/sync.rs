//! Poison-safe locking helpers for `std::sync` primitives.
//!
//! The canonical substitute for the hand-rolled
//! `m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)` pattern:
//! a panic while a lock is held must not wedge the caller permanently, so the
//! guard is recovered from the poison instead of panicking.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lock a `Mutex`, recovering from a poisoned mutex instead of panicking.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Acquire a read guard on a `RwLock`, recovering from a poisoned lock
/// instead of panicking.
pub fn lock_read<T>(m: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Acquire a write guard on a `RwLock`, recovering from a poisoned lock
/// instead of panicking.
pub fn lock_write<T>(m: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_recovers_poisoned_mutex() {
        let m = Mutex::new(42);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = m.lock().unwrap();
            panic!("boom");
        }));
        assert!(err.is_err(), "expected the closure to panic");
        assert_eq!(*lock(&m), 42);
    }

    #[test]
    fn lock_read_and_write_recovers_poisoned_rwlock() {
        let r = RwLock::new(7);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut guard = r.write().unwrap();
            *guard = 9;
            panic!("boom");
        }));
        assert!(err.is_err(), "expected the closure to panic");
        assert_eq!(*lock_read(&r), 9);
        *lock_write(&r) = 11;
        assert_eq!(*lock_read(&r), 11);
    }

    #[test]
    fn lock_returns_value_on_normal_mutex() {
        let m = Mutex::new("hello");
        assert_eq!(*lock(&m), "hello");
    }
}
