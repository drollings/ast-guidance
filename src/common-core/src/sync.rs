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

