use common_core::sync::*;
use std::sync::{Mutex, RwLock};


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
