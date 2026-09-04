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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reserve_concurrent_try_acquire_on_one() {
    // Concurrent with one permit held: exactly one holder, rest None.
    let counter = Arc::new(AtomicUsize::new(1));
    let r = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    let kept = Some(r);
    let mut handles = Vec::new();
    for _ in 0..64 {
        let c = Arc::clone(&counter);
        handles.push(tokio::spawn(async move { Reserve::try_acquire(c).is_some() }));
    }
    let mut s2 = 0;
    for h in handles {
        if h.await.unwrap() {
            s2 += 1;
        }
    }
    assert_eq!(s2, 0, "no concurrent acquire while one is held");
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    assert_ne!(counter.load(Ordering::SeqCst), usize::MAX);
    drop(kept.unwrap());
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    // Stress: 64 tasks racing for 1 permit without holder — exactly one wins.
    let counter2 = Arc::new(AtomicUsize::new(1));
    let barrier = Arc::new(tokio::sync::Barrier::new(64));
    let mut handles2 = Vec::new();
    for _ in 0..64 {
        let c = Arc::clone(&counter2);
        let b = Arc::clone(&barrier);
        handles2.push(tokio::spawn(async move {
            b.wait().await;
            Reserve::try_acquire(c)
        }));
    }
    let mut reserves = Vec::new();
    for h in handles2 {
        if let Some(r) = h.await.unwrap() {
            reserves.push(r);
        }
    }
    assert_eq!(reserves.len(), 1, "exactly one acquire should succeed with counter=1");
    assert_eq!(counter2.load(Ordering::SeqCst), 0);
    assert_ne!(counter2.load(Ordering::SeqCst), usize::MAX);
    drop(reserves);
    assert_eq!(counter2.load(Ordering::SeqCst), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reserve_never_underflows_to_max() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..64 {
        let c = Arc::clone(&counter);
        handles.push(tokio::spawn(async move { Reserve::try_acquire(c).is_some() }));
    }
    let mut successes = 0;
    for h in handles {
        if h.await.unwrap() {
            successes += 1;
        }
    }
    assert_eq!(successes, 0, "all should be None when counter=0");
    assert_eq!(counter.load(Ordering::SeqCst), 0);
    assert_ne!(counter.load(Ordering::SeqCst), usize::MAX);
}
