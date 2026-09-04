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
