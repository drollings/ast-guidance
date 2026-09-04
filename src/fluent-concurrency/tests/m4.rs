use super::*;
use crate::flow;

#[tokio::test(start_paused = true)]
async fn test_credit_flow_initial_credit_allows_n_sends() {
    let spec = flow::CreditSpec {
        initial: 3,
        more_after: 2,
    };
    let (sender, _receiver) = flow::new(spec);
    let counter = Arc::new(AtomicUsize::new(0));
    sender
        .send(|| async {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .await;
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_credit_flow_receiver_bumps_after_more_after() {
    let spec = flow::CreditSpec {
        initial: 3,
        more_after: 2,
    };
    let (sender, receiver) = flow::new(spec);

    let counter = Arc::new(AtomicUsize::new(0));
    for _ in 0..3 {
        let cnt = Arc::clone(&counter);
        sender
            .send(|| async move {
                cnt.fetch_add(1, Ordering::SeqCst);
            })
            .await;
    }
    assert_eq!(counter.load(Ordering::SeqCst), 3);

    receiver.recv();
    receiver.recv();

    let cnt = Arc::clone(&counter);
    sender
        .send(|| async move {
            cnt.fetch_add(1, Ordering::SeqCst);
        })
        .await;
    assert_eq!(counter.load(Ordering::SeqCst), 4);
}

#[tokio::test(start_paused = true)]
async fn test_credit_flow_sender_blocks_when_exhausted() {
    let spec = flow::CreditSpec {
        initial: 1,
        more_after: 2,
    };
    let (sender, receiver) = flow::new(spec);

    sender.send(|| async {}).await;

    let cnt = Arc::new(AtomicUsize::new(0));
    let cnt_clone = Arc::clone(&cnt);
    let handle = tokio::spawn(async move {
        sender
            .send(|| async move {
                cnt_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await;
    });

    receiver.recv();
    receiver.recv();

    handle.await.unwrap();
    assert_eq!(cnt.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_credit_flow_end_to_end() {
    let spec = flow::CreditSpec {
        initial: 5,
        more_after: 3,
    };
    let (sender, receiver) = flow::new(spec);

    let sent = Arc::new(AtomicUsize::new(0));
    let received = Arc::new(AtomicUsize::new(0));

    let s = Arc::clone(&sent);
    let r = Arc::clone(&received);

    let producer = tokio::spawn(async move {
        for _ in 0..10 {
            let s = Arc::clone(&s);
            sender
                .send(|| async move {
                    s.fetch_add(1, Ordering::SeqCst);
                })
                .await;
        }
    });

    let consumer = tokio::spawn(async move {
        for _ in 0..10 {
            receiver.recv();
            r.fetch_add(1, Ordering::SeqCst);
        }
    });

    producer.await.unwrap();
    consumer.await.unwrap();

    assert_eq!(sent.load(Ordering::SeqCst), 10);
    assert_eq!(received.load(Ordering::SeqCst), 10);
}

#[tokio::test(start_paused = true)]
async fn test_credit_flow_is_blocked_and_current_credit() {
    tokio::time::resume();
    let spec = flow::CreditSpec {
        initial: 2,
        more_after: 1,
    };
    let (sender, receiver) = flow::new(spec);
    assert_eq!(sender.current_credit(), 2);
    assert!(!sender.is_blocked());

    sender.send(|| async {}).await;
    assert_eq!(sender.current_credit(), 1);
    assert!(!sender.is_blocked());

    sender.send(|| async {}).await;
    assert_eq!(sender.current_credit(), 0);

    // Now sender is blocked, wrap in Arc for shared access
    let sender = std::sync::Arc::new(sender);
    let sender_clone = std::sync::Arc::clone(&sender);
    let handle = tokio::spawn(async move {
        sender_clone.send(|| async {}).await;
        assert!(!sender_clone.is_blocked());
    });

    // Allow the spawned task to start and reach the blocking point
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(sender.is_blocked());

    receiver.recv();
    handle.await.unwrap();
    // Credit went 0 -> +1 (bump) -> 0 (consumed by send)
    assert_eq!(sender.current_credit(), 0);
    assert!(!sender.is_blocked());
}
