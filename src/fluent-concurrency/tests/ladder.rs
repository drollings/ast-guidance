use super::*;

#[tokio::test]
async fn first_accept_wins() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order(
        rungs,
        |r| async move {
            if r == 2 {
                Ok(Some(r * 10))
            } else {
                Ok(None)
            }
        },
        |_: &u8| false,
    )
    .await;
    assert_eq!(out, Ok(Some(20)));
}

#[tokio::test]
async fn skip_continues_to_later_rung() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order(
        rungs,
        |r| async move {
            if r == 3 {
                Ok(Some("last"))
            } else {
                Ok(None)
            }
        },
        |_: &u8| false,
    )
    .await;
    assert_eq!(out, Ok(Some("last")));
}

#[tokio::test]
async fn stop_short_circuits_with_the_trigger_error() {
    let rungs = vec![1u32, 2, 3];
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let out = first_accept_in_order(
        rungs,
        |_| async {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err::<Option<u32>, u8>(7)
        },
        |e: &u8| *e == 7,
    )
    .await;
    assert_eq!(out, Err(7));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn log_and_continue_on_non_short_circuit_error() {
    let rungs = vec![1u32, 2];
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let out = first_accept_in_order(
        rungs,
        |r| {
            let calls = &calls;
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if r == 1 {
                    Err::<Option<u32>, u8>(1) // transient; log-and-continue
                } else {
                    Ok(Some(99))
                }
            }
        },
        |e: &u8| *e == 2, // only 2 short-circuits
    )
    .await;
    assert_eq!(out, Ok(Some(99)));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn clean_exhaustion_returns_ok_none() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order(
        rungs,
        |_| async { Ok::<Option<u32>, u8>(None) },
        |_: &u8| false,
    )
    .await;
    assert_eq!(out, Ok(None));
}

#[tokio::test]
async fn empty_rungs_returns_ok_none() {
    let rungs: Vec<u32> = Vec::new();
    let out = first_accept_in_order(
        rungs,
        |_| async { Ok::<Option<u32>, u8>(Some(1)) },
        |_: &u8| false,
    )
    .await;
    assert_eq!(out, Ok(None));
}

#[tokio::test]
async fn exhaustion_after_error_returns_last_error() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order(
        rungs,
        |r| async move {
            if r == 3 {
                Err::<Option<u32>, u8>(30) // last error, no stop
            } else {
                Ok(None)
            }
        },
        |e: &u8| *e == 99, // never stops
    )
    .await;
    assert_eq!(out, Err(30));
}

#[tokio::test]
async fn owned_rungs_are_moved_into_the_future() {
    // The rung is moved into each future (no per-rung clone); the future
    // may mutate it freely.
    let rungs = vec![vec![1u32], vec![2u32]];
    let out = first_accept_in_order(
        rungs,
        |mut r| async move {
            r.push(9);
            Ok::<Option<Vec<u32>>, u8>(Some(r))
        },
        |_: &u8| false,
    )
    .await;
    assert_eq!(out, Ok(Some(vec![1, 9])));
}

// ── Sync twin: `first_accept_in_order_sync` mirrors the async set ──────

#[test]
fn sync_first_accept_wins() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order_sync(
        rungs,
        |r| {
            if r == 2 {
                Ok(Some(r * 10))
            } else {
                Ok(None)
            }
        },
        |_: &u8| false,
    );
    assert_eq!(out, Ok(Some(20)));
}

#[test]
fn sync_skip_continues_to_later_rung() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order_sync(
        rungs,
        |r| {
            if r == 3 {
                Ok(Some("last"))
            } else {
                Ok(None)
            }
        },
        |_: &u8| false,
    );
    assert_eq!(out, Ok(Some("last")));
}

#[test]
fn sync_stop_short_circuits_with_the_trigger_error() {
    let rungs = vec![1u32, 2, 3];
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let out = first_accept_in_order_sync(
        rungs,
        |_| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err::<Option<u32>, u8>(7)
        },
        |e: &u8| *e == 7,
    );
    assert_eq!(out, Err(7));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn sync_log_and_continue_on_non_short_circuit_error() {
    let rungs = vec![1u32, 2];
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let out = first_accept_in_order_sync(
        rungs,
        |r| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r == 1 {
                Err::<Option<u32>, u8>(1) // transient; log-and-continue
            } else {
                Ok(Some(99))
            }
        },
        |e: &u8| *e == 2, // only 2 short-circuits
    );
    assert_eq!(out, Ok(Some(99)));
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[test]
fn sync_clean_exhaustion_returns_ok_none() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order_sync(
        rungs,
        |_| Ok::<Option<u32>, u8>(None),
        |_: &u8| false,
    );
    assert_eq!(out, Ok(None));
}

#[test]
fn sync_empty_rungs_returns_ok_none() {
    let rungs: Vec<u32> = Vec::new();
    let out = first_accept_in_order_sync(
        rungs,
        |_| Ok::<Option<u32>, u8>(Some(1)),
        |_: &u8| false,
    );
    assert_eq!(out, Ok(None));
}

#[test]
fn sync_exhaustion_after_error_returns_last_error() {
    let rungs = vec![1u32, 2, 3];
    let out = first_accept_in_order_sync(
        rungs,
        |r| {
            if r == 3 {
                Err::<Option<u32>, u8>(30) // last error, no stop
            } else {
                Ok(None)
            }
        },
        |e: &u8| *e == 99, // never stops
    );
    assert_eq!(out, Err(30));
}

#[test]
fn sync_owned_rungs_are_moved_into_the_closure() {
    // The rung is moved into each closure invocation (no per-rung clone);
    // the closure may mutate it freely.
    let rungs = vec![vec![1u32], vec![2u32]];
    let out = first_accept_in_order_sync(
        rungs,
        |mut r| {
            r.push(9);
            Ok::<Option<Vec<u32>>, u8>(Some(r))
        },
        |_: &u8| false,
    );
    assert_eq!(out, Ok(Some(vec![1, 9])));
}

#[tokio::test]
async fn ladder_walk_exhaustion_is_last_err() {
    let rungs = vec![1u32, 2, 3];
    let async_out = first_accept_in_order(
        rungs.clone(),
        |_| async { Err::<Option<u32>, u8>(7) },
        |_: &u8| false,
    )
    .await;
    let sync_out = first_accept_in_order_sync(
        rungs,
        |_| Err::<Option<u32>, u8>(7),
        |_: &u8| false,
    );
    assert_eq!(async_out, Err(7));
    assert_eq!(sync_out, Err(7));
    // Clean exhaustion must be Ok(None) on both
    let async_none = first_accept_in_order(
        vec![1u32, 2],
        |_| async { Ok::<Option<u32>, u8>(None) },
        |_: &u8| false,
    )
    .await;
    let sync_none =
        first_accept_in_order_sync(vec![1u32, 2], |_| Ok::<Option<u32>, u8>(None), |_: &u8| false);
    assert_eq!(async_none, Ok(None));
    assert_eq!(sync_none, Ok(None));
}
