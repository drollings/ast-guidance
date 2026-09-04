use common_core::retry::*;
use std::time::Duration;


#[test]
fn backoff_first_attempt_equals_base() {
        assert_eq!(backoff_ms(100, 1, 0), 100);
        assert_eq!(backoff_ms(1, 1, 0), 1);
        assert_eq!(backoff_ms(300, 1, 0), 300);
}

#[test]
fn backoff_grows_exponentially_without_jitter() {
        assert_eq!(backoff_ms(100, 2, 0), 200);
        assert_eq!(backoff_ms(100, 3, 0), 400);
        assert_eq!(backoff_ms(100, 4, 0), 800);
        assert_eq!(backoff_ms(25, 5, 0), 400);
}

#[test]
fn backoff_jitter_stays_within_bounds() {
        for attempt in 1..=8u32 {
            let base = 100u64;
            let max_jitter = base * 50 / 100;
            for _ in 0..50 {
                let delay = backoff_ms(base, attempt, 50);
                let floor = base << (attempt - 1);
                assert!(
                    delay >= floor && delay <= floor + max_jitter,
                    "attempt {attempt}: {delay} outside [{floor}, {floor}+{max_jitter}]"
                );
            }
        }
}

#[test]
fn backoff_clamps_jitter_pct() {
        // jitter_pct > 100 must be clamped; the deterministic component is
        // unchanged and the jitter stays within `base_ms`.
        for _ in 0..50 {
            let delay = backoff_ms(100, 2, 500);
            assert!(delay >= 200 && delay <= 300, "unclamped jitter: {delay}");
        }
        assert_eq!(backoff_ms(100, 2, 0), 200);
}

    #[tokio::test]
    async fn retry_async_succeeds_on_first_attempt() {
        let result = retry_async(3, 1, 0, |_: &()| true, || async { Ok::<_, ()>(42) }).await;
        assert_eq!(result.unwrap(), 42);
}

    #[tokio::test]
    async fn retry_async_retries_then_succeeds() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let result = retry_async(
            3,
            1,
            0,
            |_| true,
            || async {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err("transient")
                } else {
                    Ok("ok")
                }
            },
        )
        .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
}

    #[tokio::test]
    async fn retry_async_short_circuits_on_non_retryable() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let result: Result<(), &str> = retry_async(
            3,
            1,
            0,
            |e: &&str| *e != "fatal",
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err("fatal")
            },
        )
        .await;
        assert_eq!(result.unwrap_err(), "fatal");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
}

    #[tokio::test]
    async fn retry_async_exhaustion_returns_last_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let result: Result<usize, usize> = retry_async(
            2,
            1,
            0,
            |_| true,
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(calls.load(Ordering::SeqCst))
            },
        )
        .await;
        assert_eq!(result.unwrap_err(), 2);
}

#[test]
fn capped_backoff_progression_and_cap() {
        // base=5s cap=12 → 5,10,15,…,60s
        assert_eq!(capped_backoff_ms(5_000, 0, 12), 5_000);
        assert_eq!(capped_backoff_ms(5_000, 1, 12), 10_000);
        assert_eq!(capped_backoff_ms(5_000, 2, 12), 15_000);
        assert_eq!(capped_backoff_ms(5_000, 11, 12), 60_000);
        // Clamped at the cap for higher failure counts.
        assert_eq!(capped_backoff_ms(5_000, 99, 12), 60_000);
}

#[test]
fn capped_backoff_cap_one_is_base() {
        assert_eq!(capped_backoff_ms(1000, 0, 1), 1000);
        assert_eq!(capped_backoff_ms(1000, 5, 1), 1000);
}

#[test]
fn capped_backoff_base_zero_is_zero() {
        assert_eq!(capped_backoff_ms(0, 3, 12), 0);
}

    #[tokio::test]
    async fn poll_with_backoff_ready_on_first_tick() {
        let poll = PollWithBackoff::new(Duration::from_millis(10), 12);
        assert_eq!(poll.run(|| async { true }).await, PollResult::Ready);
}

    #[tokio::test]
    async fn poll_with_backoff_ready_after_ticks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let poll = PollWithBackoff::new(Duration::from_millis(1), 12);
        let ticks = AtomicUsize::new(0);
        let result = poll
            .run(|| async { ticks.fetch_add(1, Ordering::SeqCst) >= 2 })
            .await;
        assert_eq!(result, PollResult::Ready);
        assert_eq!(ticks.load(Ordering::SeqCst), 3); // initial + 2 after sleeps
}

    #[tokio::test]
    async fn poll_with_backoff_exhaustion_reports_failures() {
        let poll = PollWithBackoff::new(Duration::from_millis(1), 12).with_max_failures(3);
        let result = poll.run(|| async { false }).await;
        assert_eq!(result, PollResult::Exhausted { failures: 4 });
}

    #[tokio::test]
    async fn poll_with_backoff_no_max_runs_forever_until_ready() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let poll = PollWithBackoff::new(Duration::from_millis(1), 12);
        let ticks = AtomicUsize::new(0);
        let result = poll
            .run(|| async { ticks.fetch_add(1, Ordering::SeqCst) >= 5 })
            .await;
        assert_eq!(result, PollResult::Ready);
        assert_eq!(ticks.load(Ordering::SeqCst), 6);
}

#[test]
fn capped_exp_backoff_supervisor_schedule() {
    // M3.2: base=1000ms, max_shift=6 → 1,2,4,8,16,32,64,64,64s in ms.
    let expected = [1_000u64, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 64_000, 64_000];
    for (failures, want) in expected.iter().enumerate() {
        assert_eq!(
            capped_exp_backoff_ms(1_000, failures as u32, 6),
            *want,
            "failures={failures}"
        );
    }
}

#[test]
fn capped_exp_backoff_never_panics_or_overflows() {
    assert_eq!(capped_exp_backoff_ms(1_000, 100, 6), 64_000);
    assert_eq!(capped_exp_backoff_ms(1_000, u32::MAX, 6), 64_000);
    // Absurd shift clamps at 63; the multiply saturates instead of wrapping.
    assert_eq!(capped_exp_backoff_ms(1_000, u32::MAX, u32::MAX), u64::MAX);
    assert_eq!(capped_exp_backoff_ms(u64::MAX, 63, 63), u64::MAX);
    assert_eq!(capped_exp_backoff_ms(0, 5, 6), 0);
    assert_eq!(capped_exp_backoff_ms(5_000, 0, 6), 5_000);
}
