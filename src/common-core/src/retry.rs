//! Retry and backoff primitives — the single source of truth for the
//! jittered-exponential backoff schedule used across the workspace.
//!
//! Consumers: `RetryChatBackend` (fluent-router), `Zone`
//! (fluent-concurrency), and `fluent_wvr::retry_call` (the synchronous
//! counterpart for non-async callers). The `async` loop below is the
//! transport-retry helper; corrective re-prompting (e.g. `RetryClassifier`)
//! is a distinct concern and does not use this module.

use std::future::Future;
use std::time::Duration;

/// Compute the jittered-exponential backoff delay for a given attempt.
///
/// Delay: `base_ms * 2^(attempt-1)` plus a jitter in
/// `0..=base_ms*jitter_pct/100`. `jitter_pct` is clamped to 0–100.
/// `attempt` is 1-based: the first retry (`attempt = 1`) sleeps `≈ base_ms`,
/// the second `≈ 2*base_ms`, the third `≈ 4*base_ms`, and so on.
pub fn backoff_ms(base_ms: u64, attempt: u32, jitter_pct: u32) -> u64 {
    let exponent = attempt.saturating_sub(1).min(63);
    let exp = base_ms.saturating_mul(1u64 << exponent);
    let pct = jitter_pct.min(100);
    let jitter_max = base_ms.saturating_mul(u64::from(pct)) / 100;
    let jitter = if jitter_max > 0 {
        fastrand::u64(0..=jitter_max)
    } else {
        0
    };
    exp.saturating_add(jitter)
}

/// Retry an async operation with jittered-exponential backoff.
///
/// Runs `op` up to `max_attempts` times (must be ≥ 1). A non-retryable
/// error (per `is_retryable`) short-circuits immediately; a retryable
/// error sleeps `backoff_ms(base_ms, attempt, jitter_pct)` before the
/// next attempt. On exhaustion, the last error is returned.
pub async fn retry_async<F, Fut, T, E>(
    max_attempts: u32,
    base_ms: u64,
    jitter_pct: u32,
    is_retryable: impl Fn(&E) -> bool,
    mut op: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    assert!(max_attempts >= 1);
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match op().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                if attempt >= max_attempts || !is_retryable(&e) {
                    return Err(e);
                }
                let delay = backoff_ms(base_ms, attempt, jitter_pct);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
