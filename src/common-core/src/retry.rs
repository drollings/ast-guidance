//! Retry and backoff primitives — the single source of truth for the
//! jittered-exponential backoff schedule used across the workspace.
//!
//! Consumers: `RetryBackend` (fluent-router), `SupervisedBatch`
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

/// Compute a capped linear backoff: `base_ms * min(failures + 1, cap)`.
///
/// With `base=5s, cap=12` the schedule is 5, 10, 15, …, 60s — it grows
/// arithmetically (unlike [`backoff_ms`]'s exponential growth) and clamps at
/// `cap` times the base. `failures` is the count of consecutive failures so
/// far (0 = healthy → returns `base_ms`). This is the schedule used by the
/// residency / bootstrap / supervise poll loops.
pub fn capped_backoff_ms(base_ms: u64, failures: u32, cap: u32) -> u64 {
    base_ms.saturating_mul(u64::from(failures.saturating_add(1).min(cap)))
}

/// Compute a capped exponential backoff: `base_ms * 2^min(failures, max_shift)`.
///
/// M3: the no-jitter capped-exponential shape. With `base_ms=1000,
/// max_shift=6` the schedule is 1, 2, 4, …, 64s — the supervisor restart
/// cadence, where jitter would change restart timing. `failures` is the
/// count of consecutive failures so far (0 = first failure → `base_ms`).
/// The shift is clamped to 63 and the multiply saturates, so arbitrarily
/// large inputs can neither panic nor overflow. Do NOT add jitter here;
/// jittered callers use [`backoff_ms`].
pub fn capped_exp_backoff_ms(base_ms: u64, failures: u32, max_shift: u32) -> u64 {
    let shift = failures.min(max_shift).min(63);
    base_ms.saturating_mul(1u64 << shift)
}

/// Outcome of a [`PollWithBackoff`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollResult {
    /// `tick()` returned `true` before exhausting the failure budget.
    Ready,
    /// `tick()` never returned `true` before `max_failures` was exceeded.
    Exhausted { failures: u32 },
}

/// A poll-until-condition loop with capped linear backoff between ticks.
///
/// Runs `tick` immediately; when it returns `true` the loop stops with
/// [`PollResult::Ready`]. Otherwise the failure count is incremented, the
/// loop sleeps `capped_backoff_ms(base, failures, cap)`, and `tick` is tried
/// again. With no `max_failures` the loop runs forever (a supervised
/// forever-task); with a `max_failures` budget it gives up with
/// [`PollResult::Exhausted`]. The caller maps the outcome to its own
/// terminal error (residency log-and-continue vs bootstrap give-up).
pub struct PollWithBackoff {
    base: Duration,
    cap: u32,
    max_failures: Option<u32>,
}

impl PollWithBackoff {
    /// Build a poll loop with `base` interval and a `cap`-times-base ceiling.
    pub fn new(base: Duration, cap: u32) -> Self {
        Self {
            base,
            cap,
            max_failures: None,
        }
    }

    /// Bound the loop: give up after `max` consecutive failed ticks. The
    /// first tick is attempt 1, so `max_failures` is the number of `false`
    /// ticks tolerated before [`PollResult::Exhausted`].
    #[must_use]
    pub fn with_max_failures(self, max: u32) -> Self {
        Self {
            max_failures: Some(max),
            ..self
        }
    }

    /// Poll until `tick()` returns `true`. The tick is an async closure so
    /// pollers that need to `await` (reconcile, `/health`, a residency cycle)
    /// can be expressed directly; it stays monomorphized (no trait object).
    pub async fn run<F, Fut>(&self, mut tick: F) -> PollResult
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = bool>,
    {
        if tick().await {
            return PollResult::Ready;
        }
        let mut failures = 0u32;
        loop {
            failures += 1;
            if let Some(max) = self.max_failures {
                if failures > max {
                    return PollResult::Exhausted { failures };
                }
            }
            let delay = capped_backoff_ms(self.base.as_millis() as u64, failures, self.cap);
            tokio::time::sleep(Duration::from_millis(delay)).await;
            if tick().await {
                return PollResult::Ready;
            }
        }
    }
}

