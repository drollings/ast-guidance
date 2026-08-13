//! First-accept-wins combinator — the canonical "try rungs in order until one
//! produces a result" control flow.
//!
//! Four router sites previously re-implemented this shape by hand
//! (`Ladder::try_escalate`, `BackendChain::complete` +
//! `stream_complete`, the `dispatch_real` fallback walk). This module is the
//! single monomorphized home for it — deliberately **not** a trait-object
//! ladder (a `Vec<Arc<dyn Rung>>` would add a vtable per rung on the
//! escalation path).
//!
//! Rungs are **owned**: the caller passes an `IntoIterator<Item = T>` and each
//! rung is moved into the combinator and then into its future. `run` is a
//! plain `FnMut(T) -> Fut` — no higher-ranked borrow forcing callers to clone
//! every rung into every future. The terminal error is a first-class output,
//! so callers never instrument the loop (no `Mutex`/`Atomic` bookkeeping).
//!
//! Semantics shared by every caller:
//! - rungs are tried in order;
//! - a rung that yields `Ok(Some(result))` wins immediately (`Ok(Some(_))`);
//! - `Ok(None)` from a rung means "not applicable, skip" and the walk
//!   continues;
//! - `Err(e)` is a failure: log-and-continue unless `stop(&e)` returns `true`
//!   (short-circuit — e.g. a non-retryable 4xx), in which case the walk stops
//!   with `Err(e)`;
//! - exhaustion returns `Ok(None)` when every rung skipped cleanly, or
//!   `Err(e)` with the **last** rung error — the caller only supplies a
//!   default terminal error (e.g. `AllBackendsFailed`) for the clean-exhaustion
//!   case.

use std::future::Future;

/// Run `rungs` in order, returning the first `Some(result)`.
///
/// Each rung is executed via `run` (which may `await`). `Ok(None)` skips to
/// the next rung; `Err(e)` continues unless `stop(&e)` short-circuits the
/// whole walk (returning `Err(e)`). Exhaustion returns `Ok(None)` when no rung
/// errored, or `Err(e)` with the last rung error.
pub async fn first_accept_in_order<T, R, E, F, Fut, Stop>(
    rungs: impl IntoIterator<Item = T>,
    mut run: F,
    mut stop: Stop,
) -> Result<Option<R>, E>
where
    F: FnMut(T) -> Fut,
    Fut: Future<Output = Result<Option<R>, E>> + Send,
    Stop: FnMut(&E) -> bool,
{
    let mut last_err: Option<E> = None;
    for rung in rungs {
        match run(rung).await {
            Ok(Some(result)) => return Ok(Some(result)),
            Ok(None) => {}
            Err(e) => {
                if stop(&e) {
                    return Err(e);
                }
                last_err = Some(e);
            }
        }
    }
    match last_err {
        Some(e) => Err(e),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
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
}
