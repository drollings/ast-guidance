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
//!
//! The module ships an **async** combinator ([`first_accept_in_order`]) and a
//! **sync** twin ([`first_accept_in_order_sync`]) with identical semantics.
//! The sync variant exists for non-async ladders — today the `spacy-rs`
//! synchronous annotation ladder (`run_ladder_sync`), whose rungs already
//! return `Result<Option<_>, _>` — so a sync caller gets the same
//! compiler-checked walk instead of hand-rolling the loop.

use std::future::Future;
use std::ops::ControlFlow;

fn finish_walk<R, E>(last_err: Option<E>) -> Result<Option<R>, E> {
    match last_err {
        Some(e) => Err(e),
        None => Ok(None),
    }
}

fn walk_one<R, E>(res: Result<Option<R>, E>, stop: &mut impl FnMut(&E) -> bool, last_err: &mut Option<E>) -> ControlFlow<Result<Option<R>, E>> {
    match res {
        Ok(Some(v)) => ControlFlow::Break(Ok(Some(v))),
        Ok(None) => ControlFlow::Continue(()),
        Err(e) => {
            if stop(&e) {
                ControlFlow::Break(Err(e))
            } else {
                *last_err = Some(e);
                ControlFlow::Continue(())
            }
        }
    }
}

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
        match walk_one(run(rung).await, &mut stop, &mut last_err) {
            ControlFlow::Break(res) => return res,
            ControlFlow::Continue(()) => {}
        }
    }
    finish_walk(last_err)
}

/// The synchronous twin of [`first_accept_in_order`], mirroring its semantics
/// exactly: `Ok(Some)` wins; `Ok(None)` skips; `Err(e)` continues unless
/// `stop(&e)` short-circuits the walk with `Err(e)`; exhaustion returns the
/// last rung error or `Ok(None)` when every rung skipped cleanly. `run` is
/// synchronous (`FnMut(T) -> Result<Option<R>, E>`), so it cannot `await`.
pub fn first_accept_in_order_sync<T, R, E, F, Stop>(
    rungs: impl IntoIterator<Item = T>,
    mut run: F,
    mut stop: Stop,
) -> Result<Option<R>, E>
where
    F: FnMut(T) -> Result<Option<R>, E>,
    Stop: FnMut(&E) -> bool,
{
    let mut last_err: Option<E> = None;
    for rung in rungs {
        match walk_one(run(rung), &mut stop, &mut last_err) {
            ControlFlow::Break(res) => return res,
            ControlFlow::Continue(()) => {}
        }
    }
    finish_walk(last_err)
}

#[cfg(test)]
#[path = "../tests/ladder.rs"]
mod tests;
