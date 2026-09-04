//! The escalation ladder: deterministic context-cache first, then a configured
//! sequence of frontier/LLM modes (filter → question → team → turnover) until
//! one accepts the request, else `None` (the caller falls back).
//!
//! Split across [`modes`] (the four mode implementations + the shared frontier
//! transport), [`audit`] (the `emit_audit` record builder), and [`assemble`]
//! (the parse/assemble/scorer helpers the modes drive).

pub mod assemble;
pub mod audit;
pub mod modes;

use std::sync::{Arc, Mutex};

use fluent_concurrency::ladder::first_accept_in_order;

use crate::config::EscalationLadderConfig;
use crate::dag_session::DependencySession;
use crate::dispatch::frontier::DispatchError;
use crate::frontier::modes::EscalationMode;
use crate::server::responses::{completion_to_response, make_text_completion, HyperResponse};
use crate::types::RouterRequest;

/// A sync local-model backend — the role engines (decomposer/assembler/
/// classifier/draft/judge) run against `fluent_llm::ChatBackend`.
pub type LocalBackend = Arc<dyn fluent_llm::client::ChatBackend>;

/// The async frontier backend — every actual frontier HTTP call goes through
/// the canonical `dispatch/backend.rs` transport.
pub type FrontierBackend = Arc<dyn crate::dispatch::backend::DispatchBackend>;

/// Backends for every ladder role. Roles that are `None` disable the modes
/// that require them.
pub struct EscalationBackends {
    /// Frontier transport (required for every mode).
    pub frontier: FrontierBackend,
    /// Question-mode: breaks the anonymized query into hypotheticals.
    pub decomposer: Option<LocalBackend>,
    /// Question-mode: synthesizes + scores the final answer.
    pub assembler: Option<LocalBackend>,
    /// Team-mode: parallel classifier slots that vote on approach.
    pub classifier: Option<LocalBackend>,
    /// Team-mode: decomposes into subtasks + attempts a local draft.
    pub draft: Option<LocalBackend>,
    /// Team-mode: crafts the frontier prompt from the gap + judges the result.
    pub judge: Option<LocalBackend>,
}

/// Per-request escalation context (the `ctx_deps` bundle the ladder needs).
pub struct EscalationContext<'a> {
    /// The full normalized request.
    pub request: &'a RouterRequest,
    /// The last user message text.
    pub user_text: &'a str,
    /// The model name the client requested (used on the completion response).
    pub model_name: &'a str,
    /// Deterministic-fact cache consulted before any frontier call.
    pub context_cache: Option<&'a Arc<dyn fluent_types::ContextCache>>,
    /// The session to mark frontier-owned after a turnover handoff.
    pub session: Option<&'a Arc<Mutex<DependencySession>>>,
}

/// A configured escalation ladder for one model group.
pub struct Ladder {
    pub(super) config: EscalationLadderConfig,
    pub(super) backends: EscalationBackends,
}

impl Ladder {
    pub fn new(config: EscalationLadderConfig, backends: EscalationBackends) -> Self {
        Self { config, backends }
    }

    pub fn config(&self) -> &EscalationLadderConfig {
        &self.config
    }

    /// Direct frontier dispatch of the full request — the bypass path for a
    /// session the turnover mode already marked frontier-owned.
    /// Buffered (multi-step escalation is not streamable); `None` on failure.
    pub async fn dispatch_frontier(&self, ctx: &EscalationContext<'_>) -> Option<HyperResponse> {
        self.turnover_mode(ctx).await.ok().flatten()
    }

    /// Attempt to resolve the request through the ladder.
    ///
    /// Returns `Some(response)` when a rung accepted the request (or the
    /// context cache short-circuited); `None` when every configured mode was
    /// rejected or skipped — the caller then returns `fallback_completion`.
    pub async fn try_escalate(&self, ctx: &EscalationContext<'_>) -> Option<HyperResponse> {
        // Deterministic-first: a context hit short-circuits before any
        // frontier call (VISION §"Post-processing": the same question never
        // pays frontier cost twice).
        if let Some(cache) = ctx.context_cache {
            if let Some(hit) = cache.lookup(ctx.user_text) {
                tracing::info!(
                    target: "router.dispatch.escalation",
                    mode = "context",
                    source = %hit.source,
                    score = hit.score,
                    "context cache short-circuit"
                );
                audit::emit_audit(
                    "context",
                    true,
                    ctx.user_text,
                    &hit.content,
                    "context cache hit",
                    &serde_json::json!({ "source": hit.source, "score": hit.score }),
                );
                let completion = make_text_completion(ctx.model_name, &hit.content);
                return Some(completion_to_response(
                    &completion,
                    ctx.model_name,
                    false,
                    None,
                ));
            }
        }

        // Owned rungs (a ≤4-mode Copy enum): moved into each future, so the
        // closure borrows `self` directly — no per-rung ladder clone. Errors
        // are log-and-continue; exhaustion/`Err` maps to `None`-on-exhaustion.
        first_accept_in_order(
            self.config.modes.clone(),
            |mode| async move {
                match self.run_mode(mode, ctx).await {
                    Ok(result) => Ok(result),
                    Err(e) => {
                        tracing::warn!(
                            target: "router.dispatch.escalation",
                            mode = ?mode,
                            error = %e,
                            retryable = e.is_retryable(),
                            "escalation mode failed"
                        );
                        Err(e)
                    }
                }
            },
            |_: &DispatchError| false,
        )
        .await
        .ok()
        .flatten()
    }

    async fn run_mode(
        &self,
        mode: EscalationMode,
        ctx: &EscalationContext<'_>,
    ) -> Result<Option<HyperResponse>, DispatchError> {
        match mode {
            EscalationMode::Filter => self.filter_mode(ctx).await,
            EscalationMode::Question => self.question_mode(ctx).await,
            EscalationMode::Team => self.team_mode(ctx).await,
            EscalationMode::Turnover => self.turnover_mode(ctx).await,
        }
    }
}
#[cfg(test)]
#[path = "../../../tests/dispatch_escalation_mod.rs"]
mod tests;
