//! Router-local LLM-JSON round-trip codec.
//!
//! Every router feature that asks an LLM for JSON repeats the same envelope:
//! build a `Vec<ChatMessage>` prompt, call a [`ChatBackend`], then coerce the
//! tolerant parse into a typed value. The generic parse/coerce half lives in
//! `fluent_llm::parse::parse_typed`; this module owns the router-facing
//! call+parse envelope and its unified error, because it depends on
//! `fluent_llm::client::ChatBackend`/`ChatMessage` and (for coercion callers)
//! the router's `stages::common` helpers. It cannot live in `fluent-llm`
//! without dragging router concerns upward.
//!
//! Note on async: `ChatBackend::chat_complete` is synchronous, so `chat_json`
//! is a plain `fn` — an `async` wrapper over a blocking call would add a
//! pointless `.await` and force every caller to run it through the sync→async
//! bridge. Sites that need concurrency bounding (classifier, tree) keep their
//! `Limiter::run_sync` wrapper around `chat_complete` and call `parse_typed`
//! directly; `chat_json` is for the sites whose round-trip is a bare call.

use std::sync::Arc;

use serde::de::DeserializeOwned;

use fluent_llm::client::ChatBackend;
use fluent_llm::{parse::parse_typed, ChatMessage};

/// Errors produced by the [`chat_json`] round-trip: the LLM call failed, or
/// the returned text did not coerce into the target type.
#[derive(Debug, thiserror::Error)]
pub enum PromptParseError {
    /// The `ChatBackend` call itself failed.
    #[error("LLM call failed: {0}")]
    Call(String),
    /// The backend returned text that did not parse/coerce.
    #[error("LLM output did not parse: {0}")]
    Parse(String),
}

/// The repeated LLM-JSON round-trip: call `backend` with `messages`, then
/// coerce the tolerant parse into `T`.
///
/// See [`parse_typed`] for the parse semantics (`defaults` merge for missing
/// object fields, then `sanitize` field coercion, then deserialize). The
/// caller maps [`PromptParseError`] onto its own error type.
pub fn chat_json<T>(
    backend: &Arc<dyn ChatBackend>,
    messages: &[ChatMessage],
    defaults: &serde_json::Value,
    sanitize: impl FnOnce(&mut serde_json::Value),
) -> Result<T, PromptParseError>
where
    T: DeserializeOwned,
{
    let raw = backend
        .chat_complete(messages)
        .map_err(|e| PromptParseError::Call(e.to_string()))?;
    parse_typed(&raw, defaults, sanitize).map_err(|e| PromptParseError::Parse(e.to_string()))
}
