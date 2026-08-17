//! Opt-in live-AI smoke test for the fluent-llm chat client.
//!
//! This test performs a REAL model call. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make llm-test-live`).
//!
//! Env contract (see `tests/live/README.md`):
//! - `LLM_BASE_URL` — OpenAI-compatible chat-completions base URL.
//! - `LLM_MODEL` — model name to request.
//!
//! When either variable is absent the test skips cleanly (early `return`,
//! never panic) per the roadmap's skip-not-fail policy. Assertions are
//! structural only (well-formed, bounded response text) — never model output
//! quality.

use fluent_llm::{block_on, ChatMessage, LlmClient};

/// `LLM_BASE_URL` and `LLM_MODEL` must both be set; otherwise `None`.
fn live_env() -> Option<(String, String)> {
    let base = std::env::var("LLM_BASE_URL").ok()?;
    let model = std::env::var("LLM_MODEL").ok()?;
    Some((base, model))
}

#[test]
#[ignore = "live-AI: requires LLM_BASE_URL + LLM_MODEL; run via `make test-live`"]
fn smoke_live_chat_completion_structural() {
    let Some((base, model)) = live_env() else {
        eprintln!("LLM_BASE_URL/LLM_MODEL not set; skipping live smoke test");
        return;
    };

    let client = LlmClient::new(&base, &model);
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "Reply with the single word: ok".into(),
    }];

    let response = block_on(client.chat_complete_async(&messages))
        .expect("live chat completion should return Ok");

    // Structural invariants only: non-empty, plausibly bounded text.
    assert!(!response.trim().is_empty(), "live response must not be empty");
    assert!(
        response.len() < 1_000_000,
        "live response unexpectedly large ({} bytes)",
        response.len()
    );
}
