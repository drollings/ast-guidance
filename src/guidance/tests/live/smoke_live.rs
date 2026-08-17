//! Opt-in live-AI smoke test for the guidance-core Enhancer.
//!
//! This test performs a REAL model call. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make guidance-test-live`).
//!
//! Env contract (see `tests/live/README.md`):
//! - `LLM_BASE_URL` — OpenAI-compatible chat-completions base URL.
//! - `LLM_MODEL` — model name to request.
//!
//! When either variable is absent the test skips cleanly (early `return`,
//! never panic) per the roadmap's skip-not-fail policy. Assertions are
//! structural only (well-formed, bounded comment text) — never generation
//! quality.

use guidance_core::enhancer::Enhancer;

/// `LLM_BASE_URL` and `LLM_MODEL` must both be set; otherwise `None`.
fn live_env() -> Option<(String, String)> {
    let base = std::env::var("LLM_BASE_URL").ok()?;
    let model = std::env::var("LLM_MODEL").ok()?;
    Some((base, model))
}

#[test]
#[ignore = "live-AI: requires LLM_BASE_URL + LLM_MODEL; run via `make test-live`"]
fn smoke_live_enhance_function_structural() {
    let Some((base, model)) = live_env() else {
        eprintln!("LLM_BASE_URL/LLM_MODEL not set; skipping live smoke test");
        return;
    };

    let enhancer = Enhancer::new(&base, &model);
    let comment = enhancer
        .enhance_function(
            "add",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }",
            "math",
            "rust",
        )
        .expect("live enhance should return Ok");

    // Structural invariants only: a comment was produced and is plausibly sized.
    let text = comment.unwrap_or_default();
    assert!(!text.trim().is_empty(), "live enhancement must not be empty");
    assert!(
        text.len() < 1_000_000,
        "live enhancement unexpectedly large ({} bytes)",
        text.len()
    );
}
