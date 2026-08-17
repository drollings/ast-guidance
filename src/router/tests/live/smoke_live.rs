//! Opt-in live-AI smoke test for the fluent-router classifier pipeline.
//!
//! This test performs a REAL model call. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make router-test-live`).
//!
//! Env contract (see `tests/live/README.md`):
//! - `LLM_BASE_URL` — OpenAI-compatible chat-completions base URL.
//! - `LLM_MODEL` — model name to request.
//!
//! When either variable is absent the test skips cleanly (early `return`,
//! never panic) per the roadmap's skip-not-fail policy. Assertions are
//! structural only (stage order, valid pipeline outcome) — never the
//! classifier's routing decision quality.

use std::sync::Arc;

use fluent_llm::client::{ChatBackend, LlmClient};
use fluent_router::config::RouterConfig;
use fluent_router::pipeline::{PipelineOrchestrator, PipelineResult};
use fluent_router::pipeline_types::PipelineStage;
use fluent_router::testing::test_request;
use fluent_wvr::work::{WorkContext, WorkError};
use fluent_wvr::WorkUnit;

/// `LLM_BASE_URL` and `LLM_MODEL` must both be set; otherwise `None`.
fn live_env() -> Option<(String, String)> {
    let base = std::env::var("LLM_BASE_URL").ok()?;
    let model = std::env::var("LLM_MODEL").ok()?;
    Some((base, model))
}

/// Minimal two-stage config. The declared model's endpoint is never reached:
/// the classifier uses the injected real `LlmClient` backend, so the endpoint
/// is a refused loopback (structural safety: this test must never dial a real
/// host itself).
fn make_live_config() -> RouterConfig {
    serde_json::from_str(
        r#"{
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true, "blacklist": "env/pii-patterns.json"}},
        "models": {"fast": {"endpoint": "http://127.0.0.1:1/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.000001, "cost_output": 0.000006, "cost_cached_read": 0.0000004, "speed": 10, "total_timeout_ms": 5000, "idle_timeout_ms": 2000, "stream": false, "filter_thinking": false, "retry_count": 0, "retry_base_interval_s": 1}},
        "model_groups": {"fast": ["fast"]},
        "routes": {"fast": {"group": "fast", "pipelines": ["default"]}},
        "default_route": "fast"
    }"#,
    )
    .expect("live config JSON must deserialize")
}

fn run_pipeline(pipeline: &PipelineOrchestrator, text: &str) -> Result<PipelineResult, WorkError> {
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", &test_request(text));
    let output = pipeline.execute(&ctx)?;
    output
        .data_take()
        .map_err(|e| WorkError::Execution(e.to_string()))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live-AI: requires LLM_BASE_URL + LLM_MODEL; run via `make test-live`"]
async fn smoke_live_classifier_dispatch_structural() {
    let Some((base, model)) = live_env() else {
        eprintln!("LLM_BASE_URL/LLM_MODEL not set; skipping live smoke test");
        return;
    };

    let config = make_live_config();
    let backend: Arc<dyn ChatBackend> = Arc::new(LlmClient::new(&base, &model));
    let pipeline = config
        .build_named_pipeline_with_backend("default", Some(backend))
        .expect("live pipeline should build with a real LLM backend");

    let result = run_pipeline(&pipeline, "What is 2+2?").expect("pipeline should complete");

    // Structural invariants only: not rejected, and the deterministic →
    // classifier stage order is observed.
    assert!(!result.rejected, "a plain question must not be rejected");
    let stage_order: Vec<PipelineStage> = result.decisions.iter().map(|d| d.stage).collect();
    assert!(
        stage_order.contains(&PipelineStage::DeterministicPreFilter),
        "pipeline must run the deterministic pre-filter"
    );
    assert!(
        stage_order.contains(&PipelineStage::Classifier),
        "pipeline must run the classifier stage"
    );
    assert!(
        stage_order
            .windows(2)
            .any(|w| w[0] == PipelineStage::DeterministicPreFilter && w[1] == PipelineStage::Classifier),
        "deterministic pre-filter must precede the classifier"
    );

    // The pipeline must resolve to either a routing target or a direct answer
    // (the real model may legitimately do either).
    assert!(
        result.routing_target.is_some()
            || result
                .classifier_response
                .as_ref()
                .is_some_and(|s| !s.is_empty())
            || result.final_response.as_ref().is_some_and(|s| !s.is_empty()),
        "pipeline must produce a routing target or an answer"
    );
}
