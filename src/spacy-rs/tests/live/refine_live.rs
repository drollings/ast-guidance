//! Opt-in live-AI test for the span-scoped refine seam (ROADMAP_20260831
//! _ARCEAGER M2.5).
//!
//! This test performs a REAL model call. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make spacy-test-live`).
//!
//! Env contract (see `README.md` alongside `smoke_live.rs`):
//! - `SPACY_LIVE_LLM_URL` — an OpenAI-compatible chat-completions endpoint
//!   that answers the [`spacy_rs::LlmRefinePrompt`] contract with a
//!   corrections-object reply.
//!
//! When the variable is absent the test skips cleanly (early `return`, never
//! panic). Assertions are structural only — the ladder lands on a validated
//! parse (base or refined), never an error — never annotation quality.

use fluent_llm::protocol::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};
use spacy_rs::pipeline::{
    AnnotateError, LlmRefineFetchSync, LlmRefineRequest, NlpPipeline, RefineMode, RefinePolicy,
    RefineSeams,
};
use spacy_rs::{AnnotationSource, LlmRefinePrompt};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The live-endpoint URL; unset (or empty) skips the test.
fn live_url() -> Option<String> {
    std::env::var("SPACY_LIVE_LLM_URL")
        .ok()
        .filter(|u| !u.is_empty())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live-AI: requires SPACY_LIVE_LLM_URL; run via `make test-live`"]
async fn focused_refinement_runs_through_a_live_endpoint() {
    let Some(url) = live_url() else {
        eprintln!("skipping: SPACY_LIVE_LLM_URL unset");
        return;
    };

    let rt = fluent_concurrency::tokio_runtime();
    let client = reqwest::Client::new();
    let queue = Arc::new(LlmRequestQueue::new(
        rt,
        &LlmQueueConfig {
            worker_count: 1,
            queue_capacity: 4,
        },
        move |task: LlmTask| {
            let client = client.clone();
            let endpoint = url.clone();
            async move {
                let body = serde_json::json!({
                    "model": task.config.model,
                    "messages": task.messages.iter().map(|m| {
                        serde_json::json!({"role": m.role, "content": m.content})
                    }).collect::<Vec<_>>(),
                });
                let resp = client
                    .post(&endpoint)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| LlmError::Http(e.to_string()))?;
                let text = resp
                    .text()
                    .await
                    .map_err(|e| LlmError::Http(e.to_string()))?;
                let value: serde_json::Value =
                    serde_json::from_str(&text).map_err(|e| LlmError::Api(e.to_string()))?;
                if value.get("choices").is_some() {
                    value["choices"][0]["message"]["content"]
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| LlmError::Api("no content in reply".into()))
                } else {
                    Ok(text)
                }
            }
        },
    ));

    let consults = Arc::new(AtomicUsize::new(0));
    let consults2 = Arc::clone(&consults);
    let handle = tokio::runtime::Handle::current();
    // The sync bridge (the router's `Limiter::run_sync` pattern): the ladder
    // calls the seam from a blocking thread, which drives the queued call to
    // completion on the multi-thread runtime.
    let refine: LlmRefineFetchSync = Arc::new(move |req: LlmRefineRequest| {
        let queue = Arc::clone(&queue);
        consults2.fetch_add(1, Ordering::SeqCst);
        let task = LlmTask {
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: req.prompt.clone(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Correct only the FOCUS tokens; reply with the JSON object.".into(),
                },
            ],
            config: LlmConfig::new()
                .api_url("http://127.0.0.1:1".into())
                .model("spacy-refiner".to_string())
                .timeout_ms(60_000)
                .build(),
        };
        handle
            .block_on(queue.submit(task))
            .map_err(|e| AnnotateError::Fetch(e.to_string()))
    });

    let pipeline = Arc::new(NlpPipeline::en_default().expect("pipeline"));
    let seams = RefineSeams {
        llm_focused: Some(refine),
        ..RefineSeams::default()
    };
    let (doc, result) = tokio::task::spawn_blocking(move || {
        pipeline.process_sync_with_refine(
            "The cat sat on the mat .",
            None,
            None,
            &seams,
            None,
            RefinePolicy {
                mode: RefineMode::OnUncertain,
                min_token_score: 1.01,
                ..RefinePolicy::default()
            },
        )
    })
    .await
    .expect("join")
    .expect("the ladder always lands on a validated parse");

    assert_eq!(doc.len(), 7);
    assert_eq!(result.records().len(), doc.len());
    assert!(
        matches!(result.source(), AnnotationSource::ArcEager | AnnotationSource::Llm),
        "base kept or focused refinement adopted, never a worse rung"
    );
    // The refine contract is the one the prompt builder owns.
    assert!(LlmRefinePrompt::contract().get("properties").is_some());
    eprintln!(
        "refine consulted {} time(s), source = {:?}",
        consults.load(Ordering::SeqCst),
        result.source()
    );
}
