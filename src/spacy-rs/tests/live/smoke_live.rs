//! Opt-in live-AI smoke test for the annotation fetch seam (§10.1–10.3).
//!
//! This test performs a REAL model call. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make spacy-test-live`).
//!
//! Env contract (see `tests/live/README.md`):
//! - `SPACY_LIVE_LLM_URL` — an OpenAI-compatible chat-completions endpoint
//!   finetuned to emit the §10.1 annotation JSON array for a given token list.
//!
//! When the variable is absent the test skips cleanly (early `return`, never
//! panic) per the roadmap's skip-not-fail policy. Assertions are structural
//! only (the annotated doc passes the 7-check gate and is navigable) — never
//! annotation quality.

use fluent_llm::protocol::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};
use spacy_rs::pipeline::{AnnotateError, LlmFetch, NlpPipeline};
use std::sync::Arc;

/// The live-endpoint URL; unset (or empty) skips the test.
fn live_url() -> Option<String> {
    std::env::var("SPACY_LIVE_LLM_URL")
        .ok()
        .filter(|u| !u.is_empty())
}

#[tokio::test]
#[ignore = "live-AI: requires SPACY_LIVE_LLM_URL; run via `make test-live`"]
async fn process_async_annotates_through_a_live_endpoint() {
    let Some(url) = live_url() else {
        eprintln!("skipping: SPACY_LIVE_LLM_URL unset");
        return;
    };

    let rt = fluent_concurrency::tokio_runtime();
    let pipeline = NlpPipeline::en_default().expect("pipeline");

    let client = reqwest::Client::new();
    let endpoint = url.clone();
    let queue = Arc::new(LlmRequestQueue::new(
        Arc::clone(&rt),
        &LlmQueueConfig {
            worker_count: 1,
            queue_capacity: 4,
        },
        move |task: LlmTask| {
            let client = client.clone();
            let endpoint = endpoint.clone();
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
                // The endpoint may wrap the array in a chat-completion
                // envelope; strip it if present.
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

    let fetch: LlmFetch = Arc::new(move |tokens: Vec<String>| {
        let queue = Arc::clone(&queue);
        Box::pin(async move {
            let system = spacy_rs::AnnotationRecord::prompt(&tokens);
            let task = LlmTask {
                messages: vec![
                    ChatMessage {
                        role: "system".into(),
                        content: system,
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: "Annotate the given tokens.".into(),
                    },
                ],
                config: LlmConfig::new()
                    .api_url("http://127.0.0.1:1".into())
                    .model("spacy-annotator".to_string())
                    .timeout_ms(60_000)
                    .build(),
            };
            let json = queue
                .submit(task)
                .await
                .map_err(|e| AnnotateError::Fetch(e.to_string()))?;
            Ok(json)
        })
    });

    let text = "The cat sat on the mat .";
    let doc = pipeline
        .process_async(text, Some(fetch), rt.clone(), Default::default())
        .await
        .expect("annotated doc");
    assert_eq!(doc.len(), 7);
    // The gate passed and the tree was rebuilt, so the doc is navigable.
    let _ = doc.head_index(0);
    let _ = doc.lefts(3);
    eprintln!("annotated: {text}");
}
