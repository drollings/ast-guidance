//! HTTP-level integration tests for the router server.
//!
//! These drive the real `serve_http` accept loop over an ephemeral-port
//! `TcpListener` with `reqwest` as the client. Hermetic: no external network,
//! no real LLM calls. The classifier is a `TranscriptProvider` and dispatch
//! goes to in-process mock upstreams (or a never-responding listener for the
//! timeout regression).
//!
//! Every assertion that could hang is wrapped in `tokio::time::timeout`, and
//! the server task is aborted on teardown, so a regression fails instead of
//! hanging the test binary.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use common_core::sync::lock;
use http_body_util::BodyExt;
use http_body_util::Full;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::tests::common::{
    install_audit_capture, make_config, mock_for, post_chat, rigor_test_deps, spawn_mock_upstream,
    spawn_test_server, spawn_test_server_with_deps, spawn_test_server_with_ledger,
    spawn_test_server_with_sessions, test_deps, test_deps_with_ledger, TestServer,
};
use crate::config::RouterConfig;
use crate::routes::plan::PlanRoute;
use crate::routes::rigor::RigorRoute;
use crate::server::handler::ServerDeps;
use crate::server::serve_http;
use crate::testing::mock::{MockDispatchContext, MockTranscriptEntry, TranscriptProvider};
use fluent_llm::client::ChatBackend;

/// Spawn a server with a plan route (interview round-trip tests). The
/// chart store is seeded with `bug_triage`; no selector backend is attached,
/// so the deterministic/HNSW binding is the sole authority on executability.
/// An execution backend feeds the two chart targets (reproduce - root_cause)
/// so an exact hit executes server-side.
async fn spawn_plan_server() -> TestServer {
    use crate::charts::store::{chart_from_str, ChartStore};
    use fluent_db::hnsw::HnswIndexHandle;
    use crate::routes::plan::PlanRoute;
    use crate::test_stubs::StubChatBackend;

    let triage = r#"{
        "name": "bug_triage",
        "description": "Triage a bug report into reproduction, root cause, and fix plan",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            {
                "name": "reproduce",
                "provides": ["repro_plan"],
                "depends": [],
                "template": "reproduce {{ request }}",
                "essential": true
            },
            {
                "name": "root_cause",
                "provides": ["root_cause"],
                "depends": [
                    { "kind": "capability", "name": "repro_plan" },
                    { "kind": "entity_match", "name": "report",
                      "description": "the bug report",
                      "predicate": {
                        "fields": [
                            { "path": "title", "ty": "string", "required": true }
                        ]
                      },
                      "required": true }
                ],
                "template": "cause {{ request }}",
                "essential": true
            }
        ]
    }"#;

    let tmp = std::env::temp_dir().join(format!("plan-test-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).ok();
    let index_path = tmp.join("workflow_library.sqlite");
    let handle = HnswIndexHandle {
        name: "workflow_library".into(),
        path: index_path.display().to_string(),
    };
    let store = ChartStore::new(Some(handle));
    store
        .upsert(chart_from_str(triage).expect("chart parses"))
        .expect("upsert");

    let plan_route = Arc::new(
        PlanRoute::new()
            .with_chart_store(Arc::new(store))
            .with_execution_backend(Arc::new(StubChatBackend::new(vec![
                r#"{"plan": "minimal repro"}"#.to_string(),
                r#"{"cause": "null pointer deref in async task"}"#.to_string(),
            ]))),
    );
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let deps = test_deps(
        pipelines,
        &config,
        None,
        None,
        Some(plan_route),
        HashMap::new(),
        None,
    );
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "plan test server failed");
        }
    });

    TestServer { addr, handle }
}

/// POST a plan request, bounded by an overall timeout.
async fn post_plan(
    base_url: &str,
    body: Value,
    timeout_ms: u64,
) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client
            .post(format!("{base_url}/v1/plan"))
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| "plan request timed out".to_string())?
    .map_err(|e| format!("plan request failed: {e}"))
}

/// Extract the concatenated `delta.content` from each `data:` SSE line.
fn sse_delta_content(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| {
            v.get("choices")?
                .as_array()?
                .first()?
                .get("delta")?
                .get("content")?
                .as_str()
                .map(ToString::to_string)
        })
        .collect()
}

// -- Scenario 1: buffered happy path --------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffered_happy_path_returns_200_with_choices() {
    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    let server = spawn_test_server(config, Some(mock_for("What is 2+2?", "4"))).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);

    let value: Value = response.json().await.expect("response must be valid JSON");
    assert_eq!(value["choices"][0]["message"]["content"], "4");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
}

// -- Session-step recording on the dispatch path ----------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_step_recorded_and_completed_on_dispatch_path() {
    use crate::dag_session::SessionRegistry;
    use crate::session::StepStatus;

    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    let sessions = Arc::new(SessionRegistry::new(None));
    let server = spawn_test_server_with_sessions(
        config,
        Some(mock_for("What is 2+2?", "4")),
        Some(Arc::clone(&sessions)),
    )
    .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "session_id": "sess-http-1"
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);

    // The request was recorded as a completed step on the session keyed by
    // `session_id`, with the model name attached (rewind restores by model).
    let session = sessions.get_or_create("sess-http-1");
    let session = session.lock().unwrap();
    assert_eq!(session.model.as_deref(), Some("fast"));
    assert_eq!(session.step_count(), 1);
    let step_id = session.step_ids().first().unwrap().clone();
    let step = session.get_step(&step_id).unwrap();
    assert_eq!(step.status, StepStatus::Completed);
    assert!(step.result.as_ref().unwrap().accepted);
}

// -- Matched target's answer recorded in ledger + session --------------

/// An upstream that answers every buffered dispatch with `content`.
async fn upstream_answering(content: &'static str) -> String {
    spawn_mock_upstream(Arc::new(move |_req: &Value| {
        let body = json!({
            "id": "cmpl-m5",
            "object": "chat.completion",
            "created": 0,
            "model": "fast",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content}
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });
        let s = serde_json::to_string(&body).expect("serialize");
        hyper::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(s)).boxed_unsync())
            .expect("build response")
    }))
    .await
}

/// Read the session's only node and return its LOD0 (full text) content.
fn session_lod0(ledger: &crate::ledger::ContentNodeLedger, session_id: &str) -> Option<String> {
    let nodes = ledger.get_session_nodes(session_id, 10).ok()?;
    nodes.first().map(|n| n.lod[0].clone())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routed_dispatch_answer_recorded_in_ledger_lod0() {
    use crate::ledger::ContentNodeLedger;

    // Real upstream dispatch (buffered): the transcript classifier routes to
    // `fast` and the mock upstream answers with a fixed content.
    let upstream = upstream_answering("the ledger answer").await;
    let config = make_config(&upstream, false, false, 5000, 2000);
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().unwrap());
    let server = spawn_test_server_with_ledger(config, None, None, Some(Arc::clone(&ledger)))
        .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "session_id": "sess-m5-routed"
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("response json");
    assert_eq!(value["choices"][0]["message"]["content"], "the ledger answer");

    let nodes = ledger
        .get_session_nodes("sess-m5-routed", 10)
        .expect("session nodes");
    assert_eq!(nodes.len(), 1, "request + result recorded as one node");
    assert_eq!(
        nodes[0].lod[0], "the ledger answer",
        "matched target's answer must be durably recorded at LOD0"
    );
    assert_eq!(nodes[0].accepted, Some(true));
    assert_eq!(nodes[0].acceptance_score, Some(1.0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ledger_with_summarizer_attached_derives_lod() {
    use crate::ledger::{ContentNodeLedger, LedgerError};
    use crate::summarization::Summarizer;

    // A ledger wired with a `Summarizer` must derive lazy LOD levels at
    // runtime - `ensure_lod` returns `Ok` (with the summary) instead of
    // `LedgerError::NoSummarizer`. The routed answer is recorded at LOD0 and
    // the lazy tier is derived from LOD0 only.
    let upstream = upstream_answering("the summarized ledger answer").await;
    let config = make_config(&upstream, false, false, 5000, 2000);

    let backend: Arc<dyn ChatBackend> =
        Arc::new(crate::test_stubs::StubChatBackend::always("lazy summary"));
    let summarizer = Summarizer::new(backend, 20);
    let ledger = Arc::new(
        ContentNodeLedger::open_in_memory()
            .unwrap()
            .with_summarizer(summarizer),
    );
    let server = spawn_test_server_with_ledger(config, None, None, Some(Arc::clone(&ledger)))
        .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "session_id": "sess-m2-summarizer"
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);

    let nodes = ledger
        .get_session_nodes("sess-m2-summarizer", 10)
        .expect("session nodes");
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].lod[0], "the summarized ledger answer",
        "answer durably recorded at LOD0"
    );

    // The Summarizer is attached: ensure_lod succeeds and derives from LOD0.
    let derived = ledger
        .ensure_lod(nodes[0].id.expect("node id"), 2)
        .expect("lazy LOD derivation succeeds (Summarizer attached)");
    assert_eq!(derived.lod[2], "lazy summary");

    // Without a Summarizer the same call would return NoSummarizer - assert
    // the plain ledger still reports it, proving the wiring is load-bearing.
    let bare = Arc::new(ContentNodeLedger::open_in_memory().unwrap());
    let id = bare
        .record_request("sess-m2-bare", "r1", "some text")
        .unwrap();
    assert!(matches!(
        bare.ensure_lod(id, 2),
        Err(LedgerError::NoSummarizer)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_dispatch_path_records_answer_in_ledger() {
    use crate::ledger::ContentNodeLedger;

    // Mock dispatch path: the canned `dispatch_response` is the answer and
    // must land in the ledger LOD0 (previously only a "mock response" marker
    // was recorded).
    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().unwrap());
    let server = spawn_test_server_with_ledger(
        config,
        Some(mock_for("What is 2+2?", "mock answer")),
        None,
        Some(Arc::clone(&ledger)),
    )
    .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "session_id": "sess-m5-mock"
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);

    assert_eq!(
        session_lod0(&ledger, "sess-m5-mock").as_deref(),
        Some("mock answer"),
        "mock dispatch path must record the real answer, not a marker"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routed_stream_dispatch_answer_recorded_on_completion() {
    use crate::ledger::ContentNodeLedger;

    // Real upstream SSE stream: the assembled content is finalized once the
    // stream ends and recorded to the ledger (best-effort content).
    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let (mut tx, rx) =
            http_body_util::channel::Channel::<Bytes, std::convert::Infallible>::new(4);
        tokio::spawn(async move {
            let events = [
                r#"data: {"choices":[{"delta":{"content":"streamed "}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"ledger answer"}}]}"#,
                "data: [DONE]",
            ];
            for event in events {
                if tx
                    .send_data(Bytes::from(format!("{event}\n\n")))
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        hyper::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(rx.boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = make_config(&upstream, true, false, 5000, 2000);
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().unwrap());
    let server = spawn_test_server_with_ledger(config, None, None, Some(Arc::clone(&ledger)))
        .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "stream to me"}],
        "stream": true,
        "session_id": "sess-m5-stream"
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    let text = response.text().await.expect("read SSE body");
    let joined: String = sse_delta_content(&text).concat();
    assert_eq!(
        joined, "streamed ledger answer",
        "stream must carry the dispatched content"
    );

    // The stream finalizer runs in a detached task - poll the ledger until
    // the assembled answer lands (bounded), then assert.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut recorded = None;
    while tokio::time::Instant::now() < deadline {
        if session_lod0(&ledger, "sess-m5-stream").as_deref() == Some("streamed ledger answer") {
            recorded = Some(true);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        recorded.is_some(),
        "streamed answer must be finalized into the ledger LOD0"
    );
}

// -- Scenario 2: SSE stream -----------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_request_returns_sse_data_lines() {
    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        true,
        false,
        5000,
        2000,
    );
    let server = spawn_test_server(
        config,
        Some(mock_for("Tell me a story", "Once upon a time")),
    )
    .await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": true
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/event-stream")),
        "streaming response must be text/event-stream"
    );

    let text = response.text().await.expect("read SSE body");
    let data_lines: Vec<&str> = text.lines().filter(|l| l.starts_with("data: ")).collect();
    assert!(!data_lines.is_empty(), "expected at least one data: line");
    assert!(
        data_lines.contains(&"data: [DONE]"),
        "stream must terminate with [DONE]"
    );
    assert!(
        text.contains("Once upon a time"),
        "stream must carry the dispatched content"
    );
}

// -- Scenario 3: malformed JSON - 400 -------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_json_returns_400() {
    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    let server = spawn_test_server(config, None).await;

    let client = reqwest::Client::new();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        client
            .post(format!("{}/v1/chat/completions", server.base_url()))
            .header("content-type", "application/json")
            .body("{not json")
            .send(),
    )
    .await
    .expect("request must not hang")
    .expect("send must succeed");
    assert_eq!(response.status(), 400);
}

// -- Scenario 4: oversized payload - 413 ----------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversized_payload_returns_413() {
    let mut config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    config.server.max_payload = 64;
    let server = spawn_test_server(config, None).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "x".repeat(100)}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 413);
}

// -- Scenario 5: multi-byte UTF-8 at the 120-byte boundary

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multibyte_utf8_message_at_120_byte_boundary_returns_200() {
    // 5 ASCII bytes + 39 CJK chars (3 bytes each) = 122 bytes; byte 120 falls
    // mid-character. The old `&s[..120]` slice in the handler panicked here.
    let msg = "x".repeat(5) + &"你".repeat(39);
    assert_eq!(msg.len(), 122);
    assert!(
        !msg.is_char_boundary(120),
        "test must put byte 120 mid-char"
    );

    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        5000,
        2000,
    );
    let server = spawn_test_server(config, Some(mock_for(&msg, "ok"))).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": msg}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must not panic or hang");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("response must be valid JSON");
    assert_eq!(value["choices"][0]["message"]["content"], "ok");
}

// -- Scenario 6: regression - never-responding upstream times out ------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn never_responding_upstream_times_out() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    let held = Arc::new(std::sync::Mutex::new(Vec::new()));
    let held_for_task = held.clone();
    let _held_connections = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            lock(&held_for_task).push(stream);
        }
    });

    let total_timeout_ms = 500;
    let config = make_config(
        &format!("http://{addr}"),
        false,
        false,
        total_timeout_ms,
        total_timeout_ms,
    );
    let server = spawn_test_server(config, None).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "stall me"}]
    });
    let start = Instant::now();
    let response = post_chat(&server.base_url(), body, total_timeout_ms + 2000)
        .await
        .expect("request must fail fast, not hang");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(total_timeout_ms + 2000),
        "buffered dispatch took {elapsed:?}; total timeout not honored"
    );
    assert_eq!(response.status(), 200, "fallback response expected");
    let text = response.text().await.expect("read fallback body");
    assert!(
        text.contains("ERROR: route ends in unsupported deterministic response."),
        "expected fallback body, got: {text}"
    );
}

/// A route referencing a pipeline that was never built (the misconfigured
/// classifier case — boot logs `built=0`) must fail the request with a
/// legible error, not the canned success fallback.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_pipeline_returns_error_not_canned_success() {
    let config = make_config(
        "http://upstream.test:8080/v1/chat/completions",
        false,
        false,
        500,
        500,
    );
    // Simulate a boot where the `default` pipeline failed to build: the
    // route still references it, but the pipelines map is empty.
    let pipelines = Arc::new(std::collections::HashMap::new());
    let deps = test_deps_with_ledger(pipelines, &config, None, None, None);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });
    let server = TestServer { addr, handle };

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must not hang");
    assert_eq!(response.status(), 200);
    let text = response.text().await.expect("read body");
    assert!(
        text.contains("ERROR"),
        "expected a legible error for the unbuilt pipeline, got: {text}"
    );
    assert!(
        !text.contains("ERROR: route ends in unsupported deterministic response."),
        "must not claim success when no pipeline is built, got: {text}"
    );
}

/// The classifier-fallback dispatch (a `respond` decision that carries no
/// response text, so the pipeline resolves neither a direct response nor a
/// routing target) must honor the client's `stream` flag. Regression for the
/// `Invalid response event-stream. content-type: application/json` client
/// error: the fallback hardcoded `is_stream = false`, so a streaming client
/// got a bare JSON completion body instead of SSE.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn classifier_fallback_dispatch_honors_stream_flag() {
    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let (mut tx, rx) =
            http_body_util::channel::Channel::<Bytes, std::convert::Infallible>::new(4);
        tokio::spawn(async move {
            let events = [
                r#"data: {"choices":[{"delta":{"content":"streamed "}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"story"}}]}"#,
                "data: [DONE]",
            ];
            for event in events {
                if tx
                    .send_data(Bytes::from(format!("{event}\n\n")))
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        hyper::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(rx.boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = make_config(&upstream, true, false, 5000, 2000);
    let classifier_entry = config.models.get("fast").expect("fast model").clone();

    // The verdict picks the `fast` child with no direct-answer payload, so
    // the pipeline dispatches the request through the route's group; the
    // handler streams the dispatched content when the client asks for SSE.
    // (Tree mode has no `respond` path — the flat `action: respond` premise
    // is retired; the streaming regression this guards is unchanged.)
    let provider = TranscriptProvider::new(HashMap::new()).with_default(
        serde_json::to_string(&serde_json::json!({
            "route": "fast",
            "coherence": 0.9,
            "safety": 1.0,
            "complexity": 5,
            "reason": "no direct answer",
        }))
        .expect("classifier output serializes"),
    );
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));

    let mock = Arc::new(mock_for("fallback me", "Once upon a time"));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let mut deps = test_deps(pipelines, &config, Some(mock), None, None, HashMap::new(), None);
    deps.classifier = Some(("fast".to_string(), classifier_entry));
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });
    let server = TestServer { addr, handle };

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "fallback me"}],
        "stream": true
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("text/event-stream")),
        "classifier-fallback dispatch must stream when the client requests it"
    );
    let text = response.text().await.expect("read SSE body");
    assert!(
        text.contains("Once upon a time"),
        "fallback must carry the dispatched content, got: {text}"
    );
}

// -- Scenario 7a: filter_thinking - buffered strip ------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filter_thinking_stripped_from_buffered_response() {
    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let body = json!({
            "id": "cmpl-think",
            "object": "chat.completion",
            "created": 0,
            "model": "fast",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<think>secret reasoning</think>the answer"
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        });
        let s = serde_json::to_string(&body).expect("serialize");
        hyper::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(s)).boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = make_config(&upstream, false, true, 5000, 2000);
    let server = spawn_test_server(config, None).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is the answer?"}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("response must be valid JSON");
    assert_eq!(
        value["choices"][0]["message"]["content"], "the answer",
        "thinking block must be stripped from the buffered response"
    );
    assert!(
        !value.to_string().contains("secret"),
        "thinking content must not leak into the response"
    );
}

// -- Scenario 7b: filter_thinking - no partial tag leak across chunks -----

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filter_thinking_never_leaks_partial_tag_in_stream() {
    // The upstream splits both the `<think>` open tag and the `</think>`
    // close tag across SSE writes; the router must hold the partial tags
    // until they complete so no fragment ever reaches the client.
    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let (mut tx, rx) =
            http_body_util::channel::Channel::<Bytes, std::convert::Infallible>::new(4);
        tokio::spawn(async move {
            let events = [
                r#"data: {"choices":[{"delta":{"content":"Hello <thi"}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"nk>secret reasoning</thi"}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"nk>the answer"}}]}"#,
                "data: [DONE]",
            ];
            for event in events {
                if tx
                    .send_data(Bytes::from(format!("{event}\n\n")))
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        hyper::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(rx.boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = make_config(&upstream, true, true, 5000, 2000);
    let server = spawn_test_server(config, None).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": true
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);

    let text = response.text().await.expect("read SSE body");
    let chunks = sse_delta_content(&text);
    assert!(!chunks.is_empty(), "expected streamed content chunks");

    for chunk in &chunks {
        assert!(
            !chunk.contains("<think") && !chunk.contains("think>") && !chunk.contains("secret"),
            "stream leaked a partial tag or thinking content: {chunk:?}"
        );
    }
    let joined: String = chunks.concat();
    assert_eq!(
        joined, "Hello the answer",
        "assembled stream content is wrong (partial tags not held correctly)"
    );
}

// -- Plan route interview round-trip ----------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_route_responds_with_targeted_clarification_then_executes() {
    let server = spawn_plan_server().await;
    let request = "Please bug_triage this report";

    // Round 1: no report entity - structured clarification (never free chat).
    let body = json!({ "message": request });
    let resp = post_plan(&server.base_url(), body, 5000)
        .await
        .expect("plan round 1");
    assert_eq!(resp.status(), 200);
    let r1: Value = resp.json().await.expect("json response");
    assert_eq!(r1["status"], "clarify");
    assert_eq!(r1["source"], "template_adapted");
    assert!(
        r1["questions"].as_array().is_some_and(|q| q
            .iter()
            .any(|x| x.as_str().is_some_and(|s| s.contains("report")))),
        "targeted question must name the gap: {r1:?}"
    );
    let gaps: Vec<String> = r1["gaps"]
        .as_array()
        .expect("gaps echoed")
        .iter()
        .filter_map(|g| g.as_str().map(ToOwned::to_owned))
        .collect();
    assert_eq!(gaps, vec!["report".to_string()]);

    // Round 2: the answer arrives as an entity (kind = gap dep name) plus the
    // echoed gaps and retry=true - the chart is bound and compiled.
    let answer = json!({
        "message": request,
        "entities": [{
            "id": "issue-42",
            "kind": "report",
            "value": {"title": "Segfault on startup"}
        }],
        "gaps": gaps,
        "retry": true
    });
    let resp = post_plan(&server.base_url(), answer, 5000)
        .await
        .expect("plan round 2");
    assert_eq!(resp.status(), 200);
    let r2: Value = resp.json().await.expect("json response");
    assert_eq!(r2["status"], "executed");
    assert_eq!(r2["source"], "template_adapted");
    assert_eq!(
        r2["gaps_filled"],
        json!(["report"]),
        "the interviewed gap is reported as filled"
    );
    assert!(
        r2["final_output"].is_object(),
        "executed response carries the final output: {r2:?}"
    );
    assert_eq!(
        r2["final_output"]["cause"], "null pointer deref in async task",
        "executed result equals the golden transcript"
    );
    assert_eq!(r2["accepted"], true, "chart accepted after execution");
    assert!(
        r2["audit"].is_array() && r2["audit"].as_array().is_some_and(|a| a.len() == 2),
        "audit trail has one entry per completed target: {r2:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_route_second_failure_terminates_as_fresh_draft() {
    let server = spawn_plan_server().await;
    let request = "Please bug_triage this report";

    let body = json!({ "message": request });
    let r1: Value = post_plan(&server.base_url(), body, 5000)
        .await
        .expect("plan round 1")
        .json()
        .await
        .expect("json response");
    let gaps: Vec<String> = r1["gaps"]
        .as_array()
        .expect("gaps echoed")
        .iter()
        .filter_map(|g| g.as_str().map(ToOwned::to_owned))
        .collect();

    // Round 2 answers with an entity that does NOT satisfy the report
    // predicate - still Partial - the interview terminates as fresh_draft.
    let answer = json!({
        "message": request,
        "entities": [{
            "id": "note-1",
            "kind": "note",
            "value": {"body": "no title field"}
        }],
        "gaps": gaps,
        "retry": true
    });
    let resp = post_plan(&server.base_url(), answer, 5000)
        .await
        .expect("plan round 2");
    assert_eq!(resp.status(), 200);
    let r2: Value = resp.json().await.expect("json response");
    assert_eq!(
        r2["status"], "fresh_draft",
        "a second failure must not yield another round of questions: {r2:?}"
    );
    assert_eq!(r2["source"], "fresh_draft");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_route_unconfigured_returns_service_unavailable() {
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let server = spawn_test_server(config, None).await;
    let body = json!({ "message": "anything" });
    let resp = post_plan(&server.base_url(), body, 5000)
        .await
        .expect("plan request");
    assert_eq!(resp.status(), 503);
}

// -- Dispatch post-processing - workflow extraction -------------------

/// Spawn the real server with a plan route (extraction hook over a
/// boot-loaded chart store).
async fn spawn_server_with_plan_route(
    config: RouterConfig,
    plan_route: Arc<PlanRoute>,
) -> TestServer {
    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let deps = test_deps(
        pipelines,
        &config,
        None,
        None,
        Some(plan_route),
        HashMap::new(),
        None,
    );
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });

    TestServer { addr, handle }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_dispatch_distills_a_draft_chart() {
    use crate::charts::extract::WorkflowExtractor;
    use crate::charts::store::ChartStore;

    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let body = json!({
            "id": "cmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "fast",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": { "role": "assistant", "content": "the answer is 42" }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        });
        let s = serde_json::to_string(&body).expect("serialize");
        hyper::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(s)).boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = make_config(&upstream, false, false, 5000, 2000);

    // A shared store with the extraction hook enabled (operator opt-in).
    // Mode `"all"` keeps the blanket extraction the e2e asserts (the default
    // `"frontier"` scope would skip this single-target primary dispatch).
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone())
        .enabled(true)
        .with_extraction_mode(crate::config::WorkflowExtractionMode::All);
    let plan_route = Arc::new(
        PlanRoute::new()
            .with_chart_store(store.clone())
            .with_workflow_extractor(Arc::new(extractor)),
    );
    let server = spawn_server_with_plan_route(config, plan_route).await;

    let body = json!({
        "model": "fast",
        "messages": [{"role": "user", "content": "What is the answer?"}]
    });
    let response = post_chat(&server.base_url(), body, 5000)
        .await
        .expect("request must complete");
    assert_eq!(response.status(), 200);
    let value: Value = response.json().await.expect("valid JSON response");
    assert_eq!(
        value["choices"][0]["message"]["content"],
        "the answer is 42"
    );

    // The successful buffered dispatch was distilled into a draft chart.
    let name = "what_is_the_answer";
    assert!(
        store.get(name).is_some(),
        "a draft chart must be auto-extracted, got store = {:?}",
        store.list()
    );
    assert!(
        store.is_draft(name),
        "the auto-extracted chart is a draft until rubric-validated"
    );
    // LOD0 fidelity: the draft's template captures the real prompt shape
    // (the role-prefixed message) - not the synthesized "Solve the following
    // request-" wrapper.
    let chart = store.get(name).expect("chart exists");
    let template = &chart.targets[0].template;
    assert!(
        template.starts_with("user: {{ request }}"),
        "template must reflect the real prompt shape, got: {template:?}"
    );
    assert!(
        !template.contains("Solve the following request"),
        "no synthesized wrapper in the LOD0 template, got: {template:?}"
    );
    // And the draft is not selectable yet (excluded from selection).
    assert!(!store.charts_sorted().iter().any(|c| c.name == name));
}

// -- Escalation ladder - integration -----------------------------------

/// A config whose `fast` group carries an escalation ladder (turnover) pointed
/// at `frontier_url`. The local `fast` model's endpoint is dead
/// (`127.0.0.1:1`) so the local chain always exhausts into the ladder.
fn escalated_config(frontier_url: &str) -> RouterConfig {
    let value = json!({
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true}},
        "models": {"fast": {
            "endpoint": "http://127.0.0.1:1",
            "name": "fast",
            "intelligence": 1,
            "cost_input": 0.000001,
            "cost_output": 0.000006,
            "cost_cached_read": 0.0000004,
            "speed": 10,
            "total_timeout_ms": 2000,
            "idle_timeout_ms": 1000,
            "stream": false,
            "retry_count": 0,
            "retry_base_interval_s": 1
        }},
        "model_groups": {"fast": {
            "models": ["fast"],
            "escalation": {
                "modes": ["turnover"],
                "frontier": {"endpoint": frontier_url, "model": "claude"}
            }
        }},
        "classification": {"root": {
            "type": "classifier",
            "description": "test router",
            "model": "fast",
            "children": [
                {"key": "fast", "description": "test route", "node": {"type": "terminal", "route": "fast", "group": "fast"}}
            ]
        }},
        "default_route": "fast"
    });
    serde_json::from_value(value).expect("valid escalated test config")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn escalation_ladder_responds_after_local_chain_fails() {
    let capture = install_audit_capture();
    // Suffix-scope instead of clearing: the buffer is process-shared, and a
    // clear races sibling audit tests (their baseline index would point past
    // the truncation). See `test_support::global_capture_len`.
    let base = crate::test_support::global_capture_len(&capture);

    let upstream = spawn_mock_upstream(Arc::new(|_req: &Value| {
        let body = json!({
            "id": "cmpl-escalated",
            "object": "chat.completion",
            "created": 0,
            "model": "claude",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "frontier rescued the request"}
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        });
        let s = serde_json::to_string(&body).expect("serialize");
        hyper::Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(s)).boxed_unsync())
            .expect("build response")
    }))
    .await;

    let config = escalated_config(&upstream);
    let http_client = reqwest::Client::new();
    let ladders = config.build_escalation_ladders(&http_client);
    assert_eq!(ladders.len(), 1, "one ladder for the fast group");

    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let deps = test_deps(pipelines, &config, None, None, None, ladders, None);
    let server = spawn_test_server_with_deps(deps).await;

    let resp = post_chat(
        &server.base_url(),
        json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "what is the answer?"}]
        }),
        8000,
    )
    .await
    .expect("chat completion");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "frontier rescued the request"
    );

    // Every escalation interaction wrote a `kind = "escalation"` audit record
    // with the mode and acceptance - captured by the global subscriber.
    let lines = crate::test_support::global_capture_since(&capture, base).join("\n");
    assert!(
        lines.contains("router.audit"),
        "audit stream must carry the record, got:\n{lines}"
    );
    assert!(
        lines.contains("\"mode\":\"turnover\"") && lines.contains("\"accepted\":true"),
        "escalation audit record must carry mode/accepted, got:\n{lines}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_cache_short_circuits_before_frontier_integration() {
    let capture = install_audit_capture();
    // Suffix-scope instead of clearing (process-shared buffer — see above).
    let base = crate::test_support::global_capture_len(&capture);

    // A context hit must be returned without any frontier contact, so the
    // upstream is not even spawned - point it at a dead address.
    let config = escalated_config("http://127.0.0.1:1");
    let http_client = reqwest::Client::new();
    let ladders = config.build_escalation_ladders(&http_client);

    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));

    struct CannedCache;
    impl fluent_types::ContextCache for CannedCache {
        fn lookup(&self, query: &str) -> Option<fluent_types::ContextHit> {
            query
                .eq_ignore_ascii_case("known fact")
                .then(|| fluent_types::ContextHit {
                    source: "test-cache".into(),
                    content: "cached fact".into(),
                    score: 0.99,
                    metadata: None,
                })
        }
    }
    let context_cache: Arc<dyn fluent_types::ContextCache> = Arc::new(CannedCache);

    let deps = test_deps(
        pipelines,
        &config,
        None,
        None,
        None,
        ladders,
        Some(context_cache),
    );
    let server = spawn_test_server_with_deps(deps).await;

    let resp = post_chat(
        &server.base_url(),
        json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "known fact"}]
        }),
        8000,
    )
    .await
    .expect("chat completion");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "cached fact");

    let lines = crate::test_support::global_capture_since(&capture, base).join("\n");
    assert!(
        lines.contains("\"mode\":\"context\"") && lines.contains("\"source\":\"test-cache\""),
        "context short-circuit must be audited with the cache source, got:\n{lines}"
    );
}

// -- /v1/rigor server round-trip ------------------------------------

/// Spawn a server with a rigor route whose three role backends are stubs.
async fn spawn_rigor_server(blue: Vec<&str>, red: Vec<&str>, judge: Vec<&str>) -> TestServer {
    use crate::test_stubs::StubChatBackend;

    let rigor_route = Arc::new(
        RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::new(
                blue.into_iter().map(ToOwned::to_owned).collect(),
            )))
            .with_red_backend(Arc::new(StubChatBackend::new(
                red.into_iter().map(ToOwned::to_owned).collect(),
            )))
            .with_judge_backend(Arc::new(StubChatBackend::new(
                judge.into_iter().map(ToOwned::to_owned).collect(),
            ))),
    );
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let sessions = Arc::new(crate::dag_session::SessionRegistry::new(None));
    let ledger = Arc::new(crate::ledger::ContentNodeLedger::open_in_memory().unwrap());

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let deps = rigor_test_deps(
        pipelines,
        &config,
        Some(rigor_route),
        Some(sessions),
        Some(ledger),
    );
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "rigor test server failed");
        }
    });
    TestServer { addr, handle }
}

/// POST a rigor request, bounded by an overall timeout.
async fn post_rigor(
    base_url: &str,
    body: Value,
    timeout_ms: u64,
) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client
            .post(format!("{base_url}/v1/rigor"))
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| "rigor request timed out".to_string())?
    .map_err(|e| format!("rigor request failed: {e}"))
}

const RED_OBJECTIONS: &str =
    r#"[{"category": "factual", "description": "unsupported claim", "severity": 0.9}]"#;
const ACCEPT_VERDICT: &str =
    r#"{"verdict": "accept", "caveats": [], "reasons": [], "confidence": 0.9}"#;
const REJECT_VERDICT: &str =
    r#"{"verdict": "reject", "caveats": [], "reasons": ["x"], "confidence": 0.8}"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rigor_route_judge_accepts_returns_executed_and_audits() {
    let capture = install_audit_capture();
    // Suffix-scope like the other audit tests: node/request ids recycle, so
    // whole-buffer presence asserts can match a sibling's stale lines.
    let base = crate::test_support::global_capture_len(&capture);
    let server = spawn_rigor_server(
        vec!["the rigorous answer"],
        vec![RED_OBJECTIONS],
        vec![ACCEPT_VERDICT],
    )
    .await;
    let resp = post_rigor(
        &server.base_url(),
        json!({"message": "prove this claim", "session_id": "sess-rigor-http"}),
        10000,
    )
    .await
    .expect("rigor request");
    assert_eq!(resp.status(), 200, "judge accepts -> executed");
    let body: Value = resp.json().await.expect("rigor response json");
    assert_eq!(body["status"], "executed");
    assert_eq!(body["answer"], "the rigorous answer");
    assert_eq!(body["verdict"], "accept");
    assert_eq!(body["rewound"], false);

    let lines = crate::test_support::global_capture_since(&capture, base).join("\n");
    assert!(
        lines.contains("router.audit") && lines.contains("kind=\"rigor\""),
        "rigor execution must emit an audit record, got:\n{lines}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rigor_route_material_rejection_returns_clarify() {
    let server = spawn_rigor_server(
        vec!["first answer", "second answer"],
        vec![RED_OBJECTIONS, RED_OBJECTIONS],
        vec![REJECT_VERDICT, REJECT_VERDICT],
    )
    .await;
    let resp = post_rigor(
        &server.base_url(),
        json!({"message": "high-stakes claim"}),
        10000,
    )
    .await
    .expect("rigor request");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("rigor response json");
    assert_eq!(
        body["status"], "clarify",
        "a final rejection resolves to clarify"
    );
    assert_eq!(body["rewound"], true, "material rejection rewound for real");
    assert!(
        body["questions"].as_array().is_some_and(|q| !q.is_empty()),
        "targeted interview questions must be populated: {body:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rigor_route_unconfigured_returns_explicit_error() {
    // No rigor route wired: /v1/rigor degrades to an explicit error, never a
    // crash (the shipped env/coral-router.json has no `rigor` section).
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let deps = rigor_test_deps(pipelines, &config, None, None, None);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "rigor test server failed");
        }
    });
    let server = TestServer { addr, handle };

    let resp = post_rigor(&server.base_url(), json!({"message": "x"}), 10000)
        .await
        .expect("rigor request");
    assert_eq!(
        resp.status(),
        hyper::StatusCode::SERVICE_UNAVAILABLE,
        "unconfigured rigor route -> explicit 503"
    );
    let text = resp.text().await.unwrap_or_default();
    assert!(
        text.contains("rigor route not configured"),
        "error body must explain, got: {text}"
    );
}

// -- Shared-weight instance management API ----------------------------------

/// A stub llama-server management backend answering the `/instances` envelope,
/// create, and delete.
fn instance_stub_handler(
) -> Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> {
    Arc::new(|method, path, _body| match (method, path) {
        ("GET", "/instances") => (
            200,
            json!({
                "instances": [{
                    "id": "ledger", "aliases": [], "group": "ledger",
                    "n_ctx": 65536, "parallel": 1, "pinned": true, "is_default": true,
                    "state": "loaded", "model_bytes": 2428416000u64, "context_bytes": 100,
                    "compute_bytes": 100, "total_bytes": 2428416200u64, "vram_bytes": 200,
                    "last_used": 1,
                }],
                "snapshots": [],
                "total": { "model": 2428416000u64, "context": 100, "compute": 100, "total": 2428416200u64 },
            })
            .to_string(),
        ),
        ("POST", "/instances") => (
            201,
            json!({
                "id": "work", "group": "swarm", "n_ctx": 32768, "parallel": 1,
                "pinned": false, "is_default": false, "state": "loaded",
                "model_bytes": 0, "context_bytes": 0, "compute_bytes": 0,
                "total_bytes": 0, "vram_bytes": 0, "last_used": -1,
            })
            .to_string(),
        ),
        ("DELETE", path) if path.starts_with("/instances/") => (200, "{}".into()),
        ("POST", path) if path.starts_with("/instances/") => (200, "{}".into()),
        _ => (404, "{}".into()),
    })
}

/// Spawn a server wired with a managed instance pool backed by `stub`.
async fn spawn_instances_server(
    stub: &crate::instances::stub::StubServer,
) -> TestServer {
    use crate::config::SidecarConfig;
    use crate::instances::{InstanceClient, InstanceManager, InstancePool};

    let mut managers = HashMap::new();
    managers.insert(
        "swarm".into(),
        Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            SidecarConfig::default(),
        )),
    );
    let pool = InstancePool::from_managers(managers, None);

    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let mut deps = test_deps(pipelines, &config, None, None, None, HashMap::new(), None);
    deps.instance_pool = Some(Arc::new(pool));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "instances test server failed");
        }
    });
    TestServer { addr, handle }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn instances_api_aggregates_lists_and_proxies() {
    let stub = crate::instances::stub::StubServer::start(instance_stub_handler());
    let server = spawn_instances_server(&stub).await;

    // GET /instances aggregates with the public id grammar.
    let resp = reqwest::get(format!("{}/instances", server.base_url()))
        .await
        .expect("GET /instances");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("instances envelope json");
    let instances = body["instances"].as_array().expect("instances array");
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0]["id"], "swarm:ledger");
    assert_eq!(instances[0]["pinned"], true);
    assert_eq!(body["total"]["model"], 2428416000u64);

    // GET /v1/models lists one entry per instance plus aliases.
    let resp = reqwest::get(format!("{}/v1/models", server.base_url()))
        .await
        .expect("GET /v1/models");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("models json");
    let data = body["data"].as_array().expect("models data");
    assert_eq!(data[0]["id"], "swarm:ledger");
    let aliases = data[0]["aliases"].as_array().expect("aliases");
    assert!(aliases.iter().any(|a| a == "swarm"));
    assert!(aliases.iter().any(|a| a == "swarm:latest"));

    // GET /memory reshapes the same envelope.
    let resp = reqwest::get(format!("{}/memory", server.base_url()))
        .await
        .expect("GET /memory");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("memory json");
    assert_eq!(body["object"], "memory");
    assert_eq!(body["total"]["total"], 2428416200u64);

    // POST /instances creates a fresh context on the owning server.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/instances", server.base_url()))
        .json(&json!({"model": "swarm", "name": "work", "group": "swarm", "ctx_size": 32768}))
        .send()
        .await
        .expect("POST /instances");
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.expect("create response json");
    assert_eq!(body["id"], "work");

    // DELETE /instances/<model>:<name> proxies to the owning server.
    let resp = client
        .delete(format!("{}/instances/swarm:ledger", server.base_url()))
        .send()
        .await
        .expect("DELETE /instances");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("delete response json");
    assert_eq!(body["success"], true);

    // The id grammar also works via ?model= + bare name.
    let resp = client
        .delete(format!("{}/instances/ledger?model=swarm", server.base_url()))
        .send()
        .await
        .expect("DELETE ?model= route");
    assert_eq!(resp.status(), 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn instances_api_rejects_unknown_model_and_requires_specify_model() {
    let stub = crate::instances::stub::StubServer::start(instance_stub_handler());
    let server = spawn_instances_server(&stub).await;
    let client = reqwest::Client::new();

    // A management call that names no model (with exactly one managed model)
    // routes to it; a body carrying an unknown model is rejected.
    let resp = client
        .post(format!("{}/instances", server.base_url()))
        .json(&json!({"name": "work", "group": "swarm", "ctx_size": 16384}))
        .send()
        .await
        .expect("POST without model");
    assert_eq!(resp.status(), 201, "single managed model is the implicit target");

    let resp = client
        .post(format!("{}/instances", server.base_url()))
        .json(&json!({"model": "nope", "name": "work", "group": "swarm"}))
        .send()
        .await
        .expect("POST unknown model");
    assert_eq!(resp.status(), 400);
}

// ---------------------------------------------------------------------------
// Capability gating is real on the serving path.
// ---------------------------------------------------------------------------

/// A classifier backend that records whether the router's knowledge capability
/// is granted in the current task-local when the classifier runs, then returns
/// a routing decision so the pipeline proceeds normally. This observes the
/// `handle_request` grant from inside the request path without touching
/// dispatch.
struct KnowledgeProbeBackend {
    gate: Arc<Mutex<Option<bool>>>,
}

impl ChatBackend for KnowledgeProbeBackend {
    fn chat_complete(
        &self,
        _messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        let granted =
            fluent_wvr::capability::check_capability(&crate::knowledge::RouterKnowledgeCapability)
                .is_ok();
        *lock(&self.gate) = Some(granted);
        let out = serde_json::to_string(&crate::config::ClassifierOutput {
            action: "route".into(),
            response: None,
            target: Some("fast".into()),
            coherence_score: 0.95,
            safety_score: 0.9,
            complexity: None,
            intent: Some("question".into()),
            reason: "probe".into(),
            completeness: None,
            risk: None,
        })
        .unwrap_or_default();
        Ok(out)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn knowledge_gate_is_open_inside_http_request_and_closed_outside() {
    use fluent_wvr::capability::check_capability;

    // Outside any scope: the gate must be closed.
    assert!(
        check_capability(&crate::knowledge::RouterKnowledgeCapability).is_err(),
        "the knowledge capability must not be granted outside a request scope"
    );

    let gate = Arc::new(Mutex::new(None::<bool>));
    let probe = KnowledgeProbeBackend { gate: Arc::clone(&gate) };
    let backend: Arc<dyn ChatBackend> = Arc::new(probe);
    let config = make_config("http://127.0.0.1:1", false, false, 2000, 1000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));

    let mock = Arc::new(MockDispatchContext::new(
        vec![MockTranscriptEntry {
            user_message: "What is 2+2?".into(),
            classifier_response: String::new(),
            expected_route: Some("fast".into()),
            expect_model_group: Some("fast".into()),
            dispatch_response: Some("four".into()),
            rejected: false,
            reject_reason_contains: None,
        }],
        vec![],
    ));

    let deps = test_deps(pipelines, &config, Some(mock), None, None, HashMap::new(), None);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let handle = tokio::spawn(async move {
        let _ = serve_http(listener, deps, None).await;
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200, "canned dispatch should succeed");
    handle.abort();

    assert_eq!(
        *lock(&gate),
        Some(true),
        "the router knowledge capability must be granted inside the HTTP handler"
    );
}

// ---------------------------------------------------------------------------
// Server-owned tasks are drained on graceful shutdown.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graceful_shutdown_drains_tracked_connections_within_timeout() {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;

    let config = make_config("http://127.0.0.1:1", false, false, 2000, 1000);
    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let deps = test_deps(pipelines, &config, None, None, None, HashMap::new(), None);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let serve = tokio::spawn(async move {
        serve_http(listener, deps, Some(shutdown_rx)).await
    });

    // A completed request is served normally before shutdown.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("health");
    assert_eq!(resp.status(), 200);

    // Leave a connection half-open so an in-flight per-connection task exists
    // when shutdown fires; graceful stop must abort+await it within a timeout.
    let mut lingering = TcpStream::connect(addr).await.expect("connect");
    let _ = lingering
        .write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n")
        .await; // no terminating CRLF: connection stays open

    let _ = shutdown_tx.send(true);
    let drained = tokio::time::timeout(Duration::from_secs(3), async {
        serve.await.expect("serve task joined").expect("serve_http ok")
    })
    .await;

    assert!(
        drained.is_ok(),
        "graceful shutdown must stop accepting and drain tracked connection tasks within timeout"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn instances_api_key_auth_401_without_200_with() {
    // The management `/instances` contract is API-key gated when
    // `api_key_env_name` names an env var. `api_key_env_name` is None at every
    // other test assembly site, so this is the sole exercise of the auth path.
    let stub = crate::instances::stub::StubServer::start(instance_stub_handler());

    use crate::config::SidecarConfig;
    use crate::instances::{InstanceClient, InstanceManager, InstancePool};
    let mut managers = HashMap::new();
    managers.insert(
        "swarm".into(),
        Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            SidecarConfig::default(),
        )),
    );
    let pool = InstancePool::from_managers(managers, None);

    // Register the key in a uniquely-named env var so parallel tests can't
    // clobber each other; removed after the test.
    let key_var = "ROUTER_TEST_API_KEY_9f7c";
    unsafe { std::env::set_var(key_var, "s3cret-key") };

    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let mut deps = test_deps(pipelines, &config, None, None, None, HashMap::new(), None);
    deps.instance_pool = Some(Arc::new(pool));
    deps.api_key_env_name = Some(key_var.into());
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        let _ = serve_http(listener, deps, None).await;
    });
    let server = TestServer { addr, handle };
    let base = server.base_url();

    let client = reqwest::Client::new();
    // No Authorization header -> 401.
    let resp = client
        .get(format!("{base}/instances"))
        .send()
        .await
        .expect("GET /instances");
    assert_eq!(resp.status(), 401);
    // Wrong key -> 401.
    let resp = client
        .get(format!("{base}/instances"))
        .header("authorization", "Bearer wrong")
        .send()
        .await
        .expect("GET /instances");
    assert_eq!(resp.status(), 401);
    // Correct key -> 200.
    let resp = client
        .get(format!("{base}/instances"))
        .header("authorization", "Bearer s3cret-key")
        .send()
        .await
        .expect("GET /instances");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("instances json");
    assert_eq!(body["instances"][0]["id"], "swarm:ledger");

    // A per-instance sub-resource is also gated.
    let resp = client
        .delete(format!("{base}/instances/swarm:ledger"))
        .send()
        .await
        .expect("DELETE /instances (no key)");
    assert_eq!(resp.status(), 401);
    let resp = client
        .delete(format!("{base}/instances/swarm:ledger"))
        .header("authorization", "Bearer s3cret-key")
        .send()
        .await
        .expect("DELETE /instances (key)");
    assert_eq!(resp.status(), 200);

// Non-management endpoints (e.g. /v1/models) are NOT gated.
    let resp = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .expect("GET /v1/models");
    assert_eq!(resp.status(), 200);
}

/// A minimal per-sentence routing signal carrying an interlingua frame (the
/// shape `record_parse_node` persists as a `nlp_parse` node).
fn review_signal() -> spacy_rs::routing::RoutingSignal {
    spacy_rs::routing::RoutingSignal {
        sentence: "show me the report".into(),
        predicate: "show".into(),
        subject: None,
        direct_object: Some("report".into()),
        indirect_object: Some("me".into()),
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["show".into(), "report".into()],
        lemmas: vec!["show".into(), "report".into()],
        pos: vec!["verb".into(), "noun".into()],
        deps: vec!["root".into(), "dobj".into()],
        heads: vec![0, -1],
        interlingua: Some(spacy_rs::InterlinguaSignal {
            predicate_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0001)),
            subject_id: None,
            direct_object_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0002)),
            indirect_object_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0003)),
            concept_ids: vec![],
            token_ids: vec![
                fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0001),
                fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0002),
            ],
            confidence: None,
        }),
    }
}

/// Build `ServerDeps` with a live review worker + a durable parse node for the
/// session (the ROADMAP §12.8 endpoint path). Returns the fetch-call counter so
/// a test can assert the review model was actually consulted (H1 regression).
fn review_deps(
    config: &RouterConfig,
    session_id: &str,
) -> (
    ServerDeps,
    Arc<crate::ledger::ContentNodeLedger>,
    fluent_types::NodeId,
    Arc<std::sync::atomic::AtomicUsize>,
) {
    use crate::ledger::correction_index::SqliteCorrectionIndex;
    use crate::ledger::nlp::record_parse_node;
    use crate::ledger::ContentNodeLedger;
    use crate::server::review::{ReviewFetch, ReviewWorker};
    use fluent_concurrency::tokio_runtime;
    use spacy_rs::concept_store_mem::InMemoryConceptStore;
    use spacy_rs::CorrectionIndex;

    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let signals = vec![review_signal()];
    let parse_id = record_parse_node(&ledger, session_id, "req-1", "show me the report", &signals)
        .expect("parse node");

    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetched = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let f = Arc::clone(&fetched);
    let fetch: ReviewFetch = Arc::new(move |_| {
        f.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(r#"{"corrections":[],"linked_entities":[],"note":null}"#.into())
    });
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn spacy_rs::ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let mut deps = test_deps(pipelines, config, None, None, None, HashMap::new(), None);
    deps.ledger = Some(Arc::clone(&ledger));
    deps.review_worker = Some(worker);
    deps.review_fetch = Some(fetch);
    (deps, ledger, parse_id, fetched)
}

/// ROADMAP §12.8: `POST /v1/sessions/{id}/review-parse` enqueues a job (202 +
/// job id) and `GET` reads the review status back — the endpoint seam end to
/// end (no ledger, no worker → explicit 501/404, never a crash).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn review_parse_endpoints_enqueue_and_read_back_status() {
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let (deps, ledger, parse_id, fetched) = review_deps(&config, "sess-rv");
    let server = spawn_test_server_with_deps(deps).await;
    let base = server.base_url();
    let client = reqwest::Client::new();

    // POST enqueues a job: 202 + review_queued (the hot path returns
    // immediately; the worker processes asynchronously).
    let resp = client
        .post(format!("{base}/v1/sessions/sess-rv/review-parse"))
        .send()
        .await
        .expect("POST review-parse");
    assert_eq!(resp.status(), 202, "enqueue returns 202");
    let body: Value = resp.json().await.expect("enqueue json");
    assert_eq!(body["status"], "review_queued");
    assert_eq!(body["node_id"], parse_id.as_int());

    // GET reads the current review status back (Unreviewed while the worker
    // is in flight, Reviewed once it lands — both are valid read-backs).
    let resp = client
        .get(format!("{base}/v1/sessions/sess-rv/review-parse"))
        .send()
        .await
        .expect("GET review-parse");
    assert_eq!(resp.status(), 200, "status read-back");
    let body: Value = resp.json().await.expect("status json");
    assert_eq!(body["session_id"], "sess-rv");
    assert_eq!(body["node_id"], parse_id.as_int());
    assert!(
        body["review_status"].is_object(),
        "review_status must be present in the read-back"
    );
    assert!(
        body["candidates"].is_array(),
        "the overlay/candidate plane surfaces in the read-back (M6.5)"
    );

    // The candidate plane lists durable overlay candidates for the node (M6.5).
    // Seed one entity-link candidate and confirm the GET surfaces it.
    use crate::ledger::overlay::OverlayCandidateStore;
    let candidate_store = OverlayCandidateStore::new(ledger.node_store().shared_sqlite().expect("shared"));
    candidate_store
        .write_candidate(&crate::ledger::overlay::OverlayCandidate::entity_link(
            parse_id,
            0,
            7,
            fluent_types::InterlinguaId::from_u64(0x0200_0000_0000_0001),
            0.9,
            "entity_link",
        ))
        .expect("write candidate");
    let resp = client
        .get(format!("{base}/v1/sessions/sess-rv/review-parse"))
        .send()
        .await
        .expect("GET review-parse");
    let body: Value = resp.json().await.expect("json");
    let candidates = body["candidates"].as_array().expect("candidates array");
    assert_eq!(candidates.len(), 1, "the seeded entity-link candidate surfaces");
    assert_eq!(candidates[0]["kind"], "entity_link");
    assert_eq!(candidates[0]["source"], "entity_link");
    assert_eq!(candidates[0]["status"], "pending");

    // H1 regression: a POST with realistic (non-empty) token_ids must drive a
    // real review-model call — the endpoint is never a silent no-op. Poll
    // because the worker processes asynchronously.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while fetched.load(std::sync::atomic::Ordering::SeqCst) == 0
        && std::time::Instant::now() < deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        fetched.load(std::sync::atomic::Ordering::SeqCst) >= 1,
        "the review model fetch must be invoked after a POST with token_ids"
    );
}

/// ROADMAP §12.8: without a review worker the endpoint is 501; without a parse
/// node for the session it is 404 — never a crash.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn review_parse_endpoint_degrades_gracefully() {
    use crate::ledger::ContentNodeLedger;

    // No review worker configured.
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let mut deps = test_deps(pipelines, &config, None, None, None, HashMap::new(), None);
    deps.ledger = Some(Arc::new(ContentNodeLedger::open_in_memory().expect("ledger")));
    let server = spawn_test_server_with_deps(deps).await;
    let base = server.base_url();
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/v1/sessions/sess-x/review-parse"))
        .send()
        .await
        .expect("POST review-parse");
    assert_eq!(resp.status(), 501, "no review worker → 501 Not Implemented");

    // A worker but no parse node for the session → 404.
    let (deps, _ledger, _, _) = review_deps(&config, "sess-with-parse");
    let server = spawn_test_server_with_deps(deps).await;
    let base = server.base_url();
    let resp = client
        .post(format!("{base}/v1/sessions/sess-absent/review-parse"))
        .send()
        .await
        .expect("POST review-parse");
    assert_eq!(resp.status(), 404, "no parse node for session → 404");
}

