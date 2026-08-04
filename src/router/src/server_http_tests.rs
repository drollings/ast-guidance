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
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::config::RouterConfig;
use crate::routes::plan::PlanRoute;
use crate::server::responses::ResponseBody;
use crate::server::serve_http;
use crate::testing::mock::{MockDispatchContext, MockTranscriptEntry, TranscriptProvider};
use guidance_llm::client::ChatBackend;

/// Upstream responder: given the parsed request body, produce an HTTP response.
type UpstreamRespond = Arc<dyn Fn(&Value) -> hyper::Response<ResponseBody> + Send + Sync>;

/// A running router server bound to an ephemeral port.
struct TestServer {
    addr: std::net::SocketAddr,
    handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Build a `RouterConfig` with a single `default` pipeline, a single `fast`
/// model/route/group, and the given upstream + dispatch settings.
fn make_config(
    endpoint: &str,
    stream: bool,
    filter_thinking: bool,
    total_timeout_ms: u64,
    idle_timeout_ms: u64,
) -> RouterConfig {
    let value = json!({
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true}},
        "models": {"fast": {
            "endpoint": endpoint,
            "name": "fast",
            "intelligence": 1,
            "cost_input": 0.000001,
            "cost_output": 0.000006,
            "cost_cached_read": 0.0000004,
            "speed": 10,
            "total_timeout_ms": total_timeout_ms,
            "idle_timeout_ms": idle_timeout_ms,
            "stream": stream,
            "filter_thinking": filter_thinking,
            "retry_count": 0,
            "retry_base_interval_s": 1
        }},
        "model_groups": {"fast": ["fast"]},
        "routes": {"fast": {"group": "fast", "pipelines": ["default"]}},
        "default_route": "fast"
    });
    serde_json::from_value(value).expect("valid test config")
}

/// Spawn the real server (ephemeral port) with a transcript classifier and an
/// optional dispatch mock. The default transcript classifier routes to the
/// `fast` target.
async fn spawn_test_server(config: RouterConfig, mock: Option<MockDispatchContext>) -> TestServer {
    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let routes = Arc::new(config.routes.clone());
    let models = Arc::new(config.models.clone());
    let mock = mock.map(Arc::new);
    let max_payload = config.server.max_payload;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(
            listener,
            pipelines,
            routes,
            models,
            max_payload,
            None,
            mock,
            None,
            None,
            None,
        )
        .await
        {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });

    TestServer { addr, handle }
}

/// Spawn a server with a plan route (M8 interview round-trip tests). The
/// chart store is seeded with `bug_triage`; no selector backend is attached,
/// so the deterministic/HNSW binding is the sole authority on executability.
async fn spawn_plan_server() -> TestServer {
    use crate::charts::store::{chart_from_str, ChartStore};
    use crate::hnsw::HnswIndexHandle;
    use crate::routes::plan::PlanRoute;

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

    let plan_route = Arc::new(PlanRoute::new().with_chart_store(Arc::new(store)));
    let config = make_config("http://127.0.0.1:1", false, false, 5000, 2000);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(None));
    let routes = Arc::new(config.routes.clone());
    let models = Arc::new(config.models.clone());
    let max_payload = config.server.max_payload;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(
            listener,
            pipelines,
            routes,
            models,
            max_payload,
            None,
            None,
            None,
            None,
            Some(plan_route),
        )
        .await
        {
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

/// A dispatch mock with a single canned entry.
fn mock_for(user_message: &str, dispatch_response: &str) -> MockDispatchContext {
    MockDispatchContext::new(
        vec![MockTranscriptEntry {
            user_message: user_message.to_string(),
            classifier_response: String::new(),
            expected_route: None,
            dispatch_response: Some(dispatch_response.to_string()),
            rejected: false,
            reject_reason_contains: None,
        }],
        vec![],
    )
}

/// POST an OpenAI-style chat completion body, bounded by an overall timeout.
async fn post_chat(
    base_url: &str,
    body: Value,
    timeout_ms: u64,
) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| "request timed out".to_string())?
    .map_err(|e| format!("request failed: {e}"))
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

// ── Scenario 1: buffered happy path ──────────────────────────────────────

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

// ── Scenario 2: SSE stream ───────────────────────────────────────────────

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

// ── Scenario 3: malformed JSON → 400 ─────────────────────────────────────

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

// ── Scenario 4: oversized payload → 413 ──────────────────────────────────

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

// ── Scenario 5: M1.1 regression — multi-byte UTF-8 at the 120-byte boundary

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

// ── Scenario 6: M2 regression — never-responding upstream times out ──────

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
            held_for_task
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(stream);
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
        text.contains("pipeline completed successfully"),
        "expected fallback body, got: {text}"
    );
}

// ── Scenario 7a: filter_thinking — buffered strip ────────────────────────

/// Spawn an in-process mock upstream that answers every request via `respond`.
async fn spawn_mock_upstream(respond: UpstreamRespond) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");

    tokio::spawn(async move {
        loop {
            let Ok((stream, _peer)) = listener.accept().await else {
                break;
            };
            let io = TokioIo::new(stream);
            let respond = respond.clone();
            let service = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                let respond = respond.clone();
                async move {
                    let body_bytes = req
                        .collect()
                        .await
                        .map(http_body_util::Collected::to_bytes)
                        .unwrap_or_default();
                    let value = serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null);
                    Ok::<_, std::convert::Infallible>(respond(&value))
                }
            });
            tokio::spawn(async move {
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });

    format!("http://{addr}")
}

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

// ── Scenario 7b: filter_thinking — no partial tag leak across chunks ─────

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

// ── M8: plan route interview round-trip ──────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plan_route_responds_with_targeted_clarification_then_executes() {
    let server = spawn_plan_server().await;
    let request = "Please bug_triage this report";

    // Round 1: no report entity → structured clarification (never free chat).
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
    // echoed gaps and retry=true → the chart is bound and compiled.
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
        r2["workflow"]["workflows"]["bug_triage"]["stages"]
            .as_array()
            .is_some_and(|s| s.len() == 2),
        "executed workflow carries the compiled chart: {r2:?}"
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
    // predicate → still Partial → the interview terminates as fresh_draft.
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

// ── M10: dispatch post-processing — workflow extraction ───────────────────

/// Spawn the real server with a plan route (M10 extraction hook over a
/// boot-loaded chart store).
async fn spawn_server_with_plan_route(
    config: RouterConfig,
    plan_route: Arc<PlanRoute>,
) -> TestServer {
    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let routes = Arc::new(config.routes.clone());
    let models = Arc::new(config.models.clone());
    let max_payload = config.server.max_payload;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(
            listener,
            pipelines,
            routes,
            models,
            max_payload,
            None,
            None,
            None,
            None,
            Some(plan_route),
        )
        .await
        {
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

    // A shared store with the M10 extraction hook enabled (operator opt-in).
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
    // M10a LOD0 fidelity: the draft's template captures the real prompt shape
    // (the role-prefixed message) — not the synthesized "Solve the following
    // request…" wrapper.
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
