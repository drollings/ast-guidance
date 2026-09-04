use super::*;
#[allow(unused_imports)]
use crate::types::{RouterMessage, RouterMessageContent};

fn make_test_request(content: &str) -> RouterRequest {
    RouterRequest {
        model: "test-model".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: None,
        agent_id: None,
        adapter: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        metadata: Default::default(),
    }
}

// A stub backend for testing decorators
struct StubBackend {
    responses: std::sync::Mutex<Vec<Result<RouterResponse, DispatchError>>>,
}

impl StubBackend {
    // Returns a trait object, not Self - the retry decorator needs the
    // erased backend. Scoped-allow: clippy's new_ret_no_self false positive.
    #[allow(clippy::new_ret_no_self)]
    fn new(responses: Vec<Result<RouterResponse, DispatchError>>) -> Arc<dyn DispatchBackend> {
        Arc::new(StubBackend {
            responses: std::sync::Mutex::new(responses),
        })
    }
}

impl DispatchBackend for StubBackend {
    fn complete(
        &self,
        _request: RouterRequest,
        _model: String,
        _params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        _filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let _ = (idle_timeout_ms, total_timeout_ms);
        let mut guard = self.responses.lock().unwrap();
        let result = guard.remove(0);
        Box::pin(async move { result })
    }

    fn stream_complete_with_abort(
        &self,
        _request: RouterRequest,
        _model: String,
        _params: Option<Value>,
        _idle_timeout_ms: u64,
        _total_timeout_ms: u64,
        _filter_thinking: bool,
        _abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        let mut guard = self.responses.lock().unwrap();
        let result = guard.remove(0);
        Box::pin(async move {
            match result {
                Ok(_) => {
                    let abort = StreamAbort::new();
                    let (_, rx) = http_body_util::channel::Channel::new(32);
                    Ok(StreamHandle {
                        model: "test".into(),
                        body: StreamBody {
                            inner: rx,
                            abort: abort.clone(),
                        },
                        answer: None,
                        abort,
                    })
                }
                Err(e) => Err(e),
            }
        })
    }
}

fn dummy_response() -> RouterResponse {
    RouterResponse {
        id: "test-id".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "test-model".into(),
        choices: vec![],
        usage: crate::types::Usage::default(),
    }
}

// -----------------------------------------------------------------------
// Routing-fields body builder (instance/snapshot/id_slot)
// -----------------------------------------------------------------------

#[test]
fn routing_fields_reach_outgoing_body_only_when_set() {
    let request = make_test_request("hi");

    // None set -> no routing fields in the body.
    let params = params_with_routing_fields(None, None, None, None);
    let body = build_chat_body(&request, "m", params.as_ref(), false, false).unwrap();
    let obj = body.as_object().unwrap();
    assert!(obj.get("instance").is_none());
    assert!(obj.get("snapshot").is_none());
    assert!(obj.get("id_slot").is_none());

    // All set -> present in the outgoing body.
    let params = params_with_routing_fields(None, Some("ledger"), Some("readfiles"), Some(3));
    let body = build_chat_body(&request, "m", params.as_ref(), false, false).unwrap();
    let obj = body.as_object().unwrap();
    assert_eq!(obj["instance"], "ledger");
    assert_eq!(obj["snapshot"], "readfiles");
    assert_eq!(obj["id_slot"], 3);
}

#[test]
fn routing_fields_merge_into_existing_params() {
    let params = serde_json::json!({"temperature": 0.2});
    let merged = params_with_routing_fields(Some(params), Some("scratch"), None, Some(0));
    let obj = merged.expect("merged params");
    assert_eq!(obj["temperature"], 0.2);
    assert_eq!(obj["instance"], "scratch");
    assert_eq!(obj["id_slot"], 0);
    assert!(obj.get("snapshot").is_none());
}

// -----------------------------------------------------------------------
// RetryBackend tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn retry_success_on_first_attempt() {
    let inner = StubBackend::new(vec![Ok(dummy_response())]);
    let backend = RetryBackend::new(inner, 2, 1);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn retry_on_transient_then_succeed() {
    let inner = StubBackend::new(vec![Err(DispatchError::RateLimited), Ok(dummy_response())]);
    let backend = RetryBackend::new(inner, 2, 1);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn retry_short_circuit_on_non_retryable() {
    let inner = StubBackend::new(vec![Err(DispatchError::ResponseParse("bad json".into()))]);
    let backend = RetryBackend::new(inner, 2, 1);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        DispatchError::ResponseParse(_)
    ));
}

#[tokio::test]
async fn retry_exhaustion_returns_last_error() {
    let inner = StubBackend::new(vec![
        Err(DispatchError::RateLimited),
        Err(DispatchError::RateLimited),
    ]);
    let backend = RetryBackend::new(inner, 1, 1);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// BackendChain tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn fallback_first_backend_succeeds() {
    let b1 = StubBackend::new(vec![Ok(dummy_response())]);
    let b2 = StubBackend::new(vec![Ok(dummy_response())]);
    let backend = BackendChain::new(vec![b1, b2]);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn fallback_falls_through_on_transient_error() {
    let b1 = StubBackend::new(vec![Err(DispatchError::RateLimited)]);
    let b2 = StubBackend::new(vec![Ok(dummy_response())]);
    let backend = BackendChain::new(vec![b1, b2]);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn fallback_short_circuits_on_4xx() {
    let b1 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 400".into()))]);
    let b2 = StubBackend::new(vec![Ok(dummy_response())]);
    let backend = BackendChain::new(vec![b1, b2]);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("400"));
}

#[tokio::test]
async fn fallback_all_backends_fail() {
    let b1 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 503".into()))]);
    let b2 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 502".into()))]);
    let backend = BackendChain::new(vec![b1, b2]);
    let result = backend
        .complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), DispatchError::Http(_)));
}

#[tokio::test]
async fn retry_stream_transient_then_succeed() {
    let inner = StubBackend::new(vec![Err(DispatchError::RateLimited), Ok(dummy_response())]);
    let backend = RetryBackend::new(inner, 2, 1);
    let result = backend
        .stream_complete(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
        )
        .await;
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------
// Timeout enforcement tests
// -----------------------------------------------------------------------

/// An upstream that accepts TCP connections but never responds. A buffered
/// dispatch against it must resolve with a timeout error, not hang forever.
#[tokio::test]
async fn complete_times_out_against_never_responding_upstream() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let _server = tokio::spawn(async move {
        // Hold every accepted connection open without responding so the
        // peer's `send()` stalls until the total timeout fires.
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });

    let backend = OpenAiChatBackend::new(reqwest::Client::new(), format!("http://{addr}"));
    let total_timeout_ms = 200;
    let start = std::time::Instant::now();

    let result = tokio::time::timeout(
        Duration::from_millis(total_timeout_ms + 2000),
        backend.complete(
            make_test_request("hi"),
            "m".into(),
            None,
            total_timeout_ms,
            total_timeout_ms,
            false,
        ),
    )
    .await;

    let elapsed = start.elapsed();
    assert!(
        result.is_ok(),
        "complete() must not hang on a stalled upstream"
    );
    let err = result.unwrap().unwrap_err();
    assert!(
        matches!(&err, DispatchError::Http(msg) if msg.contains("timeout")),
        "expected a timeout DispatchError, got: {err}"
    );
    assert!(
        elapsed < Duration::from_millis(total_timeout_ms + 2000),
        "complete() returned after {elapsed:?}, expected ~{total_timeout_ms}ms"
    );
}

#[test]
fn parse_group_miss_fork_json_shape() {
    // The fork's real 503 payload is JSON with `error.message`.
    let body = r#"{"error":{"code":503,"message":"no free instance in group 'swarm'","type":"unavailable_error"}}"#;
    assert_eq!(parse_group_miss(body).as_deref(), Some("swarm"));
}

#[test]
fn parse_group_miss_raw_marker_fallback() {
    // A non-JSON body carrying the marker still resolves via the substring
    // fallback (no regression).
    let body = "upstream 503: no free instance in group 'fast'";
    assert_eq!(parse_group_miss(body).as_deref(), Some("fast"));
}

#[test]
fn parse_group_miss_generic_503_returns_none() {
    // A generic 503 without the group-miss marker -> None.
    assert_eq!(parse_group_miss(r#"{"error":{"message":"oom"}}"#), None);
    assert_eq!(parse_group_miss("Internal Server Error"), None);
}

#[tokio::test]
async fn complete_sends_bearer_token_from_api_key_env() {
    // An external endpoint's `api_key` names an env var; the backend sends
    // its value as `Authorization: Bearer`. A missing variable degrades to
    // no header (fail-open).
    use std::sync::Mutex;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let seen_for_task = seen.clone();

    let body = r#"{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
    let body_owned = body.to_string();
    let _server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        let _ = reader.read_line(&mut request_line).await;
        let mut auth = None;
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await.is_err() {
                return;
            }
            if line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_lowercase().strip_prefix("authorization:") {
                auth = Some(v.trim().to_string());
            }
            if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
        *seen_for_task.lock().unwrap() = auth;
        let mut buf = vec![0u8; content_length];
        if content_length > 0 && reader.read_exact(&mut buf).await.is_err() {
            return;
        }
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_owned}",
            body_owned.len()
        );
        let _ = stream.write_all(resp.as_bytes()).await;
        let _ = stream.flush().await;
    });

    // Preserve the previous env value so the test restores it.
    let old = std::env::var("TEST_LLAMA_CPP_API_KEY").ok();
    std::env::set_var("TEST_LLAMA_CPP_API_KEY", "s3cret-token");

    let backend = OpenAiChatBackend::new(reqwest::Client::new(), format!("http://{addr}"))
        .with_api_key(Some("TEST_LLAMA_CPP_API_KEY".to_string()));
    let result = backend
        .complete(make_test_request("hi"), "m".into(), None, 5000, 30000, false)
        .await;
    assert!(result.is_ok(), "dispatch succeeds: {result:?}");

    // Restore env.
    match old {
        Some(v) => std::env::set_var("TEST_LLAMA_CPP_API_KEY", v),
        None => std::env::remove_var("TEST_LLAMA_CPP_API_KEY"),
    }

    let auth = seen.lock().unwrap().clone();
    assert_eq!(
        auth.as_deref().map(|a| a.to_lowercase()).as_deref(),
        Some("bearer s3cret-token"),
        "external api_key env var sent as Authorization: Bearer"
    );
}

#[tokio::test]
async fn complete_omits_auth_when_api_key_env_missing() {
    // An unreadable api_key env var must not fail the request — dispatch
    // proceeds without an auth header (fail-open), matching the frontier
    // backend convention.
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let body = r#"{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
    let body_owned = body.to_string();
    let _server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        let _ = reader.read_line(&mut request_line).await;
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await.is_err() {
                return;
            }
            if line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
        let mut buf = vec![0u8; content_length];
        if content_length > 0 && reader.read_exact(&mut buf).await.is_err() {
            return;
        }
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_owned}",
            body_owned.len()
        );
        let _ = stream.write_all(resp.as_bytes()).await;
        let _ = stream.flush().await;
    });

    // No env var set: the backend must still complete the request.
    let old = std::env::var("TEST_LLAMA_CPP_MISSING_KEY").ok();
    std::env::remove_var("TEST_LLAMA_CPP_MISSING_KEY");

    let backend = OpenAiChatBackend::new(reqwest::Client::new(), format!("http://{addr}"))
        .with_api_key(Some("TEST_LLAMA_CPP_MISSING_KEY".to_string()));
    let result = backend
        .complete(make_test_request("hi"), "m".into(), None, 5000, 30000, false)
        .await;
    assert!(result.is_ok(), "dispatch succeeds without an auth header: {result:?}");

    match old {
        Some(v) => std::env::set_var("TEST_LLAMA_CPP_MISSING_KEY", v),
        None => {}
    }
}

#[tokio::test]
async fn stream_group_miss_yields_instance_group_miss() {
    // The streaming 503 branch shares `parse_group_miss`: a fork-shaped
    // JSON group-miss on the stream connection yields `InstanceGroupMiss`.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = r#"{"error":{"code":503,"message":"no free instance in group 'swarm'","type":"unavailable_error"}}"#;
    let body_len = body.len();
    let body_owned = body.to_string();
    let _server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => continue,
            };
            let body_owned = body_owned.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
                let mut reader = BufReader::new(&mut stream);
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).await.is_err() {
                    return;
                }
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).await.is_err() {
                        return;
                    }
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut buf = vec![0u8; content_length];
                if content_length > 0 && reader.read_exact(&mut buf).await.is_err() {
                    return;
                }
                let resp = format!(
                    "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n{body_owned}"
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });

    let backend = OpenAiChatBackend::new(reqwest::Client::new(), format!("http://{addr}"));
    let result = backend
        .stream_complete(make_test_request("hi"), "m".into(), None, 5000, 30000, false)
        .await;
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("streaming group-miss must error"),
    };
    assert!(
        matches!(&err, DispatchError::InstanceGroupMiss { group } if group == "swarm"),
        "expected InstanceGroupMiss(swarm), got: {err}"
    );
}

// -----------------------------------------------------------------------
// Abort propagation tests
// -----------------------------------------------------------------------

/// A stub upstream that streams one SSE chunk and then holds the connection
/// open, recording whether the router closed it (the abort must drop the
/// upstream connection instead of draining the generation to the end).
#[tokio::test]
async fn stream_abort_drops_upstream_and_finalizes_partial_answer() {
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let peer_closed = Arc::new(AtomicBool::new(false));
    let peer_closed_task = peer_closed.clone();

    let _server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        {
            let mut reader = BufReader::new(&mut stream);
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).await.is_err() {
                return;
            }
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).await.is_err() {
                    return;
                }
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; content_length];
            if content_length > 0 && reader.read_exact(&mut body).await.is_err() {
                return;
            }
        } // drop the reader so the response can be written

        // One SSE chunk, then keep the connection open (no last-chunk), so
        // the router's forwarding task parks on the next `chunk()`.
        let chunk1 = r#"data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n",
            chunk1.len(),
            chunk1,
        );
        let _ = stream.write_all(resp.as_bytes()).await;
        let _ = stream.flush().await;

        // Detect the router closing the connection on abort.
        let mut sink = [0u8; 64];
        loop {
            match stream.read(&mut sink).await {
                Ok(0) => {
                    peer_closed_task.store(true, AtomicOrdering::SeqCst);
                    break;
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });

    let backend = OpenAiChatBackend::new(reqwest::Client::new(), format!("http://{addr}"));
    let abort = StreamAbort::new();
    let handle = backend
        .stream_complete_with_abort(
            make_test_request("hi"),
            "m".into(),
            None,
            5000,
            30000,
            false,
            Some(abort),
        )
        .await
        .expect("stream connects");
    assert!(!handle.abort.is_cancelled());

    // Simulate the client abort: hyper drops the response body. The body
    // drop-guard fires the signal, and the forwarding task closes the
    // upstream connection.
    drop(handle.body);
    assert!(handle.abort.is_cancelled(), "body drop fires the abort signal");

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !peer_closed.load(AtomicOrdering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("router closed the upstream connection on abort");

    // The partial answer is finalized so the ledger records what was
    // streamed before the abort rather than a stub label. Whether the
    // first chunk was consumed before the abort is a scheduling race, so
    // accept either the full partial content or (if the abort won the
    // race) an empty-but-finalized answer — the load-bearing assertion is
    // that `finalize` ran on the abort path at all.
    let answer = handle.answer.as_ref().expect("answer present");
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while answer.get().is_none() {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("partial answer finalized on abort");
    let content = answer.get().expect("finalized on abort");
    assert!(
        content.is_empty() || content == "hello",
        "partial answer must be a prefix of the streamed content, got: {content:?}"
    );
}
