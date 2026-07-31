use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::dispatch::frontier::DispatchBackend;
use bytes::Bytes;
use common_core::drain_sse_lines;
use common_core::hash::uuid_v4;
use guidance_llm::HttpClass;
use serde_json::Value;

use crate::dispatch::frontier::{DispatchError, OpenAiBackend};
use crate::normalize;
use crate::streaming::StreamingHandler;
use crate::types::{RouterRequest, RouterResponse};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Streaming result — a channel that receives SSE-formatted response data
/// in a spawned background task.
pub struct StreamResult {
    pub model: String,
    pub body: http_body_util::channel::Channel<Bytes, std::convert::Infallible>,
}

// ---------------------------------------------------------------------------
// Trait — abstraction over LLM chat completion providers
// ---------------------------------------------------------------------------

/// Object-safe for `Arc<dyn ChatBackend>`. Provider config established at
/// construction time; per-request fields passed as arguments.
pub trait ChatBackend: Send + Sync {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>>;

    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_chat_body(
    request: &RouterRequest,
    model: &str,
    params: Option<&Value>,
    stream: bool,
) -> Result<Value, DispatchError> {
    let messages = normalize::messages_to_json(request)
        .map_err(|e| DispatchError::RequestBuild(e.to_string()))?;
    let mut body = serde_json::json!({"model": model, "messages": messages});
    if stream {
        body["stream"] = Value::Bool(true);
    }
    if let Some(p) = params {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                if k != "stream" {
                    body[k] = v.clone();
                }
            }
        }
    }
    if !body
        .as_object()
        .is_some_and(|o| o.contains_key("temperature"))
    {
        if let Some(temp) = request.temperature {
            if let Some(n) = serde_json::Number::from_f64(temp) {
                body["temperature"] = Value::Number(n);
            }
        }
    }
    if !body
        .as_object()
        .is_some_and(|o| o.contains_key("max_tokens"))
    {
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = Value::Number(serde_json::Number::from(max_tokens));
        }
    }
    Ok(body)
}

fn dispatch_url(endpoint_url: &str) -> String {
    if endpoint_url.ends_with("/chat/completions") {
        endpoint_url.to_string()
    } else {
        format!("{}/chat/completions", endpoint_url.trim_end_matches('/'))
    }
}

/// Wrap a fallible HTTP future in a timeout. Collapses the repeated
/// `tokio::time::timeout(...)` + `map_err` shapes used by both the buffered
/// and streaming dispatch paths; `label` distinguishes a total-budget expiry
/// from an idle-stall expiry in the resulting `DispatchError::Http`.
async fn with_total_timeout<T>(
    ms: u64,
    label: &str,
    fut: impl Future<Output = Result<T, DispatchError>>,
) -> Result<T, DispatchError> {
    tokio::time::timeout(Duration::from_millis(ms), fut)
        .await
        .map_err(|_| DispatchError::Http(label.to_string()))?
}

// ---------------------------------------------------------------------------
// OpenAiChatBackend — single-attempt OpenAI-compatible HTTP backend
// ---------------------------------------------------------------------------

pub struct OpenAiChatBackend {
    client: reqwest::Client,
    endpoint_url: String,
}

impl OpenAiChatBackend {
    pub fn new(client: reqwest::Client, endpoint_url: impl Into<String>) -> Self {
        Self {
            client,
            endpoint_url: endpoint_url.into(),
        }
    }
}

impl ChatBackend for OpenAiChatBackend {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let url = dispatch_url(&self.endpoint_url);
        let client = self.client.clone();

        Box::pin(async move {
            let body = build_chat_body(&request, &model, params.as_ref(), false)?;
            let response = with_total_timeout(
                total_timeout_ms,
                "total timeout exceeded",
                async {
                    client
                        .post(&url)
                        .json(&body)
                        .send()
                        .await
                        .map_err(|e| DispatchError::Http(e.to_string()))
                },
            )
            .await?;

            let status = response.status();
            if !status.is_success() {
                let class = HttpClass::from_status(status.as_u16());
                return if class.is_retryable() {
                    Err(DispatchError::RateLimited)
                } else {
                    Err(DispatchError::Http(format!("HTTP {status}")))
                };
            }

            let json: Value = with_total_timeout(
                idle_timeout_ms,
                "idle timeout exceeded reading response body",
                async {
                    response
                        .json()
                        .await
                        .map_err(|e| DispatchError::ResponseParse(e.to_string()))
                },
            )
            .await?;

            let openai = OpenAiBackend::new(&url, None);
            let mut resp = openai.parse_response(&json)?;
            if resp.model == "unknown" && model != "unknown" {
                resp.model = model;
            }
            Ok(resp)
        })
    }

    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>> {
        let url = dispatch_url(&self.endpoint_url);
        let client = self.client.clone();
        let model_for_task = model.clone();
        let request_id = uuid_v4();

        Box::pin(async move {
            let body = build_chat_body(&request, &model, params.as_ref(), true)?;
            let mut response = with_total_timeout(
                total_timeout_ms,
                "total timeout exceeded for stream connection",
                async {
                    client
                        .post(&url)
                        .json(&body)
                        .send()
                        .await
                        .map_err(|e| DispatchError::Http(e.to_string()))
                },
            )
            .await?;

            let status = response.status();
            if !status.is_success() {
                let class = HttpClass::from_status(status.as_u16());
                return if class.is_retryable() {
                    Err(DispatchError::RateLimited)
                } else {
                    Err(DispatchError::Http(format!("HTTP {status}")))
                };
            }

            let (mut tx, rx) = http_body_util::channel::Channel::new(32);

            tokio::spawn(async move {
                let mut handler = StreamingHandler::new(&request_id, &model_for_task)
                    .with_filter_thinking(filter_thinking);
                let mut buf = Vec::new();
                let mut sent_first_chunk = false;

                loop {
                    let chunk = match with_total_timeout(
                        idle_timeout_ms,
                        "idle timeout waiting for stream chunk",
                        async {
                            response
                                .chunk()
                                .await
                                .map_err(|e| DispatchError::Http(e.to_string()))
                        },
                    )
                    .await
                    {
                        Ok(Some(b)) => b,
                        Ok(None) => break,
                        Err(e) => {
                            tracing::warn!(
                                target: "router.dispatch",
                                model = %model_for_task,
                                error = %e,
                                "stream read error or idle timeout"
                            );
                            if sent_first_chunk {
                                let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                            }
                            return;
                        }
                    };

                    for line in drain_sse_lines(&mut buf, &chunk) {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        if trimmed == "data: [DONE]" {
                            let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                            return;
                        }
                        if let Some(data) = trimmed.strip_prefix("data: ") {
                            if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                                let choices = chunk_json
                                    .get("choices")
                                    .and_then(|v| v.as_array())
                                    .cloned()
                                    .unwrap_or_default();
                                if let Some(choice) = choices.first() {
                                    let delta = choice
                                        .get("delta")
                                        .and_then(|d| d.get("content"))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let finish_reason =
                                        choice.get("finish_reason").and_then(|v| v.as_str());

                                    if let Some(fr) = finish_reason {
                                        let s = handler.format_chunk(delta, Some(fr));
                                        if !s.is_empty() {
                                            let _ = tx.send_data(Bytes::from(s)).await;
                                        }
                                        let _ =
                                            tx.send_data(Bytes::from(handler.format_done())).await;
                                        return;
                                    }
                                    let s = handler.format_chunk(delta, None);
                                    if !s.is_empty() {
                                        sent_first_chunk = true;
                                        let _ = tx.send_data(Bytes::from(s)).await;
                                    }
                                }
                            }
                        }
                    }
                }
                if sent_first_chunk {
                    let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                }
            });

            Ok(StreamResult { model, body: rx })
        })
    }
}

// ---------------------------------------------------------------------------
// RetryChatBackend — wraps a ChatBackend with exponential-backoff retry
// ---------------------------------------------------------------------------

pub struct RetryChatBackend {
    inner: Arc<dyn ChatBackend>,
    retry_count: u32,
    retry_base_interval_s: u64,
}

impl RetryChatBackend {
    pub fn new(inner: Arc<dyn ChatBackend>, retry_count: u32, retry_base_interval_s: u64) -> Self {
        Self {
            inner,
            retry_count,
            retry_base_interval_s,
        }
    }
}

impl ChatBackend for RetryChatBackend {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let inner = self.inner.clone();
        let max_attempts = (self.retry_count + 1).max(1);
        let retry_base = self.retry_base_interval_s;

        Box::pin(async move {
            let mut last_err = DispatchError::Http("no attempt made".into());
            for attempt in 0..max_attempts {
                match inner
                    .complete(
                        request.clone(),
                        model.clone(),
                        params.clone(),
                        idle_timeout_ms,
                        total_timeout_ms,
                    )
                    .await
                {
                    Ok(resp) => return Ok(resp),
                    Err(e) if e.is_retryable() && attempt + 1 < max_attempts => {
                        last_err = e;
                        tokio::time::sleep(Duration::from_millis(
                            retry_base * 1000 * (1u64 << attempt),
                        ))
                        .await;
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(last_err)
        })
    }

    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>> {
        let inner = self.inner.clone();
        let max_attempts = (self.retry_count + 1).max(1);
        let retry_base = self.retry_base_interval_s;

        Box::pin(async move {
            let mut last_err = DispatchError::Http("no attempt made".into());
            for attempt in 0..max_attempts {
                let result = tokio::time::timeout(
                    Duration::from_millis(total_timeout_ms),
                    inner.stream_complete(
                        request.clone(),
                        model.clone(),
                        params.clone(),
                        idle_timeout_ms,
                        total_timeout_ms,
                        filter_thinking,
                    ),
                )
                .await;

                match result {
                    Ok(Ok(stream)) => return Ok(stream),
                    Ok(Err(e)) if e.is_retryable() && attempt + 1 < max_attempts => {
                        last_err = e;
                        tokio::time::sleep(Duration::from_millis(
                            retry_base * 1000 * (1u64 << attempt),
                        ))
                        .await;
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) if attempt + 1 < max_attempts => {
                        last_err = DispatchError::Http("total timeout".into());
                        tokio::time::sleep(Duration::from_millis(
                            retry_base * 1000 * (1u64 << attempt),
                        ))
                        .await;
                    }
                    Err(_) => return Err(DispatchError::Http("total timeout exhausted".into())),
                }
            }
            Err(last_err)
        })
    }
}

// ---------------------------------------------------------------------------
// FallbackChatBackend — tries backends in order
// ---------------------------------------------------------------------------

pub struct FallbackChatBackend {
    backends: Vec<Arc<dyn ChatBackend>>,
}

impl FallbackChatBackend {
    pub fn new(backends: Vec<Arc<dyn ChatBackend>>) -> Self {
        Self { backends }
    }
}

impl ChatBackend for FallbackChatBackend {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let backends = self.backends.clone();

        Box::pin(async move {
            let mut last_err = DispatchError::AllBackendsFailed;
            for backend in &backends {
                match backend
                    .complete(
                        request.clone(),
                        model.clone(),
                        params.clone(),
                        idle_timeout_ms,
                        total_timeout_ms,
                    )
                    .await
                {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        if let DispatchError::Http(ref msg) = e {
                            if msg.starts_with("HTTP 4")
                                && !msg.starts_with("HTTP 408")
                                && !msg.starts_with("HTTP 429")
                            {
                                return Err(e);
                            }
                        }
                        last_err = e;
                    }
                }
            }
            Err(last_err)
        })
    }

    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>> {
        let backends = self.backends.clone();

        Box::pin(async move {
            let mut last_err = DispatchError::AllBackendsFailed;
            for backend in &backends {
                match backend
                    .stream_complete(
                        request.clone(),
                        model.clone(),
                        params.clone(),
                        idle_timeout_ms,
                        total_timeout_ms,
                        filter_thinking,
                    )
                    .await
                {
                    Ok(stream) => return Ok(stream),
                    Err(e) => {
                        if let DispatchError::Http(ref msg) = e {
                            if msg.starts_with("HTTP 4")
                                && !msg.starts_with("HTTP 408")
                                && !msg.starts_with("HTTP 429")
                            {
                                return Err(e);
                            }
                        }
                        last_err = e;
                    }
                }
            }
            Err(last_err)
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
            metadata: Default::default(),
        }
    }

    // A stub backend for testing decorators
    struct StubBackend {
        responses: std::sync::Mutex<Vec<Result<RouterResponse, DispatchError>>>,
    }

    impl StubBackend {
        fn new(responses: Vec<Result<RouterResponse, DispatchError>>) -> Arc<dyn ChatBackend> {
            Arc::new(StubBackend {
                responses: std::sync::Mutex::new(responses),
            })
        }
    }

    impl ChatBackend for StubBackend {
        fn complete(
            &self,
            _request: RouterRequest,
            _model: String,
            _params: Option<Value>,
            idle_timeout_ms: u64,
            total_timeout_ms: u64,
        ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
            let _ = (idle_timeout_ms, total_timeout_ms);
            let mut guard = self.responses.lock().unwrap();
            let result = guard.remove(0);
            Box::pin(async move { result })
        }

        fn stream_complete(
            &self,
            _request: RouterRequest,
            _model: String,
            _params: Option<Value>,
            _idle_timeout_ms: u64,
            _total_timeout_ms: u64,
            _filter_thinking: bool,
        ) -> Pin<Box<dyn Future<Output = Result<StreamResult, DispatchError>> + Send>> {
            let mut guard = self.responses.lock().unwrap();
            let result = guard.remove(0);
            Box::pin(async move {
                match result {
                    Ok(_) => {
                        let (_, rx) = http_body_util::channel::Channel::new(32);
                        Ok(StreamResult {
                            model: "test".into(),
                            body: rx,
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
    // RetryChatBackend tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn retry_success_on_first_attempt() {
        let inner = StubBackend::new(vec![Ok(dummy_response())]);
        let backend = RetryChatBackend::new(inner, 2, 1);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn retry_on_transient_then_succeed() {
        let inner = StubBackend::new(vec![Err(DispatchError::RateLimited), Ok(dummy_response())]);
        let backend = RetryChatBackend::new(inner, 2, 1);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn retry_short_circuit_on_non_retryable() {
        let inner = StubBackend::new(vec![Err(DispatchError::ResponseParse("bad json".into()))]);
        let backend = RetryChatBackend::new(inner, 2, 1);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
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
        let backend = RetryChatBackend::new(inner, 1, 1);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // FallbackChatBackend tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn fallback_first_backend_succeeds() {
        let b1 = StubBackend::new(vec![Ok(dummy_response())]);
        let b2 = StubBackend::new(vec![Ok(dummy_response())]);
        let backend = FallbackChatBackend::new(vec![b1, b2]);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn fallback_falls_through_on_transient_error() {
        let b1 = StubBackend::new(vec![Err(DispatchError::RateLimited)]);
        let b2 = StubBackend::new(vec![Ok(dummy_response())]);
        let backend = FallbackChatBackend::new(vec![b1, b2]);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn fallback_short_circuits_on_4xx() {
        let b1 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 400".into()))]);
        let b2 = StubBackend::new(vec![Ok(dummy_response())]);
        let backend = FallbackChatBackend::new(vec![b1, b2]);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("400"));
    }

    #[tokio::test]
    async fn fallback_all_backends_fail() {
        let b1 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 503".into()))]);
        let b2 = StubBackend::new(vec![Err(DispatchError::Http("HTTP 502".into()))]);
        let backend = FallbackChatBackend::new(vec![b1, b2]);
        let result = backend
            .complete(make_test_request("hi"), "m".into(), None, 5000, 30000)
            .await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DispatchError::Http(_)));
    }

    #[tokio::test]
    async fn retry_stream_transient_then_succeed() {
        let inner = StubBackend::new(vec![Err(DispatchError::RateLimited), Ok(dummy_response())]);
        let backend = RetryChatBackend::new(inner, 2, 1);
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
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => held.push(stream),
                    Err(_) => break,
                }
            }
        });

        let backend = OpenAiChatBackend::new(
            reqwest::Client::new(),
            format!("http://{addr}"),
        );
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
            ),
        )
        .await;

        let elapsed = start.elapsed();
        assert!(result.is_ok(), "complete() must not hang on a stalled upstream");
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
}
