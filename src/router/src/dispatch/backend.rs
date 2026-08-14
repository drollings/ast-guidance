use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::metrics::FailureClass;
use bytes::Bytes;
use common_core::drain_sse_lines;
use common_core::hash::uuid_v4;
use fluent_concurrency::ladder::first_accept_in_order;
use fluent_llm::HttpClass;
use serde_json::Value;

use crate::dispatch::frontier::{DispatchError, OpenAiBackend};
use crate::normalize;
use crate::streaming::StreamingHandler;
use crate::types::{RouterRequest, RouterResponse};
// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Streaming result - a channel that receives SSE-formatted response data
/// in a spawned background task.
pub struct StreamResult {
    pub model: String,
    pub body: http_body_util::channel::Channel<Bytes, std::convert::Infallible>,
    /// Best-effort finalization sink for the assembled answer text.
    /// The streaming task writes `filtered_content()` here when the stream
    /// ends; `None` for backends/stubs that don't accumulate content.
    pub answer: Option<crate::streaming::StreamAnswer>,
}

// ---------------------------------------------------------------------------
// Trait - abstraction over LLM chat completion providers
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
        filter_thinking: bool,
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
    filter_thinking: bool,
) -> Result<Value, DispatchError> {
    let messages = normalize::messages_to_json(request)
        .map_err(|e| DispatchError::RequestBuild(e.to_string()))?;

    // Canonical body builder (fluent_llm::openai) supplies the
    // AGENTS.md-mandated `chat_template_kwargs: {"enable_thinking": false}`
    // default; `params` may override it. When `filter_thinking` is set we pin
    // the request-side default (belt-and-suspenders with the response-side
    // strip), so a contradictory `params` override cannot re-enable thinking.
    let mut body = fluent_llm::openai::build_openai_chat_body(
        model,
        &Value::Array(messages),
        params,
        stream,
        filter_thinking.then_some(false),
    );

    // Router-specific request defaults: temperature/max_tokens fall back to
    // the request fields when the model params don't set them.
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
    fluent_llm::url::chat_completions_url(endpoint_url)
}

/// Extract the group name from the fork's 503 group-miss payload
/// (`unavailable_error` "no free instance in group '<group>'"). `None` when the
/// body is not a group-miss (a generic 503).
///
/// The fork's real payload is JSON
/// (`{"error":{"message":"no free instance in group 'X'"}}`), so parse that
/// shape first; fall back to the raw substring scan so a non-JSON body (or an
/// older wire format) never regresses.
fn parse_group_miss(body: &str) -> Option<String> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(msg) = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            if let Some(group) = group_from_message(msg) {
                return Some(group);
            }
        }
    }
    group_from_message(body)
}

/// Extract the group name from a message carrying the group-miss marker. `None`
/// when the marker is absent (a generic 503, not a group miss).
fn group_from_message(msg: &str) -> Option<String> {
    let marker = "no free instance in group '";
    let idx = msg.find(marker)?;
    let rest = &msg[idx + marker.len()..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

/// Merge the routing request fields (`instance`/`snapshot`/`id_slot`) into the
/// params object that becomes the outgoing OpenAI body. These are top-level
/// request fields the fork reads (the body wins over the query string); they
/// are only added when set, so a bare dispatch leaves the body unchanged.
pub(crate) fn params_with_routing_fields(
    params: Option<Value>,
    instance: Option<&str>,
    snapshot: Option<&str>,
    id_slot: Option<i32>,
) -> Option<Value> {
    if instance.is_none() && snapshot.is_none() && id_slot.is_none() {
        return params;
    }
    let mut obj = params.unwrap_or_else(|| Value::Object(Default::default()));
    if let Value::Object(map) = &mut obj {
        if let Some(v) = instance {
            map.insert("instance".into(), Value::String(v.into()));
        }
        if let Some(v) = snapshot {
            map.insert("snapshot".into(), Value::String(v.into()));
        }
        if let Some(v) = id_slot {
            map.insert("id_slot".into(), Value::Number(v.into()));
        }
    }
    Some(obj)
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
// OpenAiChatBackend - single-attempt OpenAI-compatible HTTP backend
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
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let url = dispatch_url(&self.endpoint_url);
        let client = self.client.clone();

        Box::pin(async move {
            let start = Instant::now();
            let log_model = model.clone();
            let result: Result<RouterResponse, DispatchError> = async {
                let body =
                    build_chat_body(&request, &model, params.as_ref(), false, filter_thinking)?;
                let response =
                    with_total_timeout(total_timeout_ms, "total timeout exceeded", async {
                        client
                            .post(&url)
                            .json(&body)
                            .send()
                            .await
                            .map_err(|e| DispatchError::Http(e.to_string()))
                    })
                    .await?;

                let status = response.status();
                if !status.is_success() {
                    let status_u16 = status.as_u16();
                    // A 503 with the fork's group-miss payload means the pool
                    // had no free member; surface it as a dedicated error so
                    // the sidecar can allocate fresh KV and retry once.
                    if status_u16 == 503 {
                        let body = response
                            .text()
                            .await
                            .unwrap_or_else(|_| String::new());
                        if let Some(group) = parse_group_miss(&body) {
                            return Err(DispatchError::InstanceGroupMiss { group });
                        }
                        return Err(DispatchError::RateLimited);
                    }
                    let class = HttpClass::from_status(status_u16);
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
            }
            .await;

            let latency_ms = start.elapsed().as_millis() as u64;
            match &result {
                Ok(resp) => {
                    tracing::info!(
                        target: "router.dispatch.backend",
                        model = %log_model,
                        url = %url,
                        latency_ms = latency_ms,
                        total_timeout_ms = total_timeout_ms,
                        idle_timeout_ms = idle_timeout_ms,
                        choices = resp.choices.len(),
                        "chat completion ok"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "router.dispatch.backend",
                        model = %log_model,
                        url = %url,
                        latency_ms = latency_ms,
                        total_timeout_ms = total_timeout_ms,
                        idle_timeout_ms = idle_timeout_ms,
                        error = %e,
                        error_class = FailureClass::from(e).label(),
                        "chat completion failed"
                    );
                }
            }
            result
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
            let stream_start = Instant::now();
            let body = build_chat_body(&request, &model, params.as_ref(), true, filter_thinking)?;
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
                let status_u16 = status.as_u16();
                if status_u16 == 503 {
                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| String::new());
                    if let Some(group) = parse_group_miss(&body) {
                        return Err(DispatchError::InstanceGroupMiss { group });
                    }
                    return Err(DispatchError::RateLimited);
                }
                let class = HttpClass::from_status(status_u16);
                return if class.is_retryable() {
                    Err(DispatchError::RateLimited)
                } else {
                    Err(DispatchError::Http(format!("HTTP {status}")))
                };
            }

            tracing::info!(
                target: "router.dispatch.backend",
                model = %model,
                url = %url,
                connect_latency_ms = stream_start.elapsed().as_millis() as u64,
                total_timeout_ms = total_timeout_ms,
                idle_timeout_ms = idle_timeout_ms,
                "stream connected"
            );

            let (mut tx, rx) = http_body_util::channel::Channel::new(32);

            // Assemble the streamed answer and finalize it when the stream
            // ends so the handler can record it into the ledger + session step.
            let answer = crate::streaming::StreamAnswer::new();
            let answer_for_task = answer.clone();

            tokio::spawn(async move {
                let mut handler = StreamingHandler::new(&request_id, &model_for_task)
                    .with_filter_thinking(filter_thinking);
                let mut buf = Vec::new();
                let mut sent_first_chunk = false;

                let finalize = |handler: &StreamingHandler| {
                    answer_for_task.finalize(handler.filtered_content());
                };

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
                                error_class = FailureClass::from(&e).label(),
                                idle_timeout_ms = idle_timeout_ms,
                                "stream read error or idle timeout"
                            );
                            finalize(&handler);
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
                            finalize(&handler);
                            return;
                        }
                        if let Some(data) = trimmed.strip_prefix("data: ") {
                            // Shared OpenAI stream-delta parser
                            // (fluent_llm::openai).
                            let Some(delta) = normalize::parse_openai_stream_delta(data) else {
                                continue;
                            };

                            if let Some(fr) = &delta.finish_reason {
                                let s = handler.format_chunk(&delta.delta, Some(fr));
                                if !s.is_empty() {
                                    let _ = tx.send_data(Bytes::from(s)).await;
                                }
                                let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                                finalize(&handler);
                                return;
                            }
                            let s = handler.format_chunk(&delta.delta, None);
                            if !s.is_empty() {
                                if !sent_first_chunk {
                                    sent_first_chunk = true;
                                    tracing::info!(
                                        target: "router.dispatch.backend",
                                        model = %model_for_task,
                                        stream_ttfb_ms =
                                            stream_start.elapsed().as_millis() as u64,
                                        "stream first chunk"
                                    );
                                }
                                let _ = tx.send_data(Bytes::from(s)).await;
                            }
                        }
                    }
                }
                finalize(&handler);
                if sent_first_chunk {
                    let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                }
            });

            Ok(StreamResult {
                model,
                body: rx,
                answer: Some(answer),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// RetryBackend - wraps a ChatBackend with exponential-backoff retry
// ---------------------------------------------------------------------------

pub struct RetryBackend {
    inner: Arc<dyn ChatBackend>,
    retry_count: u32,
    retry_base_interval_s: u64,
}

impl RetryBackend {
    pub fn new(inner: Arc<dyn ChatBackend>, retry_count: u32, retry_base_interval_s: u64) -> Self {
        Self {
            inner,
            retry_count,
            retry_base_interval_s,
        }
    }
}

impl ChatBackend for RetryBackend {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let inner = self.inner.clone();
        let max_attempts = (self.retry_count + 1).max(1);
        let base_ms = self.retry_base_interval_s * 1000;

        Box::pin(async move {
            common_core::retry::retry_async(
                max_attempts,
                base_ms,
                0,
                DispatchError::is_retryable,
                || async {
                    inner
                        .complete(
                            request.clone(),
                            model.clone(),
                            params.clone(),
                            idle_timeout_ms,
                            total_timeout_ms,
                            filter_thinking,
                        )
                        .await
                },
            )
            .await
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
        let base_ms = self.retry_base_interval_s * 1000;

        Box::pin(async move {
            // The streaming path applies the total-timeout on each connection
            // attempt; a connection that fails to open within the budget is
            // retried like any other retryable failure.
            common_core::retry::retry_async(
                max_attempts,
                base_ms,
                0,
                DispatchError::is_retryable,
                || async {
                    match tokio::time::timeout(
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
                    .await
                    {
                        Ok(Ok(stream)) => Ok(stream),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err(DispatchError::Http("total timeout".into())),
                    }
                },
            )
            .await
        })
    }
}

// ---------------------------------------------------------------------------
// BackendChain - tries backends in order
// ---------------------------------------------------------------------------

pub struct BackendChain {
    backends: Vec<Arc<dyn ChatBackend>>,
}

impl BackendChain {
    pub fn new(backends: Vec<Arc<dyn ChatBackend>>) -> Self {
        Self { backends }
    }
}

/// A 4xx that is *not* a transient retryable (408/429) — a non-retryable
/// short-circuit for the fallback chain. Mirrors the original per-backend
/// predicate exactly.
fn is_non_retryable_4xx(e: &DispatchError) -> bool {
    if let DispatchError::Http(ref msg) = e {
        msg.starts_with("HTTP 4")
            && !msg.starts_with("HTTP 408")
            && !msg.starts_with("HTTP 429")
    } else {
        false
    }
}

impl ChatBackend for BackendChain {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let backends = self.backends.clone();

        Box::pin(async move {
            match first_accept_in_order(
                backends,
                |backend| {
                    // `complete` consumes the request, so each rung gets its
                    // own owned clone (inherent to the API); the owned rung
                    // `Arc<dyn ChatBackend>` is moved into the future — no
                    // per-rung `Arc::clone`.
                    let request = request.clone();
                    let model = model.clone();
                    let params = params.clone();
                    async move {
                        backend
                            .complete(
                                request,
                                model,
                                params,
                                idle_timeout_ms,
                                total_timeout_ms,
                                filter_thinking,
                            )
                            .await
                            .map(Some)
                    }
                },
                is_non_retryable_4xx,
            )
            .await
            {
                Ok(Some(resp)) => Ok(resp),
                Err(e) => Err(e),
                Ok(None) => Err(DispatchError::AllBackendsFailed),
            }
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
            match first_accept_in_order(
                backends,
                |backend| {
                    let request = request.clone();
                    let model = model.clone();
                    let params = params.clone();
                    async move {
                        backend
                            .stream_complete(
                                request,
                                model,
                                params,
                                idle_timeout_ms,
                                total_timeout_ms,
                                filter_thinking,
                            )
                            .await
                            .map(Some)
                    }
                },
                is_non_retryable_4xx,
            )
            .await
            {
                Ok(Some(stream)) => Ok(stream),
                Err(e) => Err(e),
                Ok(None) => Err(DispatchError::AllBackendsFailed),
            }
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
            _filter_thinking: bool,
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
                            answer: None,
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
}
