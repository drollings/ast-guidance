use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use crate::metrics::FailureClass;
use bytes::Bytes;
use common_core::hash::uuid_v4;
use fluent_concurrency::ladder::first_accept_in_order;
use fluent_concurrency::stream::StreamAbort;
use fluent_llm::HttpClass;
use fluent_llm::openai::drain_sse_lines;
use fluent_llm::thinking::strip_thinking_blocks;
use http_body::Frame;
use serde_json::Value;

use crate::dispatch::frontier::{DispatchError, OpenAiBackend};
use crate::normalize;
use crate::streaming::StreamingHandler;
use crate::types::{RouterRequest, RouterResponse};
// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The client-facing SSE response body. An `http_body_util::channel::Channel`
/// receiver whose `Drop` fires the stream's [`StreamAbort`].
///
/// The body's lifetime is the single source of truth for "the consumer is
/// gone": hyper drops it the moment it detects the client is no longer reading
/// (its next socket write fails), and the forwarding task — which holds a clone
/// of the same signal — reacts by dropping the upstream connection (and, via
/// the dispatch layer, issuing a management-plane abort). Without this, the
/// forwarding task would keep draining the upstream generation to the end.
pub struct StreamBody {
    inner: http_body_util::channel::Channel<Bytes, std::convert::Infallible>,
    abort: StreamAbort,
}

impl Drop for StreamBody {
    fn drop(&mut self) {
        self.abort.cancel();
    }
}

impl http_body::Body for StreamBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Streaming result - a channel that receives SSE-formatted response data
/// in a spawned background task, plus the abort surface shared by that task
/// and the downstream body drop-guard.
pub struct StreamHandle {
    pub model: String,
    /// The body handed to the client; dropping it fires [`Self::abort`].
    pub body: StreamBody,
    /// Best-effort finalization sink for the assembled answer text.
    /// The streaming task writes `filtered_content()` here when the stream
    /// ends (including on abort, so the ledger records the partial answer);
    /// `None` for backends/stubs that don't accumulate content.
    pub answer: Option<crate::streaming::StreamAnswer>,
    /// The cancellation signal: fires when the client stops consuming the
    /// body, and is what the forwarding task and the management-abort watcher
    /// wait on.
    pub abort: StreamAbort,
}

// ---------------------------------------------------------------------------
// Trait - abstraction over LLM chat completion providers
// ---------------------------------------------------------------------------

/// Object-safe for `Arc<dyn DispatchBackend>`. Provider config established at
/// construction time; per-request fields passed as arguments.
pub trait DispatchBackend: Send + Sync {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>>;

    /// Stream a completion with no abort signal. Delegates to
    /// [`Self::stream_complete_with_abort`] with `None`.
    fn stream_complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        self.stream_complete_with_abort(
            request,
            model,
            params,
            idle_timeout_ms,
            total_timeout_ms,
            filter_thinking,
            None,
        )
    }

    /// Stream a completion, optionally carrying the downstream abort signal.
    ///
    /// The `abort` token is the forward-facing cancellation: when the client
    /// stops consuming the response body, the token fires, the backend drops
    /// its upstream connection (the fork then interrupts the slot task), and
    /// the dispatch layer may issue an explicit management-plane abort.
    ///
    /// Wrapper backends (`RetryBackend`, `BackendChain`) MUST forward this
    /// token to the inner backend unchanged — the transparent-delegation rule
    /// from the fluent-wvr wrapper pattern — so a downstream abort reaches the
    /// transport regardless of how many retry/fallback layers sit in between.
    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>>;
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

/// Apply an optional `Authorization: Bearer` header to a request. `api_key_env`
/// names the environment variable holding the token; a missing/unreadable
/// variable degrades to no header (fail-open), matching the frontier backend's
/// `frontier_api_client` convention.
fn apply_auth(
    mut request: reqwest::RequestBuilder,
    api_key_env: Option<&str>,
) -> reqwest::RequestBuilder {
    let Some(env) = api_key_env else {
        return request;
    };
    match std::env::var(env) {
        Ok(key) if !key.is_empty() => {
            request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"));
        }
        _ => tracing::warn!(
            target: "router.dispatch.backend",
            env = %env,
            "model api_key env var unreadable - dispatch without auth header",
        ),
    }
    request
}

/// Extract the group name from the fork's 503 group-miss payload
/// (`unavailable_error` "no free instance in group '<group>'"). `None` when the
/// body is not a group-miss (a generic 503).
///
/// The fork's real payload is JSON
/// (`{"error":{"message":"no free instance in group 'X'"}}`), so parse that
/// shape first; fall back to the raw substring scan so a non-JSON body (or an
/// older wire format) never regresses.
///
/// M2: intentionally a pristine `from_str::<Value>`, NOT the tolerant LLM
/// codec. This parses our own fork's machine-generated HTTP error body —
/// never LLM-produced text. The substring fallback already covers non-JSON
/// bodies, so tolerance here would add nothing but a wider accept set on an
/// error path.
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
    api_key: Option<String>,
}

impl OpenAiChatBackend {
    pub fn new(client: reqwest::Client, endpoint_url: impl Into<String>) -> Self {
        Self {
            client,
            endpoint_url: endpoint_url.into(),
            api_key: None,
        }
    }

    /// Attach an `Authorization: Bearer` token (from an external endpoint's
    /// configured `api_key` env var, resolved at dispatch time).
    #[must_use]
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }
}

impl DispatchBackend for OpenAiChatBackend {
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
        let api_key_env = self.api_key.clone();

        Box::pin(async move {
            let start = Instant::now();
            let log_model = model.clone();
            let result: Result<RouterResponse, DispatchError> = async {
                let body =
                    build_chat_body(&request, &model, params.as_ref(), false, filter_thinking)?;
                let response =
                    with_total_timeout(total_timeout_ms, "total timeout exceeded", async {
                        apply_auth(client.post(&url), api_key_env.as_deref())
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

    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        let url = dispatch_url(&self.endpoint_url);
        let client = self.client.clone();
        let api_key_env = self.api_key.clone();
        let model_for_task = model.clone();
        let request_id = uuid_v4();

        Box::pin(async move {
            let stream_start = Instant::now();
            let body = build_chat_body(&request, &model, params.as_ref(), true, filter_thinking)?;
            let mut response = with_total_timeout(
                total_timeout_ms,
                "total timeout exceeded for stream connection",
                async {
                    apply_auth(client.post(&url), api_key_env.as_deref())
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

            // The abort token. `Some` when the dispatch layer passed one in
            // (it arms a management-plane abort on the same signal); a private
            // one when a bare `stream_complete` caller is driving, so the body
            // drop-guard still stops the upstream in that case too.
            let abort = abort.unwrap_or_default();
            // The body fires `abort` when hyper drops it (client gone); the
            // body's lifetime is the single source of truth for that fact.
            let body = StreamBody {
                inner: rx,
                abort: abort.clone(),
            };

            // Assemble the streamed answer and finalize it when the stream
            // ends (including on abort, so the ledger records the partial
            // answer) so the handler can record it into the ledger + session
            // step.
            let answer = crate::streaming::StreamAnswer::new();
            let answer_for_task = answer.clone();
            let abort_for_task = abort.clone();

            tokio::spawn(async move {
                let mut handler = StreamingHandler::new(&request_id, &model_for_task)
                    .with_filter_thinking(filter_thinking);
                let mut buf = Vec::new();
                let mut sent_first_chunk = false;

                let finalize = |handler: &StreamingHandler| {
                    answer_for_task.finalize(handler.filtered_content());
                };

                loop {
                    // Park point 1 — the upstream read, racing the downstream
                    // abort. The idle timeout still bounds a stalled read; the
                    // abort arm fires the moment the client's body is dropped
                    // (`StreamBody::drop` -> cancel) and `return`s, dropping
                    // `response` so the upstream connection closes and the
                    // fork interrupts the slot's generation.
                    let chunk = tokio::select! {
                        res = with_total_timeout(
                            idle_timeout_ms,
                            "idle timeout waiting for stream chunk",
                            async {
                                response
                                    .chunk()
                                    .await
                                    .map_err(|e| DispatchError::Http(e.to_string()))
                            },
                        ) => match res {
                            Ok(Some(b)) => Some(b),
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
                        },
                        () = abort_for_task.cancelled() => {
                            tracing::info!(
                                target: "router.dispatch",
                                model = %model_for_task,
                                "downstream disconnected - aborting upstream stream"
                            );
                            finalize(&handler);
                            return;
                        }
                    };

                    let Some(chunk) = chunk else {
                        continue;
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
                                    // Park point 2 — a blocked send is released
                                    // with `Err` the moment hyper drops the
                                    // receiver, so a dead client never wedges
                                    // the task; treat a failed send as
                                    // downstream-gone.
                                    if tx.send_data(Bytes::from(s)).await.is_err() {
                                        finalize(&handler);
                                        return;
                                    }
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
                                if tx.send_data(Bytes::from(s)).await.is_err() {
                                    finalize(&handler);
                                    return;
                                }
                            }
                        }
                    }
                }
                finalize(&handler);
                if sent_first_chunk {
                    let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                }
            });

            Ok(StreamHandle {
                model,
                body,
                answer: Some(answer),
                abort,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// RetryBackend - wraps a DispatchBackend with exponential-backoff retry
// ---------------------------------------------------------------------------

pub struct RetryBackend {
    inner: Arc<dyn DispatchBackend>,
    retry_count: u32,
    retry_base_interval_s: u64,
}

impl RetryBackend {
    pub fn new(inner: Arc<dyn DispatchBackend>, retry_count: u32, retry_base_interval_s: u64) -> Self {
        Self {
            inner,
            retry_count,
            retry_base_interval_s,
        }
    }
}

impl DispatchBackend for RetryBackend {
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

    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        let inner = self.inner.clone();
        let max_attempts = (self.retry_count + 1).max(1);
        let base_ms = self.retry_base_interval_s * 1000;

        Box::pin(async move {
            // The streaming path applies the total-timeout on each connection
            // attempt; a connection that fails to open within the budget is
            // retried like any other retryable failure. The abort token is
            // forwarded unchanged to the inner backend (a downstream abort
            // cannot fire before a stream handle exists, so the retry loop is
            // not resurrecting a dead stream).
            common_core::retry::retry_async(
                max_attempts,
                base_ms,
                0,
                DispatchError::is_retryable,
                || async {
                    let abort = abort.clone();
                    match tokio::time::timeout(
                        Duration::from_millis(total_timeout_ms),
                        inner.stream_complete_with_abort(
                            request.clone(),
                            model.clone(),
                            params.clone(),
                            idle_timeout_ms,
                            total_timeout_ms,
                            filter_thinking,
                            abort,
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
    backends: Vec<Arc<dyn DispatchBackend>>,
}

impl BackendChain {
    pub fn new(backends: Vec<Arc<dyn DispatchBackend>>) -> Self {
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

impl DispatchBackend for BackendChain {
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
                    // `Arc<dyn DispatchBackend>` is moved into the future — no
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

    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        idle_timeout_ms: u64,
        total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        let backends = self.backends.clone();

        Box::pin(async move {
            match first_accept_in_order(
                backends,
                |backend| {
                    let request = request.clone();
                    let model = model.clone();
                    let params = params.clone();
                    let abort = abort.clone();
                    async move {
                        backend
                            .stream_complete_with_abort(
                                request,
                                model,
                                params,
                                idle_timeout_ms,
                                total_timeout_ms,
                                filter_thinking,
                                abort,
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
// OnnxDispatchBackend
// ---------------------------------------------------------------------------

/// Adapts the in-process onnx generative `ChatBackend` (a
/// `fluent_llm::client::ChatBackend`, e.g. the `onnx/llm` routing model) to the
/// dispatch `DispatchBackend`, so a route whose group resolves to an onnx role is
/// served in-process rather than over HTTP. Onnx decodes are buffered-only
/// (single-shot text), so streaming requests degrade to a buffered completion
/// wrapped as one SSE chunk — the normal dispatch path never streams to an onnx
/// target (`RoutingTarget::from_onnx_role` sets `stream: false`).
pub struct OnnxDispatchBackend {
    backend: Arc<dyn fluent_llm::client::ChatBackend>,
}

impl OnnxDispatchBackend {
    pub fn new(backend: Arc<dyn fluent_llm::client::ChatBackend>) -> Self {
        Self { backend }
    }

    /// Run the buffered completion against the onnx backend and wrap its text
    /// into a `RouterResponse`. `params` carries the merged sampling + routing
    /// fields the onnx backend honors (`instance`/`snapshot`/`id_slot`).
    fn complete_impl(
        backend: &Arc<dyn fluent_llm::client::ChatBackend>,
        request: &RouterRequest,
        model: &str,
        params: Option<&Value>,
        filter_thinking: bool,
    ) -> Result<RouterResponse, DispatchError> {
        let messages: Vec<fluent_llm::ChatMessage> = request
            .messages
            .iter()
            .map(|m| fluent_llm::ChatMessage {
                role: m.role.clone(),
                content: m.content.to_string_lossy(),
            })
            .collect();
        let extras = params.cloned().unwrap_or_else(|| Value::Object(Default::default()));
        let text = backend
            .chat_complete_with_extras(&messages, &extras)
            .map_err(|e| DispatchError::Http(e.to_string()))?;
        let mut resp = crate::server::responses::make_text_completion(model, &text);
        if filter_thinking {
            for choice in &mut resp.choices {
                if let crate::types::RouterMessageContent::Text(ref mut t) = choice.message.content {
                    *t = strip_thinking_blocks(t);
                }
            }
        }
        Ok(resp)
    }
}

impl DispatchBackend for OnnxDispatchBackend {
    fn complete(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        _idle_timeout_ms: u64,
        _total_timeout_ms: u64,
        filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        let backend = self.backend.clone();
        Box::pin(async move {
            Self::complete_impl(&backend, &request, &model, params.as_ref(), filter_thinking)
        })
    }

    fn stream_complete_with_abort(
        &self,
        request: RouterRequest,
        model: String,
        params: Option<Value>,
        _idle_timeout_ms: u64,
        _total_timeout_ms: u64,
        filter_thinking: bool,
        abort: Option<StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        let backend = self.backend.clone();
        Box::pin(async move {
            let completion =
                Self::complete_impl(&backend, &request, &model, params.as_ref(), filter_thinking)?;
            let content = completion
                .choices
                .first()
                .map(|c| c.message.content.to_string_lossy())
                .unwrap_or_default();

            let (mut tx, rx) = http_body_util::channel::Channel::new(32);
            let abort = abort.unwrap_or_default();
            let body = StreamBody {
                inner: rx,
                abort: abort.clone(),
            };
            let answer = crate::streaming::StreamAnswer::new();
            let answer_for_task = answer.clone();
            let request_id = uuid_v4();
            let model_for_task = model.clone();
            tokio::spawn(async move {
                let mut handler =
                    StreamingHandler::new(&request_id, &model_for_task).with_filter_thinking(filter_thinking);
                let _ = tx.send_data(Bytes::from(handler.format_chunk(&content, Some("stop")))).await;
                let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                answer_for_task.finalize(content);
            });
            Ok(StreamHandle {
                model,
                body,
                answer: Some(answer),
                abort,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "../../tests/dispatch_backend.rs"]
mod tests;
