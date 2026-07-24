//! HTTP server exposing the router pipeline as an OpenAI-compatible endpoint.
//! Uses hyper for HTTP with SSE streaming support via http-body-util::channel.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fluent_wvr::prelude::*;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use crate::config::{ModelEntry, RouteRef, ServerConfig};
use crate::normalize;
use crate::pipeline::PipelineOrchestrator;
use crate::streaming::StreamingHandler;
use crate::testing::mock::MockDispatchContext;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse, Usage};

type ResponseBody = http_body_util::combinators::UnsyncBoxBody<Bytes, Infallible>;
type HyperResponse = hyper::Response<ResponseBody>;

const CORS_HEADERS: &[(&str, &str)] = &[
    ("access-control-allow-origin", "*"),
    ("access-control-allow-methods", "POST, GET, OPTIONS"),
    ("access-control-allow-headers", "Content-Type, Authorization"),
];

pub struct RouterServer {
    name: ArcIntern<str>,
    pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
    routes: HashMap<String, RouteRef>,
    models: HashMap<String, ModelEntry>,
    bind_addr: String,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

struct ServerStats {
    requests: AtomicU64,
    errors: AtomicU64,
    rejections: AtomicU64,
}

impl RouterServer {
    pub fn new(
        pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
        routes: HashMap<String, RouteRef>,
        models: HashMap<String, ModelEntry>,
        config: &ServerConfig,
        classifier_url: Option<String>,
    ) -> Self {
        Self {
            name: ArcIntern::from("router.server"),
            pipelines,
            routes,
            models,
            bind_addr: config.bind_addr.clone(),
            max_payload: config.max_payload,
            classifier_url,
            mock_dispatch: None,
            depends: vec![],
            provides: vec![ArcIntern::from("http.endpoint")],
        }
    }

    #[must_use]
    pub fn with_mock(mut self, mock_dispatch: MockDispatchContext) -> Self {
        tracing::info!(
            target: "router.server",
            except_count = mock_dispatch.except_models.len(),
            "mock dispatch enabled"
        );
        self.mock_dispatch = Some(Arc::new(mock_dispatch));
        self
    }

    pub async fn serve(&self) -> Result<(), String> {
        tracing::info!(
            target: "router.server",
            bind_addr = %self.bind_addr,
            has_mock = self.mock_dispatch.is_some(),
            "serving HTTP"
        );
        run_http(
            Arc::new(self.pipelines.clone()),
            Arc::new(self.routes.clone()),
            Arc::new(self.models.clone()),
            &self.bind_addr,
            self.max_payload,
            self.classifier_url.clone(),
            self.mock_dispatch.clone(),
        )
        .await
    }
}

impl WorkUnit for RouterServer {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let pipelines = Arc::new(self.pipelines.clone());
        let routes = Arc::new(self.routes.clone());
        let models = Arc::new(self.models.clone());
        let bind_addr = self.bind_addr.clone();
        let max_payload = self.max_payload;
        let classifier_url = self.classifier_url.clone();
        let mock_dispatch = self.mock_dispatch.clone();

        let rt = ctx.rt.clone();

        let _handle = rt.spawn(Box::pin(async move {
            if let Err(e) =
                run_http(pipelines, routes, models, &bind_addr, max_payload, classifier_url, mock_dispatch)
                    .await
            {
                tracing::error!(target: "router.server", error = %e, "HTTP server error");
            }
        }));

        Ok(WorkOutput::ok(format!(
            "HTTP server bound to {}",
            self.bind_addr
        )))
    }
}

impl FieldAccess for RouterServer {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "RouterServer has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "RouterServer has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for RouterServer {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(RouterServer);

async fn run_http(
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<HashMap<String, RouteRef>>,
    models: Arc<HashMap<String, ModelEntry>>,
    bind_addr: &str,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| format!("bind {bind_addr} failed: {e}"))?;

    tracing::info!(target: "router.server", addr = %bind_addr, "HTTP server listening (hyper)");

    let stats = Arc::new(ServerStats {
        requests: AtomicU64::new(0),
        errors: AtomicU64::new(0),
        rejections: AtomicU64::new(0),
    });

    let http_client = Arc::new(reqwest::Client::new());

    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "router.server", error = %e, "accept error");
                continue;
            }
        };

        let pipelines = pipelines.clone();
        let routes = routes.clone();
        let models = models.clone();
        let stats = stats.clone();
        let classifier_url = classifier_url.clone();
        let mock_dispatch = mock_dispatch.clone();
        let http_client = http_client.clone();

        tokio::spawn(async move {
            let io = TokioIo::new(stream);

            let service = hyper::service::service_fn(move |req| {
                handle_request(
                    req,
                    pipelines.clone(),
                    routes.clone(),
                    models.clone(),
                    stats.clone(),
                    max_payload,
                    classifier_url.clone(),
                    mock_dispatch.clone(),
                    http_client.clone(),
                )
            });

            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                if !e.to_string().contains("connection closed")
                    && !e.to_string().contains("shutdown")
                {
                    tracing::error!(target: "router.server", error = %e, "hyper connection error");
                }
            }
        });
    }
}

async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<HashMap<String, RouteRef>>,
    models: Arc<HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    http_client: Arc<reqwest::Client>,
) -> Result<HyperResponse, Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if method == hyper::Method::OPTIONS {
        return Ok(empty_response(hyper::StatusCode::NO_CONTENT));
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            Ok(json_response(
                hyper::StatusCode::OK,
                &serde_json::json!({"status": "ok"}),
            ))
        }
        ("GET", "/stats") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            let body = serde_json::json!({
                "requests": stats.requests.load(Ordering::Relaxed),
                "errors": stats.errors.load(Ordering::Relaxed),
                "rejections": stats.rejections.load(Ordering::Relaxed),
            });
            Ok(json_response(hyper::StatusCode::OK, &body))
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completion(
                req, pipelines, routes, models, stats, max_payload,
                classifier_url, mock_dispatch, http_client,
            )
            .await
        }
        _ => {
            let code = if path == "/v1/chat/completions" {
                hyper::StatusCode::METHOD_NOT_ALLOWED
            } else {
                hyper::StatusCode::NOT_FOUND
            };
            Ok(empty_response(code))
        }
    }
}

async fn handle_chat_completion(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<HashMap<String, RouteRef>>,
    models: Arc<HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    http_client: Arc<reqwest::Client>,
) -> Result<HyperResponse, Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(hyper::StatusCode::BAD_REQUEST, &format!("body read error: {e}")));
        }
    };

    if body_bytes.len() > max_payload {
        return Ok(empty_response(hyper::StatusCode::PAYLOAD_TOO_LARGE));
    }

    if body_bytes.is_empty() {
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(error_response(
            hyper::StatusCode::BAD_REQUEST,
            "empty body",
        ));
    }

    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    let body_json: serde_json::Value = match serde_json::from_str(body_str) {
        Ok(v) => v,
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {e}"),
            ));
        }
    };

    let router_request = match normalize::normalize_request(body_json) {
        Ok(r) => r,
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(
                hyper::StatusCode::BAD_REQUEST,
                &e.to_string(),
            ));
        }
    };

    let is_stream = router_request.stream.unwrap_or(false);
    let model_name = router_request.model.clone();
    let user_message: String = router_request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| {
            let s = m.content.to_string_lossy();
            if s.len() > 120 {
                format!("{}...", &s[..120])
            } else {
                s
            }
        })
        .unwrap_or_default();
    let request_json = serde_json::to_string(&router_request).unwrap_or_default();

    tracing::info!(
        target: "router.server",
        model = %model_name,
        user_message = %user_message,
        messages = router_request.messages.len(),
        stream = is_stream,
        "incoming request"
    );

    stats.requests.fetch_add(1, Ordering::Relaxed);

    let pipeline_result = resolve_pipeline(
        &model_name,
        &routes,
        &models,
        &pipelines,
        &request_json,
    );

    let user_text = router_request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.to_string_lossy())
        .unwrap_or_default();

    if pipeline_result.rejected {
        stats.rejections.fetch_add(1, Ordering::Relaxed);
        let reason = pipeline_result
            .reject_reason
            .as_deref()
            .unwrap_or("request rejected");

        if let Some(ref mock) = mock_dispatch {
            if let Some(entry) = mock.lookup(&user_text) {
                mock.validate_rejection(entry, reason);
            }
        }

        let error_output = format!("ERROR: {reason}");
        let completion = make_error_completion(&model_name, &error_output);
        return Ok(completion_to_response(&completion, &model_name, is_stream, None));
    }

    if let Some(ref resp_str) = pipeline_result.classifier_response {
        tracing::info!(
            target: "router.server",
            model = %model_name,
            response_len = resp_str.len(),
            "responding with classifier direct response"
        );
        let completion = make_text_completion(&model_name, resp_str);
        return Ok(completion_to_response(&completion, &model_name, is_stream, None));
    }

    if let Some(ref rt) = pipeline_result.routing_target {
        return handle_dispatch(
            rt,
            &router_request,
            &model_name,
            &user_text,
            mock_dispatch.as_ref(),
            &http_client,
            is_stream,
        )
        .await;
    }

    // Classifier fallback
    if let Some(ref url) = classifier_url {
        tracing::info!(
            target: "router.server",
            model = %model_name,
            fallback_url = %url,
            "no routing target — dispatching to classifier fallback"
        );
        let rt_for_fallback = crate::pipeline::RoutingTarget {
            url: url.clone(),
            model: model_name.clone(),
            group: None,
            target_name: None,
            params: None,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            stream: false,
            idle_timeout_ms: 30000,
            total_timeout_ms: 30000,
        };
        return handle_dispatch(
            &rt_for_fallback,
            &router_request,
            &model_name,
            &user_text,
            mock_dispatch.as_ref(),
            &http_client,
            false,
        )
        .await;
    }

    tracing::warn!(
        target: "router.server",
        model = %model_name,
        "no routing target, no classifier response, no classifier url — returning fallback"
    );
    let completion = fallback_completion(&model_name);
    Ok(completion_to_response(&completion, &model_name, is_stream, None))
}

async fn handle_dispatch(
    rt: &crate::pipeline::RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    user_text: &str,
    mock_dispatch: Option<&Arc<MockDispatchContext>>,
    http_client: &reqwest::Client,
    is_stream: bool,
) -> Result<HyperResponse, Infallible> {
    let target_streams = is_stream && rt.stream;

    if let Some(mock) = mock_dispatch {
        if let Some(entry) = mock.lookup(user_text) {
            mock.validate_route(entry, Some(rt));
            if mock.is_model_excepted(&rt.model) || mock.is_model_excepted(model_name) {
                tracing::info!(target: "router.server", model = %rt.model, "excepted model — real LLM call");
                return dispatch_real(rt, router_request, model_name, http_client, target_streams).await;
            }
            tracing::info!(target: "router.server", model = %model_name, "mock canned response");
            let completion = mock.dispatch_response(entry, model_name);
            return Ok(completion_to_response(&completion, model_name, is_stream, None));
        }
        tracing::debug!(target: "router.server", model = %model_name, transcript_found = false, "no transcript entry — real dispatch fallback");
    }

    tracing::info!(
        target: "router.server",
        model = %rt.model,
        url = %rt.url,
        stream = target_streams,
        retry = rt.retry_count,
        "real dispatch"
    );

    dispatch_real(rt, router_request, model_name, http_client, target_streams).await
}

async fn dispatch_real(
    rt: &crate::pipeline::RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    http_client: &reqwest::Client,
    stream: bool,
) -> Result<HyperResponse, Infallible> {
    let url = build_dispatch_url(&rt.url);

    if stream {
        match dispatch_to_llm_streaming(
            http_client,
            &url,
            router_request,
            &rt.model,
            model_name,
            rt.params.as_ref(),
            rt.filter_thinking,
            rt.retry_count,
            rt.retry_base_interval_s,
            rt.idle_timeout_ms,
            rt.total_timeout_ms,
        )
        {
            Ok(body) => {
                let mut resp = HyperResponse::new(body.boxed_unsync());
                *resp.status_mut() = hyper::StatusCode::OK;
                resp.headers_mut()
                    .insert(hyper::header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
                add_cors_headers(resp.headers_mut());
                Ok(resp)
            }
            Err(e) => {
                tracing::warn!(target: "router.server", error = %e, "streaming dispatch failed, using fallback");
                let completion = fallback_completion(model_name);
                Ok(completion_to_response(&completion, model_name, true, None))
            }
        }
    } else {
        let completion = dispatch_to_llm_buffered(
            http_client,
            &url,
            router_request,
            &rt.model,
            rt.params.as_ref(),
            rt.retry_count,
            rt.retry_base_interval_s,
        )
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(target: "router.server", error = %e, "dispatch failed, using fallback");
            fallback_completion(model_name)
        });
        Ok(completion_to_response(&completion, model_name, false, Some(model_name)))
    }
}

fn resolve_pipeline(
    model_name: &str,
    routes: &HashMap<String, RouteRef>,
    models: &HashMap<String, ModelEntry>,
    pipelines: &HashMap<String, Arc<PipelineOrchestrator>>,
    request_json: &str,
) -> crate::pipeline::PipelineResult {
    let route = routes.get(model_name).cloned();

    let pipeline_names: Vec<String> = if let Some(ref r) = route {
        r.pipelines.clone()
    } else if let Some(model_entry) = models.get(model_name) {
        let rt = crate::pipeline::RoutingTarget {
            url: model_entry.endpoint.clone(),
            model: model_entry
                .name
                .clone()
                .unwrap_or_else(|| model_name.to_string()),
            group: None,
            target_name: Some(model_name.to_string()),
            params: model_entry.params.clone(),
            filter_thinking: model_entry.filter_thinking,
            retry_count: model_entry.retry_count,
            retry_base_interval_s: model_entry.retry_base_interval_s,
            stream: model_entry.stream,
            idle_timeout_ms: model_entry.idle_timeout_ms,
            total_timeout_ms: model_entry.total_timeout_ms,
        };
        return crate::pipeline::PipelineResult {
            decisions: vec![],
            final_response: None,
            rejected: false,
            reject_reason: None,
            routing_target: Some(rt),
            classifier_response: None,
        };
    } else {
        routes
            .get("local")
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    };

    let mut ctx = WorkContext::default();
    ctx.metadata.insert(
        "request".into(),
        MetadataValue::String(request_json.to_string()),
    );

    let mut all_decisions = Vec::new();
    let mut last_result: Option<crate::pipeline::PipelineResult> = None;

    for name in &pipeline_names {
        let Some(pipeline) = pipelines.get(name) else {
            tracing::warn!(target: "router.server", pipeline = %name, "pipeline not found, skipping");
            continue;
        };

        let output = match pipeline.execute(&ctx) {
            Ok(o) => o,
            Err(e) => {
                return crate::pipeline::PipelineResult {
                    decisions: all_decisions,
                    final_response: None,
                    rejected: true,
                    reject_reason: Some(format!("pipeline '{name}' error: {e}")),
                    routing_target: None,
                    classifier_response: None,
                };
            }
        };

        let mut result: crate::pipeline::PipelineResult = match output.data_take() {
            Ok(r) => r,
            Err(e) => {
                return crate::pipeline::PipelineResult {
                    decisions: all_decisions,
                    final_response: None,
                    rejected: true,
                    reject_reason: Some(format!("pipeline '{name}' output decode: {e}")),
                    routing_target: None,
                    classifier_response: None,
                };
            }
        };

        if result.rejected {
            return result;
        }

        all_decisions.append(&mut result.decisions);
        last_result = Some(result);
    }

    let mut final_result = last_result.unwrap_or(crate::pipeline::PipelineResult {
        decisions: vec![],
        final_response: None,
        rejected: false,
        reject_reason: None,
        routing_target: None,
        classifier_response: None,
    });
    final_result.decisions = all_decisions;
    final_result
}

fn build_dispatch_url(endpoint_url: &str) -> String {
    if endpoint_url.ends_with("/chat/completions") {
        endpoint_url.to_string()
    } else {
        format!("{}/chat/completions", endpoint_url.trim_end_matches('/'))
    }
}

fn dispatch_to_llm_streaming(
    http_client: &reqwest::Client,
    endpoint_url: &str,
    request: &RouterRequest,
    model_name: &str,
    response_model: &str,
    params: Option<&serde_json::Value>,
    filter_thinking: bool,
    retry_count: u32,
    retry_base_interval_s: u64,
    idle_timeout_ms: u64,
    total_timeout_ms: u64,
) -> Result<http_body_util::channel::Channel<Bytes, Infallible>, String> {
    let messages = normalize::messages_to_json(request);

    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
        "stream": true,
    });

    if let Some(p) = params {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                if k != "stream" {
                    body[k] = v.clone();
                }
            }
        }
    }

    if !body.as_object().is_some_and(|o| o.contains_key("temperature")) {
        if let Some(temp) = request.temperature {
            body["temperature"] =
                serde_json::Value::Number(serde_json::Number::from_f64(temp).ok_or("invalid temperature")?);
        }
    }
    if !body.as_object().is_some_and(|o| o.contains_key("max_tokens")) {
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(max_tokens.into());
        }
    }

    let request_id = uuid_v4();
    let (mut tx, rx_body) = http_body_util::channel::Channel::<Bytes>::new(32);

    let url = endpoint_url.to_string();
    let model = model_name.to_string();
    let resp_model = response_model.to_string();
    let client = http_client.clone();

    tokio::spawn(async move {
        if let Err(e) = stream_dispatch_inner(
            &client, &url, &body, &model, &request_id, &resp_model,
            filter_thinking, retry_count, retry_base_interval_s,
            idle_timeout_ms, total_timeout_ms, &mut tx,
        )
        .await
        {
            tracing::warn!(target: "router.dispatch", error = %e, "stream dispatch ended with error");
        }
    });

    Ok(rx_body)
}

async fn stream_dispatch_inner(
    http_client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    model_name: &str,
    request_id: &str,
    response_model: &str,
    filter_thinking: bool,
    retry_count: u32,
    retry_base_interval_s: u64,
    idle_timeout_ms: u64,
    total_timeout_ms: u64,
    tx: &mut http_body_util::channel::Sender<Bytes>,
) -> Result<(), String> {
    let max_attempts = (retry_count + 1).max(1);
    let idle_dur = Duration::from_millis(idle_timeout_ms);
    let total_dur = Duration::from_millis(total_timeout_ms);

    let stream = tokio::time::timeout(total_dur, async move {
        let mut last_err = String::new();

        for attempt in 0..max_attempts {
            let req = http_client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body.to_string());

            match req.send().await {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response);
                    }
                    let status_err = format!("HTTP {status}");
                    tracing::warn!(target: "router.dispatch", url = %url, attempt = attempt + 1, error = %status_err, "non-success status");
                    last_err = status_err;
                }
                Err(e) => {
                    last_err = format!("HTTP error: {e}");
                    tracing::warn!(target: "router.dispatch", url = %url, attempt = attempt + 1, error = %last_err, "request failed");
                }
            }

            if attempt + 1 < max_attempts {
                let delay = retry_base_interval_s * 1000 * (1u64 << attempt);
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
        }

        Err(last_err)
    })
    .await
    .map_err(|_| "total timeout exceeded".to_string())?
    .map_err(|e| format!("dispatch failed after {max_attempts} attempts: {e}"))?;

    let mut response_stream = stream;

    let mut handler = StreamingHandler::new(request_id, format!("{response_model}#stream"));
    let mut line_buf = Vec::new();
    let mut thinking = filter_thinking;
    let mut sent_first_chunk = false;

    loop {
        let chunk_result = tokio::time::timeout(idle_dur, response_stream.chunk()).await;

        let chunk = match chunk_result {
            Ok(Ok(Some(bytes))) => bytes,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => {
                let err = format!("upstream stream error: {e}");
                tracing::warn!(target: "router.dispatch", error = %err, "stream read error");
                if sent_first_chunk {
                    let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                }
                return Err(err);
            }
            Err(_) => {
                let err = format!("idle timeout after {idle_timeout_ms}ms");
                tracing::warn!(target: "router.dispatch", model = %model_name, error = %err, "idle timeout");
                if sent_first_chunk {
                    let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                }
                return Err(err);
            }
        };

        line_buf.extend_from_slice(&chunk);

        while let Some(pos) = line_buf.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&line_buf[..pos]).to_string();
            line_buf.drain(..=pos);

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "data: [DONE]" {
                let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                return Ok(());
            }

            if let Some(data) = trimmed.strip_prefix("data: ") {
                if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(data) {
                    let empty = vec![];
                    let choices = chunk_json
                        .get("choices")
                        .and_then(|v| v.as_array())
                        .unwrap_or(&empty);

                    if let Some(choice) = choices.first() {
                        let delta = choice
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        let finish_reason = choice
                            .get("finish_reason")
                            .and_then(|v| v.as_str());

                        if let Some(fr) = finish_reason {
                            let _ = tx.send_data(Bytes::from(handler.format_chunk(delta, Some(fr)))).await;
                            let _ = tx.send_data(Bytes::from(handler.format_done())).await;
                            return Ok(());
                        }

                        if thinking && !delta.is_empty() {
                            thinking = false;
                        }

                        if !thinking {
                            sent_first_chunk = true;
                            let _ = tx.send_data(Bytes::from(handler.format_chunk(delta, None))).await;
                        }
                    }
                }
            }
        }
    }

    if sent_first_chunk {
        let _ = tx.send_data(Bytes::from(handler.format_done())).await;
    }
    Ok(())
}

async fn dispatch_to_llm_buffered(
    http_client: &reqwest::Client,
    endpoint_url: &str,
    request: &RouterRequest,
    model_name: &str,
    params: Option<&serde_json::Value>,
    retry_count: u32,
    retry_base_interval_s: u64,
) -> Result<RouterResponse, String> {
    let url = build_dispatch_url(endpoint_url);

    tracing::info!(
        target: "router.dispatch",
        url = %url,
        model = %model_name,
        retry_count = retry_count,
        message_count = request.messages.len(),
        "dispatching to LLM (buffered)"
    );

    let messages = normalize::messages_to_json(request);

    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
    });

    if let Some(p) = params {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                if k != "stream" {
                    body[k] = v.clone();
                }
            }
        }
    }

    if !body.as_object().is_some_and(|o| o.contains_key("temperature")) {
        if let Some(temp) = request.temperature {
            body["temperature"] =
                serde_json::Value::Number(serde_json::Number::from_f64(temp).ok_or("invalid temperature")?);
        }
    }
    if !body.as_object().is_some_and(|o| o.contains_key("max_tokens")) {
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(max_tokens.into());
        }
    }

    let max_attempts = (retry_count + 1).max(1);
    let mut last_err = String::new();

    for attempt in 0..max_attempts {
        let result = http_client.post(&url).json(&body).send().await;

        match result {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    let json: serde_json::Value = response
                        .json()
                        .await
                        .map_err(|e| format!("response JSON parse: {e}"))?;
                    return Ok(parse_openai_response(&json, model_name));
                }
                last_err = format!("HTTP {status}");
            }
            Err(e) => {
                last_err = format!("HTTP error: {e}");
            }
        }

        if attempt + 1 < max_attempts {
            let delay = retry_base_interval_s * 1000 * (1u64 << attempt);
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
    }

    Err(format!(
        "dispatch failed after {max_attempts} attempts: {last_err}"
    ))
}

fn parse_openai_response(json: &serde_json::Value, fallback_model: &str) -> RouterResponse {
    let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let created = json.get("created").and_then(serde_json::Value::as_u64).unwrap_or(0);

    let choices: Vec<crate::types::RouterChoice> = json
        .get("choices")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let index = c.get("index").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32;
                    let finish = c
                        .get("finish_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stop")
                        .to_string();
                    let msg = c.get("message")?;
                    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("assistant").to_string();
                    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    Some(crate::types::RouterChoice {
                        index,
                        message: RouterMessage {
                            role,
                            content: RouterMessageContent::Text(content),
                            tool_calls: None,
                            tool_call_id: None,
                        },
                        finish_reason: finish,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let usage = json.get("usage").map_or(Usage::default(), |u| Usage {
        prompt_tokens: u.get("prompt_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
        completion_tokens: u.get("completion_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
        total_tokens: u.get("total_tokens").and_then(serde_json::Value::as_u64).unwrap_or(0) as u32,
    });

    RouterResponse {
        id,
        object: "chat.completion".into(),
        created,
        model,
        choices,
        usage,
    }
}

fn fallback_completion(model_name: &str) -> RouterResponse {
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model_name.to_string(),
        choices: vec![crate::types::RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text("pipeline completed successfully".into()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage::default(),
    }
}

fn make_error_completion(model_name: &str, error: &str) -> RouterResponse {
    make_text_completion(model_name, &format!("ERROR: {error}"))
}

fn make_text_completion(model_name: &str, text: &str) -> RouterResponse {
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model_name.to_string(),
        choices: vec![crate::types::RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text(text.to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage::default(),
    }
}

fn completion_to_response(
    completion: &RouterResponse,
    model_name: &str,
    is_stream: bool,
    actual_model: Option<&str>,
) -> HyperResponse {
    let body_str = if is_stream {
        let mut handler = StreamingHandler::new(&completion.id, actual_model.unwrap_or(model_name));
        let mut s = String::new();
        if let Some(choice) = completion.choices.first() {
            s.push_str(&handler.format_choice_chunk(choice));
        }
        s.push_str(&handler.format_done());
        s
    } else {
        serde_json::to_string(&normalize::normalize_response(completion)).unwrap_or_default()
    };

    let content_type = if is_stream {
        "text/event-stream"
    } else {
        "application/json"
    };

    let len = body_str.len();
    let full = Full::new(Bytes::from(body_str));
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = hyper::StatusCode::OK;
    resp.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        content_type.parse().unwrap(),
    );
    resp.headers_mut().insert(
        hyper::header::CONTENT_LENGTH,
        len.to_string().parse().unwrap(),
    );
    add_cors_headers(resp.headers_mut());
    resp
}

fn json_response(status: hyper::StatusCode, value: &serde_json::Value) -> HyperResponse {
    let body_str = serde_json::to_string(value).unwrap_or_default();
    let len = body_str.len();
    let full = Full::new(Bytes::from(body_str));
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    resp.headers_mut().insert(
        hyper::header::CONTENT_LENGTH,
        len.to_string().parse().unwrap(),
    );
    add_cors_headers(resp.headers_mut());
    resp
}

fn error_response(status: hyper::StatusCode, message: &str) -> HyperResponse {
    let err = normalize::error_response(message, "invalid_request_error");
    json_response(status, &err)
}

fn empty_response(status: hyper::StatusCode) -> HyperResponse {
    let full = Full::new(Bytes::new());
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = status;
    add_cors_headers(resp.headers_mut());
    resp
}

fn add_cors_headers(headers: &mut hyper::HeaderMap) {
    for (name, value) in CORS_HEADERS {
        headers.insert(
            hyper::header::HeaderName::from_static(name),
            value.parse().unwrap(),
        );
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos();
    let pid = u64::from(std::process::id());
    let rng = fastrand::u64(..);
    let time_hi = (nanos >> 32) as u32;
    let time_mid = (nanos >> 16) as u32 & 0xFFFF;
    let time_lo = nanos as u32 & 0xFFFF;
    let version = (u64::from(pid as u32 & 0xFFF)) as u32 | 0x4000;
    let variant = (((pid >> 12) ^ rng) as u32 & 0x3FFF) | 0x8000;
    let node = (rng >> 32) as u32;
    format!("{time_hi:08x}-{time_mid:04x}-{time_lo:04x}-{version:04x}-{variant:04x}{node:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_is_formatted_correctly() {
        let id = uuid_v4();
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().filter(|&c| c == '-').count(), 4);
    }

    #[test]
    fn fallback_has_stop_reason() {
        let r = fallback_completion("test");
        assert_eq!(r.choices.len(), 1);
        assert_eq!(r.choices[0].finish_reason, "stop");
    }
}
