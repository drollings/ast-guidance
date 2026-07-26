//! HTTP server exposing the router pipeline as an OpenAI-compatible endpoint.
//! Uses hyper for HTTP with SSE streaming support via http-body-util::channel.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use common_core::hash::uuid_v4;
use common_core::now_secs;
use common_core::ResponseCache;
use fluent_wvr::prelude::*;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use crate::config::{ModelEntry, RouteRef, ServerConfig};
use crate::ledger::ContentNodeLedger;
use crate::normalize;
use crate::dispatch::backend::ChatBackend;
use crate::streaming::strip_thinking_blocks;
use crate::streaming::StreamingHandler;
use crate::pipeline::PipelineOrchestrator;
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
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

struct ServerStats {
    requests: AtomicU64,
    errors: AtomicU64,
    rejections: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
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
            ledger: None,
            cache: None,
            depends: vec![],
            provides: vec![ArcIntern::from("http.endpoint")],
        }
    }

    #[must_use]
    pub fn with_ledger(mut self, ledger: Arc<ContentNodeLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    #[must_use]
    pub fn with_cache(mut self, cache: Arc<ResponseCache>) -> Self {
        self.cache = Some(cache);
        self
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
            has_ledger = self.ledger.is_some(),
            has_cache = self.cache.is_some(),
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
            self.ledger.clone(),
            self.cache.clone(),
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
        let ledger = self.ledger.clone();
        let cache = self.cache.clone();

        let rt = ctx.rt.clone();

        let _handle = rt.spawn(Box::pin(async move {
            if let Err(e) =
                run_http(pipelines, routes, models, &bind_addr, max_payload, classifier_url, mock_dispatch, ledger, cache)
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
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| format!("bind {bind_addr} failed: {e}"))?;

    tracing::info!(target: "router.server", addr = %bind_addr, "HTTP server listening (hyper)");

    let stats = Arc::new(ServerStats {
        requests: AtomicU64::new(0),
        errors: AtomicU64::new(0),
        rejections: AtomicU64::new(0),
        cache_hits: AtomicU64::new(0),
        cache_misses: AtomicU64::new(0),
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
        let ledger = ledger.clone();
        let cache = cache.clone();
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
                    ledger.clone(),
                    cache.clone(),
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
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
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
                &serde_json::json!({
                    "status": "ok",
                    "cache_hits": stats.cache_hits.load(Ordering::Relaxed),
                    "cache_misses": stats.cache_misses.load(Ordering::Relaxed),
                }),
            ))
        }
        ("GET", "/stats") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            let body = serde_json::json!({
                "requests": stats.requests.load(Ordering::Relaxed),
                "errors": stats.errors.load(Ordering::Relaxed),
                "rejections": stats.rejections.load(Ordering::Relaxed),
                "cache_hits": stats.cache_hits.load(Ordering::Relaxed),
                "cache_misses": stats.cache_misses.load(Ordering::Relaxed),
            });
            Ok(json_response(hyper::StatusCode::OK, &body))
        }
        ("POST", "/admin/cache/invalidate") => {
            if !is_local_request(&req) {
                return Ok(forbidden_response());
            }
            if let Some(ref cache) = cache {
                cache.invalidate_all();
                stats.requests.fetch_add(1, Ordering::Relaxed);
                Ok(json_response(hyper::StatusCode::OK, &serde_json::json!({"status": "ok"})))
            } else {
                Ok(json_response(hyper::StatusCode::OK, &serde_json::json!({"status": "no_cache"})))
            }
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completion(
                req, pipelines, routes, models, stats, max_payload,
                classifier_url, mock_dispatch, ledger, cache, http_client,
            )
            .await
        }
        _ => {
            // Check for DELETE /admin/cache/{key}
            if method == "DELETE" && path.starts_with("/admin/cache/") {
                if !is_local_request(&req) {
                    return Ok(forbidden_response());
                }
                let key = &path["/admin/cache/".len()..];
                if key.is_empty() {
                    return Ok(error_response(hyper::StatusCode::BAD_REQUEST, "missing cache key"));
                }
                if let Some(ref cache_backend) = cache {
                    cache_backend.invalidate_key_raw(key);
                    stats.requests.fetch_add(1, Ordering::Relaxed);
                    Ok(json_response(hyper::StatusCode::OK, &serde_json::json!({"status": "deleted"})))
                } else {
                    Ok(json_response(hyper::StatusCode::OK, &serde_json::json!({"status": "no_cache"})))
                }
            } else {
                let code = if path == "/v1/chat/completions" {
                    hyper::StatusCode::METHOD_NOT_ALLOWED
                } else {
                    hyper::StatusCode::NOT_FOUND
                };
                Ok(empty_response(code))
            }
        }
    }
}

fn is_local_request(req: &hyper::Request<hyper::body::Incoming>) -> bool {
    req.headers()
        .get(hyper::header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host.eq_ignore_ascii_case("127.0.0.1")
                || host.eq_ignore_ascii_case("::1")
                || host.starts_with("localhost:")
                || host.starts_with("127.0.0.1:")
                || host.starts_with("[::1]:")
        })
}

fn forbidden_response() -> HyperResponse {
    error_response(hyper::StatusCode::FORBIDDEN, "admin endpoints are localhost-only")
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
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
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

    // ── Record request in ledger (LOD0) before any pipeline processing ──
    let session_id = router_request.session_id.clone().unwrap_or_else(uuid_v4);
    let request_id = uuid_v4();
    let request_text = router_request
        .messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.to_string_lossy())
        .unwrap_or_default();
    let ledger_node_id = ledger.as_ref().and_then(|l| {
        l.record_request(&session_id, &request_id, &request_text).ok()
    });

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

        // Record rejection in ledger
        if let Some(node_id) = ledger_node_id {
            if let Some(ref l) = ledger {
                let _ = l.record_result(node_id, false, Some(0.0), reason);
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
        // Record successful response in ledger
        if let Some(node_id) = ledger_node_id {
            if let Some(ref l) = ledger {
                let _ = l.record_result(node_id, true, Some(1.0), resp_str);
            }
        }
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
            ledger_node_id,
            ledger.as_ref(),
            cache.as_ref(),
            stats.as_ref(),
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
            ledger_node_id,
            ledger.as_ref(),
            cache.as_ref(),
            stats.as_ref(),
        )
        .await;
    }

    tracing::warn!(
        target: "router.server",
        model = %model_name,
        "no routing target, no classifier response, no classifier url — returning fallback"
    );
    // Record fallback in ledger
    if let Some(node_id) = ledger_node_id {
        if let Some(ref l) = ledger {
            let _ = l.record_result(node_id, true, Some(0.5), "fallback response");
        }
    }
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
    ledger_node_id: Option<fluent_types::NodeId>,
    ledger: Option<&Arc<ContentNodeLedger>>,
    cache: Option<&Arc<ResponseCache>>,
    stats: &ServerStats,
) -> Result<HyperResponse, Infallible> {
    let target_streams = is_stream && rt.stream;

    // ── Response cache check (buffered only) ─────────────────────
    if !target_streams {
        if let Some(cache_backend) = cache {
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Some(cached) = cache_backend.get(&rt.model, &request_json) {
                stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(target: "router.dispatch", model = %rt.model, "cache hit");
                let Ok(mut response) = serde_json::from_value::<RouterResponse>(cached.response_json) else {
                    stats.cache_misses.fetch_add(1, Ordering::Relaxed);
                    return dispatch_real(rt, router_request, model_name, http_client, target_streams, cache).await;
                };
                if rt.filter_thinking {
                    for choice in &mut response.choices {
                        if let crate::types::RouterMessageContent::Text(ref mut text) = choice.message.content {
                            *text = strip_thinking_blocks(text);
                        }
                    }
                }
                return Ok(completion_to_response(&response, model_name, false, Some(&response.model)));
            }
            stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    if let Some(mock) = mock_dispatch {
        if let Some(entry) = mock.lookup(user_text) {
            mock.validate_route(entry, Some(rt));
            if mock.is_model_excepted(&rt.model) || mock.is_model_excepted(model_name) {
                tracing::info!(target: "router.server", model = %rt.model, "excepted model — real LLM call");
                return dispatch_real(rt, router_request, model_name, http_client, target_streams, cache).await;
            }
            tracing::info!(target: "router.server", model = %model_name, "mock canned response");
            if let Some(node_id) = ledger_node_id {
                if let Some(l) = ledger {
                    let _ = l.record_result(node_id, true, Some(1.0), "mock response");
                }
            }
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

    dispatch_real(rt, router_request, model_name, http_client, target_streams, cache).await
}

async fn dispatch_real(
    rt: &crate::pipeline::RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
) -> Result<HyperResponse, Infallible> {
    use crate::dispatch::backend::{OpenAiChatBackend, RetryChatBackend};

    let base: Arc<dyn ChatBackend> =
        Arc::new(OpenAiChatBackend::new(http_client.clone(), &rt.url));

    let backend: Arc<dyn ChatBackend> = if rt.retry_count > 0 {
        Arc::new(RetryChatBackend::new(base, rt.retry_count, rt.retry_base_interval_s))
    } else {
        base
    };

    if stream {
        match backend
            .stream_complete(
                router_request.clone(),
                rt.model.clone(),
                rt.params.clone(),
                rt.idle_timeout_ms,
                rt.total_timeout_ms,
                rt.filter_thinking,
            )
            .await
        {
            Ok(body) => {
                let mut resp = HyperResponse::new(body.body.boxed_unsync());
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
        let filter_thinking = rt.filter_thinking;
        let completion = match backend
            .complete(
                router_request.clone(),
                rt.model.clone(),
                rt.params.clone(),
            )
            .await
        {
            Ok(mut c) => {
                if let Some(cache) = cache {
                    let request_json = serde_json::to_string(router_request).unwrap_or_default();
                    if let Ok(response_json) = serde_json::to_value(&c) {
                        cache.set(&rt.model, &request_json, response_json);
                    }
                }
                if filter_thinking {
                    for choice in &mut c.choices {
                        if let crate::types::RouterMessageContent::Text(ref mut text) = choice.message.content {
                            *text = strip_thinking_blocks(text);
                        }
                    }
                }
                c
            }
            Err(e) => {
                tracing::warn!(target: "router.server", error = %e, "dispatch failed, using fallback");
                fallback_completion(model_name)
            }
        };
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

fn fallback_completion(model_name: &str) -> RouterResponse {
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: now_secs(),
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
        created: now_secs(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use common_core::hash::uuid_v4;

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
