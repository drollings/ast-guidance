//! HTTP server exposing the router pipeline as an OpenAI-compatible endpoint.
//! Follows the job-copilot pattern: raw `tokio::net::TcpListener` with
//! hand-written HTTP/1.1 parsing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use fluent_wvr::prelude::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::config::{RouteRef, ServerConfig};
use crate::normalize;
use crate::pipeline::PipelineOrchestrator;
use crate::testing::mock::MockDispatchContext;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse, Usage};

const CORS_HEADERS: &str = "Access-Control-Allow-Origin: *\r\n\
     Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n\
     Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

/// Router HTTP server. Implements `WorkUnit` so it can be registered in a `Zone`.
pub struct RouterServer {
    name: ArcIntern<str>,
    pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
    routes: HashMap<String, RouteRef>,
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
        config: &ServerConfig,
        classifier_url: Option<String>,
    ) -> Self {
        Self {
            name: ArcIntern::from("router.server"),
            pipelines,
            routes,
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
        self.mock_dispatch = Some(Arc::new(mock_dispatch));
        self
    }

    /// Start the HTTP server and block the current task.
    pub async fn serve(&self) -> Result<(), String> {
        run_http(
            Arc::new(self.pipelines.clone()),
            Arc::new(self.routes.clone()),
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
        let bind_addr = self.bind_addr.clone();
        let max_payload = self.max_payload;
        let classifier_url = self.classifier_url.clone();
        let mock_dispatch = self.mock_dispatch.clone();

        let rt = ctx.rt.clone();

        let _handle = rt.spawn(Box::pin(async move {
            if let Err(e) =
                run_http(pipelines, routes, &bind_addr, max_payload, classifier_url, mock_dispatch)
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
    bind_addr: &str,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| format!("bind {bind_addr} failed: {e}"))?;

    tracing::info!(target: "router.server", addr = %bind_addr, "HTTP server listening");

    let stats = Arc::new(ServerStats {
        requests: AtomicU64::new(0),
        errors: AtomicU64::new(0),
        rejections: AtomicU64::new(0),
    });

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
        let stats = stats.clone();
        let classifier_url = classifier_url.clone();
        let mock_dispatch = mock_dispatch.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(
                stream,
                pipelines,
                routes,
                stats,
                max_payload,
                classifier_url,
                mock_dispatch,
            )
            .await
            {
                tracing::error!(target: "router.server", error = %e, "connection error");
            }
        });
    }
}

fn execute_pipeline_sequence(
    pipelines: &HashMap<String, Arc<PipelineOrchestrator>>,
    pipeline_names: &[String],
    ctx: &mut WorkContext,
) -> Result<crate::pipeline::PipelineResult, String> {
    let mut all_decisions = Vec::new();
    let mut last_result: Option<crate::pipeline::PipelineResult> = None;

    for name in pipeline_names {
        let Some(pipeline) = pipelines.get(name) else {
            tracing::warn!(target: "router.server", pipeline = %name, "pipeline not found, skipping");
            continue;
        };

        let output = pipeline
            .execute(ctx)
            .map_err(|e| format!("pipeline '{name}' error: {e}"))?;
        let mut result: crate::pipeline::PipelineResult = output
            .data_take()
            .map_err(|e| format!("pipeline '{name}' output decode: {e}"))?;

        if result.rejected {
            return Ok(crate::pipeline::PipelineResult {
                decisions: result.decisions,
                final_response: None,
                rejected: true,
                reject_reason: result.reject_reason,
                routing_target: None,
                classifier_response: None,
            });
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
    Ok(final_result)
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<HashMap<String, RouteRef>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
) -> Result<(), String> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 1024];

    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|e| format!("read error: {e}"))?;
        if n == 0 {
            return Err("connection closed before headers".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 16_384 {
            return Err("headers too large".into());
        }
    }

    let header_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let headers = parse_headers(&header_str);

    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = parts.first().copied().unwrap_or("");
    let path = parts.get(1).copied().unwrap_or("");

    // CORS preflight
    if method == "OPTIONS" {
        let resp =
            format!("HTTP/1.1 204 No Content\r\n{CORS_HEADERS}Connection: close\r\n\r\n");
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| format!("write: {e}"))?;
        return Ok(());
    }

    match (method, path) {
        ("GET", "/health") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{CORS_HEADERS}Connection: close\r\n\r\n{{\"status\":\"ok\"}}"
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            return Ok(());
        }
        ("GET", "/stats") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            let body = serde_json::json!({
                "requests": stats.requests.load(Ordering::Relaxed),
                "errors": stats.errors.load(Ordering::Relaxed),
                "rejections": stats.rejections.load(Ordering::Relaxed),
            });
            let body_str = serde_json::to_string(&body).unwrap_or_default();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{}",
                body_str.len(), body_str
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            return Ok(());
        }
        ("POST", "/v1/chat/completions") => {}
        _ => {
            let code = if path == "/v1/chat/completions" {
                "405 Method Not Allowed"
            } else {
                "404 Not Found"
            };
            let resp = format!("HTTP/1.1 {code}\r\nConnection: close\r\n\r\n");
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            return Ok(());
        }
    }

    // Read body
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    if content_length > max_payload {
        let resp = "HTTP/1.1 413 Payload Too Large\r\nConnection: close\r\n\r\n";
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| format!("write: {e}"))?;
        return Ok(());
    }

    if content_length == 0 {
        let body = serde_json::to_string(&normalize::error_response(
            "empty body",
            "invalid_request_error",
        ))
        .unwrap_or_default();
        let resp = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| format!("write: {e}"))?;
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    let already_read = buf.len() - header_end;
    let remaining = content_length.saturating_sub(already_read);
    if remaining > 0 {
        buf.resize(header_end + remaining, 0);
        stream
            .read_exact(&mut buf[header_end..])
            .await
            .map_err(|e| format!("body read: {e}"))?;
    }

    let body_bytes = &buf[header_end..header_end + content_length];
    let body_str = std::str::from_utf8(body_bytes).unwrap_or("");
    stats.requests.fetch_add(1, Ordering::Relaxed);

    // Parse request body
    let body_json: serde_json::Value = match serde_json::from_str(body_str) {
        Ok(v) => v,
        Err(e) => {
            let err =
                normalize::error_response(&format!("invalid JSON: {e}"), "invalid_request_error");
            let err_str = serde_json::to_string(&err).unwrap_or_default();
            let resp = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{err_str}",
                err_str.len()
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
    };

    // Normalize to RouterRequest
    let router_request = match normalize::normalize_request(&body_json) {
        Ok(r) => r,
        Err(e) => {
            let err = normalize::error_response(&e.to_string(), "invalid_request_error");
            let err_str = serde_json::to_string(&err).unwrap_or_default();
            let resp = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{err_str}",
                err_str.len()
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
    };

    let is_stream = router_request.stream.unwrap_or(false);
    let model_name = router_request.model.clone();
    let request_json = serde_json::to_string(&router_request).unwrap_or_default();

    // Determine which pipelines to run for this route
    let route = routes
        .get(&model_name)
        .or_else(|| routes.get("local"))
        .cloned();
    let pipeline_names = route
        .as_ref()
        .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone());

    // Run pipeline sequence
    let mut ctx = WorkContext::default();
    ctx.metadata
        .insert("request".into(), MetadataValue::String(request_json));

    let pipeline_result = match execute_pipeline_sequence(&pipelines, &pipeline_names, &mut ctx) {
        Ok(result) => result,
        Err(e) => {
            let err =
                normalize::error_response(&format!("pipeline error: {e}"), "server_error");
            let err_str = serde_json::to_string(&err).unwrap_or_default();
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{err_str}",
                err_str.len()
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| format!("write: {e}"))?;
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
    };

    // Extract user message for mock lookup
    let user_message = router_request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.to_string_lossy())
        .unwrap_or_default();

    if pipeline_result.rejected {
        let reason = pipeline_result
            .reject_reason
            .as_deref()
            .unwrap_or("request rejected");
        stats.rejections.fetch_add(1, Ordering::Relaxed);

        if let Some(ref mock) = mock_dispatch {
            if let Some(entry) = mock.lookup(&user_message) {
                mock.validate_rejection(entry, reason);
            }
        }

        let error_output = format!("ERROR: {reason}");
        let completion = RouterResponse {
            id: String::new(),
            object: "chat.completion".into(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: model_name.clone(),
            choices: vec![crate::types::RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text(error_output),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage::default(),
        };
        write_completion_response(&mut stream, &completion, is_stream).await?;
        return Ok(());
    }

    // Build response
    if let Some(ref resp_str) = pipeline_result.classifier_response {
        let completion = RouterResponse {
            id: String::new(),
            object: "chat.completion".into(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: model_name.clone(),
            choices: vec![crate::types::RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text(resp_str.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage::default(),
        };
        write_completion_response(&mut stream, &completion, is_stream).await?;
    } else if let Some(ref rt) = pipeline_result.routing_target {
        // Mock mode: validate routing and return canned response
        if let Some(ref mock) = mock_dispatch {
            if let Some(entry) = mock.lookup(&user_message) {
                mock.validate_route(entry, Some(rt));
                let completion = mock.dispatch_response(entry, &model_name);
                write_completion_response(&mut stream, &completion, is_stream).await?;
            } else {
                // No transcript entry for this message — use real dispatch as fallback
                let url = build_dispatch_url(&rt.url);
                let mut completion = dispatch_to_llm(
                    &url,
                    &router_request,
                    &rt.model,
                    rt.params.as_ref(),
                    rt.filter_thinking,
                    rt.retry_count,
                    rt.retry_base_interval_s,
                )
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(target: "router.server", error = %e, "dispatch failed, using fallback");
                    fallback_completion(&model_name)
                });
                completion.model = model_name.clone();
                write_completion_response(&mut stream, &completion, is_stream).await?;
            }
        } else {
            // Real dispatch
            let url = build_dispatch_url(&rt.url);
            let mut completion = dispatch_to_llm(
                &url,
                &router_request,
                &rt.model,
                rt.params.as_ref(),
                rt.filter_thinking,
                rt.retry_count,
                rt.retry_base_interval_s,
            )
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(target: "router.server", error = %e, "dispatch failed, using fallback");
                fallback_completion(&model_name)
            });
            completion.model = model_name.clone();
            write_completion_response(&mut stream, &completion, is_stream).await?;
        }
    } else if let Some(ref url) = classifier_url {
        let mut completion = dispatch_to_llm(
            url, &router_request, &model_name, None, false, 0, 1,
        )
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(target: "router.server", error = %e, "classifier dispatch failed, using fallback");
            fallback_completion(&model_name)
        });
        completion.model = model_name.clone();
        write_completion_response(&mut stream, &completion, is_stream).await?;
    } else {
        let completion = fallback_completion(&model_name);
        write_completion_response(&mut stream, &completion, is_stream).await?;
    }

    Ok(())
}

fn build_dispatch_url(endpoint_url: &str) -> String {
    if endpoint_url.ends_with("/chat/completions") {
        endpoint_url.to_string()
    } else {
        format!(
            "{}/chat/completions",
            endpoint_url.trim_end_matches('/')
        )
    }
}

async fn write_completion_response(
    stream: &mut tokio::net::TcpStream,
    completion: &RouterResponse,
    is_stream: bool,
) -> Result<(), String> {
    let body = if is_stream {
        let mut handler =
            crate::streaming::StreamingHandler::new(&completion.id, &completion.model);
        let mut resp_body = String::new();
        if let Some(choice) = completion.choices.first() {
            resp_body.push_str(&handler.format_choice_chunk(choice));
        }
        resp_body.push_str(&handler.format_done());
        resp_body
    } else {
        serde_json::to_string(&normalize::normalize_response(completion)).unwrap_or_default()
    };
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{body}",
        if is_stream { "text/event-stream" } else { "application/json" },
        body.len()
    );
    stream
        .write_all(resp.as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))
}

fn parse_headers(header_str: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in header_str.lines().skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }
    map
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

async fn dispatch_to_llm(
    endpoint_url: &str,
    request: &RouterRequest,
    model_name: &str,
    params: Option<&serde_json::Value>,
    _filter_thinking: bool,
    retry_count: u32,
    retry_base_interval_s: u64,
) -> Result<RouterResponse, String> {
    let client = reqwest::Client::new();
    let url = build_dispatch_url(endpoint_url);

    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                RouterMessageContent::Text(s) => serde_json::Value::String(s.clone()),
                RouterMessageContent::Parts(parts) => serde_json::Value::Array(
                    parts
                        .iter()
                        .map(|p| serde_json::to_value(p).unwrap())
                        .collect(),
                ),
            };
            let mut msg = serde_json::json!({"role": m.role, "content": content});
            if let Some(ref tc) = m.tool_calls {
                msg["tool_calls"] = serde_json::to_value(tc).unwrap();
            }
            if let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            msg
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
    });

    if let Some(p) = params {
        if let Some(obj) = p.as_object() {
            for (k, v) in obj {
                body[k] = v.clone();
            }
        }
    }

    if !body
        .as_object()
        .is_some_and(|o| o.contains_key("temperature"))
    {
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::Value::Number(
                serde_json::Number::from_f64(temp).ok_or("invalid temperature")?,
            );
        }
    }
    if !body
        .as_object()
        .is_some_and(|o| o.contains_key("max_tokens"))
    {
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(max_tokens.into());
        }
    }

    let max_attempts = (retry_count + 1).max(1);
    let mut last_err = String::new();

    for attempt in 0..max_attempts {
        let result = client.post(&url).json(&body).send().await;

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
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        }
    }

    Err(format!(
        "dispatch failed after {max_attempts} attempts: {last_err}"
    ))
}

fn parse_openai_response(json: &serde_json::Value, fallback_model: &str) -> RouterResponse {
    let id = json
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let created = json
        .get("created")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    let choices: Vec<crate::types::RouterChoice> = json
        .get("choices")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let index = c
                        .get("index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as u32;
                    let finish = c
                        .get("finish_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stop")
                        .to_string();
                    let msg = c.get("message")?;
                    let role = msg
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("assistant")
                        .to_string();
                    let content = msg
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
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

    let usage = json
        .get("usage")
        .map_or(Usage::default(), |u| Usage {
            prompt_tokens: u
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("completion_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("total_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_headers_extracts_content_length() {
        let raw =
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nContent-Length: 42\r\n\r\n";
        let h = parse_headers(raw);
        assert_eq!(h.get("content-length").unwrap(), "42");
    }

    #[test]
    fn parse_headers_lowercases_keys() {
        let raw =
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n";
        let h = parse_headers(raw);
        assert!(h.contains_key("content-type"));
    }

    #[test]
    fn parse_headers_empty() {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\n\r\n";
        let h = parse_headers(raw);
        assert!(h.is_empty());
    }
}
