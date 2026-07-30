use std::sync::atomic::Ordering;
use std::sync::Arc;

use common_core::hash::uuid_v4;
use common_core::ResponseCache;
use http_body_util::BodyExt;

use crate::config::{ModelEntry, RouteRef};
use crate::ledger::ContentNodeLedger;
use crate::normalize;
use crate::pipeline::PipelineOrchestrator;
use crate::pipeline::RoutingTarget;
use crate::server::dispatch::handle_dispatch;
use crate::server::responses::completion_to_response;
use crate::server::responses::empty_response;
use crate::server::responses::error_response;
use crate::server::responses::make_error_completion;
use crate::server::responses::make_text_completion;
use crate::server::responses::ServerStats;
use crate::server::responses::HyperResponse;
use crate::testing::mock::MockDispatchContext;

async fn handle_chat_completion(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<std::collections::HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<std::collections::HashMap<String, RouteRef>>,
    models: Arc<std::collections::HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    http_client: Arc<reqwest::Client>,
) -> Result<HyperResponse, std::convert::Infallible> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("body read error: {e}"),
            ));
        }
    };

    if body_bytes.len() > max_payload {
        return Ok(empty_response(hyper::StatusCode::PAYLOAD_TOO_LARGE));
    }

    if body_bytes.is_empty() {
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(error_response(hyper::StatusCode::BAD_REQUEST, "empty body"));
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
            return Ok(error_response(hyper::StatusCode::BAD_REQUEST, &e.to_string()));
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

    let session_id = router_request.session_id.clone().unwrap_or_else(uuid_v4);
    let request_id = uuid_v4();
    let request_text = router_request
        .messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.to_string_lossy())
        .unwrap_or_default();
    let ledger_node_id = ledger.as_ref().and_then(|l| {
        l.record_request(&session_id, &request_id, &request_text)
            .ok()
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

        if let Some(node_id) = ledger_node_id {
            if let Some(ref l) = ledger {
                let _ = l.record_result(node_id, false, Some(0.0), reason);
            }
        }

        let completion = make_error_completion(&model_name, reason);
        return Ok(completion_to_response(&completion, &model_name, is_stream, None));
    }

    if let Some(ref resp_str) = pipeline_result.classifier_response {
        tracing::info!(
            target: "router.server",
            model = %model_name,
            response_len = resp_str.len(),
            "responding with classifier direct response"
        );
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

    if let Some(ref url) = classifier_url {
        tracing::info!(
            target: "router.server",
            model = %model_name,
            fallback_url = %url,
            "no routing target — dispatching to classifier fallback"
        );
        let rt_for_fallback = RoutingTarget {
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
            fallbacks: vec![],
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
    if let Some(node_id) = ledger_node_id {
        if let Some(ref l) = ledger {
            let _ = l.record_result(node_id, true, Some(0.5), "fallback response");
        }
    }
    let completion = crate::server::responses::fallback_completion(&model_name);
    Ok(completion_to_response(&completion, &model_name, is_stream, None))
}

#[allow(clippy::implicit_hasher)]
pub async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<std::collections::HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<std::collections::HashMap<String, RouteRef>>,
    models: Arc<std::collections::HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier_url: Option<String>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    http_client: Arc<reqwest::Client>,
) -> Result<HyperResponse, std::convert::Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    if method == hyper::Method::OPTIONS {
        return Ok(empty_response(hyper::StatusCode::NO_CONTENT));
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            Ok(crate::server::responses::json_response(
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
            Ok(crate::server::responses::json_response(hyper::StatusCode::OK, &body))
        }
        ("POST", "/admin/cache/invalidate") => {
            if !is_local_request(&req) {
                return Ok(crate::server::responses::forbidden_response());
            }
            if let Some(ref cache) = cache {
                cache.invalidate_all();
                stats.requests.fetch_add(1, Ordering::Relaxed);
                Ok(crate::server::responses::json_response(
                    hyper::StatusCode::OK,
                    &serde_json::json!({"status": "ok"}),
                ))
            } else {
                Ok(crate::server::responses::json_response(
                    hyper::StatusCode::OK,
                    &serde_json::json!({"status": "no_cache"}),
                ))
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
            if method == hyper::Method::DELETE && path.starts_with("/admin/cache/") {
                if !is_local_request(&req) {
                    return Ok(crate::server::responses::forbidden_response());
                }
                let key = &path["/admin/cache/".len()..];
                if key.is_empty() {
                    return Ok(crate::server::responses::error_response(
                        hyper::StatusCode::BAD_REQUEST, "missing cache key",
                    ));
                }
                if let Some(ref cache_backend) = cache {
                    cache_backend.invalidate_key_raw(key);
                    stats.requests.fetch_add(1, Ordering::Relaxed);
                    Ok(crate::server::responses::json_response(
                        hyper::StatusCode::OK,
                        &serde_json::json!({"status": "deleted"}),
                    ))
                } else {
                    Ok(crate::server::responses::json_response(
                        hyper::StatusCode::OK,
                        &serde_json::json!({"status": "no_cache"}),
                    ))
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

fn resolve_pipeline(
    model_name: &str,
    routes: &std::collections::HashMap<String, RouteRef>,
    models: &std::collections::HashMap<String, ModelEntry>,
    pipelines: &std::collections::HashMap<String, Arc<PipelineOrchestrator>>,
    request_json: &str,
) -> crate::pipeline::PipelineResult {
    use fluent_wvr::prelude::*;
    use crate::pipeline::RoutingTarget;

    let route = routes.get(model_name).cloned();

    let pipeline_names: Vec<String> = if let Some(ref r) = route {
        r.pipelines.clone()
    } else if let Some(model_entry) = models.get(model_name) {
        let rt = RoutingTarget {
            url: model_entry.endpoint.clone(),
            model: model_entry.name.clone().unwrap_or_else(|| model_name.to_string()),
            group: None,
            target_name: Some(model_name.to_string()),
            params: model_entry.params.clone(),
            filter_thinking: model_entry.filter_thinking,
            retry_count: model_entry.retry_count,
            retry_base_interval_s: model_entry.retry_base_interval_s,
            stream: model_entry.stream,
            idle_timeout_ms: model_entry.idle_timeout_ms,
            total_timeout_ms: model_entry.total_timeout_ms,
            fallbacks: vec![],
        };
        return crate::pipeline::PipelineResult {
            decisions: vec![], final_response: None, rejected: false,
            reject_reason: None, routing_target: Some(rt), classifier_response: None,
        };
    } else {
        routes.get("local")
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
                    decisions: all_decisions, final_response: None, rejected: true,
                    reject_reason: Some(format!("pipeline '{name}' error: {e}")),
                    routing_target: None, classifier_response: None,
                };
            }
        };

        let mut result: crate::pipeline::PipelineResult = match output.data_take() {
            Ok(r) => r,
            Err(e) => {
                return crate::pipeline::PipelineResult {
                    decisions: all_decisions, final_response: None, rejected: true,
                    reject_reason: Some(format!("pipeline '{name}' output decode: {e}")),
                    routing_target: None, classifier_response: None,
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
        decisions: vec![], final_response: None, rejected: false,
        reject_reason: None, routing_target: None, classifier_response: None,
    });
    final_result.decisions = all_decisions;
    final_result
}
