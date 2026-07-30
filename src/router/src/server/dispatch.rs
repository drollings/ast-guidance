use std::sync::Arc;

use common_core::ResponseCache;
use http_body_util::BodyExt;

use crate::dispatch::backend::ChatBackend;
use crate::dispatch::backend::OpenAiChatBackend;
use crate::dispatch::backend::RetryChatBackend;
use crate::ledger::ContentNodeLedger;
use crate::pipeline::RoutingTarget;
use crate::server::responses::completion_to_response;
use crate::server::responses::fallback_completion;
use crate::server::responses::ServerStats;
use crate::server::responses::HyperResponse;
use common_core::string::strip_thinking_blocks;
use crate::testing::mock::MockDispatchContext;
use crate::types::{RouterMessageContent, RouterRequest, RouterResponse};

pub async fn handle_dispatch(
    rt: &RoutingTarget,
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
) -> Result<HyperResponse, std::convert::Infallible> {
    let target_streams = is_stream && rt.stream;

    if !target_streams {
        if let Some(cache_backend) = cache {
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Some(cached) = cache_backend.get(&rt.model, &request_json) {
                stats.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(target: "router.dispatch", model = %rt.model, "cache hit");
                let Ok(mut response) = serde_json::from_value::<RouterResponse>(cached.response_json)
                else {
                    stats.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return dispatch_real(rt, router_request, model_name, http_client, target_streams, cache).await;
                };
                if rt.filter_thinking {
                    for choice in &mut response.choices {
                        if let RouterMessageContent::Text(ref mut text) = choice.message.content {
                            *text = strip_thinking_blocks(text);
                        }
                    }
                }
                return Ok(completion_to_response(&response, model_name, false, Some(&response.model)));
            }
            stats.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

pub async fn dispatch_real(
    rt: &RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
) -> Result<HyperResponse, std::convert::Infallible> {
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
                let mut resp = hyper::Response::new(body.body.boxed_unsync());
                *resp.status_mut() = hyper::StatusCode::OK;
                resp.headers_mut()
                    .insert(hyper::header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
                crate::server::responses::add_cors_headers(resp.headers_mut());
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
                        if let RouterMessageContent::Text(ref mut text) = choice.message.content {
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
