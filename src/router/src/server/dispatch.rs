use std::sync::Arc;
use std::time::Instant;

use common_core::ResponseCache;
use http_body_util::BodyExt;

use crate::dispatch::backend::ChatBackend;
use crate::dispatch::backend::OpenAiChatBackend;
use crate::dispatch::backend::RetryChatBackend;
use crate::dispatch::frontier::DispatchError;
use crate::ledger::ContentNodeLedger;
use crate::pipeline::RoutingTarget;
use crate::server::responses::completion_to_response;
use crate::server::responses::fallback_completion;
use crate::server::responses::HyperResponse;
use crate::server::responses::ServerStats;
use crate::testing::mock::MockDispatchContext;
use crate::types::{RouterMessageContent, RouterRequest, RouterResponse};
use common_core::string::strip_thinking_blocks;

use crate::charts::extract::WorkflowExtractor;

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
    extractor: Option<Arc<WorkflowExtractor>>,
) -> Result<HyperResponse, std::convert::Infallible> {
    let target_streams = is_stream && rt.stream;

    if !target_streams {
        if let Some(cache_backend) = cache {
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Some(cached) = cache_backend.get(&rt.model, &request_json) {
                stats
                    .cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(target: "router.dispatch", model = %rt.model, "cache hit");
                let Ok(mut response) =
                    serde_json::from_value::<RouterResponse>(cached.response_json)
                else {
                    stats
                        .cache_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return dispatch_real(
                        rt,
                        router_request,
                        model_name,
                        http_client,
                        target_streams,
                        cache,
                        user_text,
                        extractor,
                    )
                    .await;
                };
                if rt.filter_thinking {
                    for choice in &mut response.choices {
                        if let RouterMessageContent::Text(ref mut text) = choice.message.content {
                            *text = strip_thinking_blocks(text);
                        }
                    }
                }
                return Ok(completion_to_response(
                    &response,
                    model_name,
                    false,
                    Some(&response.model),
                ));
            }
            stats
                .cache_misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    if let Some(mock) = mock_dispatch {
        if let Some(entry) = mock.lookup(user_text) {
            mock.validate_route(entry, Some(rt));
            if mock.is_model_excepted(&rt.model) || mock.is_model_excepted(model_name) {
                tracing::info!(target: "router.server", model = %rt.model, "excepted model — real LLM call");
                return dispatch_real(
                    rt,
                    router_request,
                    model_name,
                    http_client,
                    target_streams,
                    cache,
                    user_text,
                    extractor,
                )
                .await;
            }
            tracing::info!(target: "router.server", model = %model_name, "mock canned response");
            crate::server::handler::record_ledger_result(
                ledger,
                ledger_node_id,
                true,
                Some(1.0),
                "mock response".to_string(),
            )
            .await;
            let completion = mock.dispatch_response(entry, model_name);
            return Ok(completion_to_response(
                &completion,
                model_name,
                is_stream,
                None,
            ));
        }
        tracing::debug!(target: "router.server", model = %model_name, transcript_found = false, "no transcript entry — real dispatch fallback");
    }

    tracing::info!(
        target: "router.server",
        model = %rt.model,
        url = %rt.url,
        stream = target_streams,
        retry = rt.retry_count,
        idle_timeout_ms = rt.idle_timeout_ms,
        total_timeout_ms = rt.total_timeout_ms,
        filter_thinking = rt.filter_thinking,
        fallbacks = rt.fallbacks.len(),
        "real dispatch"
    );

    dispatch_real(
        rt,
        router_request,
        model_name,
        http_client,
        target_streams,
        cache,
        user_text,
        extractor,
    )
    .await
}

/// Build a `ChatBackend` (optionally wrapped in `RetryChatBackend`) for a single
/// routing target.
fn make_backend(http_client: &reqwest::Client, target: &RoutingTarget) -> Arc<dyn ChatBackend> {
    let base: Arc<dyn ChatBackend> =
        Arc::new(OpenAiChatBackend::new(http_client.clone(), &target.url));
    if target.retry_count > 0 {
        Arc::new(RetryChatBackend::new(
            base,
            target.retry_count,
            target.retry_base_interval_s,
        ))
    } else {
        base
    }
}

/// Try dispatching to a single target.  `is_primary` controls cache write.
/// Returns `Ok(HyperResponse)` on success or `Err(DispatchError)` on failure.
async fn dispatch_to_single_target(
    target: &RoutingTarget,
    router_request: &RouterRequest,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
    is_primary: bool,
    user_text: &str,
    extractor: Option<Arc<WorkflowExtractor>>,
) -> Result<HyperResponse, DispatchError> {
    let backend = make_backend(http_client, target);

    if stream {
        let result = backend
            .stream_complete(
                router_request.clone(),
                target.model.clone(),
                target.params.clone(),
                target.idle_timeout_ms,
                target.total_timeout_ms,
                target.filter_thinking,
            )
            .await?;
        let mut resp = hyper::Response::new(result.body.boxed_unsync());
        *resp.status_mut() = hyper::StatusCode::OK;
        resp.headers_mut().insert(
            hyper::header::CONTENT_TYPE,
            hyper::header::HeaderValue::from_static("text/event-stream"),
        );
        crate::server::responses::add_cors_headers(resp.headers_mut());
        return Ok(resp);
    }

    let filter_thinking = target.filter_thinking;
    let mut completion = backend
        .complete(
            router_request.clone(),
            target.model.clone(),
            target.params.clone(),
            target.idle_timeout_ms,
            target.total_timeout_ms,
        )
        .await?;

    if filter_thinking {
        for choice in &mut completion.choices {
            if let RouterMessageContent::Text(ref mut text) = choice.message.content {
                *text = strip_thinking_blocks(text);
            }
        }
    }

    // Cache only the primary (first) target's response
    if is_primary {
        if let Some(cache_backend) = cache {
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Ok(response_json) = serde_json::to_value(&completion) {
                cache_backend.set(&target.model, &request_json, response_json);
            }
        }
    }

    // M10 learning loop: a successful buffered dispatch is a solved solution —
    // distill it into a draft chart (best-effort, never fails the request).
    if let Some(extractor) = extractor {
        let answer = completion
            .choices
            .first()
            .map(|c| c.message.content.to_string_lossy())
            .unwrap_or_default();
        extractor.record_success(user_text, &target.model, &answer);
    }

    Ok(completion_to_response(
        &completion,
        "",
        false,
        Some(&target.model),
    ))
}

pub async fn dispatch_real(
    rt: &RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
    user_text: &str,
    extractor: Option<Arc<WorkflowExtractor>>,
) -> Result<HyperResponse, std::convert::Infallible> {
    let all_targets = std::iter::once(rt)
        .chain(rt.fallbacks.iter())
        .collect::<Vec<_>>();

    let mut last_error: Option<DispatchError> = None;

    for (i, target) in all_targets.iter().enumerate() {
        tracing::info!(
            target: "router.server",
            attempt = i + 1,
            total = all_targets.len(),
            model = %target.model,
            url = %target.url,
            stream = stream,
            retry_count = target.retry_count,
            idle_timeout_ms = target.idle_timeout_ms,
            total_timeout_ms = target.total_timeout_ms,
            "dispatch attempt"
        );

        let attempt_start = Instant::now();
        match dispatch_to_single_target(
            target,
            router_request,
            http_client,
            stream,
            cache,
            i == 0,
            user_text,
            extractor.clone(),
        )
        .await
        {
            Ok(resp) => {
                return Ok(resp);
            }
            Err(e) => {
                let attempt_latency_ms = attempt_start.elapsed().as_millis() as u64;
                let is_retryable = e.is_retryable();
                tracing::warn!(
                    target: "router.server",
                    attempt = i + 1,
                    total = all_targets.len(),
                    model = %target.model,
                    error = %e,
                    retryable = is_retryable,
                    attempt_latency_ms = attempt_latency_ms,
                    remaining = all_targets.len() - i - 1,
                    "dispatch attempt failed"
                );
                last_error = Some(e);
                // Non-retryable errors (e.g. 400 Bad Request) short-circuit
                if !is_retryable {
                    break;
                }
            }
        }
    }

    tracing::warn!(
        target: "router.server",
        error = ?last_error,
        "all dispatch targets failed, returning fallback response"
    );
    let completion = fallback_completion(model_name);
    Ok(completion_to_response(
        &completion,
        model_name,
        stream,
        None,
    ))
}
