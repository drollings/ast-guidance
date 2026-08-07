use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use common_core::ResponseCache;
use http_body_util::BodyExt;

use crate::dag_session::DependencySession;
use crate::dispatch::backend::ChatBackend;
use crate::dispatch::backend::OpenAiChatBackend;
use crate::dispatch::backend::RetryChatBackend;
use crate::dispatch::escalation::{EscalationContext, EscalationLadder};
use crate::dispatch::frontier::DispatchError;
use crate::pipeline::RoutingTarget;
use crate::server::responses::answer_text;
use crate::server::responses::completion_to_response;
use crate::server::responses::fallback_completion;
use crate::server::responses::HyperResponse;
use crate::server::responses::ServerStats;
use crate::streaming::StreamAnswer;
use crate::testing::mock::MockDispatchContext;
use crate::types::{RouterMessageContent, RouterRequest, RouterResponse};
use common_core::string::strip_thinking_blocks;

use crate::charts::extract::WorkflowExtractor;

/// Outcome of a dispatch: the HTTP response plus the matched target's answer
/// text when it is known synchronously (buffered path). For the streaming
/// path the answer is assembled asynchronously and surfaced via
/// [`DispatchOutcome::stream_answer`].
///
/// M5: the handler records `answer_text` (or the finalized stream content)
/// into the session ledger + session step.
pub struct DispatchOutcome {
    pub response: HyperResponse,
    pub answer_text: Option<String>,
    pub stream_answer: Option<StreamAnswer>,
}

#[allow(clippy::implicit_hasher)]
pub async fn handle_dispatch(
    rt: &RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    user_text: &str,
    mock_dispatch: Option<&Arc<MockDispatchContext>>,
    http_client: &reqwest::Client,
    is_stream: bool,
    cache: Option<&Arc<ResponseCache>>,
    stats: &ServerStats,
    extractor: Option<Arc<WorkflowExtractor>>,
    ladders: &HashMap<String, Arc<EscalationLadder>>,
    context_cache: Option<&Arc<dyn fluent_types::ContextCache>>,
    session: Option<&Arc<Mutex<DependencySession>>>,
    instance_managers: &HashMap<String, Arc<crate::instances::InstanceManager>>,
) -> Result<DispatchOutcome, std::convert::Infallible> {
    // A rewind may have restored a KV snapshot; carry its fork-facing identity
    // (snapshot/instance/id_slot) into the outgoing body via the target's
    // request fields so the next dispatch switches that snapshot into its slot.
    let pending = session.and_then(|s| common_core::sync::lock(s).pending_kv_fields());
    let pending_rt: RoutingTarget;
    let rt: &RoutingTarget = if let Some((snapshot, instance, id_slot)) = pending {
        pending_rt = apply_pending_snapshot(rt, snapshot, instance, id_slot);
        &pending_rt
    } else {
        rt
    };
    let target_streams = is_stream && rt.stream;

    if !target_streams {
        if let Some(cache_backend) = cache {
            // Boundary: the cache key is the serialized request body.
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
                        ladders,
                        context_cache,
                        session,
                        instance_managers,
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
                return Ok(DispatchOutcome {
                    response: completion_to_response(
                        &response,
                        model_name,
                        false,
                        Some(&response.model),
                    ),
                    answer_text: answer_text(&response),
                    stream_answer: None,
                });
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
                    ladders,
                    context_cache,
                    session,
                    instance_managers,
                )
                .await;
            }
            tracing::info!(target: "router.server", model = %model_name, "mock canned response");
            let completion = mock.dispatch_response(entry, model_name);
            return Ok(DispatchOutcome {
                response: completion_to_response(&completion, model_name, is_stream, None),
                answer_text: answer_text(&completion),
                stream_answer: None,
            });
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
        ladders,
        context_cache,
        session,
        instance_managers,
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

/// Apply a restored KV snapshot's fork-facing identity to a target so the
/// next dispatch sends `snapshot`/`id_slot` (and `instance` when the target
/// has none) as request fields.
fn apply_pending_snapshot(
    target: &RoutingTarget,
    snapshot: String,
    instance: Option<String>,
    id_slot: i32,
) -> RoutingTarget {
    let mut owned = target.clone();
    owned.snapshot = Some(snapshot);
    owned.id_slot = Some(id_slot);
    if owned.instance.is_none() {
        owned.instance = instance;
    }
    owned
}

/// Reconstruct the prompt actually sent to the model from the normalized
/// request messages (M10a LOD0 fidelity).
///
/// The dispatch backend serializes exactly `request.messages` via
/// `normalize::messages_to_json`, so this assembly — the role-prefixed text
/// of every message, system first — is faithful to what the model received.
/// This is the *reconstructed* prompt (the exact rendered JSON body is not
/// recoverable at the call site).
///
/// `pub(crate)` so the escalation ladder (`dispatch::escalation`) can reuse
/// it for its `payload` audit field.
pub(crate) fn render_prompt(router_request: &RouterRequest) -> String {
    router_request
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content.to_string_lossy()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Try dispatching to a single target.  `is_primary` controls cache write;
/// `is_fallback` (an index > 0 in the dispatch chain) controls M10b
/// extraction scope.
/// Returns `Ok(DispatchOutcome)` on success or `Err(DispatchError)` on failure.
async fn dispatch_to_single_target(
    target: &RoutingTarget,
    router_request: &RouterRequest,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
    is_primary: bool,
    is_fallback: bool,
    user_text: &str,
    extractor: Option<Arc<WorkflowExtractor>>,
) -> Result<DispatchOutcome, DispatchError> {
    let backend = make_backend(http_client, target);

    let params = crate::dispatch::backend::params_with_routing_fields(
        target.params.clone(),
        target.instance.as_deref(),
        target.snapshot.as_deref(),
        target.id_slot,
    );

    if stream {
        let result = backend
            .stream_complete(
                router_request.clone(),
                target.model.clone(),
                params,
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
        return Ok(DispatchOutcome {
            response: resp,
            answer_text: None,
            stream_answer: result.answer,
        });
    }

    let filter_thinking = target.filter_thinking;
    let mut completion = backend
        .complete(
            router_request.clone(),
            target.model.clone(),
            params,
            target.idle_timeout_ms,
            target.total_timeout_ms,
            filter_thinking,
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
            // Boundary: the cache key is the serialized request body.
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Ok(response_json) = serde_json::to_value(&completion) {
                cache_backend.set(&target.model, &request_json, response_json);
            }
        }
    }

    // M10 learning loop: a successful buffered dispatch is a solved solution —
    // distill it into a draft chart (best-effort, never fails the request).
    // M10a: record the *real* rendered prompt; M10b: the extractor gates on
    // `is_fallback` + its configured mode (frontier-assisted by default).
    let answer = answer_text(&completion).unwrap_or_default();
    if let Some(extractor) = extractor {
        let prompt = render_prompt(router_request);
        extractor.record_success(user_text, &prompt, &target.model, &answer, is_fallback);
    }

    Ok(DispatchOutcome {
        response: completion_to_response(
            &completion,
            "",
            false,
            Some(&target.model),
        ),
        answer_text: answer_text(&completion),
        stream_answer: None,
    })
}

#[allow(clippy::implicit_hasher)]
pub async fn dispatch_real(
    rt: &RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    http_client: &reqwest::Client,
    stream: bool,
    cache: Option<&Arc<ResponseCache>>,
    user_text: &str,
    extractor: Option<Arc<WorkflowExtractor>>,
    ladders: &HashMap<String, Arc<EscalationLadder>>,
    context_cache: Option<&Arc<dyn fluent_types::ContextCache>>,
    session: Option<&Arc<Mutex<DependencySession>>>,
    instance_managers: &HashMap<String, Arc<crate::instances::InstanceManager>>,
) -> Result<DispatchOutcome, std::convert::Infallible> {
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
            i > 0,
            user_text,
            extractor.clone(),
        )
        .await
        {
            Ok(outcome) => {
                crate::audit::emit(
                    "route",
                    serde_json::json!({
                        "stage": "dispatch",
                        "verdict": "dispatched",
                        "model": target.model,
                        "url": target.url,
                        "attempt": i + 1,
                        "total": all_targets.len(),
                        "outcome": "success",
                    }),
                );
                return Ok(outcome);
            }
            Err(e) => {
                // M4 allocate-on-503: a group-miss means the pool had no free
                // member. Ask the sidecar to allocate fresh KV for the group
                // (weights already loaded), then retry this target once.
                if let DispatchError::InstanceGroupMiss { group } = &e {
                    if let Some(mgr) = instance_managers.get(&target.url) {
                        if mgr.ensure_group(group).await.is_ok() {
                            crate::audit::emit(
                                "instances",
                                serde_json::json!({
                                    "action": "allocate_on_miss",
                                    "group": group,
                                }),
                            );
                            if let Ok(outcome) = dispatch_to_single_target(
                                target,
                                router_request,
                                http_client,
                                stream,
                                cache,
                                i == 0,
                                i > 0,
                                user_text,
                                extractor.clone(),
                            )
                            .await
                            {
                                return Ok(outcome);
                            }
                        }
                    }
                }
                let attempt_latency_ms = attempt_start.elapsed().as_millis() as u64;
                let is_retryable = e.is_retryable();
                crate::audit::emit(
                    "route",
                    serde_json::json!({
                        "stage": "dispatch",
                        "verdict": "dispatch_failed",
                        "model": target.model,
                        "url": target.url,
                        "attempt": i + 1,
                        "total": all_targets.len(),
                        "outcome": "failed",
                        "error": e.to_string(),
                        "retryable": is_retryable,
                    }),
                );
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

    // M3 escalation: only after the local chain is exhausted do we engage the
    // frontier ladder. The ladder is resolved from the resolved route's group
    // (`RoutingTarget.group`); direct-model targets (no group) get `None`.
        if let Some(ladder) = rt.group.as_deref().and_then(|g| ladders.get(g)) {
        tracing::info!(
            target: "router.server",
            group = ?rt.group,
            model = %model_name,
            last_error = ?last_error,
            "local chain exhausted — engaging escalation ladder"
        );
        let esc_ctx = EscalationContext {
            request: router_request,
            user_text,
            model_name,
            context_cache,
            session,
        };
        if let Some(resp) = ladder.try_escalate(&esc_ctx).await {
            return Ok(DispatchOutcome {
                response: resp,
                answer_text: None,
                stream_answer: None,
            });
        }
    }

    tracing::warn!(
        target: "router.server",
        error = ?last_error,
        "all dispatch targets failed, returning fallback response"
    );
    let completion = fallback_completion(model_name);
    Ok(DispatchOutcome {
        response: completion_to_response(
            &completion,
            model_name,
            stream,
            None,
        ),
        answer_text: answer_text(&completion),
        stream_answer: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_target() -> RoutingTarget {
        crate::pipeline::RoutingTarget {
            url: "http://x/v1/chat/completions".into(),
            model: "base:swarm".into(),
            group: None,
            target_name: Some("swarm".into()),
            params: None,
            instance: None,
            snapshot: None,
            id_slot: None,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            stream: true,
            idle_timeout_ms: 5000,
            total_timeout_ms: 30000,
            fallbacks: vec![],
        }
    }

    #[test]
    fn apply_pending_snapshot_sets_request_fields() {
        let rt = apply_pending_snapshot(&base_target(), "readfiles".into(), Some("scratch".into()), 2);
        assert_eq!(rt.snapshot.as_deref(), Some("readfiles"));
        assert_eq!(rt.instance.as_deref(), Some("scratch"));
        assert_eq!(rt.id_slot, Some(2));
    }

    #[test]
    fn apply_pending_snapshot_preserves_existing_instance() {
        let mut t = base_target();
        t.instance = Some("ledger".into());
        let rt = apply_pending_snapshot(&t, "readfiles".into(), Some("scratch".into()), 0);
        assert_eq!(rt.instance.as_deref(), Some("ledger"), "existing instance wins");
        assert_eq!(rt.snapshot.as_deref(), Some("readfiles"));
    }

    /// A stub that serves both the chat-completions endpoint and the
    /// management `/instances` endpoint from one listener: the chat path
    /// returns a 503 group-miss on the first call and a success completion on
    /// the second; `/instances` allocates (201). Used to assert the
    /// allocate-on-503 retry.
    #[tokio::test]
    async fn allocate_on_503_creates_instance_and_retries_once() {
        use crate::instances::stub::StubServer;
        use crate::instances::{management_base_url, InstanceClient, InstanceManager};
        use crate::config::InstanceProfile;
        use std::sync::Arc as StdArc;
        use std::sync::Mutex;

        let chat_calls = StdArc::new(Mutex::new(0usize));
        let chat_calls_c = chat_calls.clone();
        let handler: StdArc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
            StdArc::new(move |method, path, _body| {
                if method == "POST" && path.ends_with("/chat/completions") {
                    let mut n = chat_calls_c.lock().unwrap();
                    *n += 1;
                    if *n == 1 {
                        // The fork's 503 group-miss payload.
                        return (
                            503,
                            r#"{"error":{"code":503,"message":"no free instance in group 'swarm'","type":"unavailable_error"}}"#
                                .into(),
                        );
                    }
                    return (
                        200,
                        r#"{"id":"x","object":"chat.completion","model":"base:swarm","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
                            .into(),
                    );
                }
                (201, "{}".into())
            });
        let stub = StubServer::start(handler);

        let endpoint = format!("{}/v1/chat/completions", stub.base_url());
        let mut target = base_target();
        target.url = endpoint.clone();
        target.instance = Some("swarm".into());
        target.stream = false; // buffered dispatch for simplicity

        // A manager whose client points at the same server's management API.
        let client = InstanceClient::new(
            reqwest::Client::new(),
            management_base_url(&endpoint),
            None,
        );
        let profile = InstanceProfile {
            name: Some("swarm0".into()),
            group: Some("swarm".into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            params: None,
        };
        let manager = Arc::new(InstanceManager::new(
            client,
            vec![profile],
            crate::config::SidecarConfig::default(),
        ));
        let mut managers = std::collections::HashMap::new();
        managers.insert(endpoint.clone(), manager);

        let request = crate::types::RouterRequest {
            model: "base".into(),
            messages: vec![crate::types::RouterMessage {
                role: "user".into(),
                content: crate::types::RouterMessageContent::Text("hello".into()),
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
        };
        let outcome = dispatch_real(
            &target,
            &request,
            "base",
            &reqwest::Client::new(),
            false,
            None,
            "hello",
            None,
            &std::collections::HashMap::new(),
            None,
            None,
            &managers,
        )
        .await
        .expect("dispatch_real is infallible");
        assert!(outcome.response.status().is_success(), "retry succeeded");

        let recorded = stub.recorded();
        // Exactly two chat calls (first group-miss, then retry) and one
        // management `POST /instances` in between.
        let chat_hits = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p.ends_with("/chat/completions"))
            .count();
        let create_hits = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .count();
        assert_eq!(chat_hits, 2, "group-miss then retry");
        assert_eq!(create_hits, 1, "a fresh instance was allocated between");
    }
}
