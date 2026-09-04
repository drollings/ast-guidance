use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use fluent_concurrency::ladder::first_accept_in_order;
use fluent_llm::cache::ResponseCache;
use http_body_util::BodyExt;

use crate::dag_session::DependencySession;
use crate::dispatch::backend::DispatchBackend;
use crate::dispatch::backend::OpenAiChatBackend;
use crate::dispatch::backend::RetryBackend;
use crate::dispatch::escalation::{EscalationContext, Ladder};
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
use fluent_llm::thinking::strip_thinking_blocks;

use crate::charts::extract::WorkflowExtractor;

/// Outcome of a dispatch: the HTTP response plus the matched target's answer
/// text when it is known synchronously (buffered path). For the streaming
/// path the answer is assembled asynchronously and surfaced via
/// [`DispatchOutcome::stream_answer`].
///
/// The handler records `answer_text` (or the finalized stream content)
/// into the session ledger + session step.
pub struct DispatchOutcome {
    pub response: HyperResponse,
    pub answer_text: Option<String>,
    pub stream_answer: Option<StreamAnswer>,
}

/// The shared, request-invariant dependencies threaded through
/// `handle_dispatch` → `dispatch_real` → `dispatch_to_single_target`.
///
/// Pure struct bundling: collapses the former 14/12/9-argument dispatch
/// signatures into one `&DispatchDeps` (the fields `ServerDeps` already
/// bundles for the HTTP handler). Zero runtime cost — no vtable, no behavior
/// change. Build once per request in the handler and borrow it.
#[derive(Clone)]
pub struct DispatchDeps {
    pub http_client: Arc<reqwest::Client>,
    pub cache: Option<Arc<ResponseCache>>,
    pub stats: Arc<ServerStats>,
    pub extractor: Option<Arc<WorkflowExtractor>>,
    pub ladders: HashMap<String, Arc<Ladder>>,
    pub context_cache: Option<Arc<dyn fluent_types::ContextCache>>,
    pub session: Option<Arc<Mutex<DependencySession>>>,
    pub instance_pool: Option<crate::instances::InstancePool>,
    /// The in-process onnx generative `ChatBackend` (the `onnx/llm` routing
    /// model). Onnx `RoutingTarget`s are served through it. `None` (e.g. mock
    /// mode, no onnx fleet) leaves onnx targets unserved — the mock intercepts
    /// first in tests.
    pub onnx_llm_backend: Option<Arc<dyn fluent_llm::client::ChatBackend>>,
}

#[allow(clippy::implicit_hasher)]
pub async fn handle_dispatch(
    rt: &RoutingTarget,
    router_request: &RouterRequest,
    model_name: &str,
    user_text: &str,
    mock_dispatch: Option<&Arc<MockDispatchContext>>,
    is_stream: bool,
    deps: &DispatchDeps,
) -> Result<DispatchOutcome, std::convert::Infallible> {
    // A rewind may have restored a KV snapshot; carry its fork-facing identity
    // (snapshot/instance/id_slot) into the outgoing body via the target's
    // request fields so the next dispatch switches that snapshot into its slot.
    let pending = deps
        .session
        .as_ref()
        .and_then(|s| common_core::sync::lock(s).pending_kv_fields());
    // The client's explicit routing fields (`instance`/`snapshot`/`id_slot`)
    // override any the target derived from config; pending rewind fields take
    // precedence over the request's.
    let needs_overlay = router_request.instance.is_some()
        || router_request.snapshot.is_some()
        || router_request.id_slot.is_some()
        || pending.is_some();
    let owned_rt = if needs_overlay {
        let mut t = crate::server::instances_api::apply_request_routing_fields(rt, router_request);
        if let Some((snapshot, instance, id_slot)) = pending {
            t.snapshot = Some(snapshot);
            if t.instance.is_none() {
                t.instance = instance;
            }
            t.id_slot = Some(id_slot);
        }
        Some(t)
    } else {
        None
    };
    let rt: &RoutingTarget = match &owned_rt {
        Some(v) => v,
        None => rt,
    };
    let target_streams = is_stream && rt.stream;

    if !target_streams {
        if let Some(cache_backend) = deps.cache.as_ref() {
            // Boundary: the cache key is the serialized request body.
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Some(cached) = cache_backend.get(&rt.model, &request_json) {
                deps.stats
                    .cache_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(target: "router.dispatch", model = %rt.model, "cache hit");
                let Ok(mut response) =
                    serde_json::from_value::<RouterResponse>(cached.response_json)
                else {
                    deps.stats
                        .cache_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return dispatch_real(
                        rt,
                        router_request,
                        model_name,
                        deps,
                        target_streams,
                        user_text,
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
            deps.stats
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
                    deps,
                    target_streams,
                    user_text,
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
        deps,
        target_streams,
        user_text,
    )
    .await
}

/// Build a `DispatchBackend` (optionally wrapped in `RetryBackend`) for a single
/// routing target. An onnx target (`is_onnx`) is served by the in-process onnx
/// `ChatBackend`; every other target goes over HTTP to `target.url`.
fn make_backend(deps: &DispatchDeps, target: &RoutingTarget) -> Arc<dyn DispatchBackend> {
    if target.is_onnx {
        if let Some(onnx) = &deps.onnx_llm_backend {
            return Arc::new(crate::dispatch::backend::OnnxDispatchBackend::new(onnx.clone()));
        }
        tracing::warn!(
            target: "router.dispatch",
            model = %target.model,
            "onnx dispatch target but no onnx backend wired — dispatch will fail open",
        );
    }
    let base: Arc<dyn DispatchBackend> = Arc::new(
        OpenAiChatBackend::new((*deps.http_client).clone(), &target.url).with_api_key(target.api_key.clone()),
    );
    if target.retry_count > 0 {
        Arc::new(RetryBackend::new(
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
#[cfg(test)]
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
/// request messages.
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
/// `is_fallback` (an index > 0 in the dispatch chain) controls extraction
/// scope.
/// Returns `Ok(DispatchOutcome)` on success or `Err(DispatchError)` on failure.
async fn dispatch_to_single_target(
    target: &RoutingTarget,
    router_request: &RouterRequest,
    stream: bool,
    is_primary: bool,
    is_fallback: bool,
    user_text: &str,
    deps: &DispatchDeps,
) -> Result<DispatchOutcome, DispatchError> {
    let backend = make_backend(deps, target);

    let params = crate::dispatch::backend::params_with_routing_fields(
        target.params.clone(),
        target.instance.as_deref(),
        target.snapshot.as_deref(),
        target.id_slot,
    );

    if stream {
        // The downstream abort signal: the body drop-guard fires it when the
        // client stops consuming, and it is what (a) the forwarding task
        // selects on to drop the upstream connection and (b) this watcher
        // reacts to with an explicit management-plane abort.
        let abort = fluent_concurrency::stream::StreamAbort::new();

        // Approach C — the router is the process owner of the fleet, so on a
        // downstream disconnect it also asks the owning server to stop the
        // generation explicitly (belt-and-suspenders on top of the transport
        // close the forwarding task performs on the same signal). Best-effort:
        // a server that does not expose `/abort`, or a slot id that is not
        // mid-generation, answers non-2xx and is logged and ignored.
        if let Some(mgr) = deps
            .instance_pool
            .as_ref()
            .and_then(|pool| pool.manager_for_url(&target.url))
        {
            let mgr = mgr.clone();
            let abort_watch = abort.clone();
            let model = target.model.clone();
            let url = target.url.clone();
            let id_slot = target.id_slot;
            tokio::spawn(async move {
                abort_watch.cancelled().await;
                crate::audit::emit(
                    "stream",
                    serde_json::json!({
                        "stage": "dispatch",
                        "verdict": "aborted",
                        "model": model,
                        "url": url,
                        "id_slot": id_slot,
                    }),
                );
                if let Err(e) = mgr.abort_generation(id_slot).await {
                    tracing::debug!(
                        target: "router.dispatch",
                        model = %model,
                        error = %e,
                        "management abort best-effort"
                    );
                }
            });
        }

        let result = backend
            .stream_complete_with_abort(
                router_request.clone(),
                target.model.clone(),
                params,
                target.idle_timeout_ms,
                target.total_timeout_ms,
                target.filter_thinking,
                Some(abort),
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
        if let Some(cache_backend) = deps.cache.as_ref() {
            // Boundary: the cache key is the serialized request body.
            let request_json = serde_json::to_string(router_request).unwrap_or_default();
            if let Ok(response_json) = serde_json::to_value(&completion) {
                cache_backend.set(&target.model, &request_json, response_json);
            }
        }
    }

    // Learning loop: a successful buffered dispatch is a solved solution —
    // distill it into a draft chart (best-effort, never fails the request).
    // Record the *real* rendered prompt; the extractor gates on
    // `is_fallback` + its configured mode (frontier-assisted by default).
    let answer = answer_text(&completion).unwrap_or_default();
    if let Some(extractor) = deps.extractor.as_ref() {
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
    deps: &DispatchDeps,
    stream: bool,
    user_text: &str,
) -> Result<DispatchOutcome, std::convert::Infallible> {
    let all_targets = std::iter::once(rt)
        .chain(rt.fallbacks.iter())
        .collect::<Vec<_>>();

    let mut attempt = 0usize;
    let total = all_targets.len();

    // First-accept-wins over the target chain (primary + fallbacks). Each
    // rung owns the per-target residency/audit/warn side effects and the
    // allocate-on-503 retry; `stop` short-circuits on a non-retryable error
    // (e.g. 400 Bad Request). The combinator returns the terminal error (the
    // stop trigger or the last rung failure) for the post-chain
    // escalation/fallback handling.
    let result = first_accept_in_order(
        all_targets,
        |target| {
            let i = attempt;
            attempt += 1;
            async move {
                tracing::info!(
                    target: "router.server",
                    attempt = i + 1,
                    total = total,
                    model = %target.model,
                    url = %target.url,
                    stream = stream,
                    retry_count = target.retry_count,
                    idle_timeout_ms = target.idle_timeout_ms,
                    total_timeout_ms = target.total_timeout_ms,
                    "dispatch attempt"
                );

                // On-demand residency: ensure the target's managed model is
                // loaded (spawn its llama-server if it is lazy and currently
                // unloaded) and that a specifically-targeted instance exists.
                // Best-effort — a load failure surfaces as the target's own
                // dispatch error below.
                if let Some(pool) = deps.instance_pool.as_ref() {
                    pool.ensure_target_ready(&target.url, target.instance.as_deref())
                        .await;
                }
                // ROADMAP M7 resize-to-demand: a caller that declares a bigger
                // context window than the targeted instance is allocated gets
                // the instance resized to fit (bounded by `max_ctx`); a need
                // beyond `max_ctx` fails loudly with the same shape a too-large
                // llama request gets. Best-effort on the resize failure.
                if let Some(pool) = deps.instance_pool.as_ref() {
                    let num_ctx = router_request
                        .metadata
                        .get("num_ctx")
                        .and_then(serde_json::Value::as_u64);
                    if let Err(e) = pool
                        .resize_to_demand(&target.url, target.instance.as_deref(), num_ctx)
                        .await
                    {
                        tracing::warn!(
                            target: "router.instances",
                            model = %target.model,
                            instance = ?target.instance,
                            error = %e,
                            "resize-to-demand failed",
                        );
                    }
                }

                let attempt_start = Instant::now();
                match dispatch_to_single_target(
                    target,
                    router_request,
                    stream,
                    i == 0,
                    i > 0,
                    user_text,
                    deps,
                )
                .await
                {
                    Ok(outcome) => {
                        crate::audit::AuditRecord::route(
                            crate::pipeline_types::PipelineStage::Classifier,
                            crate::pipeline_types::StageVerdict::Passed,
                            Some(target),
                            None,
                            Some("dispatched"),
                        )
                        .emit();
                        Ok(Some(outcome))
                    }
                    Err(e) => {
                        // Allocate-on-503: a group-miss means the pool had
                        // no free member. Ask the sidecar to allocate fresh KV
                        // for the group (weights already loaded), then retry
                        // this target once.
                        if let DispatchError::InstanceGroupMiss { group } = &e {
                            if let Some(mgr) = deps
                                .instance_pool
                                .as_ref()
                                .and_then(|pool| pool.manager_for_url(&target.url))
                            {
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
                                        stream,
                                        i == 0,
                                        i > 0,
                                        user_text,
                                        deps,
                                    )
                                    .await
                                    {
                                        return Ok(Some(outcome));
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
                                "total": total,
                                "outcome": "failed",
                                "error": e.to_string(),
                                "retryable": is_retryable,
                            }),
                        );
                        tracing::warn!(
                            target: "router.server",
                            attempt = i + 1,
                            total = total,
                            model = %target.model,
                            error = %e,
                            retryable = is_retryable,
                            attempt_latency_ms = attempt_latency_ms,
                            remaining = total.saturating_sub(i + 1),
                            "dispatch attempt failed"
                        );
                        Err(e)
                    }
                }
            }
        },
        |e: &DispatchError| !e.is_retryable(),
    )
    .await;

    let last_error = match result {
        Ok(Some(outcome)) => return Ok(outcome),
        Err(e) => Some(e),
        Ok(None) => None,
    };

    // Escalation: only after the local chain is exhausted do we engage the
    // frontier ladder. The ladder is resolved from the resolved route's group
    // (`RoutingTarget.group`); direct-model targets (no group) get `None`.
        if let Some(ladder) = rt.group.as_deref().and_then(|g| deps.ladders.get(g)) {
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
            context_cache: deps.context_cache.as_ref(),
            session: deps.session.as_ref(),
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
#[path = "../../tests/server_dispatch.rs"]
mod tests;
