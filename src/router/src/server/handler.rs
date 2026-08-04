use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use common_core::hash::uuid_v4;
use common_core::ResponseCache;
use http_body_util::BodyExt;

use crate::config::{ModelEntry, RouteRef};
use crate::dag_session::{DependencySession, SessionRegistry, SessionStep, StepResult};
use crate::ledger::ContentNodeLedger;
use crate::normalize;
use crate::pipeline::{PipelineOrchestrator, RoutingTarget};
use crate::routes::plan::PlanRoute;
use crate::server::dispatch::handle_dispatch;
use crate::server::responses::completion_to_response;
use crate::server::responses::empty_response;
use crate::server::responses::error_response;
use crate::server::responses::make_error_completion;
use crate::server::responses::make_text_completion;
use crate::server::responses::HyperResponse;
use crate::server::responses::ServerStats;
use crate::testing::mock::MockDispatchContext;
use crate::types::RouterRequest;

/// Best-effort ledger insert, moved off the async handler via
/// `spawn_blocking` so sync rusqlite never runs on a tokio worker thread.
///
/// Both failure modes are swallowed by design: a panicked blocking task
/// (`.ok()`) and a ledger error (`.flatten()`) degrade to "no ledger row",
/// matching the documented best-effort logging contract.
pub(crate) async fn record_ledger_request(
    ledger: Option<&Arc<ContentNodeLedger>>,
    session_id: String,
    request_id: String,
    request_text: String,
) -> Option<fluent_types::NodeId> {
    let l = Arc::clone(ledger?);
    tokio::task::spawn_blocking(move || l.record_request(&session_id, &request_id, &request_text))
        .await
        .ok()
        .and_then(Result::ok)
}

/// Best-effort ledger update, off the async handler (see
/// `record_ledger_request` for the swallow semantics).
pub(crate) async fn record_ledger_result(
    ledger: Option<&Arc<ContentNodeLedger>>,
    node_id: Option<fluent_types::NodeId>,
    accepted: bool,
    score: Option<f64>,
    content: String,
) {
    let (Some(l), Some(node_id)) = (ledger, node_id) else {
        return;
    };
    let l = Arc::clone(l);
    tokio::task::spawn_blocking(move || {
        let _ = l.record_result(node_id, accepted, score, &content);
    })
    .await
    .ok();
}

/// Per-request handle into a `DependencySession` step (D6). Holds the session
/// `Arc` and the request's step id so the outcome can be recorded exactly once
/// from whichever terminal branch the request takes. Locking is scoped to the
/// `complete` call — never held across an await.
struct SessionStepHandle {
    session: Arc<Mutex<DependencySession>>,
    step_id: String,
}

impl SessionStepHandle {
    fn complete(&self, accepted: bool, score: Option<f64>, content: String, error: Option<String>) {
        let mut session = match self.session.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let _ = session.complete_step(
            &self.step_id,
            StepResult {
                content,
                accepted,
                score,
                latency_ms: 0,
                error,
            },
        );
    }
}

/// Register the request as a step in the session keyed by `session_id` (if a
/// `SessionRegistry` is wired) and return a handle to complete it when the
/// outcome is known.
fn begin_session_step(
    sessions: Option<&Arc<SessionRegistry>>,
    session_id: &str,
    model_name: &str,
    adapter: Option<&str>,
    request_id: &str,
    request_text: &str,
) -> Option<SessionStepHandle> {
    let registry = sessions?;
    let session = registry.get_or_create(session_id);
    {
        let mut s = match session.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        s.set_model(model_name);
        if let Some(adapter) = adapter {
            s.adapter = Some(adapter.to_string());
        }
        let step_id = format!("req-{request_id}");
        if s.get_step(&step_id).is_none() {
            let _ = s.add_step(SessionStep::new(step_id.clone(), request_text));
        }
    }
    Some(SessionStepHandle {
        session,
        step_id: format!("req-{request_id}"),
    })
}

async fn handle_chat_completion(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<std::collections::HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<std::collections::HashMap<String, RouteRef>>,
    models: Arc<std::collections::HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier: Option<(String, ModelEntry)>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    plan_route: Option<Arc<PlanRoute>>,
    sessions: Option<Arc<SessionRegistry>>,
    http_client: Arc<reqwest::Client>,
) -> Result<HyperResponse, std::convert::Infallible> {
    // M10: the dispatch post-processing hook (workflow extraction), if the
    // operator configured it. Passed through to successful dispatches only.
    let workflow_extractor = plan_route
        .as_ref()
        .and_then(|p| p.workflow_extractor().cloned());
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
        .map(|m| common_core::string::truncate_utf8(&m.content.to_string_lossy(), 120))
        .unwrap_or_default();

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
    let ledger_node_id = record_ledger_request(
        ledger.as_ref(),
        session_id.clone(),
        request_id.clone(),
        request_text.clone(),
    )
    .await;

    // D6 canonical session: register the request as a step and complete it at
    // whichever terminal branch the request takes (outcome recorded exactly
    // once).
    let session_step = begin_session_step(
        sessions.as_ref(),
        &session_id,
        &model_name,
        router_request.adapter.as_deref(),
        &request_id,
        &request_text,
    );

    let pipeline_result =
        resolve_pipeline(&model_name, &routes, &models, &pipelines, &router_request);

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

        record_ledger_result(
            ledger.as_ref(),
            ledger_node_id,
            false,
            Some(0.0),
            reason.to_string(),
        )
        .await;

        if let Some(ref step) = session_step {
            step.complete(
                false,
                Some(0.0),
                reason.to_string(),
                Some(reason.to_string()),
            );
        }

        let completion = make_error_completion(&model_name, reason);
        return Ok(completion_to_response(
            &completion,
            &model_name,
            is_stream,
            None,
        ));
    }

    if let Some(ref resp_str) = pipeline_result.classifier_response {
        tracing::info!(
            target: "router.server",
            model = %model_name,
            response_len = resp_str.len(),
            "responding with classifier direct response"
        );
        record_ledger_result(
            ledger.as_ref(),
            ledger_node_id,
            true,
            Some(1.0),
            resp_str.clone(),
        )
        .await;
        if let Some(ref step) = session_step {
            step.complete(true, Some(1.0), resp_str.clone(), None);
        }
        let completion = make_text_completion(&model_name, resp_str);
        return Ok(completion_to_response(
            &completion,
            &model_name,
            is_stream,
            None,
        ));
    }

    if let Some(ref rt) = pipeline_result.routing_target {
        let resp = handle_dispatch(
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
            workflow_extractor.clone(),
        )
        .await?;
        if let Some(ref step) = session_step {
            step.complete(
                resp.status().is_success(),
                None,
                format!("dispatched to {}: {}", rt.model, resp.status()),
                None,
            );
        }
        return Ok(resp);
    }

    if let Some((ref key, ref entry)) = classifier {
        let rt_for_fallback = RoutingTarget::from_model_entry(key, entry);
        tracing::info!(
            target: "router.server",
            model = %rt_for_fallback.model,
            fallback_url = %rt_for_fallback.url,
            "no routing target — dispatching to classifier fallback"
        );
        let resp = handle_dispatch(
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
            workflow_extractor.clone(),
        )
        .await?;
        if let Some(ref step) = session_step {
            step.complete(
                resp.status().is_success(),
                None,
                format!("dispatched to classifier fallback: {}", resp.status()),
                None,
            );
        }
        return Ok(resp);
    }

    tracing::warn!(
        target: "router.server",
        model = %model_name,
        "no routing target, no classifier response, no classifier url — returning fallback"
    );
    record_ledger_result(
        ledger.as_ref(),
        ledger_node_id,
        true,
        Some(0.5),
        "fallback response".to_string(),
    )
    .await;
    if let Some(ref step) = session_step {
        step.complete(true, Some(0.5), "fallback response".to_string(), None);
    }
    let completion = crate::server::responses::fallback_completion(&model_name);
    Ok(completion_to_response(
        &completion,
        &model_name,
        is_stream,
        None,
    ))
}

/// The `plan` route's HTTP surface (M8): a single targeted interview round.
///
/// The request body carries the user message and an optional entity list
/// (the client's answers to a prior clarification, serialized as entities —
/// the same shape stored under structured `entities`). The response is
/// structured JSON, never free-form chat:
///
/// - `{"status": "clarify", "questions": [...], "gaps": [...]}` — the chart
///   needs one round of targeted answers; the client replies with `entities`
///   plus `retry: true` and the echoed `gaps`.
/// - `{"status": "executed", "workflow": {...}, "source": ..., "gaps_filled":
///   [...]}` — the chart is bound and compiled.
/// - `{"status": "fresh_draft", "source": "fresh_draft"}` — no chart fit;
///   planning falls through to a blank slate.
///
/// A `retry` request that still leaves gaps terminates as `fresh_draft`
/// (VISION: terminate, don't loop).
async fn handle_plan_request(
    req: hyper::Request<hyper::body::Incoming>,
    plan_route: Option<Arc<PlanRoute>>,
    max_payload: usize,
    stats: &ServerStats,
) -> Result<HyperResponse, std::convert::Infallible> {
    let Some(route) = plan_route else {
        stats.errors.fetch_add(1, Ordering::Relaxed);
        return Ok(crate::server::responses::error_response(
            hyper::StatusCode::SERVICE_UNAVAILABLE,
            "plan route not configured",
        ));
    };

    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(crate::server::responses::error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("body read error: {e}"),
            ));
        }
    };
    if body_bytes.len() > max_payload {
        return Ok(empty_response(hyper::StatusCode::PAYLOAD_TOO_LARGE));
    }
    let body: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            stats.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(crate::server::responses::error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("invalid JSON: {e}"),
            ));
        }
    };

    let message = body
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if message.is_empty() {
        return Ok(crate::server::responses::error_response(
            hyper::StatusCode::BAD_REQUEST,
            "missing 'message'",
        ));
    }

    let entities: Vec<crate::charts::binding::Entity> = body
        .get("entities")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|e| serde_json::from_value(e.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    let retry = body
        .get("retry")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let prior_gaps: Vec<String> = body
        .get("gaps")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();

    let result = if retry {
        route
            .plan_interviewed(message, &entities, &prior_gaps)
            .await
    } else {
        route.plan(message, &entities).await
    };

    let response = match result.source {
        crate::routes::plan::PlanSource::FreshDraft => {
            serde_json::json!({ "status": "fresh_draft", "source": "fresh_draft" })
        }
        crate::routes::plan::PlanSource::HnswHit => plan_executed_response("hnsw_hit", &result),
        crate::routes::plan::PlanSource::TemplateAdapted => {
            if result.interview_questions.is_empty() {
                plan_executed_response("template_adapted", &result)
            } else {
                serde_json::json!({
                    "status": "clarify",
                    "source": "template_adapted",
                    "questions": result.interview_questions,
                    "gaps": result.gaps,
                })
            }
        }
    };
    Ok(crate::server::responses::json_response(
        hyper::StatusCode::OK,
        &response,
    ))
}

/// Build the D3 `/v1/plan` "executed" response: execution results, not a
/// compiled graph. Carries selection provenance (`fit`/`score`) and the
/// execution summary (`final_output`/`accepted`/`audit`) when the chart ran.
fn plan_executed_response(
    source: &str,
    result: &crate::routes::plan::PlanResult,
) -> serde_json::Value {
    let mut executed = serde_json::json!({
        "status": "executed",
        "source": source,
        "gaps_filled": result.gaps_filled,
    });
    if let Some(fit) = &result.fit {
        executed["fit"] = serde_json::Value::String(fit.clone());
    }
    if let Some(score) = result.score {
        executed["score"] = serde_json::json!(score);
    }
    if let Some(summary) = &result.summary {
        executed["accepted"] = serde_json::json!(summary.accepted);
        if let Some(output) = &summary.final_output {
            executed["final_output"] = output.clone();
        }
        executed["audit"] = serde_json::to_value(&summary.audit).unwrap_or_default();
        executed["completed"] = serde_json::to_value(&summary.completed).unwrap_or_default();
    }
    executed
}

#[allow(clippy::implicit_hasher)]
pub async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    pipelines: Arc<std::collections::HashMap<String, Arc<PipelineOrchestrator>>>,
    routes: Arc<std::collections::HashMap<String, RouteRef>>,
    models: Arc<std::collections::HashMap<String, ModelEntry>>,
    stats: Arc<ServerStats>,
    max_payload: usize,
    classifier: Option<(String, ModelEntry)>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    plan_route: Option<Arc<PlanRoute>>,
    sessions: Option<Arc<SessionRegistry>>,
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
            Ok(crate::server::responses::json_response(
                hyper::StatusCode::OK,
                &body,
            ))
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
                req,
                pipelines,
                routes,
                models,
                stats,
                max_payload,
                classifier,
                mock_dispatch,
                ledger,
                cache,
                plan_route,
                sessions,
                http_client,
            )
            .await
        }
        ("POST", "/v1/plan") => {
            stats.requests.fetch_add(1, Ordering::Relaxed);
            handle_plan_request(req, plan_route, max_payload, &stats).await
        }
        _ => {
            if method == hyper::Method::DELETE && path.starts_with("/admin/cache/") {
                if !is_local_request(&req) {
                    return Ok(crate::server::responses::forbidden_response());
                }
                let key = &path["/admin/cache/".len()..];
                if key.is_empty() {
                    return Ok(crate::server::responses::error_response(
                        hyper::StatusCode::BAD_REQUEST,
                        "missing cache key",
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
    router_request: &RouterRequest,
) -> crate::pipeline::PipelineResult {
    use fluent_wvr::prelude::*;

    let route = routes.get(model_name).cloned();

    let pipeline_names: Vec<String> = if let Some(ref r) = route {
        r.pipelines.clone()
    } else if let Some(model_entry) = models.get(model_name) {
        let rt = RoutingTarget::from_model_entry(model_name, model_entry);
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
    ctx.set_structured("request", router_request);

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
