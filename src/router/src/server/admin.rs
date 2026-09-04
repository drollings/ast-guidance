//! Admin endpoints for the CLI (`coral-router ps/stop/speedtest`): `POST
//! /models/unload` (stop a managed model's server) and `GET /metrics`
//! (aggregate the managed llama-servers' Prometheus expositions, optionally
//! filtered by `?model=`).

use std::collections::HashSet;

use bytes::Bytes;
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;

use crate::server::handler::ServerDeps;
use crate::server::responses::{add_cors_headers, error_response, json_response, HyperResponse};

/// `POST /models/unload` — unload a managed model's `llama-server` (frees its
/// VRAM). The spec stays registered so a later dispatch reloads it on demand.
pub async fn handle_unload_model(
    req: hyper::Request<Incoming>,
    deps: &ServerDeps,
) -> HyperResponse {
    if !crate::server::handler::is_local_request(&req) {
        return crate::server::responses::forbidden_response();
    }
    let body = match crate::server::instances_api::read_json_body(req, deps.max_payload).await {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let key = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    if key.is_empty() {
        return error_response(hyper::StatusCode::BAD_REQUEST, "missing 'model' field");
    }
    // Onnx models with `Always` residency load once at boot and are never
    // evicted or unloaded — refuse loudly instead of silently no-oping.
    if let Some(onnx) = deps.onnx.as_ref() {
        if onnx.refuses_unload(&key) {
            return error_response(
                hyper::StatusCode::CONFLICT,
                &format!(
                    "model '{key}' is an always-resident onnx model and cannot be unloaded"
                ),
            );
        }
    }
    // Route through the unified fleet when attached: onnx `Unloadable` roles
    // release through their `OnnxWeights`; llama through the supervisor
    // adapter. The `Always`/pinned refusal above is preserved exactly.
    if let Some(fleet) = deps.fleet.as_ref() {
        if fleet.is_known_model(&key) {
            return match fleet.unload(&key).await {
                Ok(()) => json_response(
                    hyper::StatusCode::OK,
                    &serde_json::json!({ "status": "ok", "model": key }),
                ),
                Err(e) => error_response(hyper::StatusCode::BAD_GATEWAY, &e.to_string()),
            };
        }
    }
    let Some(supervisor) = deps.supervisor.as_ref() else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed models");
    };
    if supervisor.base_url_for(&key).is_none() {
        return error_response(
            hyper::StatusCode::NOT_FOUND,
            &format!("unknown model: '{key}'"),
        );
    }
    supervisor.unload(&key).await;
    json_response(
        hyper::StatusCode::OK,
        &serde_json::json!({ "status": "ok", "model": key }),
    )
}

/// `GET /metrics` — Prometheus text exposition of the managed llama-servers.
///
/// With `?model=<key>` the exposition of that model's server is proxied
/// verbatim (for a managed model) or derived from the model's own endpoint
/// origin (for a standalone/remote model). Without a model filter, every
/// managed server's exposition is concatenated, with `# HELP`/`# TYPE`
/// documentation lines deduplicated by metric name.
pub async fn handle_metrics(deps: &ServerDeps, query: &[(String, String)]) -> HyperResponse {
    let model = query
        .iter()
        .find(|(k, _)| k == "model")
        .map(|(_, v)| v.clone());
    let client = reqwest::Client::new();

    if let Some(model) = model {
        let Some(url) = managed_or_endpoint_origin(deps, &model) else {
            return error_response(
                hyper::StatusCode::NOT_FOUND,
                &format!("unknown model: '{model}'"),
            );
        };
        return match fetch_metrics(&client, &url).await {
            Ok(text) => text_response(text),
            Err(e) => error_response(
                hyper::StatusCode::BAD_GATEWAY,
                &format!("metrics for '{model}' unavailable: {e}"),
            ),
        };
    }

    let Some(supervisor) = deps.supervisor.as_ref() else {
        return text_response("# no managed models\n".into());
    };
    let mut keys: Vec<String> = supervisor.model_keys();
    keys.sort_unstable();

    let mut bodies = Vec::new();
    for key in keys {
        if let Some(base_url) = supervisor.base_url_for(&key) {
            if let Ok(text) = fetch_metrics(&client, &format!("{base_url}/metrics")).await {
                bodies.push(text);
            }
        }
    }
    if bodies.is_empty() {
        return text_response("# no managed models\n".into());
    }
    text_response(dedupe_metric_docs(bodies))
}

/// The `/metrics` origin for a model: its managed server's base URL, or (for
/// a standalone model) the origin of its configured chat-completions endpoint.
fn managed_or_endpoint_origin(deps: &ServerDeps, model: &str) -> Option<String> {
    if let Some(base_url) = deps.supervisor.as_ref().and_then(|s| s.base_url_for(model)) {
        return Some(format!("{base_url}/metrics"));
    }
    let entry = deps.models.get(model)?;
    let origin = url_origin(&entry.endpoint)?;
    Some(format!("{origin}/metrics"))
}

fn url_origin(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let host_port = rest.split('/').next().unwrap_or_default();
    if host_port.is_empty() {
        None
    } else {
        Some(format!("{scheme}://{host_port}"))
    }
}

async fn fetch_metrics(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.text().await.map_err(|e| e.to_string())
}

/// Concatenate several Prometheus expositions, keeping only the first
/// `# HELP`/`# TYPE` line per metric name.
fn dedupe_metric_docs(bodies: Vec<String>) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = String::new();
    for body in bodies {
        for line in body.lines() {
            let trimmed = line.trim();
            if let Some(doc) = trimmed.strip_prefix("# ") {
                if doc.starts_with("HELP ") || doc.starts_with("TYPE ") {
                    let name = doc.split_whitespace().nth(1).unwrap_or_default();
                    let key = format!("{} {name}", &doc[..4]);
                    if !seen.insert(key) {
                        continue;
                    }
                }
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn text_response(body: String) -> HyperResponse {
    let len = body.len();
    let full = Full::new(Bytes::from(body));
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = hyper::StatusCode::OK;
    resp.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    resp.headers_mut().insert(
        hyper::header::CONTENT_LENGTH,
        hyper::header::HeaderValue::from(len as u64),
    );
    add_cors_headers(resp.headers_mut());
    resp
}
#[cfg(test)]
#[path = "../../tests/server_admin.rs"]
mod tests;
