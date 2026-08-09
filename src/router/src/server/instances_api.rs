//! Public `/instances` management API facade.
//!
//! Coral Router mirrors the llama-server `/instances` contract at its OWN
//! address as the single sidecar entry point: same envelope, same id grammar,
//! same operations - aggregated across every managed model. Public instance ids
//! are `<model_id>:<instance_name>`; per-instance operations accept either
//! `?model=<model_id>` or an id of that form. All management endpoints require
//! the API key when `sidecar.api_key_env` names a variable.
//!
//! Also hosts `/v1/models` (one entry per instance), the `/memory` compat
//! reshape, `/props`, and the model-less llama-server endpoints (`/tokenize`,
//! `/detokenize`, `/apply-template`, `/control`) which run on the pool's
//! default server.
#![allow(clippy::manual_let_else)]

use http_body_util::BodyExt;
use hyper::HeaderMap;

use crate::instances::InstanceTotals;
use crate::server::handler::ServerDeps;
use crate::server::responses::{error_response, json_response, HyperResponse};
use crate::types::RouterRequest;

/// Decode a URL query component (`%XX` and `+`).
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                if let Ok(v) = u8::from_str_radix(
                    std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                    16,
                ) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse a URL query string into key/value pairs (raw order preserved).
pub fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let mut it = pair.splitn(2, '=');
            let key = it.next().unwrap_or_default();
            let value = it.next().unwrap_or_default();
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

/// The single `model` query parameter (or none).
fn query_model(query: &[(String, String)]) -> Option<&str> {
    query
        .iter()
        .find(|(k, _)| k == "model")
        .map(|(_, v)| v.as_str())
}

/// The `?model=` query param or the body's `model` field (body wins per the
/// contract's "body wins over query" rule).
fn body_or_query_model(
    body: Option<&serde_json::Value>,
    query: &[(String, String)],
) -> Option<String> {
    body.and_then(|b| b.get("model"))
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| query_model(query).map(ToOwned::to_owned))
}

/// Validate that `model` names a managed model (or `None` when exactly one
/// managed model exists). Returns the model key.
#[allow(clippy::result_large_err)]
fn resolve_model(deps: &ServerDeps, model: Option<&str>) -> Result<String, HyperResponse> {
    let pool = deps
        .instance_pool
        .as_ref()
        .ok_or_else(|| error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"))?;
    if let Some(m) = model {
        if pool.manager(m).is_none() {
            return Err(error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("unknown model: '{m}'"),
            ));
        }
        return Ok(m.to_string());
    }
    single_model_key(deps).ok_or_else(|| {
        error_response(
            hyper::StatusCode::BAD_REQUEST,
            "cannot route instance management request: specify 'model'",
        )
    })
}

/// Resolve a per-instance operation to `(model, instance)`: the id grammar
/// `<model_id>:<name>` wins; else `?model=` + the bare name; else the single
/// running server.
#[allow(clippy::result_large_err)]
fn resolve_instance_target(
    deps: &ServerDeps,
    path_name: &str,
    query: &[(String, String)],
) -> Result<(String, String), HyperResponse> {
    let pool = deps
        .instance_pool
        .as_ref()
        .ok_or_else(|| error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"))?;
    // Id grammar `<model_id>:<name>`.
    if let Some((model, name)) = pool.resolve_instance_id(path_name) {
        return Ok((model, name));
    }
    let model = resolve_model(deps, query_model(query))?;
    if pool.manager(&model).is_none() {
        return Err(error_response(
            hyper::StatusCode::NOT_FOUND,
            &format!("instance not found: '{path_name}'"),
        ));
    }
    Ok((model, path_name.to_string()))
}

/// The model key when exactly one managed model exists.
fn single_model_key(deps: &ServerDeps) -> Option<String> {
    let pool = deps.instance_pool.as_ref()?;
    let mut keys: Vec<String> = pool
        .managers_iter()
        .map(|m| m.model_key().to_string())
        .collect();
    keys.sort();
    keys.dedup();
    if keys.len() == 1 {
        Some(keys[0].clone())
    } else {
        None
    }
}

/// Enforce the management API key when `sidecar.api_key_env` names a variable.
/// Returns `None` when allowed, or an error response to send.
pub fn check_management_key(deps: &ServerDeps, headers: &HeaderMap) -> Option<HyperResponse> {
    let key = deps
        .api_key_env_name
        .as_deref()
        .and_then(|name| std::env::var(name).ok())?;
    if key.is_empty() {
        return None;
    }
    let provided = headers
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if provided == Some(key.as_str()) {
        None
    } else {
        Some(error_response(hyper::StatusCode::UNAUTHORIZED, "invalid api key"))
    }
}

/// `GET /instances` - the aggregate envelope across every managed model.
pub async fn handle_get_instances(
    deps: &ServerDeps,
    query: &[(String, String)],
) -> HyperResponse {
    let model = query_model(query);
    let Some(pool) = deps.instance_pool.as_ref() else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances");
    };
    if let Some(m) = model {
        if pool.manager(m).is_none() {
            return error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("unknown model: '{m}'"),
            );
        }
    }
    match pool.aggregate(model).await {
        Ok(value) => json_response(hyper::StatusCode::OK, &value),
        Err(e) => error_response(hyper::StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

/// `POST /instances` - allocate a NEW context from a model's shared weights.
pub async fn handle_post_instances(
    req: hyper::Request<hyper::body::Incoming>,
    deps: &ServerDeps,
    query: &[(String, String)],
) -> HyperResponse {
    let body = match read_json_body(req, deps.max_payload).await {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let model = match body_or_query_model(Some(&body), query) {
        Some(m) => m,
        None => match single_model_key(deps) {
            Some(m) => m,
            None => {
                return error_response(
                    hyper::StatusCode::BAD_REQUEST,
                    "cannot route instance management request: specify 'model'",
                )
            }
        },
    };
    let pool = match deps.instance_pool.as_ref() {
        Some(p) => p,
        None => return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"),
    };
    if pool.manager(&model).is_none() {
        return error_response(
            hyper::StatusCode::BAD_REQUEST,
            &format!("unknown model: '{model}'"),
        );
    }
    let name = body
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let group = body
        .get("group")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(name);
    if !crate::instances::is_valid_instance_name(name)
        || !crate::instances::is_valid_instance_name(group)
    {
        return error_response(
            hyper::StatusCode::BAD_REQUEST,
            "invalid instance name/group (allowed: [A-Za-z0-9._-])",
        );
    }
    let ctx_size = body
        .get("ctx_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let parallel = body
        .get("parallel")
        .and_then(serde_json::Value::as_u64)
        .map(|p| p as u32)
        .filter(|p| *p > 0);
    let pinned = body
        .get("pinned")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let is_default = body
        .get("default")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    match pool
        .create(&model, name, group, ctx_size, parallel, pinned, is_default)
        .await
    {
        Ok(info) => json_response(
            hyper::StatusCode::CREATED,
            &serde_json::to_value(info).unwrap_or_default(),
        ),
        Err(e) => instance_error_response(e),
    }
}

/// Per-instance mutating operations: `DELETE /instances/:name`, pin/unpin,
/// resize.
pub async fn handle_instance_op(
    req: hyper::Request<hyper::body::Incoming>,
    deps: &ServerDeps,
    path_name: &str,
    operation: &str,
    query: &[(String, String)],
) -> HyperResponse {
    let (model, instance) = match resolve_instance_target(deps, path_name, query) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let pool = match deps.instance_pool.as_ref() {
        Some(p) => p,
        None => return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"),
    };
    let result = match operation {
        "delete" => pool.destroy(&model, &instance).await,
        "pin" => pool.pin(&model, &instance).await,
        "unpin" => pool.unpin(&model, &instance).await,
        "resize" => {
            let body = match read_json_body(req, deps.max_payload).await {
                Ok(b) => b,
                Err(resp) => return resp,
            };
            let ctx_size = body
                .get("ctx_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            pool.resize(&model, &instance, ctx_size).await
        }
        _ => unreachable!("unknown instance op"),
    };
    instance_op_response(result)
}

/// `POST /instances/:name/snapshot`, `GET /instances/:name/snapshots`,
/// `DELETE /instances/:name/snapshot/:snapshot`.
pub async fn handle_snapshot_op(
    req: hyper::Request<hyper::body::Incoming>,
    deps: &ServerDeps,
    path_name: &str,
    operation: &str,
    snapshot_name: Option<&str>,
    query: &[(String, String)],
) -> HyperResponse {
    let (model, instance) = match resolve_instance_target(deps, path_name, query) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let pool = match deps.instance_pool.as_ref() {
        Some(p) => p,
        None => return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"),
    };
    match operation {
        "save" => {
            let body = match read_json_body(req, deps.max_payload).await {
                Ok(b) => b,
                Err(resp) => return resp,
            };
            let name = body
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if !crate::instances::is_valid_instance_name(name) {
                return error_response(
                    hyper::StatusCode::BAD_REQUEST,
                    "invalid snapshot name (allowed: [A-Za-z0-9._-])",
                );
            }
            instance_op_response(pool.save_snapshot(&model, &instance, name).await)
        }
        "list" => match pool.list_snapshots(&model, &instance).await {
            Ok(snapshots) => json_response(
                hyper::StatusCode::OK,
                &serde_json::json!({ "snapshots": snapshots }),
            ),
            Err(e) => instance_error_response(e),
        },
        "delete" => {
            let Some(name) = snapshot_name else {
                return error_response(hyper::StatusCode::BAD_REQUEST, "missing snapshot name");
            };
            instance_op_response(pool.delete_snapshot(&model, &instance, name).await)
        }
        _ => unreachable!("unknown snapshot op"),
    }
}

/// Dispatch an `/instances/:name[/...]` operation (from `route_instance_resource`)
/// to the instance or snapshot handler.
pub async fn handle_snapshot_op_or_instance_op(
    req: hyper::Request<hyper::body::Incoming>,
    deps: &ServerDeps,
    operation: &str,
    path_name: &str,
    snapshot_name: Option<&str>,
    query: &[(String, String)],
) -> HyperResponse {
    match operation {
        "save" | "list" | "delete_snapshot" => {
            let op = match operation {
                "save" => "save",
                "list" => "list",
                _ => "delete",
            };
            handle_snapshot_op(req, deps, path_name, op, snapshot_name, query).await
        }
        op => handle_instance_op(req, deps, path_name, op, query).await,
    }
}

/// Map a management-client error to the OpenAI-shaped response the contract
/// specifies (409 duplicate, 507/503 evict triggers, 404 unknown).
fn instance_error_response(e: crate::instances::InstanceError) -> HyperResponse {
    match e {
        crate::instances::InstanceError::Duplicate => {
            error_response(hyper::StatusCode::CONFLICT, "duplicate instance name")
        }
        crate::instances::InstanceError::Transient { status: 503, .. } => {
            error_response(hyper::StatusCode::SERVICE_UNAVAILABLE, "transient management error")
        }
        crate::instances::InstanceError::Transient { .. } => error_response(
            hyper::StatusCode::INSUFFICIENT_STORAGE,
            "insufficient device memory",
        ),
        crate::instances::InstanceError::Rejected { status: 404, .. } => {
            error_response(hyper::StatusCode::NOT_FOUND, "instance not found")
        }
        crate::instances::InstanceError::Rejected { status: 400, .. } => {
            error_response(hyper::StatusCode::BAD_REQUEST, "invalid instance operation")
        }
        e => error_response(hyper::StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

/// Common response mapping for a per-instance mutating op.
fn instance_op_response(result: Result<(), crate::instances::InstanceError>) -> HyperResponse {
    match result {
        Ok(()) => json_response(hyper::StatusCode::OK, &serde_json::json!({ "success": true })),
        Err(e) => instance_error_response(e),
    }
}

/// `GET /memory` - compatibility reshape of the aggregate envelope.
pub async fn handle_memory(deps: &ServerDeps) -> HyperResponse {
    let pool = match deps.instance_pool.as_ref() {
        Some(p) => p,
        None => return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances"),
    };
    let value = match pool.aggregate(None).await {
        Ok(v) => v,
        Err(e) => return error_response(hyper::StatusCode::BAD_GATEWAY, &e.to_string()),
    };
    let total: InstanceTotals =
        serde_json::from_value(value.get("total").cloned().unwrap_or_default())
            .unwrap_or_default();
    json_response(
        hyper::StatusCode::OK,
        &serde_json::json!({
            "slots": value.get("instances").cloned().unwrap_or_default(),
            "total": total,
            "object": "memory",
        }),
    )
}

/// `GET /v1/models` / `GET /models` - one entry per instance plus aliases.
pub async fn handle_list_models(deps: &ServerDeps) -> HyperResponse {
    let Some(pool) = deps.instance_pool.as_ref() else {
        // No managed pool: fall back to the configured model keys.
        let created = common_core::now_secs();
        let mut keys: Vec<&String> = deps.models.keys().collect();
        keys.sort();
        let data: Vec<serde_json::Value> = keys
            .iter()
            .map(|k| {
                serde_json::json!({
                    "id": k,
                    "object": "model",
                    "created": created,
                    "owned_by": "coral-router",
                })
            })
            .collect();
        return json_response(
            hyper::StatusCode::OK,
            &serde_json::json!({ "object": "list", "data": data }),
        );
    };
    let data = pool.list_models().await;
    json_response(
        hyper::StatusCode::OK,
        &serde_json::json!({ "object": "list", "data": data }),
    )
}

/// `GET /props` - proxy the pool's default server's props, adding `total_slots`
/// and an `instances` array.
pub async fn handle_props(deps: &ServerDeps) -> HyperResponse {
    let Some(pool) = deps.instance_pool.as_ref() else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances");
    };
    let Some(model) = single_model_key(deps) else {
        return error_response(
            hyper::StatusCode::BAD_REQUEST,
            "cannot route model-less request: multiple managed models",
        );
    };
    let Some(base) = pool.manager(&model).map(|m| m.client().base_url().to_string()) else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances");
    };
    let props_url = format!("{base}/props");
    let client = reqwest::Client::new();
    let response = match client.get(&props_url).send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(v) => v,
            Err(_) => serde_json::json!({}),
        },
        _ => serde_json::json!({}),
    };
    let envelope = match pool.aggregate(None).await {
        Ok(v) => v,
        Err(_) => serde_json::json!({}),
    };
    let instances = envelope
        .get("instances")
        .cloned()
        .unwrap_or_else(|| serde_json::json!([]));
    let mut props = response;
    if let serde_json::Value::Object(ref mut obj) = props {
        let total_slots: u64 = instances
            .as_array()
            .map_or(0, |arr| {
                arr.iter()
                    .filter_map(|i| i.get("parallel").and_then(serde_json::Value::as_u64))
                    .sum()
            });
        obj.insert("total_slots".into(), serde_json::json!(total_slots));
        obj.insert("instances".into(), instances);
    }
    json_response(hyper::StatusCode::OK, &props)
}

/// Proxy a model-less llama-server endpoint (`/tokenize`, `/detokenize`,
/// `/apply-template`, `/control`) to the pool's default server, forwarding
/// method, query, and body verbatim.
pub async fn handle_model_less_proxy(
    req: hyper::Request<hyper::body::Incoming>,
    deps: &ServerDeps,
    path: &str,
) -> HyperResponse {
    let Some(pool) = deps.instance_pool.as_ref() else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances");
    };
    let Some(model) = single_model_key(deps) else {
        return error_response(
            hyper::StatusCode::BAD_REQUEST,
            "cannot route model-less request: multiple managed models",
        );
    };
    let Some(base) = pool.manager(&model).map(|m| m.client().base_url().to_string()) else {
        return error_response(hyper::StatusCode::NOT_FOUND, "no managed instances");
    };
    let method = req.method().clone();
    let query = req.uri().query().unwrap_or_default().to_string();
    let body_bytes = match req.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => bytes::Bytes::new(),
    };
    let target = format!("{base}{path}?{query}");
    let client = reqwest::Client::new();
    let builder = client.request(method, &target).body(body_bytes.to_vec());
    match builder.send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.bytes().await.unwrap_or_default();
            let mut out = hyper::Response::new(http_body_util::Full::new(body).boxed_unsync());
            *out.status_mut() = status;
            crate::server::responses::add_cors_headers(out.headers_mut());
            out
        }
        Err(e) => error_response(hyper::StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

/// Read and JSON-parse a request body, enforcing the max payload.
pub(crate) async fn read_json_body(
    req: hyper::Request<hyper::body::Incoming>,
    max_payload: usize,
) -> Result<serde_json::Value, HyperResponse> {
    let body_bytes = match req.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            return Err(error_response(
                hyper::StatusCode::BAD_REQUEST,
                &format!("body read error: {e}"),
            ))
        }
    };
    if body_bytes.len() > max_payload {
        return Err(crate::server::responses::empty_response(
            hyper::StatusCode::PAYLOAD_TOO_LARGE,
        ));
    }
    if body_bytes.is_empty() {
        return Ok(serde_json::json!({}));
    }
    match serde_json::from_slice(&body_bytes) {
        Ok(v) => Ok(v),
        Err(e) => Err(error_response(
            hyper::StatusCode::BAD_REQUEST,
            &format!("invalid JSON: {e}"),
        )),
    }
}

/// Build the `RoutingTarget` overlay for a request's explicit routing fields:
/// the client's `instance`/`snapshot`/`id_slot` win over any the target
/// derived from config.
pub(crate) fn apply_request_routing_fields(
    target: &crate::pipeline::RoutingTarget,
    request: &RouterRequest,
) -> crate::pipeline::RoutingTarget {
    let mut t = target.clone();
    if let Some(instance) = &request.instance {
        t.instance = Some(instance.clone());
    }
    if let Some(snapshot) = &request.snapshot {
        t.snapshot = Some(snapshot.clone());
    }
    if let Some(id_slot) = request.id_slot {
        t.id_slot = Some(id_slot);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_decodes_and_splits() {
        let pairs = parse_query("model=swarm&instance=ledger%3A0&id_slot=2&blank=");
        let map: std::collections::HashMap<&str, &str> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(map.get("model"), Some(&"swarm"));
        assert_eq!(map.get("instance"), Some(&"ledger:0"));
        assert_eq!(map.get("id_slot"), Some(&"2"));
        assert_eq!(map.get("blank"), Some(&""));
    }

    #[test]
    fn percent_decode_handles_plus_and_hex() {
        assert_eq!(percent_decode("a+b"), "a b");
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("%2F"), "/");
        assert_eq!(percent_decode("plain"), "plain");
    }

    #[test]
    fn apply_request_routing_overrides_target_fields() {
        let target = crate::pipeline::RoutingTarget {
            url: "http://x/v1/chat/completions".into(),
            model: "base:ledger".into(),
            group: None,
            target_name: None,
            params: None,
            instance: Some("ledger".into()),
            snapshot: None,
            id_slot: None,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            stream: true,
            idle_timeout_ms: 5000,
            total_timeout_ms: 30000,
            fallbacks: vec![],
        };
        let request = RouterRequest {
            model: "base".into(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            instance: Some("scratch".into()),
            snapshot: Some("readfiles".into()),
            id_slot: Some(3),
            metadata: Default::default(),
        };
        let overlaid = apply_request_routing_fields(&target, &request);
        assert_eq!(overlaid.instance.as_deref(), Some("scratch"));
        assert_eq!(overlaid.snapshot.as_deref(), Some("readfiles"));
        assert_eq!(overlaid.id_slot, Some(3));
    }
}
