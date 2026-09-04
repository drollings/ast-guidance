//! Request and response normalization — OpenAI-compatible JSON → RouterRequest
//! and RouterResponse → OpenAI-compatible JSON.
//!
//! Thin adapter over the canonical protocol-boundary helpers in
//! `fluent_llm::openai`. The shared helpers operate on `serde_json::Value`;
//! this module adapts them to the router's typed `RouterRequest`/`RouterResponse`.

use crate::types::{RouterRequest, RouterResponse};

pub use fluent_llm::openai::NormalizeError;

/// Normalize an OpenAI-compatible chat completion request JSON into a RouterRequest.
///
/// The shared `fluent_llm` normalizer strips non-OpenAI fields; the routing
/// fields the owning llama-server reads (`instance`/`snapshot`/`id_slot`) are
/// re-attached from the original body so they survive into the dispatch target.
/// `num_ctx` (the caller's declared context window) is preserved into
/// `metadata["num_ctx"]` so the dispatch path's resize-to-demand can size a
/// targeted context to the request's needs (ROADMAP M7).
pub fn normalize_request(body: serde_json::Value) -> Result<RouterRequest, NormalizeError> {
    let num_ctx = body
        .get("num_ctx")
        .and_then(serde_json::Value::as_u64);
    // Validate instance/snapshot grammar via the canonical primitive before
    // preserving them into the typed request.
    if let Some(v) = body.get("instance").and_then(|v| v.as_str()) {
        if !fluent_types::instance_id::is_valid_instance_name(v) {
            return Err(NormalizeError::Parse(format!(
                "invalid instance name '{v}' (allowed: [A-Za-z0-9._-])"
            )));
        }
    }
    if let Some(v) = body.get("snapshot").and_then(|v| v.as_str()) {
        if !fluent_types::instance_id::is_valid_instance_name(v) {
            return Err(NormalizeError::Parse(format!(
                "invalid snapshot name '{v}' (allowed: [A-Za-z0-9._-])"
            )));
        }
    }
    let routing: Vec<(String, serde_json::Value)> = ["instance", "snapshot", "id_slot"]
        .iter()
        .filter_map(|key| body.get(key).map(|v| ((*key).to_string(), v.clone())))
        .collect();
    let mut value = fluent_llm::openai::normalize_request(body)?;
    if let serde_json::Value::Object(ref mut obj) = value {
        for (key, value) in routing {
            obj.insert(key, value);
        }
    }
    let mut request: RouterRequest =
        serde_json::from_value(value).map_err(|e| NormalizeError::Parse(e.to_string()))?;
    if let Some(n) = num_ctx {
        request
            .metadata
            .insert("num_ctx".into(), serde_json::json!(n));
    }
    Ok(request)
}

/// Normalize a RouterResponse to OpenAI-compatible chat completion JSON.
pub fn normalize_response(response: &RouterResponse) -> serde_json::Value {
    let value = serde_json::to_value(response).unwrap_or(serde_json::Value::Null);
    fluent_llm::openai::normalize_response(&value)
}

/// Build an error response in OpenAI-compatible format.
pub fn error_response(message: &str, error_type: &str) -> serde_json::Value {
    fluent_llm::openai::error_response(message, error_type)
}

/// Convert RouterRequest messages into Vec<serde_json::Value> for dispatch.
/// Used by both server.rs dispatch_to_llm and dispatch::frontier backends.
pub fn messages_to_json(request: &RouterRequest) -> Result<Vec<serde_json::Value>, NormalizeError> {
    let messages = serde_json::to_value(&request.messages)
        .map_err(|e| NormalizeError::Parse(e.to_string()))?;
    let arr = messages.as_array().cloned().unwrap_or_default();
    fluent_llm::openai::messages_to_json(&arr)
}

// Keep the shared error-free helpers importable through this module for the
// small set of callers that reach into `normalize` for the wire helpers.
pub use fluent_llm::openai::{parse_openai_stream_delta, OpenAiDelta};

#[cfg(test)]
#[path = "../tests/normalize.rs"]
mod tests;
