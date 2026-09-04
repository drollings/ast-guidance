use std::sync::atomic::AtomicU64;

use bytes::Bytes;
use http_body_util::BodyExt;
use http_body_util::Full;

use crate::normalize;
use crate::streaming::StreamingHandler;
use crate::types::RouterResponse;
#[cfg(test)]
use crate::types::{RouterMessageContent, Usage};

pub type ResponseBody = http_body_util::combinators::UnsyncBoxBody<Bytes, std::convert::Infallible>;
pub type HyperResponse = hyper::Response<ResponseBody>;



pub struct ServerStats {
    pub requests: AtomicU64,
    pub errors: AtomicU64,
    pub rejections: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl ServerStats {
    pub fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }
}

impl Default for ServerStats {
    fn default() -> Self {
        Self::new()
    }
}

pub use crate::server::cors::{add_cors_headers, CORS_ALLOW_HEADERS, CORS_ALLOW_METHODS, CORS_ALLOW_ORIGIN};

pub fn completion_to_response(
    completion: &RouterResponse,
    model_name: &str,
    is_stream: bool,
    actual_model: Option<&str>,
) -> HyperResponse {
    let body_str = if is_stream {
        let mut handler = StreamingHandler::new(&completion.id, actual_model.unwrap_or(model_name));
        let mut s = String::new();
        if let Some(choice) = completion.choices.first() {
            s.push_str(&handler.format_choice_chunk(choice));
        }
        s.push_str(&handler.format_done());
        s
    } else {
        serde_json::to_string(&normalize::normalize_response(completion)).unwrap_or_default()
    };

    let content_type = if is_stream {
        "text/event-stream"
    } else {
        "application/json"
    };

    let len = body_str.len();
    let full = Full::new(Bytes::from(body_str));
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = hyper::StatusCode::OK;
    resp.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static(content_type),
    );
    resp.headers_mut().insert(
        hyper::header::CONTENT_LENGTH,
        hyper::header::HeaderValue::from(len as u64),
    );
    add_cors_headers(resp.headers_mut());
    resp
}

pub fn json_response(status: hyper::StatusCode, value: &serde_json::Value) -> HyperResponse {
    let body_str = common_core::http::json_body(value);
    let len = body_str.len();
    let full = Full::new(Bytes::from(body_str));
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );
    resp.headers_mut().insert(
        hyper::header::CONTENT_LENGTH,
        hyper::header::HeaderValue::from(len as u64),
    );
    add_cors_headers(resp.headers_mut());
    resp
}

pub fn error_response(status: hyper::StatusCode, message: &str) -> HyperResponse {
    let err = fluent_llm::openai::error_response(message, "invalid_request_error");
    json_response(status, &err)
}

pub fn empty_response(status: hyper::StatusCode) -> HyperResponse {
    let full = Full::new(Bytes::new());
    let mut resp = HyperResponse::new(full.boxed_unsync());
    *resp.status_mut() = status;
    add_cors_headers(resp.headers_mut());
    resp
}

pub fn forbidden_response() -> HyperResponse {
    error_response(
        hyper::StatusCode::FORBIDDEN,
        "admin endpoints are localhost-only",
    )
}

pub fn fallback_completion(model_name: &str) -> RouterResponse {
    RouterResponse::fallback(model_name)
}

/// The assistant's answer text from a completion (first choice), or `None`
/// when the response carries no choices.
///
/// The single extraction used by the dispatch path (workflow
/// extractor, `server/dispatch.rs`) and by the handler when it records the
/// matched target's answer into the ledger + session step.
pub fn answer_text(completion: &RouterResponse) -> Option<String> {
    completion.answer_text()
}

pub fn make_error_completion(model_name: &str, error: &str) -> RouterResponse {
    RouterResponse::error(model_name, error)
}

pub fn make_text_completion(model_name: &str, text: &str) -> RouterResponse {
    RouterResponse::text(model_name, text)
}
#[cfg(test)]
#[path = "../../tests/server_responses.rs"]
mod tests;
