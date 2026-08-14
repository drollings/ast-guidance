use std::sync::atomic::AtomicU64;

use bytes::Bytes;
use common_core::now_secs;
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::HeaderMap;

use crate::normalize;
use crate::streaming::StreamingHandler;
use crate::types::{RouterChoice, RouterMessage, RouterMessageContent, RouterResponse, Usage};

pub type ResponseBody = http_body_util::combinators::UnsyncBoxBody<Bytes, std::convert::Infallible>;
pub type HyperResponse = hyper::Response<ResponseBody>;

const CORS_HEADERS: &[(&str, &str)] = &[
    ("access-control-allow-origin", "*"),
    ("access-control-allow-methods", "POST, GET, OPTIONS"),
    (
        "access-control-allow-headers",
        "Content-Type, Authorization",
    ),
];

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

pub fn add_cors_headers(headers: &mut HeaderMap) {
    for (name, value) in CORS_HEADERS {
        headers.insert(
            hyper::header::HeaderName::from_static(name),
            hyper::header::HeaderValue::from_static(value),
        );
    }
}

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
    let body_str = serde_json::to_string(value).unwrap_or_default();
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
    let err = normalize::error_response(message, "invalid_request_error");
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
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: now_secs(),
        model: model_name.to_string(),
        choices: vec![RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text("pipeline completed successfully".into()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage::default(),
    }
}

/// The assistant's answer text from a completion (first choice), or `None`
/// when the response carries no choices.
///
/// The single extraction used by the dispatch path (workflow
/// extractor, `server/dispatch.rs`) and by the handler when it records the
/// matched target's answer into the ledger + session step.
pub fn answer_text(completion: &RouterResponse) -> Option<String> {
    completion
        .choices
        .first()
        .map(|c| c.message.content.to_string_lossy())
}

pub fn make_error_completion(model_name: &str, error: &str) -> RouterResponse {
    make_text_completion(model_name, &format!("ERROR: {error}"))
}

pub fn make_text_completion(model_name: &str, text: &str) -> RouterResponse {
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: now_secs(),
        model: model_name.to_string(),
        choices: vec![RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text(text.to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_is_formatted_correctly() {
        let id = common_core::hash::uuid_v4();
        assert_eq!(id.len(), 36);
        assert_eq!(id.chars().filter(|&c| c == '-').count(), 4);
    }

    #[test]
    fn fallback_has_stop_reason() {
        let r = fallback_completion("test");
        assert_eq!(r.choices.len(), 1);
        assert_eq!(r.choices[0].finish_reason, "stop");
    }

    #[test]
    fn answer_text_extracts_first_choice() {
        let c = make_text_completion("fast", "the answer");
        assert_eq!(answer_text(&c).as_deref(), Some("the answer"));
    }

    #[test]
    fn answer_text_is_none_without_choices() {
        let c = RouterResponse {
            id: String::new(),
            object: "chat.completion".into(),
            created: 0,
            model: "fast".into(),
            choices: vec![],
            usage: Usage::default(),
        };
        assert_eq!(answer_text(&c), None);
    }

    #[test]
    fn answer_text_concatenates_text_parts() {
        let mut c = make_text_completion("fast", "ignored");
        c.choices[0].message.content = RouterMessageContent::Parts(vec![
            crate::types::ContentPart::Text { text: "hello".into() },
            crate::types::ContentPart::Text { text: "world".into() },
        ]);
        assert_eq!(answer_text(&c).as_deref(), Some("hello world"));
    }
}
