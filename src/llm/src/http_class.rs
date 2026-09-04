/// HTTP status classification for LLM API retry decisions.
///
/// Consumed by `LlmClient` to produce the correct `LlmError` variant, and by
/// callers that talk directly to LLM endpoints via raw `reqwest` (e.g. the
/// coral-router dispatch path) for their own retry loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpClass {
    /// 400-405, 410, 413-414, 422 — permanent rejection. Never retried.
    HardReject,
    /// 429, 500, 502-504 — transient. Eligible for backoff + retry.
    TransientFailure,
    /// 409 Conflict — internal escalation signal (routers / multi-hop).
    /// Not retried.
    EscalationRequired,
    /// Unknown 5xx or non-LLM status (including 200). Treat as
    /// non-retryable unless the caller knows better.
    UpstreamFailure,
}

impl HttpClass {
    pub fn from_status(status: u16) -> Self {
        match status {
            409 => Self::EscalationRequired,
            429 | 500 | 502 | 503 | 504 => Self::TransientFailure,
            400 | 401 | 403 | 404 | 405 | 410 | 413 | 414 | 422 => Self::HardReject,
            _ if status >= 500 => Self::UpstreamFailure,
            _ => Self::UpstreamFailure,
        }
    }

    pub fn is_retryable(self) -> bool {
        matches!(self, Self::TransientFailure)
    }
}

/// High-level failure classes for error classification.
///
/// Every variant serializes to a stable string label for backward
/// compatibility with existing metrics consumers. This is the canonical
/// workspace taxonomy (D2): `HttpClass` drives retry decisions, while
/// `FailureClass` drives metrics/telemetry classification.
///
/// Single definition owned by `common_core::telemetry` (the PII-free
/// observability contract); re-exported here because `fluent-llm` depends on
/// `common-core` and the dependency cannot run the other way. There must be
/// exactly one enum definition — do not reintroduce a duplicate.
pub use common_core::telemetry::FailureClass;

/// Coarse `HttpClass` → `FailureClass` mapping for callers that hold an
/// `HttpClass` without the raw status code. Retryable transient failures map
/// to `RateLimit`; permanent rejections to `InputValidation`; escalation and
/// upstream failures to `Network`.
impl From<HttpClass> for FailureClass {
    fn from(class: HttpClass) -> Self {
        match class {
            HttpClass::HardReject => FailureClass::InputValidation,
            HttpClass::TransientFailure => FailureClass::RateLimit,
            HttpClass::EscalationRequired | HttpClass::UpstreamFailure => FailureClass::Network,
        }
    }
}

/// Classify a raw HTTP status into a `FailureClass`, preserving the
/// fine-grained distinctions metrics consumers rely on (authentication,
/// input-validation, timeout, network). Expressed via `HttpClass` so there is
/// a single status-taxonomy in the workspace; only the fine-grained splits
/// that `HttpClass` intentionally merges are re-derived here.
pub fn classify_http_status(status: u16) -> FailureClass {
    match HttpClass::from_status(status) {
        HttpClass::TransientFailure => FailureClass::RateLimit,
        HttpClass::HardReject if matches!(status, 401 | 403) => FailureClass::Authentication,
        HttpClass::HardReject => FailureClass::InputValidation,
        _ if status == 408 => FailureClass::Timeout,
        _ => FailureClass::Network,
    }
}

#[cfg(test)]
#[path = "../tests/http_class.rs"]
mod tests;
