use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Network,
    Authentication,
    RateLimit,
    InputValidation,
    Storage,
    Timeout,
    Internal,
    Unknown,
}

impl FailureClass {
    /// Stable string label for the failure class.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Authentication => "authentication",
            Self::RateLimit => "rate_limit",
            Self::InputValidation => "input_validation",
            Self::Storage => "storage",
            Self::Timeout => "timeout",
            Self::Internal => "internal",
            Self::Unknown => "unknown",
        }
    }
}

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
mod tests {
    use super::*;

    #[test]
    fn hard_reject_422() {
        assert_eq!(HttpClass::from_status(422), HttpClass::HardReject);
        assert!(!HttpClass::HardReject.is_retryable());
    }

    #[test]
    fn escalation_409() {
        assert_eq!(HttpClass::from_status(409), HttpClass::EscalationRequired);
        assert!(!HttpClass::EscalationRequired.is_retryable());
    }

    #[test]
    fn rate_limit_429_is_transient() {
        assert_eq!(HttpClass::from_status(429), HttpClass::TransientFailure);
        assert!(HttpClass::TransientFailure.is_retryable());
    }

    #[test]
    fn server_errors_are_transient() {
        for status in [500, 502, 503, 504] {
            assert_eq!(HttpClass::from_status(status), HttpClass::TransientFailure);
        }
    }

    #[test]
    fn client_errors_are_hard_reject() {
        for status in [400, 401, 403, 404] {
            assert_eq!(HttpClass::from_status(status), HttpClass::HardReject);
        }
    }

    #[test]
    fn unknown_5xx_is_upstream() {
        assert_eq!(HttpClass::from_status(501), HttpClass::UpstreamFailure);
        assert_eq!(HttpClass::from_status(505), HttpClass::UpstreamFailure);
        assert!(!HttpClass::UpstreamFailure.is_retryable());
    }

    #[test]
    fn success_is_upstream() {
        assert_eq!(HttpClass::from_status(200), HttpClass::UpstreamFailure);
        assert_eq!(HttpClass::from_status(201), HttpClass::UpstreamFailure);
    }

    // ── FailureClass labels ─────────────────────────────────────────

    #[test]
    fn failure_class_labels() {
        assert_eq!(FailureClass::Network.label(), "network");
        assert_eq!(FailureClass::Authentication.label(), "authentication");
        assert_eq!(FailureClass::RateLimit.label(), "rate_limit");
        assert_eq!(FailureClass::InputValidation.label(), "input_validation");
        assert_eq!(FailureClass::Storage.label(), "storage");
        assert_eq!(FailureClass::Timeout.label(), "timeout");
        assert_eq!(FailureClass::Internal.label(), "internal");
        assert_eq!(FailureClass::Unknown.label(), "unknown");
    }

    #[test]
    fn failure_class_round_trips_through_json() {
        for class in &[
            FailureClass::Network,
            FailureClass::Authentication,
            FailureClass::RateLimit,
            FailureClass::InputValidation,
            FailureClass::Storage,
            FailureClass::Timeout,
            FailureClass::Internal,
            FailureClass::Unknown,
        ] {
            let json = serde_json::to_string(class).unwrap();
            let parsed: FailureClass = serde_json::from_str(&json).unwrap();
            assert_eq!(*class, parsed);
        }
    }

    // ── classify_http_status ────────────────────────────────────────

    #[test]
    fn transient_statuses_are_rate_limit() {
        for status in [429, 500, 502, 503, 504] {
            assert_eq!(
                classify_http_status(status),
                FailureClass::RateLimit,
                "status {status}"
            );
        }
    }

    #[test]
    fn auth_statuses_are_authentication() {
        for status in [401, 403] {
            assert_eq!(
                classify_http_status(status),
                FailureClass::Authentication,
                "status {status}"
            );
        }
    }

    #[test]
    fn validation_statuses_are_input_validation() {
        for status in [400, 404, 405, 410, 413, 414, 422] {
            assert_eq!(
                classify_http_status(status),
                FailureClass::InputValidation,
                "status {status}"
            );
        }
    }

    #[test]
    fn request_timeout_is_timeout() {
        assert_eq!(classify_http_status(408), FailureClass::Timeout);
    }

    #[test]
    fn other_statuses_are_network() {
        for status in [200, 201, 409, 418, 501, 505, 599] {
            assert_eq!(
                classify_http_status(status),
                FailureClass::Network,
                "status {status}"
            );
        }
    }

    // ── From<HttpClass> ─────────────────────────────────────────────

    #[test]
    fn http_class_to_failure_class() {
        assert_eq!(
            FailureClass::from(HttpClass::HardReject),
            FailureClass::InputValidation
        );
        assert_eq!(
            FailureClass::from(HttpClass::TransientFailure),
            FailureClass::RateLimit
        );
        assert_eq!(
            FailureClass::from(HttpClass::EscalationRequired),
            FailureClass::Network
        );
        assert_eq!(
            FailureClass::from(HttpClass::UpstreamFailure),
            FailureClass::Network
        );
    }
}
