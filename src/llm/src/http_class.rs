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
}
