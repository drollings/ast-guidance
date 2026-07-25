/// HTTP status classification for retry decisions — MOA_ROUTER_SPEC §3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpClass {
    /// 422 Unprocessable Entity — filter/policy rejection. Never retried.
    HardReject,
    /// 503 Service Unavailable — transient. Eligible for retry_count/backoff.
    TransientFailure,
    /// 409 Conflict — internal escalation signal. Not retried.
    EscalationRequired,
    /// Provider's own 4xx/5xx — passthrough. Treated as non-retryable unless
    /// the provider code indicates transient (e.g., 429 Rate Limit).
    UpstreamFailure,
}

impl HttpClass {
    pub fn from_status(status: u16) -> Self {
        match status {
            422 => Self::HardReject,
            409 => Self::EscalationRequired,
            429 => Self::TransientFailure,
            500 | 502 | 503 | 504 => Self::TransientFailure,
            400 | 401 | 403 | 404 | 405 | 410 | 413 | 414 => Self::HardReject,
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
        assert_eq!(HttpClass::from_status(500), HttpClass::TransientFailure);
        assert_eq!(HttpClass::from_status(502), HttpClass::TransientFailure);
        assert_eq!(HttpClass::from_status(503), HttpClass::TransientFailure);
        assert_eq!(HttpClass::from_status(504), HttpClass::TransientFailure);
    }

    #[test]
    fn client_errors_are_hard_reject() {
        assert_eq!(HttpClass::from_status(400), HttpClass::HardReject);
        assert_eq!(HttpClass::from_status(401), HttpClass::HardReject);
        assert_eq!(HttpClass::from_status(403), HttpClass::HardReject);
        assert_eq!(HttpClass::from_status(404), HttpClass::HardReject);
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
