//! Failure classification for the router pipeline.
//!
//! Typed-first: `FailureClass` is derived from the typed error
//! (`DispatchError`/`WorkError`/`ServerError` via `From`), with the string
//! regex classifier retained only as a fallback for opaque payloads (shell
//! output, compiler diagnostics like `[E0425]`).

use std::sync::LazyLock;

use fluent_wvr::WorkError;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::dispatch::frontier::DispatchError;
use crate::error::ServerError;

// ── FailureClass ───────────────────────────────────────────────────────

/// High-level failure classes for error classification.
///
/// Every variant serializes to a stable string label for backward
/// compatibility with existing metrics consumers.
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

/// Classify an error message string into a `FailureClass`.
///
/// Uses regex patterns ordered by specificity, with a fast prefix
/// check for compiler diagnostic codes as an optimization.
pub fn classify_error(message: &str) -> FailureClass {
    if message.is_empty() {
        return FailureClass::Unknown;
    }

    // Fast prefix check for compiler diagnostic codes (e.g. [E0425])
    if message.starts_with('[') {
        if let Some(end) = message.find(']') {
            let code = &message[1..end];
            if code.len() >= 3 && code.chars().all(char::is_alphanumeric) {
                return FailureClass::Internal;
            }
        }
    }

    static PATTERNS: LazyLock<Vec<(Regex, FailureClass)>> = LazyLock::new(|| {
        vec![
            (Regex::new(r"(?i)\b(?:timeout|timed?\s*out|deadline exceeded|deadline_exceeded)\b").unwrap(), FailureClass::Timeout),
            (Regex::new(r"(?i)\b(?:rate.?limit|too many requests|429|503|throttle)\b").unwrap(), FailureClass::RateLimit),
            (Regex::new(r"(?i)\b(?:auth|unauthorized|forbidden|401|403|invalid.*(?:key|token|credential)|access denied)\b").unwrap(), FailureClass::Authentication),
            (Regex::new(r"(?i)\b(?:network|connection.*(?:reset|refused|refuse|timeout|closed)|dns.*(?:fail|error)|tcp|socket)\b").unwrap(), FailureClass::Network),
            (Regex::new(r"(?i)\b(?:io error|file.*not.*found|no such file|disk.*full|read.?only|permission.*denied)\b").unwrap(), FailureClass::Storage),
            (Regex::new(r"(?i)\b(?:parse.*error|invalid.*(?:input|format|argument|parameter|request|json)|malformed|bad request|422|400)\b").unwrap(), FailureClass::InputValidation),
            (Regex::new(r"(?i)\b(?:syntax error|type error|compiler? error|compilation failed|build.*(?:fail|error)|panic|internal error)\b").unwrap(), FailureClass::Internal),
            (Regex::new(r"(?i)\b(?:test.*(?:fail|error)|assert.*failed|assertion.*failed)\b").unwrap(), FailureClass::Internal),
            (Regex::new(r"(?i)\b(?:not found|404)\b").unwrap(), FailureClass::InputValidation),
            (Regex::new(r"(?i)\b(?:runtime error|segfault|segmentation fault|abort|fatal)\b").unwrap(), FailureClass::Internal),
        ]
    });

    for (re, class) in PATTERNS.iter() {
        if re.is_match(message) {
            return *class;
        }
    }

    FailureClass::Unknown
}

/// Classify a `DispatchError` typed-first, falling back to the string
/// classifier for opaque payloads (e.g. reqwest error strings carried in
/// `Http`).
impl From<&DispatchError> for FailureClass {
    fn from(err: &DispatchError) -> Self {
        match err {
            DispatchError::RateLimited => FailureClass::RateLimit,
            DispatchError::Http(msg) => classify_http_status(msg),
            DispatchError::RequestBuild(_)
            | DispatchError::ResponseParse(_)
            | DispatchError::StreamParse(_) => FailureClass::InputValidation,
            DispatchError::AllBackendsFailed | DispatchError::UnsupportedProvider(_) => {
                FailureClass::Unknown
            }
        }
    }
}

/// Classify a `WorkError` typed-first. `Execution` carries an opaque
/// shell/command string — the documented regex fallback; `Dependency` and
/// `Timeout` are typed directly.
impl From<&WorkError> for FailureClass {
    fn from(err: &WorkError) -> Self {
        match err {
            WorkError::Execution(msg) => classify_error(msg),
            WorkError::Dependency(_) => FailureClass::Internal,
            WorkError::Timeout { .. } => FailureClass::Timeout,
        }
    }
}

/// Classify a `ServerError` typed-first. `Dispatch` delegates to the
/// `DispatchError` mapping; `Http`/`Bind` reuse the status/string logic.
impl From<&ServerError> for FailureClass {
    fn from(err: &ServerError) -> Self {
        match err {
            ServerError::Dispatch(e) => FailureClass::from(e),
            ServerError::Http(msg) => classify_http_status(msg),
            ServerError::Bind { .. } => FailureClass::Network,
            ServerError::Addr(_) => FailureClass::InputValidation,
            ServerError::FrontierNotImplemented(_) => FailureClass::Unknown,
        }
    }
}

/// Extract an HTTP status from a `DispatchError::Http`/`ServerError::Http`
/// message of the form `"HTTP <code>..."` and map it typed-first. Free-form
/// strings (reqwest errors, timeout labels) fall back to `classify_error`.
fn classify_http_status(msg: &str) -> FailureClass {
    let status = msg
        .strip_prefix("HTTP ")
        .and_then(|rest| rest.split([' ', ':']).next())
        .and_then(|s| s.parse::<u16>().ok());
    match status {
        Some(429 | 500 | 502 | 503 | 504) => FailureClass::RateLimit,
        Some(401 | 403) => FailureClass::Authentication,
        Some(400 | 404 | 405 | 410 | 413 | 414 | 422) => FailureClass::InputValidation,
        Some(408) => FailureClass::Timeout,
        Some(_) => FailureClass::Network,
        None => classify_error(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 4.1 FailureClass label round-trip ───────────────────────────

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

    // ── 4.2 Error classification ────────────────────────────────────

    #[test]
    fn classify_type_error() {
        assert_eq!(
            classify_error("type error: expected i32, found String"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_syntax_error() {
        assert_eq!(
            classify_error("syntax error near unexpected token"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_test_failure() {
        assert_eq!(
            classify_error("test failed: assertion failed at line 42"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_build_failure() {
        assert_eq!(
            classify_error("build failed: compilation error"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_permission_denied() {
        assert_eq!(
            classify_error("permission denied: cannot open file"),
            FailureClass::Storage
        );
    }

    #[test]
    fn classify_timeout() {
        assert_eq!(
            classify_error("timeout: request took too long"),
            FailureClass::Timeout
        );
    }

    #[test]
    fn classify_not_found() {
        assert_eq!(
            classify_error("not found: resource does not exist"),
            FailureClass::InputValidation
        );
    }

    #[test]
    fn classify_runtime_error() {
        assert_eq!(
            classify_error("runtime error: segmentation fault"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_unknown_gibberish() {
        assert_eq!(classify_error("xyzzy flobnob glup"), FailureClass::Unknown);
    }

    #[test]
    fn classify_compiler_diagnostic_code() {
        assert_eq!(
            classify_error("[E0425] cannot find value `x` in this scope"),
            FailureClass::Internal
        );
    }

    #[test]
    fn classify_rate_limit_429() {
        assert_eq!(
            classify_error("429 Too Many Requests"),
            FailureClass::RateLimit
        );
    }

    #[test]
    fn classify_auth_401() {
        assert_eq!(
            classify_error("401 Unauthorized"),
            FailureClass::Authentication
        );
    }

    #[test]
    fn classify_network_refused() {
        assert_eq!(
            classify_error("connection refused (os error 111)"),
            FailureClass::Network
        );
    }

    #[test]
    fn classify_empty_returns_unknown() {
        assert_eq!(classify_error(""), FailureClass::Unknown);
    }

    // ── Typed-first mapping (D10) ──────────────────────────────────────

    #[test]
    fn dispatch_rate_limited_maps_to_rate_limit() {
        assert_eq!(
            FailureClass::from(&DispatchError::RateLimited),
            FailureClass::RateLimit
        );
    }

    #[test]
    fn dispatch_http_status_maps_by_status_code() {
        assert_eq!(
            FailureClass::from(&DispatchError::Http("HTTP 429".into())),
            FailureClass::RateLimit
        );
        assert_eq!(
            FailureClass::from(&DispatchError::Http("HTTP 503".into())),
            FailureClass::RateLimit
        );
        assert_eq!(
            FailureClass::from(&DispatchError::Http("HTTP 401".into())),
            FailureClass::Authentication
        );
        assert_eq!(
            FailureClass::from(&DispatchError::Http("HTTP 422".into())),
            FailureClass::InputValidation
        );
        assert_eq!(
            FailureClass::from(&DispatchError::Http("HTTP 408".into())),
            FailureClass::Timeout
        );
    }

    #[test]
    fn dispatch_http_free_string_falls_back_to_regex() {
        assert_eq!(
            FailureClass::from(&DispatchError::Http("connection reset by peer".into())),
            FailureClass::Network
        );
        assert_eq!(
            FailureClass::from(&DispatchError::Http("total timeout exceeded".into())),
            FailureClass::Timeout
        );
    }

    #[test]
    fn dispatch_build_and_parse_errors_are_input_validation() {
        assert_eq!(
            FailureClass::from(&DispatchError::RequestBuild("bad".into())),
            FailureClass::InputValidation
        );
        assert_eq!(
            FailureClass::from(&DispatchError::ResponseParse("bad".into())),
            FailureClass::InputValidation
        );
        assert_eq!(
            FailureClass::from(&DispatchError::StreamParse("bad".into())),
            FailureClass::InputValidation
        );
    }

    #[test]
    fn dispatch_misc_maps_to_unknown() {
        assert_eq!(
            FailureClass::from(&DispatchError::AllBackendsFailed),
            FailureClass::Unknown
        );
        assert_eq!(
            FailureClass::from(&DispatchError::UnsupportedProvider("x".into())),
            FailureClass::Unknown
        );
    }

    #[test]
    fn work_timeout_maps_directly() {
        assert_eq!(
            FailureClass::from(&WorkError::Timeout {
                duration_ms: 30_000,
                unit: "x".into(),
            }),
            FailureClass::Timeout
        );
    }

    #[test]
    fn work_execution_uses_regex_fallback() {
        assert_eq!(
            FailureClass::from(&WorkError::Execution(
                "build failed: compilation error".into()
            )),
            FailureClass::Internal
        );
        assert_eq!(
            FailureClass::from(&WorkError::Execution("[E0425] cannot find value".into())),
            FailureClass::Internal
        );
    }

    #[test]
    fn work_dependency_maps_to_internal() {
        assert_eq!(
            FailureClass::from(&WorkError::Dependency("artifact".into())),
            FailureClass::Internal
        );
    }

    #[test]
    fn server_error_delegates_to_dispatch_and_http() {
        assert_eq!(
            FailureClass::from(&ServerError::Dispatch(DispatchError::RateLimited)),
            FailureClass::RateLimit
        );
        assert_eq!(
            FailureClass::from(&ServerError::Http("HTTP 401".into())),
            FailureClass::Authentication
        );
        assert_eq!(
            FailureClass::from(&ServerError::Bind {
                addr: "0.0.0.0:1".into(),
                source: std::io::Error::new(std::io::ErrorKind::Other, "in use"),
            }),
            FailureClass::Network
        );
    }
}
