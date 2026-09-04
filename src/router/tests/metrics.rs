use super::*;
use fluent_wvr::WorkError;

/// Classify a `WorkError`. `Execution` carries an opaque
/// shell/command string — the documented regex fallback; `Dependency` and
/// `Timeout` are typed directly. Expressed as a free function rather than a
/// `From` impl because the orphan rule forbids `impl From<&WorkError> for
/// FailureClass` when both `WorkError` (fluent-wvr) and `FailureClass`
/// (fluent-llm) are foreign to the router.
fn work_error_class(err: &WorkError) -> FailureClass {
    match err {
        WorkError::Execution(msg) => classify_error(msg),
        WorkError::Dependency(_) => FailureClass::Internal,
        WorkError::Timeout { .. } => FailureClass::Timeout,
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

#[test]
fn classify_first_match_wins_arm_order() {
    // Matches both the Timeout arm (first) and the Network arm
    // ("connection refused"): Timeout must win because its regex is earlier.
    assert_eq!(
        classify_error("timeout: connection refused by peer"),
        FailureClass::Timeout
    );
    // Matches both RateLimit ("429") and Authentication ("401"): RateLimit arm
    // is earlier, so it wins.
    assert_eq!(
        classify_error("429 Unauthorized: rate limit exceeded"),
        FailureClass::RateLimit
    );
}

#[test]
fn dispatch_http_unknown_code_maps_to_network() {
    assert_eq!(
        FailureClass::from(&DispatchError::Http("HTTP 999".into())),
        FailureClass::Network
    );
    assert_eq!(
        FailureClass::from(&DispatchError::Http("HTTP 418".into())),
        FailureClass::Network
    );
}

// ── Typed-first mapping ──────────────────────────────────────

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
        work_error_class(&WorkError::Timeout {
            duration_ms: 30_000,
            unit: "x".into(),
        }),
        FailureClass::Timeout
    );
}

#[test]
fn work_execution_uses_regex_fallback() {
    assert_eq!(
        work_error_class(&WorkError::Execution(
            "build failed: compilation error".into()
        )),
        FailureClass::Internal
    );
    assert_eq!(
        work_error_class(&WorkError::Execution("[E0425] cannot find value".into())),
        FailureClass::Internal
    );
}

#[test]
fn work_dependency_maps_to_internal() {
    assert_eq!(
        work_error_class(&WorkError::Dependency("artifact".into())),
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
            source: std::io::Error::other("in use"),
        }),
        FailureClass::Network
    );
}
