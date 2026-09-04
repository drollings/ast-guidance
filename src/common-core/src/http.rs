//! Process-wide shared HTTP client (and test variants).
//!
//! Use `shared_http_client()` in production code to amortize TLS handshakes
//! and connection pool setup. Use `test_http_client()` in tests to get a
//! client with a consistent User-Agent for log filtering.

use std::sync::OnceLock;
use std::time::Duration;

static SHARED: OnceLock<reqwest::Client> = OnceLock::new();

/// Returns a process-wide shared `reqwest::Client` with sensible defaults.
/// Amortizes TLS handshakes and connection pool setup across all callers.
pub fn shared_http_client() -> &'static reqwest::Client {
    SHARED.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("common-core/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .expect("build shared reqwest client")
    })
}

#[cfg(any(test, feature = "test-util"))]
static TEST: OnceLock<reqwest::Client> = OnceLock::new();

/// Returns a shared `reqwest::Client` for test code.
/// Uses default settings with no custom timeouts for test flexibility.
#[cfg(any(test, feature = "test-util"))]
pub fn test_http_client() -> &'static reqwest::Client {
    TEST.get_or_init(reqwest::Client::new)
}

// NOTE (ROADMAP_20260903_LLM M11): the CORS contract (`CORS_*`,
// `add_cors_headers`, now owned by `fluent_router::server::cors`) and the
// error envelope (`error_value`, now `fluent_llm::openai::error_response`)
// lived here through M10 as deprecated byte-identical shims; M11 deleted
// them. The generic client + body helpers below stay.

/// JSON body string for `value`.
pub fn json_body(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

