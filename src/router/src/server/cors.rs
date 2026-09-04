//! CORS headers — the single owner (ROADMAP_20260903_LLM M7).
//!
//! Moved verbatim from `common_core::http` (`CORS_ALLOW_ORIGIN`,
//! `CORS_ALLOW_METHODS`, `CORS_ALLOW_HEADERS`, `add_cors_headers`). CORS
//! lives with the HTTP server that attaches it: every response built in
//! `super::responses` (plus `instances_api` and `admin` via `responses`)
//! carries these headers.
//!
//! M11 deleted the `common-core::http` byte-identical shim copies (kept
//! through M10 under `#[deprecated]`); the owner values locked by
//! `cors_headers_unchanged` (`tests/server_responses.rs`) are the lasting
//! contract.

use http::HeaderMap;

/// CORS header set the router attaches to every response.
pub const CORS_ALLOW_ORIGIN: &str = "*";
pub const CORS_ALLOW_METHODS: &str = "POST, GET, OPTIONS";
pub const CORS_ALLOW_HEADERS: &str = "Content-Type, Authorization";

/// Insert the standard CORS headers into `headers`.
pub fn add_cors_headers(headers: &mut HeaderMap) {
    headers.insert(
        http::header::HeaderName::from_static("access-control-allow-origin"),
        http::header::HeaderValue::from_static(CORS_ALLOW_ORIGIN),
    );
    headers.insert(
        http::header::HeaderName::from_static("access-control-allow-methods"),
        http::header::HeaderValue::from_static(CORS_ALLOW_METHODS),
    );
    headers.insert(
        http::header::HeaderName::from_static("access-control-allow-headers"),
        http::header::HeaderValue::from_static(CORS_ALLOW_HEADERS),
    );
}
