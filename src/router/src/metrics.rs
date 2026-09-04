//! Failure classification for the router pipeline.
//!
//! `FailureClass` (the enum + stable labels) is the canonical shared
//! taxonomy in `fluent_llm::http_class`; this module re-exports it.
//! Typed-first: `FailureClass` is derived from the typed error
//! (`DispatchError`/`WorkError`/`ServerError` via `From`), with the string
//! regex classifier retained only as a fallback for opaque payloads (shell
//! output, compiler diagnostics like `[E0425]`).

use std::sync::LazyLock;

use regex::Regex;

use crate::dispatch::frontier::DispatchError;
use crate::error::ServerError;

pub use fluent_llm::http_class::FailureClass;

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
            DispatchError::RateLimited | DispatchError::InstanceGroupMiss { .. } => {
                FailureClass::RateLimit
            }
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

/// Classify a `ServerError` typed-first. `Dispatch` delegates to the
/// `DispatchError` mapping; `Http`/`Bind` reuse the status/string logic.
impl From<&ServerError> for FailureClass {
    fn from(err: &ServerError) -> Self {
        match err {
            ServerError::Dispatch(e) => FailureClass::from(e),
            ServerError::Http(msg) => classify_http_status(msg),
            ServerError::Bind { .. } => FailureClass::Network,
            ServerError::Addr(_) => FailureClass::InputValidation,
        }
    }
}

/// Extract an HTTP status from a `DispatchError::Http`/`ServerError::Http`
/// message of the form `"HTTP <code>..."` and map it typed-first via the
/// shared `fluent_llm::http_class::classify_http_status`. Free-form
/// strings (reqwest errors, timeout labels) fall back to `classify_error`.
fn classify_http_status(msg: &str) -> FailureClass {
    let status = msg
        .strip_prefix("HTTP ")
        .and_then(|rest| rest.split([' ', ':']).next())
        .and_then(|s| s.parse::<u16>().ok());
    match status {
        Some(status) => fluent_llm::http_class::classify_http_status(status),
        None => classify_error(msg),
    }
}

#[cfg(test)]
#[path = "../tests/metrics.rs"]
mod tests;
