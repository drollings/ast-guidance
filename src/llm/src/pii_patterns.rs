use std::sync::LazyLock;

use regex::Regex;

pub static EMAIL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap());
pub static API_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(api[_-]?key|apikey|secret|token)\s*[=:]\s*[a-zA-Z0-9_-]{16,}").unwrap()
});
pub static IPV4_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap());
pub static PHONE_US_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap());
pub static CREDIT_CARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap());
pub static SSN_US_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap());
pub static NINO_UK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2}\d{6}[A-Z]\b").unwrap());
pub static SIN_CA_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d{3}-\d{3}-\d{3}\b").unwrap());
pub static BEARER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Bearer\s+[A-Za-z0-9_\-]{8,}").unwrap());
pub static AWS_KEY_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap());
pub static GENERIC_API_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-zA-Z0-9]{32,}\b").unwrap());
pub static IPV6_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b").unwrap());

/// Canonical regex pattern strings for PII detection. Used by the router's
/// deterministic pre-filter to build `PatternEntry` values.
pub struct PiiPattern {
    pub name: &'static str,
    pub regex: &'static str,
}

pub fn pii_patterns() -> Vec<PiiPattern> {
    vec![
        PiiPattern { name: "ssn", regex: r"\b\d{3}-\d{2}-\d{4}\b" },
        PiiPattern { name: "card_number", regex: r"\b(?:\d[ -]*?){13,19}\b" },
        PiiPattern { name: "email", regex: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b" },
        PiiPattern { name: "phone", regex: r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b" },
        PiiPattern { name: "api_key", regex: r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]{8,}" },
    ]
}
