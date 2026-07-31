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
pub static SSN_US_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap());
pub static NINO_UK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2}\d{6}[A-Z]\b").unwrap());
pub static SIN_CA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3}-\d{3}-\d{3}\b").unwrap());
pub static BEARER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Bearer\s+[A-Za-z0-9_\-]{8,}").unwrap());
pub static AWS_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap());
pub static GENERIC_API_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-zA-Z0-9]{32,}\b").unwrap());
pub static IPV6_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b").unwrap());

/// Canonical PII detection patterns.
///
/// The compiled statics above are the single source of truth. This table is
/// derived from them so the names and regex strings can never drift from the
/// compiled patterns used by `anonymize`.
pub struct PiiPattern {
    pub name: &'static str,
    pub regex: &'static LazyLock<Regex>,
}

pub fn pii_patterns() -> &'static [PiiPattern] {
    static PATTERNS: &[PiiPattern] = &[
        PiiPattern {
            name: "ssn",
            regex: &SSN_US_RE,
        },
        PiiPattern {
            name: "card_number",
            regex: &CREDIT_CARD_RE,
        },
        PiiPattern {
            name: "email",
            regex: &EMAIL_RE,
        },
        PiiPattern {
            name: "phone",
            regex: &PHONE_US_RE,
        },
        PiiPattern {
            name: "api_key",
            regex: &API_KEY_RE,
        },
    ];
    PATTERNS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_table_has_expected_shape() {
        let patterns = pii_patterns();
        let names: Vec<&str> = patterns.iter().map(|p| p.name).collect();
        assert_eq!(names, ["ssn", "card_number", "email", "phone", "api_key"]);
        for p in patterns {
            assert!(!p.regex.as_str().is_empty());
            let compiled = Regex::new(p.regex.as_str()).expect("table regex compiles");
            assert_eq!(compiled.as_str(), p.regex.as_str());
        }
    }

    #[test]
    fn patterns_table_matches_statics() {
        let patterns = pii_patterns();
        assert_eq!(patterns[0].regex.as_str(), SSN_US_RE.as_str());
        assert_eq!(patterns[1].regex.as_str(), CREDIT_CARD_RE.as_str());
        assert_eq!(patterns[2].regex.as_str(), EMAIL_RE.as_str());
        assert_eq!(patterns[3].regex.as_str(), PHONE_US_RE.as_str());
        assert_eq!(patterns[4].regex.as_str(), API_KEY_RE.as_str());
    }

    #[test]
    fn ssn_pattern_matches() {
        assert!(SSN_US_RE.is_match("123-45-6789"));
        assert!(!SSN_US_RE.is_match("1234-56-789"));
    }
}
