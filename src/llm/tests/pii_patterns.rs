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

#[test]
fn twelve_vs_five_exposure_gap_is_deliberate() {
    // M6.2 lock: 12 compiled statics exist, exactly 5 are exposed via
    // `pii_patterns()`. The 7 anonymize-only statics must keep firing inside
    // `anonymize` (golden table) while staying out of this table (pre-filter
    // behavior unchanged). Any change to either side fails here first.
    assert_eq!(pii_patterns().len(), 5);
    let exposed: Vec<&str> = pii_patterns().iter().map(|p| p.name).collect();
    for name in ["nino", "sin", "bearer", "aws_key", "generic_api_key", "ipv6", "ipv4"] {
        assert!(!exposed.contains(&name), "{name} must stay unexposed");
    }
    // And the anonymize-only statics still match (they are load-bearing).
    assert!(NINO_UK_RE.is_match("AB123456C"));
    assert!(SIN_CA_RE.is_match("046-454-286"));
    assert!(BEARER_RE.is_match("Bearer abcdefgh"));
    assert!(AWS_KEY_RE.is_match("AKIAIOSFODNN7EXAMPLE"));
    assert!(GENERIC_API_KEY_RE.is_match("abcdef1234567890abcdef1234567890"));
    assert!(IPV6_RE.is_match("2001:0db8:85a3:0000:0000:8a2e:0370:7334"));
    assert!(IPV4_RE.is_match("192.168.1.1"));
}
