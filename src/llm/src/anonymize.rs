pub fn anonymize(text: &str) -> String {
    use crate::pii_patterns::{
        API_KEY_RE, AWS_KEY_RE, BEARER_RE, CREDIT_CARD_RE, EMAIL_RE, GENERIC_API_KEY_RE, IPV4_RE,
        IPV6_RE, NINO_UK_RE, PHONE_US_RE, SIN_CA_RE, SSN_US_RE,
    };

    let text = EMAIL_RE.replace_all(text, "[EMAIL]");
    let text = CREDIT_CARD_RE.replace_all(&text, "[CREDIT_CARD]");
    let text = SSN_US_RE.replace_all(&text, "[SSN]");
    let text = NINO_UK_RE.replace_all(&text, "[NINO]");
    let text = SIN_CA_RE.replace_all(&text, "[SIN]");
    let text = BEARER_RE.replace_all(&text, "[BEARER_TOKEN]");
    let text = AWS_KEY_RE.replace_all(&text, "[AWS_KEY]");
    let text = GENERIC_API_KEY_RE.replace_all(&text, "[API_KEY]");
    let text = IPV6_RE.replace_all(&text, "[IPv6]");
    let text = IPV4_RE.replace_all(&text, "[IP_ADDRESS]");
    let text = PHONE_US_RE.replace_all(&text, "[PHONE]");
    let text = API_KEY_RE.replace_all(&text, "[REDACTED]");
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anonymize_email() {
        let result = anonymize("Contact me at user@example.com");
        assert_eq!(result, "Contact me at [EMAIL]");
    }

    #[test]
    fn test_anonymize_credit_card() {
        let result = anonymize("Card: 1234-5678-9012-3456");
        assert!(result.contains("[CREDIT_CARD]"));
    }

    #[test]
    fn test_anonymize_ssn_us() {
        let result = anonymize("SSN: 123-45-6789");
        assert!(result.contains("[SSN]"));
        assert!(!result.contains("123-45-6789"));
    }

    #[test]
    fn test_anonymize_nino_uk() {
        let result = anonymize("NINO: AB123456C");
        assert!(result.contains("[NINO]"));
    }

    #[test]
    fn test_anonymize_sin_ca() {
        let result = anonymize("SIN: 046-454-286");
        assert!(result.contains("[SIN]"));
    }

    #[test]
    fn test_anonymize_bearer_token() {
        let result = anonymize("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9");
        assert!(result.contains("[BEARER_TOKEN]"));
    }

    #[test]
    fn test_anonymize_aws_key() {
        let result = anonymize("Key: AKIAIOSFODNN7EXAMPLE");
        assert!(result.contains("[AWS_KEY]"));
        assert!(!result.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn test_anonymize_generic_api_key() {
        let result = anonymize("api_key=abcdef1234567890abcdef1234567890");
        assert!(result.contains("[API_KEY]"));
    }

    #[test]
    fn test_anonymize_ipv6() {
        let result = anonymize("IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334");
        assert!(result.contains("[IPv6]"));
    }

    #[test]
    fn test_anonymize_ipv4() {
        let result = anonymize("Server at 192.168.1.1");
        assert_eq!(result, "Server at [IP_ADDRESS]");
    }

    #[test]
    fn test_anonymize_phone() {
        let result = anonymize("Call 555-123-4567 for info");
        assert!(result.contains("[PHONE]"));
    }

    #[test]
    fn test_anonymize_multiple() {
        let result = anonymize("user@test.com api_key=abcdefghijklmnop12345 from 10.0.0.1");
        assert!(result.contains("[EMAIL]"));
        assert!(result.contains("[REDACTED]"));
        assert!(result.contains("[IP_ADDRESS]"));
    }

    #[test]
    fn test_no_pii_unchanged() {
        let text = "This is a normal sentence with no PII.";
        let result = anonymize(text);
        assert_eq!(result, text);
    }
}
