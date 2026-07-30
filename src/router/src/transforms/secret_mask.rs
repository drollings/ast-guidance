use std::sync::LazyLock;

use regex::Regex;

use crate::transforms::{TransformError, TransformStrategy};
use crate::types::{RouterMessageContent, RouterRequest};

static SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)",  // case-insensitive
        // Bearer token: preserve "Bearer " prefix, mask the token
        r"(?<bearer>\bBearer\s+)[A-Za-z0-9\-._~+/=]+",
        r"|",
        // Basic auth: preserve "Basic " prefix, mask the token
        r"(?<basic>\bBasic\s+)[A-Za-z0-9+/=]+",
        r"|",
        // API key with sk- prefix
        r"\bsk-[A-Za-z0-9]{10,}",
        r"|",
        // AWS access keys: AKIA + 16 alphanumeric
        r"\bAKIA[A-Z0-9]{16}",
        r"|",
        // GitHub personal access tokens: ghp_ + alphanumeric
        r"\bghp_[A-Za-z0-9]{36,}",
        r"|",
        // GitHub OAuth tokens: gho_ + alphanumeric
        r"\bgho_[A-Za-z0-9]{36,}",
        r"|",
        // GitHub user-to-server tokens: ghu_ + alphanumeric
        r"\bghu_[A-Za-z0-9]{36,}",
        r"|",
        // GitHub SSH user keys: ghs_ + alphanumeric
        r"\bghs_[A-Za-z0-9]{36,}",
        r"|",
        // Named secrets: key=value or key:value (preserve key name, mask value)
        r"(?<keyname>\b(?:password|token|key|secret|api_key|apikey|access_token|private_key)\s*[=:]\s*)\S+",
    )).unwrap()
});

pub struct SecretMask;

impl TransformStrategy for SecretMask {
    fn name(&self) -> &str {
        "secret_mask"
    }

    fn transform(
        &self,
        request: &RouterRequest,
        _pii_classes: &[String],
    ) -> Result<RouterRequest, TransformError> {
        let mut transformed = request.clone();

        for message in &mut transformed.messages {
            let text_ref = match &message.content {
                RouterMessageContent::Text(s) => s,
                RouterMessageContent::Parts(_) => continue,
            };
            if !SECRET_RE.is_match(text_ref) {
                continue;
            }
            let text = text_ref.clone();

            let masked = SECRET_RE.replace_all(&text, |caps: &regex::Captures| {
                if let Some(m) = caps.name("bearer") {
                    format!("{}****", m.as_str())
                } else if let Some(m) = caps.name("basic") {
                    format!("{}****", m.as_str())
                } else if let Some(m) = caps.name("keyname") {
                    format!("{}****", m.as_str())
                } else {
                    "****".to_string()
                }
            });

            if masked != text {
                message.content = RouterMessageContent::Text(masked.to_string());
            }
        }

        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{test_request, text_of};
    use crate::types::RouterRequest;

    fn make_request(text: &str) -> RouterRequest {
        let mut req = test_request(text);
        req.model = "test".into();
        req
    }

    #[test]
    fn bearer_token_masked_prefix_preserved() {
        let m = SecretMask;
        let req = make_request("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(output.contains("Bearer ****"), "got: {output}");
        assert!(!output.contains("eyJhbGci"), "got: {output}");
    }

    #[test]
    fn basic_auth_masked() {
        let m = SecretMask;
        let req = make_request("Basic dXNlcjpwYXNzd29yZA==");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(output.contains("Basic ****"), "got: {output}");
        assert!(!output.contains("dXNlcjpwYXNzd29yZA=="), "got: {output}");
    }

    #[test]
    fn password_keyvalue_masked() {
        let m = SecretMask;
        let req = make_request("password=superSecret123!");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(output.contains("password=****"), "got: {output}");
        assert!(!output.contains("superSecret123"), "got: {output}");
    }

    #[test]
    fn sk_prefix_key_masked() {
        let m = SecretMask;
        let req = make_request("Use key sk-abc123def456ghijklmnopqrstuvwxyz");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(!output.contains("sk-abc123def456ghijklmnopqrstuvwxyz"), "got: {output}");
    }

    #[test]
    fn akia_key_masked() {
        let m = SecretMask;
        let req = make_request("AWS key: AKIA1234567890ABCDEF");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(!output.contains("AKIA1234567890ABCDEF"), "got: {output}");
    }

    #[test]
    fn github_token_masked() {
        let m = SecretMask;
        let req = make_request("Token: ghp_abc123def456ghijklmnopqrstuvwxyz123456");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(!output.contains("ghp_abc"), "got: {output}");
    }

    #[test]
    fn multiple_secrets_all_masked() {
        let m = SecretMask;
        let req = make_request("Bearer tok123 password=abc api_key=xyz");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(output.contains("Bearer ****"), "got: {output}");
        assert!(output.contains("password=****"), "got: {output}");
        assert!(output.contains("api_key=****"), "got: {output}");
        assert!(!output.contains("tok123"), "got: {output}");
        assert!(!output.contains("=abc"), "got: {output}");
        assert!(!output.contains("=xyz"), "got: {output}");
    }

    #[test]
    fn clean_text_passes_unchanged() {
        let m = SecretMask;
        let req = make_request("Hello, how are you?");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "Hello, how are you?");
    }

    #[test]
    fn matching_case_insensitive_bearer() {
        let m = SecretMask;
        let req = make_request("BEARER token123");
        let result = m.transform(&req, &[]).unwrap();
        let output = text_of(&result);
        assert!(output.to_lowercase().contains("bearer ****"), "got: {output}");
    }

    #[test]
    fn all_key_names_handled() {
        let m = SecretMask;
        for name in &["apikey", "access_token", "private_key", "token", "key", "secret"] {
            let req = make_request(&format!("{name}=value123"));
            let result = m.transform(&req, &[]).unwrap();
            let output = text_of(&result);
            assert!(!output.contains("value123"), "{name} value not masked: {output}");
        }
    }
}
