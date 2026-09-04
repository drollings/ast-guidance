use std::sync::LazyLock;

use regex::Regex;

use crate::transforms::{rewrite_text_messages, TransformError, TransformStrategy};
use crate::types::RouterRequest;

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

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        rewrite_text_messages(request, |content| {
            if !SECRET_RE.is_match(content) {
                return Ok(content.to_string());
            }
            let masked = SECRET_RE.replace_all(content, |caps: &regex::Captures| {
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
            Ok(masked.to_string())
        })
    }
}
#[cfg(test)]
#[path = "../../tests/transforms_secret_mask.rs"]
mod tests;
