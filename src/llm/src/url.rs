use thiserror::Error;

#[derive(Debug, Error)]
pub enum UrlError {
    #[error("invalid API URL: no scheme")]
    InvalidApiUrl,
    #[error("insecure API URL: non-local HTTP")]
    InsecureApiUrl,
    #[error("SSRF blocked URL: private IP over HTTPS")]
    SsrfBlockedUrl,
}

pub fn is_local_host(host: &str) -> bool {
    let h = host.trim().to_lowercase();
    h == "localhost" || h == "127.0.0.1" || h == "::1" || h.starts_with("127.")
}

pub fn is_private_ip(host: &str) -> bool {
    let h = host.trim();
    let h = h.strip_prefix('[').unwrap_or(h);
    let h = h.strip_suffix(']').unwrap_or(h);

    if h.starts_with("10.")
        || h.starts_with("192.168.")
        || h.starts_with("169.254.")
        || h.starts_with("0.")
    {
        return true;
    }
    if h.starts_with("172.") {
        let parts: Vec<&str> = h.split('.').collect();
        if parts.len() >= 2 {
            if let Ok(second) = parts[1].parse::<u8>() {
                if (16..=31).contains(&second) {
                    return true;
                }
            }
        }
    }
    let lower = h.to_lowercase();
    if lower.starts_with("fc") || lower.starts_with("fd") || lower.starts_with("fe80") {
        return true;
    }
    false
}

fn extract_host(url: &str) -> &str {
    if let Some(rest) = url.split("://").nth(1) {
        let end = rest.find([':', '/']).unwrap_or(rest.len());
        &rest[..end]
    } else {
        url
    }
}

pub fn validate_https_or_local_http(url: &str) -> Result<(), UrlError> {
    if url.is_empty() || !url.contains("://") {
        return Err(UrlError::InvalidApiUrl);
    }
    let is_https = url.starts_with("https://");
    let is_http = url.starts_with("http://");
    if !is_https && !is_http {
        return Err(UrlError::InvalidApiUrl);
    }
    let host = extract_host(url);
    if is_http && !is_local_host(host) {
        return Err(UrlError::InsecureApiUrl);
    }
    if is_https && is_private_ip(host) {
        return Err(UrlError::SsrfBlockedUrl);
    }
    Ok(())
}

/// Derive the chat-completions endpoint URL from a base URL.
///
/// If `base` already ends in `/chat/completions` it is returned as-is
/// (after trimming trailing slashes); otherwise `/chat/completions` is
/// appended.
pub fn chat_completions_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

/// Derive an OpenAI-compatible embeddings base URL from a chat-completions
/// endpoint: `http://host:port/v1/chat/completions` → `http://host:port/v1`
/// (the embeddings client appends `/embeddings`).
pub fn derive_embeddings_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    if let Some(base) = trimmed.strip_suffix("/v1/chat/completions") {
        format!("{base}/v1")
    } else if let Some(base) = trimmed.strip_suffix("/chat/completions") {
        format!("{base}/v1")
    } else {
        trimmed.to_string()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_local_host_variants() {
        assert!(is_local_host("localhost"));
        assert!(is_local_host("127.0.0.1"));
        assert!(is_local_host("::1"));
        assert!(!is_local_host("example.com"));
    }

    #[test]
    fn is_private_ip_ranges() {
        assert!(is_private_ip("10.0.0.1"));
        assert!(is_private_ip("192.168.1.1"));
        assert!(is_private_ip("172.16.0.1"));
        assert!(!is_private_ip("8.8.8.8"));
    }

    #[test]
    fn validate_https_accepts() {
        assert!(validate_https_or_local_http("https://api.openai.com/v1").is_ok());
    }

    #[test]
    fn validate_local_http_accepts() {
        assert!(validate_https_or_local_http("http://localhost:11434").is_ok());
    }

    #[test]
    fn validate_rejects_remote_http() {
        assert!(validate_https_or_local_http("http://evil.com").is_err());
    }

    #[test]
    fn validate_rejects_bare_hostname() {
        assert!(validate_https_or_local_http("localhost").is_err());
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(validate_https_or_local_http("").is_err());
    }

    #[test]
    fn allows_localhost_http() {
        assert!(validate_https_or_local_http("http://localhost:11434/api/embed").is_ok());
    }

    #[test]
    fn allows_public_https() {
        assert!(validate_https_or_local_http("https://api.openai.com/v1/embeddings").is_ok());
    }

    #[test]
    fn blocks_aws_metadata_http() {
        assert!(validate_https_or_local_http("http://169.254.169.254/latest/meta-data").is_err());
    }

    #[test]
    fn blocks_aws_metadata_https() {
        assert!(validate_https_or_local_http("https://169.254.169.254/latest/meta-data").is_err());
    }

    #[test]
    fn blocks_private_class_a_https() {
        assert!(validate_https_or_local_http("https://10.0.0.1/api").is_err());
    }

    #[test]
    fn blocks_private_class_c_https() {
        assert!(validate_https_or_local_http("https://192.168.1.1/api").is_err());
    }

    #[test]
    fn chat_completions_url_already_suffixed() {
        assert_eq!(
            chat_completions_url("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_appends() {
        assert_eq!(
            chat_completions_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_trims_trailing_slash() {
        assert_eq!(
            chat_completions_url("http://localhost:11434/v1/"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn derive_embeddings_url_v1_chat() {
        assert_eq!(
            derive_embeddings_url("http://host:port/v1/chat/completions"),
            "http://host:port/v1"
        );
    }

    #[test]
    fn derive_embeddings_url_plain_chat() {
        assert_eq!(
            derive_embeddings_url("http://host:port/chat/completions"),
            "http://host:port/v1"
        );
    }

    #[test]
    fn derive_embeddings_url_passthrough() {
        assert_eq!(
            derive_embeddings_url("http://host:port/v1"),
            "http://host:port/v1"
        );
    }
}
