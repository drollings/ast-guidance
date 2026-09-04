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
    hosts_equivalent_with(host, "127.0.0.1", HostEquivalence::LOCAL_SSRF)
}

/// Options for the parameterized host-equivalence primitive (P5).
///
/// The two call sites keep DIFFERENT equivalence classes (different threat
/// models — SSRF vs self-routing-loop — so the classes are never unified):
/// - `LOCAL_SSRF` (`fold_case`, full `127/8`): the LLM-transport class that
///   [`is_local_host`] preserves.
/// - `EXACT_LOOPBACK` (trim only, three exact forms): the router's
///   `hosts_equivalent` class (see `router/src/config/addr.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostEquivalence {
    /// Lowercase before comparing (`"LocalHost" == "localhost"`).
    pub fold_case: bool,
    /// Treat the whole `127.*` range as loopback (else only `127.0.0.1`).
    pub loopback_range_127: bool,
}

impl HostEquivalence {
    /// The LLM-transport (SSRF) class: case-folded, full `127/8` loopback.
    pub const LOCAL_SSRF: Self = Self { fold_case: true, loopback_range_127: true };
    /// The router self-routing class: trim-only, three exact loopback forms.
    pub const EXACT_LOOPBACK: Self = Self { fold_case: false, loopback_range_127: false };
}

fn canonical_host(host: &str, opts: HostEquivalence) -> String {
    let trimmed = host.trim();
    if opts.fold_case {
        let lower = trimmed.to_lowercase();
        if lower == "localhost"
            || lower == "::1"
            || lower == "127.0.0.1"
            || (opts.loopback_range_127 && lower.starts_with("127."))
        {
            return "127.0.0.1".to_string();
        }
        return lower;
    }
    match trimmed {
        "localhost" | "127.0.0.1" | "::1" => "127.0.0.1".to_string(),
        _ => trimmed.to_string(),
    }
}

/// Parameterized host equivalence: compare canonical forms under `opts`.
/// Each side's current equivalence class is preserved by its own options
/// (see [`HostEquivalence`]) — the classes are never merged.
pub fn hosts_equivalent_with(a: &str, b: &str, opts: HostEquivalence) -> bool {
    canonical_host(a, opts) == canonical_host(b, opts)
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

/// Extract the host from a URL (`scheme://host[:port][/path]` → `host`).
/// Scheme-less inputs pass through unchanged. Shared primitive (P5) — note
/// it is IPv6-bracket naive (cuts at the first `':'`), so host:port parsers
/// that must handle `[::1]` (e.g. the router's `parse_bind_addr` flow) keep
/// their own bracket-aware shape instead of composing this.
pub fn extract_host(url: &str) -> &str {
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
#[path = "../tests/url.rs"]
mod tests;
