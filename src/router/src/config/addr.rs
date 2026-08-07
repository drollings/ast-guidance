//! Address parsing, host equivalence, and self-routing validation utilities.

use std::collections::HashMap;

use super::ModelEntry;
use crate::error::ServerError;

/// Normalize a hostname to a canonical comparison form.
/// Returns `true` if two hosts should be considered equivalent
/// (e.g. `"localhost"` and `"127.0.0.1"`).
fn normalize_host(h: &str) -> String {
    match h.trim() {
        "localhost" | "127.0.0.1" | "::1" => "127.0.0.1".into(),
        other => other.to_string(),
    }
}

pub fn hosts_equivalent(a: &str, b: &str) -> bool {
    normalize_host(a) == normalize_host(b)
}

/// Parse a `host:port` string into its components.
pub fn parse_bind_addr(addr: &str) -> Result<(&str, u16), ServerError> {
    let addr = addr.trim();
    if addr.is_empty() {
        return Err(ServerError::Addr("bind_addr is empty".into()));
    }
    // Handle IPv6: [::1]:port
    if addr.starts_with('[') {
        let close_bracket = addr
            .rfind(']')
            .ok_or_else(|| ServerError::Addr("unclosed '[' in bind_addr".into()))?;
        let host = &addr[1..close_bracket];
        let rest = addr[close_bracket + 1..].trim_start_matches(':');
        let port: u16 = rest
            .parse()
            .map_err(|e| ServerError::Addr(format!("invalid port in bind_addr '{addr}': {e}")))?;
        return Ok((host, port));
    }
    if let Some(colon_pos) = addr.rfind(':') {
        let host = &addr[..colon_pos];
        let port: u16 = addr[colon_pos + 1..]
            .parse()
            .map_err(|e| ServerError::Addr(format!("invalid port in bind_addr '{addr}': {e}")))?;
        Ok((host, port))
    } else {
        Err(ServerError::Addr(format!(
            "bind_addr '{addr}' missing port (expected host:port)"
        )))
    }
}

/// Validate that none of the configured model endpoints point back at the
/// router's own bind address.  This prevents accidental self-routing loops.
#[allow(clippy::implicit_hasher)]
pub fn validate_no_self_routing(
    bind_addr: &str,
    models: &HashMap<String, ModelEntry>,
) -> Result<(), ServerError> {
    if bind_addr.is_empty() {
        return Err(ServerError::Addr("server.bind_addr must be set".into()));
    }
    let (my_host, my_port) = parse_bind_addr(bind_addr)?;

    for (name, entry) in models {
        let url = entry.endpoint.trim();
        // Parse host:port from the endpoint URL
        let url_no_scheme = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .unwrap_or(url);
        let (host, port) = match url_no_scheme.find('/') {
            Some(pos) => {
                let hp = &url_no_scheme[..pos];
                parse_bind_addr(hp)?
            }
            None => parse_bind_addr(url_no_scheme)?,
        };

        if hosts_equivalent(host, my_host) && port == my_port {
            return Err(ServerError::Addr(format!(
                "model '{name}' endpoint ({}) points to the router's own bind address ({}) — would create a routing loop",
                entry.endpoint, bind_addr
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_equivalent_same_host() {
        assert!(hosts_equivalent("localhost", "localhost"));
    }

    #[test]
    fn hosts_equivalent_localhost_and_ip() {
        assert!(hosts_equivalent("localhost", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_ipv6_local() {
        assert!(hosts_equivalent("::1", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_different_hosts() {
        assert!(!hosts_equivalent("upstream.test", "127.0.0.1"));
        assert!(!hosts_equivalent("0.0.0.0", "127.0.0.1"));
    }

    #[test]
    fn hosts_equivalent_works_with_whitespace() {
        assert!(hosts_equivalent("  localhost  ", "127.0.0.1"));
    }

    #[test]
    fn parse_bind_addr_simple() {
        let (host, port) = parse_bind_addr("0.0.0.0:8079").unwrap();
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8079);
    }

    #[test]
    fn parse_bind_addr_empty_fails() {
        assert!(parse_bind_addr("").is_err());
    }

    #[test]
    fn parse_bind_addr_missing_port_fails() {
        assert!(parse_bind_addr("localhost").is_err());
    }

    #[test]
    fn parse_bind_addr_ipv6() {
        let (host, port) = parse_bind_addr("[::1]:8079").unwrap();
        assert_eq!(host, "::1");
        assert_eq!(port, 8079);
    }

    #[test]
    fn validate_ok_when_no_models() {
        let models = HashMap::new();
        assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
    }

    #[test]
    fn validate_ok_when_models_point_upstream() {
        let mut models = HashMap::new();
        models.insert(
            "fast".into(),
            ModelEntry {
                endpoint: "http://upstream.test:8080/v1/chat/completions".into(),
                name: Some("fast".into()),
                intelligence: 1,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cached_read: 0.0,
                speed: 10,
                total_timeout_ms: 5000,
                idle_timeout_ms: 2000,
                stream: false,
                filter_thinking: false,
                retry_count: 0,
                retry_base_interval_s: 1,
                params: None,
                instances: None,
            },
        );
        assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
    }

    #[test]
    fn validate_rejects_self_loop_localhost() {
        let mut models = HashMap::new();
        models.insert(
            "fast".into(),
            ModelEntry {
                endpoint: "http://localhost:8079/v1/chat/completions".into(),
                name: Some("fast".into()),
                intelligence: 1,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cached_read: 0.0,
                speed: 10,
                total_timeout_ms: 5000,
                idle_timeout_ms: 2000,
                stream: false,
                filter_thinking: false,
                retry_count: 0,
                retry_base_interval_s: 1,
                params: None,
                instances: None,
            },
        );
        let err = validate_no_self_routing("127.0.0.1:8079", &models)
            .expect_err("should reject self-routing model");
        assert!(
            err.to_string().contains("routing loop"),
            "error should mention routing loop: {err}"
        );
    }

    #[test]
    fn validate_rejects_self_loop_exact_match() {
        let mut models = HashMap::new();
        models.insert(
            "fast".into(),
            ModelEntry {
                endpoint: "http://127.0.0.1:8079/v1/chat/completions".into(),
                name: Some("fast".into()),
                intelligence: 1,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cached_read: 0.0,
                speed: 10,
                total_timeout_ms: 5000,
                idle_timeout_ms: 2000,
                stream: false,
                filter_thinking: false,
                retry_count: 0,
                retry_base_interval_s: 1,
                params: None,
                instances: None,
            },
        );
        let err = validate_no_self_routing("127.0.0.1:8079", &models)
            .expect_err("should reject self-routing model");
        assert!(err.to_string().contains("routing loop"));
    }

    #[test]
    fn validate_rejects_when_port_differs_but_host_is_same() {
        let mut models = HashMap::new();
        models.insert(
            "fast".into(),
            ModelEntry {
                endpoint: "http://127.0.0.1:8080/v1/chat/completions".into(),
                name: Some("fast".into()),
                intelligence: 1,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cached_read: 0.0,
                speed: 10,
                total_timeout_ms: 5000,
                idle_timeout_ms: 2000,
                stream: false,
                filter_thinking: false,
                retry_count: 0,
                retry_base_interval_s: 1,
                params: None,
                instances: None,
            },
        );
        // Different port (8080 vs 8079) should be OK
        assert!(validate_no_self_routing("127.0.0.1:8079", &models).is_ok());
    }

    #[test]
    fn validate_empty_bind_addr_errors() {
        let models = HashMap::new();
        let err = validate_no_self_routing("", &models).expect_err("empty bind_addr should error");
        assert!(err.to_string().contains("must be set"));
    }
}
