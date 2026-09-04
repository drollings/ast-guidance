//! Address parsing, host equivalence, and self-routing validation utilities.

use std::collections::HashMap;

use super::ModelEntry;
use crate::error::ServerError;

/// `true` when two hosts name the same loopback (`"localhost"`, `"127.0.0.1"`,
/// `"::1"` — trim-only, no case folding, no `127/8` range).
///
/// Thin wrapper over `fluent_llm::url::hosts_equivalent_with` (`EXACT_LOOPBACK`).
/// The class is deliberately NOT unified with the LLM-transport class
/// (SSRF vs self-routing-loop threat models).
pub fn hosts_equivalent(a: &str, b: &str) -> bool {
    // Delegates to the shared primitive; the wrapper (and the divergence
    // tests) lock the EXACT_LOOPBACK class — any change here is a
    // security-relevant behavior change, not a refactor.
    let opts = fluent_llm::url::HostEquivalence::EXACT_LOOPBACK;
    fluent_llm::url::hosts_equivalent_with(a, b, opts)
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
        // Managed models (weights/hf_repo/instances) declare no endpoint in the
        // config; Coral Router assigns and rewrites it to a spawned
        // llama-server's localhost address at boot. Skip them here — an empty
        // endpoint is not a self-routing risk.
        if url.is_empty() {
            continue;
        }
        // Parse host:port from the endpoint URL.
        // NOTE (P5, evaluated): this keeps its own scheme-strip + `parse_bind_addr`
        // shape instead of composing `fluent_llm::url::extract_host` — that helper
        // is host-only (drops the port this check needs) and IPv6-bracket naive
        // (`extract_host("http://[::1]:8079/x") == "["`), while this flow must
        // handle `[::1]:port` via `parse_bind_addr`. Sharing it here would be a
        // security-relevant behavior change, not a refactor.
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
#[path = "../../tests/config_addr.rs"]
mod tests;
