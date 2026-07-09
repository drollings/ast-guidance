use std::collections::HashMap;
use std::net::SocketAddr;

use crate::error::CopilotError;

/// Check that the peer address is loopback. Returns `Err` for non-loopback.
pub fn check_loopback_peer(peer: &SocketAddr) -> Result<(), CopilotError> {
    if !peer.ip().is_loopback() {
        return Err(CopilotError::Auth(format!(
            "non-loopback peer rejected: {peer}"
        )));
    }
    Ok(())
}

/// Check the `Authorization: Bearer <token>` header against an expected token.
///
/// If `expected` is `None`, any/no token is accepted (auth disabled).
pub fn check_bearer_token(
    headers: &HashMap<String, String, std::collections::hash_map::RandomState>,
    expected: Option<&str>,
) -> Result<(), CopilotError> {
    let Some(expected) = expected else {
        return Ok(());
    };

    if expected.is_empty() {
        return Ok(());
    }

    let auth = headers.get("authorization").map_or("", String::as_str);

    let token = auth.strip_prefix("Bearer ").unwrap_or("");

    if token != expected {
        return Err(CopilotError::Auth("invalid bearer token".into()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn loopback_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 12345)
    }

    fn public_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)), 12345)
    }

    #[test]
    fn check_loopback_accepts_127_0_0_1() {
        assert!(check_loopback_peer(&loopback_addr()).is_ok());
    }

    #[test]
    fn check_loopback_rejects_public_ip() {
        let err = check_loopback_peer(&public_addr()).unwrap_err();
        assert!(format!("{err}").contains("non-loopback peer rejected"));
    }

    #[test]
    fn bearer_token_none_accepts_any() {
        let headers = HashMap::new();
        assert!(check_bearer_token(&headers, None).is_ok());
    }

    #[test]
    fn bearer_token_empty_expected_accepts_any() {
        let headers = HashMap::new();
        assert!(check_bearer_token(&headers, Some("")).is_ok());
    }

    #[test]
    fn bearer_token_valid() {
        let mut headers = HashMap::new();
        headers.insert("authorization".into(), "Bearer my-secret".into());
        assert!(check_bearer_token(&headers, Some("my-secret")).is_ok());
    }

    #[test]
    fn bearer_token_wrong() {
        let mut headers = HashMap::new();
        headers.insert("authorization".into(), "Bearer wrong".into());
        let err = check_bearer_token(&headers, Some("my-secret")).unwrap_err();
        assert!(format!("{err}").contains("invalid bearer token"));
    }

    #[test]
    fn bearer_token_missing_header() {
        let headers = HashMap::new();
        let err = check_bearer_token(&headers, Some("my-secret")).unwrap_err();
        assert!(format!("{err}").contains("invalid bearer token"));
    }
}
