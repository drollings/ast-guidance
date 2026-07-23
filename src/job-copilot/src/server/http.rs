use std::collections::HashMap;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::config::DaemonConfig;
use crate::error::CopilotError;
use crate::server::auth::{check_bearer_token, check_loopback_peer};
use crate::server::handler::DaemonHandler;
use common_core::jsonrpc::JsonRpcHandler;

const CORS_HEADERS: &str = "Access-Control-Allow-Origin: *\r\n\
     Access-Control-Allow-Methods: POST, OPTIONS\r\n\
     Access-Control-Allow-Headers: Authorization, Content-Type\r\n";

/// Run the HTTP loopback JSON-RPC server.
///
/// Binds to `config.rest_bind_addr:rest_port` and dispatches `POST /rpc`
/// to the same `DaemonHandler` used by Native Messaging.
pub async fn run_http(
    config: &DaemonConfig,
    handler: Arc<DaemonHandler>,
) -> Result<(), CopilotError> {
    let addr = (config.rest_bind_addr.as_str(), config.rest_port);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| CopilotError::LoopbackBind(format!("{addr:?}: {e}")))?;

    tracing::info!(
        "HTTP loopback listening on {}:{}",
        config.rest_bind_addr,
        config.rest_port
    );

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("accept error: {e}");
                continue;
            }
        };

        if let Err(e) = check_loopback_peer(&peer) {
            tracing::warn!("rejected non-loopback peer: {e}");
            let mut stream = stream;
            let _ = stream
                .write_all(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                .await;
            continue;
        }

        let handler = handler.clone();
        let auth_token = config.auth_token.clone();
        let max_payload = config.max_nm_payload;

        tokio::spawn(async move {
            if let Err(e) =
                handle_connection(stream, handler, auth_token.as_deref(), max_payload).await
            {
                tracing::error!("connection error: {e}");
            }
        });
    }
}

/// Run the HTTP server once (for testing), returning the bound address.
pub async fn run_http_once(
    config: &DaemonConfig,
    handler: Arc<DaemonHandler>,
) -> Result<std::net::SocketAddr, CopilotError> {
    let addr = (config.rest_bind_addr.as_str(), 0);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| CopilotError::LoopbackBind(format!("{addr:?}: {e}")))?;

    let bound = listener.local_addr().unwrap();

    let auth_token_owned = config.auth_token.clone();
    let max_payload = config.max_nm_payload;

    tokio::spawn(async move {
        loop {
            let Ok((stream, peer)) = listener.accept().await else {
                break;
            };

            if check_loopback_peer(&peer).is_err() {
                let mut stream = stream;
                let _ = stream
                    .write_all(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                    .await;
                continue;
            }

            let handler = handler.clone();
            let auth_token = auth_token_owned.clone();
            tokio::spawn(async move {
                let _ =
                    handle_connection(stream, handler, auth_token.as_deref(), max_payload).await;
            });
        }
    });

    Ok(bound)
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    handler: Arc<DaemonHandler>,
    auth_token: Option<&str>,
    max_payload: usize,
) -> Result<(), CopilotError> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 1024];

    // Read until we see the end of headers (\r\n\r\n).
    loop {
        let n = stream
            .read(&mut tmp)
            .await
            .map_err(|e| CopilotError::Http(format!("read error: {e}")))?;
        if n == 0 {
            return Err(CopilotError::Http(
                "connection closed before headers".into(),
            ));
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 16_384 {
            return Err(CopilotError::Http("headers too large".into()));
        }
    }

    let header_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;

    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let headers = parse_headers(&header_str);

    // Parse request line.
    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = parts.first().copied().unwrap_or("");
    let path = parts.get(1).copied().unwrap_or("");

    // CORS preflight.
    if method == "OPTIONS" && path == "/rpc" {
        let resp = format!("HTTP/1.1 204 No Content\r\n{CORS_HEADERS}Connection: close\r\n\r\n");
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
        return Ok(());
    }

    if method != "POST" || path != "/rpc" {
        let code = if path == "/rpc" {
            "405 Method Not Allowed"
        } else {
            "404 Not Found"
        };
        let resp = format!("HTTP/1.1 {code}\r\nConnection: close\r\n\r\n");
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
        return Ok(());
    }

    // Auth check.
    if let Err(e) = check_bearer_token(&headers, auth_token) {
        tracing::warn!("auth rejected: {e}");
        let resp = "HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n";
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
        return Ok(());
    }

    // Read body based on Content-Length.
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    if content_length > max_payload {
        let resp = "HTTP/1.1 413 Payload Too Large\r\nConnection: close\r\n\r\n";
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
        return Ok(());
    }

    if content_length == 0 {
        let resp = "HTTP/1.1 400 Bad Request: empty body\r\nConnection: close\r\n\r\n";
        stream
            .write_all(resp.as_bytes())
            .await
            .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
        return Ok(());
    }

    // We already have some data after headers; count it.
    let already_read = buf.len() - header_end;
    let remaining = content_length.saturating_sub(already_read);
    if remaining > 0 {
        buf.resize(header_end + remaining, 0);
        stream
            .read_exact(&mut buf[header_end..])
            .await
            .map_err(|e| CopilotError::Http(format!("body read: {e}")))?;
    }

    let body = &buf[header_end..header_end + content_length];

    // Dispatch to handler.
    let resp_str = match handler.handle_request(std::str::from_utf8(body).unwrap_or("")) {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("internal error: {}", e.message);
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{msg}",
                msg.len()
            );
            stream
                .write_all(resp.as_bytes())
                .await
                .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
            return Ok(());
        }
    };

    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS_HEADERS}Connection: close\r\n\r\n{resp_str}",
        resp_str.len()
    );

    stream
        .write_all(resp.as_bytes())
        .await
        .map_err(|e| CopilotError::Http(format!("write: {e}")))?;
    Ok(())
}

fn parse_headers(header_str: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in header_str.lines().skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AnalyzeFormComponent;
    use crate::dispatcher::{FieldValueDispatcher, LocalDispatcher, TieredDispatcher};
    use crate::profile::Profile;
    use crate::server::handler::DaemonHandler;
    use dag::middleware::{RetryMiddleware, TimingMiddleware};
    use fluent_wvr::wrapper::MiddlewareChain;
    use fluent_wvr::Component;
    use std::sync::{Arc, RwLock};

    fn test_handler() -> Arc<DaemonHandler> {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let dispatcher: Arc<dyn FieldValueDispatcher> =
            Arc::new(TieredDispatcher::new().with(local));

        let base: Arc<dyn Component> = Arc::new(
            AnalyzeFormComponent::builder()
                .dispatcher(dispatcher)
                .profile(shared.clone())
                .build(),
        );
        let chain = MiddlewareChain::new()
            .push(Box::new(TimingMiddleware::new()))
            .push(Box::new(RetryMiddleware::new(2, 50)));
        let unit = chain.apply(base);

        Arc::new(DaemonHandler::new(shared, unit))
    }

    #[test]
    fn parse_headers_extracts_content_length() {
        let raw = "POST /rpc HTTP/1.1\r\nHost: 127.0.0.1:7182\r\nContent-Length: 42\r\n\r\n";
        let h = parse_headers(raw);
        assert_eq!(h.get("content-length").unwrap(), "42");
    }

    #[test]
    fn parse_headers_lowercases_keys() {
        let raw = "POST /rpc HTTP/1.1\r\nContent-Type: application/json\r\n\r\n";
        let h = parse_headers(raw);
        assert!(h.contains_key("content-type"));
    }

    #[test]
    fn parse_headers_empty() {
        let raw = "POST /rpc HTTP/1.1\r\n\r\n";
        let h = parse_headers(raw);
        assert!(h.is_empty());
    }

    #[tokio::test]
    async fn http_post_rpc_returns_json_response() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "daemon.health"
        });
        let resp = client
            .post(format!("http://{addr}/rpc"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert!(json["result"].is_object());
    }

    #[tokio::test]
    async fn http_options_returns_cors() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let resp = client
            .request(reqwest::Method::OPTIONS, format!("http://{addr}/rpc"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 204);
        assert_eq!(
            resp.headers().get("access-control-allow-origin").unwrap(),
            "*"
        );
    }

    #[tokio::test]
    async fn http_get_returns_405() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let resp = client
            .get(format!("http://{addr}/rpc"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 405);
    }

    #[tokio::test]
    async fn http_wrong_path_returns_404() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let resp = client
            .post(format!("http://{addr}/other"))
            .body("test")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn http_wrong_bearer_token_returns_401() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .auth_token("secret".to_string())
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let resp = client
            .post(format!("http://{addr}/rpc"))
            .bearer_auth("wrong")
            .json(&serde_json::json!({"jsonrpc":"2.0","id":1,"method":"daemon.health"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401);
    }

    #[tokio::test]
    async fn http_valid_bearer_token_returns_200() {
        let handler = test_handler();
        let config = DaemonConfig::new()
            .profile_path(std::path::PathBuf::from("/tmp/test-profile.toml"))
            .rest_port(0)
            .auth_token("secret".to_string())
            .build();
        let addr = run_http_once(&config, handler).await.unwrap();

        let client = common_core::http::test_http_client();
        let resp = client
            .post(format!("http://{addr}/rpc"))
            .bearer_auth("secret")
            .json(&serde_json::json!({"jsonrpc":"2.0","id":1,"method":"daemon.health"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }
}
