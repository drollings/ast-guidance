//! Capability-gated network I/O (TCP connect, HTTP GET/POST, streaming).

use std::time::Duration;

use bytes::Bytes;
use common_core::error::IoError;
use fluent_wvr::Capability;
use futures_util::stream::{Stream, StreamExt};
use tokio::net::{TcpStream, ToSocketAddrs};

use crate::io::check_capability;

/// Capability-gated network operations.
pub struct NetCapability {
    client: reqwest::Client,
}

impl NetCapability {
    pub fn new() -> Self {
        Self::with_config(&NetConfig::default())
    }

    pub fn with_config(config: &NetConfig) -> Self {
        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(config.max_idle_per_host)
            .pool_idle_timeout(config.idle_timeout)
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout);

        if let Some(user_agent) = &config.user_agent {
            builder = builder.user_agent(user_agent);
        }

        let client = builder.build().expect("failed to build reqwest client");

        Self { client }
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

pub struct NetConfig {
    pub max_idle_per_host: usize,
    pub idle_timeout: Duration,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub user_agent: Option<String>,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            max_idle_per_host: 4,
            idle_timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            user_agent: None,
        }
    }
}

impl Default for NetCapability {
    fn default() -> Self {
        Self::new()
    }
}

impl Capability for NetCapability {
    fn name(&self) -> &'static str {
        "net"
    }
}

impl NetCapability {
    pub async fn tcp_connect(&self, addr: impl ToSocketAddrs) -> Result<TcpStream, IoError> {
        check_capability(self)?;
        Ok(TcpStream::connect(addr).await?)
    }

    pub async fn http_get(&self, url: &str) -> Result<String, IoError> {
        check_capability(self)?;
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(std::io::Error::other)?;
        let body = response.text().await.map_err(std::io::Error::other)?;
        Ok(body)
    }

    pub async fn http_post(&self, url: &str, body: &str) -> Result<String, IoError> {
        check_capability(self)?;
        let response = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(std::io::Error::other)?;
        let response_body = response.text().await.map_err(std::io::Error::other)?;
        Ok(response_body)
    }

    /// POST a JSON body and return a streaming `Stream<Item = Result<Bytes, IoError>>`
    /// of response body chunks. Use this when the caller wants incremental
    /// delivery (e.g. SSE forwarding) rather than buffering the entire response.
    ///
    /// The capability is checked once, before the request is dispatched. The
    /// returned stream surfaces transport errors from `reqwest` as `IoError`
    /// items; the caller is responsible for parsing and idle-timeout
    /// enforcement, because those are domain-specific.
    pub async fn http_post_json_stream(
        &self,
        url: &str,
        body: &str,
    ) -> Result<impl Stream<Item = Result<Bytes, IoError>> + Send + 'static, IoError> {
        check_capability(self)?;
        let response = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(std::io::Error::other)?;

        let status = response.status();
        if !status.is_success() {
            return Err(IoError(std::io::Error::other(format!(
                "HTTP {status} from {url}"
            ))));
        }

        let mapped = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| std::io::Error::other(e).into()));
        Ok(mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::CURRENT_CAPS;
    use fluent_wvr::CapabilitySet;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn net_config_defaults() {
        let cfg = NetConfig::default();
        assert_eq!(cfg.max_idle_per_host, 4);
        assert_eq!(cfg.idle_timeout, Duration::from_secs(30));
        assert_eq!(cfg.connect_timeout, Duration::from_secs(10));
        assert_eq!(cfg.request_timeout, Duration::from_secs(30));
        assert!(cfg.user_agent.is_none());
    }

    #[test]
    fn net_capability_default_via_default_trait() {
        let _cap = NetCapability::default();
        let _cap2 = NetCapability::new();
    }

    #[tokio::test(start_paused = true)]
    async fn tcp_connect_echo_roundtrip() {
        // A `TcpListener`-backed echo server on loopback: the capability-gated
        // `tcp_connect` must establish the connection and round-trip bytes.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let echo = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.expect("accept");
            let mut buf = [0u8; 64];
            let n = sock.read(&mut buf).await.expect("read");
            sock.write_all(&buf[..n]).await.expect("echo");
        });

        let net = NetCapability::new();
        let caps = CapabilitySet::new().with(NetCapability::new());
        let echoed = CURRENT_CAPS
            .scope(caps, async {
                let mut stream = net.tcp_connect(addr).await.expect("connect");
                stream.write_all(b"ping").await.expect("write");
                let mut buf = [0u8; 4];
                stream.read_exact(&mut buf).await.expect("read echo");
                buf.to_vec()
            })
            .await;
        echo.await.expect("echo task");
        assert_eq!(echoed, b"ping".to_vec());
    }

    #[tokio::test(start_paused = true)]
    async fn tcp_connect_refused_errors() {
        // `127.0.0.1:1` is a refused loopback port: the connect must error, not
        // hang or fire anything.
        let net = NetCapability::new();
        let caps = CapabilitySet::new().with(NetCapability::new());
        let result = CURRENT_CAPS
            .scope(caps, async { net.tcp_connect("127.0.0.1:1").await })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn tcp_connect_without_capability_is_denied() {
        // Outside a `CURRENT_CAPS` scope the capability check fails with
        // `PermissionDenied` before any network I/O occurs.
        let net = NetCapability::new();
        let err = net.tcp_connect("127.0.0.1:1").await.expect_err("denied");
        assert_eq!(err.0.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(err.0.to_string().contains("missing capability"));
    }

    #[test]
    fn net_capability_name() {
        assert_eq!(NetCapability::new().name(), "net");
    }
}
