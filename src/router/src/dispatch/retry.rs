/// Retry an HTTP POST to an LLM-compatible endpoint with exponential backoff.
use std::time::Duration;

use guidance_llm::HttpClass;

use crate::error::ServerError;

/// Sends `POST url` with `body` as JSON, retrying on transient failures.
///
/// Returns `Ok(response)` on success (HTTP 2xx) or on a non-retryable HTTP
/// status — the caller inspects the response status to distinguish.  Returns
/// `Err` when all retry attempts are exhausted on retryable errors
/// (429, 5xx) or transport failures.
pub async fn retry_http_request(
    client: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    retry_count: u32,
    retry_base_interval_s: u64,
) -> Result<reqwest::Response, ServerError> {
    let max_attempts = (retry_count + 1).max(1);
    let mut last_err = String::new();

    for attempt in 0..max_attempts {
        let result = client.post(url).json(body).send().await;

        match result {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return Ok(response);
                }
                let class = HttpClass::from_status(status.as_u16());
                if class.is_retryable() && attempt + 1 < max_attempts {
                    last_err = format!("HTTP {status}");
                    let delay = retry_base_interval_s * 1000 * (1u64 << attempt);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    continue;
                }
                if class.is_retryable() {
                    last_err = format!("HTTP {status}");
                    break;
                }
                return Ok(response);
            }
            Err(e) => {
                last_err = format!("HTTP error: {e}");
                if attempt + 1 < max_attempts {
                    let delay = retry_base_interval_s * 1000 * (1u64 << attempt);
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    continue;
                }
                break;
            }
        }
    }

    Err(ServerError::Http(format!(
        "dispatch failed after {max_attempts} attempts: {last_err}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::net::TcpListener;

    /// A minimal HTTP server that responds with a given status code and body,
    /// then tracks how many requests it received.
    struct TestServer {
        addr: String,
        stop_tx: Option<tokio::sync::oneshot::Sender<()>>,
    }

    impl TestServer {
        async fn new(responses: Vec<http::StatusCode>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();

            let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel::<()>();

            tokio::spawn(async move {
                let mut response_iter = responses.into_iter().cycle();
                loop {
                    tokio::select! {
                        _ = &mut stop_rx => break,
                        result = listener.accept() => {
                            let (mut stream, _) = result.unwrap();
                            let status = response_iter.next().unwrap_or(http::StatusCode::OK);
                            let mut buf = [0u8; 4096];
                            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
                            let body = format!("{{ \"status\": {} }}", status.as_u16());
                            let resp = format!(
                                "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                                status.as_u16(),
                                status.canonical_reason().unwrap_or("Unknown"),
                                body.len(),
                                body
                            );
                            let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, resp.as_bytes()).await;
                        }
                    }
                }
            });

            TestServer {
                addr,
                stop_tx: Some(stop_tx),
            }
        }

        fn url(&self) -> String {
            format!("http://{}/chat/completions", self.addr)
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            if let Some(tx) = self.stop_tx.take() {
                let _ = tx.send(());
            }
        }
    }

    /// A shared client for the retry tests (reuse the connection pool rather
    /// than constructing a fresh `reqwest::Client` per test).
    fn test_client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn success_on_first_attempt() {
        let server = TestServer::new(vec![http::StatusCode::OK]).await;
        let client = test_client();
        let result =
            retry_http_request(&client, &server.url(), &json!({"model": "test"}), 2, 1).await;
        assert!(result.is_ok());
        assert!(result.unwrap().status().is_success());
    }

    #[tokio::test]
    async fn retry_on_500_then_succeed() {
        let server = TestServer::new(vec![
            http::StatusCode::INTERNAL_SERVER_ERROR,
            http::StatusCode::OK,
        ])
        .await;
        let client = test_client();
        let result =
            retry_http_request(&client, &server.url(), &json!({"model": "test"}), 2, 1).await;
        assert!(result.is_ok());
        assert!(result.unwrap().status().is_success());
    }

    #[tokio::test]
    async fn short_circuit_on_400() {
        let server = TestServer::new(vec![http::StatusCode::BAD_REQUEST]).await;
        let client = test_client();
        let result =
            retry_http_request(&client, &server.url(), &json!({"model": "test"}), 2, 1).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status(), http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn exhaust_retries_then_err() {
        let server = TestServer::new(vec![
            http::StatusCode::INTERNAL_SERVER_ERROR,
            http::StatusCode::INTERNAL_SERVER_ERROR,
            http::StatusCode::INTERNAL_SERVER_ERROR,
        ])
        .await;
        let client = test_client();
        let result =
            retry_http_request(&client, &server.url(), &json!({"model": "test"}), 1, 1).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed after"),
            "expected 'failed after', got: {msg}"
        );
    }

    #[tokio::test]
    async fn transport_error_triggers_retry() {
        // Point at a port that's not listening — will get connection refused.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap();
        let result = retry_http_request(
            &client,
            "http://127.0.0.1:1/chat/completions",
            &json!({"model": "test"}),
            1,
            1,
        )
        .await;
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed after"),
            "expected 'failed after', got: {msg}"
        );
    }
}
