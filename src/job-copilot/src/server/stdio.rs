use std::io::{self, Read, Write};

use common_core::jsonrpc::{JsonRpcHandler, JsonRpcResponse};

use crate::config::DaemonConfig;
use crate::error::CopilotError;

/// Read a Native Messaging frame from the given reader.
///
/// Returns `None` on EOF (stdin closed), or `Some(body)` on success.
/// Rejects frames exceeding `max_nm_payload`.
fn read_frame(
    reader: &mut impl Read,
    max_nm_payload: usize,
) -> Result<Option<String>, CopilotError> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(CopilotError::Io(common_core::error::IoError(e))),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > max_nm_payload {
        return Err(CopilotError::NativeMessaging(format!(
            "frame too large: {len} (max {max_nm_payload})"
        )));
    }
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .map_err(|e| CopilotError::Io(common_core::error::IoError(e)))?;
    let body = String::from_utf8(body)
        .map_err(|e| CopilotError::NativeMessaging(format!("invalid UTF-8: {e}")))?;
    Ok(Some(body))
}

/// Write a Native Messaging frame to the given writer.
fn write_frame(writer: &mut impl Write, body: &str) -> Result<(), CopilotError> {
    let len = body.len() as u32;
    writer
        .write_all(&len.to_le_bytes())
        .map_err(|e| CopilotError::Io(common_core::error::IoError(e)))?;
    writer
        .write_all(body.as_bytes())
        .map_err(|e| CopilotError::Io(common_core::error::IoError(e)))?;
    writer
        .flush()
        .map_err(|e| CopilotError::Io(common_core::error::IoError(e)))?;
    Ok(())
}

/// Run the Native Messaging STDIO transport.
///
/// Reads frames from stdin, dispatches via `handler`, and writes response
/// frames to stdout. This is synchronous and intended to run inside
/// `tokio::task::spawn_blocking`.
pub fn run_native_messaging<H: JsonRpcHandler>(
    handler: &H,
    config: &DaemonConfig,
) -> Result<(), CopilotError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let max_nm_payload = config.max_nm_payload;

    loop {
        match read_frame(&mut stdin.lock(), max_nm_payload)? {
            None => return Ok(()),
            Some(body) => {
                let response = handler.handle_request(&body);
                match response {
                    Ok(resp_json) => {
                        write_frame(&mut stdout, &resp_json)?;
                    }
                    Err(jsonrpc_err) => {
                        let resp = JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: None,
                            result: None,
                            error: Some(jsonrpc_err),
                        };
                        let err_json = serde_json::to_string(&resp)
                            .map_err(|e| CopilotError::Internal(e.to_string()))?;
                        write_frame(&mut stdout, &err_json)?;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DaemonConfig;
    use crate::error::CopilotError;
    use common_core::jsonrpc::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

    struct EchoHandler;

    impl JsonRpcHandler for EchoHandler {
        fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError> {
            let req: JsonRpcRequest = serde_json::from_str(raw)?;
            let resp = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({"echo": req.method})),
                error: None,
            };
            Ok(serde_json::to_string(&resp).unwrap())
        }
    }

    fn make_config(max_nm_payload: usize) -> DaemonConfig {
        let dir = std::env::temp_dir();
        let profile = dir.join("test_profile.toml");
        if !profile.exists() {
            std::fs::write(&profile, "").unwrap();
        }
        DaemonConfig::new()
            .max_nm_payload(max_nm_payload)
            .profile_path(profile)
            .build()
    }

    #[test]
    fn read_frame_rejects_oversized_frame() {
        let len_bytes = 2_000_000u32.to_le_bytes();
        let mut input = Vec::new();
        input.extend_from_slice(&len_bytes);
        input.extend_from_slice(&vec![b'x'; 2_000_000]);

        let mut cursor = io::Cursor::new(input);
        let result = read_frame(&mut cursor, 1_000_000);
        match result {
            Err(CopilotError::NativeMessaging(msg)) => {
                assert!(msg.contains("frame too large"));
            }
            other => panic!("expected NativeMessaging error, got {other:?}"),
        }
    }

    #[test]
    fn read_frame_returns_none_on_empty() {
        let cursor = io::Cursor::new(Vec::new());
        let mut cursor = cursor;
        let result = read_frame(&mut cursor, 1_000_000).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn read_write_roundtrip() {
        let input_json = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let mut buf = Vec::new();
        // Write a frame
        write_frame(&mut buf, input_json).unwrap();
        // Read it back
        let mut cursor = io::Cursor::new(buf);
        let config = make_config(1_000_000);
        let body = read_frame(&mut cursor, config.max_nm_payload)
            .unwrap()
            .unwrap();
        assert_eq!(body, input_json);
    }

    #[test]
    fn frame_roundtrip_with_response() {
        let mut buf = Vec::new();
        write_frame(&mut buf, r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#).unwrap();

        let mut cursor = io::Cursor::new(buf);
        let config = make_config(1_000_000);
        let body = read_frame(&mut cursor, config.max_nm_payload)
            .unwrap()
            .unwrap();

        let handler = EchoHandler;
        let resp_json = handler.handle_request(&body).unwrap();

        let mut out_buf = Vec::new();
        write_frame(&mut out_buf, &resp_json).unwrap();

        let mut cursor = io::Cursor::new(out_buf);
        let resp_body = read_frame(&mut cursor, config.max_nm_payload)
            .unwrap()
            .unwrap();
        assert!(resp_body.contains("echo"));
    }

    #[test]
    fn handle_request_returns_valid_jsonrpc() {
        let handler = EchoHandler;
        let req = r#"{"jsonrpc":"2.0","id":42,"method":"test"}"#;
        let resp_str = handler.handle_request(req).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, Some(serde_json::json!(42)));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn handle_request_parse_error_returns_jsonrpc_error() {
        let handler = EchoHandler;
        let result = handler.handle_request("not json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, -32700);
    }
}
