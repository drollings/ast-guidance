//! Shared JSON-RPC 2.0 types and STDIO transport.
//!
//! Deduplicates `JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcError`, and the
//! line-delimited STDIO read/write loop that was previously copy-pasted in
//! `bin/guidance/src/mcp.rs` and `coral/src/mcp.rs`.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// JSON-RPC 2.0 request object.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub method: String,
    pub id: Option<serde_json::Value>,
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 response object.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl From<serde_json::Error> for JsonRpcError {
    fn from(e: serde_json::Error) -> Self {
        JsonRpcError {
            code: -32700,
            message: format!("parse error: {e}"),
        }
    }
}

/// Standard JSON-RPC "method not found" error code.
pub const METHOD_NOT_FOUND: i32 = -32601;

/// Build a JSON-RPC error response for a method-not-found condition.
pub fn method_not_found(id: Option<serde_json::Value>, method: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code: METHOD_NOT_FOUND,
            message: format!("method not found: {method}"),
        }),
    }
}

/// Trait for handling JSON-RPC requests.
///
/// Implementors define per-method dispatch. The `serve_stdio` function drives
/// the line-delimited STDIO loop and calls `handle_request` for each incoming
/// JSON object.
///
/// # Examples
///
/// ```no_run
/// use common_core::jsonrpc::{JsonRpcHandler, JsonRpcRequest, JsonRpcResponse, JsonRpcError};
///
/// struct EchoHandler;
///
/// impl JsonRpcHandler for EchoHandler {
///     fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError> {
///         let req: JsonRpcRequest = serde_json::from_str(raw)
///             .map_err(|e| JsonRpcError { code: -32700, message: format!("parse: {e}") })?;
///         let resp = JsonRpcResponse {
///             jsonrpc: "2.0".into(),
///             id: req.id,
///             result: Some(serde_json::json!({ "echo": req.params })),
///             error: None,
///         };
///         serde_json::to_string(&resp)
///             .map_err(|e| JsonRpcError { code: -32603, message: format!("serialize: {e}") })
///     }
/// }
/// ```
pub trait JsonRpcHandler: Send + Sync {
    /// Process a raw JSON-RPC request and return a serialized response.
    fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError>;
}

/// Drive a line-delimited JSON-RPC 2.0 server over STDIO.
///
/// Reads lines from stdin, calls `handler.handle_request` for each non-empty
/// line, and writes the response to stdout. Returns when stdin is closed.
///
/// # Examples
///
/// ```no_run
/// use common_core::jsonrpc::{serve_stdio, JsonRpcHandler, JsonRpcError,
///     JsonRpcRequest, JsonRpcResponse, method_not_found};
///
/// struct MyHandler;
/// impl JsonRpcHandler for MyHandler {
///     fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError> {
///         let req: JsonRpcRequest = serde_json::from_str(raw)
///             .map_err(|e| JsonRpcError { code: -32700, message: format!("parse: {e}") })?;
///         match req.method.as_str() {
///             "ping" => {
///                 let resp = JsonRpcResponse {
///                     jsonrpc: "2.0".into(), id: req.id,
///                     result: Some(serde_json::json!("pong")), error: None,
///                 };
///                 Ok(serde_json::to_string(&resp).unwrap())
///             }
///             _ => Ok(serde_json::to_string(&method_not_found(req.id, &req.method)).unwrap()),
///         }
///     }
/// }
///
/// serve_stdio(&MyHandler).unwrap();
/// ```
pub fn serve_stdio<H: JsonRpcHandler>(handler: &H) -> Result<(), IoError> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        match handler.handle_request(&trimmed) {
            Ok(response) => {
                writeln!(stdout, "{response}")?;
                stdout.flush()?;
            }
            // Parse/deserialization errors — write a JSON-RPC error response
            // so the client gets something structured instead of silence.
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: None,
                    error: Some(e),
                };
                let json = serde_json::to_string(&resp).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "failed to serialize error")
                })?;
                writeln!(stdout, "{json}")?;
                stdout.flush()?;
            }
        }
    }

    Ok(())
}

