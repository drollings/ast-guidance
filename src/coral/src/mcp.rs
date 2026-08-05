use std::path::Path;
use std::sync::Arc;

use common_core::jsonrpc::{JsonRpcError, JsonRpcHandler, JsonRpcRequest, JsonRpcResponse};
use fluent_types::ContentNode;
use serde::Deserialize;
use thiserror::Error;

use crate::cache::reactor::QueueReactor;
use crate::db::Library;

// Promoted to `common-core::constants` (ROADMAP_20260804_SHARED_CORE M4.3).
use common_core::constants::MAX_MCP_REQUEST_SIZE;

#[derive(Error, Debug)]
pub enum McpError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("library error: {0}")]
    Library(#[from] crate::db::LibraryError),
    #[error("method not found: {0}")]
    MethodNotFound(String),
}

common_core::impl_from_io_error!(McpError);

pub struct McpServer {
    library: Arc<Library>,
    reactor: Option<Arc<QueueReactor>>,
}

impl McpServer {
    pub fn new(library: Arc<Library>) -> Self {
        Self {
            library,
            reactor: None,
        }
    }

    /// Create an MCP server with a reactor for cache tier stats.
    pub fn with_reactor(library: Arc<Library>, reactor: Arc<QueueReactor>) -> Self {
        Self {
            library,
            reactor: Some(reactor),
        }
    }

    fn dispatch(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "coral_query" => self.handle_coral_query(request),
            "coral_insert" => self.handle_coral_insert(request),
            "coral_traverse" => self.handle_coral_traverse(request),
            "coral_stats" => self.handle_coral_stats(request),
            _ => common_core::jsonrpc::method_not_found(request.id.clone(), &request.method),
        }
    }

    fn handle_coral_query(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let name = request
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match self.library.find_node_by_name(name) {
            Ok(Some(node_id)) => {
                let node = self.library.get_node(node_id).ok().flatten();
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id.clone(),
                    result: Some(serde_json::json!({
                        "found": true,
                        "node_id": node_id.as_int(),
                        "node": node,
                    })),
                    error: None,
                }
            }
            Ok(None) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: Some(serde_json::json!({"found": false})),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                error: Some(JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }),
                result: None,
            },
        }
    }

    fn handle_coral_insert(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        // M8.8: deserialize from the borrowed `&Value` (Value implements
        // `serde::Deserializer`) instead of cloning the params first.
        let node: ContentNode = match request.params.as_ref().map(ContentNode::deserialize) {
            Some(Ok(n)) => n,
            Some(Err(e)) => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id.clone(),
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: format!("invalid params: {e}"),
                    }),
                    result: None,
                };
            }
            None => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id.clone(),
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "missing params".into(),
                    }),
                    result: None,
                };
            }
        };

        match self.library.insert_node(&node) {
            Ok(node_id) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: Some(serde_json::json!({"node_id": node_id.as_int()})),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                error: Some(JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }),
                result: None,
            },
        }
    }

    fn handle_coral_traverse(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let node_id = request
            .params
            .as_ref()
            .and_then(|p| p.get("node_id"))
            .and_then(serde_json::Value::as_i64)
            .map_or(
                fluent_types::NodeId::from_int(0),
                fluent_types::NodeId::from_int,
            );

        let max_depth = request
            .params
            .as_ref()
            .and_then(|p| p.get("max_depth"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(3) as u8;

        match self.library.traverse_from(node_id, max_depth) {
            Ok(nodes) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: Some(serde_json::json!({"nodes": nodes, "count": nodes.len()})),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                error: Some(JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }),
                result: None,
            },
        }
    }

    fn handle_coral_stats(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match &self.reactor {
            Some(reactor) => {
                let stats = reactor.coral_stats();
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id.clone(),
                    result: Some(serde_json::to_value(&stats).unwrap_or_default()),
                    error: None,
                }
            }
            None => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                error: Some(JsonRpcError {
                    code: -32000,
                    message: "no reactor configured".into(),
                }),
                result: None,
            },
        }
    }
}

impl JsonRpcHandler for McpServer {
    fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError> {
        // M9.2b: bound the incoming request size so an oversized payload is
        // rejected before JSON parsing.
        if raw.len() > MAX_MCP_REQUEST_SIZE {
            return Err(JsonRpcError {
                code: -32600,
                message: format!(
                    "request exceeds MAX_MCP_REQUEST_SIZE ({MAX_MCP_REQUEST_SIZE} bytes)"
                ),
            });
        }
        let request: JsonRpcRequest = serde_json::from_str(raw)?;
        let response = self.dispatch(&request);
        Ok(serde_json::to_string(&response)?)
    }
}

/// Open a library from a path (or create in-memory if not found) and serve MCP over STDIO.
pub fn serve_stdio_from_path(db_path: &Path) -> Result<(), McpError> {
    let lib = if db_path.exists() {
        Library::open(db_path)?
    } else {
        Library::open_in_memory()?
    };
    let server = McpServer::new(Arc::new(lib));
    common_core::jsonrpc::serve_stdio(&server).map_err(McpError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> McpServer {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        McpServer::new(lib)
    }

    #[test]
    fn test_method_not_found() {
        let server = make_server();
        let req = r#"{"jsonrpc":"2.0","method":"unknown","id":1}"#;
        let response = server.handle_request(req).expect("handle");
        let resp: JsonRpcResponse = serde_json::from_str(&response).expect("parse");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_coral_query_not_found() {
        let server = make_server();
        let req =
            r#"{"jsonrpc":"2.0","method":"coral_query","id":1,"params":{"name":"nonexistent"}}"#;
        let response = server.handle_request(req).expect("handle");
        let resp: serde_json::Value = serde_json::from_str(&response).expect("parse");
        assert_eq!(resp["result"]["found"], false);
    }

    #[test]
    fn test_coral_insert_and_query() {
        let server = make_server();
        let req = r#"{"jsonrpc":"2.0","method":"coral_insert","id":1,"params":{"name":"test_mcp","source":"test source","lod":[],"embedding":null}}"#;
        let response = server.handle_request(req).expect("handle");
        let resp: serde_json::Value = serde_json::from_str(&response).expect("parse");
        let node_id = resp["result"]["node_id"].as_i64().expect("node_id");
        assert!(node_id > 0);

        let query_req =
            r#"{"jsonrpc":"2.0","method":"coral_query","id":2,"params":{"name":"test_mcp"}}"#
                .to_string();
        let query_resp = server.handle_request(&query_req).expect("handle");
        let qr: serde_json::Value = serde_json::from_str(&query_resp).expect("parse");
        assert_eq!(qr["result"]["found"], true);
    }

    #[test]
    fn test_coral_traverse() {
        let server = make_server();
        let insert = r#"{"jsonrpc":"2.0","method":"coral_insert","id":1,"params":{"name":"root","source":"root","lod":[],"embedding":null}}"#;
        let resp = server.handle_request(insert).expect("handle");
        let rv: serde_json::Value = serde_json::from_str(&resp).expect("parse");
        let root_id = rv["result"]["node_id"].as_i64().expect("id");

        let traverse_req = format!(
            r#"{{"jsonrpc":"2.0","method":"coral_traverse","id":2,"params":{{"node_id":{root_id},"max_depth":3}}}}"#
        );
        let trav_resp = server.handle_request(&traverse_req).expect("handle");
        let tr: serde_json::Value = serde_json::from_str(&trav_resp).expect("parse");
        assert_eq!(tr["result"]["count"].as_i64(), Some(1));
    }

    #[test]
    fn test_serve_stdio_handles_multiple_lines() {
        let server = make_server();
        let req1 =
            r#"{"jsonrpc":"2.0","method":"coral_query","id":1,"params":{"name":"nonexistent"}}"#;
        let req2 = r#"{"jsonrpc":"2.0","method":"coral_insert","id":2,"params":{"name":"test_stdio","source":"test","lod":[],"embedding":null}}"#;

        let resp1 = server.handle_request(req1).expect("handle");
        let r1: JsonRpcResponse = serde_json::from_str(&resp1).expect("parse");
        assert_eq!(
            r1.result.as_ref().and_then(|r| r.get("found")),
            Some(&serde_json::json!(false))
        );

        let resp2 = server.handle_request(req2).expect("handle");
        let r2: JsonRpcResponse = serde_json::from_str(&resp2).expect("parse");
        assert!(r2.result.as_ref().and_then(|r| r.get("node_id")).is_some());
    }

    #[test]
    fn test_serve_stdio_parse_error_for_invalid_json() {
        let server = make_server();
        let resp = server.handle_request("not-json");
        assert!(resp.is_err());
    }

    #[test]
    fn test_coral_stats_no_reactor() {
        let server = make_server();
        let req = r#"{"jsonrpc":"2.0","method":"coral_stats","id":1}"#;
        let resp = server.handle_request(req).expect("handle");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("parse");
        assert!(v["error"].is_object());
        assert_eq!(v["error"]["code"], -32000);
    }

    #[test]
    fn test_coral_stats_with_reactor() {
        use crate::cache::{QueueReactor, QueueReactorCreateArgs};
        use std::sync::Arc;

        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let args = QueueReactorCreateArgs::builder()
            .library(Arc::clone(&lib))
            .build();
        let reactor = Arc::new(QueueReactor::new(args));
        let server = McpServer::with_reactor(lib, reactor);

        let req = r#"{"jsonrpc":"2.0","method":"coral_stats","id":1}"#;
        let resp = server.handle_request(req).expect("handle");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("parse");
        assert!(v["result"].is_object());
        assert!(v["error"].is_null());
        // No routes yet, so total_count should be 0
        assert_eq!(v["result"]["total_count"], 0);
        // But we should have at least one tier histogram (L3)
        assert!(v["result"]["tier_count"].as_u64().unwrap() > 0);
    }
}
