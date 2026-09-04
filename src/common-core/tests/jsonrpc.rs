use common_core::jsonrpc::*;


#[test]
fn method_not_found_response() {
        let resp = method_not_found(Some(serde_json::json!(1)), "unknown");
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, Some(serde_json::json!(1)));
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("unknown"));
}

#[test]
fn response_roundtrip() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(42)),
            result: Some(serde_json::json!({"ok": true})),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: JsonRpcResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, Some(serde_json::json!(42)));
        assert!(parsed.result.is_some());
        assert!(parsed.error.is_none());
}
