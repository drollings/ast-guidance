// NOTE (ROADMAP_20260903_LLM M11): the `error_value` envelope golden moved to
// `fluent-llm --test openai` (canonical owner `fluent_llm::openai`) in M7,
// the CORS golden to the router `cors_headers_unchanged` test (canonical
// owner `fluent-router::server::cors`), and M11 deleted the
// `common_core::http` shims (with the shim-lock tests that pinned them).
// This file keeps the generic suites (`shared_http_client`, `json_body` —
// stay in `common-core`).

#[cfg(feature = "http")]
use common_core::http::*;


#[cfg(feature = "http")]
#[test]
fn shared_http_client_returns_same_instance() {
        let a = shared_http_client();
        let b = shared_http_client();
        assert!(std::ptr::eq(a, b));
}

#[cfg(feature = "http")]
#[test]
fn json_body_round_trips() {
        let v = serde_json::json!({"ok":1});
        assert_eq!(json_body(&v), r#"{"ok":1}"#);
}
