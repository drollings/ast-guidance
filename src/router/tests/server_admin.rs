use std::sync::Arc;

use super::*;

#[test]
fn url_origin_extracts_host_port() {
    assert_eq!(
        url_origin("http://127.0.0.1:1234/v1/chat/completions").as_deref(),
        Some("http://127.0.0.1:1234")
    );
    assert_eq!(
        url_origin("https://api.example.com/v1/chat/completions").as_deref(),
        Some("https://api.example.com")
    );
    assert_eq!(url_origin("not a url"), None);
}

#[test]
fn dedupe_metric_docs_keeps_first_help_and_type() {
    let body1 = "# HELP a count\n# TYPE a counter\na 1\n".to_string();
    let body2 = "# HELP a count\n# TYPE a counter\n# TYPE b gauge\nb 2\n".to_string();
    let merged = dedupe_metric_docs(vec![body1, body2]);
    assert_eq!(merged.matches("# HELP a").count(), 1);
    assert_eq!(merged.matches("# TYPE a").count(), 1);
    assert_eq!(merged.matches("# TYPE b").count(), 1);
    assert!(merged.contains("a 1"));
    assert!(merged.contains("b 2"));
}

#[tokio::test]
async fn unload_refuses_always_resident_onnx_model() {
    use fluent_onnx::{OnnxConfig, OnnxTask, OrtSessionRegistry};

    let config = crate::tests::common::make_config(
        "http://127.0.0.1:1/v1/chat/completions",
        true,
        false,
        60_000,
        30_000,
    );
    // An Always-resident (resident default true) onnx model.
    let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
    registry
        .register(
            "onnx-encoder",
            OnnxConfig::new()
                .model_path("/models/encoder.onnx")
                .tokenizer_path("/models/tokenizer.json")
                .task(OnnxTask::FillMask)
                .build(),
        )
        .expect("register");

    let mut deps = crate::tests::common::test_deps(
        Arc::new(std::collections::HashMap::new()),
        &config,
        None,
        None,
        None,
        std::collections::HashMap::new(),
        None,
    );
    deps.onnx = Some(Arc::new(registry));

    let server = crate::tests::common::spawn_test_server_with_deps(deps).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/models/unload", server.base_url()))
        .json(&serde_json::json!({ "model": "onnx-encoder" }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), hyper::StatusCode::CONFLICT);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("cannot be unloaded")
    );
}

/// Stub loader for the hermetic unload-refusal test (no real ort session).
#[derive(Default)]
struct StubLoader;

impl fluent_onnx::SessionLoader for StubLoader {
    fn load(
        &self,
        _config: &fluent_onnx::OnnxConfig,
        _model_key: &str,
    ) -> Result<fluent_onnx::SessionHandle, fluent_onnx::OrtError> {
        Ok(fluent_onnx::SessionHandle::new("stub"))
    }
}
