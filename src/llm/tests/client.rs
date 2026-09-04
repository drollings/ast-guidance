use super::*;

#[test]
fn test_client_creation() {
    let client = LlmClient::new("http://localhost:11434/v1", "llama3");
    assert_eq!(client.model(), "llama3");
}

#[test]
fn test_chat_message_serde() {
    let msg = ChatMessage {
        role: "user".into(),
        content: "hello".into(),
    };
    let json = serde_json::to_string(&msg).unwrap();
    let msg2: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(msg.content, msg2.content);
}

#[test]
fn test_llm_config_builder() {
    let config = LlmConfig::new()
        .api_url("http://localhost:11434/v1".into())
        .model("llama3".into())
        .think(true)
        .timeout_ms(5000)
        .debug(true)
        .show_prompts(false)
        .build();
    assert_eq!(config.model, "llama3");
    assert_eq!(config.think, Some(true));
    assert_eq!(config.timeout_ms, 5000);
}

#[test]
fn test_client_with_config() {
    let config = LlmConfig::new()
        .api_url("http://localhost:11434/v1".into())
        .model("llama3".into())
        .build();
    let client = LlmClient::with_config(config);
    assert_eq!(client.model(), "llama3");
}

#[test]
fn test_merge_extra_params_none_both() {
    assert_eq!(merge_extra_params(None, None), None);
}

#[test]
fn test_merge_extra_params_base_only() {
    let base = serde_json::json!({"temperature": 0.5});
    assert_eq!(
        merge_extra_params(Some(&base), None),
        Some(serde_json::json!({"temperature": 0.5}))
    );
}

#[test]
fn test_merge_extra_params_extras_only() {
    let extras = serde_json::json!({"response_format": {"type": "json_object"}});
    assert_eq!(
        merge_extra_params(None, Some(&extras)),
        Some(extras.clone())
    );
}

#[test]
fn test_merge_extra_params_extras_win_on_conflict() {
    let base = serde_json::json!({"temperature": 0.5, "num_ctx": 8192});
    let extras = serde_json::json!({"temperature": 0.1});
    assert_eq!(
        merge_extra_params(Some(&base), Some(&extras)),
        Some(serde_json::json!({"temperature": 0.1, "num_ctx": 8192}))
    );
}

#[test]
fn test_merge_extra_params_merges_disjoint_keys() {
    let base = serde_json::json!({"temperature": 0.5});
    let extras = serde_json::json!({"response_format": {"type": "json_object"}});
    assert_eq!(
        merge_extra_params(Some(&base), Some(&extras)),
        Some(serde_json::json!({
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }))
    );
}

#[test]
fn test_merge_extra_params_non_object_base_degrades_to_empty() {
    // A non-object base must not panic or leak; extras still win.
    let base = serde_json::json!(42);
    let extras = serde_json::json!({"response_format": {"type": "json_object"}});
    assert_eq!(
        merge_extra_params(Some(&base), Some(&extras)),
        Some(extras.clone())
    );
}

#[test]
fn test_merge_extra_params_non_object_extras_ignored() {
    let base = serde_json::json!({"temperature": 0.5});
    let extras = serde_json::json!("not-an-object");
    assert_eq!(
        merge_extra_params(Some(&base), Some(&extras)),
        Some(serde_json::json!({"temperature": 0.5}))
    );
}

#[test]
fn test_chat_complete_with_extras_invokes_seam() {
    // The seam must be the same callable surface as chat_complete for a
    // queue-less client: it compiles and routes through the shared async
    // path (unreachable endpoint here, so we assert the error shape only).
    let client = LlmClient::new("http://127.0.0.1:1/v1", "llama3");
    let messages = vec![ChatMessage { role: "user".into(), content: "hi".into() }];
    let extras = serde_json::json!({"response_format": {"type": "json_object"}});
    let result = client.chat_complete_with_extras(&messages, &extras);
    assert!(result.is_err(), "unreachable endpoint must surface an error");
}

#[test]
fn test_model_name_strips_prefix() {
    assert_eq!(model_name("ollama:embeddinggemma"), "embeddinggemma");
    assert_eq!(model_name("model"), "model");
    assert_eq!(model_name("a:b:c"), "b:c");
    assert_eq!(model_name(""), "");
}

#[test]
fn test_strip_think_block_html() {
    let result = strip_think_block("<think>hidden</think>visible");
    assert_eq!(result, "visible");
}

#[test]
fn test_strip_think_block_bracket() {
    let result = strip_think_block("[THINK]hidden[/THINK]visible");
    assert_eq!(result, "visible");
}

#[test]
fn test_strip_think_block_no_tags() {
    let result = strip_think_block("no tags here");
    assert_eq!(result, "no tags here");
}

#[test]
fn test_strip_preamble_let_me() {
    let result = strip_preamble("let me explain\nfoo bar");
    assert_eq!(result, "foo bar");
}

#[test]
fn test_strip_preamble_here_is() {
    let result = strip_preamble("here is the answer\n42");
    assert_eq!(result, "42");
}

#[test]
fn test_strip_preamble_no_match() {
    let result = strip_preamble("hello world");
    assert_eq!(result, "hello world");
}

#[test]
fn test_is_malformed_response_empty() {
    assert!(is_malformed_response(""));
    assert!(is_malformed_response("   "));
}

#[test]
fn test_is_malformed_response_dangling_end() {
    assert!(is_malformed_response("something with"));
    assert!(is_malformed_response("answer is to"));
}

#[test]
fn test_is_malformed_response_ends_with_question() {
    assert!(is_malformed_response("what is this?"));
}

#[test]
fn test_is_malformed_response_generic_self_ref() {
    assert!(is_malformed_response("this function"));
}

#[test]
fn test_is_malformed_response_overly_generic() {
    assert!(is_malformed_response("function"));
    assert!(is_malformed_response("helper"));
}

#[test]
fn test_is_malformed_response_llm_preamble() {
    assert!(is_malformed_response(
        "here's a function that does something"
    ));
}

#[test]
fn test_is_malformed_response_valid() {
    assert!(!is_malformed_response(
        "Computes the SHA-256 hash of the input string."
    ));
}

#[test]
fn test_is_malformed_response_valid_long() {
    assert!(!is_malformed_response(
        "Parses command-line arguments and prints the result."
    ));
}

#[test]
fn test_extract_comment_tag() {
    let result = extract_comment_tag("prefix<comment>hello world</comment>suffix");
    assert_eq!(result, Some("hello world"));
}

#[test]
fn test_extract_comment_tag_no_match() {
    let result = extract_comment_tag("no tags here");
    assert_eq!(result, None);
}

#[test]
fn test_is_blank_or_plausible() {
    assert!(is_blank_or_plausible(""));
    assert!(is_blank_or_plausible("Computes the hash."));
    assert!(!is_blank_or_plausible("ab"));
    assert!(!is_blank_or_plausible("function"));
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "hello world"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    })
}

fn one_message() -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: "hi".into() }]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_http_happy_path() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/chat/completions");
        then.status(200).json_body(chat_body());
    });
    let client = LlmClient::new(&server.base_url(), "m");
    let result = client.chat_complete(&one_message()).expect("chat complete");
    assert_eq!(result, "hello world");
    mock.assert_hits(1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_http_rate_limited_is_retryable() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/chat/completions");
        then.status(503)
            .json_body(serde_json::json!({"error": {"message": "overloaded"}}));
    });
    let client = LlmClient::new(&server.base_url(), "m");
    let err = client.chat_complete(&one_message()).expect_err("rate limited");
    assert_eq!(err, LlmError::RateLimited);
    assert!(err.is_retryable());
    mock.assert_hits(1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_http_client_error_is_api() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method("POST").path("/chat/completions");
        then.status(404).body("not found");
    });
    let client = LlmClient::new(&server.base_url(), "m");
    let err = client.chat_complete(&one_message()).expect_err("404");
    assert!(matches!(err, LlmError::Api(_)));
    assert!(!err.is_retryable());
    mock.assert_hits(1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_empty_content_is_no_response() {
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.method("POST").path("/chat/completions");
        then.status(200).json_body(serde_json::json!({
            "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }));
    });
    let client = LlmClient::new(&server.base_url(), "m");
    let err = client.chat_complete(&one_message()).expect_err("empty content");
    assert_eq!(err, LlmError::NoResponse);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_malformed_json_is_api_error() {
    let server = httpmock::MockServer::start();
    server.mock(|when, then| {
        when.method("POST").path("/chat/completions");
        then.status(200).body("this is not json");
    });
    let client = LlmClient::new(&server.base_url(), "m");
    let err = client.chat_complete(&one_message()).expect_err("malformed");
    assert!(matches!(err, LlmError::Api(_)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chat_complete_unreachable_endpoint_is_http_error() {
    // A refused loopback port is a transport failure -> Http (retryable).
    let client = LlmClient::new("http://127.0.0.1:1", "m");
    let err = client.chat_complete(&one_message()).expect_err("refused");
    assert!(matches!(err, LlmError::Http(_)));
    assert!(err.is_retryable());
}
