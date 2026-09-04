use super::*;
#[test]
fn create_noop_provider() {
    let p = create_embedding_provider("none", None, None, None, 768, None, None).unwrap();
    assert_eq!(p.name(), "none");
    assert_eq!(p.dimensions(), 768);
}

#[test]
fn create_unknown_provider() {
    let result = create_embedding_provider("bogus", None, None, None, 0, None, None);
    assert!(result.is_err());
}

#[test]
fn noop_embedding_returns_empty() {
    let p = NoopEmbedding::new(768);
    let vec = p.embed("hello").unwrap();
    assert!(vec.is_empty());
}

#[test]
fn noop_embed_batch() {
    let p = NoopEmbedding::new(768);
    let batch = p.embed_batch(&[]).unwrap();
    assert_eq!(batch.count, 0);
    assert_eq!(batch.dims, 0);
    assert!(batch.flat.is_empty());

    let batch = p.embed_batch(&["a"]).unwrap();
    assert_eq!(batch.count, 1);
    assert_eq!(batch.dims, 0);

    let batch = p.embed_batch(&["a", "b", "c"]).unwrap();
    assert_eq!(batch.count, 3);
}

#[test]
fn batch_embedding_vector_access() {
    let batch = BatchEmbedding {
        flat: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        count: 2,
        dims: 3,
    };
    assert_eq!(batch.vector(0), &[0.1, 0.2, 0.3]);
    assert_eq!(batch.vector(1), &[0.4, 0.5, 0.6]);
}

#[test]
fn batch_embedding_try_vector_bounds() {
    let batch = BatchEmbedding {
        flat: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        count: 2,
        dims: 3,
    };
    assert_eq!(batch.try_vector(0), Some(&[0.1, 0.2, 0.3][..]));
    assert_eq!(batch.try_vector(1), Some(&[0.4, 0.5, 0.6][..]));
    assert_eq!(batch.try_vector(2), None);
    assert_eq!(batch.try_vector(100), None);
}

#[test]
fn ollama_embedding_init() {
    let p = OllamaEmbedding::new(Some("llama3"), Some("http://localhost:11434"), 4096).unwrap();
    assert_eq!(p.name(), "ollama");
}

#[test]
fn parse_ollama_response_valid() {
    let json = br#"{"embeddings": [[0.1, 0.2, 0.3]]}"#;
    let vec = parse_ollama_response(json).unwrap();
    assert_eq!(vec.len(), 3);
    assert!((vec[0] - 0.1).abs() < 1e-6);
}

#[test]
fn parse_ollama_response_truncated_json() {
    let json = br#"{"embeddings": ["#;
    let result = parse_ollama_response(json);
    assert!(result.is_err());
}

#[test]
fn parse_ollama_response_wrong_structure() {
    let json = br#"{"foo": "bar"}"#;
    let result = parse_ollama_response(json);
    assert!(result.is_err());
}

#[test]
fn parse_ollama_batch_response_valid() {
    let json = br#"{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#;
    let batch = parse_ollama_batch_response(json).unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    assert_eq!(batch.flat.len(), 4);
    assert!((batch.vector(0)[0] - 0.1).abs() < 1e-6);
    assert!((batch.vector(1)[1] - 0.4).abs() < 1e-6);
}

#[test]
fn parse_openai_response_valid() {
    let json = br#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}"#;
    let vec = parse_openai_response(json).unwrap();
    assert_eq!(vec.len(), 3);
}

#[test]
fn parse_openai_response_truncated_json() {
    let json = br#"{"data": ["#;
    let result = parse_openai_response(json);
    assert!(result.is_err());
}

#[test]
fn parse_openai_batch_response_valid() {
    let json = br#"{"data": [{"embedding": [0.1, 0.2], "index": 0}, {"embedding": [0.3, 0.4], "index": 1}]}"#;
    let batch = parse_openai_batch_response(json).unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    assert_eq!(batch.flat.len(), 4);
}

#[test]
fn factory_ollama_prefix() {
    let p = create_embedding_provider(
        "ollama:llama3",
        None,
        Some("http://localhost:11434"),
        None,
        4096,
        None,
        None,
    )
    .unwrap();
    assert_eq!(p.name(), "ollama");
}

#[test]
fn factory_custom_prefix() {
    let p = create_embedding_provider(
        "custom:https://upstream.test:8080",
        None,
        None,
        Some("sk-test"),
        768,
        None,
        None,
    )
    .unwrap();
    assert_eq!(p.name(), "openai");
}

#[test]
fn content_hash_deterministic_and_model_sensitive() {
    let h1 = content_hash_with_model("text", "model-a");
    let h2 = content_hash_with_model("text", "model-a");
    assert_eq!(h1, h2);
    let h3 = content_hash_with_model("text", "model-b");
    assert_ne!(h1, h3);
}

#[test]
fn empty_text_returns_empty_skip_http() {
    let p = OllamaEmbedding::new(None, Some("http://localhost:11434"), 768).unwrap();
    let vec = p.embed("").unwrap();
    assert!(vec.is_empty());
}

#[test]
fn ollama_embed_with_mock_http() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2, 0.3]]}"#);
    });
    let p = OllamaEmbedding::new(Some("test"), Some(&server.url("")), 3).unwrap();
    let vec = p.embed("hello").unwrap();
    assert_eq!(vec.len(), 3);
    assert!((vec[0] - 0.1).abs() < 1e-6);
    mock.assert();
}

#[test]
fn ollama_embed_http_error() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(500).body("Internal Server Error");
    });
    let p = OllamaEmbedding::new(Some("test"), Some(&server.url("")), 3).unwrap();
    let result = p.embed("hello");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn ollama_embed_batch_with_mock_http() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#);
    });
    let p = OllamaEmbedding::new(Some("test"), Some(&server.url("")), 2).unwrap();
    let batch = p.embed_batch(&["a", "b"]).unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    mock.assert();
}

#[test]
fn do_http_post_outside_runtime_uses_fallback() {
    // Plain #[test] thread has no active tokio runtime; do_http_post must
    // fall back to the process-wide runtime instead of panicking.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2]]}"#);
    });
    let body = serde_json::json!({"model": "test", "input": ["x"]});
    let bytes = do_http_post(&server.url("/api/embed"), &body, None).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&bytes).is_ok());
    mock.assert();
}

#[tokio::test(flavor = "multi_thread")]
async fn do_http_post_inside_runtime_uses_block_in_place() {
    // Inside a tokio runtime, do_http_post must use block_in_place, not
    // construct a second runtime (which would panic).
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2]]}"#);
    });
    let body = serde_json::json!({"model": "test", "input": ["x"]});
    let bytes = do_http_post(&server.url("/api/embed"), &body, None).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&bytes).is_ok());
    mock.assert();
}

#[test]
fn openai_embed_with_mock_http() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings")
            .header("Authorization", "Bearer sk-test");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}"#);
    });
    let p = OpenAiEmbedding::new(
        Some("text-embedding-3-small"),
        Some(&server.url("")),
        Some("sk-test"),
        3,
    )
    .unwrap();
    let vec = p.embed("hello").unwrap();
    assert_eq!(vec.len(), 3);
    mock.assert();
}

#[test]
fn openai_embed_http_error() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/v1/embeddings");
        then.status(401).body("Unauthorized");
    });
    let p = OpenAiEmbedding::new(None, Some(&server.url("")), Some("sk-bad"), 3).unwrap();
    let result = p.embed("hello");
    assert!(result.is_err());
    mock.assert();
}

#[test]
fn openai_embed_batch_with_mock_http() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2], "index": 0}, {"embedding": [0.3, 0.4], "index": 1}]}"#);
    });
    let p = OpenAiEmbedding::new(None, Some(&server.url("")), Some("sk-test"), 2).unwrap();
    let batch = p.embed_batch(&["a", "b"]).unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    mock.assert();
}

#[test]
fn parse_ollama_batch_empty() {
    let json = br#"{"embeddings": []}"#;
    let batch = parse_ollama_batch_response(json).unwrap();
    assert_eq!(batch.count, 0);
    assert_eq!(batch.dims, 0);
    assert!(batch.flat.is_empty());
}

#[test]
fn parse_openai_batch_empty() {
    let json = br#"{"data": []}"#;
    let batch = parse_openai_batch_response(json).unwrap();
    assert_eq!(batch.count, 0);
    assert_eq!(batch.dims, 0);
    assert!(batch.flat.is_empty());
}

#[tokio::test]
async fn test_ollama_embed_async_with_mock() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2, 0.3]]}"#);
    });
    let p = OllamaEmbedding::new(Some("test"), Some(&server.url("")), 3).unwrap();
    let vec = p.embed_async("hello").await.unwrap();
    assert_eq!(vec.len(), 3);
    mock.assert();
}

#[tokio::test]
async fn test_openai_embed_async_with_mock() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings")
            .header("Authorization", "Bearer sk-test");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}"#);
    });
    let p = OpenAiEmbedding::new(
        Some("text-embedding-3-small"),
        Some(&server.url("")),
        Some("sk-test"),
        3,
    )
    .unwrap();
    let vec = p.embed_async("hello").await.unwrap();
    assert_eq!(vec.len(), 3);
    mock.assert();
}

#[test]
fn noop_embedding_dimensions() {
    let p = NoopEmbedding::new(512);
    assert_eq!(p.dimensions(), 512);
}

#[test]
fn ollama_embedding_dimensions() {
    let p = OllamaEmbedding::new(None, Some("http://localhost:11434"), 768).unwrap();
    assert_eq!(p.dimensions(), 768);
}



#[test]
fn ollama_default_constructor() {
    let p = OllamaEmbedding::new(None, None, 768).unwrap();
    assert_eq!(p.name(), "ollama");
}

#[tokio::test]
async fn ollama_embed_async_empty_text() {
    let p = OllamaEmbedding::new(None, Some("http://localhost:11434"), 3).unwrap();
    let vec = p.embed_async("").await.unwrap();
    assert!(vec.is_empty());
}

#[tokio::test]
async fn noop_embed_async_delegates() {
    let p = NoopEmbedding::new(768);
    let vec = p.embed_async("hello").await.unwrap();
    assert!(vec.is_empty());
}

#[tokio::test]
async fn noop_embed_batch_async_delegates() {
    let p = NoopEmbedding::new(768);
    let batch = p.embed_batch_async(&["a", "b"]).await.unwrap();
    assert_eq!(batch.count, 2);
}

#[tokio::test]
async fn ollama_embed_batch_async_with_mock() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/api/embed");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}"#);
    });
    let p = OllamaEmbedding::new(Some("test"), Some(&server.url("")), 2).unwrap();
    let batch = p.embed_batch_async(&["a", "b"]).await.unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    mock.assert();
}

#[tokio::test]
async fn openai_embed_batch_async_with_mock() {
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2], "index": 0}, {"embedding": [0.3, 0.4], "index": 1}]}"#);
    });
    let p = OpenAiEmbedding::new(
        Some("text-embedding-3-small"),
        Some(&server.url("")),
        Some("sk-test"),
        2,
    )
    .unwrap();
    let batch = p.embed_batch_async(&["a", "b"]).await.unwrap();
    assert_eq!(batch.count, 2);
    assert_eq!(batch.dims, 2);
    mock.assert();
}

#[test]
fn create_provider_ollama_prefix_custom_url() {
    let result = create_embedding_provider(
        "ollama:llama3",
        None,
        Some("http://localhost:11434"),
        None,
        4096,
        None,
        None,
    );
    assert!(result.is_ok());
}

#[test]
fn cached_provider_evicts_when_full() {
    let limit = 5;
    let provider = CachedEmbeddingProvider::new_with_limit(NoopEmbedding::new(768), limit);

    // Insert `limit + 10` entries
    for i in 0..limit + 10 {
        let text = format!("text_{i}");
        let _ = provider.embed(&text);
    }

    let cache = provider.cache.lock().unwrap();
    assert_eq!(cache.len(), limit);
}

#[test]
fn openai_embed_forwards_slot_params_in_request_body() {
    // The chart HNSW embedder must send the same `num_ctx` slot sizing as
    // the chat classifier — otherwise the gguf server opens a second
    // default-context instance for the same model (pipeline.rs test
    // documents the same failure mode for chat).
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings")
            .body_contains("num_ctx")
            .body_contains("\"num_ctx\":98304");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}"#);
    });
    let p = OpenAiEmbedding::new_with_params(
        Some("text-embedding-3-small"),
        Some(&server.url("")),
        Some("sk-test"),
        3,
        Some(serde_json::json!({"num_ctx": 98304, "parallel": 3})),
    )
    .unwrap();
    let vec = p.embed("hello").unwrap();
    assert_eq!(vec.len(), 3);
    mock.assert();
}

#[test]
fn openai_embed_params_cannot_override_model_or_input() {
    // Core fields are authoritative: `params` may supply slot sizing but
    // must never clobber `model`/`input`.
    let server = httpmock::MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/embeddings")
            .body_contains("\"model\":\"text-embedding-3-small\"")
            .body_contains("\"input\":\"hello\"");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]}"#);
    });
    let p = OpenAiEmbedding::new_with_params(
        Some("text-embedding-3-small"),
        Some(&server.url("")),
        Some("sk-test"),
        3,
        Some(serde_json::json!({"model": "evil", "input": "evil", "num_ctx": 98304})),
    )
    .unwrap();
    let vec = p.embed("hello").unwrap();
    assert_eq!(vec.len(), 3);
    mock.assert();
}
