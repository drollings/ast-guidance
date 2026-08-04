use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::url::validate_https_or_local_http;
use async_trait::async_trait;
use common_core::hash::{content_hash_with_model, hex_encode};
use lru::LruCache;

const DEFAULT_CACHE_LIMIT: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("embedding request failed: {0}")]
    RequestFailed(String),
    #[error("invalid API URL")]
    InvalidApiUrl,
    #[error("insecure API URL")]
    InsecureApiUrl,
    #[error("SSRF blocked URL")]
    SsrfBlockedUrl,
    #[error("no API key provided")]
    NoApiKey,
    #[error("parse error: {0}")]
    ParseError(String),
}

#[derive(Debug, Clone)]
pub struct BatchEmbedding {
    pub flat: Vec<f32>,
    pub count: usize,
    pub dims: usize,
}

impl BatchEmbedding {
    pub fn vector(&self, i: usize) -> &[f32] {
        let start = i * self.dims;
        &self.flat[start..start + self.dims]
    }

    /// Checked accessor: returns `None` when `i` is out of bounds instead of
    /// panicking. Prefer over `vector` for input-derived indices.
    pub fn try_vector(&self, i: usize) -> Option<&[f32]> {
        let start = i.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        self.flat.get(start..end)
    }
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn dimensions(&self) -> u32;
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError>;

    async fn embed_async(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed(text)
    }

    async fn embed_batch_async(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        self.embed_batch(texts)
    }
}

pub struct NoopEmbedding {
    dims: u32,
}

impl NoopEmbedding {
    pub fn new(dims: u32) -> Self {
        Self { dims }
    }
}

impl EmbeddingProvider for NoopEmbedding {
    fn name(&self) -> &'static str {
        "none"
    }

    fn dimensions(&self) -> u32 {
        self.dims
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(Vec::new())
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let _ = texts;
        Ok(BatchEmbedding {
            flat: vec![],
            count: texts.len(),
            dims: 0,
        })
    }
}

pub struct OllamaEmbedding {
    base_url: String,
    model: String,
    dims: u32,
}

impl OllamaEmbedding {
    pub fn new(
        model: Option<&str>,
        base_url: Option<&str>,
        dims: u32,
    ) -> Result<Self, EmbeddingError> {
        let base_url = base_url.unwrap_or("http://localhost:11434").to_string();
        validate_url(&base_url)?;
        Ok(Self {
            base_url,
            model: model.unwrap_or("nomic-embed-text").to_string(),
            dims,
        })
    }

    pub fn embed_raw(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let resp_bytes = do_http_post(&url, &body, None)?;
        parse_ollama_response(&resp_bytes)
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbedding {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn dimensions(&self) -> u32 {
        self.dims
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_raw(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let inputs: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::Value::String(t.to_string()))
            .collect();
        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let resp_bytes = do_http_post(&url, &body, None)?;
        parse_ollama_batch_response(&resp_bytes)
    }

    async fn embed_async(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let resp_bytes = do_http_post_async(&url, &body, None).await?;
        parse_ollama_response(&resp_bytes)
    }

    async fn embed_batch_async(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let inputs: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::Value::String(t.to_string()))
            .collect();
        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });
        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let resp_bytes = do_http_post_async(&url, &body, None).await?;
        parse_ollama_batch_response(&resp_bytes)
    }
}

/// Generic caching wrapper around any EmbeddingProvider.
/// Uses content_hash_with_model for cache keys.
/// Bounded by an LRU cache (default 1024 entries).
pub struct CachedEmbeddingProvider<T: EmbeddingProvider> {
    inner: T,
    cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
}

impl<T: EmbeddingProvider> CachedEmbeddingProvider<T> {
    pub fn new(inner: T) -> Self {
        Self::new_with_limit(inner, DEFAULT_CACHE_LIMIT)
    }

    pub fn new_with_limit(inner: T, limit: usize) -> Self {
        let capacity = NonZeroUsize::new(limit)
            .or_else(|| NonZeroUsize::new(DEFAULT_CACHE_LIMIT))
            .unwrap_or(NonZeroUsize::MIN);
        Self {
            inner,
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }

    fn cache_key(&self, text: &str) -> String {
        let bytes = content_hash_with_model(text, self.inner.name());
        hex_encode(&bytes)
    }
}

#[async_trait]
impl<T: EmbeddingProvider + Send + Sync + 'static> EmbeddingProvider
    for CachedEmbeddingProvider<T>
{
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.cache_key(text);
        {
            let mut map = self
                .cache
                .lock()
                .map_err(|_| EmbeddingError::RequestFailed("cache lock".into()))?;
            if let Some(cached) = map.get(&key) {
                return Ok(cached.clone());
            }
        }
        let result = self.inner.embed(text)?;
        if let Ok(mut map) = self.cache.lock() {
            map.put(key, result.clone());
        }
        Ok(result)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        // M8.5: the sync variant is a thin runtime shim over the single async
        // core — the cache loop lives in exactly one place. Uses the canonical
        // `block_in_place` / fallback-runtime pattern from `do_http_post`
        // (which mirrors `client.rs::chat_complete`).
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                tokio::task::block_in_place(move || handle.block_on(self.embed_batch_async(texts)))
            }
            Err(_) => fallback_runtime().block_on(self.embed_batch_async(texts)),
        }
    }

    async fn embed_async(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let key = self.cache_key(text);
        {
            let mut map = self
                .cache
                .lock()
                .map_err(|_| EmbeddingError::RequestFailed("cache lock".into()))?;
            if let Some(cached) = map.get(&key) {
                return Ok(cached.clone());
            }
        }
        let result = self.inner.embed_async(text).await?;
        if let Ok(mut map) = self.cache.lock() {
            map.put(key, result.clone());
        }
        Ok(result)
    }

    async fn embed_batch_async(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let dims = self.dimensions() as usize;
        let mut flat = Vec::with_capacity(texts.len().saturating_mul(dims));
        let mut uncached_texts: Vec<&str> = Vec::new();
        let mut uncached_positions: Vec<usize> = Vec::new();
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                results[i] = Some(Vec::new());
                continue;
            }
            let key = self.cache_key(text);
            if let Ok(mut map) = self.cache.lock() {
                if let Some(cached) = map.get(&key) {
                    results[i] = Some(cached.clone());
                    continue;
                }
            }
            uncached_texts.push(text);
            uncached_positions.push(i);
        }

        if !uncached_texts.is_empty() {
            let batch = self.inner.embed_batch_async(&uncached_texts).await?;
            for (j, &pos) in uncached_positions.iter().enumerate() {
                let vec = batch
                    .try_vector(j)
                    .ok_or_else(|| {
                        EmbeddingError::ParseError(format!(
                            "batch returned fewer vectors than requested (requested {j})"
                        ))
                    })?
                    .to_vec();
                let key = self.cache_key(uncached_texts[j]);
                // M8.8: move the fresh vector into the result slot; the cache
                // receives a clone (the result slot must keep its value).
                if let Ok(mut map) = self.cache.lock() {
                    map.put(key, vec.clone());
                }
                results[pos] = Some(vec);
            }
        }

        for v in results.iter().flatten() {
            flat.extend_from_slice(v);
        }

        let count = results.len();
        let actual_dims = if count > 0 {
            results[0].as_ref().map_or(0, Vec::len)
        } else {
            0
        };
        Ok(BatchEmbedding {
            flat,
            count,
            dims: actual_dims,
        })
    }
}

pub struct OpenAiEmbedding {
    base_url: String,
    api_key: String,
    model: String,
    dims: u32,
    /// Model-level inference params merged into every embeddings request body
    /// (e.g. `num_ctx` for llama.cpp slot sizing). Kept on the embedding so the
    /// chart HNSW embedder reuses the same slot configuration as the chat
    /// classifier instead of opening a second default-context instance.
    params: Option<serde_json::Value>,
}

impl OpenAiEmbedding {
    pub fn new(
        model: Option<&str>,
        base_url: Option<&str>,
        api_key: Option<&str>,
        dims: u32,
    ) -> Result<Self, EmbeddingError> {
        Self::new_with_params(model, base_url, api_key, dims, None)
    }

    pub fn new_with_params(
        model: Option<&str>,
        base_url: Option<&str>,
        api_key: Option<&str>,
        dims: u32,
        params: Option<serde_json::Value>,
    ) -> Result<Self, EmbeddingError> {
        let base_url = base_url.unwrap_or("https://api.openai.com/v1").to_string();
        let api_key = api_key.ok_or(EmbeddingError::NoApiKey)?;
        validate_url(&base_url)?;
        Ok(Self {
            base_url,
            api_key: api_key.to_string(),
            model: model.unwrap_or("text-embedding-3-small").to_string(),
            dims,
            params,
        })
    }

    /// Build the request body merging model-level `params` into the standard
    /// `{"model", "input"}` shape. Core fields cannot be overwritten.
    fn request_body(&self, input: &serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": input,
        });
        if let Some(params) = self.params.as_ref() {
            if let Some(obj) = params.as_object() {
                for (k, v) in obj {
                    if k != "model" && k != "input" {
                        body[k] = v.clone();
                    }
                }
            }
        }
        body
    }

    fn embeddings_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/embeddings") {
            return base.to_string();
        }
        let has_path = base[8..].contains('/'); // after https:// or http://
        if has_path {
            format!("{base}/embeddings")
        } else {
            format!("{base}/v1/embeddings")
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbedding {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn dimensions(&self) -> u32 {
        self.dims
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.request_body(&serde_json::json!(text));
        let resp_bytes = do_http_post(&self.embeddings_url(), &body, Some(&self.api_key))?;
        parse_openai_response(&resp_bytes)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let inputs: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::Value::String(t.to_string()))
            .collect();
        let body = self.request_body(&serde_json::Value::Array(inputs));
        let resp_bytes = do_http_post(&self.embeddings_url(), &body, Some(&self.api_key))?;
        parse_openai_batch_response(&resp_bytes)
    }

    async fn embed_async(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.request_body(&serde_json::json!(text));
        let resp_bytes =
            do_http_post_async(&self.embeddings_url(), &body, Some(&self.api_key)).await?;
        parse_openai_response(&resp_bytes)
    }

    async fn embed_batch_async(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let inputs: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::Value::String(t.to_string()))
            .collect();
        let body = self.request_body(&serde_json::Value::Array(inputs));
        let resp_bytes =
            do_http_post_async(&self.embeddings_url(), &body, Some(&self.api_key)).await?;
        parse_openai_batch_response(&resp_bytes)
    }
}

fn async_http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

async fn do_http_post_async(
    url: &str,
    body: &serde_json::Value,
    auth_header: Option<&str>,
) -> Result<Vec<u8>, EmbeddingError> {
    let client = async_http_client();
    let mut req = client.post(url);
    req = req.header("Content-Type", "application/json");
    if let Some(token) = auth_header {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req
        .json(&body)
        .send()
        .await
        .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
    let status = resp.status();
    if status.as_u16() >= 400 {
        let text = resp.text().await.unwrap_or_default();
        return Err(EmbeddingError::RequestFailed(format!(
            "HTTP {status}: {text}",
        )));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))
}

/// Sync HTTP POST adapter for embedding calls.
///
/// Uses `tokio::task::block_in_place` when called from inside a tokio runtime
/// (which the coral L4 tier does) so it does not panic with
/// "Cannot start a runtime from within a runtime"; falls back to the
/// process-wide runtime when no runtime is active. The `client.rs`
/// `chat_complete` adapter owns the canonical pattern.
fn do_http_post(
    url: &str,
    body: &serde_json::Value,
    auth_header: Option<&str>,
) -> Result<Vec<u8>, EmbeddingError> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(move || {
            handle.block_on(do_http_post_async(url, body, auth_header))
        }),
        Err(_) => fallback_runtime().block_on(do_http_post_async(url, body, auth_header)),
    }
}

/// Fallback runtime used when the sync adapter is called with no active tokio
/// runtime (e.g. a plain `fn main()`). Mirrors `client.rs::fallback_runtime`.
fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build fallback tokio runtime for embeddings")
    })
}

fn parse_float_array(arr: &[serde_json::Value]) -> Result<Vec<f32>, EmbeddingError> {
    arr.iter()
        .map(|x| {
            x.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| EmbeddingError::ParseError("non-float in embedding".into()))
        })
        .collect()
}

fn parse_batch_embeddings(arrays: &[Vec<f32>]) -> BatchEmbedding {
    let count = arrays.len();
    let dims = arrays.first().map_or(0, Vec::len);
    let flat = arrays.iter().flatten().copied().collect();
    BatchEmbedding { flat, count, dims }
}

pub fn parse_ollama_response(json: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
    let embeddings = v["embeddings"]
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("missing embeddings array".into()))?;
    let first = embeddings
        .first()
        .ok_or_else(|| EmbeddingError::ParseError("empty embeddings array".into()))?;
    let arr = first
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("embedding is not an array".into()))?;
    parse_float_array(arr)
}

pub fn parse_ollama_batch_response(json: &[u8]) -> Result<BatchEmbedding, EmbeddingError> {
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
    let embeddings = v["embeddings"]
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("missing embeddings array".into()))?;
    let arrays: Result<Vec<Vec<f32>>, EmbeddingError> = embeddings
        .iter()
        .map(|emb| {
            let arr = emb
                .as_array()
                .ok_or_else(|| EmbeddingError::ParseError("embedding is not an array".into()))?;
            parse_float_array(arr)
        })
        .collect();
    Ok(parse_batch_embeddings(&arrays?))
}

pub fn parse_openai_response(json: &[u8]) -> Result<Vec<f32>, EmbeddingError> {
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
    let data = v["data"]
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("missing data array".into()))?;
    let first = data
        .first()
        .ok_or_else(|| EmbeddingError::ParseError("empty data array".into()))?;
    let embedding = first["embedding"]
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("missing embedding field".into()))?;
    parse_float_array(embedding)
}

pub fn parse_openai_batch_response(json: &[u8]) -> Result<BatchEmbedding, EmbeddingError> {
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
    let data = v["data"]
        .as_array()
        .ok_or_else(|| EmbeddingError::ParseError("missing data array".into()))?;
    let arrays: Result<Vec<Vec<f32>>, EmbeddingError> = data
        .iter()
        .map(|entry| {
            let embedding = entry["embedding"]
                .as_array()
                .ok_or_else(|| EmbeddingError::ParseError("missing embedding field".into()))?;
            parse_float_array(embedding)
        })
        .collect();
    Ok(parse_batch_embeddings(&arrays?))
}

fn validate_url(url: &str) -> Result<(), EmbeddingError> {
    use crate::url::UrlError;
    validate_https_or_local_http(url).map_err(|e| match e {
        UrlError::InvalidApiUrl => EmbeddingError::InvalidApiUrl,
        UrlError::InsecureApiUrl => EmbeddingError::InsecureApiUrl,
        UrlError::SsrfBlockedUrl => EmbeddingError::SsrfBlockedUrl,
    })
}

pub fn create_embedding_provider(
    name: &str,
    model: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
    dims: u32,
    cache_limit: Option<usize>,
    params: Option<&serde_json::Value>,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    let limit = cache_limit.unwrap_or(DEFAULT_CACHE_LIMIT);
    let provider: Box<dyn EmbeddingProvider> = match name {
        "none" => Box::new(NoopEmbedding::new(dims)),
        "ollama" => Box::new(CachedEmbeddingProvider::new_with_limit(
            OllamaEmbedding::new(model, base_url, dims)?,
            limit,
        )),
        "openai" => Box::new(CachedEmbeddingProvider::new_with_limit(
            OpenAiEmbedding::new_with_params(model, base_url, api_key, dims, params.cloned())?,
            limit,
        )),
        _ => {
            if let Some(ollama_model) = name.strip_prefix("ollama:") {
                Box::new(CachedEmbeddingProvider::new_with_limit(
                    OllamaEmbedding::new(Some(ollama_model), base_url, dims)?,
                    limit,
                ))
            } else if let Some(custom_url) = name.strip_prefix("custom:") {
                Box::new(CachedEmbeddingProvider::new_with_limit(
                    OpenAiEmbedding::new_with_params(
                        model,
                        Some(custom_url),
                        api_key,
                        dims,
                        params.cloned(),
                    )?,
                    limit,
                ))
            } else {
                return Err(EmbeddingError::UnknownProvider(name.to_string()));
            }
        }
    };
    Ok(provider)
}

#[cfg(test)]
mod tests {
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
    fn ollama_rejects_insecure_remote() {
        let result = OllamaEmbedding::new(None, Some("http://evil.com"), 0);
        assert!(result.is_err());
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
    fn openai_embeddings_url_standard() {
        let e = OpenAiEmbedding::new(
            None,
            Some("https://api.openai.com/v1"),
            Some("sk-test"),
            768,
        )
        .unwrap();
        assert_eq!(e.embeddings_url(), "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn openai_embeddings_url_already_embeddings() {
        let e = OpenAiEmbedding::new(
            None,
            Some("https://api.openai.com/v1/embeddings"),
            Some("sk-test"),
            768,
        )
        .unwrap();
        assert_eq!(e.embeddings_url(), "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn openai_embeddings_url_custom_path() {
        let e = OpenAiEmbedding::new(
            None,
            Some("https://my-server.com/custom/path"),
            Some("sk-test"),
            768,
        )
        .unwrap();
        assert_eq!(
            e.embeddings_url(),
            "https://my-server.com/custom/path/embeddings"
        );
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
    fn openai_embedding_dimensions() {
        let p = OpenAiEmbedding::new(
            None,
            Some("https://api.openai.com/v1"),
            Some("sk-test"),
            1536,
        )
        .unwrap();
        assert_eq!(p.dimensions(), 1536);
    }

    #[test]
    fn openai_embed_empty_text_returns_empty() {
        let p = OpenAiEmbedding::new(
            None,
            Some("https://api.openai.com/v1"),
            Some("sk-test"),
            768,
        )
        .unwrap();
        let vec = p.embed("").unwrap();
        assert!(vec.is_empty());
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
    async fn openai_embed_async_empty_text() {
        let p = OpenAiEmbedding::new(None, Some("https://api.openai.com/v1"), Some("sk-test"), 3)
            .unwrap();
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
}
