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
        // The sync variant is a thin runtime shim over the single async
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
                // Move the fresh vector into the result slot; the cache
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
#[path = "../tests/embeddings.rs"]
mod tests;
