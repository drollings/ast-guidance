use std::sync::Arc;

use common_core::http::shared_http_client;

use crate::protocol::{ChatMessage, LlmConfig, LlmError, LlmRequestQueue};

use crate::http_class::HttpClass;

/// Trait for chat backends — sends messages and returns a response string.
///
/// Implemented by `LlmClient` (production) and test stubs.
pub trait ChatBackend: Send + Sync {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError>;

    /// Chat completion with per-call extra body parameters merged on top of
    /// the backend's configured defaults (e.g. `response_format` for
    /// constrained decoding). The default implementation ignores `extras` and
    /// delegates to [`chat_complete`](Self::chat_complete), so stubs, mock
    /// backends, and transports without a per-call extras seam keep working
    /// unchanged — only backends that honor per-call extras need to override.
    fn chat_complete_with_extras(
        &self,
        messages: &[ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, LlmError> {
        let _ = extras;
        self.chat_complete(messages)
    }
}

pub struct LlmClient {
    pub api_base: String,
    pub model: String,
    pub config: LlmConfig,
    pub queue: Option<Arc<LlmRequestQueue>>,
}

impl Clone for LlmClient {
    fn clone(&self) -> Self {
        Self {
            api_base: self.api_base.clone(),
            model: self.model.clone(),
            config: self.config.clone(),
            queue: self.queue.clone(),
        }
    }
}

impl LlmClient {
    pub fn new(api_base: &str, model: &str) -> Self {
        let config = LlmConfig::new()
            .api_url(api_base.to_string())
            .model(model.to_string())
            .build();
        Self {
            api_base: api_base.trim_end_matches('/').to_string(),
            model: model.to_string(),
            config,
            queue: None,
        }
    }

    pub fn with_queue(api_base: &str, model: &str, queue: Arc<LlmRequestQueue>) -> Self {
        let config = LlmConfig::new()
            .api_url(api_base.to_string())
            .model(model.to_string())
            .build();
        Self {
            api_base: api_base.trim_end_matches('/').to_string(),
            model: model.to_string(),
            config,
            queue: Some(queue),
        }
    }

    /// Attaches a pre-built `LlmRequestQueue` (worker pool) and a fully
    /// resolved `LlmConfig` (think flag, debug, show_prompts, timeout).
    /// Prefer this over [`with_queue`](Self::with_queue) when the config
    /// carries non-default fields, since `with_queue` rebuilds a default
    /// `LlmConfig` and silently drops `think`, `debug`, etc.
    pub fn with_queue_and_config(queue: Arc<LlmRequestQueue>, config: LlmConfig) -> Self {
        let api_base = config.api_url.trim_end_matches('/').to_string();
        let model = config.model.clone();
        Self {
            api_base,
            model,
            config,
            queue: Some(queue),
        }
    }

    pub fn with_config(config: LlmConfig) -> Self {
        let api_base = config.api_url.trim_end_matches('/').to_string();
        let model = config.model.clone();
        Self {
            api_base,
            model,
            config,
            queue: None,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Native async chat completion. Uses the caller's tokio runtime and a
    /// shared `reqwest::Client`. If a custom `LlmRequestQueue` is configured,
    /// the request is submitted through that queue's worker pool; otherwise
    /// the HTTP call is made directly.
    ///
    /// When a queue is attached, the call awaits the queue's worker future
    /// — this is fully async, so it composes correctly with `tokio::spawn`,
    /// `Scope::spawn`, and `Limiter::run` without requiring `Handle::block_on`.
    pub async fn chat_complete_async(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.chat_complete_with_extra_async(messages, None).await
    }

    /// Async chat completion with per-call extra body parameters merged on top
    /// of the client's configured `extra_body_params`. This is the seam that
    /// lets callers request constrained decoding (`response_format` /
    /// `json_schema`) for a single call without changing the shared config.
    pub async fn chat_complete_with_extras_async(
        &self,
        messages: &[ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, LlmError> {
        self.chat_complete_with_extra_async(messages, Some(extras)).await
    }

    /// Core async chat completion. `extras` (per-call body params) are merged
    /// on top of the client's configured `extra_body_params`; `None` sends the
    /// configured defaults unchanged.
    async fn chat_complete_with_extra_async(
        &self,
        messages: &[ChatMessage],
        extras: Option<&serde_json::Value>,
    ) -> Result<String, LlmError> {
        let merged = merge_extra_params(self.config.extra_body_params.as_ref(), extras);
        match &self.queue {
            Some(q) => {
                let mut config = self.config.clone();
                config.extra_body_params = merged;
                let task = crate::protocol::LlmTask {
                    messages: messages.to_vec(),
                    config,
                };
                q.submit(task).await
            }
            None => {
                chat_complete_http_async(
                    &self.api_base,
                    messages,
                    &self.model,
                    self.config.think,
                    self.config.timeout_ms,
                    merged.as_ref(),
                    self.config.debug,
                    self.config.show_prompts,
                )
                .await
            }
        }
    }

    /// Sync adapter for `chat_complete_async`. Uses `tokio::task::block_in_place`
    /// when inside a tokio runtime to prevent worker-thread starvation; falls
    /// back to the process-wide fallback runtime when no runtime is active.
    pub fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let client = self.clone();
        let messages = messages.to_vec();
        block_on(client.chat_complete_async(&messages))
    }

    /// Sync adapter for `chat_complete_with_extras_async` — the per-call
    /// extras seam for synchronous callers (e.g. the classifier stage).
    pub fn chat_complete_with_extras(
        &self,
        messages: &[ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, LlmError> {
        let client = self.clone();
        let messages = messages.to_vec();
        let extras = extras.clone();
        block_on(client.chat_complete_with_extras_async(&messages, &extras))
    }
}

impl ChatBackend for LlmClient {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.chat_complete(messages)
    }

    fn chat_complete_with_extras(
        &self,
        messages: &[ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, LlmError> {
        self.chat_complete_with_extras(messages, extras)
    }
}

/// Merge per-call `extras` on top of the client's configured `base`
/// `extra_body_params` (extras win on key conflicts). Returns `None` only when
/// both are absent; non-object values degrade to an empty object so the merge
/// never drops configured params.
fn merge_extra_params(
    base: Option<&serde_json::Value>,
    extras: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let mut merged = base
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(extras) = extras {
        if let Some(obj) = extras.as_object() {
            for (k, v) in obj {
                merged.insert(k.clone(), v.clone());
            }
        }
    }
    if merged.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(merged))
    }
}

/// Run a future to completion synchronously, reusing the canonical runtime
/// bridging: inside a multi-threaded tokio runtime it drives the future via
/// `tokio::task::block_in_place` (avoiding worker-thread starvation); inside a
/// current-thread runtime (where `block_in_place` panics) and with no active
/// runtime it falls back to plain `block_on` / the process-wide fallback
/// runtime.
///
/// The implementation delegates to `common_core::runtime::block_on` — the
/// workspace's single canonical sync→async bridge.
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    common_core::runtime::block_on(fut)
}

/// Async chat completion HTTP call. Honors `LlmConfig::timeout_ms` via
/// `tokio::time::timeout` on the whole request/response.
///
/// `extra_body_params` are merged into the HTTP request body to supply
/// model-level inference parameters (e.g. `num_ctx`, `temperature`).
pub async fn chat_complete_http_async(
    api_base: &str,
    messages: &[ChatMessage],
    model: &str,
    think: Option<bool>,
    timeout_ms: u64,
    extra_body_params: Option<&serde_json::Value>,
    debug: bool,
    show_prompts: bool,
) -> Result<String, LlmError> {
    let url = crate::url::chat_completions_url(api_base);
    let result = chat_complete_http_inner_async(
        &url,
        messages,
        model,
        think,
        timeout_ms,
        extra_body_params,
        debug,
        show_prompts,
    )
    .await?;
    if result.is_empty() {
        Err(LlmError::NoResponse)
    } else {
        Ok(result)
    }
}

/// Sync adapter for `chat_complete_http_async`. Forwards to the caller's
/// tokio runtime when one is active, otherwise the fallback runtime.
pub fn chat_complete_http(
    api_base: &str,
    messages: &[ChatMessage],
    model: &str,
    think: Option<bool>,
    extra_body_params: Option<&serde_json::Value>,
    debug: bool,
    show_prompts: bool,
) -> Result<String, LlmError> {
    // timeout_ms=0 disables the timeout to preserve the original behavior
    // of the sync `chat_complete_http` (which had no timeout).
    let fut = chat_complete_http_async(
        api_base,
        messages,
        model,
        think,
        0,
        extra_body_params,
        debug,
        show_prompts,
    );
    block_on(fut)
}

/// Inner async request — returns `Ok(content)` or `Ok("")` on empty.
/// Never returns reasoning_content as the answer.
///
/// `extra_body_params` are merged into the JSON body after the built-in
/// fields but before stream/think overrides, so they can supply model-level
/// defaults (e.g. `num_ctx`, `temperature`, `max_tokens`) that are then
/// superseded by call-specific overrides.
async fn chat_complete_http_inner_async(
    url: &str,
    messages: &[ChatMessage],
    model: &str,
    think: Option<bool>,
    timeout_ms: u64,
    extra_body_params: Option<&serde_json::Value>,
    debug: bool,
    show_prompts: bool,
) -> Result<String, LlmError> {
    let messages_json = serde_json::to_value(messages).map_err(|e| LlmError::Api(e.to_string()))?;
    let body = crate::openai::build_openai_chat_body(
        model,
        &messages_json,
        extra_body_params,
        false,
        think,
    );

    if show_prompts {
        eprintln!("=== LLM PROMPT ===");
        eprintln!("URL: {url}");
        eprintln!("Model: {model}");
        for msg in messages {
            eprintln!("--- {} ---", msg.role);
            eprintln!("{}", msg.content);
        }
        eprintln!("=== END PROMPT ===");
    } else if debug {
        eprintln!("[llm] model={model} url={url} messages={}", messages.len());
    }

    let client = shared_http_client();
    let body_str = serde_json::to_string(&body).map_err(|e| LlmError::Api(e.to_string()))?;

    let send_fut = async {
        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .body(body_str)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;
        if !response.status().is_success() {
            let class = HttpClass::from_status(response.status().as_u16());
            return Err(if class.is_retryable() {
                LlmError::RateLimited
            } else {
                LlmError::Api(format!("HTTP {}", response.status()))
            });
        }
        response
            .text()
            .await
            .map_err(|e| LlmError::Api(e.to_string()))
    };

    let start = std::time::Instant::now();
    tracing::debug!(
        target: "llm.client",
        model = %model,
        url = %url,
        timeout_ms = timeout_ms,
        "chat completion request"
    );

    let body_str = if timeout_ms == 0 {
        send_fut.await?
    } else {
        match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), send_fut).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                tracing::warn!(
                    target: "llm.client",
                    model = %model,
                    url = %url,
                    timeout_ms = timeout_ms,
                    latency_ms = start.elapsed().as_millis() as u64,
                    "chat completion timed out"
                );
                return Err(LlmError::Http(format!(
                    "request timed out after {timeout_ms}ms"
                )));
            }
        }
    };

    tracing::debug!(
        target: "llm.client",
        model = %model,
        url = %url,
        latency_ms = start.elapsed().as_millis() as u64,
        body_len = body_str.len(),
        "chat completion response"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&body_str).map_err(|e| LlmError::Api(e.to_string()))?;

    // Extract content from choices[0].message.content.
    // reasoning_content is NEVER the answer — it is the model's internal
    // think block and must never leak to callers.
    let content = parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|choices| choices.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if show_prompts {
        eprintln!("=== LLM RESPONSE ===");
        eprintln!("{content}");
        eprintln!("=== END RESPONSE ===");
    }

    Ok(content.to_string())
}

/// Strips provider: prefix from model reference strings.
/// e.g. "ollama:embeddinggemma" → "embeddinggemma"
pub fn model_name(model_ref: &str) -> &str {
    model_ref
        .split_once(':')
        .map_or(model_ref, |(_, name)| name)
}

/// Removes think-block tags from LLM output (e.g. `<think>reasoning</think>`).
/// Canonical owner is [`crate::thinking`]; kept here so existing
/// `fluent_llm::strip_think_block` / `client::strip_think_block` imports work.
pub use crate::thinking::strip_think_block;

/// Removes leading preamble lines from LLM output.
pub fn strip_preamble(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return trimmed;
    }
    let first_newline = trimmed.find('\n').unwrap_or(trimmed.len());
    let first_line = &trimmed[..first_newline];
    let first_lower = first_line.to_lowercase();

    let preambles = [
        "let's ",
        "let me ",
        "we need to ",
        "here's ",
        "here is ",
        "i'll ",
        "i will ",
        "the answer is ",
        "to answer ",
        "okay, ",
        "ok, ",
        "sure, ",
        "alright, ",
    ];

    for &preamble in &preambles {
        if first_lower.starts_with(preamble) {
            if first_newline >= trimmed.len() {
                return "";
            }
            return trimmed[first_newline + 1..].trim();
        }
    }
    trimmed
}

const LLM_PREAMBLE_PATTERNS: &[&str] = &[
    "here's a",
    "here is a",
    "i'll ",
    "to summarize",
    "okay,",
    "ok,",
    "we need ",
    "let's think",
    "let's craft",
    "let's count",
    "let me think",
    "i need to ",
];

/// Returns true if the LLM response appears malformed.
pub fn is_malformed_response(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if llm_has_dangling_end(trimmed) {
        return true;
    }
    let rtrimmed = trimmed.trim_end_matches([' ', '\t']);
    if !rtrimmed.is_empty() && rtrimmed.ends_with('?') {
        return true;
    }
    if llm_is_generic_self_ref(trimmed) {
        return true;
    }
    if llm_is_overly_generic(trimmed) {
        return true;
    }
    for &pattern in LLM_PREAMBLE_PATTERNS {
        if common_core::string::contains_ignore_case(trimmed, pattern) {
            return true;
        }
    }
    false
}

fn llm_has_dangling_end(body: &str) -> bool {
    let trimmed = body.trim_end_matches([' ', '\t', '.', '?']);
    if trimmed.is_empty() {
        return false;
    }
    let last_word = trimmed.rsplit(' ').next().unwrap_or("");
    let danglers = ["of", "in", "for", "from", "with", "to", "a", "an", "the"];
    danglers.iter().any(|&d| last_word.eq_ignore_ascii_case(d))
}

fn llm_is_generic_self_ref(body: &str) -> bool {
    let patterns = [
        "this function",
        "this method",
        "this class",
        "this struct",
        "this type",
        "this module",
    ];
    let trimmed = body.trim_end_matches([' ', '\t', '\r', '\n', '.']);
    patterns.iter().any(|&p| trimmed.eq_ignore_ascii_case(p))
}

fn llm_is_overly_generic(body: &str) -> bool {
    let generics = [
        "function",
        "method",
        "helper",
        "util",
        "utility",
        "handler",
        "callback",
        "wrapper",
        "implementation",
    ];
    let trimmed = body.trim_end_matches([' ', '\t', '\r', '\n', '.']);
    if trimmed.len() > 20 {
        return false;
    }
    if trimmed.contains(' ') {
        return false;
    }
    generics.iter().any(|&g| trimmed.eq_ignore_ascii_case(g))
}

/// Extracts content from `<comment>` tags in LLM output.
pub fn extract_comment_tag(text: &str) -> Option<&str> {
    let start = text.find("<comment>")?;
    let content_start = start + 9;
    let end = text[content_start..].find("</comment>")?;
    let content = text[content_start..content_start + end].trim();
    if content.is_empty() {
        None
    } else {
        Some(content)
    }
}

/// Returns true if text is blank or a plausible doc comment.
pub fn is_blank_or_plausible(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.len() < 3 {
        return false;
    }
    !is_malformed_response(trimmed)
}

#[cfg(test)]
#[path = "../tests/client.rs"]
mod tests;
