use std::sync::Arc;

use common_core::http::shared_http_client;

use fluent_concurrency::llm_queue::{ChatMessage, LlmConfig, LlmError, LlmRequestQueue};

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
                let task = fluent_concurrency::llm_queue::LlmTask {
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
/// Delegates to the canonical implementation in `common_core::string`.
pub use common_core::string::strip_think_block;

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
mod tests {
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
}
