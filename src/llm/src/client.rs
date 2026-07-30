use std::sync::{Arc, OnceLock};

use common_core::http::shared_http_client;

use fluent_concurrency::llm_queue::{ChatMessage, LlmConfig, LlmError, LlmRequestQueue};

use crate::http_class::HttpClass;

/// Trait for chat backends — sends messages and returns a response string.
///
/// Implemented by `LlmClient` (production) and test stubs.
pub trait ChatBackend: Send + Sync {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError>;
}

/// Fallback runtime used only when the sync adapter is called from a context
/// that has no tokio runtime (e.g. a plain `fn main()`). Production callers
/// inside an async runtime use `Handle::current().block_on` instead.
fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to build fallback tokio runtime for LlmClient")
    })
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
        match &self.queue {
            Some(q) => {
                let task = fluent_concurrency::llm_queue::LlmTask {
                    messages: messages.to_vec(),
                    config: self.config.clone(),
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
                    self.config.extra_body_params.as_ref(),
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
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let client = self.clone();
                let messages = messages.to_vec();
                tokio::task::block_in_place(move || {
                    handle.block_on(client.chat_complete_async(&messages))
                })
            }
            Err(_) => fallback_runtime().block_on(self.chat_complete_async(messages)),
        }
    }
}

impl ChatBackend for LlmClient {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.chat_complete(messages)
    }
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
    let trimmed = api_base.trim_end_matches('/');
    let url = if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    };
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
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(fut),
        Err(_) => fallback_runtime().block_on(fut),
    }
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
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false},
    });

    // Merge model-level params (can override defaults above, e.g. when
    // `extra_body_params` contains `chat_template_kwags`).
    if let Some(params) = extra_body_params {
        if let Some(obj) = params.as_object() {
            for (k, v) in obj {
                if k != "model" && k != "messages" && k != "stream" {
                    body[k] = v.clone();
                }
            }
        }
    }
    // think flag from LlmConfig overrides everything.
    if think == Some(true) {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    }

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

    let body_str = if timeout_ms == 0 {
        send_fut.await?
    } else {
        match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), send_fut).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(LlmError::Http(format!(
                    "request timed out after {timeout_ms}ms"
                )));
            }
        }
    };

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
}
