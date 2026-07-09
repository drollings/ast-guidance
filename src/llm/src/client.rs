use std::sync::{Arc, LazyLock};

use bon::Builder;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::llm_queue::{LlmQueueConfig, LlmRequestQueue};

/// Trait for chat backends — sends messages and returns a response string.
///
/// Implemented by `LlmClient` (production) and test stubs.
pub trait ChatBackend: Send + Sync {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError>;
}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("API error: {0}")]
    Api(String),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("no response from model")]
    NoResponse,
    #[error("rate limited")]
    RateLimited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(start_fn = new)]
pub struct LlmConfig {
    pub api_url: String,
    pub model: String,
    pub think: Option<bool>,
    #[builder(default = 2000)]
    pub timeout_ms: u64,
    #[builder(default)]
    pub debug: bool,
    #[builder(default)]
    pub show_prompts: bool,
}

struct DefaultQueue {
    #[allow(dead_code)]
    runtime: tokio::runtime::Runtime,
    queue: Arc<LlmRequestQueue>,
}

impl DefaultQueue {
    fn get() -> &'static Self {
        static INSTANCE: LazyLock<DefaultQueue> = LazyLock::new(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let queue = runtime.block_on(async {
                Arc::new(LlmRequestQueue::new(
                    Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime),
                    &LlmQueueConfig::default(),
                ))
            });
            DefaultQueue { runtime, queue }
        });
        &INSTANCE
    }
}

static BLOCKING_CLIENT: std::sync::LazyLock<reqwest::blocking::Client> =
    std::sync::LazyLock::new(reqwest::blocking::Client::new);

pub struct LlmClient {
    pub api_base: String,
    pub model: String,
    pub config: LlmConfig,
    pub queue: Option<Arc<LlmRequestQueue>>,
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

    pub fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let dq = DefaultQueue::get();
        let queue = self.queue.clone().unwrap_or_else(|| dq.queue.clone());
        dq.runtime
            .block_on(queue.submit_async(messages.to_vec(), self.config.clone()))
    }
}

impl ChatBackend for LlmClient {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.chat_complete(messages)
    }
}

pub fn chat_complete_http(
    api_base: &str,
    messages: &[ChatMessage],
    model: &str,
    think: Option<bool>,
    debug: bool,
    show_prompts: bool,
) -> Result<String, LlmError> {
    let trimmed = api_base.trim_end_matches('/');
    let url = if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    };
    let result = chat_complete_http_inner(&url, messages, model, think, debug, show_prompts)?;
    if result.is_empty() {
        Err(LlmError::NoResponse)
    } else {
        Ok(result)
    }
}

/// Inner request — returns `Ok(content)` or `Ok("")` on empty.
/// Never returns reasoning_content as the answer.
fn chat_complete_http_inner(
    url: &str,
    messages: &[ChatMessage],
    model: &str,
    think: Option<bool>,
    debug: bool,
    show_prompts: bool,
) -> Result<String, LlmError> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": 1024u32,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false},
    });
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

    let response = BLOCKING_CLIENT
        .post(url)
        .body(serde_json::to_string(&body).map_err(|e| LlmError::Api(e.to_string()))?)
        .send()
        .map_err(|e| LlmError::Http(e.to_string()))?;

    let body_str = response.text().map_err(|e| LlmError::Api(e.to_string()))?;

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
pub fn strip_think_block(text: &str) -> String {
    let result = if let Some(start) = text.find("<think>") {
        if let Some(end) = text[start + 7..].find("</think>") {
            let after = start + 7 + end + 8;
            if after >= text.len() {
                String::new()
            } else {
                text[after..].trim_start().to_string()
            }
        } else {
            text[..start].trim().to_string()
        }
    } else if let Some(start) = text.find("[THINK]") {
        if let Some(end) = text[start + 7..].find("[/THINK]") {
            let after = start + 7 + end + 8;
            if after >= text.len() {
                String::new()
            } else {
                text[after..].trim_start().to_string()
            }
        } else {
            text[..start].trim().to_string()
        }
    } else {
        text.to_string()
    };
    result
}

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
