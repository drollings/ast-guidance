use bon::Builder;

use crate::client::{strip_think_block, LlmClient};
use crate::LlmConfig;

/// Trait for task decomposition — splits a query into subtasks.
///
/// Implemented by `LocalDecomposer` (LLM-backed) and test stubs.
pub trait Decomposer: Send + Sync {
    fn decompose(&self, task: &str) -> Vec<String>;
}

#[derive(Debug, Clone, Builder)]
pub struct DecomposerConfig {
    pub llm: LlmConfig,
    #[builder(default = 5)]
    pub max_subtasks: usize,
    #[builder(default = 2)]
    pub max_depth: u8,
}

pub struct LocalDecomposer {
    pub config: DecomposerConfig,
}

impl Decomposer for LocalDecomposer {
    fn decompose(&self, task: &str) -> Vec<String> {
        self.decompose(task)
    }
}

const SYSTEM_PROMPT: &str = r#"You are a task planner. Given a user query, decompose it into at most 5
concrete, ordered sub-tasks. Reply with ONLY a JSON array of strings, no
preamble, no explanation. Example:
["Find relevant documents","Filter by date","Summarize results"]"#;

impl LocalDecomposer {
    pub fn new(config: DecomposerConfig) -> Self {
        Self { config }
    }

    pub fn decompose(&self, task: &str) -> Vec<String> {
        let client = LlmClient::with_config(self.config.llm.clone());
        let messages = vec![
            crate::ChatMessage {
                role: "system".into(),
                content: SYSTEM_PROMPT.to_string(),
            },
            crate::ChatMessage {
                role: "user".into(),
                content: task.to_string(),
            },
        ];

        let Ok(raw) = client.chat_complete(&messages) else {
            return vec![task.to_string()];
        };

        let stripped = strip_think_block(&raw);
        if is_malformed_json_array(&stripped) {
            return vec![task.to_string()];
        }

        match parse_json_array(&stripped, self.config.max_subtasks) {
            Ok(tasks) => tasks,
            Err(_) => vec![task.to_string()],
        }
    }
}

fn is_malformed_json_array(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    if !t.starts_with('[') {
        return true;
    }
    if !t.ends_with(']') {
        return true;
    }
    false
}

fn parse_json_array(text: &str, limit: usize) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("json parse: {e}"))?;
    let serde_json::Value::Array(ref arr) = parsed else {
        return Err("not an array".into());
    };
    if arr.is_empty() {
        return Err("empty array".into());
    }
    let count = arr.len().min(limit);
    let mut result = Vec::with_capacity(count);
    for item in arr.iter().take(count) {
        match item {
            serde_json::Value::String(s) => result.push(s.clone()),
            _ => return Err("not a string array".into()),
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decomposer() -> LocalDecomposer {
        let config = DecomposerConfig::builder()
            .llm(
                LlmConfig::new()
                    .api_url("http://localhost:11434/v1".into())
                    .model("llama3".into())
                    .build(),
            )
            .max_subtasks(5)
            .max_depth(2)
            .build();
        LocalDecomposer::new(config)
    }

    #[test]
    fn test_decomposer_creation() {
        let d = make_decomposer();
        assert_eq!(d.config.max_subtasks, 5);
        assert_eq!(d.config.max_depth, 2);
    }

    #[test]
    fn test_decomposer_config_builder() {
        let config = DecomposerConfig::builder()
            .llm(
                LlmConfig::new()
                    .api_url("http://localhost:11434/v1".into())
                    .model("llama3".into())
                    .build(),
            )
            .build();
        assert_eq!(config.max_subtasks, 5);
        assert_eq!(config.max_depth, 2);
    }

    #[test]
    fn test_is_malformed_json_array() {
        assert!(is_malformed_json_array(""));
        assert!(is_malformed_json_array("hello"));
        assert!(!is_malformed_json_array("[\"a\"]"));
        assert!(!is_malformed_json_array("[ \"a\", \"b\" ]"));
    }

    #[test]
    fn test_parse_json_array_valid() {
        let result = parse_json_array("[\"a\", \"b\", \"c\"]", 5).unwrap();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_json_array_with_limit() {
        let result = parse_json_array("[\"a\", \"b\", \"c\"]", 2).unwrap();
        assert_eq!(result, vec!["a", "b"]);
    }

    #[test]
    fn test_parse_json_array_invalid() {
        assert!(parse_json_array("not json", 5).is_err());
        assert!(parse_json_array("{}", 5).is_err());
    }

    #[test]
    fn test_parse_json_array_empty() {
        assert!(parse_json_array("[]", 5).is_err());
    }

    #[test]
    fn test_decomposer_fallback_on_malformed_response() {
        // With no LLM server running, decompose should fallback gracefully
        let d = make_decomposer();
        let tasks = d.decompose("test task");
        assert_eq!(tasks, vec!["test task"]);
    }
}
