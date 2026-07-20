use crate::ChatMessage;
use common_core::tokens::estimate_tokens;

pub struct ContextPacker {
    max_tokens: usize,
}

impl ContextPacker {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn estimate_tokens(text: &str) -> usize {
        estimate_tokens(text)
    }

    pub fn truncate_to_budget(&self, text: &str) -> String {
        let budget = self.max_tokens;
        let estimated = Self::estimate_tokens(text);
        if estimated <= budget {
            return text.to_string();
        }
        let ratio = budget as f64 / estimated as f64;
        let target_len = (text.len() as f64 * ratio).max(1.0) as usize;
        let mut result = text.chars().take(target_len).collect::<String>();
        result.push_str("...");
        result
    }

    pub fn pack_context(
        &self,
        system_prompt: &str,
        context: &str,
        query: &str,
    ) -> Vec<ChatMessage> {
        let truncated_context = self.truncate_to_budget(context);
        vec![
            ChatMessage {
                role: "system".into(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".into(),
                content: format!("Context:\n{truncated_context}\n\nQuery: {query}"),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(ContextPacker::estimate_tokens("hi"), 1);
        assert_eq!(ContextPacker::estimate_tokens("a"), 1);
        assert_eq!(ContextPacker::estimate_tokens(""), 0);
    }

    #[test]
    fn test_no_truncation_when_within_budget() {
        let packer = ContextPacker::new(100);
        let text = "short text";
        let result = packer.truncate_to_budget(text);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncation_when_over_budget() {
        let packer = ContextPacker::new(2);
        let text = "this is a longer text that exceeds the budget";
        let result = packer.truncate_to_budget(text);
        assert!(result.ends_with("..."));
        assert!(result.len() < text.len());
    }

    #[test]
    fn test_pack_context() {
        let packer = ContextPacker::new(100);
        let messages = packer.pack_context(
            "You are a helpful assistant.",
            "Some context here.",
            "What is this?",
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert!(messages[1].content.contains("Some context here."));
        assert!(messages[1].content.contains("What is this?"));
    }
}
