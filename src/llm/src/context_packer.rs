use crate::ChatMessage;
use common_core::tokens::TokenBudget;

pub struct ContextPacker {
    budget: TokenBudget,
}

impl ContextPacker {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            budget: TokenBudget(max_tokens),
        }
    }

    pub fn max_tokens(&self) -> usize {
        self.budget.0
    }

    pub fn pack_context(
        &self,
        system_prompt: &str,
        context: &str,
        query: &str,
    ) -> Vec<ChatMessage> {
        let truncated_context = self.budget.truncate_to_budget(context);
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
    fn test_no_truncation_when_within_budget() {
        let packer = ContextPacker::new(100);
        let text = "short text";
        let result = packer.budget.truncate_to_budget(text);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncation_when_over_budget() {
        let packer = ContextPacker::new(2);
        let text = "this is a longer text that exceeds the budget";
        let result = packer.budget.truncate_to_budget(text);
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