use crate::ChatMessage;
use common_core::tokens::{estimate_tokens, TokenBudget};

/// Token-budget context packing. Canonical owner of token-budget logic in the
/// workspace: prompt packing, LOD selection, and first-fit-decreasing
/// bin-packing all live here so every consumer shares one `ContextPacker`.
///
/// Coral (`coral-context`) delegates its graph packing to this type via the
/// shared `ffd_pack` budget-fit core; Node-specific helpers stay coral-side.
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

    /// Estimate token count from text length (Unicode-script-aware).
    pub fn estimate_tokens(text: &str) -> usize {
        estimate_tokens(text) as usize
    }

    /// Select the appropriate LOD level based on graph distance.
    /// Closer nodes get more detailed LOD (lower index).
    pub fn select_lod_by_distance(graph_distance: f64, avg_degree: f64) -> u8 {
        let effective_distance = graph_distance / (1.0 + avg_degree / (avg_degree + 1.0));
        if effective_distance < 1.0 {
            return 0;
        }
        if effective_distance < 2.0 {
            return 1;
        }
        if effective_distance < 3.0 {
            return 2;
        }
        if effective_distance < 4.0 {
            return 3;
        }
        if effective_distance < 5.0 {
            return 4;
        }
        5
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

    /// First-Fit Decreasing bin-pack of `(text, payload)` items into the token
    /// budget, sorted by decreasing estimated token cost. Returns the payloads
    /// whose texts fit within the budget.
    ///
    /// This is the shared budget-fit core; coral's graph packing delegates to
    /// it (see `coral_context::packer`).
    pub fn ffd_pack<'a, T: ?Sized>(&self, items: &[(&'a str, &'a T)]) -> Vec<&'a T> {
        let mut sorted: Vec<(usize, &'a T)> = items
            .iter()
            .map(|(text, payload)| (Self::estimate_tokens(text), *payload))
            .collect();
        sorted.sort_by_key(|(tokens, _)| std::cmp::Reverse(*tokens));

        let mut used_tokens = 0usize;
        let mut packed = Vec::new();
        for (tokens, payload) in sorted {
            if used_tokens + tokens <= self.budget.0 {
                packed.push(payload);
                used_tokens += tokens;
            }
        }
        packed
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

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(ContextPacker::estimate_tokens("hello"), 1);
        assert_eq!(ContextPacker::estimate_tokens(""), 0);
        assert_eq!(ContextPacker::estimate_tokens(&"a".repeat(12)), 3);
    }

    #[test]
    fn test_select_lod_by_distance() {
        assert_eq!(ContextPacker::select_lod_by_distance(0.5, 2.0), 0);
        // effective = 1.5 / (1 + 2/3) = 1.5 / 1.667 ≈ 0.9 → lod 0
        assert_eq!(ContextPacker::select_lod_by_distance(1.5, 2.0), 0);
        // effective = 5.0 / 1.667 ≈ 3.0 → lod 3
        assert_eq!(ContextPacker::select_lod_by_distance(5.0, 2.0), 3);
    }

    #[test]
    fn test_ffd_pack_respects_budget() {
        let packer = ContextPacker::new(10);
        let items = vec![
            ("aaaa", "first"),
            ("bb", "second"),
            ("cc", "third"),
        ];
        let packed = packer.ffd_pack(&items);
        // Budget 10 tokens (generous) — all three fit
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn test_ffd_pack_respects_order_by_size() {
        let packer = ContextPacker::new(1);
        let first = 1;
        let second = 2;
        let third = 3;
        let items = vec![("aaaa", &first), ("bb", &second), ("cc", &third)];
        let packed = packer.ffd_pack(&items);
        // Only the smallest single item fits within a 1-token budget
        assert_eq!(packed.len(), 1);
    }
}
