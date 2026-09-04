use crate::ChatMessage;
use crate::tokens::{estimate_tokens, TokenBudget};

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
#[path = "../tests/context_packer.rs"]
mod tests;
