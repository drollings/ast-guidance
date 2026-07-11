//! Token budget helpers: `estimate_tokens` and `TokenBudget`.

pub const DEFAULT_CHARS_PER_TOKEN: usize = 4;

pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(DEFAULT_CHARS_PER_TOKEN)
}

pub fn estimate_tokens_with(text: &str, chars_per_token: usize) -> usize {
    if chars_per_token == 0 {
        return text.len();
    }
    text.len().div_ceil(chars_per_token)
}

/// A remaining token budget that can be checked and consumed.
///
/// Useful for truncating prompts or responses to fit within LLM context windows.
///
/// # Examples
///
/// ```
/// use common_core::tokens::{TokenBudget, estimate_tokens};
///
/// let mut budget = TokenBudget(1000);
/// let text = "This is a test prompt.";
/// let tokens = estimate_tokens(text);
///
/// assert!(budget.fits(tokens));
/// assert!(budget.consume(tokens));
/// assert_eq!(budget.remaining(0), 1000 - tokens);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenBudget(pub usize);

impl TokenBudget {
    pub fn fits(&self, tokens: usize) -> bool {
        tokens <= self.0
    }

    pub fn remaining(&self, used: usize) -> usize {
        self.0.saturating_sub(used)
    }

    pub fn consume(&mut self, tokens: usize) -> bool {
        if tokens <= self.0 {
            self.0 -= tokens;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("hi"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn test_estimate_tokens_with() {
        assert_eq!(estimate_tokens_with("abcdefgh", 2), 4);
        assert_eq!(estimate_tokens_with("abcdefgh", 0), 8);
    }

    #[test]
    fn test_token_budget() {
        let mut budget = TokenBudget(10);
        assert!(budget.fits(10));
        assert!(!budget.fits(11));
        assert_eq!(budget.remaining(3), 7);
        assert!(budget.consume(4));
        assert_eq!(budget.0, 6);
        assert!(!budget.consume(10));
        assert_eq!(budget.0, 6);
    }
}
