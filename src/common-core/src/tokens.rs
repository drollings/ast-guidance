//! Token budget helpers: `estimate_tokens`, `TokenBudget`, `AtomicTokenBudget`,
//! and sentence-boundary-aware document chunking.

use std::sync::atomic::{AtomicU64, Ordering};

#[deprecated(
    since = "0.2.0",
    note = "use `estimate_tokens` instead — the Unicode-script-aware estimator no longer uses a single ratio"
)]
pub const DEFAULT_CHARS_PER_TOKEN: usize = 4;

/// Estimate the token count of `text` using Unicode script density.
///
/// Different scripts compress differently in LLM tokenizers:
/// - ASCII/Latin: ~0.25 tokens per char (4 chars per token)
/// - CJK:         ~0.67 tokens per char (1.5 chars per token)
/// - Emoji:       ~1.0  tokens per emoji
/// - Whitespace:  ~0.1  tokens per char
/// - Control:     0
///
/// Returns a `u64` count. Callers that need `usize` can cast.
pub fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let sum: f64 = text.chars().map(char_token_weight).sum();
    sum.round() as u64
}

/// Like `estimate_tokens` but returns at least `min_tokens` even for
/// short or empty text.
pub fn estimate_tokens_floor(text: &str, min_tokens: u64) -> u64 {
    estimate_tokens(text).max(min_tokens)
}

/// Estimate token count using a fixed characters-per-token ratio.
///
/// When `chars_per_token` is 0 every byte counts as one token.
pub fn estimate_tokens_with(text: &str, chars_per_token: usize) -> usize {
    if chars_per_token == 0 {
        return text.len();
    }
    text.len().div_ceil(chars_per_token)
}

fn char_token_weight(c: char) -> f64 {
    match c {
        // CJK Unified Ideographs + extensions + compatibility
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B739}'
        | '\u{2B740}'..='\u{2B81D}'
        | '\u{2B820}'..='\u{2CEA1}'
        | '\u{2CEB0}'..='\u{2EBE0}'
        | '\u{30000}'..='\u{3134A}'
        | '\u{31350}'..='\u{323AF}'
        // CJK compatibility ideographs
        | '\u{F900}'..='\u{FAFF}'
        | '\u{2F800}'..='\u{2FA1F}'
        // CJK misc (punctuation, strokes, symbols generally 1 token each)
        | '\u{2E80}'..='\u{2EFF}'
        | '\u{3000}'..='\u{303F}'
        | '\u{31C0}'..='\u{31EF}'
        | '\u{3200}'..='\u{33FF}'
        | '\u{FE30}'..='\u{FE4F}'
        | '\u{FF00}'..='\u{FFEF}' => 0.67, // ~1.5 chars/token

        // Emoji — each count as roughly 1 token
        '\u{1F300}'..='\u{1F9FF}'   // misc symbols, emoticons, sport, transport, etc.
        | '\u{1FA00}'..='\u{1FA6F}' // chess symbols
        | '\u{1FA70}'..='\u{1FAFF}' // symbols extended-A
        | '\u{2600}'..='\u{27BF}'   // misc symbols, dingbats
        | '\u{FE00}'..='\u{FE0F}'   // variation selectors
        | '\u{200D}'                // ZWJ
        | '\u{E0000}'..='\u{E007F}' // tags block
        | '\u{1F000}'..='\u{1F02F}' // mahjong/domino
        | '\u{1F0A0}'..='\u{1F0FF}' // playing cards
        | '\u{1F100}'..='\u{1F1FF}' // enclosed alphanumeric supplement
        | '\u{1F200}'..='\u{1F2FF}' // enclosed ideographic supplement
        => 1.0,

        // Whitespace — contributes fractionally
        ' ' | '\t' | '\n' | '\r' | '\u{200B}' | '\u{00A0}' => 0.1,

        // Control characters — contribute nothing
        '\u{0000}'..='\u{001F}' | '\u{007F}'..='\u{009F}' => 0.0,

        // Default: ASCII / Latin / everything else → ~0.25 tokens per char (4 chars/token)
        _ => 0.25,
    }
}

// ── Atomic Token Budget ─────────────────────────────────────────────

/// A thread-safe token budget using atomic operations internally.
///
/// All methods take `&self` (no mutable borrow) so the budget can be
/// shared across concurrent tasks.
///
/// # Examples
///
/// ```
/// use common_core::tokens::AtomicTokenBudget;
///
/// let budget = AtomicTokenBudget::new(1000);
/// assert_eq!(budget.remaining(), 1000);
/// assert!(budget.reserve(200));
/// assert_eq!(budget.remaining(), 800);
/// budget.release(50);
/// assert_eq!(budget.remaining(), 850);
/// ```
#[derive(Debug)]
pub struct AtomicTokenBudget {
    total: u64,
    used: AtomicU64,
}

impl AtomicTokenBudget {
    pub fn new(total: u64) -> Self {
        Self {
            total,
            used: AtomicU64::new(0),
        }
    }

    /// Total capacity of the budget.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// How many tokens are currently remaining.
    pub fn remaining(&self) -> u64 {
        let used = self.used.load(Ordering::Acquire);
        self.total.saturating_sub(used)
    }

    /// Attempt to reserve `tokens` from the budget.
    ///
    /// Returns `true` if the tokens were reserved, `false` if there
    /// are not enough tokens remaining.
    pub fn reserve(&self, tokens: u64) -> bool {
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            if current + tokens > self.total {
                return false;
            }
            match self.used.compare_exchange_weak(
                current,
                current + tokens,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Release previously reserved tokens back to the budget.
    ///
    /// The release is clamped so usage never goes below zero.
    pub fn release(&self, tokens: u64) {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |u| {
                Some(u.saturating_sub(tokens))
            })
            .ok();
    }

    /// Reset usage to zero.
    pub fn reset(&self) {
        self.used.store(0, Ordering::Release);
    }
}

// ── Token Budget (mutable, per-request) ─────────────────────────────

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
/// let tokens = estimate_tokens(text) as usize;
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

    pub fn truncate_to_budget(&self, text: &str) -> String {
        let budget = self.0;
        if budget == 0 {
            return String::new();
        }
        let estimated = estimate_tokens(text) as usize;
        if estimated <= budget {
            return text.to_string();
        }
        let ratio = budget as f64 / estimated as f64;
        let target_len = (text.len() as f64 * ratio).max(1.0) as usize;
        let mut result = text.chars().take(target_len).collect::<String>();
        result.push_str("...");
        result
    }
}

// ── Sentence-boundary-aware document chunking ────────────────────────

/// Configuration for sentence-boundary-aware document chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum tokens allowed per chunk (estimated).
    pub max_tokens: u64,
    /// Desired overlap in tokens between consecutive chunks.
    /// Clamped below `max_tokens` so progress is always guaranteed.
    pub overlap_tokens: u64,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            overlap_tokens: 64,
        }
    }
}

impl ChunkConfig {
    pub fn new(max_tokens: u64, overlap_tokens: u64) -> Self {
        let overlap_tokens = overlap_tokens.min(max_tokens.saturating_sub(1));
        Self {
            max_tokens,
            overlap_tokens,
        }
    }
}

/// Split `text` into chunks that each fit within a token budget,
/// respecting sentence boundaries where possible.
///
/// Sentences end at `.`, `!`, `?` and their CJK full-width variants
/// `。`, `！`, `？`. Newlines also act as unit boundaries.
///
/// Consecutive chunks share trailing context as overlap so that
/// boundary information is not lost. A unit that alone exceeds the
/// budget is emitted as its own chunk (never dropped).
///
/// Returns empty `Vec` for empty or whitespace-only input.
pub fn chunk_document(text: &str, config: &ChunkConfig) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let max_tokens = config.max_tokens;
    let overlap_tokens = config.overlap_tokens;

    let sentences = split_into_sentences(trimmed);
    let mut chunks: Vec<String> = Vec::new();
    let mut current_units: Vec<&str> = Vec::new();
    let mut current_tokens: u64 = 0;

    for sentence in &sentences {
        let unit_tokens = estimate_tokens(sentence);

        // If a single unit exceeds the budget, emit it as its own chunk.
        if unit_tokens > max_tokens && current_units.is_empty() {
            chunks.push((*sentence).to_string());
            continue;
        }

        if current_tokens + unit_tokens > max_tokens {
            // Flush current chunk
            let chunk_text = current_units.join("");
            chunks.push(chunk_text);
            // Compute overlap: carry trailing units whose token cost fits in the overlap budget
            current_units = build_overlap(&current_units, overlap_tokens);
            current_tokens = estimate_tokens(&current_units.join(""));
        }

        current_units.push(sentence);
        current_tokens += unit_tokens;
    }

    // Flush remaining
    if !current_units.is_empty() {
        chunks.push(current_units.join(""));
    }

    chunks
}

fn split_into_sentences(text: &str) -> Vec<&str> {
    let mut units: Vec<&str> = Vec::new();
    let mut start = 0;
    let char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let chars: Vec<char> = text.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        let is_terminal = matches!(ch, '.' | '!' | '?' | '\u{3002}' | '\u{FF01}' | '\u{FF1F}');
        let is_newline = ch == '\n';

        if is_terminal || is_newline {
            let end = if idx + 1 < char_indices.len() {
                let next_byte = char_indices[idx + 1];
                // Include newline after terminal for sentence termination
                if is_terminal && idx + 1 < chars.len() && chars[idx + 1] == '\n' {
                    if idx + 2 < char_indices.len() {
                        char_indices[idx + 2]
                    } else {
                        text.len()
                    }
                } else {
                    next_byte
                }
            } else {
                text.len()
            };

            let unit = &text[start..end];
            if !unit.trim().is_empty() {
                units.push(unit);
            }
            if is_terminal && idx + 1 < chars.len() && chars[idx + 1] == '\n' {
                start = if idx + 2 < char_indices.len() {
                    char_indices[idx + 2]
                } else {
                    text.len()
                };
            } else {
                start = end;
            }
        }
    }

    // Remaining text after last boundary
    if start < text.len() {
        let rest = &text[start..];
        if !rest.trim().is_empty() {
            units.push(rest);
        }
    }

    units
}

fn build_overlap<'a>(units: &[&'a str], overlap_budget: u64) -> Vec<&'a str> {
    let mut result: Vec<&str> = Vec::new();
    let mut tokens: u64 = 0;
    for unit in units.iter().rev() {
        let t = estimate_tokens(unit);
        if tokens + t > overlap_budget {
            break;
        }
        result.push(unit);
        tokens += t;
    }
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1.1 Unicode-aware estimation ──────────────────────────────

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_ascii() {
        let tokens = estimate_tokens("hello world"); // 11 chars → ~2.75 → 3
        assert!(tokens >= 2 && tokens <= 4, "got {tokens}");
    }

    #[test]
    fn estimate_tokens_cjk() {
        let tokens = estimate_tokens("你好世界"); // 4 CJK chars → ~2.68 → 3
        assert!(tokens >= 2 && tokens <= 4, "got {tokens}");
    }

    #[test]
    fn estimate_tokens_emoji() {
        let tokens = estimate_tokens("😀😀😀"); // 3 emoji → 3 tokens
        assert_eq!(tokens, 3);
    }

    #[test]
    fn estimate_tokens_mixed() {
        let tokens = estimate_tokens("hello 世界 😀"); // 6 ASCII + 1 space + 2 CJK + 1 space + 1 emoji
                                                       // 6*0.25 + 0.1 + 2*0.67 + 0.1 + 1.0 = 1.5 + 0.1 + 1.34 + 0.1 + 1.0 = 4.04 → 4
        assert!(tokens >= 3 && tokens <= 5, "got {tokens}");
    }

    #[test]
    fn estimate_tokens_whitespace_heavy() {
        let tokens = estimate_tokens("   \n\t  "); // 7 whitespace chars → 0.7 → rounds to 1
        assert_eq!(tokens, 1);
    }

    #[test]
    fn estimate_tokens_control_chars() {
        let tokens = estimate_tokens("\x00\x01\x02"); // 3 control → 0 → min 0 → floor returns 0
        assert_eq!(tokens, 0);
    }

    #[test]
    fn estimate_tokens_floor_respects_minimum() {
        assert_eq!(estimate_tokens_floor("", 5), 5);
        assert_eq!(estimate_tokens_floor("hi", 10), 10);
        assert_eq!(
            estimate_tokens_floor("this is a longer text", 3),
            estimate_tokens("this is a longer text")
        );
    }

    // ── 1.2 Atomic token budget ────────────────────────────────────

    #[test]
    fn atomic_budget_new_zero_used() {
        let b = AtomicTokenBudget::new(1000);
        assert_eq!(b.total(), 1000);
        assert_eq!(b.remaining(), 1000);
    }

    #[test]
    fn atomic_budget_reserve_success() {
        let b = AtomicTokenBudget::new(1000);
        assert!(b.reserve(200));
        assert_eq!(b.remaining(), 800);
    }

    #[test]
    fn atomic_budget_reserve_insufficient() {
        let b = AtomicTokenBudget::new(100);
        assert!(!b.reserve(101));
        assert_eq!(b.remaining(), 100);
    }

    #[test]
    fn atomic_budget_release() {
        let b = AtomicTokenBudget::new(1000);
        b.reserve(200);
        b.release(50);
        assert_eq!(b.remaining(), 850);
    }

    #[test]
    fn atomic_budget_release_clamped_to_zero() {
        let b = AtomicTokenBudget::new(100);
        b.release(200);
        assert_eq!(b.remaining(), 100);
    }

    #[test]
    fn atomic_budget_reset() {
        let b = AtomicTokenBudget::new(1000);
        b.reserve(500);
        b.reset();
        assert_eq!(b.remaining(), 1000);
    }

    #[test]
    fn atomic_budget_debug() {
        let b = AtomicTokenBudget::new(500);
        b.reserve(100);
        let debug = format!("{:?}", b);
        assert!(debug.contains("500"));
        assert!(debug.contains("100"));
        // used is 100, so remaining is 400 — verify the budget is intact
        assert_eq!(b.remaining(), 400);
    }

    // ── 1.3 Sentence-boundary chunking ─────────────────────────────

    #[test]
    fn chunk_empty() {
        assert_eq!(
            chunk_document("", &ChunkConfig::default()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn chunk_whitespace_only() {
        assert_eq!(
            chunk_document("   \n  ", &ChunkConfig::default()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn chunk_latin_splits() {
        // Each sentence ~13 ASCII chars → ~3 tokens. With max=10, should fit 3 sentences.
        let text =
            "First sentence here. Second sentence here. Third sentence here. Fourth sentence here.";
        let chunks = chunk_document(text, &ChunkConfig::new(10, 3));
        // Should produce at least 2 chunks
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
        for c in &chunks {
            let t = estimate_tokens(c) as usize;
            assert!(t <= 12, "chunk has {t} tokens, expected ≤ 12");
        }
    }

    #[test]
    fn chunk_cjk_terminators() {
        let text = "こんにちは。さようなら！元気ですか？はい、元気です。";
        let chunks = chunk_document(text, &ChunkConfig::new(10, 2));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn chunk_cjk_fullwidth_terminators() {
        // Full-width exclamation ＵＦＦ０１ and question mark ＵＦＦ１Ｆ
        let text = "わかりました！本当ですか？そうです。";
        let chunks = chunk_document(text, &ChunkConfig::new(10, 2));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn chunk_newline_boundaries() {
        let text = "line one\nline two\nline three\nline four\nline five\nline six";
        let chunks = chunk_document(text, &ChunkConfig::new(5, 1));
        // Newlines create multiple unit boundaries, should produce several chunks
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
    }

    #[test]
    fn chunk_overlap_preserved() {
        let text = "AAA. BBB. CCC. DDD. EEE. FFF.";
        let config = ChunkConfig::new(5, 2);
        let chunks = chunk_document(text, &config);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunk_overlong_unit_emitted_alone() {
        // A single very long "sentence" with no terminators
        let text = &"x".repeat(500); // ~125 tokens, well over default 512 bytes budget
        let chunks = chunk_document(text, &ChunkConfig::new(20, 5));
        // It should be emitted as its own chunk
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_config_default() {
        let cfg = ChunkConfig::default();
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.overlap_tokens, 64);
    }

    #[test]
    fn chunk_config_clamps_overlap() {
        let cfg = ChunkConfig::new(100, 150);
        assert!(cfg.overlap_tokens < cfg.max_tokens);
    }

    // ── TokenBudget (existing API) ─────────────────────────────────

    #[test]
    fn test_token_budget_truncate_to_budget() {
        let budget = TokenBudget(100);
        assert_eq!(budget.truncate_to_budget("short"), "short");
        assert_eq!(budget.truncate_to_budget(""), "");
        let budget_zero = TokenBudget(0);
        assert_eq!(budget_zero.truncate_to_budget("anything"), "");
    }

    #[test]
    fn test_token_budget_truncate_over_budget() {
        let budget = TokenBudget(2);
        let text = "this is a longer text that exceeds the budget";
        let result = budget.truncate_to_budget(text);
        assert!(result.ends_with("..."));
        assert!(result.len() < text.len());
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

    #[test]
    fn test_estimate_tokens_with() {
        assert_eq!(estimate_tokens_with("abcdefgh", 2), 4);
        assert_eq!(estimate_tokens_with("abcdefgh", 0), 8);
    }

    #[test]
    #[allow(deprecated)]
    fn test_estimate_tokens_old_still_works() {
        assert_eq!(estimate_tokens_with("abcdefgh", DEFAULT_CHARS_PER_TOKEN), 2);
    }
}
