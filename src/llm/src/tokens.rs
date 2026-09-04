//! Token budgets — the single owner (ROADMAP_20260903_LLM M3).
//!
//! Moved verbatim from `common_core::tokens` (`estimate_tokens`,
//! `estimate_tokens_floor`, `estimate_tokens_with`,
//! `estimate_chars_for_tokens`, `AtomicTokenBudget`, `TokenBudget`
//! (+ `truncate_to_budget`), `ChunkConfig`, `chunk_document`,
//! `split_into_sentences`, `build_overlap`). Zero cross-crate dependencies
//! (pure `std` + atomics).
//!
//! M11 deleted the `common-core::tokens` byte-identical shim copies (kept
//! through M10 under `#[deprecated]`); the owner goldens in
//! `tests/tokens.rs` are the lasting contract.
//!
//! Calibration (roadmap §1, M10): the script weights below (0.25 ASCII /
//! 0.67 CJK / 1.0 emoji / 0.1 whitespace / 0 control) are a task-value
//! budget fit, not producer confidence — a low estimate is never "the model
//! is sure", and an over-budget truncation is data loss even when the
//! producer was confident. The weights move unchanged here; calibrating the
//! weights themselves is M10 and must never be silently retuned in a move.
//!
//! Token budget helpers: `estimate_tokens`, `TokenBudget`, `AtomicTokenBudget`,
//! and sentence-boundary-aware document chunking.

use std::sync::atomic::{AtomicU64, Ordering};

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

/// Estimate the character budget for a token count at a fixed ratio.
///
/// M4: the inverse of [`estimate_tokens_with`] for callers that budget in
/// characters (no tokenizer dependency — a straight character estimate).
/// A `chars_per_token` of 0 is floored to 1 (every token costs at least one
/// char); the multiply saturates. Note the zero-guard differs deliberately
/// from `estimate_tokens_with` (which maps 0 → `text.len()`): the two run in
/// opposite directions and neither may change semantics for existing callers.
pub fn estimate_chars_for_tokens(tokens: usize, chars_per_token: usize) -> usize {
    tokens.saturating_mul(chars_per_token.max(1))
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
/// use fluent_llm::tokens::AtomicTokenBudget;
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
/// use fluent_llm::tokens::{TokenBudget, estimate_tokens};
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

