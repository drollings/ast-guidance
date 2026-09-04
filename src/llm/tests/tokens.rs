//! ROADMAP_20260903_LLM M3.3 — token-budget goldens (moved, not copied).
//!
//! Canonical home for every token-budget golden: the estimation / budget /
//! chunking assertions moved from `src/common-core/tests/tokens.rs`, plus
//! the must-NOT-over-count control group (whitespace/control-heavy inputs
//! stay fractional/zero) and the `chars_per_token = 0` divergence lock.
//! Behavior is byte-identical to the removed `common_core::tokens` shims
//! (M11 deleted them with `parity_new_eq_old`).
//!
//! Calibration (roadmap §1, M10): these weights are a task-value budget
//! fit, not producer confidence — a low estimate is never "the model is
//! sure", and an over-budget truncation is data loss even when the
//! producer was confident. The weights move unchanged; calibrating them
//! is M10.

use fluent_llm::tokens::*;

// ── Moved from common-core/tests/tokens.rs: estimation ───────────────────

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

// ── Moved: fixed-ratio estimators + divergence lock ──────────────────────

#[test]
fn test_estimate_tokens_with() {
    assert_eq!(estimate_tokens_with("abcdefgh", 2), 4);
    assert_eq!(estimate_tokens_with("abcdefgh", 0), 8);
}

#[test]
fn test_estimate_chars_for_tokens() {
    // Exact integer equality; the zero-guard differs deliberately from
    // `estimate_tokens_with` (inverse floored to 1, not `text.len()`) —
    // locked, not unified.
    assert_eq!(estimate_chars_for_tokens(100, 4), 400);
    assert_eq!(estimate_chars_for_tokens(8192, 4), 32768);
    assert_eq!(estimate_chars_for_tokens(100, 0), 100, "zero ratio floors at 1");
    assert_eq!(estimate_chars_for_tokens(0, 4), 0);
    assert_eq!(estimate_chars_for_tokens(0, 0), 0);
    assert_eq!(estimate_chars_for_tokens(usize::MAX, 4), usize::MAX, "saturates");
}

// ── Moved: atomic budget ─────────────────────────────────────────────────

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
    let debug = format!("{b:?}");
    assert!(debug.contains("500"));
    assert!(debug.contains("100"));
    assert_eq!(b.remaining(), 400);
}

// ── Moved: TokenBudget ───────────────────────────────────────────────────

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

// ── Moved: chunking ──────────────────────────────────────────────────────

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
    let text = "わかりました！本当ですか？そうです。";
    let chunks = chunk_document(text, &ChunkConfig::new(10, 2));
    assert!(!chunks.is_empty());
}

#[test]
fn chunk_newline_boundaries() {
    let text = "line one\nline two\nline three\nline four\nline five\nline six";
    let chunks = chunk_document(text, &ChunkConfig::new(5, 1));
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
    let text = &"x".repeat(500); // ~125 tokens, well over budget
    let chunks = chunk_document(text, &ChunkConfig::new(20, 5));
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

// ── Controls: must NOT over-count (precision guard) ──────────────────────

#[test]
fn control_whitespace_and_control_stay_fractional_or_zero() {
    // Whitespace-heavy input must not inflate: 7 ws chars → 0.7 → 1.
    assert_eq!(estimate_tokens("   \n\t  "), 1);
    // A long whitespace run scales fractionally, never 1:1.
    let ws = " ".repeat(100);
    assert!(estimate_tokens(&ws) <= 15, "got {}", estimate_tokens(&ws));
    // Control characters contribute nothing.
    assert_eq!(estimate_tokens("\x00\x01\x02\x7f"), 0);
    // `chars_per_token = 0` keeps its documented divergent semantics:
    // forward counts every byte, inverse floors the ratio at 1.
    assert_eq!(estimate_tokens_with("abcdefgh", 0), 8);
    assert_eq!(estimate_chars_for_tokens(100, 0), 100);
}
