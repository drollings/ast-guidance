//! Irreversible write-path scrubber for the Content-Node ledger (M1).
//!
//! The durable ledger can never cache text matching the builtin filter engine:
//! every write-path payload is scrubbed through
//! [`scrub_for_ledger`] before it reaches `NodeStore`. The guard lives here
//! (write policy), NOT in `NodeStore` — the store stays a pure, policy-free
//! shared store (D1).
//!
//! The scrub is **irreversible by design**: `Redact`/`Anonymize` both collapse
//! to `[REDACTED:<pattern>]` and no codeword map is retained. This is the
//! `transform` hook's implementation from `crate::views` (M2) — one
//! implementation, two callers.
//!
//! The engine is the same `DeterministicFilterEngine` built by
//! `crate::stages::deterministic::builtin_filter_engine`, evaluated with the
//! `ContentNodeWrite` scope active (`FilterContext::ledger_write`). It is
//! module-scoped and shared via `LazyLock` — never reconstructed per call.

use std::sync::LazyLock;

use crate::config::FilterAction;
use crate::filters::{DeterministicFilterEngine, FilterContext, FilterDecision};
use crate::stages::deterministic::builtin_filter_engine;

/// The shared builtin engine. Evaluated with the `ContentNodeWrite` scope
/// active so `[Any]` filters apply while `FrontierBound`-only filters do not.
static BUILTIN_ENGINE: LazyLock<DeterministicFilterEngine> = LazyLock::new(builtin_filter_engine);

/// Result of scrubbing one write-path payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubbedText {
    /// The scrubbed text, safe to persist.
    pub text: String,
    /// Whether any filter decision fired.
    pub flagged: bool,
    /// The matching pattern name, when flagged.
    pub pattern: Option<String>,
}

/// Evaluate + apply the builtin filter engine with the `ContentNodeWrite`
/// scope active. Always available (builtin engine) — no config flag (D1).
pub fn scrub_for_ledger(text: &str) -> ScrubbedText {
    let ctx = FilterContext::ledger_write(text.to_string());
    match BUILTIN_ENGINE.evaluate(&ctx) {
        Some(decision) => apply_filter_decision(text, &decision),
        None => ScrubbedText {
            text: text.to_string(),
            flagged: false,
            pattern: None,
        },
    }
}

/// Apply a `FilterDecision` to a plain string (not a `RouterRequest`).
///
/// - `OutputFilter`: replace every span rightmost-first (mirroring
///   `transforms::codeword_anonymize`'s sort) so earlier byte offsets stay
///   valid; `Redact`/`Anonymize` → `[REDACTED:<pattern>]`, `Omit` → the span is
///   deleted. Irreversible: no codeword map is retained.
/// - `HardReject`: the whole payload collapses to `[rejected: <pattern>]`.
/// - `SoftRedirect`: unchanged (a redirect target is meaningless on the write
///   path), `flagged = false`.
fn apply_filter_decision(text: &str, decision: &FilterDecision) -> ScrubbedText {
    match decision {
        FilterDecision::HardReject { pattern, .. } => ScrubbedText {
            text: format!("[rejected: {pattern}]"),
            flagged: true,
            pattern: Some(pattern.clone()),
        },
        FilterDecision::OutputFilter {
            action,
            matched_pattern,
            matches,
            ..
        } => {
            let mut sorted = matches.clone();
            sorted.sort_by_key(|m| std::cmp::Reverse(m.start));
            let mut result = text.to_string();
            for m in &sorted {
                let replacement = match action {
                    FilterAction::Redact | FilterAction::Anonymize => {
                        format!("[REDACTED:{matched_pattern}]")
                    }
                    FilterAction::Omit => String::new(),
                };
                result.replace_range(m.start..m.end, &replacement);
            }
            ScrubbedText {
                text: result,
                flagged: true,
                pattern: Some(matched_pattern.clone()),
            }
        }
        FilterDecision::SoftRedirect { .. } => ScrubbedText {
            text: text.to_string(),
            flagged: false,
            pattern: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::RegexMatch;

    fn output_decision(
        action: FilterAction,
        pattern: &str,
        matches: Vec<(usize, usize)>,
    ) -> FilterDecision {
        FilterDecision::OutputFilter {
            action: action.clone(),
            matched_pattern: pattern.to_string(),
            codewords: Default::default(),
            matches: matches
                .into_iter()
                .map(|(start, end)| RegexMatch {
                    pattern_name: pattern.to_string(),
                    matched_text: "x".to_string(),
                    start,
                    end,
                    action: action.clone(),
                })
                .collect(),
        }
    }

    #[test]
    fn clean_text_unchanged_and_unflagged() {
        let s = scrub_for_ledger("What is the capital of France?");
        assert!(!s.flagged);
        assert_eq!(s.text, "What is the capital of France?");
        assert_eq!(s.pattern, None);
    }

    #[test]
    fn email_is_redacted() {
        let s = scrub_for_ledger("Contact user@example.com for help.");
        assert!(s.flagged);
        assert_eq!(s.pattern.as_deref(), Some("email"));
        assert_eq!(s.text, "Contact [REDACTED:email] for help.");
        assert!(!s.text.contains("user@example.com"), "email must be gone");
    }

    #[test]
    fn phone_is_redacted() {
        let s = scrub_for_ledger("Call me at 555-123-4567 soon.");
        assert!(s.flagged);
        assert_eq!(s.pattern.as_deref(), Some("phone"));
        assert_eq!(s.text, "Call me at [REDACTED:phone] soon.");
    }

    #[test]
    fn ssn_is_redacted() {
        let s = scrub_for_ledger("My ssn is 123-45-6789.");
        assert!(s.flagged);
        assert_eq!(s.pattern.as_deref(), Some("ssn"));
        assert_eq!(s.text, "My ssn is [REDACTED:ssn].");
    }

    #[test]
    fn api_key_is_rejected() {
        let s = scrub_for_ledger("here is the key: api_key = abcdefghijklmnop");
        assert!(s.flagged);
        assert_eq!(s.pattern.as_deref(), Some("api_key"));
        assert_eq!(s.text, "[rejected: api_key]");
    }

    #[test]
    fn adjacent_matches_replaced_rightmost_first_without_index_corruption() {
        let decision = output_decision(FilterAction::Redact, "p1", vec![(0, 3), (3, 6), (6, 9)]);
        let s = apply_filter_decision("aaaBBBccc", &decision);
        assert_eq!(s.text, "[REDACTED:p1][REDACTED:p1][REDACTED:p1]");
        assert!(s.flagged);
    }

    #[test]
    fn omit_action_deletes_the_span() {
        let decision = output_decision(FilterAction::Omit, "secret", vec![(7, 18)]);
        let s = apply_filter_decision("prefix SECRETVALUE suffix", &decision);
        assert_eq!(s.text, "prefix  suffix");
        assert!(s.flagged);
    }

    #[test]
    fn anonymize_collapses_to_redact_marker() {
        // Anonymize is irreversible on the write path: same marker as Redact.
        let decision = output_decision(FilterAction::Anonymize, "email", vec![(8, 24)]);
        let s = apply_filter_decision("Contact user@example.com for help.", &decision);
        assert_eq!(s.text, "Contact [REDACTED:email] for help.");
    }
}
