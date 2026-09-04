/// The ONE request-path anonymize entry (M6): twelve sequential regex
/// replacements with stable `[TYPE]` placeholders.
///
/// KNOWN CALLERS (the exhaustive list — every upstream pipeline routes
/// through this function, never a private copy):
/// - `fluent-router` `transforms::pii_anonymize::PiiAnonymize` (+ reverse map)
/// - `fluent-router` `transforms::DecomposeToAnonymizedHypothetical`
/// - `fluent-router` `dispatch::escalation` question mode (hypotheticals)
/// - `coral` `tier_units` (pre-LLM query scrub)
///
/// Deliberately SEPARATE vocabularies (not this function — do not merge;
/// each is locked by its own goldens):
/// - `SecretMask` (`****` transport masking, key-name preserving)
/// - `CodewordAnonymizer` (`CODEWORD_<TYPE>_N`, session-scoped reversible)
/// - `ledger_guard::scrub_for_ledger` (`[REDACTED:<pattern>]`, irreversible
///   write-path policy driven by the filter engine)
///
/// Idempotent (fixed-point) and placeholder/codeword pass-through — locked
/// by `tests/anonymize.rs` (M6.1 taxonomy + M6.2 golden table).
pub fn anonymize(text: &str) -> String {
    use crate::pii_patterns::{
        API_KEY_RE, AWS_KEY_RE, BEARER_RE, CREDIT_CARD_RE, EMAIL_RE, GENERIC_API_KEY_RE, IPV4_RE,
        IPV6_RE, NINO_UK_RE, PHONE_US_RE, SIN_CA_RE, SSN_US_RE,
    };

    let text = EMAIL_RE.replace_all(text, "[EMAIL]");
    let text = CREDIT_CARD_RE.replace_all(&text, "[CREDIT_CARD]");
    let text = SSN_US_RE.replace_all(&text, "[SSN]");
    let text = NINO_UK_RE.replace_all(&text, "[NINO]");
    let text = SIN_CA_RE.replace_all(&text, "[SIN]");
    let text = BEARER_RE.replace_all(&text, "[BEARER_TOKEN]");
    let text = AWS_KEY_RE.replace_all(&text, "[AWS_KEY]");
    let text = GENERIC_API_KEY_RE.replace_all(&text, "[API_KEY]");
    let text = IPV6_RE.replace_all(&text, "[IPv6]");
    let text = IPV4_RE.replace_all(&text, "[IP_ADDRESS]");
    let text = PHONE_US_RE.replace_all(&text, "[PHONE]");
    let text = API_KEY_RE.replace_all(&text, "[REDACTED]");
    text.to_string()
}

/// Rebuild the placeholder → original-value reverse map by byte-diffing
/// `original` against its `anonymized` form (M6: extracted verbatim from
/// `fluent-router` `transforms::pii_anonymize` — same walk, same keying).
///
/// Placeholders are `[...]` spans (≤64 chars) in the anonymized text; the
/// first occurrence of a placeholder keys as the placeholder itself, repeats
/// as `{placeholder}_{n}`. Entries accumulate into `map` (never cleared, so
/// multi-message callers share one map). Pure string diff — PII-independent:
/// it never consults a pattern table, only the two texts.
///
/// Callers must pass the exact `anonymize` output as `anonymized`; any other
/// pairing yields a best-effort diff, never an error.
pub fn build_anonymize_map(
    original: &str,
    anonymized: &str,
    map: &mut std::collections::HashMap<String, String, impl std::hash::BuildHasher>,
) {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let orig_bytes = original.as_bytes();
    let anon_bytes = anonymized.as_bytes();
    let mut orig_idx = 0;
    let mut anon_idx = 0;

    while orig_idx < orig_bytes.len() && anon_idx < anon_bytes.len() {
        if orig_bytes[orig_idx] == anon_bytes[anon_idx] {
            orig_idx += 1;
            anon_idx += 1;
            continue;
        }

        let remaining_orig = &original[orig_idx..];
        let placeholder = extract_placeholder(&anonymized[anon_idx..]);
        if let Some(ph) = placeholder {
            let ph_len = ph.len();
            let matched_len = find_matching_len(remaining_orig, ph);
            if matched_len > 0 {
                let original_value = &remaining_orig[..matched_len];
                let count = counts.entry(ph).or_insert(0);
                let key = if *count == 0 {
                    ph.to_string()
                } else {
                    format!("{ph}_{count}")
                };
                *count += 1;
                map.insert(key, original_value.to_string());

                orig_idx += matched_len;
                anon_idx += ph_len;
                continue;
            }
        }

        orig_idx += 1;
        anon_idx += 1;
    }
}

fn extract_placeholder(s: &str) -> Option<&str> {
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let candidate = &s[..=end];
            if candidate.len() <= 64 {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_matching_len(text: &str, _placeholder: &str) -> usize {
    let non_anon_end = text
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric() || c.is_ascii_punctuation())
        .map_or(text.len(), |(i, c)| i + c.len_utf8());
    non_anon_end.max(1)
}

#[cfg(test)]
#[path = "../tests/anonymize.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/anonymize_map.rs"]
mod anonymize_map_tests;
