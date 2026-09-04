//! Think-block stripping — the single owner (ROADMAP_20260903_LLM M1).
//!
//! Moved verbatim from `common_core::string` (`THINKING_PAIRS`,
//! `strip_tag_pairs`, `strip_plain_thinking`, `strip_thinking_blocks`,
//! `strip_think_block`, `StreamingThinkFilter`). The only cross-crate
//! dependency is the generic `common_core::string::find_subseq` (a byte
//! subsequence search — stays in `common-core`).
//!
//! M11 deleted the `common-core::string` byte-identical shim copies (kept
//! through M10 under `#[deprecated]`); the owner goldens in
//! `tests/thinking.rs` are the lasting contract.

use common_core::string::find_subseq;

// ── Thinking block stripping ──────────────────────────────────────────

const THINKING_PAIRS: &[(&[u8], &[u8])] = &[
    (b"\x3cthink\x3e", b"\x3c/think\x3e"), // Ollama: <think>...</think>
    (b"\x3cthinking\x3e", b"\x3c/thinking\x3e"), // Claude/Gemini: <thinking>...</thinking>
    (b"[THINK]", b"[/THINK]"),             // Bracket format
];

/// Strip content between start and end markers. Returns the text with content
/// between each matching pair removed. If a start marker is found without a
/// matching end marker, everything from the start marker onward is stripped.
fn strip_tag_pairs(text: &str, pairs: &[(&[u8], &[u8])]) -> String {
    let mut result = String::with_capacity(text.len());
    let mut pos = 0;
    let bytes = text.as_bytes();

    while pos < bytes.len() {
        let mut earliest: Option<usize> = None;
        let mut matched_pair: Option<(&[u8], &[u8])> = None;

        for &(start_mark, end_mark) in pairs {
            if let Some(start) = find_subseq(bytes, pos, start_mark) {
                if earliest.is_none_or(|e| start < e) {
                    earliest = Some(start);
                    matched_pair = Some((start_mark, end_mark));
                }
            }
        }

        if let Some((start_mark, end_mark)) = matched_pair {
            let start = earliest.unwrap();
            result.push_str(&text[pos..start]);
            let after_start = start + start_mark.len();
            if let Some(end) = find_subseq(bytes, after_start, end_mark) {
                pos = end + end_mark.len();
            } else {
                return result;
            }
        } else {
            result.push_str(&text[pos..]);
            return result;
        }
    }

    result
}

/// Strip plain-text ` thinking ...  response\n` delimiters
/// (DeepSeek R1, unsloth thinking). The end delimiter must be followed by a
/// newline or end-of-string.
fn strip_plain_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut pos = 0;
    let bytes = text.as_bytes();

    while pos < bytes.len() {
        if let Some(start) = find_subseq(bytes, pos, b" thinking") {
            let after_start = start + 9;
            match find_subseq(bytes, after_start, b" response") {
                Some(end) if end + 9 >= bytes.len() || bytes[end + 9] == b'\n' => {
                    result.push_str(&text[pos..start]);
                    let after_end = end + 9;
                    pos = if after_end < bytes.len() {
                        after_end + 1
                    } else {
                        after_end
                    };
                }
                _ => return result,
            }
        } else {
            result.push_str(&text[pos..]);
            return result;
        }
    }

    result
}

/// Strip thinking blocks from the given text. Handles multiple formats:
/// - `<think>...</think>` (Ollama-style XML tags)
/// - `<thinking>...</thinking>` (Claude, Gemini, some local models)
/// - ` thinking ...  response\n` (DeepSeek R1, unsloth thinking)
///
/// Tags can appear anywhere in the content and blocks may be unclosed.
pub fn strip_thinking_blocks(text: &str) -> String {
    let tagged = strip_tag_pairs(text, THINKING_PAIRS);
    if tagged != text {
        return tagged;
    }
    strip_plain_thinking(text)
}

/// Strip `<think>` and `[THINK]` tag pairs. Delegates to
/// `strip_thinking_blocks` which handles all known thinking-block formats.
pub fn strip_think_block(text: &str) -> String {
    strip_thinking_blocks(text)
}

// ── Streaming think-block filter ──────────────────────────────────────

/// Streaming think-block filter for token-chunked LLM output.
///
/// Feeds delta chunks via [`Self::push`] and returns only the text safe to
/// emit. Trailing text that is a proper prefix of a think open/close tag
/// (e.g. `<thi`) is held back until the next chunk completes it, so a tag
/// split across chunk boundaries never leaks a partial tag to the client.
/// Content inside an open think block is discarded; [`Self::finish`] returns
/// any trailing text that never resolved into a tag (the caller decides
/// whether to emit it at end-of-stream).
#[derive(Default)]
pub struct StreamingThinkFilter {
    in_think_block: bool,
    pending: String,
}

impl StreamingThinkFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one delta chunk and return the text safe to emit.
    pub fn push(&mut self, delta: &str) -> String {
        const OPEN_TAGS: &[&str] = &["<think>", "<thinking>"];
        const CLOSE_TAGS: &[&str] = &["</think>", "</thinking>"];

        let mut combined = String::with_capacity(self.pending.len() + delta.len());
        combined.push_str(&self.pending);
        combined.push_str(delta);
        self.pending.clear();

        let mut remaining: &str = &combined;
        let mut output = String::new();

        loop {
            if self.in_think_block {
                // Check for any closing tag in remaining.
                let mut earliest_close: Option<(usize, &str)> = None;
                for ct in CLOSE_TAGS {
                    if let Some(pos) = remaining.find(ct) {
                        if earliest_close.is_none_or(|(e, _)| pos < e) {
                            earliest_close = Some((pos, ct));
                        }
                    }
                }

                if let Some((pos, ct)) = earliest_close {
                    // Close the think block — emit nothing for its content.
                    self.in_think_block = false;
                    remaining = &remaining[pos + ct.len()..];
                    continue;
                }
                // Still inside the block — discard the definite content but
                // hold back a trailing close-tag prefix that may complete in
                // the next chunk (otherwise a split `</thi` + `nk>` close
                // would be missed and the tail would leak).
                let hold = Self::tag_prefix_len(remaining, CLOSE_TAGS);
                self.pending.push_str(&remaining[remaining.len() - hold..]);
                return output;
            }

            // Not inside a think block — scan for opening tags.
            let mut earliest_open: Option<(usize, &str)> = None;
            for ot in OPEN_TAGS {
                if let Some(pos) = remaining.find(ot) {
                    if earliest_open.is_none_or(|(e, _)| pos < e) {
                        earliest_open = Some((pos, ot));
                    }
                }
            }

            if let Some((pos, ot)) = earliest_open {
                output.push_str(&remaining[..pos]);
                self.in_think_block = true;
                remaining = &remaining[pos + ot.len()..];
                continue;
            }

            // No complete open tag — emit everything except a trailing run
            // that could be the start of a tag split across chunks.
            let hold = Self::tag_prefix_len(remaining, OPEN_TAGS);
            output.push_str(&remaining[..remaining.len() - hold]);
            self.pending.push_str(&remaining[remaining.len() - hold..]);
            return output;
        }
    }

    /// Trailing held-back text that never resolved into a tag. Call at
    /// end-of-stream to flush (e.g. a lone `<` that was never completed).
    pub fn finish(&self) -> String {
        self.pending.clone()
    }

    /// Length of the longest suffix of `s` that is a proper prefix of any tag
    /// in `tags` — i.e. a run that could grow into a full tag once the next
    /// chunk arrives.
    fn tag_prefix_len(s: &str, tags: &[&str]) -> usize {
        let mut best = 0;
        for tag in tags {
            for len in 1..tag.len() {
                if s.ends_with(&tag[..len]) {
                    best = best.max(len);
                }
            }
        }
        best
    }
}
