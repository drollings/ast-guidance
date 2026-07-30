//! 20+ string utilities: case-insensitive search, slug, truncation, identifier detection.

use std::collections::HashSet;
use std::sync::LazyLock;

pub static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    for w in &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "used", "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "mine", "yours", "hers", "ours", "theirs", "and", "but", "or", "nor", "not", "so",
        "yet", "for", "in", "on", "at", "to", "by", "with", "from", "of", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once",
    ] {
        s.insert(*w);
    }
    s
});

pub fn trim_right<'a>(slice: &'a [u8], pattern: &[u8]) -> &'a [u8] {
    let mut end = slice.len();
    while end > 0 && pattern.contains(&slice[end - 1]) {
        end -= 1;
    }
    &slice[..end]
}

pub fn trim_left<'a>(slice: &'a [u8], pattern: &[u8]) -> &'a [u8] {
    let mut start = 0;
    while start < slice.len() && pattern.contains(&slice[start]) {
        start += 1;
    }
    &slice[start..]
}

fn contains_word_with_boundary(text: &str, word: &str, is_boundary: fn(u8) -> bool) -> bool {
    let lower = text.to_lowercase();
    let lower_word = word.to_lowercase();
    let bytes = lower.as_bytes();
    let wb = lower_word.as_bytes();
    if wb.is_empty() || wb.len() > bytes.len() {
        return false;
    }
    let mut i = 0;
    while i + wb.len() <= bytes.len() {
        if &bytes[i..i + wb.len()] == wb {
            let left_boundary = i == 0 || is_boundary(bytes[i - 1]);
            let right_boundary = i + wb.len() == bytes.len() || is_boundary(bytes[i + wb.len()]);
            if left_boundary && right_boundary {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_ident_boundary(c: u8) -> bool {
    !c.is_ascii_alphanumeric()
}

pub fn contains_ident_word(haystack: &str, needle: &str) -> bool {
    contains_word_with_boundary(haystack, needle, is_ident_boundary)
}

pub fn contains_any(text: &str, keywords: &[&str]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|k| lower.contains(&k.to_lowercase()))
}

pub fn contains_any_word(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|k| contains_word(text, k))
}

pub fn contains_ignore_case(text: &str, pattern: &str) -> bool {
    text.to_lowercase().contains(&pattern.to_lowercase())
}

pub fn contains_word(text: &str, word: &str) -> bool {
    contains_word_with_boundary(text, word, |c| !c.is_ascii_alphanumeric())
}

pub fn first_comment_line(text: &str) -> Option<String> {
    let line = text.lines().next()?;
    let trimmed = line
        .trim()
        .trim_start_matches("///")
        .trim()
        .trim_start_matches("//!")
        .trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn has_extension(path: &str, ext: &str) -> bool {
    let ext = ext.trim_start_matches('.');
    path.to_lowercase()
        .ends_with(&format!(".{}", ext.to_lowercase()))
}

pub fn looks_like_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn lower_into<'a>(dst: &'a mut [u8], src: &[u8]) -> &'a [u8] {
    let len = src.len().min(dst.len());
    for i in 0..len {
        dst[i] = src[i].to_ascii_lowercase();
    }
    &dst[..len]
}

/// Lightweight HTML tag stripper — no regex dependency. Strips `<script>`,
/// `<style>`, and all other tags, decodes common entities, replaces
/// block-level tags (`<br>`, `</p>`, `</div>`, `</li>`, `</tr>`) with
/// newlines, and collapses whitespace.
/// Suitable for untrusted text going to an LLM or field validator.
pub fn strip_html(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_script = false;
    let mut in_style = false;
    let mut chars = s.char_indices().peekable();

    while let Some((i, c)) = chars.next() {
        if c == '<' {
            let rest = &s[i + 1..];
            let tag_start = rest.trim_start();
            let lower: String = tag_start
                .chars()
                .take_while(|ch| ch.is_alphanumeric())
                .collect();
            let lower = lower.to_ascii_lowercase();

            // Check for closing tags first — in_script/in_style flags block the
            // normal tag handler, so we must detect </script>/</style> here.
            if in_script && lower.is_empty() {
                if let Some(past_slash) = tag_start.strip_prefix('/') {
                    let rest_name: String = past_slash
                        .chars()
                        .take_while(|ch| ch.is_alphanumeric())
                        .collect();
                    if rest_name.eq_ignore_ascii_case("script") {
                        in_script = false;
                        while let Some(&(_, ch)) = chars.peek() {
                            chars.next();
                            if ch == '>' {
                                break;
                            }
                        }
                        continue;
                    }
                }
            }
            if in_style && lower.is_empty() {
                if let Some(past_slash) = tag_start.strip_prefix('/') {
                    let rest_name: String = past_slash
                        .chars()
                        .take_while(|ch| ch.is_alphanumeric())
                        .collect();
                    if rest_name.eq_ignore_ascii_case("style") {
                        in_style = false;
                        while let Some(&(_, ch)) = chars.peek() {
                            chars.next();
                            if ch == '>' {
                                break;
                            }
                        }
                        continue;
                    }
                }
            }

            if lower.starts_with("script") {
                in_script = true;
                continue;
            }
            if lower.starts_with("style") {
                in_style = true;
                continue;
            }
            // Check for block-level break tags and replace with newline.
            if lower.is_empty() {
                if let Some(past_slash) = tag_start.strip_prefix('/') {
                    let closing_name: String = past_slash
                        .chars()
                        .take_while(|ch| ch.is_alphanumeric())
                        .collect();
                    if matches!(closing_name.as_str(), "br" | "p" | "div" | "li" | "tr") {
                        result.push('\n');
                    }
                }
            } else if matches!(lower.as_str(), "br" | "p" | "div" | "li" | "tr") {
                result.push('\n');
            }
            // Skip everything until '>'
            while let Some(&(_, ch)) = chars.peek() {
                chars.next();
                if ch == '>' {
                    break;
                }
            }
            continue;
        }

        if in_script || in_style {
            continue;
        }

        result.push(c);
    }

    // Decode common HTML entities
    let result = result.replace("&amp;", "&");
    let result = result.replace("&lt;", "<");
    let result = result.replace("&gt;", ">");
    let result = result.replace("&quot;", "\"");
    let result = result.replace("&#39;", "'");

    // Collapse whitespace (including newlines from block-tag substitution)
    let mut out = String::with_capacity(result.len());
    let mut prev_space = false;
    for c in result.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

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

pub fn slugify(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
        .map(|c| if c == ' ' { '-' } else { c })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

pub fn truncate_at_sentence(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    // Find the byte offset for the max_chars-th character
    let byte_offset = text
        .char_indices()
        .nth(max_chars)
        .map_or(text.len(), |(i, _)| i);
    let truncated = &text[..byte_offset];
    if let Some(last_period) = truncated.rfind('.') {
        // Find the char index of the period to check if it's past the midpoint
        let period_char_count = truncated[..last_period].chars().count();
        if period_char_count > max_chars / 2 {
            return text[..=last_period].to_string();
        }
    }
    truncated.to_string()
}

/// Return the first sentence of `text` — trimmed text up to and including
/// the first `.`, `!`, or `?`. If no sentence-ending punctuation is found,
/// returns up to 120 characters.
pub fn first_sentence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(idx) = trimmed.find(['.', '!', '?']) {
        trimmed[..=idx].trim().to_string()
    } else {
        trimmed.chars().take(120).collect()
    }
}

pub fn is_path_token(s: &str) -> bool {
    s.len() >= 3 && (s.contains('/') || s.contains('\\'))
}

pub fn is_test_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.contains("test")
        || lower.contains("spec")
        || lower.ends_with("_test.zig")
        || lower.ends_with("_tests.zig")
}

pub fn strip_boilerplate(text: &str, prefix: &str) -> String {
    if let Some(stripped) = text.strip_prefix(prefix) {
        stripped.trim_start().to_string()
    } else {
        text.to_string()
    }
}

const NL_PREFIXES: &[&str] = &[
    "what is ",
    "what are ",
    "what does ",
    "what's ",
    "where is ",
    "where are ",
    "where does ",
    "where can i find ",
    "how does ",
    "how do ",
    "how can i ",
    "how to ",
    "why is ",
    "why does ",
    "why do ",
    "when is ",
    "when does ",
    "when do ",
    "who is ",
    "who are ",
    "who does ",
    "which is ",
    "which are ",
    "which does ",
    "explain ",
    "define ",
    "describe ",
    "tell me about ",
];

pub fn strip_nl_prefix(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    for &prefix in NL_PREFIXES {
        if lower.starts_with(prefix) {
            return text[prefix.len()..].to_string();
        }
    }
    text.to_string()
}

pub fn is_noisy_comment(comment: &str) -> bool {
    if comment.len() < 10 {
        return true;
    }
    let non_alpha: usize = comment
        .chars()
        .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
        .count();
    non_alpha as f64 / comment.len() as f64 > 0.5
}

pub fn skill_name_from_ref(ref_path: &str) -> String {
    let normalized = ref_path.replace('\\', "/");
    let basename = normalized.rsplit('/').next().unwrap_or(ref_path);
    if basename.eq_ignore_ascii_case("SKILL.md") {
        normalized
            .rsplit('/')
            .nth(1)
            .unwrap_or(basename)
            .to_string()
    } else {
        basename.to_string()
    }
}

/// Deferred-UTF-8-decode SSE line drainer.
///
/// Appends `chunk` to `buffer`, drains every complete newline-terminated line,
/// decodes it via `String::from_utf8_lossy`, trims trailing whitespace, and
/// returns the lines.  Any unterminated tail is left in `buffer` for the next
/// call.
///
/// Because `\n` (0x0A) is never a UTF-8 lead or continuation byte, splitting on
/// it cannot cut a codepoint, so every drained line is a whole number of
/// codepoints and decodes losslessly — safe for CJK, emoji, etc.
pub fn drain_sse_lines(buffer: &mut Vec<u8>, chunk: &[u8]) -> Vec<String> {
    buffer.extend_from_slice(chunk);
    let mut lines = Vec::new();
    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buffer.drain(..=pos).collect();
        lines.push(String::from_utf8_lossy(&line).trim_end().to_string());
    }
    lines
}

/// Find the first occurrence of a byte subsequence in a slice starting from `start`.
pub fn find_subseq(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| start + i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_sse_lines_cjk_split_across_chunks() {
        let mut buf = Vec::new();
        // "data: 안녕\n" — Korean "annyeong" — UTF-8: EC 95 88 EB 85 95
        // Split at byte 7, which falls inside the "안" character (EC 95 88).
        let chunk1 = b"data: \xEC\x95";
        let chunk2 = b"\x88\xEB\x85\x95\n";
        let lines = drain_sse_lines(&mut buf, chunk1);
        assert!(lines.is_empty(), "no complete line yet");
        let lines = drain_sse_lines(&mut buf, chunk2);
        assert_eq!(lines.len(), 1);
        let decoded = &lines[0];
        assert!(
            !decoded.contains('\u{FFFD}'),
            "got replacement character in: {decoded:?}"
        );
        assert!(decoded.contains("안녕"), "expected 안녕, got: {decoded:?}");
        assert!(buf.is_empty(), "buffer should be drained");
    }

    #[test]
    fn drain_sse_lines_partial_tail_reassembly() {
        let mut buf = Vec::new();
        let lines1 = drain_sse_lines(&mut buf, b"event: ping\r\ndata: {}\npartial");
        assert_eq!(lines1.len(), 2);
        assert_eq!(lines1[0], "event: ping");
        assert_eq!(lines1[1], "data: {}");
        assert_eq!(&buf, b"partial", "tail should remain in buffer");
        let lines2 = drain_sse_lines(&mut buf, b" tail\n");
        assert_eq!(lines2.len(), 1);
        assert_eq!(lines2[0], "partial tail");
        assert!(buf.is_empty(), "buffer should be drained");
    }

    #[test]
    fn contains_ignore_case_basic() {
        assert!(contains_ignore_case("Hello World", "hello"));
        assert!(contains_ignore_case("Hello World", "WORLD"));
        assert!(!contains_ignore_case("Hello World", "goodbye"));
    }

    #[test]
    fn contains_word_boundary() {
        assert!(contains_word("test builder", "builder"));
        assert!(!contains_word("test builders", "builder"));
    }

    #[test]
    fn first_comment_line_strips_prefix() {
        assert_eq!(
            first_comment_line("/// This is a doc comment\n/// more"),
            Some("This is a doc comment".into())
        );
    }

    #[test]
    fn has_extension_variants() {
        assert!(has_extension("file.zig", "zig"));
        assert!(has_extension("file.ZIG", ".zig"));
        assert!(!has_extension("file.zig", "rs"));
    }

    #[test]
    fn looks_like_identifier_various() {
        assert!(looks_like_identifier("foo"));
        assert!(looks_like_identifier("_private"));
        assert!(!looks_like_identifier("123abc"));
    }

    #[test]
    fn slugify_converts() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("FooBar"), "foobar");
    }

    #[test]
    fn stop_words_contains_expected() {
        assert!(STOP_WORDS.contains("the"));
        assert!(STOP_WORDS.contains("is"));
        assert!(!STOP_WORDS.contains("zig"));
    }

    #[test]
    fn contains_ident_word_no_false_positive_on_substring() {
        assert!(!contains_ident_word("mystructfield", "struct"));
        assert!(!contains_ident_word("unstructured", "struct"));
        assert!(contains_ident_word("test_struct", "struct"));
    }

    #[test]
    fn truncate_at_sentence_boundary() {
        let text = "Hello world. This is a test. More content.";
        let result = truncate_at_sentence(text, 20);
        assert_eq!(result, "Hello world.");
    }

    #[test]
    fn truncate_at_sentence_within_limit() {
        let text = "Short";
        assert_eq!(truncate_at_sentence(text, 100), "Short");
    }

    #[test]
    fn trim_right_basic() {
        assert_eq!(trim_right(b"hello   ", b" "), b"hello");
    }

    #[test]
    fn trim_right_noop() {
        assert_eq!(trim_right(b"hello", b" "), b"hello");
    }

    #[test]
    fn trim_right_all_matching() {
        assert_eq!(trim_right(b"   ", b" "), b"");
    }

    #[test]
    fn trim_right_pattern_subset() {
        assert_eq!(trim_right(b"hello!?!", b"!?"), b"hello");
    }

    #[test]
    fn trim_left_basic() {
        assert_eq!(trim_left(b"   hello", b" "), b"hello");
    }

    #[test]
    fn trim_left_noop() {
        assert_eq!(trim_left(b"hello", b" "), b"hello");
    }

    #[test]
    fn contains_ident_word_basic() {
        assert!(contains_ident_word("my_struct_field", "struct"));
        assert!(!contains_ident_word("mystructfield", "struct"));
    }

    #[test]
    fn contains_ident_word_underscore_boundary() {
        assert!(contains_ident_word("test_foo_bar", "foo"));
        assert!(!contains_ident_word("testfoobar", "foo"));
    }

    #[test]
    fn lower_into_short_src() {
        let mut buf = [0u8; 16];
        let result = lower_into(&mut buf, b"HELLO");
        assert_eq!(result, b"hello");
    }

    #[test]
    fn lower_into_long_src_truncated() {
        let mut buf = [0u8; 4];
        let result = lower_into(&mut buf, b"HELLO WORLD");
        assert_eq!(result, b"hell");
    }

    #[test]
    fn lower_into_empty_src() {
        let mut buf = [0u8; 4];
        let result = lower_into(&mut buf, b"");
        assert_eq!(result, b"");
    }

    #[test]
    fn contains_any_basic() {
        assert!(contains_any("hello world", &["hello"]));
        assert!(contains_any("hello world", &["world", "foo"]));
        assert!(!contains_any("hello world", &["foo"]));
    }

    #[test]
    fn contains_any_word_basic() {
        assert!(contains_any_word("test builder", &["test", "builder"]));
        assert!(contains_any_word("test builder", &["builder"]));
        assert!(!contains_any_word("test builders", &["builder"]));
    }

    #[test]
    fn truncate_at_sentence_no_period() {
        let text = "This is a long string with no period at all in the first half";
        let result = truncate_at_sentence(text, 20);
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn truncate_at_sentence_period_too_early() {
        let text = "A. very long string that continues past the limit";
        let result = truncate_at_sentence(text, 20);
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn contains_ident_word_empty_needle() {
        assert!(!contains_ident_word("test", ""));
    }

    #[test]
    fn contains_ident_word_needle_longer_than_haystack() {
        assert!(!contains_ident_word("abc", "abcdef"));
    }

    #[test]
    fn looks_like_identifier_empty() {
        assert!(!looks_like_identifier(""));
    }

    #[test]
    fn looks_like_identifier_starts_with_digit() {
        assert!(!looks_like_identifier("123abc"));
    }

    #[test]
    fn slugify_trims_dashes() {
        assert_eq!(slugify("-hello-"), "hello");
    }

    #[test]
    fn first_comment_line_with_notice_prefix() {
        assert_eq!(
            first_comment_line("//! Module level doc\n/// member"),
            Some("Module level doc".into())
        );
    }

    #[test]
    fn first_comment_line_empty_after_strip() {
        assert_eq!(first_comment_line("///"), None);
    }

    #[test]
    fn has_extension_case_sensitivity() {
        assert!(has_extension("file.ZIG", ".zig"));
        assert!(!has_extension("file.rs", ".zig"));
    }

    #[test]
    fn contains_ident_word_boundary_special_chars() {
        assert!(contains_ident_word("foo->bar", "bar"));
        assert!(!contains_ident_word("foobar", "bar"));
    }

    #[test]
    fn is_path_token_min_length() {
        assert!(!is_path_token("ab"));
        assert!(is_path_token("a/b"));
    }

    #[test]
    fn is_noisy_comment_checks() {
        assert!(is_noisy_comment("x"));
        assert!(!is_noisy_comment("Cosine similarity for vector search"));
    }

    #[test]
    fn is_test_path_detection() {
        assert!(is_test_path("src/test.rs"));
        assert!(is_test_path("tests/foo.rs"));
        assert!(is_test_path("foo_test.zig"));
        assert!(!is_test_path("src/main.rs"));
    }

    #[test]
    fn strip_boilerplate_removes_prefix() {
        assert_eq!(strip_boilerplate("fn foo()", "fn "), "foo()");
    }

    #[test]
    fn strip_boilerplate_no_match() {
        assert_eq!(strip_boilerplate("hello world", "fn "), "hello world");
    }

    #[test]
    fn strip_nl_prefix_new_semantics_what_is() {
        assert_eq!(strip_nl_prefix("what is X"), "X");
    }

    #[test]
    fn strip_nl_prefix_new_semantics_how_does() {
        assert_eq!(strip_nl_prefix("how does Y work"), "Y work");
    }

    #[test]
    fn strip_nl_prefix_new_semantics_no_match() {
        assert_eq!(strip_nl_prefix("hello world"), "hello world");
    }

    #[test]
    fn strip_nl_prefix_new_semantics_explain() {
        assert_eq!(strip_nl_prefix("explain Z"), "Z");
    }

    #[test]
    fn first_sentence_with_period() {
        assert_eq!(first_sentence("Hello world. More text"), "Hello world.");
    }

    #[test]
    fn first_sentence_with_exclamation() {
        assert_eq!(first_sentence("Great answer! Follow up"), "Great answer!");
    }

    #[test]
    fn first_sentence_with_question() {
        assert_eq!(first_sentence("What is this? More text."), "What is this?");
    }

    #[test]
    fn first_sentence_no_punctuation() {
        let result = first_sentence("Single sentence no punctuation");
        assert!(result.len() <= 120);
        assert_eq!(result, "Single sentence no punctuation");
    }

    #[test]
    fn first_sentence_empty() {
        assert_eq!(first_sentence(""), "");
    }

    #[test]
    fn first_sentence_whitespace_only() {
        assert_eq!(first_sentence("  "), "");
    }

    #[test]
    fn first_sentence_trims_leading_whitespace() {
        assert_eq!(first_sentence("  Hello. World"), "Hello.");
    }

    #[test]
    fn skill_name_from_ref_skil_md() {
        assert_eq!(
            skill_name_from_ref("doc/skills/zig-current/SKILL.md"),
            "zig-current"
        );
    }

    #[test]
    fn skill_name_from_ref_fallback() {
        assert_eq!(skill_name_from_ref("doc/skills/foo.md"), "foo.md");
    }
}
