//! 20+ string utilities: case-insensitive search, slug, truncation, identifier detection.

use std::collections::HashSet;
use std::str::Chars;
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

// ── ANSI / control-character sanitizers ───────────────────────────────
// NOTE (ROADMAP_20260903_LLM M11): think-block stripping
// (`strip_thinking_blocks`, `strip_think_block`, `StreamingThinkFilter`)
// and the SSE drainer (`drain_sse_lines`) lived here through M10 as
// deprecated byte-identical shims of `fluent_llm::{thinking, sse}`; M11
// deleted them. The generic `find_subseq` stays (composed by
// `fluent_llm::thinking`).

/// Remove unsafe control / formatting characters: C0/C1 controls, bidi
/// overrides (U+202A–U+202E), line/paragraph separators (U+2028/U+2029),
/// and Plane-14 tags (U+E0000–U+E007F).
pub fn filter_unsafe_chars(text: &str) -> String {
    text.chars().filter(|&c| is_safe_char(c)).collect()
}

fn is_safe_char(c: char) -> bool {
    !matches!(
        c,
        '\u{0000}'
            | '\u{007F}'..='\u{009F}'
            | '\u{0001}'..='\u{0008}'
            | '\u{000B}'..='\u{000C}'
            | '\u{000E}'..='\u{001F}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{202A}'
            | '\u{202B}'
            | '\u{202C}'
            | '\u{202D}'
            | '\u{202E}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{E0000}'..='\u{E007F}'
    )
}

/// Iterator that strips ANSI escape sequences (CSI control sequences, e.g.
/// SGR color codes) from text. A lone ESC not followed by `[` is preserved.
pub struct AnsiStripper<'a> {
    chars: Chars<'a>,
}

impl<'a> AnsiStripper<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            chars: text.chars(),
        }
    }
}

impl Iterator for AnsiStripper<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        if c == '\u{1B}' {
            // Check for '[' following ESC — that starts a CSI sequence
            if let Some('[') = self.chars.clone().next() {
                self.chars.next(); // consume '['
                skip_csi_params(&mut self.chars);
                skip_csi_final(&mut self.chars);
                // Recurse to get the next visible character
                self.next()
            } else {
                // Lone ESC, not part of a CSI sequence
                Some('\u{1B}')
            }
        } else {
            Some(c)
        }
    }
}

fn skip_csi_params(chars: &mut Chars<'_>) {
    // Skip parameter bytes (0x30-0x3F) and intermediate bytes (0x20-0x2F)
    loop {
        let mut peek = chars.clone();
        match peek.next() {
            Some(p)
                if ('\u{0030}'..='\u{003F}').contains(&p)
                    || ('\u{0020}'..='\u{002F}').contains(&p) =>
            {
                chars.next();
            }
            _ => break,
        }
    }
}

fn skip_csi_final(chars: &mut Chars<'_>) {
    // Skip the final byte (0x40-0x7E) if present
    let mut peek = chars.clone();
    if let Some(f) = peek.next() {
        if ('\u{0040}'..='\u{007E}').contains(&f) {
            chars.next();
        }
    }
}

// ── Doc/identifier helpers ─────────────────────────────────────────────

/// Strip Rust `///`, `//!`, and `#` doc-comment prefixes from every line,
/// preserving inner indentation. The `#` arm also strips the Markdown-hidden
/// `# ` prefix on doc examples.
pub fn trim_doc_prefix(text: &str) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    for line in &mut lines {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("///") {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        } else if let Some(rest) = trimmed.strip_prefix("//!") {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        } else if let Some(rest) = trimmed.strip_prefix('#') {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        }
    }
    lines.join("\n")
}

/// Identifier case-style classification for a query string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentifierKind {
    CamelCase,
    PascalCase,
    SnakeCase,
    KebabCase,
    DottedPath,
    Other,
}

/// Detect the case-style of `query` without a regex. Returns `None` for empty
/// or whitespace-only input, `Some(kind)` otherwise. The pure classification
/// logic — callers that additionally require a syntactically valid identifier
/// (e.g. guidance's `detect_identifier_pattern`) apply their own gate on top.
pub fn detect_identifier_kind(query: &str) -> Option<IdentifierKind> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.contains('.') && !trimmed.contains(' ') {
        return Some(IdentifierKind::DottedPath);
    }

    let kind = if trimmed.contains('-') && !trimmed.contains(' ') {
        IdentifierKind::KebabCase
    } else if trimmed.contains('_')
        && trimmed
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
    {
        IdentifierKind::SnakeCase
    } else if trimmed
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
        && !trimmed.contains('_')
        && !trimmed.contains('-')
    {
        IdentifierKind::PascalCase
    } else if trimmed
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        IdentifierKind::CamelCase
    } else {
        IdentifierKind::Other
    };
    Some(kind)
}

pub fn slugify(text: &str) -> String {
    slugify_with(text, &SlugOptions::legacy_default())
}

/// Options for the parameterized slug primitive (P4).
///
/// `slugify(text) == slugify_with(text, &SlugOptions::legacy_default())`
/// byte-for-byte (locked in by `slugify_characterization_table`); the
/// router's chart-name slug is `CHART_OPTIONS` (separator `'_'`, run
/// collapsing, trailing-separator strip, char-bounded length cap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlugOptions {
    /// `true`: ASCII-lowercase and map every other non-kept char to the
    /// separator. `false` (legacy): Unicode-lowercase, map only `' '` to the
    /// separator, drop all other non-kept chars.
    pub ascii_only: bool,
    /// The separator emitted for blanks/other chars (`'-'` legacy, `'_'` charts).
    pub separator: char,
    /// Collapse consecutive separators into one (charts; legacy keeps runs).
    pub collapse_runs: bool,
    /// Strip leading separators (legacy trims `'-'` ends; charts never strip
    /// leading — a leading kept char is preserved and a leading separator is
    /// never emitted thanks to collapsing).
    pub trim_leading: bool,
    /// Strip trailing separators.
    pub trim_trailing: bool,
    /// Char-boundary truncation (charts cap at 64; legacy is uncapped).
    pub max_chars: Option<usize>,
}

impl SlugOptions {
    /// Options reproducing the legacy [`slugify`] byte-for-byte.
    pub fn legacy_default() -> Self {
        Self {
            ascii_only: false,
            separator: '-',
            collapse_runs: false,
            trim_leading: true,
            trim_trailing: true,
            max_chars: None,
        }
    }
}

/// Parameterized slugify: lowercase alphanumerics plus `_`/`-` kept, blanks
/// and (in `ascii_only` mode) every other non-kept char folded to
/// [`SlugOptions::separator`], optional run-collapsing, end-trimming, and
/// char-boundary truncation.
pub fn slugify_with(text: &str, opts: &SlugOptions) -> String {
    let lowered = if opts.ascii_only {
        text.to_ascii_lowercase()
    } else {
        text.to_lowercase()
    };
    let mut out = String::with_capacity(lowered.len());
    for c in lowered.chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            out.push(c);
        } else if c == ' ' || opts.ascii_only {
            if opts.collapse_runs && (out.is_empty() || out.ends_with(opts.separator)) {
                continue;
            }
            out.push(opts.separator);
        }
        // Else (!ascii_only, non-space, non-kept): drop (legacy behavior).
    }
    if opts.trim_leading {
        let prefix = out.len() - out.trim_start_matches(opts.separator).len();
        out.drain(..prefix);
    }
    if opts.trim_trailing {
        while out.ends_with(opts.separator) {
            out.pop();
        }
    }
    if let Some(max) = opts.max_chars {
        if out.chars().count() > max {
            let byte_offset = out
                .char_indices()
                .nth(max)
                .map_or(out.len(), |(i, _)| i);
            out.truncate(byte_offset);
        }
    }
    out
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

/// Truncate to `max_bytes` at a UTF-8 char boundary, appending `…` if the
/// input was truncated. Never panics on mid-character boundaries (unlike a
/// raw `&s[..n]` byte slice) and never exceeds `max_bytes`.
pub fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = s[..end].to_string();
    truncated.push('…');
    truncated
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

/// Truncate `s` to at most `max_chars` Unicode scalar values, never cutting
/// a code point. Unlike `truncate_utf8` (which is byte-based and appends a
/// `…` ellipsis when truncating), this is a hard char cap with no suffix —
/// the caller decides whether to append anything.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

