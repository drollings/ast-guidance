//! Boundary-string coercion: turning untrusted strings into typed member values.
//!
//! This is the shared vocabulary for the fluent-wvr Boundary Rule — "data that
//! arrives from outside the process is always a string at the boundary." The
//! transforms here are what a caller applies to a raw boundary string (an LLM
//! classifier's response, a user-supplied form field, an IPC payload field, a
//! file/DB record column) before it becomes a typed member via
//! [`FieldAccess::set_field`] or a `serde` deserialize.
//!
//! The derive macro's `#[field(coerce = "trim,strip_quotes")]` attribute applies
//! a [`Coercion`] pipeline inside `set_field`, so the coercion policy lives with
//! the field definition (single source of truth) and every boundary decode
//! shares the same vocabulary instead of reimplementing string surgery per
//! consumer.
//!
//! Nothing here is JSON-specific: the same modes apply to key=value user input,
//! IPC text fields, and DB record columns.

use std::fmt::Write as _;

/// A single string→string shaping step applied to a boundary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coercion {
    /// Trim leading/trailing whitespace.
    Trim,
    /// Lowercase the whole string.
    Lowercase,
    /// Strip a matching outer `'` or `"` pair (escaping inner quotes and raw
    /// control characters so the result is JSON-safe).
    StripQuotes,
    /// Escape raw control characters (`\n`, `\t`, `\r`, `\b`, `\f`, `\uXXXX`)
    /// so the result is a valid JSON string body.
    JsonEscape,
    /// Map the null-ish literal spellings small models emit (`undefined`,
    /// `None`, `NaN`, `Infinity`, `null`) to an empty string, so an
    /// `Option<String>` member becomes `None`. For an optional member that is
    /// the honest "absent" outcome; leave non-literal text untouched.
    NormalizeLiteral,
}

impl Coercion {
    /// The attribute spelling (`#[field(coerce = "trim,strip_quotes")]`).
    pub fn name(self) -> &'static str {
        match self {
            Coercion::Trim => "trim",
            Coercion::Lowercase => "lowercase",
            Coercion::StripQuotes => "strip_quotes",
            Coercion::JsonEscape => "json_escape",
            Coercion::NormalizeLiteral => "normalize_literal",
        }
    }

    /// Resolve an attribute-spelling name, or `None` for an unknown mode.
    pub fn from_name(name: &str) -> Option<Coercion> {
        match name {
            "trim" => Some(Coercion::Trim),
            "lowercase" => Some(Coercion::Lowercase),
            "strip_quotes" => Some(Coercion::StripQuotes),
            "json_escape" => Some(Coercion::JsonEscape),
            "normalize_literal" => Some(Coercion::NormalizeLiteral),
            _ => None,
        }
    }

    /// Apply this single mode to a string.
    pub fn apply(self, s: &str) -> String {
        match self {
            Coercion::Trim => s.trim().to_string(),
            Coercion::Lowercase => s.to_lowercase(),
            Coercion::StripQuotes => strip_outer_quotes(s),
            Coercion::JsonEscape => escape_control_chars(s),
            Coercion::NormalizeLiteral => {
                if is_null_literal(s.trim()) {
                    String::new()
                } else {
                    s.to_string()
                }
            }
        }
    }
}

/// Apply a coercion pipeline in order. The modes run left-to-right, so
/// `[Trim, StripQuotes]` trims first and then strips a quote pair.
pub fn coerce(s: &str, modes: &[Coercion]) -> String {
    let mut out = s.to_string();
    for mode in modes {
        out = mode.apply(&out);
    }
    out
}

/// Escape a single raw control character to its JSON escape form, or `None`
/// when `c` is not a control character. This is the single source of truth for
/// control-character escaping: the boundary lexer, `escape_control_chars`, and
/// `strip_outer_quotes` all share it.
pub fn escaped_char(c: char) -> Option<String> {
    match c {
        '\n' => Some("\\n".to_string()),
        '\t' => Some("\\t".to_string()),
        '\r' => Some("\\r".to_string()),
        '\u{0008}' => Some("\\b".to_string()),
        '\u{000C}' => Some("\\f".to_string()),
        c if c.is_control() => {
            let mut s = String::with_capacity(6);
            let _ = write!(s, "\\u{:04x}", c as u32);
            Some(s)
        }
        _ => None,
    }
}

/// Escape every raw control character in `s`; all other characters (including
/// already-escaped sequences and quote characters) are left verbatim.
pub fn escape_control_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match escaped_char(c) {
            Some(e) => out.push_str(&e),
            None => out.push(c),
        }
    }
    out
}

/// Decode a `\x` escape sequence into its logical character plus the number of
/// input characters consumed, or `None` when the sequence is not a known JSON
/// escape. `\uXXXX` decodes a single code point (surrogate pairs are not
/// combined).
fn decode_escape(chars: &[char], i: usize, n: usize, quote: char) -> Option<(String, usize)> {
    let e = *chars.get(i + 1)?;
    match e {
        'n' => Some(("\n".to_string(), 2)),
        't' => Some(("\t".to_string(), 2)),
        'r' => Some(("\r".to_string(), 2)),
        'b' => Some(("\u{0008}".to_string(), 2)),
        'f' => Some(("\u{000C}".to_string(), 2)),
        '"' | '\\' | '/' => Some((e.to_string(), 2)),
        c if c == quote => Some((c.to_string(), 2)),
        'u' => {
            if i + 5 < n {
                let hex: String = chars[i + 2..i + 6].iter().collect();
                u32::from_str_radix(&hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .map(|ch| (ch.to_string(), 6))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Strip a matching outer single- or double-quote pair and decode the inner
/// content to its logical value:
///
/// - `'…'` / `"…"` pairs are removed and their content is JSON-escape-decoded
///   (`\"`, `\\`, `\'`, `\n`, `\t`, `\r`, `\b`, `\f`, `\uXXXX`); unknown
///   escapes are preserved verbatim.
/// - An opening quote with no matching close is treated as unterminated: the
///   decoded content is still produced (a dangling trailing backslash is
///   preserved).
/// - No quote pair at all (a bare value): returned unchanged.
///
/// The result is the *logical* member content — what `serde_json` will re-escape
/// when the value is serialized. The document-repair lexer keeps escape
/// sequences verbatim precisely because it rebuilds a JSON document; this
/// function extracts the value underneath the quotes.
#[allow(clippy::many_single_char_names)]
pub fn strip_outer_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    let open = chars[0];
    if open != '\'' && open != '"' {
        return s.to_string();
    }
    let quote = open;
    let mut out = String::with_capacity(s.len());
    let mut i = 1usize;
    let n = chars.len();
    while i < n {
        let c = chars[i];
        if c == '\\' {
            if let Some((decoded, consumed)) = decode_escape(&chars, i, n, quote) {
                out.push_str(&decoded);
                i += consumed;
            } else {
                out.push('\\');
                if i + 1 < n {
                    out.push(chars[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }
        if c == quote {
            break;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// The literal spellings small models emit for a JSON `null`.
fn is_null_literal(s: &str) -> bool {
    matches!(
        s,
        "null" | "None" | "undefined" | "NaN" | "Infinity" | "-Infinity" | "-NaN"
    )
}

/// Classification of a boundary string into the JSON literal it spells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralKind {
    /// A `null`-ish spelling (`undefined`, `None`, `NaN`, `Infinity`, `null`).
    Null,
    /// A boolean spelling (`true`/`True`, `false`/`False`).
    Bool(bool),
    /// Anything else — plain text, not a literal.
    Text,
}

/// Classify a boundary string as a JSON literal. Trims surrounding whitespace
/// first. Used by callers that must know whether a value is a literal (`null`,
/// a boolean) before deciding how to coerce it.
pub fn literal_kind(s: &str) -> LiteralKind {
    let t = s.trim();
    if is_null_literal(t) {
        LiteralKind::Null
    } else {
        match t {
            "true" | "True" => LiteralKind::Bool(true),
            "false" | "False" => LiteralKind::Bool(false),
            _ => LiteralKind::Text,
        }
    }
}

/// Tolerant float parse: trims, strips a surrounding quote pair, removes inner
/// whitespace and `_` separators, drops a trailing `f`/`F`/`d`/`D` numeric
/// suffix, and rejects `Infinity`/`NaN`/empty input.
pub fn parse_number(s: &str) -> Option<f64> {
    let t = strip_outer_quotes(s);
    let t = t.trim();
    if t.is_empty() || is_null_literal(t) {
        return None;
    }
    let t: String = t.chars().filter(|c| !c.is_whitespace()).collect();
    let t = t.replace('_', "");
    let t = t
        .strip_suffix(['f', 'F', 'd', 'D'])
        .unwrap_or(&t)
        .to_string();
    t.parse::<f64>().ok()
}

/// Tolerant integer parse over [`parse_number`].
pub fn parse_int(s: &str) -> Option<i64> {
    let n = parse_number(s)?;
    if n.fract() != 0.0 || n < i64::MIN as f64 || n > i64::MAX as f64 {
        return None;
    }
    Some(n as i64)
}

/// Tolerant boolean parse: `true`/`True`/`1`/`yes` → `Some(true)`,
/// `false`/`False`/`0`/`no` → `Some(false)`, anything else → `None`.
pub fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "true" | "True" | "1" | "yes" | "on" => Some(true),
        "false" | "False" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Split a boundary string on a separator character at the top nesting level,
/// respecting quote pairs and `()`/`[]`/`{}` nesting so separators inside a
/// quoted or nested region are never split. This is the primitive for parsing
/// list-valued members (e.g. a `route` list or a `"tags"` value) out of loose
/// text.
pub fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth: i32 = 0;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in s.chars() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            cur.push(c);
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                cur.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' => {
                depth = (depth - 1).max(0);
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coerce_pipeline_runs_in_order() {
        assert_eq!(
            coerce("  'hello'  ", &[Coercion::Trim, Coercion::StripQuotes]),
            "hello"
        );
        assert_eq!(
            coerce("  'Hello'  ", &[Coercion::Trim, Coercion::StripQuotes, Coercion::Lowercase]),
            "hello"
        );
    }

    #[test]
    fn coercion_names_round_trip() {
        for c in [
            Coercion::Trim,
            Coercion::Lowercase,
            Coercion::StripQuotes,
            Coercion::JsonEscape,
            Coercion::NormalizeLiteral,
        ] {
            assert_eq!(Coercion::from_name(c.name()), Some(c));
        }
        assert_eq!(Coercion::from_name("bogus"), None);
    }

    #[test]
    fn strip_outer_quotes_handles_single_and_double() {
        assert_eq!(strip_outer_quotes("'code'"), "code");
        assert_eq!(strip_outer_quotes("\"code\""), "code");
        assert_eq!(strip_outer_quotes("'it\\'s'"), "it's");
        // Escape sequences decode to their logical content.
        assert_eq!(strip_outer_quotes("\"say \\\"hi\\\"\""), "say \"hi\"");
        assert_eq!(strip_outer_quotes("'say \"hi\"'"), "say \"hi\"");
        assert_eq!(strip_outer_quotes("\"a\\nb\""), "a\nb");
        assert_eq!(strip_outer_quotes("\"tab\\there\""), "tab\there");
        // Unknown escapes are preserved verbatim.
        assert_eq!(strip_outer_quotes("\"a\\qb\""), "a\\qb");
        // `\uXXXX` decodes a code point.
        assert_eq!(strip_outer_quotes("\"caf\\u00e9\""), "caf\u{e9}");
    }

    #[test]
    fn strip_outer_quotes_passes_raw_controls_through() {
        // Raw (unescaped) control characters are logical content: the caller's
        // serializer re-escapes them on output.
        assert_eq!(strip_outer_quotes("'a\nb'"), "a\nb");
        assert_eq!(strip_outer_quotes("\"tab\there\""), "tab\there");
    }

    #[test]
    fn strip_outer_quotes_unterminated() {
        assert_eq!(strip_outer_quotes("'hello"), "hello");
        // Dangling backslash is preserved for the caller to re-escape.
        assert_eq!(strip_outer_quotes("'trailing\\"), "trailing\\");
    }

    #[test]
    fn strip_outer_quotes_bare_value() {
        assert_eq!(strip_outer_quotes("route"), "route");
        assert_eq!(strip_outer_quotes("  spaced  "), "  spaced  ");
    }

    #[test]
    fn escape_control_chars_maps_common_and_unicode() {
        assert_eq!(escape_control_chars("a\nb\tc"), "a\\nb\\tc");
        assert_eq!(escape_control_chars("\u{0008}\u{000C}"), "\\b\\f");
        assert_eq!(escape_control_chars("a\u{0001}b"), "a\\u0001b");
        // Non-controls pass through untouched.
        assert_eq!(escape_control_chars("plain \" text"), "plain \" text");
    }

    #[test]
    fn normalize_literal_maps_null_spellings_to_empty() {
        assert_eq!(Coercion::NormalizeLiteral.apply("undefined"), "");
        assert_eq!(Coercion::NormalizeLiteral.apply("None"), "");
        assert_eq!(Coercion::NormalizeLiteral.apply("NaN"), "");
        assert_eq!(Coercion::NormalizeLiteral.apply("Infinity"), "");
        assert_eq!(Coercion::NormalizeLiteral.apply("null"), "");
        assert_eq!(Coercion::NormalizeLiteral.apply("plain text"), "plain text");
    }

    #[test]
    fn literal_kind_classifies() {
        assert_eq!(literal_kind("undefined"), LiteralKind::Null);
        assert_eq!(literal_kind("-Infinity"), LiteralKind::Null);
        assert_eq!(literal_kind("true"), LiteralKind::Bool(true));
        assert_eq!(literal_kind("False"), LiteralKind::Bool(false));
        assert_eq!(literal_kind("route"), LiteralKind::Text);
    }

    #[test]
    fn parse_number_tolerates_junk() {
        assert_eq!(parse_number("0.9"), Some(0.9));
        assert_eq!(parse_number("'0.9'"), Some(0.9));
        assert_eq!(parse_number("\"7\""), Some(7.0));
        assert_eq!(parse_number("1_000"), Some(1000.0));
        assert_eq!(parse_number("12f"), Some(12.0));
        assert_eq!(parse_number("3.5 d"), Some(3.5));
        assert_eq!(parse_number(" 5 "), Some(5.0));
        assert_eq!(parse_number("undefined"), None);
        assert_eq!(parse_number("Infinity"), None);
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("abc"), None);
    }

    #[test]
    fn parse_int_and_bool() {
        assert_eq!(parse_int("7"), Some(7));
        assert_eq!(parse_int("7.0"), Some(7));
        assert_eq!(parse_int("7.5"), None);
        assert_eq!(parse_bool("true"), Some(true));
        assert_eq!(parse_bool("True"), Some(true));
        assert_eq!(parse_bool("yes"), Some(true));
        assert_eq!(parse_bool("false"), Some(false));
        assert_eq!(parse_bool("0"), Some(false));
        assert_eq!(parse_bool("route"), None);
    }

    #[test]
    fn split_top_level_respects_nesting() {
        assert_eq!(
            split_top_level("a, b, c", ','),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            split_top_level("a, 'b, c', d", ','),
            vec!["a".to_string(), "'b, c'".to_string(), "d".to_string()]
        );
        assert_eq!(
            split_top_level("[1, 2], [3, 4]", ','),
            vec!["[1, 2]".to_string(), "[3, 4]".to_string()]
        );
        assert_eq!(split_top_level("solo", ','), vec!["solo".to_string()]);
        assert_eq!(split_top_level("", ','), Vec::<String>::new());
    }
}
