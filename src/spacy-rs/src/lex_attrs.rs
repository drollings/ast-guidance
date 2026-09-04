//! Deterministic lexeme attribute functions — the port of
//! `spacy/lang/lex_attrs.py` `LEX_ATTRS`.
//!
//! Every function is a pure `&str → value` computation with no model in the
//! loop, filling `LexemeC` at lexeme creation. Category flags use the
//! `unicode-properties` crate, whose General_Category lookups match the
//! Unicode properties CPython's `str.is*` methods consult, so golden-table
//! parity against real spaCy holds for the category-based attributes.
//!
//! Known approximations (documented divergences from spaCy):
//! - `like_url` omits the final `URL_MATCH` regex fallback (added with the
//!   tokenizer data in the tokenizer milestone).
//! - `like_email` matches the whole token rather than a leading substring.

use std::collections::HashSet;
use std::sync::OnceLock;

use fancy_regex::Regex;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

use crate::lang::url::URL_PATTERN;

/// The base `URL_MATCH` regex (`tokenizer_exceptions.py:57`), compiled lazily.
fn url_match(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(URL_PATTERN).expect("URL_PATTERN is a valid regex"));
    re.is_match(text).unwrap_or(false)
}

/// `str.isalpha()` — every char has the Unicode `Alphabetic` property.
#[must_use]
pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}

/// `str.isascii()` — every char is in the ASCII range. (Empty string is
/// trivially ASCII, matching CPython.)
#[must_use]
pub fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

/// `str.isdigit()` — every char is a decimal digit (Nd) or one of the
/// superscript digits CPython special-cases (U+00B2, U+00B3, U+00B9).
#[must_use]
pub fn is_digit(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.general_category() == GeneralCategory::DecimalNumber
                || matches!(u32::from(c), 0x00B2 | 0x00B3 | 0x00B9)
        })
}

/// `str.islower()` — at least one cased char and no uppercase chars.
#[must_use]
pub fn is_lower(s: &str) -> bool {
    s.chars().any(char::is_lowercase) && s.chars().all(|c| !c.is_uppercase())
}

/// `str.isspace()` — every char has the Unicode `White_Space` property.
#[must_use]
pub fn is_space(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_whitespace)
}

/// `str.istitle()` — each word (run of cased chars) begins with an uppercase
/// or titlecase char and continues with lowercase chars.
#[must_use]
pub fn is_title(s: &str) -> bool {
    let mut cased = false;
    let mut at_word_start = true;
    for c in s.chars() {
        if c.is_uppercase() || c.general_category() == GeneralCategory::TitlecaseLetter {
            cased = true;
            if at_word_start {
                at_word_start = false;
            } else {
                return false;
            }
        } else if c.is_lowercase() {
            cased = true;
            if at_word_start {
                return false;
            }
            at_word_start = false;
        } else {
            at_word_start = true;
        }
    }
    cased
}

/// `str.isupper()` — at least one cased char and no lowercase chars.
#[must_use]
pub fn is_upper(s: &str) -> bool {
    s.chars().any(char::is_uppercase) && s.chars().all(|c| !c.is_lowercase())
}

/// `is_punct` (`lex_attrs.py:25-29`) — every char has General_Category P*.
#[must_use]
pub fn is_punct(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.general_category_group() == GeneralCategoryGroup::Punctuation)
}

/// `is_bracket` (`lex_attrs.py:53-55`).
#[must_use]
pub fn is_bracket(s: &str) -> bool {
    matches!(s, "(" | ")" | "[" | "]" | "{" | "}" | "<" | ">")
}

/// `is_quote` (`lex_attrs.py:58-62`).
#[must_use]
pub fn is_quote(s: &str) -> bool {
    matches!(
        s,
        "\"" | "'"
            | "`"
            | "«"
            | "»"
            | "‘"
            | "’"
            | "‚"
            | "‛"
            | "“"
            | "”"
            | "„"
            | "‟"
            | "‹"
            | "›"
            | "❮"
            | "❯"
            | "''"
            | "``"
    )
}

/// `is_left_punct` (`lex_attrs.py:65-69`).
#[must_use]
pub fn is_left_punct(s: &str) -> bool {
    matches!(
        s,
        "(" | "["
            | "{"
            | "<"
            | "\""
            | "'"
            | "«"
            | "‘"
            | "‚"
            | "‛"
            | "“"
            | "„"
            | "‟"
            | "‹"
            | "❮"
            | "``"
    )
}

/// `is_right_punct` (`lex_attrs.py:72-74`).
#[must_use]
pub fn is_right_punct(s: &str) -> bool {
    matches!(
        s,
        ")" | "]" | "}" | ">" | "\"" | "'" | "»" | "’" | "”" | "›" | "❯" | "''"
    )
}

/// `is_currency` (`lex_attrs.py:77-82`) — every char has General_Category Sc.
#[must_use]
pub fn is_currency(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.general_category() == GeneralCategory::CurrencySymbol)
}

/// `like_num` (`lex_attrs.py:39-50`) — leading sign stripped, optional
/// thousands/decimal separators, integer or one-slash fraction.
#[must_use]
pub fn like_num(text: &str) -> bool {
    let stripped = match text.strip_prefix(['+', '-', '±', '~']) {
        Some(rest) => rest,
        None => text,
    };
    let normalized = stripped.replace(['.', ','], "");
    if !normalized.is_empty() && normalized.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    if stripped.matches('/').count() == 1 {
        let (num, denom) = stripped.split_once('/').expect("one slash");
        if !num.is_empty()
            && !denom.is_empty()
            && num.bytes().all(|b| b.is_ascii_digit())
            && denom.bytes().all(|b| b.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

/// The English `LIKE_NUM` (`spacy/lang/en/lex_attrs.py`) — the base shape
/// plus the language's cardinal/ordinal number words. Replaces the base
/// `like_num` when the lexicon config supplies `num_words`.
#[must_use]
pub fn like_num_en<S: std::hash::BuildHasher>(text: &str, num_words: &HashSet<String, S>) -> bool {
    let stripped = match text.strip_prefix(['+', '-', '±', '~']) {
        Some(rest) => rest,
        None => text,
    };
    let normalized = stripped.replace(['.', ','], "");
    if is_digit(&normalized) {
        return true;
    }
    if stripped.matches('/').count() == 1 {
        let (num, denom) = stripped.split_once('/').expect("one slash");
        if is_digit(num) && is_digit(denom) {
            return true;
        }
    }
    let lower = text.to_lowercase();
    if num_words.contains(&lower) {
        return true;
    }
    if (lower.ends_with("st")
        || lower.ends_with("nd")
        || lower.ends_with("rd")
        || lower.ends_with("th"))
        && is_digit(&lower[..lower.len() - 2])
    {
        return true;
    }
    false
}

/// The TLD table from `lex_attrs.py:9-22`.
const TLDS: &[&str] = &[
    "com", "org", "edu", "gov", "net", "mil", "aero", "asia", "biz", "cat", "coop", "info", "int",
    "jobs", "mobi", "museum", "name", "pro", "tel", "travel", "xyz", "icu", "xxx", "ac", "ad",
    "ae", "af", "ag", "ai", "al", "am", "an", "ao", "aq", "ar", "as", "at", "au", "aw", "ax", "az",
    "ba", "bb", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bm", "bn", "bo", "br", "bs", "bt", "bv",
    "bw", "by", "bz", "ca", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cr",
    "cs", "cu", "cv", "cx", "cy", "cz", "dd", "de", "dj", "dk", "dm", "do", "dz", "ec", "ee", "eg",
    "eh", "er", "es", "et", "eu", "fi", "fj", "fk", "fm", "fo", "fr", "ga", "gb", "gd", "ge", "gf",
    "gg", "gh", "gi", "gl", "gm", "gn", "gp", "gq", "gr", "gs", "gt", "gu", "gw", "gy", "hk", "hm",
    "hn", "hr", "ht", "hu", "id", "ie", "il", "im", "in", "io", "iq", "ir", "is", "it", "je", "jm",
    "jo", "jp", "ke", "kg", "kh", "ki", "km", "kn", "kp", "kr", "kw", "ky", "kz", "la", "lb", "lc",
    "li", "lk", "lr", "ls", "lt", "lu", "lv", "ly", "ma", "mc", "md", "me", "mg", "mh", "mk", "ml",
    "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv", "mw", "mx", "my", "mz", "na", "nc",
    "ne", "nf", "ng", "ni", "nl", "no", "np", "nr", "nu", "nz", "om", "pa", "pe", "pf", "pg", "ph",
    "pk", "pl", "pm", "pn", "pr", "ps", "pt", "pw", "py", "qa", "re", "ro", "rs", "ru", "rw", "sa",
    "sb", "sc", "sd", "se", "sg", "sh", "si", "sj", "sk", "sl", "sm", "sn", "so", "sr", "ss", "st",
    "su", "sv", "sy", "sz", "tc", "td", "tf", "tg", "th", "tj", "tk", "tl", "tm", "tn", "to", "tp",
    "tr", "tt", "tv", "tw", "tz", "ua", "ug", "uk", "us", "uy", "uz", "va", "vc", "ve", "vg", "vi",
    "vn", "vu", "wf", "ws", "ye", "yt", "za", "zm", "zw",
];

/// `like_url` (`lex_attrs.py:89-114`) — scheme/`www.` prefixes, dotted TLD
/// membership, email exclusion. Omits the `URL_MATCH` regex last resort.
#[must_use]
pub fn like_url(text: &str) -> bool {
    if text.starts_with("http://") || text.starts_with("https://") {
        return true;
    }
    if text.starts_with("www.") && text.len() >= 5 {
        return true;
    }
    if text.starts_with('.') || text.ends_with('.') {
        return false;
    }
    if text.contains('@') {
        return false;
    }
    if !text.contains('.') {
        return false;
    }
    let tld = text
        .rsplit_once('.')
        .map(|(_, tld)| tld.split(':').next().unwrap_or(tld))
        .unwrap_or_default();
    if tld.ends_with('/') {
        return true;
    }
    if tld.chars().all(char::is_alphabetic) && TLDS.contains(&tld) {
        return true;
    }
    url_match(text)
}

/// `_like_email` (`lex_attrs.py:8`) — `local@domain.tld` shape. The regex
/// anchors at the start; we require the whole token to match.
#[must_use]
pub fn like_email(s: &str) -> bool {
    let Some(at) = s.find('@') else {
        return false;
    };
    let local = &s[..at];
    let rest = &s[at + 1..];
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let domain = &rest[..dot];
    let tld = &rest[dot + 1..];
    !local.is_empty()
        && !domain.is_empty()
        && !tld.is_empty()
        && local
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '+' | '-'))
        && domain
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
        && tld
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
}

/// `word_shape` (`lex_attrs.py:117-141`) — maps chars to X (upper), x (lower),
/// d (digit), or the literal char; collapses runs beyond 4; returns `"LONG"`
/// for strings ≥ 100 chars.
#[must_use]
pub fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".to_string();
    }
    let mut shape = String::with_capacity(text.len());
    let mut last: Option<char> = None;
    let mut seq = 0usize;
    for c in text.chars() {
        let shape_char = if c.is_alphabetic() {
            if c.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if c.is_numeric() {
            'd'
        } else {
            c
        };
        if Some(shape_char) == last {
            seq += 1;
        } else {
            seq = 0;
            last = Some(shape_char);
        }
        if seq < 4 {
            shape.push(shape_char);
        }
    }
    shape
}

/// `lower` (`lex_attrs.py:144-145`) — the lowercased form.
#[must_use]
pub fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// `prefix` (`lex_attrs.py:148-149`) — the first char.
#[must_use]
pub fn prefix(s: &str) -> String {
    s.chars().next().map_or_else(String::new, |c| c.to_string())
}

/// `suffix` (`lex_attrs.py:152-153`) — the last 3 chars.
#[must_use]
pub fn suffix(s: &str) -> String {
    s.chars()
        .rev()
        .take(3)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

#[cfg(test)]
#[path = "../tests/lex_attrs.rs"]
mod tests;
