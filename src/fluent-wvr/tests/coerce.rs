#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::coerce::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


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
