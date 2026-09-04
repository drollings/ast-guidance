use common_core::string::*;

// NOTE (ROADMAP_20260903_LLM M2): `drain_sse_lines` framing goldens moved to
// `fluent-llm --test sse` (canonical owner `fluent_llm::sse`).

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
fn truncate_at_sentence_basic() {
        assert_eq!(truncate_at_sentence("Hello world. This is a test. More content.", 20), "Hello world.");
        assert_eq!(truncate_at_sentence("Short", 100), "Short");
}

#[test]
fn truncate_utf8_empty_input() {
        assert_eq!(truncate_utf8("", 120), "");
}

#[test]
fn truncate_utf8_shorter_than_cap() {
        assert_eq!(truncate_utf8("hello", 120), "hello");
}

#[test]
fn truncate_utf8_boundary_exact_char() {
        // 4 ASCII chars, cap exactly at a char boundary
        assert_eq!(truncate_utf8("abcd", 4), "abcd");
        // 5 ASCII chars capped at 4 → "abcd…"
        assert_eq!(truncate_utf8("abcde", 4), "abcd…");
}

#[test]
fn truncate_utf8_mid_char_no_panic() {
        // CJK chars are 3 bytes each; 10 chars = 30 bytes.
        let s = "汉".repeat(10);
        assert_eq!(s.len(), 30);
        // cap at 29 bytes falls inside the 10th char (starts at byte 27)
        let out = truncate_utf8(&s, 29);
        assert!(out.starts_with(&"汉".repeat(9)));
        assert!(out.ends_with('…'));
        assert_eq!(out, format!("{}…", "汉".repeat(9)));
        // cap at 28 also lands mid-char
        let out = truncate_utf8(&s, 28);
        assert!(out.starts_with(&"汉".repeat(9)));
        assert!(out.ends_with('…'));
        // cap at 30 lands exactly on a boundary → no truncation
        assert_eq!(truncate_utf8(&s, 30), s);
}

#[test]
fn truncate_utf8_max_bytes_zero() {
        assert_eq!(truncate_utf8("anything", 0), "…");
        assert_eq!(truncate_utf8("", 0), "");
}

#[test]
fn truncate_utf8_never_exceeds_cap_before_ellipsis() {
        let emoji = "🚀".repeat(50);
        for cap in 0..emoji.len() {
            let out = truncate_utf8(&emoji, cap);
            let content = out.strip_suffix('…').unwrap_or(&out);
            assert!(content.len() <= cap);
        }
}

#[test]
fn trim_right_various() {
        assert_eq!(trim_right(b"hello   ", b" "), b"hello");
        assert_eq!(trim_right(b"hello", b" "), b"hello");
        assert_eq!(trim_right(b"   ", b" "), b"");
        assert_eq!(trim_right(b"hello!?!", b"!?"), b"hello");
}

#[test]
fn trim_left_various() {
        assert_eq!(trim_left(b"   hello", b" "), b"hello");
        assert_eq!(trim_left(b"hello", b" "), b"hello");
}


#[test]
fn contains_ident_word_underscore_boundary() {
        assert!(contains_ident_word("test_foo_bar", "foo"));
        assert!(!contains_ident_word("testfoobar", "foo"));
}

#[test]
fn lower_into_various() {
        let mut buf = [0u8; 16];
        assert_eq!(lower_into(&mut buf, b"HELLO"), b"hello");
        let mut buf = [0u8; 4];
        assert_eq!(lower_into(&mut buf, b"HELLO WORLD"), b"hell");
        let mut buf = [0u8; 4];
        assert_eq!(lower_into(&mut buf, b""), b"");
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
fn truncate_at_sentence_no_usable_boundary_falls_back_to_hard_cut() {
        for text in ["This is a long string with no period at all in the first half", "A. very long string that continues past the limit"] {
            assert_eq!(truncate_at_sentence(text, 20).len(), 20);
        }
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
fn slugify_trims_dashes() {
        assert_eq!(slugify("-hello-"), "hello");}

#[test]
fn slugify_characterization_table() {
        // Characterization (M4): verbatim outputs of `slugify` — the P4
        // migration must preserve them byte-for-byte.
        let cases = [
                ("", ""),
                ("---", ""),
                ("___", "___"),
                ("Hello World", "hello-world"),
                ("a  --  b", "a------b"),
                ("  padded  ", "padded"),
                ("-hello-", "hello"),
                ("_lead", "_lead"),
                ("trail_", "trail_"),
                ("Café Münchén", "caf-mnchn"),
                ("already-kebab", "already-kebab"),
                ("a_b c", "a_b-c"),
        ];
        for (input, want) in cases {
                assert_eq!(slugify(input), want, "slugify({input:?})");
        }
        // Parity: the parameterized primitive with legacy options.
        for (input, want) in cases {
                assert_eq!(
                        slugify_with(input, &SlugOptions::legacy_default()),
                        want,
                        "slugify_with(legacy, {input:?})"
                );
        }
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
fn strip_boilerplate_various() {
        assert_eq!(strip_boilerplate("fn foo()", "fn "), "foo()");
        assert_eq!(strip_boilerplate("hello world", "fn "), "hello world");
}

#[test]
fn strip_nl_prefix_various() {
        assert_eq!(strip_nl_prefix("what is X"), "X");
        assert_eq!(strip_nl_prefix("how does Y work"), "Y work");
        assert_eq!(strip_nl_prefix("explain Z"), "Z");
        assert_eq!(strip_nl_prefix("hello world"), "hello world");
}

#[test]
fn first_sentence_punctuation_variants() {
        assert_eq!(first_sentence("Hello world. More text"), "Hello world.");
        assert_eq!(first_sentence("Great answer! Follow up"), "Great answer!");
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

#[test]
fn truncate_chars_short_input_unchanged() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert_eq!(truncate_chars("", 5), "");
}

#[test]
fn truncate_chars_exact_boundary() {
        assert_eq!(truncate_chars("hello", 5), "hello");
}

#[test]
fn truncate_chars_cuts_at_char_boundary() {
        // 10 CJK chars (3 bytes each); cap at 4 → 4 chars, no partial char.
        let s = "汉".repeat(10);
        let out = truncate_chars(&s, 4);
        assert_eq!(out, "汉".repeat(4));
        assert_eq!(out.chars().count(), 4);
}

#[test]
fn truncate_chars_no_ellipsis() {
        assert_eq!(truncate_chars("abcdef", 3), "abc");
}

    // ── AnsiStripper ─────────────────────────────────────────────────

#[test]
fn ansi_stripper_passthrough_plain_text() {
        let result: String = AnsiStripper::new("hello world").collect();
        assert_eq!(result, "hello world");
}

#[test]
fn ansi_stripper_removes_sgr_color() {
        let result: String = AnsiStripper::new("\u{1B}[31mRED\u{1B}[0m").collect();
        assert_eq!(result, "RED");
}

#[test]
fn ansi_stripper_removes_extended_color() {
        assert_eq!(AnsiStripper::new("\u{1B}[38;5;196mbright").collect::<String>(), "bright");
        assert_eq!(AnsiStripper::new("\u{1B}[38;2;255;0;0mRGB red").collect::<String>(), "RGB red");
}

#[test]
fn ansi_stripper_lone_esc_preserved() {
        let result: String = AnsiStripper::new("a\u{1B}x").collect();
        assert_eq!(result, "a\u{1B}x");
}

#[test]
fn ansi_stripper_cjk_preserved() {
        let result: String = AnsiStripper::new("こんにちは").collect();
        assert_eq!(result, "こんにちは");
}

#[test]
fn ansi_stripper_empty_input() {
        let result: String = AnsiStripper::new("").collect();
        assert_eq!(result, "");
}

    // ── filter_unsafe_chars ──────────────────────────────────────────

#[test]
fn filter_unsafe_chars_various() {
        assert_eq!(filter_unsafe_chars("hello\u{0000}world"), "helloworld");
        assert_eq!(filter_unsafe_chars("a\u{0081}b"), "ab");
        assert_eq!(filter_unsafe_chars("before\x00after"), "beforeafter");
        assert_eq!(filter_unsafe_chars("plain text"), "plain text");
        assert_eq!(filter_unsafe_chars("hello\u{202E}world"), "helloworld");
        assert_eq!(filter_unsafe_chars("a\u{2028}b"), "ab");
        assert_eq!(filter_unsafe_chars("text\u{E0001}more"), "textmore");
}

    // ── trim_doc_prefix ───────────────────────────────────────────────

#[test]
fn trim_doc_prefix_various() {
        assert_eq!(trim_doc_prefix("/// Hello world\n/// more"), "Hello world\nmore");
        assert_eq!(trim_doc_prefix("//! Module\n# hidden"), "Module\nhidden");
        assert_eq!(trim_doc_prefix("///     indented"), "    indented");
        assert_eq!(trim_doc_prefix("plain line\nsecond"), "plain line\nsecond");
}

    // ── detect_identifier_kind ────────────────────────────────────────

#[test]
fn identifier_kind_cases() {
        assert_eq!(
            detect_identifier_kind("hello_world"),
            Some(IdentifierKind::SnakeCase)
        );
        assert_eq!(
            detect_identifier_kind("HelloWorld"),
            Some(IdentifierKind::PascalCase)
        );
        assert_eq!(
            detect_identifier_kind("helloWorld"),
            Some(IdentifierKind::CamelCase)
        );
        assert_eq!(
            detect_identifier_kind("kebab-case"),
            Some(IdentifierKind::KebabCase)
        );
        assert_eq!(
            detect_identifier_kind("a.b.c"),
            Some(IdentifierKind::DottedPath)
        );
        assert_eq!(
            detect_identifier_kind("two words"),
            Some(IdentifierKind::Other)
        );
        assert_eq!(detect_identifier_kind(""), None);
        assert_eq!(detect_identifier_kind("   "), None);
}

#[test]
fn identifier_kind_snake_allows_digits_but_not_upper() {
        assert_eq!(
            detect_identifier_kind("field_1"),
            Some(IdentifierKind::SnakeCase)
        );
        assert_eq!(
            detect_identifier_kind("Field_1"),
            Some(IdentifierKind::Other),
            "an uppercase start with underscores is not snake (guidance parity)"
        );
}
