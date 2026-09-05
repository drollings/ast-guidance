use super::*;

// The legacy literals these tests pin are the exact spellings previously
// hard-coded across arc_eager.rs (suffix args to `has_suffix_ci`, the three
// punctuation sets, the clause comma). The English blob must carry them
// byte-for-byte: that equality IS the zero-behavior-change proof for the
// blob migration at every call site.
fn legacy_trailing() -> Vec<&'static str> {
    vec![".", "!", "?", ";", ":", ",", "—", "--"]
}

fn legacy_boundary() -> Vec<&'static str> {
    vec![".", "!", "?", ";", ":", "-", "—", "--"]
}

fn legacy_bare_follow() -> Vec<&'static str> {
    vec![".", "!", "?", ";", ":", "—", "--", ",", "(", ")", "..."]
}

fn english() -> TaggerOrtho<'static> {
    TaggerOrtho::from_bytes(crate::lang::en::ORTHO_BLOB).expect("English ortho blob parses")
}

#[test]
fn english_blob_carries_exact_legacy_sets() {
    let o = english();
    assert_eq!(o.trailing_punct(), legacy_trailing().as_slice());
    assert_eq!(o.sentence_boundary(), legacy_boundary().as_slice());
    assert_eq!(o.bare_follow(), legacy_bare_follow().as_slice());
    assert_eq!(o.plural_s(), "s");
    assert_eq!(o.manner_ly(), "ly");
    assert_eq!(o.past_ed(), "ed");
    assert_eq!(o.part_ing(), "ing");
    assert_eq!(o.comma(), ",");
}

#[test]
fn predicates_match_legacy_spellings() {
    let o = english();
    // Suffix predicates vs the legacy case-insensitive idiom (total:
    // short words are false, where the raw idiom would underflow).
    let lic = |word: &str, suffix: &str| {
        word.len() >= suffix.len()
            && word
                .get(word.len() - suffix.len()..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case(suffix))
    };
    let words = [
        "calls", "CALLS", "as", "quarterly", "QUARTERLY", "July", "opened", "OPENED", "red",
        "smiling", "SMILING", "king", "rain", "raining", "go", "a", "today", "statuses",
    ];
    for w in words {
        assert_eq!(o.is_plural_s(w), lic(w, "s"), "plural {w:?}");
        assert_eq!(o.is_manner_ly(w), lic(w, "ly"), "manner {w:?}");
        assert_eq!(o.is_past_ed(w), lic(w, "ed"), "past {w:?}");
        assert_eq!(o.is_part_ing(w), lic(w, "ing"), "part {w:?}");
    }
    // Token predicates vs the legacy matches! sets.
    for t in [".", "!", "?", ";", ":", ",", "-", "—", "--", "(", ")", "...", "calls", "x", ""] {
        assert_eq!(
            o.is_trailing_punct(t),
            legacy_trailing().contains(&t),
            "trailing {t:?}"
        );
        assert_eq!(
            o.is_boundary(t),
            legacy_boundary().contains(&t),
            "boundary {t:?}"
        );
        assert_eq!(
            o.is_bare_follow(t),
            legacy_bare_follow().contains(&t),
            "bare {t:?}"
        );
        assert_eq!(o.is_comma(t), t == ",", "comma {t:?}");
    }
    // Span predicates vs the legacy slice-alls.
    let streams: Vec<Vec<String>> = [
        vec!["calls", "."],
        vec!["yet", "."],
        vec!["work", "?"],
        vec![".", "calls"],
        vec!["?", "x"],
        vec!["Call", "the", "office", "."],
        vec!["Sit", ".", "Call", "."],
    ]
    .iter()
    .map(|s| s.iter().map(|t| t.to_string()).collect())
    .collect();
    for texts in &streams {
        for from in 0..=texts.len() {
            let legacy = texts[from..].iter().all(|t| legacy_trailing().contains(&t.as_str()));
            assert_eq!(o.trailing_only(texts, from), legacy, "texts={texts:?}");
        }
        for i in 0..texts.len() {
            let legacy = i == 0 || legacy_boundary().contains(&texts[i - 1].as_str());
            assert_eq!(o.starts_sentence(texts, i), legacy, "texts={texts:?} i={i}");
        }
    }
}

#[test]
fn loader_rejects_bad_blobs() {
    // Wrong magic.
    let mut bad = crate::lang::en::ORTHO_BLOB.to_vec();
    bad[0] ^= 0xFF;
    assert!(TaggerOrtho::from_bytes(Box::leak(bad.into_boxed_slice())).is_err());
    // Wrong version.
    let mut bad = crate::lang::en::ORTHO_BLOB.to_vec();
    bad[4] ^= 0xFF;
    assert!(TaggerOrtho::from_bytes(Box::leak(bad.into_boxed_slice())).is_err());
    // Truncation (every prefix must fail, never panic).
    for len in 0..crate::lang::en::ORTHO_BLOB.len() {
        assert!(
            TaggerOrtho::from_bytes(&crate::lang::en::ORTHO_BLOB[..len]).is_err(),
            "prefix len {len} must fail"
        );
    }
    // Empty scalar / NUL entry / non-UTF8 are build-time rejects; the
    // loader defends the same invariants.
    assert!(TaggerOrtho::from_bytes(&[]).is_err());
}
