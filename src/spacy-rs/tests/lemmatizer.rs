use super::*;
use crate::morph::Morphology;
use std::sync::Arc;

fn en() -> Lemmatizer {
    let strings = Arc::new(StringStore::new());
    rule_lemmatizer_with_strings(strings)
}

#[test]
fn plural_nouns_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("cats", Upos::Noun, 0)[0], "cat");
    assert_eq!(l.lemmatize("boxes", Upos::Noun, 0)[0], "box");
    assert_eq!(l.lemmatize("wives", Upos::Noun, 0)[0], "wife");
    assert_eq!(l.lemmatize("children", Upos::Noun, 0)[0], "child", "exception");
}

#[test]
fn verb_inflections_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("running", Upos::Verb, 0)[0], "run");
    assert_eq!(l.lemmatize("studies", Upos::Verb, 0)[0], "study");
    assert_eq!(l.lemmatize("went", Upos::Verb, 0)[0], "go", "exception");
}

#[test]
fn aux_be_forms_resolve_to_be() {
    // UD + spaCy lookup parity: auxiliary be-forms resolve to the lemma
    // "be" (8 bench pins, unanimous where pinned; all other be-forms carry
    // empty refs and auto-credit either way). Closed irregular class, same
    // shape as the ca/wo/n't splinter map below.
    let l = en();
    for surface in ["is", "are", "was", "were", "am", "be", "been", "being"] {
        assert_eq!(l.lemmatize(surface, Upos::Aux, 0)[0], "be", "{surface}");
    }
}

#[test]
fn verb_fallback_lowercases_sentence_case() {
    // Sentence case carries no lexical information on verbs (never proper
    // nouns): a capitalized VERB with no applicable rule falls back to the
    // lowercased form, not the surface. 16 bench pins, unanimous.
    let l = en();
    for surface in ["Let", "Close", "Send", "Buy", "Run", "Work"] {
        assert_eq!(
            l.lemmatize(surface, Upos::Verb, 0)[0],
            surface.to_lowercase(),
            "{surface}"
        );
    }
}

#[test]
fn acronym_fallback_lowercases() {
    // Must-NOT-fire (scope): all-caps tokens are acronyms, never
    // title-case names — "CEO" lowercases while "French" keeps surface.
    let l = en();
    assert_eq!(l.lemmatize("CEO", Upos::Noun, 0)[0], "ceo");
}

#[test]
fn titlecase_nominal_fallback_keeps_surface() {
    // Must-NOT-fire: title-case nominals may be proper (`French`, `John`)
    // — the fallback keeps surface, per the proper-noun convention pinned
    // by refs and `proper_nouns_keep_case`. (Common title-initials like
    // `Study` stay a documented gap: proper/common needs §8.2 evidence.)
    let l = en();
    assert_eq!(l.lemmatize("French", Upos::Noun, 0)[0], "French");
}

#[test]
fn be_lemma_verb_path_uses_tables() {
    // Invariance: a VERB-tagged "was" already resolved to "be" through the
    // blob verb exceptions before the Aux map existed — both routes agree,
    // so the Aux gate changes no VERB behavior. Pins the table route
    // against future blob regressions.
    let l = en();
    assert_eq!(l.lemmatize("was", Upos::Verb, 0)[0], "be");
    assert_eq!(l.lemmatize("was", Upos::Aux, 0)[0], "be");
}

#[test]
fn adjective_comparatives_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("faster", Upos::Adj, 0)[0], "fast");
    // spaCy's exception table for "better" is ["good", "well"]; exceptions
    // are inserted at position 0 in order, so the LAST lemma wins.
    assert_eq!(l.lemmatize("better", Upos::Adj, 0)[0], "well", "exception");
}

#[test]
fn proper_nouns_keep_case() {
    let l = en();
    assert_eq!(l.lemmatize("Apple", Upos::Propn, 0)[0], "Apple");
}

#[test]
fn unknown_pos_lowercases() {
    let l = en();
    assert_eq!(l.lemmatize("THE", Upos::Det, 0)[0], "the");
    assert_eq!(l.lemmatize("123", Upos::Num, 0)[0], "123");
}

#[test]
fn base_form_skips_lemmatization() {
    let strings = Arc::new(StringStore::new());
    let m = Arc::new(Morphology::new(strings));
    let l = en().with_morphology(m);
    // Number=Sing noun is a base form
    let sing = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("Number=Sing");
    assert!(l.is_base_form(Upos::Noun, sing));
    assert_eq!(l.lemmatize("cat", Upos::Noun, sing)[0], "cat");
    // VerbForm=Inf is a base form
    let inf = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("VerbForm=Inf");
    assert!(l.is_base_form(Upos::Verb, inf));
    // Plural is not
    let plur = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("Number=Plur");
    assert!(!l.is_base_form(Upos::Noun, plur));
    assert_eq!(l.lemmatize("cats", Upos::Noun, plur)[0], "cat");
}

#[test]
fn cache_reuses_analyses() {
    let l = en();
    assert_eq!(l.cache_len(), 0);
    let a = l.lemmatize("cats", Upos::Noun, 0);
    assert_eq!(a, vec!["cat"]);
    assert_eq!(l.cache_len(), 1);
    let b = l.lemmatize("cats", Upos::Noun, 0);
    assert_eq!(a, b);
    assert_eq!(l.cache_len(), 1);
}

#[test]
fn lookup_mode() {
    let table = HashMap::from([
        ("went".to_string(), "go".to_string()),
        ("cats".to_string(), "cat".to_string()),
    ]);
    let l = Lemmatizer::lookup(table);
    assert_eq!(l.lemmatize("went", Upos::Verb, 0)[0], "go");
    assert_eq!(l.lemmatize("cats", Upos::Noun, 0)[0], "cat");
    assert_eq!(l.lemmatize("unknown", Upos::Noun, 0)[0], "unknown");
}

#[test]
fn english_rule_data_is_loaded() {
    // Wiring only: the rule-mode lemmatizer carries the English blob.
    // Blob content (keys, counts, index, exceptions) is pinned in
    // `lemma_blob.rs` — not duplicated here.
    let l = en();
    assert!(l.blob.as_ref().is_some_and(|b| !b.rules("noun").is_empty()));
    assert_eq!(l.mode(), LemmatizerMode::Rule);
}

#[test]
fn lemma_key_is_display_without_alloc() {
    // Single source of truth for the `Upos` → lemma-blob-key mapping: the
    // hot path must match on the enum (zero-cost, typo-impossible) instead
    // of round-tripping through `to_string()` plus `== "noun"` literals.
    // Every key renders exactly as `Display` (the blob's contract), as a
    // `&'static str` with no per-token allocation.
    for pos in Upos::UPOS {
        let key: &'static str = pos.lemma_key();
        assert_eq!(key, pos.to_string(), "{pos:?}");
    }
    assert_eq!(Upos::Propn.lemma_key(), "propn");
    assert_eq!(Upos::Noun.lemma_key(), "noun");
    assert_eq!(Upos::Verb.lemma_key(), "verb");
    assert_eq!(Upos::Adj.lemma_key(), "adj");
}
