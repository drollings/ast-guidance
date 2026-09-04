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
    let l = en();
    let blob = l.blob.as_ref().expect("rule mode carries a blob");
    assert_eq!(blob.pos_count(), 5);
    assert!(!blob.rules("noun").is_empty());
    assert!(blob.index_contains("noun", "aardvark"));
    assert!(blob.exc_for("verb", "went").is_some());
    assert_eq!(l.mode(), LemmatizerMode::Rule);
}
