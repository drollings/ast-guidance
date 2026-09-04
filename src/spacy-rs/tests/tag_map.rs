use super::*;
use crate::lang::en;

#[test]
fn english_default_maps_fine_grained_tags() {
    let map = en::tag_map();
    assert_eq!(map.get("NN"), Some(Upos::Noun));
    assert_eq!(map.get("NNP"), Some(Upos::Propn));
    assert_eq!(map.get("VBD"), Some(Upos::Verb));
    assert_eq!(map.get("IN"), Some(Upos::Adp));
    assert_eq!(map.get("."), Some(Upos::Punct));
    assert_eq!(map.get("_SP"), Some(Upos::Space));
    assert_eq!(map.get("BOGUS"), None);
    assert_eq!(map.len(), 56);
}

#[test]
fn from_str_and_display_roundtrip() {
    let map: TagMap = "NN=NOUN,VBD=VERB,NNP=PROPN".parse().expect("parse");
    assert_eq!(map.get("NN"), Some(Upos::Noun));
    // Upos renders its lowercase name (the UPOS table key); parsing is
    // case-insensitive, so the round-trip is case-preserving only for
    // lowercase input.
    assert_eq!(map.to_string(), "NN=noun,NNP=propn,VBD=verb");
    let again: TagMap = map.to_string().parse().expect("reparse");
    assert_eq!(again, map);
}

#[test]
fn from_str_rejects_bad_pos() {
    assert!(matches!(
        "NN=NOTAPOS".parse::<TagMap>(),
        Err(SpacyError::UnknownPos(_))
    ));
    assert!(matches!(
        "NNNOUN".parse::<TagMap>(),
        Err(SpacyError::Annotation(_))
    ));
}

#[test]
fn from_pairs_is_case_insensitive_on_pos() {
    let map = TagMap::from_pairs(&[("NN", "noun"), ("VBD", "VERB")]).expect("pairs");
    assert_eq!(map.get("NN"), Some(Upos::Noun));
    assert_eq!(map.get("VBD"), Some(Upos::Verb));
}

#[test]
fn insert_and_len() {
    let mut map = TagMap::default();
    assert!(map.is_empty());
    map.insert("NN", Upos::Noun);
    map.insert("NN", Upos::Propn); // overwrite
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("NN"), Some(Upos::Propn));
}
