use super::*;

#[test]
fn upos_display_and_fromstr_roundtrip() {
    for tag in Upos::UPOS {
        let text = tag.to_string();
        let back: Upos = text.parse().expect("parse");
        assert_eq!(back, *tag, "roundtrip {text}");
    }
}

#[test]
fn upos_fromstr_is_case_insensitive() {
    assert_eq!("NOUN".parse::<Upos>().unwrap(), Upos::Noun);
    assert_eq!("Noun".parse::<Upos>().unwrap(), Upos::Noun);
    assert_eq!("noun".parse::<Upos>().unwrap(), Upos::Noun);
}

#[test]
fn upos_unknown_rejected() {
    assert!(matches!(
        "not-a-tag".parse::<Upos>(),
        Err(SpacyError::UnknownPos(_))
    ));
}

#[test]
fn upos_no_tag_renders_empty() {
    assert_eq!(Upos::NoTag.to_string(), "");
    assert_eq!("".parse::<Upos>().unwrap(), Upos::NoTag);
}

#[test]
fn upos_ids_match_spacy_symbols() {
    assert_eq!(Upos::NoTag.id(), 0);
    assert_eq!(Upos::Adj.id(), 84);
    assert_eq!(Upos::Noun.id(), 92);
    assert_eq!(Upos::X.id(), 101);
    assert_eq!(Upos::Space.id(), 103);
}

#[test]
fn ent_iob_display_and_parse() {
    assert_eq!(EntIoB::Missing.to_string(), "");
    assert_eq!(EntIoB::Inside.to_string(), "I");
    assert_eq!(EntIoB::Outside.to_string(), "O");
    assert_eq!(EntIoB::Begin.to_string(), "B");
    assert_eq!("B".parse::<EntIoB>().unwrap(), EntIoB::Begin);
    assert!(matches!(
        "L".parse::<EntIoB>(),
        Err(SpacyError::InvalidEntIobText(_))
    ));
    assert!(EntIoB::from_id(3).is_ok());
    assert!(matches!(
        EntIoB::from_id(4),
        Err(SpacyError::InvalidEntIob(4))
    ));
}

#[test]
fn ner_type_roundtrip() {
    assert_eq!("PERSON".parse::<NerType>().unwrap(), NerType::Person);
    assert_eq!("person".parse::<NerType>().unwrap(), NerType::Person);
    assert_eq!(NerType::Person.to_string(), "PERSON");
    assert_eq!(NerType::Person.id(), 380);
    assert_eq!(NerType::Cardinal.id(), 397);
}

#[test]
fn dep_rel_reference_set_roundtrips() {
    // The walkthrough's required reference labels must exist verbatim.
    for label in [
        "nsubj", "aux", "root", "prep", "pcomp", "compound", "dobj", "quantmod", "pobj",
    ] {
        let rel: DepRel = label.parse().expect("reference label");
        assert_eq!(rel.to_string(), label);
    }
    assert_eq!("nsubj".parse::<DepRel>().unwrap(), DepRel::Nsubj);
    assert_eq!(DepRel::Nsubj.id(), 429);
    assert_eq!(DepRel::Root.id(), 449);
}

#[test]
fn dep_label_set_accepts_ud_and_reference() {
    let set = DepLabelSet::ud_default();
    assert!(set.contains("nsubj"));
    assert!(set.contains("compound"));
    assert!(set.contains("case"));
    assert!(!set.contains("bogus_relation"));
}

#[test]
fn dep_label_set_contains_is_case_insensitive() {
    let set = DepLabelSet::ud_default();
    assert!(set.contains("ROOT"));
    assert!(set.contains("NSUBJ"));
}

#[test]
fn dep_label_set_serde_roundtrip() {
    let set = DepLabelSet::ud_default();
    let json = serde_json::to_string(&set).expect("serialize");
    let back: DepLabelSet = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.to_sorted_vec(), set.to_sorted_vec());
    assert!(back.contains("compound"));
}

#[test]
fn dep_label_set_fromstr_display_roundtrip() {
    let set: DepLabelSet = "nsubj,root,compound".parse().expect("parse");
    assert_eq!(set.to_sorted_vec(), vec!["compound", "nsubj", "root"]);
    assert_eq!(set.to_string(), "compound,nsubj,root");
    let again: DepLabelSet = set.to_string().parse().expect("reparse");
    assert_eq!(again.to_sorted_vec(), set.to_sorted_vec());
    assert!("nsubj,bogus".parse::<DepLabelSet>().is_err());
}
