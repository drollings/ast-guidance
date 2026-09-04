use super::*;

#[test]
fn ids_match_spacy_attrs() {
    assert_eq!(Attribute::IsAlpha.id(), 1);
    assert_eq!(Attribute::IsCurrency.id(), 18);
    assert_eq!(Attribute::Id.id(), 64);
    assert_eq!(Attribute::Orth.id(), 65);
    assert_eq!(Attribute::Lemma.id(), 73);
    assert_eq!(Attribute::Pos.id(), 74);
    assert_eq!(Attribute::Dep.id(), 76);
    assert_eq!(Attribute::EntIob.id(), 77);
    assert_eq!(Attribute::Head.id(), 79);
    assert_eq!(Attribute::SentStart.id(), 80);
    assert_eq!(Attribute::Idx.id(), 87);
}

#[test]
fn from_id_roundtrips() {
    for id in [0u16, 1, 18, 19, 63, 64, 73, 88, 100, 200] {
        let attr = Attribute::from_id(id);
        assert_eq!(attr.id(), id);
    }
}

#[test]
fn flag_boundary() {
    assert!(Attribute::IsAlpha.is_flag());
    assert!(Attribute::Other(19).is_flag());
    assert!(!Attribute::Other(64).is_flag());
    assert!(!Attribute::Orth.is_flag());
}

#[test]
fn from_name_case_insensitive() {
    assert_eq!(Attribute::from_name("orth").unwrap(), Attribute::Orth);
    assert_eq!(Attribute::from_name("ORTH").unwrap(), Attribute::Orth);
    assert_eq!(Attribute::from_name("ent_iob").unwrap(), Attribute::EntIob);
    assert!(matches!(
        Attribute::from_name("bogus"),
        Err(SpacyError::UnknownAttributeText(_))
    ));
}
