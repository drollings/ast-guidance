use super::*;

fn strings() -> Arc<StringStore> {
    Arc::new(StringStore::new())
}

fn morphology() -> Morphology {
    Morphology::new(strings())
}

#[test]
fn feats_to_dict_roundtrip() {
    let dict = Morphology::feats_to_dict("Case=Nom,Acc|Number=Sing");
    assert_eq!(dict.get("Case").map(String::as_str), Some("Acc,Nom"));
    assert_eq!(dict.get("Number").map(String::as_str), Some("Sing"));
    // empty / placeholder
    assert!(Morphology::feats_to_dict("").is_empty());
    assert!(Morphology::feats_to_dict("_").is_empty());
}

#[test]
fn dict_to_feats_sorts_fields_and_values() {
    let dict = HashMap::from([
        ("Number".to_string(), "Plur".to_string()),
        ("Case".to_string(), "Nom,Acc".to_string()),
    ]);
    assert_eq!(Morphology::dict_to_feats(&dict), "Case=Acc,Nom|Number=Plur");
    assert_eq!(Morphology::dict_to_feats(&HashMap::new()), "_");
}

#[test]
fn normalize_features_canonicalizes() {
    let m = morphology();
    assert_eq!(m.normalize_features("Number=Sing|Case=Nom"), "Case=Nom|Number=Sing");
    assert_eq!(m.normalize_features("Case=Nom,Acc|Number=Sing"), "Case=Acc,Nom|Number=Sing");
    // POS canonicalized to the UPOS name
    assert_eq!(m.normalize_features("POS=noun|Number=Sing"), "Number=Sing|POS=NOUN");
    // empty and placeholder both normalize to the placeholder
    assert_eq!(m.normalize_features(""), "_");
    assert_eq!(m.normalize_features("_"), "_");
}

#[test]
fn add_is_idempotent_and_content_addressed() {
    let m = morphology();
    let a = m.add("Case=Nom|Number=Sing");
    let b = m.add("Number=Sing|Case=Nom");
    assert_eq!(a, b);
    assert_eq!(m.get(a).as_deref(), Some("Case=Nom|Number=Sing"));
    // a different analysis gets a different key
    let c = m.add("Case=Acc|Number=Sing");
    assert_ne!(a, c);
}

#[test]
fn empty_analysis_key() {
    let m = morphology();
    assert_eq!(m.add(""), m.empty_key());
    assert_eq!(m.add("_"), m.empty_key());
}

#[test]
fn resolution_helpers() {
    let m = morphology();
    let key = m.add("Number=Sing|Gender=Fem|VerbForm=Inf");
    assert!(m.has_feature(key, "Number=Sing"));
    assert!(m.has_feature(key, "VerbForm=Inf"));
    assert!(!m.has_feature(key, "Number=Plur"));
    assert_eq!(m.get_by_field(key, "Number"), vec!["Sing"]);
    assert_eq!(m.get_by_field(key, "Tense"), Vec::<String>::new());
    assert_eq!(m.get_by_field(u64::MAX, "Number"), Vec::<String>::new());
    let dict = m.to_dict(key).expect("to_dict");
    assert_eq!(dict.get("Gender").map(String::as_str), Some("Fem"));
}

#[test]
fn normalize_attrs_pos_and_multi_values() {
    let dict = HashMap::from([
        ("POS".to_string(), "noun".to_string()),
        ("Case".to_string(), "Nom,Acc".to_string()),
    ]);
    let out = Morphology::normalize_attrs(&dict);
    assert_eq!(out.get("POS").map(String::as_str), Some("NOUN"));
    assert_eq!(out.get("Case").map(String::as_str), Some("Acc,Nom"));
}
