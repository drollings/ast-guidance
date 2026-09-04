use super::*;
use crate::lexeme::LexiconConfig;
use crate::vocab::Vocab;
use std::sync::Arc;

fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(LexiconConfig::default()))
}

const GOLDEN_JSON: &str = r#"[
    {"text":"The","pos":"det","tag":"DT","dep":"det","head":1,"lemma":"the"},
    {"text":"cat","pos":"noun","tag":"NN","dep":"nsubj","head":1,"lemma":"cat"},
    {"text":"sat","pos":"verb","tag":"VBD","dep":"root","head":0,"lemma":"sit"},
    {"text":".","pos":"punct","tag":".","dep":"punct","head":-1,"lemma":"."}
]"#;

fn doc_for(tokens: &[(&str, bool)]) -> Doc {
    let mut doc = Doc::new(vocab());
    for (t, s) in tokens {
        doc.push_back(t, *s).expect("push");
    }
    doc
}

#[test]
fn parse_json_matches_contract() {
    let set = AnnotationSet::parse_json(GOLDEN_JSON).expect("parse");
    assert_eq!(set.len(), 4);
    assert_eq!(set.0[0].text, "The");
    assert_eq!(set.0[0].dep, "det");
    assert_eq!(set.0[0].head, 1);
    assert_eq!(set.0[2].dep, "root");
    assert_eq!(set.0[3].head, -1);
    // optional fields default
    assert_eq!(set.0[0].ent_iob, "");
    assert_eq!(set.0[0].morph, "");
}

#[test]
fn parse_json_rejects_malformed() {
    let err =
        AnnotationSet::parse_json(r#"[{"text":"The"}]"#).expect_err("missing pos/dep/head");
    assert!(matches!(err, AnnotationError::Json { .. }));
    let err = AnnotationSet::parse_json("not json").expect_err("garbage");
    assert!(matches!(err, AnnotationError::Json { .. }));
}

#[test]
fn attach_writes_fields_and_rebuilds_tree() {
    let mut doc = doc_for(&[("The", true), ("cat", true), ("sat", true), (".", false)]);
    let set = AnnotationSet::parse_json(GOLDEN_JSON).expect("parse");
    attach(&mut doc, &set).expect("attach");
    assert_eq!(doc.token(0).pos, Upos::Det);
    assert_eq!(doc.token(1).pos, Upos::Noun);
    assert_eq!(doc.token(2).pos, Upos::Verb);
    assert_eq!(doc.token(0).dep, hash_utf8("det"));
    assert_eq!(doc.token(2).dep, hash_utf8("root"));
    assert_eq!(doc.token(0).head, 1);
    assert_eq!(doc.token(2).head, 0);
    assert_eq!(doc.token(2).lemma, hash_utf8("sit"));
    assert_eq!(doc.token(0).lemma, hash_utf8("the"));
    // tree rebuilt
    assert_eq!(doc.lefts(2), vec![1]);
    assert_eq!(doc.rights(2), vec![3]);
    assert_eq!(doc.ancestors(0), vec![1, 2]);
}

#[test]
fn attach_defaults_lemma_to_lowercase() {
    let mut doc = doc_for(&[("Apple", false)]);
    let set =
        AnnotationSet::parse_json(r#"[{"text":"Apple","pos":"noun","dep":"root","head":0}]"#)
            .expect("parse");
    attach(&mut doc, &set).expect("attach");
    assert_eq!(doc.token(0).lemma, hash_utf8("apple"));
}

#[test]
fn attach_converts_biluo_to_stored_iob() {
    let mut doc = doc_for(&[("IBM", true), ("bought", false)]);
    let set = AnnotationSet::parse_json(
        r#"[{"text":"IBM","pos":"propn","dep":"nsubj","head":1,"ent_iob":"U","ent_type":"ORG"},
            {"text":"bought","pos":"verb","dep":"root","head":0}]"#,
    )
    .expect("parse");
    attach(&mut doc, &set).expect("attach");
    assert_eq!(doc.token(0).ent_iob, EntIoB::Begin); // U → B in storage
    assert_eq!(doc.token(0).ent_type, hash_utf8("ORG"));
    assert_eq!(doc.token(1).ent_iob, EntIoB::Outside);
    assert_eq!(doc.token(1).ent_type, 0);
}

#[test]
fn attach_rejects_count_mismatch() {
    let mut doc = doc_for(&[("The", false)]);
    let set = AnnotationSet::parse_json(GOLDEN_JSON).expect("parse");
    assert!(matches!(
        attach(&mut doc, &set),
        Err(AnnotationError::Apply(_))
    ));
}

#[test]
fn apply_gates_before_attach() {
    let mut doc = doc_for(&[("The", true), ("cat", true), ("sat", true), (".", false)]);
    let mut set = AnnotationSet::parse_json(GOLDEN_JSON).expect("parse");
    set.0[0].dep = "bogus".into(); // fails check 2
    assert!(matches!(
        apply(&mut doc, &set),
        Err(AnnotationError::UnknownDep(_))
    ));
    // nothing was applied (pos untouched)
    assert_eq!(doc.token(0).pos, Upos::NoTag);
    assert_eq!(doc.token(0).dep, 0);
}

#[test]
fn contract_is_derived_from_closed_vocabularies() {
    let contract = AnnotationRecord::contract();
    let pos_enum = contract["items"]["properties"]["pos"]["enum"]
        .as_array()
        .unwrap();
    let pos: Vec<String> = Upos::UPOS.iter().map(ToString::to_string).collect();
    assert_eq!(
        pos_enum
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        pos
    );
    let dep_enum = contract["items"]["properties"]["dep"]["enum"]
        .as_array()
        .unwrap();
    assert!(dep_enum.iter().any(|v| v == "nsubj"));
    assert!(dep_enum.iter().any(|v| v == "root"));
    assert!(dep_enum.iter().any(|v| v == "compound"));
}

#[test]
fn is_confidence_bearing_classifies_every_variant() {
    assert!(!AnnotationSource::Llm.is_confidence_bearing());
    assert!(!AnnotationSource::RuleRung.is_confidence_bearing());
    assert!(!AnnotationSource::HumanReview.is_confidence_bearing());
    assert!(AnnotationSource::ArcEager.is_confidence_bearing());
    assert!(AnnotationSource::Encoder.is_confidence_bearing());
}

#[test]
fn is_confidence_bearing_equals_arc_eager_or_encoder() {
    for source in [
        AnnotationSource::Llm,
        AnnotationSource::ArcEager,
        AnnotationSource::RuleRung,
        AnnotationSource::HumanReview,
        AnnotationSource::Encoder,
        AnnotationSource::Frontier,
    ] {
        assert_eq!(
            source.is_confidence_bearing(),
            matches!(source, AnnotationSource::ArcEager | AnnotationSource::Encoder),
            "{source:?}"
        );
    }
}

#[test]
fn tier_maps_every_source_exhaustively() {
    use fluent_types::ProvenanceTier;
    for source in [
        AnnotationSource::Llm,
        AnnotationSource::ArcEager,
        AnnotationSource::RuleRung,
        AnnotationSource::HumanReview,
        AnnotationSource::Encoder,
        AnnotationSource::Frontier,
    ] {
        let _: ProvenanceTier = source.tier();
    }
    assert_eq!(AnnotationSource::ArcEager.tier(), ProvenanceTier::Deterministic);
    assert_eq!(AnnotationSource::RuleRung.tier(), ProvenanceTier::Deterministic);
    assert_eq!(AnnotationSource::Llm.tier(), ProvenanceTier::LocalModel);
    assert_eq!(AnnotationSource::Encoder.tier(), ProvenanceTier::LocalModel);
    assert_eq!(AnnotationSource::Frontier.tier(), ProvenanceTier::Frontier);
    assert_eq!(AnnotationSource::HumanReview.tier(), ProvenanceTier::HumanReview);
    // Authority ordering: deterministic frames are provisional until a
    // higher tier confirms; human review beats every local producer.
    assert!(AnnotationSource::ArcEager.tier() < AnnotationSource::Llm.tier());
    assert!(AnnotationSource::Llm.tier() < AnnotationSource::HumanReview.tier());
}

#[test]
fn refine_prompt_marks_focus_and_shows_the_base() {
    let set = AnnotationSet::parse_json(GOLDEN_JSON).expect("parse");
    let base = AnnotationResult::new(set, AnnotationSource::ArcEager);
    let tokens: Vec<String> = base
        .records()
        .records()
        .iter()
        .map(|r| r.text.clone())
        .collect();
    let prompt = LlmRefinePrompt::prompt(&tokens, &base, &[1]);
    assert!(prompt.contains("Reconsider ONLY"), "focus discipline instruction");
    assert!(prompt.contains("1: cat / noun / nsubj / 1 / cat FOCUS"));
    assert!(
        !prompt.contains("0: The / det / det / 1 / the FOCUS"),
        "non-focus tokens carry no FOCUS mark"
    );
    assert!(prompt.contains("Focus token indices: [1]"));
}

#[test]
fn refine_contract_lists_the_closed_fields() {
    let contract = LlmRefinePrompt::contract();
    let field = contract["properties"]["corrections"]["items"]["properties"]["field"]["enum"]
        .as_array()
        .unwrap();
    let names: Vec<&str> = field.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, vec!["dep", "head", "lemma", "pos"]);
    assert_eq!(
        contract["properties"]["corrections"]["items"]["required"]
            .as_array()
            .unwrap()
            .len(),
        3,
        "token_index / field / new_value — no old_value on the refine wire"
    );
}
