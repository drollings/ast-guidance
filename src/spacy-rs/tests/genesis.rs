use super::*;
use crate::review::CorrectionField;

fn corr(pos: &str) -> Correction {
    Correction {
        token_index: 0,
        field: CorrectionField::Pos,
        old_value: String::new(),
        new_value: pos.into(),
    }
}

#[test]
fn promotes_after_threshold() {
    let g = InMemoryGenesisIndex::with_threshold(3);
    assert!(g.get_pos("hello").is_none());
    g.record(&corr("verb"), "hello");
    assert!(!g.is_promoted("hello"));
    g.record(&corr("verb"), "hello");
    assert!(!g.is_promoted("hello"));
    g.record(&corr("verb"), "hello");
    assert!(g.is_promoted("hello"));
    assert_eq!(g.get_pos("hello"), Some(Upos::Verb));
    assert_eq!(g.count_for("hello"), 3);
}

#[test]
fn first_pos_wins() {
    let g = InMemoryGenesisIndex::with_threshold(2);
    g.record(&corr("noun"), "hello");
    g.record(&corr("verb"), "hello");
    assert_eq!(g.get_pos("hello"), Some(Upos::Noun));
}

#[test]
fn save_and_load_roundtrip() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("genesis.json");
    let g = InMemoryGenesisIndex::with_threshold(1);
    g.record(&corr("verb"), "run");
    g.save(&path).expect("save");
    let h = InMemoryGenesisIndex::load_or_empty(&path);
    assert_eq!(h.get_pos("run"), Some(Upos::Verb));
}

#[test]
fn save_and_load_roundtrip_preserves_ner() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("genesis.json");
    let g = InMemoryGenesisIndex::with_threshold(1);
    let ner = Correction {
        token_index: 0,
        field: CorrectionField::Ner,
        old_value: String::new(),
        new_value: "Person".into(),
    };
    g.record(&ner, "london");
    g.save(&path).expect("save");
    let h = InMemoryGenesisIndex::load_or_empty(&path);
    assert_eq!(h.get_ner("london"), Some(NerType::Person));
}

#[test]
fn non_pos_correction_ignored() {
    let g = InMemoryGenesisIndex::new();
    for field in [CorrectionField::Dep, CorrectionField::Head, CorrectionField::Lemma] {
        let c = Correction {
            token_index: 0,
            field,
            old_value: String::new(),
            new_value: "cat".into(),
        };
        g.record(&c, "cat");
    }
    assert_eq!(g.len(), 0);
}

fn ner_corr(ner: &str) -> Correction {
    Correction {
        token_index: 0,
        field: CorrectionField::Ner,
        old_value: String::new(),
        new_value: ner.into(),
    }
}

#[test]
fn genesis_ner_promotes_after_threshold_and_overrides_rule() {
    let g = std::sync::Arc::new(InMemoryGenesisIndex::with_threshold(3));
    for _ in 0..3 {
        g.record(&ner_corr("Person"), "london");
    }
    assert_eq!(g.get_ner("london"), Some(NerType::Person));
    // RuleAnnotator consults genesis.get_ner for ent_type
    let annotator = crate::pipeline::RuleAnnotator::en_default()
        .with_genesis(g.clone() as std::sync::Arc<dyn crate::genesis::GenesisIndex>);
    let vocab = std::sync::Arc::new(crate::vocab::Vocab::new(crate::lexeme::LexiconConfig::default()));
    let mut doc = crate::doc::Doc::new(vocab);
    doc.push_back("london", true).expect("push");
    let set = annotator.annotate(&doc);
    assert_eq!(set.0[0].ent_type, "PERSON");
}

#[test]
fn ner_correction_ignored_before_threshold() {
    let g = InMemoryGenesisIndex::with_threshold(3);
    g.record(&ner_corr("Person"), "london");
    g.record(&ner_corr("Person"), "london");
    assert!(g.get_ner("london").is_none());
    assert!(!g.is_promoted("london"));
}

#[test]
fn first_ner_wins() {
    let g = InMemoryGenesisIndex::with_threshold(2);
    g.record(&ner_corr("Person"), "hello");
    g.record(&ner_corr("Loc"), "hello");
    assert_eq!(g.get_ner("hello"), Some(NerType::Person));
}

#[test]
fn ner_threshold_is_higher_than_pos_and_isolated() {
    // Production defaults: POS 3, NER 5. POS evidence must not accelerate
    // NER promotion — entity type is context-variant (Washington/Jordan).
    let g = InMemoryGenesisIndex::new();
    // 3 POS evidence promotes POS
    for _ in 0..3 {
        g.record(&corr("noun"), "washington");
    }
    assert_eq!(g.get_pos("washington"), Some(Upos::Noun));
    // Same orth: NER still not promoted — shared count would have falsely
    // promoted it if the counter were shared.
    assert!(g.get_ner("washington").is_none());
    // 4 NER corrections still below NER bar (5)
    for _ in 0..4 {
        g.record(&ner_corr("Loc"), "washington");
    }
    assert!(g.get_ner("washington").is_none());
    assert_eq!(g.ner_count_for("washington"), 4);
    // 5th NER promotes NER (POS win intact)
    g.record(&ner_corr("Loc"), "washington");
    assert_eq!(g.get_ner("washington"), Some(NerType::Loc));
    assert_eq!(g.get_pos("washington"), Some(Upos::Noun));
}

#[test]
fn adversarial_same_orth_two_entity_types_first_wins() {
    // Same surface string ("Washington") corrected to two different entity
    // types across contexts — first NER value must win (monotonic), never
    // flip to the second.
    let g = InMemoryGenesisIndex::with_threshold(2);
    g.record(&ner_corr("Person"), "washington");
    g.record(&ner_corr("Loc"), "washington");
    // Even after more evidence for the second type, the first stays.
    g.record(&ner_corr("Loc"), "washington");
    g.record(&ner_corr("Loc"), "washington");
    assert_eq!(g.get_ner("washington"), Some(NerType::Person));
}

#[test]
fn pos_evidence_does_not_count_toward_ner_promotion() {
    let g = InMemoryGenesisIndex::with_thresholds(3, 5);
    for _ in 0..10 {
        g.record(&corr("noun"), "paris");
    }
    // 10 POS corrections but zero NER corrections → NER still not promoted
    assert!(g.get_ner("paris").is_none());
    assert_eq!(g.ner_count_for("paris"), 0);
    assert_eq!(g.count_for("paris"), 10);
}

#[test]
fn old_file_migration_preserves_ner_promotion() {
    // Old JSON (pre-split counters): {count:3, promoted:true, ner:"Person"}
    // with ner_count/ner_promoted missing must still load as NER-promoted.
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("genesis.json");
    let raw = r#"{"washington":{"pos":null,"ner":"Person","count":3,"promoted":true}}"#;
    std::fs::write(&path, raw).expect("write old format");
    let g = InMemoryGenesisIndex::load_or_empty(&path);
    assert_eq!(g.get_ner("washington"), Some(NerType::Person));
    assert!(g.is_promoted("washington"));
}
