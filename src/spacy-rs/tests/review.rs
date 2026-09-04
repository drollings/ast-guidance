use super::*;
use crate::llm::{AnnotationRecord, AnnotationSet};
use fluent_types::InterlinguaNamespace;

fn parse() -> AnnotationResult {
    let set = AnnotationSet(vec![
        AnnotationRecord {
            text: "The".into(),
            pos: "det".into(),
            tag: String::new(),
            dep: "det".into(),
            head: 1,
            lemma: "the".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
        AnnotationRecord {
            text: "cat".into(),
            pos: "noun".into(),
            tag: String::new(),
            dep: "dep".into(),
            head: 1,
            lemma: "cat".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
    ]);
    AnnotationResult::new(set, AnnotationSource::ArcEager)
}

#[test]
fn review_types_serde_roundtrip() {
    let review = ParseReview {
        corrections: vec![Correction {
            token_index: 1,
            field: CorrectionField::Dep,
            old_value: "dep".into(),
            new_value: "nsubj".into(),
        }],
        linked_entities: vec![],
        note: None,
    };
    let json = serde_json::to_string(&review).unwrap();
    let back: ParseReview = serde_json::from_str(&json).unwrap();
    assert_eq!(back, review);

    let status = ReviewStatus::Reviewed {
        review: review.clone(),
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: ReviewStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back, status);
}

#[test]
fn apply_corrections_modifies_target_tokens() {
    let p = parse();
    let corrected = apply_corrections(
        &p,
        &[Correction {
            token_index: 1,
            field: CorrectionField::Dep,
            old_value: "dep".into(),
            new_value: "nsubj".into(),
        }],
    );
    assert_eq!(corrected.source(), AnnotationSource::HumanReview);
    assert_eq!(corrected.records().records()[1].dep, "nsubj");
    assert_eq!(corrected.records().records()[0].dep, "det");
    // Empty corrections preserve the original source.
    let same = apply_corrections(&p, &[]);
    assert_eq!(same.source(), AnnotationSource::ArcEager);
    // Out-of-range token index is ignored.
    let guarded = apply_corrections(
        &p,
        &[Correction {
            token_index: 99,
            field: CorrectionField::Lemma,
            old_value: String::new(),
            new_value: "x".into(),
        }],
    );
    assert_eq!(guarded.records().len(), p.records().len());
}

#[test]
fn review_prompt_lists_concepts_and_parse() {
    let p = parse();
    let concepts = vec![ConceptMetadata {
        id: InterlinguaId::from_u64(1),
        canonical_name: "schema:Cat".into(),
        namespace: InterlinguaNamespace::YagoClass,
        yago_iri: Some("iri".into()),
        yago_class_iri: None,
        label: Some("cat".into()),
        node_id: None,
        parent_class_id: None,
    }];
    let prompt = review_prompt("The cat", &p, &concepts);
    assert!(prompt.contains("Sentence: The cat"));
    assert!(prompt.contains("schema:Cat"));
    assert!(prompt.contains("iri"));
    assert!(prompt.contains("linked_entities"));
}

#[test]
fn review_prompt_handles_empty_concepts() {
    let p = parse();
    let prompt = review_prompt("The cat", &p, &[]);
    assert!(prompt.contains("(none registered)"));
}

#[test]
fn apply_edits_counts_landed_edits() {
    let mut records = parse().records().records().to_vec();
    let edits = [
        Correction {
            token_index: 1,
            field: CorrectionField::Lemma,
            old_value: String::new(),
            new_value: "cats".into(),
        },
        Correction {
            // out-of-range: skipped, not counted
            token_index: 99,
            field: CorrectionField::Lemma,
            old_value: String::new(),
            new_value: "x".into(),
        },
        Correction {
            // unparseable head: rejected, not counted
            token_index: 0,
            field: CorrectionField::Head,
            old_value: String::new(),
            new_value: "not-a-number".into(),
        },
    ];
    assert_eq!(apply_edits(&mut records, &edits), 1);
    assert_eq!(records[1].lemma, "cats");
    assert_eq!(records[0].head, 1, "bad head left the record untouched");
}

#[test]
fn refine_reply_parses_without_old_value_or_entities() {
    // The span-scoped refine contract omits `old_value` and
    // `linked_entities` — the same `Correction`/`ParseReview` types must
    // deserialize it (M2.2 DRY amendment vocabulary).
    let review =
        ParseReview::parse_json(r#"{"corrections":[{"token_index":2,"field":"dep","new_value":"nsubj"}]}"#)
            .expect("parse");
    assert_eq!(review.corrections.len(), 1);
    assert_eq!(review.corrections[0].field, CorrectionField::Dep);
    assert_eq!(review.corrections[0].old_value, "");
    assert!(review.linked_entities.is_empty());
    // An empty reply still parses (the refiner treats it as "nothing to change").
    ParseReview::parse_json("{}").expect("empty parse");
}

#[test]
fn apply_edits_stale_old_value_skipped_and_warned() {
    let mut records = parse().records().records().to_vec();
    // records[1].lemma == "cat", but old_value says "dog" → stale, skipped
    let edits = [Correction {
        token_index: 1,
        field: CorrectionField::Lemma,
        old_value: "dog".into(),
        new_value: "cats".into(),
    }];
    assert_eq!(apply_edits(&mut records, &edits), 0);
    assert_eq!(records[1].lemma, "cat", "stale edit must not apply");
}

#[test]
fn apply_edits_empty_old_value_applies() {
    let mut records = parse().records().records().to_vec();
    let edits = [Correction {
        token_index: 1,
        field: CorrectionField::Lemma,
        old_value: String::new(),
        new_value: "cats".into(),
    }];
    assert_eq!(apply_edits(&mut records, &edits), 1);
    assert_eq!(records[1].lemma, "cats");
}

#[test]
fn parse_review_json_error_preserves_span() {
    let err = ParseReview::parse_json(r#"{"corrections": "not_an_array"}"#).expect_err("must fail");
    let display = format!("{err}");
    // serde_json::Error Display contains line/column for structural mismatches
    assert!(
        display.contains("line") || display.contains("column"),
        "error should preserve span, got: {display}"
    );
    // source chain also contains it
    let src = std::error::Error::source(&err);
    assert!(src.is_some(), "Json error must have source");
    let src_display = format!("{}", src.unwrap());
    assert!(
        src_display.contains("line") || src_display.contains("column"),
        "source should contain line/column, got: {src_display}"
    );
}
