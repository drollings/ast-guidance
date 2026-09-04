//! Seam round-trip for `NodeAnnotation` (moved from
//! `spacy-rs::arcready::ArcReadyAnnotation`, renamed): spacy-rs produces the
//! inputs (attached `Doc` + ladder `AnnotationResult` + routing signals), the
//! ledger's `NodeAnnotation` consumes them. No behavior change for existing
//! callers — every assertion below is ported verbatim from the pre-move
//! suites (`spacy-rs/tests/arcready.rs`, `spacy-rs/tests/pipeline.rs`
//! `arc_ready_*`), with only the type path and constructor updated.

use super::*;
use std::sync::Arc;

use spacy_rs::{
    extract_routing_signals, hash_utf8, AnnotationResult, AnnotationSet, AnnotationSource,
    Doc, LexiconConfig, LlmFetchSync, NlpPipeline, ParseConfidence, RefineMode, RefinePolicy,
    Sentencizer, Vocab,
};
use spacy_rs::llm::attach;

fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(LexiconConfig::default()))
}

fn doc_for(tokens: &[&str]) -> Doc {
    let mut doc = Doc::new(vocab());
    for t in tokens {
        doc.push_back(t, true).expect("push");
    }
    doc
}

/// Attach a full UD parse and re-run the sentencizer so the doc carries
/// sentence boundaries (the state `process_sync` produces).
fn attached(text_json: &str, tokens: &[&str]) -> Doc {
    let mut doc = doc_for(tokens);
    let set = AnnotationSet::parse_json(text_json).expect("parse json");
    attach(&mut doc, &set).expect("attach");
    Sentencizer::new().process(&mut doc);
    doc
}

const FULL_PARSE: &str = r#"[
    {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
    {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
    {"text":"the","pos":"det","dep":"det","head":2,"lemma":"the"},
    {"text":"sales","pos":"noun","dep":"compound","head":1,"lemma":"sales"},
    {"text":"report","pos":"noun","dep":"dobj","head":-4,"lemma":"report"},
    {"text":"for","pos":"adp","dep":"prep","head":-5,"lemma":"for"},
    {"text":"yesterday","pos":"noun","dep":"pobj","head":-1,"lemma":"yesterday"},
    {"text":"please","pos":"adv","dep":"advmod","head":-7,"lemma":"please"}
]"#;

fn annotation_for(doc: &Doc) -> NodeAnnotation {
    let result = AnnotationResult::new(
        AnnotationSet::parse_json(FULL_PARSE).expect("parse"),
        AnnotationSource::Llm,
    );
    let signals = extract_routing_signals(doc);
    NodeAnnotation::from_doc(doc, &result, signals)
}

#[test]
fn materializes_from_doc_result_and_signals() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let ann = annotation_for(&doc);

    // Records come from the ladder result (validated wire output).
    assert_eq!(ann.records.len(), 8);
    assert_eq!(ann.records.records()[0].text, "Show");
    assert_eq!(ann.source, AnnotationSource::Llm);
    assert!(ann.token_confidence.is_none(), "LLM rung carries no confidence");
    assert!(ann.parse_confidence.is_none());
    assert_eq!(ann.collision_count, 0);

    // Signals are per-sentence routing frames.
    assert_eq!(ann.signals.len(), 1);
    assert_eq!(ann.signals[0].predicate, "show");
    assert_eq!(ann.signals[0].direct_object.as_deref(), Some("report"));

    // The tokens detail baseline is the tokenizer's exact token array.
    assert_eq!(ann.tokens.len(), 8);
    assert_eq!(ann.tokens[0].idx, 0);
    assert_eq!(ann.tokens[1].idx, 5, "Show (4) + spacy (1)");
    assert_eq!(ann.tokens[4].lemma, hash_utf8("report"));
}

#[test]
fn carries_parse_confidence_from_result() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let mut doc = attached(FULL_PARSE, &tokens);
    // Stamp per-token confidence so the extracted signals carry it too.
    for i in 0..doc.len() {
        doc.token_mut(i).confidence = Some(0.8);
    }
    let result = AnnotationResult::new(
        AnnotationSet::parse_json(FULL_PARSE).expect("parse"),
        AnnotationSource::ArcEager,
    )
    .with_confidence(
        Some(vec![0.8; 8]),
        Some(ParseConfidence {
            overall: 0.8,
            token_scores: vec![0.8; 8],
            role_coverage: 0.9,
            oracle_tie_count: 0,
            oracle_margins: Vec::new(),
            semantic_plausibility: None,
        }),
    );
    let signals = extract_routing_signals(&doc);
    let ann = NodeAnnotation::from_doc(&doc, &result, signals);

    assert_eq!(ann.source, AnnotationSource::ArcEager);
    assert_eq!(ann.token_confidence.as_deref(), Some(&[0.8; 8][..]));
    let parse = ann.parse_confidence.as_ref().expect("parse confidence");
    assert_eq!(parse.overall, 0.8);
    assert_eq!(parse.role_coverage, 0.9);
    // The per-sentence signal confidence derives from the token confidence
    // (mean of 8 × 0.8 ≈ 0.8 within floating-point tolerance).
    let sig_conf = ann.signals[0].interlingua.as_ref().unwrap().confidence;
    assert!(
        sig_conf.is_some_and(|c| (c - 0.8).abs() < 1e-9),
        "signal confidence is ~0.8, got {sig_conf:?}"
    );
}

#[test]
fn carries_collision_count_from_result() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let mut result = AnnotationResult::new(
        AnnotationSet::parse_json(FULL_PARSE).expect("parse"),
        AnnotationSource::Llm,
    );
    result.collision_count = 3;
    let signals = extract_routing_signals(&doc);
    let ann = NodeAnnotation::from_doc(&doc, &result, signals);
    assert_eq!(ann.collision_count, 3);
}

/// Immutability: every field is owned immutable data. The struct contains
/// no interior mutability, so cloning (the `Arc` share path) is a plain
/// value copy and reads never require a lock. This test locks in the
/// absence of `Mutex`/`RefCell`/atomics by construction (the type is plain
/// data) and asserts the shared `Arc` read path yields an owned value.
#[test]
fn immutable_plain_data_shared_via_arc() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let ann = annotation_for(&doc);
    let shared: Arc<NodeAnnotation> = Arc::new(ann);
    // Two independent reads of the same Arc see identical, owned data.
    let a: NodeAnnotation = Arc::clone(&shared).as_ref().clone();
    let b: NodeAnnotation = Arc::clone(&shared).as_ref().clone();
    assert_eq!(a.records, b.records);
    assert_eq!(a.source, b.source);
    assert_eq!(a.signals, b.signals);
    assert_eq!(a.tokens.len(), b.tokens.len());
    assert_eq!(a.primary_signal(), b.primary_signal());
}

#[test]
fn primary_signal_single_sentence_is_the_signal() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let ann = annotation_for(&doc);
    let primary = ann.primary_signal().expect("one signal");
    assert_eq!(primary.predicate, "show");
    // The primary signal is exactly the whole (single-sentence) text.
    assert_eq!(primary.sentence, "Show me the sales report for yesterday please");
}

#[test]
fn primary_signal_multi_sentence_picks_most_confident() {
    // Two sentences: "Show me the report." (conf 0.4) and "I need it now."
    // (conf 0.9). The primary signal must be the more-confident second.
    let json = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"report","pos":"noun","dep":"dobj","head":-3,"lemma":"report"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."},
        {"text":"I","pos":"pron","dep":"nsubj","head":1,"lemma":"I"},
        {"text":"need","pos":"verb","dep":"root","head":0,"lemma":"need"},
        {"text":"it","pos":"pron","dep":"dobj","head":-1,"lemma":"it"},
        {"text":"now","pos":"adv","dep":"advmod","head":-2,"lemma":"now"},
        {"text":".","pos":"punct","dep":"punct","head":-4,"lemma":"."}
    ]"#;
    let tokens = ["Show", "me", "the", "report", ".", "I", "need", "it", "now", "."];
    let mut doc = attached(json, &tokens);
    // Stamp per-token confidence: sentence 1 (tokens 0..5) conf 0.4, sentence 2 (5..10) conf 0.9.
    for i in 0..doc.len() {
        doc.token_mut(i).confidence = Some(if i < 5 { 0.4 } else { 0.9 });
    }
    let result = AnnotationResult::new(
        AnnotationSet::parse_json(json).expect("parse"),
        AnnotationSource::ArcEager,
    )
    .with_confidence(Some(vec![0.4; 5].into_iter().chain(vec![0.9; 5]).collect()), None);
    let signals = extract_routing_signals(&doc);
    let ann = NodeAnnotation::from_doc(&doc, &result, signals);

    assert_eq!(ann.signals.len(), 2);
    let primary = ann.primary_signal().expect("two signals");
    assert_eq!(primary.predicate, "need", "most-confident sentence wins");
    assert_eq!(primary.sentence, "I need it now .");
}

#[test]
fn primary_signal_tie_breaks_to_first() {
    // Two sentences with equal confidence → the earliest is primary.
    let json = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"report","pos":"noun","dep":"dobj","head":-3,"lemma":"report"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."},
        {"text":"Now","pos":"adv","dep":"advmod","head":1,"lemma":"now"},
        {"text":"run","pos":"verb","dep":"root","head":0,"lemma":"run"},
        {"text":"it","pos":"pron","dep":"dobj","head":-1,"lemma":"it"},
        {"text":".","pos":"punct","dep":"punct","head":-2,"lemma":"."}
    ]"#;
    let tokens = ["Show", "me", "the", "report", ".", "Now", "run", "it", "."];
    let mut doc = attached(json, &tokens);
    for i in 0..doc.len() {
        doc.token_mut(i).confidence = Some(0.5);
    }
    let result = AnnotationResult::new(
        AnnotationSet::parse_json(json).expect("parse"),
        AnnotationSource::ArcEager,
    );
    let signals = extract_routing_signals(&doc);
    let ann = NodeAnnotation::from_doc(&doc, &result, signals);
    assert_eq!(ann.signals.len(), 2);
    let primary = ann.primary_signal().expect("two signals");
    assert_eq!(primary.predicate, "show", "equal confidence keeps the first");
}

#[test]
fn primary_signal_none_when_no_signals() {
    // A genuinely empty doc has no tokens and no signals.
    let doc = Doc::new(vocab());
    let result = AnnotationResult::new(AnnotationSet::default(), AnnotationSource::RuleRung);
    let ann = NodeAnnotation::from_doc(&doc, &result, Vec::new());
    assert!(ann.tokens.is_empty());
    assert!(ann.signals.is_empty());
    assert!(ann.primary_signal().is_none());
}

#[test]
fn primary_signal_falls_back_to_first_when_no_confidence() {
    // No per-token confidence (LLM rung) → primary is the first signal.
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let ann = annotation_for(&doc);
    let primary = ann.primary_signal().expect("one signal");
    assert_eq!(primary.predicate, "show");
    assert!(ann.signals[0].interlingua.as_ref().unwrap().confidence.is_none());
}

// ── Ladder round-trip: spacy-rs produces the inputs, the ledger consumes ──

fn en_pipeline() -> NlpPipeline {
    NlpPipeline::en_default().expect("en pipeline")
}

fn stub_fetch_sync(reply: String) -> LlmFetchSync {
    Arc::new(move |_tokens: Vec<String>| Ok(reply.clone()))
}

fn llm_first_policy() -> RefinePolicy {
    RefinePolicy {
        mode: RefineMode::Always,
        ..RefinePolicy::default()
    }
}

#[test]
fn node_annotation_materializes_from_sync_ladder_run() {
    let pipeline = en_pipeline();
    let (doc, result) = pipeline
        .process_sync_with_confidence("Show me the sales report", None, None, RefinePolicy::default())
        .expect("sync");
    let ann = node_annotation(&doc, &result);

    // The annotation is the validated ladder output, not the working doc.
    assert_eq!(ann.records, result.records().clone());
    assert_eq!(ann.source, result.source());
    assert_eq!(ann.collision_count, 0);
    // Signals derive from the sentencized doc (the constructor wires extraction).
    assert_eq!(ann.signals.len(), 1);
    // The predicate is the root verb's lemma as surfaced by the ArcEager rung
    // (the deterministic lemmatizer does not lowercase a capitalized verb).
    assert_eq!(ann.signals[0].predicate, "Show");
    // The primary signal is the whole (single-sentence) text.
    let primary = ann.primary_signal().expect("one signal");
    assert_eq!(primary.predicate, "Show");
    assert_eq!(primary.sentence, "Show me the sales report");
    // The token baseline is the tokenizer's exact array (detail baseline).
    assert_eq!(ann.tokens.len(), doc.len());
    assert_eq!(ann.tokens[1].idx, 5, "Show (4) + spacy (1)");
}

#[test]
fn node_annotation_materializes_from_llm_rung_with_real_deps() {
    let pipeline = en_pipeline();
    let full = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":2,"lemma":"the"},
        {"text":"sales","pos":"noun","dep":"compound","head":1,"lemma":"sales"},
        {"text":"report","pos":"noun","dep":"dobj","head":-3,"lemma":"report"}
    ]"#;
    let (doc, result) = pipeline
        .process_sync_with_confidence(
            "Show me the sales report",
            Some(&stub_fetch_sync(full.into())),
            None,
            llm_first_policy(),
        )
        .expect("sync");
    let ann = node_annotation(&doc, &result);
    assert_eq!(ann.source, AnnotationSource::Llm);
    assert_eq!(ann.records.records()[4].dep, "dobj");
    assert_eq!(ann.records.records()[4].lemma, "report");
    assert_eq!(ann.signals[0].direct_object.as_deref(), Some("report"));
    // The immutable document shares cleanly behind an `Arc`.
    let shared: Arc<NodeAnnotation> = Arc::new(ann);
    assert_eq!(shared.primary_signal().unwrap().predicate, "show");
}

/// The ledger overlay contract: a `NodeAnnotation` rides the opaque
/// `ContentNode.annotation` slot and reads back typed via `annotation_as`
/// (the same slot `ContentNodeStore::annotation_for` installs into).
#[test]
fn node_annotation_rides_the_content_node_overlay_slot() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let ann = annotation_for(&doc);
    let node = fluent_types::ContentNode {
        annotation: Some(Arc::new(ann)),
        ..fluent_types::ContentNode::default()
    };
    let got = node.annotation_as::<NodeAnnotation>().expect("downcast");
    assert_eq!(got.signals.len(), 1);
    assert_eq!(got.primary_signal().expect("signal").predicate, "show");
    assert_eq!(got.tokens.len(), 8);
}
