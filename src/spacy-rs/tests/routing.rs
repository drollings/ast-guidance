use super::*;
use std::sync::Arc;

use crate::doc::Doc;
use crate::lexeme::LexiconConfig;
use crate::llm::{attach, AnnotationSet};
use crate::sentencizer::Sentencizer;
use crate::vocab::Vocab;

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

#[test]
fn full_ud_parse_fills_every_role() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let signals = extract_routing_signals(&doc);
    assert_eq!(signals.len(), 1);
    let s = &signals[0];
    assert_eq!(s.predicate, "show");
    assert_eq!(s.direct_object.as_deref(), Some("report"));
    assert_eq!(s.indirect_object.as_deref(), Some("me"));
    assert_eq!(s.modifiers, vec!["sales"]);
    assert_eq!(s.arguments, vec![("for".to_string(), "yesterday".to_string())]);
    assert_eq!(s.qualifiers, vec!["please"]);
    assert!(s.subject.is_none());
    // `det` is not a role label — the determiner is a residual dependent.
    assert_eq!(s.dependents, vec!["the"]);
}

#[test]
fn transcript_matches_tokens_in_order() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let s = &extract_routing_signals(&doc)[0];
    assert_eq!(s.tokens, tokens);
    assert_eq!(s.pos[0], "verb");
    assert_eq!(s.pos[3], "noun");
    assert_eq!(s.deps[0], "root");
    assert_eq!(s.deps[4], "dobj");
    assert_eq!(s.deps[5], "prep");
    assert_eq!(s.heads[5], -5, "for is 5 before its root at index 0");
    assert_eq!(s.lemmas[4], "report");
}

#[test]
fn subject_and_qualifiers_on_intransitive() {
    let json = r#"[
        {"text":"the","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"dog","pos":"noun","dep":"nsubj","head":3,"lemma":"dog"},
        {"text":"will","pos":"aux","dep":"aux","head":2,"lemma":"will"},
        {"text":"not","pos":"adv","dep":"neg","head":1,"lemma":"not"},
        {"text":"sleep","pos":"verb","dep":"root","head":0,"lemma":"sleep"}
    ]"#;
    let doc = attached(json, &["the", "dog", "will", "not", "sleep"]);
    let s = &extract_routing_signals(&doc)[0];
    assert_eq!(s.predicate, "sleep");
    assert_eq!(s.subject.as_deref(), Some("dog"));
    assert_eq!(s.qualifiers, vec!["will", "not"]);
    assert!(s.direct_object.is_none());
    assert!(s.arguments.is_empty());
}

#[test]
fn star_parse_degrades_to_predicate_and_dependents() {
    let mut doc = doc_for(&["show", "me", "the", "report"]);
    let set = crate::pipeline::RuleAnnotator::en_default().annotate(&doc);
    attach(&mut doc, &set).expect("attach");
    let signals = extract_routing_signals(&doc);
    assert_eq!(signals.len(), 1);
    let s = &signals[0];
    // The star parse's ROOT is the last non-stop content word.
    assert_eq!(s.predicate, "report");
    assert!(s.subject.is_none());
    assert!(s.direct_object.is_none());
    assert!(s.arguments.is_empty());
    assert_eq!(s.dependents.len(), 3, "star parse flattens every non-root");
    assert!(s.dependents.contains(&"show".to_string()));
    assert!(s.dependents.contains(&"me".to_string()));
    assert!(s.dependents.contains(&"the".to_string()));
}

#[test]
fn multi_sentence_doc_yields_one_signal_per_sentence() {
    let json = r#"[
        {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"cat","pos":"noun","dep":"nsubj","head":1,"lemma":"cat"},
        {"text":"sat","pos":"verb","dep":"root","head":0,"lemma":"sit"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."},
        {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"dog","pos":"noun","dep":"nsubj","head":1,"lemma":"dog"},
        {"text":"ran","pos":"verb","dep":"root","head":0,"lemma":"run"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."}
    ]"#;
    let doc = attached(
        json,
        &["The", "cat", "sat", ".", "The", "dog", "ran", "."],
    );
    let signals = extract_routing_signals(&doc);
    assert_eq!(signals.len(), 2);
    assert_eq!(signals[0].predicate, "sit");
    assert_eq!(signals[0].subject.as_deref(), Some("cat"));
    assert_eq!(signals[1].predicate, "run");
    assert_eq!(signals[1].subject.as_deref(), Some("dog"));
}

#[test]
fn empty_doc_yields_no_signals() {
    let doc = Doc::new(vocab());
    assert!(extract_routing_signals(&doc).is_empty());
}

#[test]
fn routing_signal_serde_round_trip() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let signal = &extract_routing_signals(&doc)[0];
    let json = serde_json::to_string(signal).expect("serialize");
    let back: RoutingSignal = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, *signal);
}

#[test]
fn interlingua_signal_carries_role_ids() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let mut doc = attached(FULL_PARSE, &tokens);
    // Stamp content-addressed lemma ids on every token (the resolve stage
    // does this; here we simulate it so the extraction is testable).
    let ids: Vec<fluent_types::InterlinguaId> = (0..doc.len())
        .map(|i| {
            fluent_types::InterlinguaId::new(
                fluent_types::InterlinguaNamespace::SpacyLemma,
                i as i64 + 1,
            )
        })
        .collect();
    for i in 0..doc.len() {
        doc.token_mut(i).interlingua_lemma_id = Some(ids[i]);
    }
    // A PROPN token with a YaGO entity id → concept id.
    doc.token_mut(3).interlingua_entity_id =
        Some(fluent_types::InterlinguaId::new(
            fluent_types::InterlinguaNamespace::YagoClass,
            0xabc,
        ));

    let s = &extract_routing_signals(&doc)[0];
    let il = s.interlingua.as_ref().expect("interlingua frame");
    // predicate = "Show" (idx 0), iobj = "me" (idx 1), dobj = "report" (idx 4).
    assert_eq!(il.predicate_id, Some(ids[0]));
    assert_eq!(il.indirect_object_id, Some(ids[1]));
    assert_eq!(il.direct_object_id, Some(ids[4]));
    assert!(il.subject_id.is_none());
    assert_eq!(il.concept_ids, vec![fluent_types::InterlinguaId::new(
        fluent_types::InterlinguaNamespace::YagoClass,
        0xabc,
    )]);
    assert_eq!(il.token_ids, ids, "aligned per-token lemma ids");
}

#[test]
fn interlingua_signal_confidence_is_sentence_mean() {
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let mut doc = attached(FULL_PARSE, &tokens);
    let conf = [0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
    for (i, c) in conf.iter().enumerate() {
        doc.token_mut(i).confidence = Some(*c);
    }
    let il = extract_routing_signals(&doc)[0]
        .interlingua
        .clone()
        .expect("interlingua frame");
    // mean of [0.8,0.9,0.7,0.6,0.5,0.4,0.3,0.2] = 4.4 / 8 = 0.55
    assert_eq!(il.confidence, Some(0.55));
}

#[test]
fn interlingua_signal_confidence_none_when_unresolved() {
    // `attach` (the LLM/rule rung path) carries no per-token confidence →
    // the sentence confidence is `None`.
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let il = extract_routing_signals(&doc)[0]
        .interlingua
        .clone()
        .expect("interlingua frame");
    assert!(il.confidence.is_none());
}

#[test]
fn interlingua_signal_serde_skips_none_confidence() {
    // The `default` + `skip_serializing_if` pair keeps already-persisted
    // parse nodes (signals without `confidence`) round-tripping unchanged.
    let tokens = ["Show", "me", "the", "sales", "report", "for", "yesterday", "please"];
    let doc = attached(FULL_PARSE, &tokens);
    let signal = &extract_routing_signals(&doc)[0];
    let json = serde_json::to_string(signal).expect("serialize");
    assert!(!json.contains("confidence"), "None confidence is skipped");
    let back: RoutingSignal = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.interlingua.as_ref().unwrap().confidence, None);
}
