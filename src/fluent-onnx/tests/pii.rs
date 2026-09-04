use super::*;

#[test]
fn pii_span_serde_and_slice() {
    let span = PiiSpan::new(2, 9, "credential.password", 0.9);
    let back: PiiSpan =
        serde_json::from_str(&serde_json::to_string(&span).unwrap()).unwrap();
    assert_eq!(back, span);
    assert_eq!(span.slice("xxhunter2xx"), Some("hunter2"));
    // A zero-width span at a valid boundary slices empty; an out-of-bounds
    // range yields None (the offsets contract is byte-exact).
    assert_eq!(PiiSpan::new(0, 0, "x", 0.0).slice("abc"), Some(""));
    assert_eq!(PiiSpan::new(0, 10, "x", 0.0).slice("abc"), None);
}

#[test]
fn decode_biluo_single_and_begin_inside_end() {
    let offsets = [(0, 6), (6, 12), (12, 17), (17, 24)];
    let labels = [
        "B-credential.password".to_string(),
        "I-credential.password".to_string(),
        "E-credential.password".to_string(),
        "O".to_string(),
    ];
    let scores = [0.9, 0.8, 0.7, 0.0];
    let spans = decode_biluo(&offsets, &labels, &scores);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 17);
    assert_eq!(spans[0].label, "credential.password");
    // Mean of the three covered token scores.
    assert!((spans[0].score - 0.8).abs() < 1e-9);
}

#[test]
fn decode_biluo_single_labels_are_their_own_spans() {
    let offsets = [(0, 4), (5, 9)];
    let labels = [
        "S-contact.email".to_string(),
        "S-contact.email".to_string(),
    ];
    let spans = decode_biluo(&offsets, &labels, &[0.9, 0.7]);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0], PiiSpan::new(0, 4, "contact.email", 0.9));
    assert_eq!(spans[1], PiiSpan::new(5, 9, "contact.email", 0.7));
}

#[test]
fn decode_biluo_ignores_zero_width_special_tokens() {
    // [CLS] (0,0), then a real S-span.
    let offsets = [(0, 0), (0, 6)];
    let labels = ["O".to_string(), "S-identity.ssn".to_string()];
    let spans = decode_biluo(&offsets, &labels, &[0.0, 0.9]);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 6);
}

#[test]
fn decode_biluo_type_mismatch_splits_the_span() {
    // B-A then E-B: the E closes A as an A span and opens a single B span.
    let offsets = [(0, 3), (4, 8)];
    let labels = ["B-contact.email".to_string(), "E-contact.phone".to_string()];
    let spans = decode_biluo(&offsets, &labels, &[0.9, 0.8]);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "contact.email");
    assert_eq!(spans[0].end, 3);
    assert_eq!(spans[1].label, "contact.phone");
    assert_eq!(spans[1].start, 4);
    assert_eq!(spans[1].end, 8);
}

#[test]
fn decode_biluo_unclosed_span_closes_at_document_end() {
    let offsets = [(0, 5), (5, 10)];
    let labels = ["B-org.company_name".to_string(), "I-org.company_name".to_string()];
    let spans = decode_biluo(&offsets, &labels, &[0.8, 0.6]);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].end, 10);
    assert!((spans[0].score - 0.7).abs() < 1e-9);
}

#[test]
fn load_id2label_accepts_object_and_array() {
    let dir = tempfile::tempdir().unwrap();
    let object = dir.path().join("id2label.json");
    std::fs::write(
        &object,
        r#"{"0": "O", "1": "B-contact.email", "2": "S-contact.email"}"#,
    )
    .unwrap();
    let map = load_id2label(&object).unwrap();
    assert_eq!(map.get(&0).map(String::as_str), Some("O"));
    assert_eq!(map.get(&2).map(String::as_str), Some("S-contact.email"));

    let array = dir.path().join("labels.json");
    std::fs::write(&array, r#"["O", "B-x", "I-x"]"#).unwrap();
    let map = load_id2label(&array).unwrap();
    assert_eq!(map.get(&1).map(String::as_str), Some("B-x"));
}

#[test]
fn load_id2label_rejects_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.json");
    std::fs::write(&bad, r#"{"a": "O"}"#).unwrap();
    assert!(load_id2label(&bad).is_err());
    assert!(load_id2label(&dir.path().join("missing.json")).is_err());
}

#[test]
fn regex_detector_reports_pattern_matches_with_offsets() {
    let detector = RegexPiiDetector;
    // The canonical table covers email/ssn/phone/api_key/card_number.
    let spans = detector
        .detect("reach me at a@b.com or call 123-45-6789")
        .expect("detect");
    let labels: Vec<&str> = spans.iter().map(|s| s.label.as_str()).collect();
    assert!(labels.contains(&"email"), "email matched: {labels:?}");
    assert!(labels.contains(&"ssn"), "ssn matched: {labels:?}");
    for span in &spans {
        assert_eq!(span.score, 1.0);
        assert!(span.end > span.start);
    }
}

#[test]
fn regex_detector_empty_text_yields_nothing() {
    assert!(RegexPiiDetector.detect("").unwrap().is_empty());
    assert!(RegexPiiDetector.detect("nothing sensitive here").unwrap().is_empty());
}

#[cfg(feature = "onnx")]
#[test]
fn fresh_state_shape_replaces_batch_only() {
    use super::ort_pii::fresh_state_shape;
    // conv cache: [batch(0), hidden(1024), cache_len(3)] → [1, 1024, 3].
    assert_eq!(fresh_state_shape(&[0, 1024, 3]), vec![1, 1024, 3]);
    // kv prefix: [batch(0), heads(8), seq(0), head_dim(64)] → empty prefix.
    assert_eq!(fresh_state_shape(&[0, 8, 0, 64]), vec![1, 8, 0, 64]);
    // Symbolic dims clamp to empty (0), never a negative tensor dim.
    assert_eq!(fresh_state_shape(&[-1, -1]), vec![1, 0]);
    assert_eq!(fresh_state_shape(&[0, 8, -1, 64]), vec![1, 8, 0, 64]);
}
