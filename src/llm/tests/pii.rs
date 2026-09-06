use super::*;

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
