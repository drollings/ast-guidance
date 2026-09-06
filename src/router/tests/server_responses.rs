use super::*;

#[test]
fn fallback_has_stop_reason() {
    let r = fallback_completion("test");
    assert_eq!(r.choices.len(), 1);
    assert_eq!(r.choices[0].finish_reason, "stop");
}

#[test]
fn answer_text_extracts_first_choice() {
    let c = make_text_completion("fast", "the answer");
    assert_eq!(answer_text(&c).as_deref(), Some("the answer"));
}

#[test]
fn answer_text_is_none_without_choices() {
    let c = RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: 0,
        model: "fast".into(),
        choices: vec![],
        usage: Usage::default(),
    };
    assert_eq!(answer_text(&c), None);
}

#[test]
fn answer_text_concatenates_text_parts() {
    let mut c = make_text_completion("fast", "ignored");
    c.choices[0].message.content = RouterMessageContent::Parts(vec![
        crate::types::ContentPart::Text { text: "hello".into() },
        crate::types::ContentPart::Text { text: "world".into() },
    ]);
    assert_eq!(answer_text(&c).as_deref(), Some("hello world"));
}
