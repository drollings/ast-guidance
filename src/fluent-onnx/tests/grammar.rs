use super::*;

/// A fixed token-id → text map for hermetic tests.
struct TestVocab {
    tokens: Vec<String>,
}

impl TestVocab {
    fn from_list(tokens: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
        })
    }
}

impl TokenVocab for TestVocab {
    fn token_text(&self, id: u32) -> Option<String> {
        self.tokens.get(id as usize).cloned()
    }
}

/// The standard annotation-ish schema used across tests.
fn sample_schema() -> JsonSchema {
    JsonSchema::new(vec![
        JsonField::required("action", JsonType::String),
        JsonField::required("score", JsonType::Number),
        JsonField::optional("reason", JsonType::String),
    ])
}

fn id_of(vocab: &TestVocab, text: &str) -> u32 {
    tokens_for_literal(vocab, vocab.tokens.len(), text)
        .first()
        .copied()
        .unwrap_or_else(|| panic!("token '{text}' must be present exactly once"))
}

/// Drive a grammar through an expected token stream, asserting each token
/// is allowed and committed without rejection.
fn drive_accept(grammar: &mut dyn Grammar, vocab: &TestVocab, stream: &[&str]) {
    grammar.reset();
    for text in stream {
        let id = id_of(vocab, text);
        let allowed = grammar.allowed_ids(vocab.tokens.len());
        assert!(
            allowed.contains(&id),
            "token '{text}' must be allowed at this step (allowed={allowed:?})"
        );
        grammar.advance(id);
    }
}

#[test]
fn object_accepts_well_formed_stream() {
    let vocab = TestVocab::from_list(&[
        "{", "}", ":", ",", "\"action\"", "\"run\"", "\"score\"", "1.5", "\"reason\"", "\"ok\"",
    ]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    drive_accept(
        &mut g,
        &vocab,
        &["{", "\"action\"", ":", "\"run\"", ",", "\"score\"", ":", "1.5", "}"],
    );
}

#[test]
fn object_requires_all_required_fields_before_close() {
    let vocab = TestVocab::from_list(&["{", "}", ":", ",", "\"action\"", "\"run\"", "\"score\"", "1.5"]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    g.reset();
    g.advance(id_of(&vocab, "{"));
    g.advance(id_of(&vocab, "\"action\""));
    g.advance(id_of(&vocab, ":"));
    g.advance(id_of(&vocab, "\"run\""));
    // After the value, `}` (close) must NOT be allowed until `score` seen.
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "}")),
        "closing '}}' before required 'score' must be rejected (allowed={allowed:?})"
    );
    // The comma is allowed (continue to the next field).
    assert!(allowed.contains(&id_of(&vocab, ",")));
}

#[test]
fn object_rejects_unknown_key() {
    let vocab = TestVocab::from_list(&["{", ":", "\"bogus\"", "\"x\""]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    g.reset();
    g.advance(id_of(&vocab, "{"));
    // The only declared key tokens are action/score/reason — "bogus" is not
    // in the allowed set.
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "\"bogus\"")),
        "undeclared key must be rejected"
    );
}

#[test]
fn object_rejects_repeated_key() {
    let vocab = TestVocab::from_list(&["{", "}", ":", ",", "\"action\"", "\"run\"", "\"score\"", "1.5"]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    g.reset();
    drive_accept(&mut g, &vocab, &["{", "\"action\"", ":", "\"run\"", ","]);
    // After the comma, "action" is already seen — it must not be allowed.
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "\"action\"")),
        "repeated key must be rejected"
    );
}

#[test]
fn object_type_checks_values() {
    let vocab = TestVocab::from_list(&[
        "{", "}", ":", ",", "\"action\"", "\"run\"", "\"score\"", "not-a-number",
    ]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    g.reset();
    g.advance(id_of(&vocab, "{"));
    g.advance(id_of(&vocab, "\"action\""));
    g.advance(id_of(&vocab, ":"));
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "not-a-number")),
        "a non-number token must be rejected for a Number field"
    );
    assert!(allowed.contains(&id_of(&vocab, "\"run\"")));
}

#[test]
fn malformed_order_rejects_terminally() {
    let vocab = TestVocab::from_list(&["{", "}", ":", "\"action\"", "\"run\""]);
    let mut g = JsonObjectGrammar::new(sample_schema(), vocab.clone());
    g.reset();
    g.advance(id_of(&vocab, "{"));
    // A value before a colon is malformed.
    g.advance(id_of(&vocab, "\"run\""));
    assert!(
        g.allowed_ids(vocab.tokens.len()).is_empty(),
        "rejected → empty allowed"
    );
}

#[test]
fn array_values_round_trip() {
    let schema = JsonSchema::new(vec![JsonField::required("tags", JsonType::String)])
        .with_arrays();
    let vocab = TestVocab::from_list(&[
        "{", "}", ":", ",", "[", "]", "\"tags\"", "\"a\"", "\"b\"",
    ]);
    let mut g = JsonObjectGrammar::new(schema, vocab.clone());
    drive_accept(
        &mut g,
        &vocab,
        &["{", "\"tags\"", ":", "[", "\"a\"", ",", "\"b\"", "]", "}"],
    );
}

#[test]
fn object_values_nest() {
    let schema = JsonSchema::new(vec![JsonField::required("meta", JsonType::Object)]);
    let vocab = TestVocab::from_list(&["{", "}", ":", "\"meta\"", "\"k\"", "\"v\""]);
    let mut g = JsonObjectGrammar::new(schema, vocab.clone());
    // `{ "meta": { "k": "v" } }` — the object value accepts balanced braces.
    drive_accept(&mut g, &vocab, &["{", "\"meta\"", ":", "{", "\"k\"", "\"v\"", "}", "}"]);
}

#[test]
fn batch_grammar_accepts_well_formed_array() {
    let schema = JsonSchema::new(vec![JsonField::required("action", JsonType::String)]);
    let vocab = TestVocab::from_list(&["[", "]", ",", "{", "}", ":", "\"action\"", "\"run\"", "\"halt\""]);
    let mut g = BatchPromptGrammar::new(&[schema.clone(), schema], vocab.clone());
    drive_accept(
        &mut g,
        &vocab,
        &[
            "[",
            "{", "\"action\"", ":", "\"run\"", "}",
            ",",
            "{", "\"action\"", ":", "\"halt\"", "}",
            "]",
        ],
    );
}

#[test]
fn batch_grammar_rejects_missing_comma_between_objects() {
    let schema = JsonSchema::new(vec![JsonField::required("action", JsonType::String)]);
    let vocab = TestVocab::from_list(&["[", "]", ",", "{", "}", ":", "\"action\"", "\"run\""]);
    let mut g = BatchPromptGrammar::new(&[schema.clone(), schema], vocab.clone());
    g.reset();
    g.advance(id_of(&vocab, "["));
    g.advance(id_of(&vocab, "{"));
    g.advance(id_of(&vocab, "\"action\""));
    g.advance(id_of(&vocab, ":"));
    g.advance(id_of(&vocab, "\"run\""));
    g.advance(id_of(&vocab, "}"));
    // After the first object, a second `{` without a `,` must be rejected.
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "{")),
        "second object without a comma must be rejected"
    );
    assert!(allowed.contains(&id_of(&vocab, ",")));
    assert!(allowed.contains(&id_of(&vocab, "]")));
}

#[test]
fn batch_grammar_empty_and_single() {
    let schema = JsonSchema::new(vec![JsonField::required("a", JsonType::String)]);
    let vocab = TestVocab::from_list(&["[", "]"]);
    let g = BatchPromptGrammar::new(&[schema.clone(), schema], vocab.clone());
    assert_eq!(g.len(), 2);
    assert!(!g.is_empty());
    let empty = BatchPromptGrammar::new(&[], vocab.clone());
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    // `[]` is a well-formed empty batch.
    let mut empty = empty;
    drive_accept(&mut empty, &vocab, &["[", "]"]);
}

#[test]
fn tokens_for_literal_finds_exact_matches() {
    let vocab = TestVocab::from_list(&["{", "action", "run", "something_else"]);
    let ids = tokens_for_literal(&*vocab, vocab.tokens.len(), "action");
    assert_eq!(ids, vec![1]);
    assert!(tokens_for_literal(&*vocab, vocab.tokens.len(), "missing").is_empty());
}

#[test]
fn is_valid_json_prefix_checks_complete_and_truncated() {
    assert!(is_valid_json_prefix(""));
    assert!(is_valid_json_prefix("{"));
    assert!(is_valid_json_prefix("{\n"));
    assert!(is_valid_json_prefix("{\"action\": \"run\""));
    assert!(is_valid_json_prefix("{\"a\":[1,"));
    assert!(is_valid_json_prefix("{\"answer\":\"four\",\"score\":4}"));
    assert!(is_valid_json_prefix("\"str"));
    assert!(is_valid_json_prefix("true"));
    assert!(is_valid_json_prefix("12"));
    // Malformed (not a prefix of any valid JSON) must be rejected.
    assert!(!is_valid_json_prefix("{x"));
    assert!(!is_valid_json_prefix("{x:"));
    assert!(!is_valid_json_prefix("}"));
    assert!(!is_valid_json_prefix("trux"));
    assert!(!is_valid_json_prefix("[1,,2]"));
}

#[test]
fn normalize_strips_whitespace_and_byte_markers() {
    assert_eq!(normalize_token("{").as_ref(), "{");
    assert_eq!(normalize_token("Ġ{").as_ref(), "{");
    assert_eq!(normalize_token(" Ġ{ ").as_ref(), "{");
    assert_eq!(normalize_token("\u{010A}\u{0120}:").as_ref(), ":");
    assert_eq!(normalize_token("\"action\"").as_ref(), "\"action\"");
}

#[test]
fn object_accepts_whitespace_marked_tokens() {
    let vocab = TestVocab::from_list(&["Ġ{", "Ġ\"action\"", "Ġ:", "Ġ\"run\"", "Ġ}", "Ġ,"]);
    let schema = JsonSchema::new(vec![JsonField::required("action", JsonType::String)]);
    let mut g = JsonObjectGrammar::new(schema, vocab.clone());
    drive_accept(&mut g, &vocab, &["Ġ{", "Ġ\"action\"", "Ġ:", "Ġ\"run\"", "Ġ}"]);
}

#[test]
fn number_and_integer_validation() {
    assert!(is_json_number("0"));
    assert!(is_json_number("-1.5"));
    assert!(is_json_number("1e3"));
    assert!(is_json_number("1E-3"));
    assert!(!is_json_number(""));
    assert!(!is_json_number("-"));
    assert!(!is_json_number("1.2.3"));
    assert!(!is_json_number("abc"));
    assert!(is_json_integer("42"));
    assert!(is_json_integer("-7"));
    assert!(is_json_integer("+3"));
    assert!(!is_json_integer("4.2"));
    assert!(!is_json_integer("-"));
    assert!(!is_json_integer(""));
    assert!(scalar_matches(JsonType::Integer, "42"));
    assert!(scalar_matches(JsonType::Boolean, "true"));
    assert!(!scalar_matches(JsonType::Boolean, "yes"));
    assert!(scalar_matches(JsonType::String, "\"hi\""));
    assert!(!scalar_matches(JsonType::String, "hi"));
}

#[test]
fn array_grammar_accepts_dynamic_length_object_array() {
    let item = JsonSchema::new(vec![
        JsonField::required("text", JsonType::String),
        JsonField::required("pos", JsonType::String),
        JsonField::required("head", JsonType::Integer),
    ]);
    let vocab = TestVocab::from_list(&[
        "[", "]", ",", "{", "}", ":", "\"text\"", "\"run\"", "\"pos\"", "\"verb\"", "\"head\"",
        "0", "1",
    ]);
    let mut g = JsonArrayGrammar::new(item, vocab.clone());
    drive_accept(
        &mut g,
        &vocab,
        &[
            "[",
            "{", "\"text\"", ":", "\"run\"", ",", "\"pos\"", ":", "\"verb\"", ",", "\"head\"",
            ":", "0", "}",
            ",",
            "{", "\"text\"", ":", "\"run\"", ",", "\"pos\"", ":", "\"verb\"", ",", "\"head\"",
            ":", "1", "}",
            "]",
        ],
    );
}

#[test]
fn array_grammar_accepts_empty_and_single() {
    let item = JsonSchema::new(vec![JsonField::required("a", JsonType::String)]);
    let vocab = TestVocab::from_list(&["[", "]", "{", "}", ":", "\"a\"", "\"x\""]);
    let mut g = JsonArrayGrammar::new(item.clone(), vocab.clone());
    drive_accept(&mut g, &vocab, &["[", "]"]);
    let mut g = JsonArrayGrammar::new(item, vocab.clone());
    drive_accept(&mut g, &vocab, &["[", "{", "\"a\"", ":", "\"x\"", "}", "]"]);
}

#[test]
fn array_grammar_rejects_missing_comma_between_objects() {
    let item = JsonSchema::new(vec![JsonField::required("a", JsonType::String)]);
    let vocab = TestVocab::from_list(&["[", "]", ",", "{", "}", ":", "\"a\"", "\"x\""]);
    let mut g = JsonArrayGrammar::new(item, vocab.clone());
    g.reset();
    drive_accept(&mut g, &vocab, &["[", "{", "\"a\"", ":", "\"x\"", "}"]);
    let allowed = g.allowed_ids(vocab.tokens.len());
    assert!(
        !allowed.contains(&id_of(&vocab, "{")),
        "a second object without a comma must be rejected"
    );
    assert!(allowed.contains(&id_of(&vocab, ",")));
    assert!(allowed.contains(&id_of(&vocab, "]")));
}

#[test]
fn from_json_schema_parses_object_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "score": {"type": "number"},
            "count": {"type": "integer"},
            "flag": {"type": "boolean"},
            "note": {"type": "string"}
        },
        "required": ["text", "score"]
    });
    let s = JsonSchema::from_json_schema(&schema).expect("parse");
    assert_eq!(s.fields.len(), 5);
    let text = s.fields.iter().find(|f| f.name == "text").unwrap();
    assert_eq!(text.ty, JsonType::String);
    assert!(text.required);
    let count = s.fields.iter().find(|f| f.name == "count").unwrap();
    assert_eq!(count.ty, JsonType::Integer);
    assert!(!count.required);
}

#[test]
fn from_json_schema_rejects_array_typed_property() {
    // A review-style schema with array-of-objects fields is not
    // representable — the whole schema rejects so the caller degrades to
    // free text rather than forbid valid output.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "corrections": {"type": "array", "items": {"type": "object"}}
        },
        "required": ["corrections"]
    });
    assert!(JsonSchema::from_json_schema(&schema).is_none());
}

#[test]
fn from_json_schema_rejects_non_object_and_missing_properties() {
    assert!(JsonSchema::from_json_schema(&serde_json::json!({"type": "array"})).is_none());
    assert!(JsonSchema::from_json_schema(&serde_json::json!({"type": "object"})).is_none());
    assert!(JsonSchema::from_json_schema(&serde_json::json!("nope")).is_none());
}

#[test]
fn grammar_from_json_schema_builds_object_and_array() {
    let vocab = TestVocab::from_list(&[
        "[", "]", "{", "}", ":", ",", "\"a\"", "\"x\"", "\"b\"", "1",
    ]);
    // Object form → JsonObjectGrammar.
    let obj_schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"]
    });
    let mut g = grammar_from_json_schema(&obj_schema, vocab.clone()).expect("object grammar");
    drive_accept(g.as_mut(), &vocab, &["{", "\"a\"", ":", "\"x\"", "}"]);

    // Array-of-objects form → JsonArrayGrammar.
    let arr_schema = serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"]
        }
    });
    let mut g = grammar_from_json_schema(&arr_schema, vocab.clone()).expect("array grammar");
    drive_accept(g.as_mut(), &vocab, &["[", "{", "\"a\"", ":", "\"x\"", "}", "]"]);

    // Unrepresentable → None (free text).
    assert!(grammar_from_json_schema(&serde_json::json!("nope"), vocab.clone()).is_none());
    assert!(grammar_from_json_schema(&serde_json::json!({"type":"number"}), vocab.clone()).is_none());
}
