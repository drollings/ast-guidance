//! Grammar-constrained decoding primitives — ort-free, fully hermetic.
//!
//! A [`Grammar`] is a token-level state machine that, given the token stream
//! decoded so far, yields the set of token ids that are valid to emit next.
//! Because the automaton is a pure function of the already-decoded tokens, a
//! *structurally invalid* object is impossible to emit — it is never rejected
//! after the fact, it cannot be produced in the first place (ROADMAP G3).
//!
//! The two shapes the ROADMAP needs:
//!
//! - [`JsonObjectGrammar`] — a single JSON object whose keys are a declared
//!   `JsonSchema` and whose values are checked against each field's type. Only
//!   `{`, the declared field names, `:`, type-checked values, `,`, and `}` in
//!   valid order are ever permitted.
//! - [`BatchPromptGrammar`] — an array wrapper over N `JsonObjectGrammar`s: the
//!   M3 "one grammar-constrained call per `SupervisedBatch` wave" shape.
//!
//! The grammar reasons over *token strings*, so it needs a token-id → text
//! lookup. That lookup is injected via the [`TokenVocab`] trait (the real
//! implementation maps over the `tokenizers` vocab; hermetic tests inject a
//! small fixed map), keeping this module free of any `ort`/`tokenizers`
//! dependency.
//!
//! ## Semantics
//!
//! - `allowed_ids(vocab_size)` returns every id in `0..vocab_size` whose token
//!   text is a valid next transition. Once the object is complete — or a
//!   malformed token has been advanced — the allowed set is empty (the decoder
//!   stops, or the caller observes the empty set as a rejection).
//! - `advance(token_id)` decodes the token and commits it to the automaton. A
//!   token that is not a valid transition at the current state puts the
//!   grammar into a permanent *rejected* state (allowed set empty thereafter).
//!
//! The decoder must therefore never `advance` a token it did not first see in
//! `allowed_ids`; the empty-allowed set is the signal to stop. This mirrors the
//! llama.cpp fork's GBNF semantics (the classifier's `response_format` seam)
//! without depending on it.

use std::sync::Arc;

/// A token-id → text lookup. Injected so the grammar can reason over token
/// strings without depending on `tokenizers` (which is behind the `onnx`
/// feature).
pub trait TokenVocab: Send + Sync {
    /// The decoded text of a token id, or `None` when the id is out of range.
    /// Owned so the grammar can inspect a candidate without aliasing the
    /// vocab's interior (the `tokenizers` lookup returns an owned `String`).
    fn token_text(&self, id: u32) -> Option<String>;
}

/// A `tokenizers`-vocab-backed [`TokenVocab`]. Lives behind the `onnx`
/// feature; hermetic tests use a fixed in-memory map.
#[cfg(feature = "onnx")]
pub struct HuggingFaceVocab {
    inner: tokenizers::Tokenizer,
}

#[cfg(feature = "onnx")]
impl HuggingFaceVocab {
    /// Build a vocab lookup over an already-loaded tokenizer.
    pub fn new(inner: tokenizers::Tokenizer) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "onnx")]
impl TokenVocab for HuggingFaceVocab {
    fn token_text(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// The grammar interface the decoder drives. A single instance owns the
/// automaton state and the vocab; `reset` restarts a generation.
pub trait Grammar: Send {
    /// Return to the initial state (start a fresh generation).
    fn reset(&mut self);

    /// The token ids that are valid to emit next, given the tokens advanced so
    /// far. Empty when the grammar is complete or has rejected the stream.
    fn allowed_ids(&self, vocab_size: usize) -> Vec<u32>;

    /// Commit a decoded token to the automaton. Only tokens returned by
    /// `allowed_ids` should be advanced; advancing a malformed token puts the
    /// grammar into a permanent rejected state (empty allowed set).
    fn advance(&mut self, token_id: u32);
}

/// The JSON type a field's value must satisfy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonType {
    /// A JSON string (`"..."`).
    String,
    /// A JSON number (`0`, `-1.5`, `1e3`).
    Number,
    /// A JSON integer (digits, optional sign).
    Integer,
    /// A JSON boolean (`true` / `false`).
    Boolean,
    /// A nested JSON object (`{...}`).
    Object,
}

/// A declared field of a [`JsonSchema`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonField {
    /// The field name (the JSON string key, without quotes).
    pub name: String,
    /// The type the field's value must satisfy.
    pub ty: JsonType,
    /// Whether the field must be present for the object to be complete.
    pub required: bool,
}

impl JsonField {
    /// A required field.
    pub fn required(name: impl Into<String>, ty: JsonType) -> Self {
        Self {
            name: name.into(),
            ty,
            required: true,
        }
    }

    /// An optional field.
    pub fn optional(name: impl Into<String>, ty: JsonType) -> Self {
        Self {
            name: name.into(),
            ty,
            required: false,
        }
    }
}

/// The structural contract a [`JsonObjectGrammar`] enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSchema {
    /// The declared fields, in declaration order (keys may appear in any
    /// order, but only these names, each at most once).
    pub fields: Vec<JsonField>,
    /// Whether a field's value may be an array of the field's type
    /// (`[elem, ...]`).
    pub allow_array: bool,
}

impl JsonSchema {
    /// A schema over the given fields, with array values disabled.
    pub fn new(fields: Vec<JsonField>) -> Self {
        Self {
            fields,
            allow_array: false,
        }
    }

    /// Enable array-typed values.
    #[must_use]
    pub fn with_arrays(mut self) -> Self {
        self.allow_array = true;
        self
    }

    /// Parse a JSON-schema `Value` (`{"type":"object","properties":{...},
    /// "required":[...]}`) into the structural [`JsonSchema`]. This is the
    /// bridge from the llama-fork `response_format.schema` vocabulary the
    /// router's consumers speak (`AnnotationRecord::contract`, the classifier
    /// schema, a review schema) to the automaton's declared-field contract.
    ///
    /// Returns `None` when the schema is not a representable object form — a
    /// non-`object` top level, missing `properties`, or a property whose `type`
    /// is not one of the scalar/object set this automaton enforces. Array-typed
    /// properties (e.g. `corrections` in a review schema) are **not**
    /// representable and reject the whole schema so the caller degrades to free
    /// text rather than emit a grammar that would make valid output impossible.
    pub fn from_json_schema(v: &serde_json::Value) -> Option<Self> {
        if v.get("type")?.as_str()? != "object" {
            return None;
        }
        let properties = v.get("properties")?.as_object()?;
        let required: Vec<String> = v
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let mut fields = Vec::with_capacity(properties.len());
        for (name, prop) in properties {
            let ty = match prop.get("type").and_then(serde_json::Value::as_str) {
                Some("string") | None => JsonType::String,
                Some("number") => JsonType::Number,
                Some("integer") => JsonType::Integer,
                Some("boolean") => JsonType::Boolean,
                Some("object") => JsonType::Object,
                // Array-typed properties are not representable by the automaton.
                Some("array" | _) => return None,
            };
            fields.push(JsonField {
                name: name.clone(),
                ty,
                required: required.iter().any(|r| r == name),
            });
        }
        Some(Self {
            fields,
            allow_array: false,
        })
    }
}

/// The automaton's structural cursor. `Clone` so the grammar can `peek` a
/// transition without mutating its live state. Every "inside the object"
/// state carries the `seen` key set so it survives commas (no duplicate keys,
/// required-field check at close).
#[derive(Clone, Debug, PartialEq, Eq)]
enum MachineState {
    /// Before the opening `{`.
    Start,
    /// After `{`: expecting a declared, unseen key string, or `}` when every
    /// required field has been seen.
    ExpectKey { seen: Vec<String> },
    /// After a key: expecting `:`.
    ExpectColon { field: usize, seen: Vec<String> },
    /// After `:`: expecting the value for `field`.
    Value {
        field: usize,
        in_array: bool,
        seen: Vec<String>,
    },
    /// After `[` for an array-typed value: expecting the first element or `]`.
    ArrayStart { field: usize, seen: Vec<String> },
    /// After a scalar array element: expecting `,` or `]`.
    ArrayElem { field: usize, seen: Vec<String> },
    /// Inside a nested Object value: `depth` open braces must be matched.
    ObjectValue { depth: usize, seen: Vec<String> },
    /// After a complete value: expecting `,` or `}`.
    ExpectCommaOrClose { seen: Vec<String> },
    /// The object is complete (accepting terminal; no further tokens valid).
    Complete,
}

/// Normalize a candidate token for JSON matching: strip JSON-insignificant
/// whitespace and the BPE byte-prefix markers (`Ġ`=space, `Ċ`=newline, …) that
/// HuggingFace subword tokenizers prefix onto tokens, so a token like `Ġ{` or
/// `Ġ"answer"` matches the structural literal `{` / `"answer"`. JSON ignores
/// the stripped whitespace, so this never changes the emitted document's
/// meaning.
fn normalize_token(token: &str) -> std::borrow::Cow<'_, str> {
    let trimmed: String = token
        .chars()
        .filter(|c| !(c.is_whitespace() || is_byte_marker(*c)))
        .collect();
    if trimmed.len() == token.len() {
        std::borrow::Cow::Borrowed(token)
    } else {
        std::borrow::Cow::Owned(trimmed)
    }
}

/// Whether a char is a BPE byte-prefix marker (the `tokenizers` byte-level
/// fallback encodes a raw byte as the char in `U+0100..=U+01FF`).
fn is_byte_marker(c: char) -> bool {
    let cp = c as u32;
    (0x0100..=0x01FF).contains(&cp)
}

/// The pure JSON-object automaton. Works over token *strings* and has no
/// vocab dependency — [`JsonObjectGrammar`] supplies the vocab and the
/// `Grammar` trait surface.
#[derive(Debug, Clone)]
struct JsonObjectMachine {
    state: MachineState,
    schema: JsonSchema,
    /// Whether a malformed token was advanced; terminal (allowed set empty).
    rejected: bool,
}

impl JsonObjectMachine {
    fn new(schema: JsonSchema) -> Self {
        Self {
            state: MachineState::Start,
            schema,
            rejected: false,
        }
    }

    fn reset(&mut self) {
        self.state = MachineState::Start;
        self.rejected = false;
    }

    fn is_rejected(&self) -> bool {
        self.rejected
    }

    fn is_complete(&self) -> bool {
        self.state == MachineState::Complete
    }

    /// Whether the object has not yet been opened (still in `Start`).
    fn is_at_start(&self) -> bool {
        matches!(self.state, MachineState::Start)
    }

    /// The field index for a declared field name, or `None`.
    fn field_index(&self, name: &str) -> Option<usize> {
        self.schema.fields.iter().position(|f| f.name == name)
    }

    /// The `JsonType` for a declared field index.
    fn field_type(&self, field: usize) -> JsonType {
        self.schema
            .fields
            .get(field)
            .map_or(JsonType::String, |f| f.ty)
    }

    /// Whether every `required` field is present in `seen`.
    fn all_required_seen(&self, seen: &[String]) -> bool {
        self.schema
            .fields
            .iter()
            .filter(|f| f.required)
            .all(|f| seen.iter().any(|s| s == &f.name))
    }

    /// The state transition for `token` at the current state, or `None` when
    /// the token is not a valid transition.
    fn transition(&self, token: &str) -> Option<MachineState> {
        if self.rejected || self.is_complete() {
            return None;
        }
        let token = normalize_token(token);
        let token = token.as_ref();
        match &self.state {
            MachineState::Start => {
                if token == "{" {
                    Some(MachineState::ExpectKey { seen: Vec::new() })
                } else {
                    None
                }
            }
            MachineState::ExpectKey { seen } => {
                // A declared, not-yet-seen field name (as a quoted string key).
                if is_quoted_string(token) {
                    let name = &token[1..token.len() - 1];
                    if let Some(field) = self.field_index(name) {
                        if !seen.iter().any(|s| s == name) {
                            let mut next = seen.clone();
                            next.push(name.to_string());
                            return Some(MachineState::ExpectColon { field, seen: next });
                        }
                    }
                    return None;
                }
                // Close the object once every required field is present.
                if token == "}" && self.all_required_seen(seen) {
                    return Some(MachineState::Complete);
                }
                None
            }
            MachineState::ExpectColon { field, seen } => {
                if token == ":" {
                    Some(MachineState::Value {
                        field: *field,
                        in_array: false,
                        seen: seen.clone(),
                    })
                } else {
                    None
                }
            }
            MachineState::Value {
                field,
                in_array,
                seen,
            } => {
                let ty = self.field_type(*field);
                if self.schema.allow_array && !*in_array && token == "[" {
                    return Some(MachineState::ArrayStart {
                        field: *field,
                        seen: seen.clone(),
                    });
                }
                if *in_array && token == "]" {
                    return Some(MachineState::ExpectCommaOrClose { seen: seen.clone() });
                }
                if scalar_matches(ty, token) {
                    return Some(if *in_array {
                        MachineState::ArrayElem {
                            field: *field,
                            seen: seen.clone(),
                        }
                    } else {
                        MachineState::ExpectCommaOrClose { seen: seen.clone() }
                    });
                }
                if ty == JsonType::Object && token == "{" {
                    return Some(MachineState::ObjectValue {
                        depth: 1,
                        seen: seen.clone(),
                    });
                }
                None
            }
            MachineState::ArrayStart { field, seen } => {
                if token == "]" {
                    return Some(MachineState::ExpectCommaOrClose { seen: seen.clone() });
                }
                let ty = self.field_type(*field);
                if scalar_matches(ty, token) {
                    return Some(MachineState::ArrayElem {
                        field: *field,
                        seen: seen.clone(),
                    });
                }
                None
            }
            MachineState::ArrayElem { field, seen } => {
                if token == "," {
                    return Some(MachineState::ArrayStart {
                        field: *field,
                        seen: seen.clone(),
                    });
                }
                if token == "]" {
                    return Some(MachineState::ExpectCommaOrClose { seen: seen.clone() });
                }
                None
            }
            MachineState::ObjectValue { depth, seen } => match token {
                "{" => Some(MachineState::ObjectValue {
                    depth: depth + 1,
                    seen: seen.clone(),
                }),
                "}" if *depth > 1 => Some(MachineState::ObjectValue {
                    depth: depth - 1,
                    seen: seen.clone(),
                }),
                "}" => Some(MachineState::ExpectCommaOrClose { seen: seen.clone() }),
                _ => Some(MachineState::ObjectValue {
                    depth: *depth,
                    seen: seen.clone(),
                }),
            },
            MachineState::ExpectCommaOrClose { seen } => {
                if token == "," {
                    Some(MachineState::ExpectKey { seen: seen.clone() })
                } else if token == "}" && self.all_required_seen(seen) {
                    Some(MachineState::Complete)
                } else {
                    None
                }
            }
            MachineState::Complete => None,
        }
    }

    /// Commit a token; `true` when accepted.
    fn feed(&mut self, token: &str) -> bool {
        if let Some(next) = self.transition(token) {
            self.state = next;
            true
        } else {
            self.rejected = true;
            false
        }
    }

    /// Whether `token` is a valid next transition (no state mutation).
    fn peek(&self, token: &str) -> bool {
        self.transition(token).is_some()
    }
}

/// Whether a token is a complete, non-empty quoted string (`"..."`).
fn is_quoted_string(token: &str) -> bool {
    token.len() >= 2 && token.starts_with('"') && token.ends_with('"')
}

/// Whether a token is a valid scalar value of `ty`.
fn scalar_matches(ty: JsonType, token: &str) -> bool {
    match ty {
        JsonType::String => is_quoted_string(token),
        JsonType::Number => is_json_number(token),
        JsonType::Integer => is_json_integer(token),
        JsonType::Boolean => token == "true" || token == "false",
        JsonType::Object => false, // handled via `{` in the transition
    }
}

/// A lenient JSON-number check (digits with optional sign, `.`, `e`/`E`).
fn is_json_number(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let bytes = token.as_bytes();
    let mut i = 0;
    if matches!(bytes[0], b'-' | b'+') {
        i = 1;
    }
    let mut has_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    while i < bytes.len() {
        match bytes[i] {
            b'0'..=b'9' => has_digit = true,
            b'.' if !seen_dot && !seen_exp => seen_dot = true,
            b'e' | b'E' if !seen_exp && has_digit => {
                seen_exp = true;
                // Allow an optional sign after the exponent.
                if i + 1 < bytes.len() && matches!(bytes[i + 1], b'+' | b'-') {
                    i += 1;
                }
            }
            _ => return false,
        }
        i += 1;
    }
    has_digit
}

/// A strict JSON-integer check (optional sign, digits only).
fn is_json_integer(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let bytes = token.as_bytes();
    let mut i = 0;
    if matches!(bytes[0], b'-' | b'+') {
        i = 1;
    }
    if i >= bytes.len() {
        return false;
    }
    bytes[i..].iter().all(u8::is_ascii_digit)
}

/// A concrete [`Grammar`] producing exactly one JSON object conforming to a
/// [`JsonSchema`]. Values are type-checked; keys are restricted to the declared
/// field names; repeated keys and a closing `}` before every required field are
/// rejected.
pub struct JsonObjectGrammar {
    machine: JsonObjectMachine,
    vocab: Arc<dyn TokenVocab>,
}

impl JsonObjectGrammar {
    /// Build a grammar over `schema`, decoding token strings via `vocab`.
    pub fn new(schema: JsonSchema, vocab: Arc<dyn TokenVocab>) -> Self {
        Self {
            machine: JsonObjectMachine::new(schema),
            vocab,
        }
    }
}

impl Grammar for JsonObjectGrammar {
    fn reset(&mut self) {
        self.machine.reset();
    }

    fn allowed_ids(&self, vocab_size: usize) -> Vec<u32> {
        if self.machine.is_rejected() || self.machine.is_complete() {
            return Vec::new();
        }
        (0..vocab_size as u32)
            .filter(|id| {
                self.vocab
                    .token_text(*id)
                    .is_some_and(|t| self.machine.peek(&t))
            })
            .collect()
    }

    fn advance(&mut self, token_id: u32) {
        let Some(text) = self.vocab.token_text(token_id) else {
            self.machine.rejected = true;
            return;
        };
        self.machine.feed(&text);
    }
}

/// A [`Grammar`] producing a JSON array of N [`JsonObjectGrammar`] objects:
/// `[ obj, obj, ... ]` — the M3 one-call-per-wave shape.
pub struct BatchPromptGrammar {
    /// The per-element object machines.
    machines: Vec<JsonObjectMachine>,
    /// The index of the object currently being produced.
    current: usize,
    /// Whether the opening `[` has been emitted.
    opened: bool,
    /// Whether the closing `]` has been emitted (terminal).
    closed: bool,
    /// Whether a malformed token was advanced (terminal).
    rejected: bool,
    vocab: Arc<dyn TokenVocab>,
}

impl BatchPromptGrammar {
    /// Build a batch grammar over `schemas` (one object per element, in order),
    /// decoding token strings via `vocab`.
    pub fn new(schemas: &[JsonSchema], vocab: Arc<dyn TokenVocab>) -> Self {
        let machines = schemas
            .iter()
            .cloned()
            .map(JsonObjectMachine::new)
            .collect();
        Self {
            machines,
            current: 0,
            opened: false,
            closed: false,
            rejected: false,
            vocab,
        }
    }

    /// The number of objects in the batch.
    pub fn len(&self) -> usize {
        self.machines.len()
    }

    /// Whether the batch has zero objects.
    pub fn is_empty(&self) -> bool {
        self.machines.is_empty()
    }

    /// Whether `token` is a valid next transition (no mutation).
    fn peek(&self, token: &str) -> bool {
        if self.rejected || self.closed {
            return false;
        }
        if !self.opened {
            return token == "[";
        }
        if self.current >= self.machines.len() {
            // Every object emitted: expect the closing `]`.
            return token == "]";
        }
        // Inside the current object: after it completes, expect `,` or `]`.
        if self.machines[self.current].is_complete() {
            return token == "," || token == "]";
        }
        self.machines[self.current].peek(token)
    }

    /// Commit `token`; `true` when accepted.
    fn feed(&mut self, token: &str) -> bool {
        if self.rejected || self.closed {
            self.rejected = true;
            return false;
        }
        if !self.opened {
            if token == "[" {
                self.opened = true;
                return true;
            }
            self.rejected = true;
            return false;
        }
        if self.current >= self.machines.len() {
            if token == "]" {
                self.closed = true;
                return true;
            }
            self.rejected = true;
            return false;
        }
        if self.machines[self.current].is_complete() {
            // Object boundary: `,` advances to the next object, `]` finishes.
            if token == "," {
                self.current += 1;
                return true;
            }
            if token == "]" {
                self.closed = true;
                return true;
            }
            self.rejected = true;
            return false;
        }
        if self.machines[self.current].feed(token) {
            true
        } else {
            self.rejected = true;
            false
        }
    }
}

impl Grammar for BatchPromptGrammar {
    fn reset(&mut self) {
        self.current = 0;
        self.opened = false;
        self.closed = false;
        self.rejected = false;
        for m in &mut self.machines {
            m.reset();
        }
    }

    fn allowed_ids(&self, vocab_size: usize) -> Vec<u32> {
        if self.rejected || self.closed {
            return Vec::new();
        }
        (0..vocab_size as u32)
            .filter(|id| self.vocab.token_text(*id).is_some_and(|t| self.peek(&t)))
            .collect()
    }

    fn advance(&mut self, token_id: u32) {
        let Some(text) = self.vocab.token_text(token_id) else {
            self.rejected = true;
            return;
        };
        self.feed(&text);
    }
}

/// A [`Grammar`] producing a JSON array of **arbitrarily many** objects, each
/// conforming to the same item [`JsonSchema`]: `[ obj, obj, ... ]`. Unlike
/// [`BatchPromptGrammar`] (a fixed N — the M3 wave shape), the length here is
/// dynamic — the shape of a batch annotation response (one object per token).
///
/// An empty `[]` is well-formed; otherwise objects are `,`-separated and the
/// array closes with `]`. Each object is a fresh [`JsonObjectMachine`] (reset on
/// the comma boundary), so the item schema's required-field / duplicate-key
/// discipline applies per element.
pub struct JsonArrayGrammar {
    machine: JsonObjectMachine,
    vocab: Arc<dyn TokenVocab>,
    opened: bool,
    closed: bool,
    rejected: bool,
}

impl JsonArrayGrammar {
    /// Build a grammar producing `[ obj, obj, ... ]` over `item_schema`.
    pub fn new(item_schema: JsonSchema, vocab: Arc<dyn TokenVocab>) -> Self {
        Self {
            machine: JsonObjectMachine::new(item_schema),
            vocab,
            opened: false,
            closed: false,
            rejected: false,
        }
    }

    /// Whether `token` is a valid next transition (no mutation).
    fn peek(&self, token: &str) -> bool {
        if self.rejected || self.closed {
            return false;
        }
        if !self.opened {
            return token == "[";
        }
        if self.machine.is_at_start() {
            // No element started yet: allow the closing `]` (empty array) or a
            // first `{`.
            return token == "]" || self.machine.peek(token);
        }
        if self.machine.is_complete() {
            return token == "," || token == "]";
        }
        self.machine.peek(token)
    }

    /// Commit `token`; `true` when accepted.
    fn feed(&mut self, token: &str) -> bool {
        if self.rejected || self.closed {
            self.rejected = true;
            return false;
        }
        if !self.opened {
            if token == "[" {
                self.opened = true;
                return true;
            }
            self.rejected = true;
            return false;
        }
        if self.machine.is_at_start() && token == "]" {
            self.closed = true;
            return true;
        }
        if self.machine.is_complete() {
            // Object boundary: `,` starts a fresh object, `]` finishes.
            if token == "," {
                self.machine.reset();
                return true;
            }
            if token == "]" {
                self.closed = true;
                return true;
            }
            self.rejected = true;
            return false;
        }
        if self.machine.feed(token) {
            true
        } else {
            self.rejected = true;
            false
        }
    }
}

impl Grammar for JsonArrayGrammar {
    fn reset(&mut self) {
        self.machine.reset();
        self.opened = false;
        self.closed = false;
        self.rejected = false;
    }

    fn allowed_ids(&self, vocab_size: usize) -> Vec<u32> {
        if self.rejected || self.closed {
            return Vec::new();
        }
        (0..vocab_size as u32)
            .filter(|id| self.vocab.token_text(*id).is_some_and(|t| self.peek(&t)))
            .collect()
    }

    fn advance(&mut self, token_id: u32) {
        let Some(text) = self.vocab.token_text(token_id) else {
            self.rejected = true;
            return;
        };
        self.feed(&text);
    }
}

/// Build a [`Grammar`] from a JSON-schema `Value` — the llama-fork
/// `response_format.schema` vocabulary the router's constrained callers speak.
///
/// - `{"type":"object",...}` → a [`JsonObjectGrammar`],
/// - `{"type":"array","items":{object}}` → a [`JsonArrayGrammar`] over the
///   item schema,
/// - anything else → `None` (free text; the caller's post-hoc validator remains
///   the backstop — the grammar is a strictness improvement, never a gate).
pub fn grammar_from_json_schema(
    schema: &serde_json::Value,
    vocab: Arc<dyn TokenVocab>,
) -> Option<Box<dyn Grammar>> {
    let ty = schema.get("type")?.as_str()?;
    match ty {
        "array" => {
            let item = schema.get("items")?;
            let item_schema = JsonSchema::from_json_schema(item)?;
            Some(Box::new(JsonArrayGrammar::new(item_schema, vocab)))
        }
        "object" => {
            let obj = JsonSchema::from_json_schema(schema)?;
            Some(Box::new(JsonObjectGrammar::new(obj, vocab)))
        }
        _ => None,
    }
}

/// Return the token ids whose decoded text equals `literal` exactly.
pub fn tokens_for_literal(vocab: &dyn TokenVocab, vocab_size: usize, literal: &str) -> Vec<u32> {
    (0..vocab_size as u32)
        .filter(|id| vocab.token_text(*id).as_deref() == Some(literal))
        .collect()
}

/// Whether `text` is a valid *prefix* of a JSON document — i.e. it is either a
/// complete JSON value or a truncated-but-never-malformed prefix of one.
///
/// This is the decodable guarantee the structural grammar provides: a
/// grammar-constrained decode never emits text that cannot be extended to a
/// valid JSON value. A complete value parses cleanly; an incomplete one fails
/// serde only because input ended (`Category::Eof`), never with a syntax error.
pub fn is_valid_json_prefix(text: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(_) => true,
        Err(e) => e.classify() == serde_json::error::Category::Eof,
    }
}

#[cfg(test)]
#[path = "../tests/grammar.rs"]
mod tests;
