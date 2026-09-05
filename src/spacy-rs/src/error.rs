//! Error taxonomy for the spaCy core.

use thiserror::Error;

/// Errors produced by the spacy-rs core: doc construction, array
/// round-trips, tree rebuilds, and label parsing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SpacyError {
    /// A dependency `head` relative offset points outside the document
    /// (`doc.pyx:1115-1128` bounds check).
    #[error(
        "invalid head: token {token} has relative head {head}, absolute index {abs} not in 0..{len}"
    )]
    HeadOutOfBounds {
        token: usize,
        head: i32,
        abs: i64,
        len: usize,
    },

    /// `from_array` row count does not match the doc token count.
    #[error("array length mismatch: array has {array} rows, doc has {doc} tokens")]
    ArrayLengthMismatch { array: usize, doc: usize },

    /// `from_array`/`EntIoB::from_id` got an IOB value ≥ 4.
    #[error("invalid entity IOB value: {0} (valid: 0..=3)")]
    InvalidEntIob(u64),

    /// `EntIoB::from_str` got a non-I, -O, -B, -empty string.
    #[error("invalid entity IOB text: {0:?}")]
    InvalidEntIobText(String),

    /// `Attribute::from_id` got an id with no known meaning.
    #[error("unknown attribute id: {0}")]
    UnknownAttribute(u16),

    /// `Attribute::from_name` got a string outside the attribute vocabulary.
    #[error("unknown attribute name: {0}")]
    UnknownAttributeText(String),

    /// `from_array`/`set_token_attr` tried to write an attribute that is
    /// derived at lexeme creation (flags, orth-derived strings, id).
    #[error("attribute {0} is read-only: computed at lexeme creation")]
    ReadOnlyAttribute(u16),

    /// `Upos::from_str` got a string outside the 17-tag vocabulary.
    #[error("unknown POS tag: {0}")]
    UnknownPos(String),

    /// `DepRel::from_str` got a string outside the reference set.
    #[error("unknown dependency label: {0}")]
    UnknownDepLabel(String),

    /// `NerType::from_str` got a string outside the entity-type vocabulary.
    #[error("unknown NER type: {0}")]
    UnknownNerType(String),

    /// The annotation validator rejected a token array.
    #[error("annotation rejected: {0}")]
    Annotation(String),

    /// The tokenizer got an input of ≥ 2^30 characters (`tokenizer.pyx:171`,
    /// `Errors.E025`).
    #[error("text of length {0} exceeds the tokenizer limit of 2^30 characters")]
    TextTooLong(usize),

    /// A tokenizer regex failed to compile or to run.
    #[error("tokenizer regex error: {0}")]
    Regex(String),

    /// A special-case rule failed validation: the concatenated ORTH tokens
    /// must equal the rule key, and only ORTH/NORM attrs are allowed
    /// (`tokenizer.pyx:_validate_special_case`, `Errors.E997/E1005`).
    #[error("invalid special case for {key:?}: {detail}")]
    SpecialCase { key: String, detail: String },

    /// A versioned lemma blob (`build.rs` output) failed to parse: bad
    /// magic/version, a truncated or out-of-range section, or invalid UTF-8.
    #[error("invalid lemma blob: {0}")]
    LemmaBlob(String),

    /// A versioned tagger-orthography blob (`build.rs` output) failed to
    /// parse: bad magic/version, truncation, framing drift, or an empty
    /// entry — the data bug must fail at load, never mis-parse.
    #[error("invalid orthography blob: {0}")]
    OrthoBlob(String),
}
