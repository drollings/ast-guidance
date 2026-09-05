//! The closed label vocabularies — fixed `repr` enums mirroring
//! `spacy/symbols.pxd` / `spacy/parts_of_speech.pxd`.
//!
//! These are the *canonical* label sets the walkthrough requires (§11.2): the
//! 17 UPOS tags (plus `NO_TAG`/`EOL`/`SPACE`), the reference dependency
//! relations, the NER entity types, and the IOB marker. Discriminants equal
//! the spaCy symbol ids so a `Doc::to_array`/`from_array` matrix round-trips
//! against real spaCy data.
//!
//! `Display` mirrors the spaCy `*_` accessors (`token.pos_` → `"noun"`,
//! `token.dep_` → `"nsubj"`, `token.ent_type_` → `"PERSON"`); `FromStr`
//! accepts any casing. Open vocabularies (orth, lemma, fine-grained tag, and
//! model-specific dep labels such as UD's `compound`/`case`) are *not* enums —
//! they are stored as `u64` hashes resolved through the vocabulary's
//! `StringStore`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::SpacyError;

/// Universal part-of-speech tag. Discriminants are the `symbol_t` ids from
/// `spacy/symbols.pxd` (which `univ_pos_t` aliases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum Upos {
    /// Unset / unknown tag; `pos_` renders as the empty string.
    #[serde(rename = "no_tag")]
    NoTag = 0,
    Adj = 84,
    Adp = 85,
    Adv = 86,
    Aux = 87,
    /// Deprecated alias of `CCONJ` (Universal Dependencies 2.0); kept for id
    /// parity but never produced by the validator.
    Conj = 88,
    Cconj = 89,
    Det = 90,
    Intj = 91,
    Noun = 92,
    Num = 93,
    Part = 94,
    Pron = 95,
    Propn = 96,
    Punct = 97,
    Sconj = 98,
    Sym = 99,
    Verb = 100,
    X = 101,
    /// Internal end-of-line tag; not part of the 17-tag contract.
    Eol = 102,
    /// Internal whitespace-token tag; not part of the 17-tag contract.
    Space = 103,
}

impl Upos {
    /// The 17 Universal Dependencies tags exposed to the LLM JSON contract.
    pub const UPOS: &'static [Self] = &[
        Self::Adj,
        Self::Adp,
        Self::Adv,
        Self::Aux,
        Self::Cconj,
        Self::Det,
        Self::Intj,
        Self::Noun,
        Self::Num,
        Self::Part,
        Self::Pron,
        Self::Propn,
        Self::Punct,
        Self::Sconj,
        Self::Sym,
        Self::Verb,
        Self::X,
    ];

    /// The label's `symbol_t` id.
    #[must_use]
    pub const fn id(self) -> u64 {
        self as u64
    }

    /// Reconstruct from a `symbol_t` id (the `to_array`/`from_array` value).
    pub fn from_id(value: u64) -> Result<Self, SpacyError> {
        match value {
            0 => Ok(Self::NoTag),
            84 => Ok(Self::Adj),
            85 => Ok(Self::Adp),
            86 => Ok(Self::Adv),
            87 => Ok(Self::Aux),
            88 => Ok(Self::Conj),
            89 => Ok(Self::Cconj),
            90 => Ok(Self::Det),
            91 => Ok(Self::Intj),
            92 => Ok(Self::Noun),
            93 => Ok(Self::Num),
            94 => Ok(Self::Part),
            95 => Ok(Self::Pron),
            96 => Ok(Self::Propn),
            97 => Ok(Self::Punct),
            98 => Ok(Self::Sconj),
            99 => Ok(Self::Sym),
            100 => Ok(Self::Verb),
            101 => Ok(Self::X),
            102 => Ok(Self::Eol),
            103 => Ok(Self::Space),
            other => Err(SpacyError::UnknownPos(other.to_string())),
        }
    }

    /// The lemma-blob table key for this tag: the same lowercase label
    /// [`Display`](fmt::Display) renders, as a `&'static str` with no
    /// allocation. Single source of truth — the lemmatizer matches on the
    /// enum and keys the blob through this, never through `to_string()`
    /// plus string literals (typo-impossible, zero-cost on the hot path).
    #[must_use]
    pub const fn lemma_key(self) -> &'static str {
        match self {
            Self::NoTag => "",
            Self::Adj => "adj",
            Self::Adp => "adp",
            Self::Adv => "adv",
            Self::Aux => "aux",
            Self::Conj => "conj",
            Self::Cconj => "cconj",
            Self::Det => "det",
            Self::Intj => "intj",
            Self::Noun => "noun",
            Self::Num => "num",
            Self::Part => "part",
            Self::Pron => "pron",
            Self::Propn => "propn",
            Self::Punct => "punct",
            Self::Sconj => "sconj",
            Self::Sym => "sym",
            Self::Verb => "verb",
            Self::X => "x",
            Self::Eol => "eol",
            Self::Space => "space",
        }
    }
}

impl fmt::Display for Upos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NoTag => "",
            Self::Adj => "adj",
            Self::Adp => "adp",
            Self::Adv => "adv",
            Self::Aux => "aux",
            Self::Conj => "conj",
            Self::Cconj => "cconj",
            Self::Det => "det",
            Self::Intj => "intj",
            Self::Noun => "noun",
            Self::Num => "num",
            Self::Part => "part",
            Self::Pron => "pron",
            Self::Propn => "propn",
            Self::Punct => "punct",
            Self::Sconj => "sconj",
            Self::Sym => "sym",
            Self::Verb => "verb",
            Self::X => "x",
            Self::Eol => "eol",
            Self::Space => "space",
        };
        f.write_str(s)
    }
}

impl FromStr for Upos {
    type Err = SpacyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let tag = match lower.as_str() {
            "" => Self::NoTag,
            "adj" => Self::Adj,
            "adp" => Self::Adp,
            "adv" => Self::Adv,
            "aux" => Self::Aux,
            "conj" => Self::Conj,
            "cconj" => Self::Cconj,
            "det" => Self::Det,
            "intj" => Self::Intj,
            "noun" => Self::Noun,
            "num" => Self::Num,
            "part" => Self::Part,
            "pron" => Self::Pron,
            "propn" => Self::Propn,
            "punct" => Self::Punct,
            "sconj" => Self::Sconj,
            "sym" => Self::Sym,
            "verb" => Self::Verb,
            "x" => Self::X,
            "eol" => Self::Eol,
            "space" => Self::Space,
            other => return Err(SpacyError::UnknownPos(other.to_string())),
        };
        Ok(tag)
    }
}

/// Named-entity IOB marker, matching spaCy's `IOB_STRINGS = ("", "I", "O", "B")`
/// (`spacy/attrs.pyx:4`). The transition parser works in BILUO internally, but
/// the stored `ent_iob` is classic IOB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EntIoB {
    /// No entity annotation.
    Missing = 0,
    Inside = 1,
    Outside = 2,
    Begin = 3,
}

impl EntIoB {
    /// The id as stored in `TokenC.ent_iob` and exported by `to_array`.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Construct from the stored integer; rejects values ≥ 4 (the
    /// `from_array` bound, `doc.pyx:1130-1142`).
    pub const fn from_id(value: u8) -> Result<Self, SpacyError> {
        match value {
            0 => Ok(Self::Missing),
            1 => Ok(Self::Inside),
            2 => Ok(Self::Outside),
            3 => Ok(Self::Begin),
            other => Err(SpacyError::InvalidEntIob(other as u64)),
        }
    }
}

impl fmt::Display for EntIoB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Missing => "",
            Self::Inside => "I",
            Self::Outside => "O",
            Self::Begin => "B",
        };
        f.write_str(s)
    }
}

impl FromStr for EntIoB {
    type Err = SpacyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(Self::Missing),
            "I" => Ok(Self::Inside),
            "O" => Ok(Self::Outside),
            "B" => Ok(Self::Begin),
            other => Err(SpacyError::InvalidEntIobText(other.to_string())),
        }
    }
}

/// Named-entity type. Discriminants are the `symbol_t` ids for `PERSON` …
/// `CARDINAL` (`spacy/symbols.pxd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
#[serde(rename_all = "UPPERCASE")]
pub enum NerType {
    Person = 380,
    Norp = 381,
    Facility = 382,
    Org = 383,
    Gpe = 384,
    Loc = 385,
    Product = 386,
    Event = 387,
    WorkOfArt = 388,
    Language = 389,
    Law = 390,
    Date = 391,
    Time = 392,
    Percent = 393,
    Money = 394,
    Quantity = 395,
    Ordinal = 396,
    Cardinal = 397,
}

impl NerType {
    /// The `symbol_t` id.
    #[must_use]
    pub const fn id(self) -> u64 {
        self as u64
    }
}

impl fmt::Display for NerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Person => "PERSON",
            Self::Norp => "NORP",
            Self::Facility => "FACILITY",
            Self::Org => "ORG",
            Self::Gpe => "GPE",
            Self::Loc => "LOC",
            Self::Product => "PRODUCT",
            Self::Event => "EVENT",
            Self::WorkOfArt => "WORK_OF_ART",
            Self::Language => "LANGUAGE",
            Self::Law => "LAW",
            Self::Date => "DATE",
            Self::Time => "TIME",
            Self::Percent => "PERCENT",
            Self::Money => "MONEY",
            Self::Quantity => "QUANTITY",
            Self::Ordinal => "ORDINAL",
            Self::Cardinal => "CARDINAL",
        };
        f.write_str(s)
    }
}

impl FromStr for NerType {
    type Err = SpacyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let upper = s.to_ascii_uppercase();
        let t = match upper.as_str() {
            "PERSON" => Self::Person,
            "NORP" => Self::Norp,
            "FACILITY" => Self::Facility,
            "ORG" => Self::Org,
            "GPE" => Self::Gpe,
            "LOC" => Self::Loc,
            "PRODUCT" => Self::Product,
            "EVENT" => Self::Event,
            "WORK_OF_ART" => Self::WorkOfArt,
            "LANGUAGE" => Self::Language,
            "LAW" => Self::Law,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "PERCENT" => Self::Percent,
            "MONEY" => Self::Money,
            "QUANTITY" => Self::Quantity,
            "ORDINAL" => Self::Ordinal,
            "CARDINAL" => Self::Cardinal,
            other => return Err(SpacyError::UnknownNerType(other.to_string())),
        };
        Ok(t)
    }
}

/// Canonical dependency relation. Discriminants are the `symbol_t` ids for
/// `acomp` … `acl` (`spacy/symbols.pxd`), plus the modern Universal-Dependency
/// labels that a current model actually emits (`compound`, `case`, `flat`, …)
/// which have **no** spaCy symbol id — those get ids in the reserved
/// 2000+ range (the stored `dep` field is always the content hash, so the
/// numeric id is only for `to_array`/`from_array` interop on the symbol set).
/// The validator accepts open labels via [`crate::labels::DepLabelSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
#[serde(rename_all = "lowercase")]
pub enum DepRel {
    Acomp = 398,
    Advcl = 399,
    Advmod = 400,
    Agent = 401,
    Amod = 402,
    Appos = 403,
    Attr = 404,
    Aux = 405,
    Auxpass = 406,
    Cc = 407,
    Ccomp = 408,
    Complm = 409,
    Conj = 410,
    Cop = 411,
    Csubj = 412,
    Csubjpass = 413,
    Dep = 414,
    Det = 415,
    Dobj = 416,
    Expl = 417,
    Hmod = 418,
    Hyph = 419,
    Infmod = 420,
    Intj = 421,
    Iobj = 422,
    Mark = 423,
    Meta = 424,
    Neg = 425,
    Nmod = 426,
    Nn = 427,
    Npadvmod = 428,
    Nsubj = 429,
    Nsubjpass = 430,
    Num = 431,
    Number = 432,
    Oprd = 433,
    Obj = 434,
    Obl = 435,
    Parataxis = 436,
    Partmod = 437,
    Pcomp = 438,
    Pobj = 439,
    Poss = 440,
    Possessive = 441,
    Preconj = 442,
    Prep = 443,
    Prt = 444,
    Punct = 445,
    Quantmod = 446,
    Relcl = 447,
    Rcmod = 448,
    Root = 449,
    Xcomp = 450,
    Acl = 451,
    // ── Modern UD labels (no spaCy symbol id; reserved 2000+ range) ──
    Compound = 2000,
    Case = 2001,
    Fixed = 2002,
    Flat = 2003,
    Discourse = 2004,
    Dislocated = 2005,
    Goeswith = 2006,
    List = 2007,
    Mixed = 2008,
    Nummod = 2009,
    Orphan = 2010,
    Reparandum = 2011,
    Vocative = 2012,
}

impl DepRel {
    /// The `symbol_t` id.
    #[must_use]
    pub const fn id(self) -> u64 {
        self as u64
    }
}

impl fmt::Display for DepRel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Acomp => "acomp",
            Self::Advcl => "advcl",
            Self::Advmod => "advmod",
            Self::Agent => "agent",
            Self::Amod => "amod",
            Self::Appos => "appos",
            Self::Attr => "attr",
            Self::Aux => "aux",
            Self::Auxpass => "auxpass",
            Self::Cc => "cc",
            Self::Ccomp => "ccomp",
            Self::Complm => "complm",
            Self::Conj => "conj",
            Self::Cop => "cop",
            Self::Csubj => "csubj",
            Self::Csubjpass => "csubjpass",
            Self::Dep => "dep",
            Self::Det => "det",
            Self::Dobj => "dobj",
            Self::Expl => "expl",
            Self::Hmod => "hmod",
            Self::Hyph => "hyph",
            Self::Infmod => "infmod",
            Self::Intj => "intj",
            Self::Iobj => "iobj",
            Self::Mark => "mark",
            Self::Meta => "meta",
            Self::Neg => "neg",
            Self::Nmod => "nmod",
            Self::Nn => "nn",
            Self::Npadvmod => "npadvmod",
            Self::Nsubj => "nsubj",
            Self::Nsubjpass => "nsubjpass",
            Self::Num => "num",
            Self::Number => "number",
            Self::Oprd => "oprd",
            Self::Obj => "obj",
            Self::Obl => "obl",
            Self::Parataxis => "parataxis",
            Self::Partmod => "partmod",
            Self::Pcomp => "pcomp",
            Self::Pobj => "pobj",
            Self::Poss => "poss",
            Self::Possessive => "possessive",
            Self::Preconj => "preconj",
            Self::Prep => "prep",
            Self::Prt => "prt",
            Self::Punct => "punct",
            Self::Quantmod => "quantmod",
            Self::Relcl => "relcl",
            Self::Rcmod => "rcmod",
            Self::Root => "root",
            Self::Xcomp => "xcomp",
            Self::Acl => "acl",
            Self::Compound => "compound",
            Self::Case => "case",
            Self::Fixed => "fixed",
            Self::Flat => "flat",
            Self::Discourse => "discourse",
            Self::Dislocated => "dislocated",
            Self::Goeswith => "goeswith",
            Self::List => "list",
            Self::Mixed => "mixed",
            Self::Nummod => "nummod",
            Self::Orphan => "orphan",
            Self::Reparandum => "reparandum",
            Self::Vocative => "vocative",
        };
        f.write_str(s)
    }
}

impl FromStr for DepRel {
    type Err = SpacyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let rel = match lower.as_str() {
            "acomp" => Self::Acomp,
            "advcl" => Self::Advcl,
            "advmod" => Self::Advmod,
            "agent" => Self::Agent,
            "amod" => Self::Amod,
            "appos" => Self::Appos,
            "attr" => Self::Attr,
            "aux" => Self::Aux,
            "auxpass" => Self::Auxpass,
            "cc" => Self::Cc,
            "ccomp" => Self::Ccomp,
            "complm" => Self::Complm,
            "conj" => Self::Conj,
            "cop" => Self::Cop,
            "csubj" => Self::Csubj,
            "csubjpass" => Self::Csubjpass,
            "dep" => Self::Dep,
            "det" => Self::Det,
            "dobj" => Self::Dobj,
            "expl" => Self::Expl,
            "hmod" => Self::Hmod,
            "hyph" => Self::Hyph,
            "infmod" => Self::Infmod,
            "intj" => Self::Intj,
            "iobj" => Self::Iobj,
            "mark" => Self::Mark,
            "meta" => Self::Meta,
            "neg" => Self::Neg,
            "nmod" => Self::Nmod,
            "nn" => Self::Nn,
            "npadvmod" => Self::Npadvmod,
            "nsubj" => Self::Nsubj,
            "nsubjpass" => Self::Nsubjpass,
            "num" => Self::Num,
            "number" => Self::Number,
            "oprd" => Self::Oprd,
            "obj" => Self::Obj,
            "obl" => Self::Obl,
            "parataxis" => Self::Parataxis,
            "partmod" => Self::Partmod,
            "pcomp" => Self::Pcomp,
            "pobj" => Self::Pobj,
            "poss" => Self::Poss,
            "possessive" => Self::Possessive,
            "preconj" => Self::Preconj,
            "prep" => Self::Prep,
            "prt" => Self::Prt,
            "punct" => Self::Punct,
            "quantmod" => Self::Quantmod,
            "relcl" => Self::Relcl,
            "rcmod" => Self::Rcmod,
            "root" => Self::Root,
            "xcomp" => Self::Xcomp,
            "acl" => Self::Acl,
            "compound" => Self::Compound,
            "case" => Self::Case,
            "fixed" => Self::Fixed,
            "flat" => Self::Flat,
            "discourse" => Self::Discourse,
            "dislocated" => Self::Dislocated,
            "goeswith" => Self::Goeswith,
            "list" => Self::List,
            "mixed" => Self::Mixed,
            "nummod" => Self::Nummod,
            "orphan" => Self::Orphan,
            "reparandum" => Self::Reparandum,
            "vocative" => Self::Vocative,
            other => return Err(SpacyError::UnknownDepLabel(other.to_string())),
        };
        Ok(rel)
    }
}

/// A configurable set of dependency labels accepted by the annotation
/// validator (§10.2 check 2). The default starts from the canonical
/// [`DepRel`] reference set and adds the Universal-Dependencies labels a
/// modern `en_core_web_sm` actually emits (`compound`, `case`, `flat`, …),
/// so a finetuned model is not rejected for using current UD labels while
/// still failing closed on garbage.
///
/// The set is serde-able (serialized as the label list) and round-trips
/// through [`FromStr`]/[`Display`] (comma-joined), so a model's `label_data`
/// can override the default accepted set (§10.6, roadmap §5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DepLabelSet {
    labels: std::collections::HashSet<String>,
}

impl DepLabelSet {
    /// The canonical reference set plus the UD labels from `en_core_web_sm`
    /// v3.8's tag map (observed label set).
    #[must_use]
    pub fn ud_default() -> Self {
        let mut set = Self::default();
        for rel in [
            "acomp",
            "advcl",
            "advmod",
            "agent",
            "amod",
            "appos",
            "attr",
            "aux",
            "auxpass",
            "case",
            "cc",
            "ccomp",
            "complm",
            "compound",
            "conj",
            "cop",
            "csubj",
            "csubjpass",
            "dep",
            "det",
            "discourse",
            "dislocated",
            "dobj",
            "expl",
            "fixed",
            "flat",
            "goeswith",
            "hmod",
            "hyph",
            "infmod",
            "intj",
            "iobj",
            "list",
            "mark",
            "meta",
            "mixed",
            "neg",
            "nmod",
            "nn",
            "npadvmod",
            "nsubj",
            "nsubjpass",
            "num",
            "number",
            "nummod",
            "obj",
            "obl",
            "oprd",
            "orphan",
            "parataxis",
            "partmod",
            "pcomp",
            "pobj",
            "poss",
            "possessive",
            "preconj",
            "prep",
            "prt",
            "punct",
            "quantmod",
            "rcmod",
            "reparandum",
            "relcl",
            "root",
            "vocative",
            "xcomp",
        ] {
            set.insert(rel.to_string());
        }
        set
    }

    /// Add a label to the accepted set.
    pub fn insert(&mut self, label: impl Into<String>) {
        self.labels.insert(label.into());
    }

    /// Whether `label` is accepted (case-insensitive).
    #[must_use]
    pub fn contains(&self, label: &str) -> bool {
        self.labels.contains(&label.to_ascii_lowercase())
    }

    /// Number of accepted labels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The accepted labels, sorted for determinism.
    #[must_use]
    pub fn to_sorted_vec(&self) -> Vec<String> {
        let mut labels: Vec<String> = self.labels.iter().cloned().collect();
        labels.sort();
        labels
    }
}

impl fmt::Display for DepLabelSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_sorted_vec().join(","))
    }
}

impl FromStr for DepLabelSet {
    type Err = SpacyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut set = Self::default();
        for label in s.split(',').map(str::trim).filter(|l| !l.is_empty()) {
            // Reject garbage eagerly: every accepted label must resolve to the
            // canonical reference set or a known UD label.
            let parsed: DepRel = label.parse()?;
            set.insert(parsed.to_string());
        }
        Ok(set)
    }
}

#[cfg(test)]
#[path = "../tests/labels.rs"]
mod tests;
