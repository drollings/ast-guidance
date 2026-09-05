//! The doc model: contiguous `TokenRecord` array, the shared canvas every
//! pipeline component reads and mutates — spaCy's `Doc` / `TokenC` /
//! `set_children_from_heads` (`spacy/tokens/doc.pyx`, `spacy/structs.pxd`).
//!
//! Design points preserved from spaCy:
//! - **Two-level model**: each token references an `Arc<Lexeme>` word-type;
//!   context lives in the record itself.
//! - **`head` is a signed relative offset** from self (`token.i + head ==
//!   head_index`), making the token array relocatable/serializable.
//! - **`l_edge`/`r_edge` bounds** give O(children) tree traversal.
//! - `to_array`/`from_array` is the canonical `(n_tokens, n_attrs)` u64 matrix
//!   contract with head-bounds and IOB validation.
//!
//! Safe Rust deliberately replaces the padded raw pointer array with a plain
//! `Vec` and bounds-checked accessors — the *semantics* (windowed reads, no
//! segfaults on feature windows) are preserved, not the pointer trick.

use internment::ArcIntern;
use std::collections::HashMap;
use std::sync::Arc;

use crate::attrs::Attribute;
use crate::error::SpacyError;
use crate::labels::{EntIoB, Upos};
use crate::lexeme::Lexeme;
use crate::vocab::Vocab;
use fluent_types::InterlinguaId;

/// Sentence-boundary tri-state, matching `TokenC.sent_start`
/// (`structs.pxd:53`): 1 = sentence start, -1 = not a start, 0 = unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i8)]
pub enum SentStart {
    NotStart = -1,
    #[default]
    Unset = 0,
    Start = 1,
}

impl From<i8> for SentStart {
    fn from(value: i8) -> Self {
        match value {
            -1 => Self::NotStart,
            1 => Self::Start,
            _ => Self::Unset,
        }
    }
}

/// A word-token record — field-for-field against `TokenC`
/// (`spacy/structs.pxd:36-57`).
#[derive(Debug, Clone)]
pub struct TokenRecord {
    /// Pointer to the shared word-type.
    pub lexeme: Arc<Lexeme>,
    /// Hash key into the morphology table.
    pub morph: u64,
    /// Coarse UPOS tag.
    pub pos: Upos,
    /// Had a trailing space.
    pub spacy: bool,
    /// Fine-grained tag hash (e.g. `"NNP"`).
    pub tag: u64,
    /// Character offset of the token start in the doc.
    pub idx: u32,
    /// Lemma hash.
    pub lemma: u64,
    /// Token-level norm override; `0` means "use the lexeme norm".
    pub norm: u64,
    /// Relative signed head offset (`self + head == head_index`).
    pub head: i32,
    /// Dependency label hash.
    pub dep: u64,
    /// Count of left children.
    pub l_kids: u32,
    /// Count of right children.
    pub r_kids: u32,
    /// Leftmost descendant index.
    pub l_edge: u32,
    /// Rightmost descendant index.
    pub r_edge: u32,
    /// Sentence-boundary marker.
    pub sent_start: SentStart,
    /// IOB entity marker.
    pub ent_iob: EntIoB,
    /// Entity type hash.
    pub ent_type: u64,
    /// Knowledge-base entity id hash.
    pub ent_kb_id: u64,
    /// Entity id hash.
    pub ent_id: u64,
    /// The token lemma's interlingua id, stamped by the resolver (ROADMAP
    /// §10.1 — additive, `None` until the resolve stage runs).
    pub interlingua_lemma_id: Option<InterlinguaId>,
    /// The token lemma's entity interlingua id (PROPN tokens with a YaGO
    /// match), stamped by the resolver.
    pub interlingua_entity_id: Option<InterlinguaId>,
    /// Per-token parse confidence (the ArcEager rung fills it; LLM/rule set
    /// None). Gates downstream routing, never rung fallthrough (F7).
    pub confidence: Option<f64>,
}

impl TokenRecord {
    /// A record referencing `lexeme` with all context fields zeroed/unset.
    #[must_use]
    pub fn new(lexeme: Arc<Lexeme>) -> Self {
        Self {
            lexeme,
            morph: 0,
            pos: Upos::NoTag,
            spacy: false,
            tag: 0,
            idx: 0,
            lemma: 0,
            norm: 0,
            head: 0,
            dep: 0,
            l_kids: 0,
            r_kids: 0,
            l_edge: 0,
            r_edge: 0,
            sent_start: SentStart::Unset,
            ent_iob: EntIoB::Missing,
            ent_type: 0,
            ent_kb_id: 0,
            ent_id: 0,
            interlingua_lemma_id: None,
            interlingua_entity_id: None,
            confidence: None,
        }
    }

    /// Whether this token has an annotated head (`head` unset when
    /// `dep == 0`, the `MISSING_DEP` convention, `token.pxd:14,103`).
    #[must_use]
    pub fn has_head(&self) -> bool {
        self.dep != 0
    }
}

impl Default for TokenRecord {
    fn default() -> Self {
        Self::new(Lexeme::empty())
    }
}

/// The shared mutable canvas: a vocabulary reference plus a contiguous token
/// array.
#[derive(Debug, Clone)]
pub struct Doc {
    vocab: Arc<Vocab>,
    tokens: Vec<TokenRecord>,
    cats: HashMap<ArcIntern<str>, f32>,
}

impl Doc {
    /// An empty doc over `vocab`.
    #[must_use]
    pub fn new(vocab: Arc<Vocab>) -> Self {
        Self {
            vocab,
            tokens: Vec::new(),
            cats: HashMap::new(),
        }
    }

    /// The doc's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// The token array.
    #[must_use]
    pub fn tokens(&self) -> &[TokenRecord] {
        &self.tokens
    }

    /// Mutable access to the token vector — the tokenizer's special-case
    /// splice path.
    pub fn tokens_mut(&mut self) -> &mut Vec<TokenRecord> {
        &mut self.tokens
    }

    /// Token `i`.
    #[must_use]
    pub fn token(&self, i: usize) -> &TokenRecord {
        &self.tokens[i]
    }

    /// Mutable token `i`.
    #[must_use]
    pub fn token_mut(&mut self, i: usize) -> &mut TokenRecord {
        &mut self.tokens[i]
    }

    /// Token count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the doc has no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Append a token for `text` (creating its lexeme), with the `spacy`
    /// trailing-space flag. Computes `idx` from the previous token as
    /// `prev.idx + lex.length + spacy` (`doc.pyx:948-969`). Rejects the empty
    /// string (orth 0). Returns the new token count.
    pub fn push_back(&mut self, text: &str, has_space: bool) -> Result<usize, SpacyError> {
        if text.is_empty() {
            return Err(SpacyError::Annotation(
                "cannot push an empty token (orth 0)".into(),
            ));
        }
        let lexeme = self.vocab.lexicon().get_or_create(text);
        Ok(self.push_lexeme(lexeme, has_space))
    }

    /// Append a token referencing an existing lexeme, with the `spacy` flag.
    /// Returns the new token count.
    pub fn push_lexeme(&mut self, lexeme: Arc<Lexeme>, has_space: bool) -> usize {
        let i = self.tokens.len();
        let idx = if i == 0 {
            0
        } else {
            let prev = &self.tokens[i - 1];
            prev.idx + prev.lexeme.length + u32::from(prev.spacy)
        };
        let mut record = TokenRecord::new(lexeme);
        record.idx = idx;
        record.l_edge = i as u32;
        record.r_edge = i as u32;
        record.spacy = has_space;
        self.tokens.push(record);
        if i == 0 {
            self.tokens[0].sent_start = SentStart::Start;
        }
        i + 1
    }

    /// Append a full token record, recomputing only the position bookkeeping
    /// (`idx`, `l_edge`/`r_edge`, `spacy`) exactly as spaCy's `push_back`
    /// does (`doc.pyx:948-969`) while preserving the record's other fields
    /// (norm overrides, POS, dep, ...). This is the tokenizer's path for
    /// special-case tokens that carry per-token attributes.
    pub fn push_record(&mut self, mut record: TokenRecord, has_space: bool) -> usize {
        let i = self.tokens.len();
        let idx = if i == 0 {
            0
        } else {
            let prev = &self.tokens[i - 1];
            prev.idx + prev.lexeme.length + u32::from(prev.spacy)
        };
        record.idx = idx;
        record.l_edge = i as u32;
        record.r_edge = i as u32;
        record.spacy = has_space;
        self.tokens.push(record);
        if i == 0 {
            self.tokens[0].sent_start = SentStart::Start;
        }
        i + 1
    }

    /// The reconstructed doc text: each token's orth plus `" "` where its
    /// `spacy` flag is set (the last token's flag is authoritative, so a
    /// trailing space in the source is preserved).
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        let strings = self.vocab.strings();
        for token in &self.tokens {
            out.push_str(&token.lexeme.orth_text(strings));
            if token.spacy {
                out.push(' ');
            }
        }
        out
    }

    /// The resolved orth text of token `i`.
    #[must_use]
    pub fn token_text(&self, i: usize) -> String {
        self.tokens[i].lexeme.orth_text(self.vocab.strings())
    }

    // ── Dependency-tree navigation (Token.lefts/rights/ancestors/…) ──

    /// The absolute head index of token `i`; itself when the head is missing.
    #[must_use]
    pub fn head_index(&self, i: usize) -> usize {
        let token = &self.tokens[i];
        if token.has_head() {
            (i as i64 + i64::from(token.head)) as usize
        } else {
            i
        }
    }

    /// Whether token `i` has an annotated head.
    #[must_use]
    pub fn has_head(&self, i: usize) -> bool {
        self.tokens[i].has_head()
    }

    /// Immediate left children of token `i`, in order
    /// (`token.pyx:569-587`).
    #[must_use]
    pub fn lefts(&self, i: usize) -> Vec<usize> {
        let l_edge = self.tokens[i].l_edge as usize;
        (l_edge..i).filter(|&j| self.head_index(j) == i).collect()
    }

    /// Immediate right children of token `i`, in order (`token.pyx:589-610`).
    #[must_use]
    pub fn rights(&self, i: usize) -> Vec<usize> {
        let r_edge = self.tokens[i].r_edge as usize;
        (i + 1..=r_edge)
            .rev()
            .filter(|&j| self.head_index(j) == i)
            .collect()
    }

    /// Immediate children of token `i` (lefts then rights).
    #[must_use]
    pub fn children(&self, i: usize) -> Vec<usize> {
        let mut children = self.lefts(i);
        children.extend(self.rights(i));
        children
    }

    /// The leftmost descendant index of token `i`.
    #[must_use]
    pub fn left_edge(&self, i: usize) -> usize {
        self.tokens[i].l_edge as usize
    }

    /// The rightmost descendant index of token `i`.
    #[must_use]
    pub fn right_edge(&self, i: usize) -> usize {
        self.tokens[i].r_edge as usize
    }

    /// Ancestors of token `i`, nearest first, terminating at ROOT
    /// (`token.pyx:655-671`).
    #[must_use]
    pub fn ancestors(&self, i: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = i;
        let mut guard = 0;
        while guard < self.tokens.len() {
            let token = &self.tokens[cur];
            if token.head == 0 || !token.has_head() {
                break;
            }
            let head = (cur as i64 + i64::from(token.head)) as usize;
            out.push(head);
            cur = head;
            guard += 1;
        }
        out
    }

    /// The token and all its descendants (`token.pyx:623-636`).
    #[must_use]
    pub fn subtree(&self, i: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for left in self.lefts(i) {
            out.extend(self.subtree(left));
        }
        out.push(i);
        for right in self.rights(i) {
            out.extend(self.subtree(right));
        }
        out
    }

    /// Whether `i` is an ancestor of `j`.
    #[must_use]
    pub fn is_ancestor(&self, i: usize, j: usize) -> bool {
        self.ancestors(j).contains(&i)
    }

    // ── Attribute dispatch / serialization contract ──

    /// `Token.get_struct_attr` dispatch for one token (`token.pxd:42-63`).
    pub fn get_token_attr(&self, i: usize, attr: Attribute) -> Result<u64, SpacyError> {
        get_token_attr(&self.tokens[i], attr)
    }

    /// Export an `(n_tokens, n_attrs)` matrix, `doc.pyx:971-1023`.
    pub fn to_array(&self, attrs: &[Attribute]) -> Result<Vec<Vec<u64>>, SpacyError> {
        self.tokens
            .iter()
            .map(|token| attrs.iter().map(|&a| get_token_attr(token, a)).collect())
            .collect()
    }

    /// Load attributes from an `(M, N)` matrix with spaCy's validation
    /// (`doc.pyx:1070-1157`): row count must match; `HEAD` offsets must stay
    /// in bounds; `ENT_IOB` must be 0..=3; and when both `HEAD` and `DEP` are
    /// present with any dep set, the child/edge indexes are rebuilt.
    pub fn from_array(
        &mut self,
        attrs: &[Attribute],
        array: &[Vec<u64>],
    ) -> Result<(), SpacyError> {
        if array.len() != self.tokens.len() {
            return Err(SpacyError::ArrayLengthMismatch {
                array: array.len(),
                doc: self.tokens.len(),
            });
        }

        if let Some(col) = attrs.iter().position(|&a| a == Attribute::Head) {
            for (i, row) in array.iter().enumerate() {
                // spaCy casts the stored value to int32 before adding `i`.
                let head = i64::from(row[col] as i32);
                let abs = i as i64 + head;
                if abs < 0 || abs >= self.tokens.len() as i64 {
                    return Err(SpacyError::HeadOutOfBounds {
                        token: i,
                        head: head as i32,
                        abs,
                        len: self.tokens.len(),
                    });
                }
            }
        }

        if let Some(col) = attrs.iter().position(|&a| a == Attribute::EntIob) {
            for row in array {
                let value = row[col];
                if value >= 4 {
                    return Err(SpacyError::InvalidEntIob(value));
                }
            }
        }

        for (i, row) in array.iter().enumerate() {
            for (j, &attr) in attrs.iter().enumerate() {
                set_token_attr(&mut self.tokens[i], attr, row[j])?;
            }
        }

        if attrs.contains(&Attribute::Head) && attrs.contains(&Attribute::Dep) {
            let has_dep = self.tokens.iter().any(|t| t.dep != 0);
            if has_dep {
                set_children_from_heads(&mut self.tokens)?;
            }
        }
        Ok(())
    }

    /// Rebuild `l_kids`/`r_kids`/`l_edge`/`r_edge` and sentence starts from
    /// the `head`/`dep` arrays (`doc.pyx:1815-1834`).
    pub fn set_children_from_heads(&mut self) -> Result<(), SpacyError> {
        set_children_from_heads(&mut self.tokens)
    }

    /// Text-category scores (`doc.cats`).
    #[must_use]
    pub fn cats(&self) -> &HashMap<ArcIntern<str>, f32> {
        &self.cats
    }

    /// Set the text-category score for `label`.
    pub fn set_cat(&mut self, label: impl Into<ArcIntern<str>>, score: f32) {
        self.cats.insert(label.into(), score);
    }
}

/// The single attribute dispatch (`token.pxd:42-63`).
pub fn get_token_attr(token: &TokenRecord, attr: Attribute) -> Result<u64, SpacyError> {
    let value = match attr {
        // Boolean lexeme flags: bit test on the shared word-type.
        Attribute::IsAlpha
        | Attribute::IsAscii
        | Attribute::IsDigit
        | Attribute::IsLower
        | Attribute::IsPunct
        | Attribute::IsSpace
        | Attribute::IsTitle
        | Attribute::IsUpper
        | Attribute::LikeUrl
        | Attribute::LikeNum
        | Attribute::LikeEmail
        | Attribute::IsStop
        | Attribute::IsOovDeprecated
        | Attribute::IsBracket
        | Attribute::IsQuote
        | Attribute::IsLeftPunct
        | Attribute::IsRightPunct
        | Attribute::IsCurrency
        | Attribute::IsDetWord
        | Attribute::IsAdpWord
        | Attribute::IsAuxWord
        | Attribute::IsCconjWord
        | Attribute::IsSconjWord
        | Attribute::IsPronWord
        | Attribute::IsVerbWord
        | Attribute::IsBeVerb
        | Attribute::IsBareInfHost
        | Attribute::IsNegator
        | Attribute::IsNominative
        | Attribute::IsPossessive
        | Attribute::IsRelativizer
        | Attribute::IsSensoryVerb
        | Attribute::IsEpistemicVerb
        | Attribute::IsDiscourseMarker
        | Attribute::IsAdverbWord
        | Attribute::IsSubordComplement
        | Attribute::IsSubordAdverbial
        | Attribute::IsWhereWord
        | Attribute::IsLocative
        | Attribute::IsDemonstrative
        | Attribute::IsTodayWord
        | Attribute::IsAsWord
        | Attribute::IsAfterWord
        | Attribute::IsThatWord
        | Attribute::IsTwiceWord
        | Attribute::IsYetWord
        | Attribute::IsPleaseWord
        | Attribute::IsBeCliticS
        | Attribute::IsBeClitic
        | Attribute::IsThereWord => u64::from(token.lexeme.flags.check(attr.id())),
        Attribute::Id => token.lexeme.id,
        Attribute::Orth => token.lexeme.orth,
        Attribute::Lower => token.lexeme.lower,
        Attribute::Norm => {
            if token.norm == 0 {
                token.lexeme.norm
            } else {
                token.norm
            }
        }
        Attribute::Shape => token.lexeme.shape,
        Attribute::Prefix => token.lexeme.prefix,
        Attribute::Suffix => token.lexeme.suffix,
        Attribute::Length => u64::from(token.lexeme.length),
        Attribute::Cluster | Attribute::Prob | Attribute::SentEnd => 0,
        Attribute::Lemma => token.lemma,
        Attribute::Pos => token.pos.id(),
        Attribute::Tag => token.tag,
        Attribute::Dep => token.dep,
        Attribute::EntIob => u64::from(token.ent_iob.id()),
        Attribute::EntType => token.ent_type,
        Attribute::Head => token.head as u64,
        Attribute::SentStart => token.sent_start as u64,
        Attribute::Spacy => u64::from(token.spacy),
        Attribute::Lang => token.lexeme.lang,
        Attribute::EntKbId => token.ent_kb_id,
        Attribute::Morph => token.morph,
        Attribute::EntId => token.ent_id,
        Attribute::Idx => u64::from(token.idx),
        Attribute::InterlinguaLemmaId => token.interlingua_lemma_id.map_or(0, InterlinguaId::as_u64),
        Attribute::InterlinguaEntityId => token.interlingua_entity_id.map_or(0, InterlinguaId::as_u64),
        Attribute::AnnotationConfidence => token.confidence.map_or(0, f64::to_bits),
        Attribute::Other(_) => return Err(SpacyError::UnknownAttribute(attr.id())),
    };
    Ok(value)
}

/// Write one attribute from a matrix cell (`token.pxd:72-96`). Only context
/// attributes are writable; lexeme-derived and flag attributes are read-only.
pub fn set_token_attr(
    token: &mut TokenRecord,
    attr: Attribute,
    value: u64,
) -> Result<(), SpacyError> {
    match attr {
        Attribute::Lemma => token.lemma = value,
        Attribute::Pos => token.pos = Upos::from_id(value)?,
        Attribute::Tag => token.tag = value,
        Attribute::Dep => token.dep = value,
        Attribute::EntIob => {
            token.ent_iob = EntIoB::from_id(
                u8::try_from(value).map_err(|_| SpacyError::InvalidEntIob(value))?,
            )?;
        }
        Attribute::EntType => token.ent_type = value,
        Attribute::Head => token.head = value as i32,
        Attribute::SentStart => token.sent_start = SentStart::from(value as i8),
        Attribute::Spacy => token.spacy = value != 0,
        Attribute::Norm => token.norm = value,
        Attribute::Morph => token.morph = value,
        Attribute::EntKbId => token.ent_kb_id = value,
        Attribute::EntId => token.ent_id = value,
        Attribute::Idx => token.idx = value as u32,
        Attribute::InterlinguaLemmaId => {
            token.interlingua_lemma_id = if value == 0 {
                None
            } else {
                Some(InterlinguaId::from_u64(value))
            };
        }
        Attribute::InterlinguaEntityId => {
            token.interlingua_entity_id = if value == 0 {
                None
            } else {
                Some(InterlinguaId::from_u64(value))
            };
        }
        Attribute::AnnotationConfidence => {
            token.confidence = if value == 0 {
                None
            } else {
                Some(f64::from_bits(value))
            };
        }
        // Flags and lexeme-derived values are computed at lexeme creation.
        Attribute::IsAlpha
        | Attribute::IsAscii
        | Attribute::IsDigit
        | Attribute::IsLower
        | Attribute::IsPunct
        | Attribute::IsSpace
        | Attribute::IsTitle
        | Attribute::IsUpper
        | Attribute::LikeUrl
        | Attribute::LikeNum
        | Attribute::LikeEmail
        | Attribute::IsStop
        | Attribute::IsOovDeprecated
        | Attribute::IsBracket
        | Attribute::IsQuote
        | Attribute::IsLeftPunct
        | Attribute::IsRightPunct
        | Attribute::IsCurrency
        | Attribute::IsDetWord
        | Attribute::IsAdpWord
        | Attribute::IsAuxWord
        | Attribute::IsCconjWord
        | Attribute::IsSconjWord
        | Attribute::IsPronWord
        | Attribute::IsVerbWord
        | Attribute::IsBeVerb
        | Attribute::IsBareInfHost
        | Attribute::IsNegator
        | Attribute::IsNominative
        | Attribute::IsPossessive
        | Attribute::IsRelativizer
        | Attribute::IsSensoryVerb
        | Attribute::IsEpistemicVerb
        | Attribute::IsDiscourseMarker
        | Attribute::IsAdverbWord
        | Attribute::IsSubordComplement
        | Attribute::IsSubordAdverbial
        | Attribute::IsWhereWord
        | Attribute::IsLocative
        | Attribute::IsDemonstrative
        | Attribute::IsTodayWord
        | Attribute::IsAsWord
        | Attribute::IsAfterWord
        | Attribute::IsThatWord
        | Attribute::IsTwiceWord
        | Attribute::IsYetWord
        | Attribute::IsPleaseWord
        | Attribute::IsBeCliticS
        | Attribute::IsBeClitic
        | Attribute::IsThereWord
        | Attribute::Id
        | Attribute::Orth
        | Attribute::Lower
        | Attribute::Shape
        | Attribute::Prefix
        | Attribute::Suffix
        | Attribute::Length
        | Attribute::Cluster
        | Attribute::Prob
        | Attribute::Lang
        | Attribute::SentEnd
        | Attribute::Other(_) => return Err(SpacyError::ReadOnlyAttribute(attr.id())),
    }
    Ok(())
}

/// Port of `set_children_from_heads` (`doc.pyx:1815-1834` +
/// `_set_lr_kids_and_edges`). First validates that every `head` stays in
/// bounds, then runs the multi-pass left/right edge propagation (up to 12
/// iterations to settle non-projective parses), and finally marks sentence
/// starts at each root's left edge.
pub fn set_children_from_heads(tokens: &mut [TokenRecord]) -> Result<(), SpacyError> {
    let len = tokens.len();
    for (i, token) in tokens.iter().enumerate() {
        let abs = i as i64 + i64::from(token.head);
        if abs < 0 || abs >= len as i64 {
            return Err(SpacyError::HeadOutOfBounds {
                token: i,
                head: token.head,
                abs,
                len,
            });
        }
    }

    for (i, token) in tokens.iter_mut().enumerate() {
        token.l_kids = 0;
        token.r_kids = 0;
        token.l_edge = i as u32;
        token.r_edge = i as u32;
    }

    let mut loop_count = 0usize;
    loop {
        let within = set_lr_kids_and_edges(tokens, loop_count);
        if within {
            break;
        }
        if loop_count > 10 {
            break;
        }
        loop_count += 1;
    }

    for token in &mut *tokens {
        token.sent_start = SentStart::NotStart;
    }
    for i in 0..len {
        let token = &tokens[i];
        if token.head == 0 && token.has_head() {
            let edge = token.l_edge as usize;
            tokens[edge].sent_start = SentStart::Start;
        }
    }
    Ok(())
}

/// One iteration of the left/right kid-and-edge propagation. Returns whether
/// every head now lies within its current sentence segment.
fn set_lr_kids_and_edges(tokens: &mut [TokenRecord], loop_count: usize) -> bool {
    let len = tokens.len();

    // Left pass: count left kids and propagate edges leftward.
    for i in 0..len {
        let head_rel = tokens[i].head;
        let hi = (i as i64 + i64::from(head_rel)) as usize;
        if hi == i {
            continue;
        }
        let (l_edge, r_edge, is_left) = {
            let child = &tokens[i];
            (child.l_edge, child.r_edge, i < hi)
        };
        let head = &mut tokens[hi];
        if loop_count == 0 && is_left {
            head.l_kids += 1;
        }
        if l_edge < head.l_edge {
            head.l_edge = l_edge;
        }
        if r_edge > head.r_edge {
            head.r_edge = r_edge;
        }
    }

    // Right pass: count right kids and propagate edges rightward.
    for i in (0..len).rev() {
        let head_rel = tokens[i].head;
        let hi = (i as i64 + i64::from(head_rel)) as usize;
        if hi == i {
            continue;
        }
        let (r_edge, l_edge, is_right) = {
            let child = &tokens[i];
            (child.r_edge, child.l_edge, i > hi)
        };
        let head = &mut tokens[hi];
        if loop_count == 0 && is_right {
            head.r_kids += 1;
        }
        if r_edge > head.r_edge {
            head.r_edge = r_edge;
        }
        if l_edge < head.l_edge {
            head.l_edge = l_edge;
        }
    }

    // Sentence segments from the current state, then check that no head
    // crosses its segment boundary.
    let mut sent_starts = std::collections::HashSet::new();
    for token in &*tokens {
        if token.head == 0 {
            sent_starts.insert(token.l_edge);
        }
    }
    let mut curr_start = 0usize;
    for i in 0..len {
        if (i > 0 && sent_starts.contains(&(i as u32))) || i == len - 1 {
            let curr_end = i;
            for (j, token) in tokens.iter().enumerate().take(curr_end).skip(curr_start) {
                let abs = j as i64 + i64::from(token.head);
                if abs < curr_start as i64 || abs > curr_end as i64 {
                    return false;
                }
            }
            curr_start = i;
        }
    }
    true
}

#[cfg(test)]
#[path = "../tests/doc.rs"]
mod tests;
