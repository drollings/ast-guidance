//! The two-level lexicon: shared word-types (`Lexeme`) owned by a `Lexicon`,
//! mirroring spaCy's `LexemeC` / `Vocab._by_orth` (`spacy/structs.pxd`,
//! `spacy/vocab.pyx`).
//!
//! A `Lexeme` holds the surface-shape facts shared by every token of the same
//! orth form; per-token context lives in [`crate::doc::TokenRecord`]. String
//! attributes are stored as MurmurHash64A hashes (resolved via the shared
//! `StringStore`), and the lexicon owns one immutable `Arc<Lexeme>` per orth.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::attrs::Attribute;
use crate::lang::base_norm;
use crate::lex_attrs;
use crate::strings::StringStore;

/// Out-of-vocabulary rank sentinel (`lexeme.pyx:38`: `0xffff_ffff_ffff_ffff`).
pub const OOV_RANK: u64 = u64::MAX;

/// The 64-bit lexeme flag bitmask. Bit `i` is set iff attribute id `i` is
/// true (ids 1–18 are the named surface flags from `spacy/attrs.pxd`;
/// ids 19–47 are the closed-class function-word flags populated per
/// language from [`LexiconConfig::function_words`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LexemeFlags(u64);

impl LexemeFlags {
    /// A new bitmask.
    #[must_use]
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    /// The raw 64 bits.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether the flag at attribute id `flag_id` is set.
    #[must_use]
    pub const fn check(self, flag_id: u16) -> bool {
        self.0 & (1u64 << flag_id) != 0
    }

    /// Set the flag at attribute id `flag_id`.
    #[must_use]
    pub const fn set(mut self, flag_id: u16) -> Self {
        self.0 |= 1u64 << flag_id;
        self
    }

    /// `IS_ALPHA`.
    #[must_use]
    pub const fn is_alpha(self) -> bool {
        self.check(Attribute::IsAlpha.id())
    }

    /// `IS_ASCII`.
    #[must_use]
    pub const fn is_ascii(self) -> bool {
        self.check(Attribute::IsAscii.id())
    }

    /// `IS_DIGIT`.
    #[must_use]
    pub const fn is_digit(self) -> bool {
        self.check(Attribute::IsDigit.id())
    }

    /// `IS_LOWER`.
    #[must_use]
    pub const fn is_lower(self) -> bool {
        self.check(Attribute::IsLower.id())
    }

    /// `IS_PUNCT`.
    #[must_use]
    pub const fn is_punct(self) -> bool {
        self.check(Attribute::IsPunct.id())
    }

    /// `IS_SPACE`.
    #[must_use]
    pub const fn is_space(self) -> bool {
        self.check(Attribute::IsSpace.id())
    }

    /// `IS_TITLE`.
    #[must_use]
    pub const fn is_title(self) -> bool {
        self.check(Attribute::IsTitle.id())
    }

    /// `IS_UPPER`.
    #[must_use]
    pub const fn is_upper(self) -> bool {
        self.check(Attribute::IsUpper.id())
    }

    /// `LIKE_URL`.
    #[must_use]
    pub const fn like_url(self) -> bool {
        self.check(Attribute::LikeUrl.id())
    }

    /// `LIKE_NUM`.
    #[must_use]
    pub const fn like_num(self) -> bool {
        self.check(Attribute::LikeNum.id())
    }

    /// `LIKE_EMAIL`.
    #[must_use]
    pub const fn like_email(self) -> bool {
        self.check(Attribute::LikeEmail.id())
    }

    /// `IS_STOP`.
    #[must_use]
    pub const fn is_stop(self) -> bool {
        self.check(Attribute::IsStop.id())
    }

    /// `IS_BRACKET`.
    #[must_use]
    pub const fn is_bracket(self) -> bool {
        self.check(Attribute::IsBracket.id())
    }

    /// `IS_QUOTE`.
    #[must_use]
    pub const fn is_quote(self) -> bool {
        self.check(Attribute::IsQuote.id())
    }

    /// `IS_LEFT_PUNCT`.
    #[must_use]
    pub const fn is_left_punct(self) -> bool {
        self.check(Attribute::IsLeftPunct.id())
    }

    /// `IS_RIGHT_PUNCT`.
    #[must_use]
    pub const fn is_right_punct(self) -> bool {
        self.check(Attribute::IsRightPunct.id())
    }

    /// `IS_CURRENCY`.
    #[must_use]
    pub const fn is_currency(self) -> bool {
        self.check(Attribute::IsCurrency.id())
    }

    /// Closed-class function-word flags (ids 19–47). Each is set iff the
    /// lowercased orth is a member of the language's word set for that
    /// category (see [`LexiconConfig::function_words`]); the parser matches
    /// these bits instead of hard-coded word lists.
    #[must_use]
    pub const fn is_det_word(self) -> bool {
        self.check(Attribute::IsDetWord.id())
    }
    /// Closed POS-class words (adpositions).
    #[must_use]
    pub const fn is_adp_word(self) -> bool {
        self.check(Attribute::IsAdpWord.id())
    }
    /// Closed POS-class words (auxiliaries).
    #[must_use]
    pub const fn is_aux_word(self) -> bool {
        self.check(Attribute::IsAuxWord.id())
    }
    /// Closed POS-class words (coordinating conjunctions).
    #[must_use]
    pub const fn is_cconj_word(self) -> bool {
        self.check(Attribute::IsCconjWord.id())
    }
    /// Closed POS-class words (subordinating conjunctions).
    #[must_use]
    pub const fn is_sconj_word(self) -> bool {
        self.check(Attribute::IsSconjWord.id())
    }
    /// Closed POS-class words (pronouns).
    #[must_use]
    pub const fn is_pron_word(self) -> bool {
        self.check(Attribute::IsPronWord.id())
    }
    /// Closed verb forms (finite lexicon for the heuristic predicate).
    #[must_use]
    pub const fn is_verb_word(self) -> bool {
        self.check(Attribute::IsVerbWord.id())
    }
    /// Be-forms (copula/auxiliary hosts, incl. clitics).
    #[must_use]
    pub const fn is_be_verb(self) -> bool {
        self.check(Attribute::IsBeVerb.id())
    }
    /// Bare-infinitive hosts (do-support, modals, `n't`-split stubs).
    #[must_use]
    pub const fn is_bare_inf_host(self) -> bool {
        self.check(Attribute::IsBareInfHost.id())
    }
    /// Auxiliary-hosted negators (`n't`, `not`).
    #[must_use]
    pub const fn is_negator(self) -> bool {
        self.check(Attribute::IsNegator.id())
    }
    /// Nominative pronoun surfaces (finite-clause subjects).
    #[must_use]
    pub const fn is_nominative(self) -> bool {
        self.check(Attribute::IsNominative.id())
    }
    /// Possessive determiners (obligatorily head a nominal rightward).
    #[must_use]
    pub const fn is_possessive(self) -> bool {
        self.check(Attribute::IsPossessive.id())
    }
    /// Nominal relativizers with corpus evidence.
    #[must_use]
    pub const fn is_relativizer(self) -> bool {
        self.check(Attribute::IsRelativizer.id())
    }
    /// Sensory linking verbs.
    #[must_use]
    pub const fn is_sensory_verb(self) -> bool {
        self.check(Attribute::IsSensoryVerb.id())
    }
    /// Epistemic linking verbs.
    #[must_use]
    pub const fn is_epistemic_verb(self) -> bool {
        self.check(Attribute::IsEpistemicVerb.id())
    }
    /// Discourse-imperative markers.
    #[must_use]
    pub const fn is_discourse_marker(self) -> bool {
        self.check(Attribute::IsDiscourseMarker.id())
    }
    /// Closed time/manner adverbials.
    #[must_use]
    pub const fn is_adverb_word(self) -> bool {
        self.check(Attribute::IsAdverbWord.id())
    }
    /// Complement subordinators (`because` class).
    #[must_use]
    pub const fn is_subord_complement(self) -> bool {
        self.check(Attribute::IsSubordComplement.id())
    }
    /// Adjunct subordinators (`when`/`if`/`after` class).
    #[must_use]
    pub const fn is_subord_adverbial(self) -> bool {
        self.check(Attribute::IsSubordAdverbial.id())
    }
    /// Interrogative `where` (clause-initial gate).
    #[must_use]
    pub const fn is_where_word(self) -> bool {
        self.check(Attribute::IsWhereWord.id())
    }
    /// Locative/existential pro-forms (`there`, `here`).
    #[must_use]
    pub const fn is_locative(self) -> bool {
        self.check(Attribute::IsLocative.id())
    }
    /// Demonstratives (`this`, `these`, `those`).
    #[must_use]
    pub const fn is_demonstrative(self) -> bool {
        self.check(Attribute::IsDemonstrative.id())
    }
    /// Temporal adverbial with frozen refs (`today` class).
    #[must_use]
    pub const fn is_today_word(self) -> bool {
        self.check(Attribute::IsTodayWord.id())
    }
    /// Comparative/comment `as`.
    #[must_use]
    pub const fn is_as_word(self) -> bool {
        self.check(Attribute::IsAsWord.id())
    }
    /// Dual-class `after` (preposition vs. subordinator).
    #[must_use]
    pub const fn is_after_word(self) -> bool {
        self.check(Attribute::IsAfterWord.id())
    }
    /// Complementizer/demonstrative `that`.
    #[must_use]
    pub const fn is_that_word(self) -> bool {
        self.check(Attribute::IsThatWord.id())
    }
    /// Multiplicative `twice` (discourse-complement gate).
    #[must_use]
    pub const fn is_twice_word(self) -> bool {
        self.check(Attribute::IsTwiceWord.id())
    }
    /// Temporal `yet` (CCONJ→ADV gate).
    #[must_use]
    pub const fn is_yet_word(self) -> bool {
        self.check(Attribute::IsYetWord.id())
    }
    /// Interjection `please` (Intj vs. Adv split).
    #[must_use]
    pub const fn is_please_word(self) -> bool {
        self.check(Attribute::IsPleaseWord.id())
    }
    /// Possessive/copula clitic `'s` (pronoun-hosted AUX gate).
    #[must_use]
    pub const fn is_be_clitic_s(self) -> bool {
        self.check(Attribute::IsBeCliticS.id())
    }
    /// Be-clitics `'s`/`'re`/`'m` (progressive-participle host gate).
    #[must_use]
    pub const fn is_be_clitic(self) -> bool {
        self.check(Attribute::IsBeClitic.id())
    }
    /// Expletive `there` alone (subject slot; locatives share
    /// [`Attribute::IsLocative`]).
    #[must_use]
    pub const fn is_there_word(self) -> bool {
        self.check(Attribute::IsThereWord.id())
    }
}

/// A word-type record: the surface-shape facts for one orth, shared by all
/// tokens of that form. Field-for-field against `LexemeC`
/// (`spacy/structs.pxd:10-24`).
#[derive(Debug, Clone)]
pub struct Lexeme {
    pub flags: LexemeFlags,
    /// Language id hash.
    pub lang: u64,
    /// Rank / index into a vectors table, or [`OOV_RANK`].
    pub id: u64,
    /// Character length of the surface form.
    pub length: u32,
    /// Orth (verbatim text) hash.
    pub orth: u64,
    /// Lowercased form hash.
    pub lower: u64,
    /// Normalized form hash.
    pub norm: u64,
    /// Shape hash (e.g. `"Xxxx"`).
    pub shape: u64,
    /// First-char hash.
    pub prefix: u64,
    /// Last-3-chars hash.
    pub suffix: u64,
}

impl Lexeme {
    /// The global empty lexeme: a zeroed sentinel with `id = OOV_RANK`
    /// (`lexeme.pyx:38-40`), used for unset / out-of-bounds positions.
    #[must_use]
    pub fn empty() -> Arc<Self> {
        static EMPTY: std::sync::OnceLock<Arc<Lexeme>> = std::sync::OnceLock::new();
        EMPTY
            .get_or_init(|| {
                Arc::new(Self {
                    flags: LexemeFlags::default(),
                    lang: 0,
                    id: OOV_RANK,
                    length: 0,
                    orth: 0,
                    lower: 0,
                    norm: 0,
                    shape: 0,
                    prefix: 0,
                    suffix: 0,
                })
            })
            .clone()
    }

    /// The orth text, resolved through the vocabulary's string store.
    #[must_use]
    pub fn orth_text(&self, strings: &StringStore) -> String {
        strings
            .get(self.orth)
            .map_or_else(String::new, |s| s.to_string())
    }
}

/// Per-lexicon attribute configuration: language, stop words, and norm
/// overrides. Supplied by the language data (e.g. English) at construction.
#[derive(Debug, Clone, Default)]
pub struct LexiconConfig {
    /// Language tag hash (e.g. `"en"`); `0` leaves `Lexeme.lang` unset.
    pub lang: u64,
    /// Stop-word set matched on the lowercased orth.
    pub stop_words: HashSet<String>,
    /// Norm overrides keyed by surface form (`NORM_EXCEPTIONS`).
    pub norm_exceptions: HashMap<String, String>,
    /// Number words (cardinal + ordinal) for the language-specific `LIKE_NUM`
    /// override; empty uses the base digit/fraction `like_num`.
    pub num_words: HashSet<String>,
    /// Closed-class function-word categories: lowercased orth → extra
    /// [`LexemeFlags`] bits (attribute ids 19–47), ORed into the lexeme at
    /// intern time. This is the per-language data the parser matches instead
    /// of hard-coded word lists — a multilingual port supplies a different
    /// map (the future blob-backed `FunctionWordView` fills this same
    /// field). Words absent from every set gain no bits.
    pub function_words: HashMap<String, u64>,
}

/// The lexicon: one immutable `Arc<Lexeme>` per orth, computed on first sight
/// from the deterministic [`crate::lex_attrs`] table. Thread-safe and shared
/// across docs and tokenizers.
#[derive(Debug)]
pub struct Lexicon {
    strings: Arc<StringStore>,
    by_orth: RwLock<HashMap<u64, Arc<Lexeme>>>,
    config: LexiconConfig,
}

impl Lexicon {
    /// A lexicon over `strings` with the given configuration.
    #[must_use]
    pub fn new(strings: Arc<StringStore>, config: LexiconConfig) -> Self {
        Self {
            strings,
            by_orth: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// A lexicon with empty configuration (no stop words, no norms).
    #[must_use]
    pub fn default_with_strings(strings: Arc<StringStore>) -> Self {
        Self::new(strings, LexiconConfig::default())
    }

    /// The shared string store backing the lexicon.
    #[must_use]
    pub fn strings(&self) -> &Arc<StringStore> {
        &self.strings
    }

    /// The lexicon configuration.
    #[must_use]
    pub fn config(&self) -> &LexiconConfig {
        &self.config
    }

    /// Number of distinct lexemes interned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_orth
            .read()
            .expect("Lexicon read lock poisoned")
            .len()
    }

    /// Whether the lexicon is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The lexeme for `orth` if already interned, else `None`.
    #[must_use]
    pub fn get_by_orth(&self, orth: u64) -> Option<Arc<Lexeme>> {
        if orth == 0 {
            return Some(Lexeme::empty());
        }
        self.by_orth
            .read()
            .expect("Lexicon read lock poisoned")
            .get(&orth)
            .cloned()
    }

    /// Get (or create and intern) the lexeme for `text`. The empty string maps
    /// to the global empty lexeme, matching `Vocab.get` (`vocab.pyx:191-207`).
    #[must_use]
    pub fn get_or_create(&self, text: &str) -> Arc<Lexeme> {
        if text.is_empty() {
            return Lexeme::empty();
        }
        let orth = self.strings.add(text);
        if let Some(lexeme) = self.get_by_orth(orth) {
            return lexeme;
        }
        let lexeme = Arc::new(self.compute_lexeme(text, orth));
        let mut map = self.by_orth.write().expect("Lexicon write lock poisoned");
        map.entry(orth).or_insert_with(|| lexeme.clone());
        map.get(&orth).expect("just inserted").clone()
    }

    /// Compute a fresh `Lexeme` record for `text` (no interning).
    #[must_use]
    fn compute_lexeme(&self, text: &str, orth: u64) -> Lexeme {
        let lower_text = lex_attrs::lower(text);
        let norm_text = self
            .config
            .norm_exceptions
            .get(text)
            .cloned()
            .or_else(|| base_norm(text).map(str::to_string))
            .unwrap_or_else(|| lower_text.clone());

        let mut flags = LexemeFlags::default();
        flags = flags
            .set_bool(Attribute::IsAlpha.id(), lex_attrs::is_alpha(text))
            .set_bool(Attribute::IsAscii.id(), lex_attrs::is_ascii(text))
            .set_bool(Attribute::IsDigit.id(), lex_attrs::is_digit(text))
            .set_bool(Attribute::IsLower.id(), lex_attrs::is_lower(text))
            .set_bool(Attribute::IsPunct.id(), lex_attrs::is_punct(text))
            .set_bool(Attribute::IsSpace.id(), lex_attrs::is_space(text))
            .set_bool(Attribute::IsTitle.id(), lex_attrs::is_title(text))
            .set_bool(Attribute::IsUpper.id(), lex_attrs::is_upper(text))
            .set_bool(Attribute::LikeUrl.id(), lex_attrs::like_url(text))
            .set_bool(
                Attribute::LikeNum.id(),
                if self.config.num_words.is_empty() {
                    lex_attrs::like_num(text)
                } else {
                    lex_attrs::like_num_en(text, &self.config.num_words)
                },
            )
            .set_bool(Attribute::LikeEmail.id(), lex_attrs::like_email(text))
            .set_bool(
                Attribute::IsStop.id(),
                self.config.stop_words.contains(&lower_text),
            )
            .set_bool(Attribute::IsBracket.id(), lex_attrs::is_bracket(text))
            .set_bool(Attribute::IsQuote.id(), lex_attrs::is_quote(text))
            .set_bool(Attribute::IsLeftPunct.id(), lex_attrs::is_left_punct(text))
            .set_bool(
                Attribute::IsRightPunct.id(),
                lex_attrs::is_right_punct(text),
            )
            .set_bool(Attribute::IsCurrency.id(), lex_attrs::is_currency(text));
        if let Some(&extra) = self.config.function_words.get(&lower_text) {
            flags = LexemeFlags::new(flags.bits() | extra);
        }

        Lexeme {
            flags,
            lang: self.config.lang,
            id: OOV_RANK,
            length: text.chars().count() as u32,
            orth,
            lower: self.strings.add(&lower_text),
            norm: self.strings.add(&norm_text),
            shape: self.strings.add(&lex_attrs::word_shape(text)),
            prefix: self.strings.add(&lex_attrs::prefix(text)),
            suffix: self.strings.add(&lex_attrs::suffix(text)),
        }
    }
}

impl LexemeFlags {
    /// Set (true) or clear (false) the flag at `flag_id`.
    #[must_use]
    const fn set_bool(mut self, flag_id: u16, value: bool) -> Self {
        if value {
            self.0 |= 1u64 << flag_id;
        } else {
            self.0 &= !(1u64 << flag_id);
        }
        self
    }
}

#[cfg(test)]
#[path = "../tests/lexeme.rs"]
mod tests;
