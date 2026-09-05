//! The lookup/rule lemmatizer (walkthrough §8.3; `spacy/pipeline/lemmatizer.py`
//! + `spacy/lang/en/lemmatizer.py`).
//!
//! `rule` mode is the parity surface: per-lowercased-UPOS suffix rules over
//! the `lemma_index` / `lemma_exc` / `lemma_rules` tables, loaded from the
//! versioned binary blob [`crate::lemma_blob`] that `build.rs` compiles out of
//! `../../env/en_lemmatizer.json` (the auditable source of truth generated from
//! `spacy-lookups-data`), and gated by the language's `is_base_form` (the
//! English morph-feature check). `lookup` mode is the flat `surface → lemma`
//! table fallback for languages without rule data. Every surface writes the
//! same lemma string, so downstream never knows which mode produced it.

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::hash::hash_utf8;
use crate::labels::Upos;
use crate::lemma_blob::LemmaBlob;
use crate::lex_attrs;
use crate::morph::Morphology;
use crate::strings::StringStore;

/// The lemma cache key: `(orth hash, pos id, morph key)`.
type CacheKey = (u64, u8, u64);

/// How the lemmatizer resolves a lemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LemmatizerMode {
    /// Per-POS suffix rules over the index/exc/rules tables.
    Rule,
    /// Flat `surface → lemma` lookup table.
    Lookup,
}

/// The rule/lookup lemmatizer. Rule tables come from a parsed [`LemmaBlob`]
/// (zero-alloc binary-search lookups); the per-token result cache is
/// interior-mutable and shared.
#[derive(Debug)]
pub struct Lemmatizer {
    mode: LemmatizerMode,
    /// Rule-mode tables (`rules`/`index`/`exc`), one per POS key.
    blob: Option<LemmaBlob>,
    /// Optional flat lookup table (`lemma_lookup`), for `Lookup` mode.
    lookup: Option<HashMap<String, Vec<String>>>,
    /// The morphology table used to resolve a token's morph key for
    /// `is_base_form`. `None` disables the morph checks (empty features).
    morphology: Option<std::sync::Arc<Morphology>>,
    cache: RwLock<HashMap<CacheKey, Vec<String>>>,
}

impl Lemmatizer {
    /// A rule-mode lemmatizer over a parsed versioned lemma blob.
    pub fn from_blob(blob: LemmaBlob) -> Self {
        Self {
            mode: LemmatizerMode::Rule,
            blob: Some(blob),
            lookup: None,
            morphology: None,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// From a shared `LemmaView` (taxonomy_blob) — read-biased `RwLock` retained, no `dashmap` swap this roadmap.
    pub fn from_view(view: std::sync::Arc<LemmaBlob>) -> Self {
        Self::from_blob((*view).clone())
    }

    /// A rule-mode lemmatizer over the generated English tables
    /// (`en_core_web_sm`'s `mode=rule` configuration).
    #[must_use]
    pub fn english_rule() -> Self {
        let blob = LemmaBlob::from_bytes(crate::lang::en::LEMMAS_BLOB)
            .expect("embedded English lemma blob is valid (build.rs)");
        Self::from_blob(blob)
    }

    /// A lookup-mode lemmatizer over a flat `surface → lemma` table.
    pub fn lookup(table: HashMap<String, String>) -> Self {
        let mut lookup = HashMap::with_capacity(table.len());
        for (surface, lemma) in table {
            lookup.insert(surface, vec![lemma]);
        }
        Self {
            mode: LemmatizerMode::Lookup,
            blob: None,
            lookup: Some(lookup),
            morphology: None,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Attach the morphology table used to resolve morph keys in `is_base_form`.
    #[must_use]
    pub fn with_morphology(mut self, morphology: std::sync::Arc<Morphology>) -> Self {
        self.morphology = Some(morphology);
        self
    }

    /// The mode.
    #[must_use]
    pub fn mode(&self) -> LemmatizerMode {
        self.mode
    }

    /// Number of cached analyses.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache.read().expect("lemma cache lock poisoned").len()
    }

    /// The lemma forms for a token surface under a coarse POS and morph key.
    /// The first element is the primary lemma (spaCy's `token.lemma_`).
    pub fn lemmatize(&self, orth: &str, pos: Upos, morph_key: u64) -> Vec<String> {
        match self.mode {
            LemmatizerMode::Rule => self.rule_lemmatize(orth, pos, morph_key),
            LemmatizerMode::Lookup => self.lookup_lemmatize(orth),
        }
    }

    /// The lookup path (`lookup_lemmatize`, `lemmatizer.py:158-170`).
    fn lookup_lemmatize(&self, orth: &str) -> Vec<String> {
        let Some(table) = &self.lookup else {
            return vec![orth.to_string()];
        };
        table.get(orth).cloned().unwrap_or_else(|| vec![orth.to_string()])
    }

    /// The rule path (`rule_lemmatize`, `lemmatizer.py:172-240`), ported
    /// verbatim: base-form shortcut, per-POS suffix rules over the index, then
    /// exceptions, with the oov/exception ordering preserved.
    fn rule_lemmatize(&self, orth: &str, pos: Upos, morph_key: u64) -> Vec<String> {
        // Closed contraction-splinter lemmas first (spaCy lookup parity): the
        // tokenizer splits can't→ca/n't and won't→wo/n't, so the rule tables
        // never see the base form. POS-conditioned — possessive 's
        // (PART/case) keeps its surface lemma; only aux-classified clitics
        // resolve. 'd is excluded (would/had is underdetermined).
        if let Some(lemma) = contraction_lemma(orth, pos) {
            return vec![lemma.to_string()];
        }
        let cache_key = (hash_utf8(orth), pos.id() as u8, morph_key);
        if let Some(cached) = self.cache.read().expect("lemma cache lock poisoned").get(&cache_key) {
            return cached.clone();
        }

        let base = matches!(pos, Upos::NoTag | Upos::Eol | Upos::Space)
            || self.is_base_form(pos, morph_key);
        let forms = if base {
            vec![orth.to_lowercase()]
        } else if !self.has_tables(pos) {
            if pos == Upos::Propn {
                vec![orth.to_string()]
            } else {
                vec![orth.to_lowercase()]
            }
        } else {
            // No rule fired: fall back to the surface — lowercased for
            // verbs (never proper nouns: sentence case on a verb is
            // orthographic, never lexical) and for all-caps tokens
            // (acronyms, never title-case names). Title-case nominals keep
            // surface per the proper-noun convention (`French`, `John`;
            // unit-pinned alongside `proper_nouns_keep_case`).
            let shouty = orth.chars().any(|c| c.is_alphabetic())
                && orth.chars().all(|c| !c.is_lowercase());
            self.apply_rules(orth, pos, pos == Upos::Verb || shouty)
        };

        self.cache
            .write()
            .expect("lemma cache lock poisoned")
            .insert(cache_key, forms.clone());
        forms
    }

    /// Whether any rule table covers `pos` (enum match — no string probe).
    fn has_tables(&self, pos: Upos) -> bool {
        self.blob.as_ref().is_some_and(|b| b.has_pos(pos))
    }

    /// The core rule application (`lemmatizer.py:207-240`): endswith rules,
    /// index-validated forms first, oov forms on fallback, exceptions up front.
    /// `lowercase_fallback` selects the lowercased surface when no rule
    /// fires (verbs / acronyms); otherwise the surface is kept for
    /// title-case nominals (proper-noun convention).
    fn apply_rules(&self, orth: &str, pos: Upos, lowercase_fallback: bool) -> Vec<String> {
        let orig = orth.to_string();
        let string = orth.to_lowercase();

        let blob = self.blob.as_ref().expect("rule mode carries a blob");
        let rules = blob.rules_for(pos);

        let mut forms: Vec<String> = Vec::new();
        let mut oov_forms: Vec<String> = Vec::new();
        for (old, new) in rules {
            if string.ends_with(old) {
                let form = format!("{}{}", &string[..string.len() - old.len()], new);
                if form.is_empty() {
                    continue;
                }
                let in_index = blob.index_contains_pos(pos, &form);
                if in_index || !lex_attrs::is_alpha(&form) {
                    if in_index {
                        forms.insert(0, form);
                    } else {
                        forms.push(form);
                    }
                } else {
                    oov_forms.push(form);
                }
            }
        }

        // Remove duplicates, preserving the order of applied rules.
        let mut seen: HashSet<String> = HashSet::new();
        forms.retain(|f| seen.insert(f.clone()));

        // Exceptions go first, so they get priority.
        if let Some(lemmas) = blob.exc_for_pos(pos, string.as_str()) {
            for lemma in lemmas.split(|&b| b == 0) {
                if lemma.is_empty() {
                    continue;
                }
                let lemma = std::str::from_utf8(lemma).expect("lemma blob is UTF-8");
                if seen.insert(lemma.to_string()) {
                    forms.insert(0, lemma.to_string());
                }
            }
        }

        if forms.is_empty() {
            forms.extend(oov_forms);
        }
        if forms.is_empty() {
            if lowercase_fallback {
                forms.push(string.clone());
            } else {
                forms.push(orig);
            }
        }
        forms
    }

    /// The English `is_base_form` check (`lang/en/lemmatizer.py:8-40`): an
    /// uninflected paradigm needs no lemmatization. Morph features come from
    /// the token's morphology key via the attached table.
    #[must_use]
    pub fn is_base_form(&self, pos: Upos, morph_key: u64) -> bool {
        let dict = self
            .morphology
            .as_ref()
            .and_then(|m| m.to_dict(morph_key))
            .unwrap_or_default();
        let get = |k: &str| dict.get(k).map(String::as_str);
        if pos == Upos::Noun && get("Number") == Some("Sing") {
            return true;
        }
        if pos == Upos::Verb && get("VerbForm") == Some("Inf") {
            return true;
        }
        if pos == Upos::Verb
            && get("VerbForm") == Some("Fin")
            && get("Tense") == Some("Pres")
            && get("Number").is_none()
        {
            return true;
        }
        if pos == Upos::Adj && get("Degree") == Some("Pos") {
            return true;
        }
        if get("VerbForm") == Some("Inf") {
            return true;
        }
        if get("VerbForm") == Some("None") {
            return true;
        }
        if get("Degree") == Some("Pos") {
            return true;
        }
        false
    }
}

/// Closed contraction-splinter lemma map (UD + spaCy lookup parity).
/// `ca` is the bound allomorph of `can`, `wo` of `will` (both occur only
/// pre-`n't`); `n't` is the negator `not`; clitic `be`/`have`/`will` forms
/// resolve to their host. Full auxiliary `be`-forms resolve to `be` — the
/// blob carries no `aux` tables, so without this map they fall through to
/// lowercased surface (`is` ≠ `be`). Gated on the parser's POS so
/// possessive `'s` (PART/case, `Bell's theorem`) and nominal uses keep
/// surface lemmas.
/// Returns the canonical lemma, or `None` when the tables should decide.
fn contraction_lemma(orth: &str, pos: Upos) -> Option<&'static str> {
    if matches!(pos, Upos::Part) && orth.eq_ignore_ascii_case("n't") {
        return Some("not");
    }
    if matches!(pos, Upos::Aux) {
        if orth.eq_ignore_ascii_case("ca") {
            return Some("can");
        }
        if orth.eq_ignore_ascii_case("wo") {
            return Some("will");
        }
        if orth.eq_ignore_ascii_case("is")
            || orth.eq_ignore_ascii_case("are")
            || orth.eq_ignore_ascii_case("was")
            || orth.eq_ignore_ascii_case("were")
            || orth.eq_ignore_ascii_case("am")
            || orth.eq_ignore_ascii_case("be")
            || orth.eq_ignore_ascii_case("been")
            || orth.eq_ignore_ascii_case("being")
        {
            return Some("be");
        }
        if orth.eq_ignore_ascii_case("'s")
            || orth.eq_ignore_ascii_case("'re")
            || orth.eq_ignore_ascii_case("'m")
        {
            return Some("be");
        }
        if orth.eq_ignore_ascii_case("'ve") {
            return Some("have");
        }
        if orth.eq_ignore_ascii_case("'ll") {
            return Some("will");
        }
    }
    if matches!(pos, Upos::Aux | Upos::Verb)
        && (orth.eq_ignore_ascii_case("did") || orth.eq_ignore_ascii_case("does"))
    {
        return Some("do");
    }
    None
}

/// A lemmatizer whose morphology table resolves through a `StringStore`
/// (helper for callers without a full vocab).
#[must_use]
pub fn rule_lemmatizer_with_strings(strings: std::sync::Arc<StringStore>) -> Lemmatizer {    let morphology = std::sync::Arc::new(Morphology::new(strings));
    Lemmatizer::english_rule().with_morphology(morphology)
}

#[cfg(test)]
#[path = "../tests/lemmatizer.rs"]
mod tests;
