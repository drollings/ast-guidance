//! The deterministic, punctuation-rule sentence segmenter (walkthrough §8.2;
//! `spacy/pipeline/sentencizer.pyx`).
//!
//! `Sentencizer` is the non-trainable `Pipe` alternative to the statistical
//! `SentenceRecognizer`: it marks the first token and every token after a run
//! of punctuation characters as a sentence start (`sent_start` = 1, others
//! -1), respecting a pre-existing annotation unless `overwrite` is set. Pure
//! `predict`/`set_annotations` split, matching the pipeline purity contract.

use std::collections::HashSet;

use crate::doc::{Doc, SentStart};

/// The default sentence-ending punctuation, exactly the 128 chars from
/// `sentencizer.pyx:25-37`.
pub const DEFAULT_PUNCT_CHARS: &str = "!.?։؟۔܀܁܂߹।॥၊။።፧፨᙮᜵᜶᠃᠉᥄᥅᪨᪩᪪᪫᭚᭛᭞᭟᰻᰼᱾᱿‼‽⁇⁈⁉⸮⸼꓿꘎꘏꛳꛷꡶꡷꣎꣏꤯꧈꧉꩝꩞꩟꫰꫱꯫﹒﹖﹗！．？𐩖𐩗𑁇𑁈𑂾𑂿𑃀𑃁𑅁𑅂𑅃𑇅𑇆𑇍𑇞𑇟𑈸𑈹𑈻𑈼𑊩𑑋𑑌𑗂𑗃𑗉𑗊𑗋𑗌𑗍𑗎𑗏𑗐𑗑𑗒𑗓𑗔𑗕𑗖𑗗𑙁𑙂𑜼𑜽𑜾𑩂𑩃𑪛𑪜𑱁𑱂𖩮𖩯𖫵𖬷𖬸𖭄𛲟𝪈｡。";

/// The deterministic sentence segmenter.
#[derive(Debug, Clone)]
pub struct Sentencizer {
    punct_chars: HashSet<char>,
    overwrite: bool,
}

impl Default for Sentencizer {
    fn default() -> Self {
        Self {
            punct_chars: Self::default_punct_chars(),
            overwrite: false,
        }
    }
}

impl Sentencizer {
    /// A sentencizer over the default punctuation set, `overwrite = false`
    /// (spaCy's `BACKWARD_OVERWRITE`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A sentencizer over a custom punctuation set.
    #[must_use]
    pub fn with_punct_chars(chars: HashSet<char>) -> Self {
        Self {
            punct_chars: chars,
            overwrite: false,
        }
    }

    /// Whether existing `sent_start` annotations are overwritten.
    #[must_use]
    pub fn overwrite(&self) -> bool {
        self.overwrite
    }

    /// Set the `overwrite` flag.
    pub fn set_overwrite(&mut self, overwrite: bool) {
        self.overwrite = overwrite;
    }

    /// The default sentence-ending punctuation set (128 chars).
    #[must_use]
    pub fn default_punct_chars() -> HashSet<char> {
        DEFAULT_PUNCT_CHARS.chars().collect()
    }

    /// Pure `predict` (`sentencizer.pyx:80-108`): one `bool` per token —
    /// `true` for the first token and for the token after each run of
    /// punctuation characters.
    #[must_use]
    pub fn predict(&self, doc: &Doc) -> Vec<bool> {
        let mut guesses = vec![false; doc.len()];
        if doc.is_empty() {
            return guesses;
        }
        let mut start = 0usize;
        let mut seen_period = false;
        guesses[0] = true;
        for i in 0..doc.len() {
            let text = doc.token_text(i);
            let is_in_punct_chars = text.chars().count() == 1 && self.punct_chars.contains(&text.chars().next().expect("one char"));
            let is_punct = doc.token(i).lexeme.flags.is_punct();
            if seen_period && !is_punct && !is_in_punct_chars {
                guesses[start] = true;
                start = i;
                seen_period = false;
            } else if is_in_punct_chars {
                seen_period = true;
            }
        }
        if start < doc.len() {
            guesses[start] = true;
        }
        guesses
    }

    /// `set_annotations` (`sentencizer.pyx:110-126`): write `sent_start` 1/-1,
    /// honoring a pre-existing annotation unless `overwrite` is set.
    pub fn set_annotations(&self, doc: &mut Doc, guesses: &[bool]) {
        for (i, &guess) in guesses.iter().enumerate() {
            let token = doc.token_mut(i);
            if token.sent_start == SentStart::Unset || self.overwrite {
                token.sent_start = if guess { SentStart::Start } else { SentStart::NotStart };
            }
        }
    }

    /// `predict` + `set_annotations` on one doc (the `Pipe.__call__` path).
    pub fn process(&self, doc: &mut Doc) {
        let guesses = self.predict(doc);
        self.set_annotations(doc, &guesses);
    }
}

#[cfg(test)]
#[path = "../tests/sentencizer.rs"]
mod tests;
