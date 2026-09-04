//! LFM ↔ spacy-rs token alignment (ROADMAP_20260827_ORT §4.1).
//!
//! The trained-encoder annotation rung must map LFM-token predictions back
//! onto the spacy-rs orth baseline (validator check 1 anchors every record's
//! `text` to the deterministic tokenizer's orth). The mapping is **pure string
//! alignment by byte offsets** — it consumes plain spans, so `fluent-onnx`
//! stays spacy-free (no `spacy-rs` import): the caller supplies the spacy
//! token spans (built from the tokenizer's `idx` + orth length) and the LFM
//! tokenizer's per-token byte offsets. Subwords, special tokens (`[CLS]` /
//! `[SEP]` / `<|startoftext|>` ...) and OOV disagreements all fall out of the
//! same offset math.
//!
//! This is the largest un-quantified cost in the proposal (red-team risk #2),
//! so the pure core is exhaustively golden-tested before any head is built on
//! top of it.

use std::ops::Range;

/// The alignment between the LFM (HF) token stream and the spacy-rs token
/// stream, computed purely from byte offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpacyTokenAlignment {
    /// For each LFM token: the spacy-rs token index whose byte span contains
    /// it. `None` for special / zero-width tokens (`[CLS]`, `[SEP]`,
    /// `<|startoftext|>` ...) whose span covers no spacy token, and for an
    /// LFM token whose span matches no spacy span (OOV / tokenizer drift).
    pub lfm_to_spacy: Vec<Option<usize>>,
    /// For each spacy-rs token: the contiguous range of LFM token indices that
    /// cover it. Empty (`s..s`) when no LFM token covers the spacy span — the
    /// caller must fall back (the token's hidden states are unavailable).
    pub spacy_to_lfm: Vec<Range<usize>>,
}

impl SpacyTokenAlignment {
    /// The LFM token index range covering spacy token `s`, or an empty range
    /// when the token has no covering LFM tokens.
    #[must_use]
    pub fn lfm_range(&self, spacy_index: usize) -> Range<usize> {
        self.spacy_to_lfm
            .get(spacy_index)
            .cloned()
            .unwrap_or(0..0)
    }

    /// Whether every spacy token has at least one covering LFM token (a
    /// round-trip sanity check the golden tests assert).
    #[must_use]
    pub fn is_total(&self) -> bool {
        self.spacy_to_lfm.iter().all(|r| !r.is_empty())
    }
}

/// The pure byte-offset aligner. Stateless — one shared instance serves every
/// worker (fluent-wvr "pure and shared" principle).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpacyTokenAligner;

impl SpacyTokenAligner {
    /// Build the spacy token spans from the tokenizer's per-token `idx` (byte
    /// offset of the token start) and orth lengths: `span = (idx, idx+len)`.
    /// Pure; the caller supplies the orth texts (slices suffice).
    #[must_use]
    pub fn spacy_spans(orth: &[impl AsRef<str>], idx: &[usize]) -> Vec<(usize, usize)> {
        orth
            .iter()
            .zip(idx.iter())
            .map(|(text, &i)| (i, i + text.as_ref().len()))
            .collect()
    }

    /// Align the LFM token stream to the spacy-rs token stream by byte offsets.
    ///
    /// `spacy_spans[i]` is the `[start, end)` byte range of spacy token `i`
    /// within the shared source text; `lfm_spans[j]` is the LFM tokenizer's
    /// per-token range (its `get_offsets()` output). Both must index the
    /// **same** source text.
    ///
    /// Mapping rule: an LFM token maps to the spacy token whose span *contains*
    /// it (`ss <= s && e <= se`). A zero-width LFM span (a special token) maps
    /// to nothing. A span that matches no single spacy token (tokenizer drift)
    /// maps to the spacy token with the greatest overlap, or nothing when no
    /// overlap exists.
    #[must_use]
    pub fn align(
        spacy_spans: &[(usize, usize)],
        lfm_spans: &[(usize, usize)],
    ) -> SpacyTokenAlignment {
        let n_spacy = spacy_spans.len();
        let n_lfm = lfm_spans.len();
        let mut lfm_to_spacy = vec![None; n_lfm];
        // Two-pointer scan: both streams are ordered and non-overlapping, so
        // one pass finds each LFM token's containing spacy token.
        let mut j = 0usize;
        for (i, &(s, e)) in lfm_spans.iter().enumerate() {
            if s == e {
                continue; // zero-width special token ([CLS]/[SEP]/...)
            }
            while j < n_spacy && spacy_spans[j].1 <= s {
                j += 1;
            }
            if j >= n_spacy {
                // LFM tail past the last spacy token: try the last spacy
                // span for a partial overlap (e.g. a trailing special token
                // whose span is the full text).
                if n_spacy > 0 {
                    lfm_to_spacy[i] = best_overlap(spacy_spans, (s, e));
                }
                continue;
            }
            let (ss, se) = spacy_spans[j];
            if ss <= s && e <= se {
                lfm_to_spacy[i] = Some(j);
            } else {
                lfm_to_spacy[i] = best_overlap(spacy_spans, (s, e));
            }
        }

        // Mapped LFM indices per spacy token are contiguous (both streams are
        // in text order, so LFM tokens covering one spacy token are adjacent).
        let mut starts = vec![usize::MAX; n_spacy];
        let mut ends = vec![0usize; n_spacy];
        for (i, mapped) in lfm_to_spacy.iter().enumerate() {
            if let Some(j) = mapped {
                starts[*j] = starts[*j].min(i);
                ends[*j] = ends[*j].max(i + 1);
            }
        }
        let spacy_to_lfm = (0..n_spacy)
            .map(|j| {
                if starts[j] <= ends[j] {
                    starts[j]..ends[j]
                } else {
                    0..0
                }
            })
            .collect();

        SpacyTokenAlignment {
            lfm_to_spacy,
            spacy_to_lfm,
        }
    }
}

/// The spacy token index whose span has the greatest byte overlap with `span`,
/// or `None` when no spacy span overlaps it.
fn best_overlap(spacy_spans: &[(usize, usize)], span: (usize, usize)) -> Option<usize> {
    let (s, e) = span;
    let mut best: Option<(usize, usize)> = None; // (index, overlap)
    for (i, &(ss, se)) in spacy_spans.iter().enumerate() {
        let overlap = e.min(se).saturating_sub(s.max(ss));
        if overlap == 0 {
            continue;
        }
        match best {
            Some((_, cur)) if cur >= overlap => {}
            _ => best = Some((i, overlap)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
#[path = "../tests/align.rs"]
mod tests;
