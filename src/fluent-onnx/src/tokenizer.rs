//! LFM tokenizer wrapper — the first consumer of the `tokenizers` crate.
//!
//! `LfmTokenizer` wraps a HuggingFace `tokenizer.json` with config-driven
//! truncation and exposes byte-offset output (`encoding.get_offsets()`), which
//! everything later (two-tower label-region pooling, LFM↔spacy alignment)
//! depends on. Offsets are byte offsets into the source string, valid for
//! `&text[start..end]` slicing.

use std::path::Path;
use std::sync::Arc;

use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::error::OrtError;

/// A single tokenized sequence with the tensors a session needs.
#[derive(Debug, Clone)]
pub struct LfmEncoding {
    /// Token ids (`u32`), truncated to `max_seq_len`.
    pub ids: Vec<u32>,
    /// Attention mask (1 = real token, 0 = pad), aligned with `ids`.
    pub attention_mask: Vec<u32>,
    /// Per-token byte offsets into the source string.
    pub offsets: Vec<(usize, usize)>,
}

impl LfmEncoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// The LFM tokenizer, shareable via `Arc` (immutable after construction).
pub struct LfmTokenizer {
    inner: Tokenizer,
    max_seq_len: usize,
}

impl LfmTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file, truncating to
    /// `max_seq_len` on the right (LongestFirst) — the encoder's hard cap.
    pub fn from_file(path: &Path, max_seq_len: usize) -> Result<Arc<Self>, OrtError> {
        let mut inner = Tokenizer::from_file(path)
            .map_err(|e| OrtError::tokenization(format!("load {}: {e}", path.display())))?;
        inner
            .with_truncation(Some(TruncationParams {
                max_length: max_seq_len,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| OrtError::tokenization(format!("set truncation: {e}")))?;
        Ok(Arc::new(Self { inner, max_seq_len }))
    }

    /// The configured truncation cap.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Encode a single string with truncation, keeping ids, attention mask,
    /// and per-token byte offsets.
    pub fn encode(&self, text: &str) -> Result<LfmEncoding, OrtError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| OrtError::tokenization(format!("encode: {e}")))?;
        Ok(LfmEncoding {
            ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
        })
    }

    /// Decode a token-id sequence back into text, skipping special tokens.
    /// Used by the generative decoder to render a generation.
    pub fn decode(&self, ids: &[u32]) -> Result<String, OrtError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| OrtError::tokenization(format!("decode: {e}")))
    }

    /// Access the inner `tokenizers::Tokenizer` (for vocab introspection).
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
#[path = "../tests/tokenizer.rs"]
mod tests;
