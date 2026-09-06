use super::*;

/// A shared tokenizer-backed fixture: tokenize `text` with the sample
/// WordPiece tokenizer and return its per-token byte spans.
/// Tokenizer-gated: needs the real `tokenizers` backend (`onnx` feature).
#[cfg(feature = "onnx")]
fn lfm_spans(text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, SAMPLE_TOKENIZER).unwrap();
    let tok = crate::tokenizer::LfmTokenizer::from_file(&path, 64).unwrap();
    let enc = tok.encode(text).unwrap();
    let ids = &enc.ids;
    let specials: &[u32] = &[0, 1, 2, 3]; // [PAD] [UNK] [CLS] [SEP]
    let surfaces = ids
        .iter()
        .zip(&enc.offsets)
        .map(|(&id, &(s, e))| {
            if specials.contains(&id) {
                String::new() // special tokens carry no surface text
            } else {
                text[s..e].to_string()
            }
        })
        .collect();
    (surfaces, enc.offsets)
}

/// A WordPiece tokenizer whose vocab produces subwords (`##` prefix),
/// OOV (`[UNK]`), and special tokens — enough surface for the golden
/// corpus to exercise every alignment branch.
/// Tokenizer-gated with the golden corpus below.
#[cfg(feature = "onnx")]
const SAMPLE_TOKENIZER: &str = r####"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {"id": 0, "content": "[PAD]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 1, "content": "[UNK]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 2, "content": "[CLS]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 3, "content": "[SEP]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
    ],
    "normalizer": null,
    "pre_tokenizer": {"type": "WhitespaceSplit"},
    "post_processor": null,
    "decoder": {"type": "Wordpiece", "prefix": "##", "cleanup": true, "handle_chinese_chars": true},
    "model": {
        "type": "WordPiece",
        "vocab": {
            "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
            "un": 4, "##happy": 5, "hello": 6, "world": 7, "the": 8,
            "cat": 9, "sat": 10, "##ting": 11
        },
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100
    }
}"####;

#[test]
fn exact_word_mapping() {
    let spacy_spans = vec![(0, 5), (6, 11)];
    let lfm_spans = vec![(0, 5), (6, 11)];
    let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    assert_eq!(a.lfm_to_spacy, vec![Some(0), Some(1)]);
    assert_eq!(a.spacy_to_lfm, vec![0..1, 1..2]);
    assert!(a.is_total());
}

#[test]
fn subwords_map_to_their_containing_spacy_token() {
    // spacy "unhappy" covers bytes 0..7; LFM splits it into "un" (0..2)
    // and "##happy" (2..7).
    let spacy_spans = vec![(0, 7), (8, 13)];
    let lfm_spans = vec![(0, 2), (2, 7), (8, 13)];
    let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    assert_eq!(a.lfm_to_spacy, vec![Some(0), Some(0), Some(1)]);
    assert_eq!(a.spacy_to_lfm, vec![0..2, 2..3]);
}

#[test]
fn special_tokens_map_to_nothing() {
    // [CLS] (0,0) and [SEP] (11,11) are zero-width and cover no spacy span.
    let spacy_spans = vec![(0, 5), (6, 11)];
    let lfm_spans = vec![(0, 0), (0, 5), (6, 11), (11, 11)];
    let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    assert_eq!(a.lfm_to_spacy, vec![None, Some(0), Some(1), None]);
    assert_eq!(a.spacy_to_lfm, vec![1..2, 2..3]);
    assert!(a.is_total(), "special tokens never empty a spacy range");
}

#[test]
fn oov_spacy_token_with_no_covering_lfm_yields_empty_range() {
    // spacy token 2 ("!!") covers bytes 7..9 but the LFM tokenizer merged
    // it with the previous token into one span (5,9). The merged LFM token
    // maps to the first max-overlap spacy token (1); spacy token 2 has no
    // covering LFM token → its range is empty and the caller falls back.
    let spacy_spans = vec![(0, 5), (5, 7), (7, 9)];
    let lfm_spans = vec![(0, 5), (5, 9)];
    let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    assert_eq!(a.lfm_to_spacy, vec![Some(0), Some(1)]);
    assert_eq!(a.spacy_to_lfm, vec![0..1, 1..2, 0..0]);
    assert!(!a.is_total());
}

#[test]
fn lfm_tail_past_last_spacy_token_uses_overlap() {
    // A trailing [SEP] whose span is (12, 12) → None; an LFM token
    // straddling the final boundary falls back to the last spacy token.
    let spacy_spans = vec![(0, 5)];
    let lfm_spans = vec![(0, 5), (3, 8)];
    let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    assert_eq!(a.lfm_to_spacy, vec![Some(0), Some(0)]);
}

#[test]
fn empty_inputs_align_to_empty() {
    let a = SpacyTokenAligner::align(&[], &[]);
    assert!(a.lfm_to_spacy.is_empty());
    assert!(a.spacy_to_lfm.is_empty());
}

#[test]
fn spacy_spans_from_idx_and_orth_lengths() {
    let orth = ["Hello", "world", "!"];
    let idx = [0usize, 6, 12];
    assert_eq!(
        SpacyTokenAligner::spacy_spans(&orth, &idx),
        vec![(0, 5), (6, 11), (12, 13)]
    );
}

#[test]
fn spacy_spans_handle_multibyte_utf8() {
    // "中文" is 6 bytes; idx is a byte offset.
    let orth = ["中文", "abc"];
    let idx = [0usize, 6];
    assert_eq!(
        SpacyTokenAligner::spacy_spans(&orth, &idx),
        vec![(0, 6), (6, 9)]
    );
}

// ── Golden corpus (tokenizer-gated: asserts the live tokenizer's offset
// behavior alongside the aligner's math) ──

/// A golden case: sentence text, its spacy orth list, the expected
/// `lfm_to_spacy` mapping, and the expected `spacy_to_lfm` ranges. The
/// LFM spans are produced live from `SAMPLE_TOKENIZER`, so this corpus
/// asserts BOTH the tokenizer's offset behavior and the aligner's math.
#[cfg(feature = "onnx")]
struct Golden {
    text: &'static str,
    spacy_orth: &'static [&'static str],
    spacy_idx: &'static [usize],
    lfm_to_spacy: &'static [Option<usize>],
    spacy_to_lfm: &'static [Range<usize>],
}

#[cfg(feature = "onnx")]
static GOLDEN_CORPUS: std::sync::LazyLock<Vec<Golden>> = std::sync::LazyLock::new(|| {
    vec![
        Golden {
            text: "hello world",
            spacy_orth: &["hello", "world"],
            spacy_idx: &[0, 6],
            lfm_to_spacy: &[Some(0), Some(1)],
            spacy_to_lfm: &[0..1, 1..2],
        },
        Golden {
            text: "the cat sat",
            spacy_orth: &["the", "cat", "sat"],
            spacy_idx: &[0, 4, 8],
            lfm_to_spacy: &[Some(0), Some(1), Some(2)],
            spacy_to_lfm: &[0..1, 1..2, 2..3],
        },
        // A subword split: "unhappy" → "un" + "##happy" under WordPiece.
        Golden {
            text: "unhappy world",
            spacy_orth: &["unhappy", "world"],
            spacy_idx: &[0, 8],
            lfm_to_spacy: &[Some(0), Some(0), Some(1)],
            spacy_to_lfm: &[0..2, 2..3],
        },
        // OOV: "zzqq" is not in the vocab → a single [UNK] token covering
        // the full word span.
        Golden {
            text: "zzqq hello",
            spacy_orth: &["zzqq", "hello"],
            spacy_idx: &[0, 5],
            lfm_to_spacy: &[Some(0), Some(1)],
            spacy_to_lfm: &[0..1, 1..2],
        },
    ]
});

#[cfg(feature = "onnx")]
#[test]
fn golden_corpus_aligns_live_tokenizer_output() {
    for (i, g) in GOLDEN_CORPUS.iter().enumerate() {
        let spacy_spans = SpacyTokenAligner::spacy_spans(g.spacy_orth, g.spacy_idx);
        let (lfm_surfaces, lfm_spans) = lfm_spans(g.text);
        let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
        assert_eq!(
            a.lfm_to_spacy, g.lfm_to_spacy,
            "golden {i} lfm_to_spacy ({:?} vs {})",
            lfm_surfaces, g.text,
        );
        assert_eq!(
            a.spacy_to_lfm,
            g.spacy_to_lfm,
            "golden {i} spacy_to_lfm ({:?} vs {})",
            lfm_surfaces, g.text,
        );
    }
}

#[cfg(feature = "onnx")]
#[test]
fn golden_corpus_maps_every_lfm_surface_back_through_spans() {
    // A stronger invariant: for every golden case, slicing the text by each
    // LFM token's span and looking up its spacy token's span recovers the
    // containing text — i.e. the alignment is consistent with the bytes.
    for (i, g) in GOLDEN_CORPUS.iter().enumerate() {
        let spacy_spans = SpacyTokenAligner::spacy_spans(g.spacy_orth, g.spacy_idx);
        let (lfm_surfaces, lfm_spans) = lfm_spans(g.text);
        let a = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
        for (j, mapped) in a.lfm_to_spacy.iter().enumerate() {
            let Some(s) = mapped else { continue };
            let (ss, se) = spacy_spans[*s];
            let spacy_surface = &g.text[ss..se];
            let lfm_surface = &lfm_surfaces[j];
            if !lfm_surface.is_empty() {
                assert!(
                    spacy_surface.contains(lfm_surface) || lfm_surface.contains(spacy_surface),
                    "golden {i} token {j}: spacy {spacy_surface:?} vs lfm {lfm_surface:?}"
                );
            }
        }
    }
}
