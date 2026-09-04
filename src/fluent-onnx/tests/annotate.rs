use super::*;
use crate::align::SpacyTokenAligner;
use crate::config::AnnotationLabels;

fn labels() -> AnnotationLabels {
    AnnotationLabels {
        upos: vec!["NOUN".into(), "VERB".into(), "ADJ".into(), "DET".into()],
        dep: vec!["root".into(), "nsubj".into(), "obj".into(), "det".into()],
    }
}

#[test]
fn argmax_picks_highest_index() {
    assert_eq!(argmax(&[1.0, 5.0, 2.0]), 1);
    assert_eq!(argmax(&[-1.0, -0.5, -2.0]), 1);
    assert_eq!(argmax(&[0.1, 0.1, 0.9]), 2);
}

#[test]
fn decode_heads_produces_per_token_labels_and_heads() {
    let labels = labels();
    let seq = 2;
    // token 0 → NOUN(0) nsubj(1) head=1; token 1 → VERB(1) root(0) head=1 (self).
    let upos = vec![10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0];
    let dep = vec![0.0, 9.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0];
    let mut head = vec![0.0f32; seq * seq];
    head[1] = 7.0; // token 0 → head 1
    head[seq + 1] = 9.0; // token 1 → head 1 (self)
    let decoded = decode_heads(&upos, &dep, &head, &labels, seq).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].pos, "NOUN");
    assert_eq!(decoded[0].dep, "nsubj");
    assert_eq!(decoded[0].head_abs, 1);
    assert_eq!(decoded[1].pos, "VERB");
    assert_eq!(decoded[1].dep, "root");
    assert_eq!(decoded[1].head_abs, 1);
}

#[test]
fn decode_heads_rejects_mis_shaped_tensors() {
    let labels = labels();
    // upos too short for seq=2 * n_pos=4.
    assert!(decode_heads(&[0.0], &[0.0; 8], &[0.0; 4], &labels, 2).is_err());
    // head too short for seq=2 * seq=2.
    assert!(decode_heads(&[0.0; 8], &[0.0; 8], &[0.0; 3], &labels, 2).is_err());
}

#[test]
fn argmax_of_mean_averages_subword_rows() {
    let labels = labels();
    let n_pos = labels.upos.len();
    // token0 row → VERB(1), token1 row → NOUN(0). Mean → argmax = 1 (VERB).
    let logits = [0.0, 10.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0];
    assert_eq!(argmax_of_mean(&logits, n_pos, 0..2), 1);
    // Single-token range: argmax of that row.
    assert_eq!(argmax_of_mean(&logits, n_pos, 0..1), 1);
    assert_eq!(argmax_of_mean(&logits, n_pos, 1..2), 0);
    // Empty range yields index 0 (all-zero mean → argmax 0).
    assert_eq!(argmax_of_mean(&logits, n_pos, 2..2), 0);
}

#[test]
fn aggregate_maps_subwords_and_resolves_heads_to_spacy() {
    let labels = labels();
    // spacy spans: "unhappy"(0..7), "world"(8..13). LFM: [CLS](0,0),
    // "un"(0,2), "##happy"(2,7), "world"(8,13), [SEP](13,13).
    let spacy_spans = vec![(0usize, 7usize), (8, 13)];
    let lfm_spans = vec![(0, 0), (0, 2), (2, 7), (8, 13), (13, 13)];
    let align = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    let seq = 5;
    let n_pos = labels.upos.len();
    let n_dep = labels.dep.len();

    // Per-LFM rows: token1 "un" and token2 "##happy" are the subwords of
    // spacy token 0. Give both NOUN(0). token3 "world" = VERB(1).
    let mut upos = vec![0.0f32; seq * n_pos];
    upos[1 * n_pos + 0] = 8.0; // "un" → NOUN
    upos[2 * n_pos + 0] = 9.0; // "##happy" → NOUN
    upos[3 * n_pos + 1] = 7.0; // "world" → VERB

    let mut dep = vec![0.0f32; seq * n_dep];
    dep[1 * n_dep + 2] = 6.0; // "un" → obj
    dep[2 * n_dep + 2] = 7.0; // "##happy" → obj (mean keeps obj)
    dep[3 * n_dep + 0] = 5.0; // "world" → root

    // Head rows. spacy token 0's first subword is LFM token 1 → head 3
    // (world). spacy token 1 (LFM 3) → head 0 (a spacy token) → spacy 0.
    let mut head = vec![0.0f32; seq * seq];
    head[1 * seq + 3] = 5.0; // token1 → head 3 (world, spacy 1)
    head[3 * seq + 0] = 6.0; // token3 → head 0 (special [CLS]) → None → ROOT
    let tokens = aggregate_to_spacy(&align, &upos, &dep, &head, &labels, seq).unwrap();
    assert_eq!(tokens.len(), 2);
    let t0 = tokens[0].as_ref().expect("spacy 0 covered");
    // argmax-of-mean over subwords "un"+"##happy" = NOUN / obj.
    assert_eq!(t0.pos, "NOUN");
    assert_eq!(t0.dep, "obj");
    // first subword (LFM 1) head row argmax = 3 → resolves to spacy 1.
    assert_eq!(t0.head_abs, Some(1));
    let t1 = tokens[1].as_ref().expect("spacy 1 covered");
    assert_eq!(t1.pos, "VERB");
    assert_eq!(t1.dep, "root");
    // head LFM 0 is [CLS] → maps to nothing → None → ROOT.
    assert_eq!(t1.head_abs, None);
}

#[test]
fn aggregate_returns_none_for_empty_spacy_range() {
    let labels = labels();
    // spacy token 1 ("!!") has no covering LFM subword → None (fail-open).
    let spacy_spans = vec![(0usize, 5usize), (5, 7), (7, 9)];
    let lfm_spans = vec![(0, 0), (0, 5), (5, 9)];
    let align = SpacyTokenAligner::align(&spacy_spans, &lfm_spans);
    let seq = 3;
    let upos = vec![0.0f32; seq * 4];
    let dep = vec![0.0f32; seq * 4];
    let head = vec![0.0f32; seq * seq];
    let tokens = aggregate_to_spacy(&align, &upos, &dep, &head, &labels, seq).unwrap();
    assert_eq!(tokens.len(), 3);
    assert!(tokens[2].is_none(), "spacy token with no LFM range is None");
}
