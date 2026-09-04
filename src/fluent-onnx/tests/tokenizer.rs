use super::*;
use std::path::PathBuf;

fn write_sample_tokenizer(dir: &Path) -> PathBuf {
    // A minimal BPE-style tokenizer that splits words on whitespace and
    // punctuation so subword offsets are observable.
    let json = r####"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "[PAD]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
            {"id": 1, "content": "[CLS]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
            {"id": 2, "content": "[SEP]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": {"type": "Wordpiece", "prefix": "##", "cleanup": true, "handle_chinese_chars": true},
        "model": {
            "type": "WordPiece",
            "vocab": {
                "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                "hello": 4, "world": 5, "the": 6, "cat": 7
            },
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100
        }
    }"####;
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    path
}

#[test]
fn encode_returns_ids_mask_and_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let tok = LfmTokenizer::from_file(&write_sample_tokenizer(dir.path()), 16).unwrap();
    let enc = tok.encode("hello world").unwrap();
    assert_eq!(enc.ids, vec![4, 5]);
    assert_eq!(enc.attention_mask, vec![1, 1]);
    assert_eq!(enc.offsets, vec![(0, 5), (6, 11)]);
    assert_eq!(&"hello world"[enc.offsets[0].0..enc.offsets[0].1], "hello");
}

#[test]
fn truncation_is_applied() {
    let dir = tempfile::tempdir().unwrap();
    let tok = LfmTokenizer::from_file(&write_sample_tokenizer(dir.path()), 1).unwrap();
    let enc = tok.encode("hello world").unwrap();
    assert_eq!(enc.len(), 1);
    assert_eq!(tok.max_seq_len(), 1);
}

#[test]
fn missing_file_is_an_error() {
    let err = LfmTokenizer::from_file(Path::new("/nonexistent/tokenizer.json"), 16);
    assert!(matches!(err, Err(OrtError::Tokenization { .. })));
}
