//! Parity benchmark: spacy-rs English tokenizer + StringStore against the
//! shared golden corpus (`src/spacy-rs/tests/data/en_tokenization.json`), the
//! byte-for-byte parity surface with pinned spaCy 3.8.15.
//!
//! Workload A (tokenize): tokenize every corpus case and materialize exactly
//! the attribute surface the golden test asserts (orth/idx/spacy/norm/lower/
//! shape/prefix/suffix + the 17 lexeme flags), folding everything into a
//! printed checksum so no work is optimized away.
//!
//! Workload B (strings): intern every distinct orth, then serialize +
//! deserialize the store (the first-wins round-trip both suites assert).
//!
//! The fixture is embedded at compile time via `CARGO_MANIFEST_DIR`, so the
//! binary is self-contained and rebuilds automatically when the corpus
//! changes. Usage: `spacy-bench [passes]`.

use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use spacy_rs::lang::en;
use spacy_rs::strings::StringStore;
use spacy_rs::vocab::Vocab;

/// The shared golden corpus fixture, embedded from the spacy-rs crate.
const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../src/spacy-rs/tests/data/en_tokenization.json"
));

#[derive(Debug, Deserialize)]
struct GoldenCase {
    text: String,
    tokens: Vec<GoldenToken>,
}

#[derive(Debug, Deserialize)]
struct GoldenToken {
    orth: String,
}

fn peak_rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == "VmHWM").then(|| {
                v.trim()
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0)
            })
        })
        .unwrap_or(0)
}

fn norm_hash(lexeme_norm: u64, token_norm: u64) -> u64 {
    if token_norm != 0 {
        token_norm
    } else {
        lexeme_norm
    }
}

fn fold_flags(acc: u64, f: &spacy_rs::lexeme::LexemeFlags) -> u64 {
    acc.wrapping_add(f.is_alpha() as u64)
        .wrapping_add((f.is_ascii() as u64) << 1)
        .wrapping_add((f.is_digit() as u64) << 2)
        .wrapping_add((f.is_lower() as u64) << 3)
        .wrapping_add((f.is_punct() as u64) << 4)
        .wrapping_add((f.is_space() as u64) << 5)
        .wrapping_add((f.is_title() as u64) << 6)
        .wrapping_add((f.is_upper() as u64) << 7)
        .wrapping_add((f.like_url() as u64) << 8)
        .wrapping_add((f.like_num() as u64) << 9)
        .wrapping_add((f.like_email() as u64) << 10)
        .wrapping_add((f.is_stop() as u64) << 11)
        .wrapping_add((f.is_bracket() as u64) << 12)
        .wrapping_add((f.is_quote() as u64) << 13)
        .wrapping_add((f.is_left_punct() as u64) << 14)
        .wrapping_add((f.is_right_punct() as u64) << 15)
        .wrapping_add((f.is_currency() as u64) << 16)
}

fn tokenize_pass(
    tokenizer: &spacy_rs::tokenizer::Tokenizer,
    strings: &spacy_rs::strings::StringStore,
    corpus: &[GoldenCase],
) -> u64 {
    let mut checksum: u64 = 0;
    for case in corpus {
        let doc = tokenizer.tokenize(&case.text).expect("tokenize");
        for t in doc.tokens() {
            checksum = checksum
                .wrapping_add(t.lexeme.orth_text(strings).len() as u64)
                .wrapping_add(t.idx as u64)
                .wrapping_add(t.spacy as u64)
                .wrapping_add(norm_hash(t.lexeme.norm, t.norm))
                .wrapping_add(t.lexeme.lower)
                .wrapping_add(t.lexeme.shape)
                .wrapping_add(t.lexeme.prefix)
                .wrapping_add(t.lexeme.suffix);
            checksum = fold_flags(checksum, &t.lexeme.flags);
        }
    }
    checksum
}

fn main() {
    let passes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let baseline_rss = peak_rss_kb();
    let t_start = Instant::now();

    let corpus: Vec<GoldenCase> = serde_json::from_str(FIXTURE).expect("parse fixture");
    let total_tokens: usize = corpus.iter().map(|c| c.tokens.len()).sum();

    let vocab = Arc::new(Vocab::new(en::lexicon_config()));
    let tokenizer = en::tokenizer(Arc::clone(&vocab)).expect("en tokenizer");
    let strings = vocab.strings();

    // Integrity + warmup (the byte-for-byte parity surface).
    for case in &corpus {
        let doc = tokenizer.tokenize(&case.text).expect("tokenize");
        assert_eq!(doc.len(), case.tokens.len(), "parity: token count");
    }
    let warm_checksum = tokenize_pass(&tokenizer, strings, &corpus);
    let startup = t_start.elapsed();

    // Workload A: tokenize + materialize the golden attribute surface.
    let t0 = Instant::now();
    let mut a_checksum: u64 = 0;
    for _ in 0..passes {
        a_checksum = a_checksum.wrapping_add(tokenize_pass(&tokenizer, strings, &corpus));
    }
    let a_elapsed = t0.elapsed();
    let a_cases = passes as u64 * corpus.len() as u64;
    let a_tokens = passes as u64 * total_tokens as u64;

    // Workload B: intern distinct orths + serialize/deserialize round-trip.
    let mut distinct: Vec<String> = Vec::new();
    for case in &corpus {
        for t in &case.tokens {
            if !distinct.contains(&t.orth) {
                distinct.push(t.orth.clone());
            }
        }
    }
    distinct.sort();
    let store = StringStore::new();
    for o in &distinct {
        store.add(o);
    }
    let t1 = Instant::now();
    let mut b_checksum: u64 = 0;
    for _ in 0..passes {
        let s = StringStore::new();
        for o in &distinct {
            s.add(o);
        }
        let bytes = s.to_bytes().expect("serialize");
        let reloaded = StringStore::from_bytes(&bytes).expect("deserialize");
        b_checksum = b_checksum.wrapping_add(reloaded.len() as u64);
    }
    let b_elapsed = t1.elapsed();

    let peak = peak_rss_kb();
    let total = t_start.elapsed();

    println!("=== spacy-rs parity benchmark ===");
    println!("passes: {passes}");
    println!("corpus_cases: {}", corpus.len());
    println!("corpus_tokens: {total_tokens}");
    println!("distinct_orths: {}", distinct.len());
    println!("startup_s: {:.3}", startup.as_secs_f64());
    println!("a_cases: {a_cases}");
    println!("a_tokens: {a_tokens}");
    println!("a_elapsed_s: {:.4}", a_elapsed.as_secs_f64());
    println!("a_cases_per_s: {:.1}", a_cases as f64 / a_elapsed.as_secs_f64());
    println!("a_tokens_per_s: {:.1}", a_tokens as f64 / a_elapsed.as_secs_f64());
    println!("a_ns_per_token: {:.0}", a_elapsed.as_nanos() as f64 / a_tokens as f64);
    println!("a_checksum: {a_checksum:x}");
    println!("b_elapsed_s: {:.4}", b_elapsed.as_secs_f64());
    println!("b_roundtrips_per_s: {:.1}", passes as f64 / b_elapsed.as_secs_f64());
    println!("b_checksum: {b_checksum:x}");
    println!("rss_baseline_kb: {baseline_rss}");
    println!("rss_peak_kb: {peak}");
    println!("rss_incremental_kb: {}", peak.saturating_sub(baseline_rss));
    println!("warm_checksum: {warm_checksum:x}");
    println!("total_s: {:.3}", total.as_secs_f64());
}