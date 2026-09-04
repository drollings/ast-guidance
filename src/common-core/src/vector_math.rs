//! Scalar embedding-vector math (P1 canonical home).
//!
//! `cosine_similarity_f32` was moved verbatim from `fluent-db::vector`
//! (`cosine_similarity`) so the dependency direction stays acyclic:
//! `fluent-db` re-exports it, and crates that must not depend on
//! `fluent-db`'s sqlite/HNSW weight (notably spacy-rs, the deterministic
//! spine) compose it from here. Byte-identical contract: `0.0` on length
//! mismatch, empty input, or zero magnitude; NaN propagates per IEEE-754.

/// Cosine similarity between two equal-length vectors. Returns `0.0` when the
/// vectors have mismatched lengths or are empty.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let mag = na.sqrt() * nb.sqrt();
    if mag == 0.0 {
        0.0
    } else {
        dot / mag
    }
}
