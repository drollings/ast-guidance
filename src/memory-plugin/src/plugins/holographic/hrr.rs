//! Holographic Reduced Representations — pure Rust implementation.
//!
//! Phase vector HRR algebra with no numpy dependency. Uses `f64` phase
//! vectors with modular arithmetic for deterministic, cross-platform
//! compositional semantics.
//!
//! # Mathematical Foundation
//!
//! HRRs (Plate 1995) encode compositional structure into fixed-width
//! distributed representations using phase vectors:
//!
//! - `bind(a, b)` = circular convolution = element-wise phase addition mod 2π
//! - `unbind(memory, key)` = circular correlation = phase subtraction mod 2π
//! - `bundle(vectors)` = superposition via circular mean of complex exponentials
//!
//! Atoms are generated deterministically from SHA-256 so representations
//! are identical across processes, machines, and language versions.

use common_core::hash::sha256_digest;

/// 2π — the full circle in radians.
pub const TWO_PI: f64 = std::f64::consts::TAU;

/// Deterministic phase vector from SHA-256 counter blocks.
///
/// Each concept maps to a unique vector of phases in [0, 2π).
/// The mapping is deterministic: `encode_atom("foo", 1024)` always
/// produces the same vector regardless of platform or process.
///
/// # Algorithm
///
/// 1. Hash `"{word}:{i}"` for i = 0, 1, 2, ...
/// 2. Interpret each 32-byte digest as 16 `u16` values
/// 3. Scale to [0, 2π): `phase = value * (2π / 65536)`
/// 4. Truncate to `dim` elements
pub fn encode_atom(word: &str, dim: usize) -> Vec<f64> {
    let values_per_block = 16; // 32 bytes / 2 bytes per u16
    let blocks_needed = dim.div_ceil(values_per_block);

    let mut uint16_values: Vec<u16> = Vec::with_capacity(dim);
    for i in 0..blocks_needed {
        let input = format!("{word}:{i}");
        let digest = sha256_digest(input.as_bytes());
        for chunk in digest.chunks_exact(2) {
            uint16_values.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }

    uint16_values[..dim]
        .iter()
        .map(|&v| (v as f64) * (TWO_PI / 65536.0))
        .collect()
}

/// Circular convolution: element-wise phase addition mod 2π.
///
/// Binding associates two concepts into a single composite vector.
/// The result is dissimilar to both inputs (quasi-orthogonal).
pub fn bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "bind: vector length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai + bi) % TWO_PI)
        .collect()
}

/// Circular correlation: element-wise phase subtraction mod 2π.
///
/// Unbinding retrieves the value associated with a key from a memory vector.
/// `unbind(bind(a, b), a) ≈ b` (up to superposition noise).
pub fn unbind(memory: &[f64], key: &[f64]) -> Vec<f64> {
    assert_eq!(memory.len(), key.len(), "unbind: vector length mismatch");
    memory
        .iter()
        .zip(key.iter())
        .map(|(&mi, &ki)| (mi - ki) % TWO_PI)
        .collect()
}

/// Superposition via circular mean of complex exponentials.
///
/// Bundling merges multiple vectors into one that is similar to each input.
/// The result can hold O(√dim) items before similarity degrades.
pub fn bundle(vectors: &[Vec<f64>]) -> Vec<f64> {
    assert!(!vectors.is_empty(), "bundle: empty input");
    let dim = vectors[0].len();
    let _n = vectors.len() as f64;
    (0..dim)
        .map(|j| {
            let sum_re: f64 = vectors.iter().map(|v| v[j].cos()).sum();
            let sum_im: f64 = vectors.iter().map(|v| v[j].sin()).sum();
            sum_im.atan2(sum_re).rem_euclid(TWO_PI)
        })
        .collect()
}

/// Phase cosine similarity. Range [-1, 1].
///
/// Returns 1.0 for identical vectors, ~0.0 for random vectors,
/// -1.0 for perfectly anti-correlated vectors.
pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "similarity: vector length mismatch");
    let n = a.len() as f64;
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).cos())
        .sum();
    sum / n
}

/// Encode a bag-of-words text into a single phase vector.
///
/// Tokenizes by lowercasing, splitting on whitespace, and stripping
/// leading/trailing punctuation. Returns the bundle of all token atoms.
/// If text produces no tokens, returns `encode_atom("__hrr_empty__", dim)`.
pub fn encode_text(text: &str, dim: usize) -> Vec<f64> {
    let tokens: Vec<&str> = text
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return encode_atom("__hrr_empty__", dim);
    }

    let atoms: Vec<Vec<f64>> = tokens
        .iter()
        .map(|t| encode_atom(&t.to_lowercase(), dim))
        .collect();
    bundle(&atoms)
}

/// Structured fact encoding: content bound to ROLE_CONTENT,
/// each entity bound to ROLE_ENTITY, all bundled.
///
/// This enables algebraic extraction:
/// `unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector`
pub fn encode_fact(content: &str, entities: &[String], dim: usize) -> Vec<f64> {
    let role_content = encode_atom("__hrr_role_content__", dim);
    let role_entity = encode_atom("__hrr_role_entity__", dim);

    let mut components: Vec<Vec<f64>> = vec![bind(&encode_text(content, dim), &role_content)];

    for entity in entities {
        components.push(bind(
            &encode_atom(&entity.to_lowercase(), dim),
            &role_entity,
        ));
    }

    bundle(&components)
}

/// Serialize phase vector to bytes. `f64` → 8 bytes per element.
pub fn phases_to_bytes(phases: &[f64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(phases.len() * 8);
    for &p in phases {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    buf
}

/// Deserialize bytes back to phase vector.
pub fn bytes_to_phases(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Signal-to-noise ratio estimate for holographic storage.
///
/// `SNR = √(dim / n_items)` when n_items > 0, else `∞`.
///
/// SNR falls below 2.0 when `n_items > dim / 4`, meaning retrieval
/// errors become likely.
pub fn snr_estimate(dim: usize, n_items: usize) -> f64 {
    if n_items == 0 {
        return f64::INFINITY;
    }
    (dim as f64 / n_items as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_atom_deterministic() {
        let a = encode_atom("hello", 128);
        let b = encode_atom("hello", 128);
        assert_eq!(a, b);
        assert_eq!(a.len(), 128);
    }

    #[test]
    fn encode_atom_different_words() {
        let a = encode_atom("foo", 128);
        let b = encode_atom("bar", 128);
        assert_ne!(a, b);
    }

    #[test]
    fn bind_unbind_roundtrip() {
        let a = encode_atom("alpha", 256);
        let b = encode_atom("beta", 256);
        let bound = bind(&a, &b);
        let recovered = unbind(&bound, &a);
        // recovered ≈ b (up to superposition noise)
        let sim = similarity(&recovered, &b);
        assert!(sim > 0.3, "bind/unbind roundtrip similarity too low: {sim}");
    }

    #[test]
    fn bundle_similarity() {
        let a = encode_atom("x", 512);
        let b = encode_atom("y", 512);
        let bundled = bundle(&[a.clone(), b.clone()]);
        let sim_a = similarity(&bundled, &a);
        let sim_b = similarity(&bundled, &b);
        assert!(sim_a > 0.1, "bundle should be similar to input a: {sim_a}");
        assert!(sim_b > 0.1, "bundle should be similar to input b: {sim_b}");
    }

    #[test]
    fn encode_text_empty() {
        let v = encode_text("", 128);
        let empty = encode_atom("__hrr_empty__", 128);
        assert_eq!(v, empty);
    }

    #[test]
    fn encode_fact_with_entities() {
        let fact = encode_fact("Alice likes Rust", &["Alice".into(), "Rust".into()], 256);
        assert_eq!(fact.len(), 256);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn bytes_roundtrip() {
        let phases = vec![0.0, 1.0, 2.0, 3.14];
        let bytes = phases_to_bytes(&phases);
        let recovered = bytes_to_phases(&bytes);
        assert_eq!(phases, recovered);
    }

    #[test]
    fn snr_degrades_with_count() {
        let dim = 1024;
        assert_eq!(snr_estimate(dim, 0), f64::INFINITY);
        assert!(snr_estimate(dim, 1) > 30.0);
        assert!(snr_estimate(dim, 256) > 1.9);
        assert!(snr_estimate(dim, 1024) > 0.9);
    }

    #[test]
    fn bind_commutative() {
        let a = encode_atom("x", 64);
        let b = encode_atom("y", 64);
        let ab = bind(&a, &b);
        let ba = bind(&b, &a);
        assert_eq!(ab, ba);
    }
}
