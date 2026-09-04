//! spaCy-compatible hashing: MurmurHash64A with seed 1.
//!
//! spaCy interns every string as a `uint64_t` hash via
//! `hash_utf8` → `hash64(data, len, 1)` (`spacy/strings.pyx:67-68`), which is
//! Austin Appleby's **MurmurHash64A** (see `murmurhash/MurmurHash2.h` in the
//! `explosion/murmurhash` package). This module reproduces that function
//! byte-exactly so saved `.spacy`/model data and golden hash tables from a
//! real spaCy install interoperate.
//!
//! Verified against the authoritative C implementation on 2026-08-25 for a
//! corpus of empty, short, long, multi-byte-UTF-8 and multi-block inputs
//! (see the `hash_parity` tests).

/// The fixed seed spaCy passes to `hash64`.
pub const HASH_SEED: u64 = 1;

/// MurmurHash64A, matching `murmurhash.mrmr.hash64(data, len, seed)`.
///
/// Reads 8-byte blocks little-endian (matching the little-endian fast path of
/// the C `getblock`), mixes the 1..7-byte tail with a fall-through switch, and
/// finalizes with two `h ^= h >> 47` rounds. Pure, allocation-free, and safe.
#[must_use]
pub fn murmur64a(data: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;

    let len = data.len();
    let mut h = seed ^ (len as u64).wrapping_mul(M);

    let mut i = 0;
    while i + 8 <= len {
        let mut k = u64::from_le_bytes(data[i..i + 8].try_into().expect("slice is 8 bytes"));
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
        i += 8;
    }

    let mut tail = 0u64;
    let remaining = len - i;
    // The C `switch` falls through, accumulating every tail byte; sequential
    // `if`s reproduce that exactly.
    if remaining >= 7 {
        tail |= u64::from(data[i + 6]) << 48;
    }
    if remaining >= 6 {
        tail |= u64::from(data[i + 5]) << 40;
    }
    if remaining >= 5 {
        tail |= u64::from(data[i + 4]) << 32;
    }
    if remaining >= 4 {
        tail |= u64::from(data[i + 3]) << 24;
    }
    if remaining >= 3 {
        tail |= u64::from(data[i + 2]) << 16;
    }
    if remaining >= 2 {
        tail |= u64::from(data[i + 1]) << 8;
    }
    if remaining >= 1 {
        tail |= u64::from(data[i]);
    }
    if remaining != 0 {
        h ^= tail;
        h = h.wrapping_mul(M);
    }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// spaCy's `hash_utf8` / `hash_string`: MurmurHash64A over the UTF-8 bytes
/// with the fixed seed.
#[must_use]
pub fn hash_utf8(s: &str) -> u64 {
    murmur64a(s.as_bytes(), HASH_SEED)
}

#[cfg(test)]
#[path = "../tests/hash.rs"]
mod tests;
