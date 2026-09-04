use super::*;

/// Golden values produced by compiling the authoritative
/// `MurmurHash2.cpp` `MurmurHash64A` with seed 1 (and cross-checked
/// against a pure-Python port) on 2026-08-25.
#[test]
fn hash_parity_with_spacy_murmur64a() {
    let cases: &[(&str, u64)] = &[
        ("", 14313749767032693980),
        ("hello", 5983625672228268878),
        ("hello world", 2758594965276909933),
        ("Apple", 6418411030699964375),
        ("nsubj", 1638336668109737677),
        ("ROOT", 8206900633647566924),
        ("X", 4918752717281726814),
        ("éclair", 4594911457355527430),
        ("dyn-o-mite-dave", 6759597906574577614),
        ("do n't", 16989667434526043584),
        ("’s", 614914527630368944),
        ("the", 7425985699627899538),
        ("I", 4690420944186131903),
        ("don't", 8627107437989221290),
        ("n't", 2043519015752540944),
        ("www.example.com", 11376590935768553373),
        ("3.5%", 3579893109067563421),
        ("A", 14862748245026736845),
        ("AB", 3916325639175504915),
        ("ABC", 125840725284221120),
        ("ABCD", 7946017431112816765),
        ("ABCDE", 12382937101347572856),
        ("ABCDEF", 7176860159450443390),
        ("ABCDEFG", 6037090127816012894),
        ("ABCDEFGH", 16318043397631462693),
        ("ABCDEFGHI", 8133142713256632090),
    ];

    for (s, expected) in cases {
        assert_eq!(hash_utf8(s), *expected, "hash mismatch for {s:?}");
    }
}

#[test]
fn empty_string_hashes_to_seed_mix() {
    // len 0: h = seed ^ (0 * m) = seed, then finalize.
    let h = murmur64a(b"", 1);
    assert_eq!(h, murmur64a(b"", HASH_SEED));
    assert_ne!(h, 0);
}

#[test]
fn distinct_inputs_distinct_hashes() {
    assert_ne!(hash_utf8("the"), hash_utf8("teh"));
    assert_ne!(hash_utf8("apple"), hash_utf8("Apple"));
}

#[test]
fn block_boundary_handling() {
    // 8 bytes (one full block) vs 9 bytes (block + 1 tail byte) must differ.
    assert_ne!(hash_utf8("12345678"), hash_utf8("123456789"));
    // 7 vs 8 tail transitions.
    assert_ne!(hash_utf8("1234567"), hash_utf8("12345678"));
}
