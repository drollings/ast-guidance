//! Crate-typed test fixtures shared by fluent-dag Tier-1 suites.
//!
//! Every helper that ≥2 suites need lives here — never copy it into a new
//! test module. `make_bitset` and `make_registry` are the migrated homes of
//! the former `resolver.rs` / `closure.rs` / `narrowing.rs` /
//! `target_work_unit.rs` duplicates (see `ROADMAP_20260816_TESTS.md` §1.3).

use bitvec::vec::BitVec;
use fluent_types::{ExecutorKind, TargetType};

use crate::target::{Target, TargetRegistry};

/// Build a `BitVec` with the given bits set (bit indices as `usize`).
pub fn make_bitset(bits: &[usize]) -> BitVec {
    let max = bits.iter().max().copied().unwrap_or(0) + 1;
    let mut bv = BitVec::with_capacity(max);
    bv.resize(max, false);
    for &bit in bits {
        if bit < bv.len() {
            bv.set(bit, true);
        }
    }
    bv
}

/// Register every target into a fresh `TargetRegistry`.
pub fn make_registry(targets: Vec<Target>) -> TargetRegistry {
    let mut reg = TargetRegistry::new();
    for t in targets {
        reg.register(t).unwrap();
    }
    reg
}

/// Build a `Target` with a numeric-provides/depends `BitVec` (bit indices).
#[allow(clippy::too_many_arguments)]
pub fn target(id: i64, name: &str, depends: &[usize], provides: &[usize], essential: bool) -> Target {
    Target::new()
        .id(id)
        .name(name.into())
        .target_type(TargetType::File)
        .executor(ExecutorKind::Native)
        .depends(make_bitset(depends))
        .provides(make_bitset(provides))
        .essential(essential)
        .build()
}

/// The canonical 3-target linear chain: `0 (compile) → 1 (link) → 2 (build)`.
pub fn linear_chain() -> Vec<Target> {
    vec![
        target(0, "compile", &[], &[0], false),
        target(1, "link", &[0], &[1], false),
        target(2, "build", &[1], &[2], false),
    ]
}

/// The canonical 4-target diamond: `base → left|right → top`.
pub fn diamond() -> Vec<Target> {
    vec![
        target(0, "base", &[], &[0], false),
        target(1, "left", &[0], &[1], false),
        target(2, "right", &[0], &[2], false),
        target(3, "top", &[1, 2], &[3], false),
    ]
}

/// A single self-providing target: depends on `0`, provides `0`.
pub fn self_providing() -> Vec<Target> {
    vec![target(0, "self_provider", &[0], &[0], false)]
}