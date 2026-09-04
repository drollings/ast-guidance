//! Bit-set drift analysis: compute "missing capability" follow-ups from `BitVec` masks.

use bitvec::prelude::*;
use common_core::drift::*;
use internment::ArcIntern;
use std::collections::HashMap;


fn make_interner(names: &[&str]) -> HashMap<ArcIntern<str>, usize> {
        let mut m = HashMap::new();
        for (i, &name) in names.iter().enumerate() {
            m.insert(ArcIntern::from(name), i);
        }
        m
}

#[test]
fn is_resolved_empty_needed_always_true() {
        let needed = BitVec::repeat(false, 8);
        let available = BitVec::repeat(false, 8);
        assert!(BitSetDrift::is_resolved(&needed, &available));
        let available2 = BitVec::repeat(true, 8);
        assert!(BitSetDrift::is_resolved(&needed, &available2));
}

#[test]
fn is_resolved_fully_covered() {
        let mut needed = BitVec::repeat(false, 8);
        needed.set(1, true);
        needed.set(3, true);
        let mut available = BitVec::repeat(false, 8);
        available.set(1, true);
        available.set(3, true);
        available.set(5, true);
        assert!(BitSetDrift::is_resolved(&needed, &available));
}

#[test]
fn is_resolved_partially_covered() {
        let mut needed = BitVec::repeat(false, 8);
        needed.set(1, true);
        needed.set(3, true);
        let mut available = BitVec::repeat(false, 8);
        available.set(1, true);
        assert!(!BitSetDrift::is_resolved(&needed, &available));
}

#[test]
fn generate_follow_ups_produces_follow_ups() {
        let interner = make_interner(&["compile", "link", "test"]);
        let drift = BitSetDrift::new(interner);
        let mut needed = BitVec::repeat(false, 3);
        needed.set(0, true);
        needed.set(1, true);
        let mut available = BitVec::repeat(false, 3);
        available.set(0, true);
        let follow_ups = drift.generate_follow_ups(&needed, &available);
        assert_eq!(follow_ups, vec!["Provide link"]);
}

#[test]
fn generate_follow_ups_no_follow_ups_when_resolved() {
        let interner = make_interner(&["compile", "link"]);
        let drift = BitSetDrift::new(interner);
        let needed = BitVec::repeat(false, 2);
        let available = BitVec::repeat(false, 2);
        let follow_ups = drift.generate_follow_ups(&needed, &available);
        assert!(follow_ups.is_empty());
}
