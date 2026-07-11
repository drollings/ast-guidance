//! Bit-set drift analysis: compute "missing capability" follow-ups from `BitVec` masks.

use bitvec::prelude::*;
use internment::ArcIntern;
use std::collections::HashMap;

pub struct BitSetDrift {
    interner: HashMap<ArcIntern<str>, usize>,
    names: Vec<ArcIntern<str>>,
}

impl BitSetDrift {
    pub fn new(interner: HashMap<ArcIntern<str>, usize>) -> Self {
        let max_idx = interner.values().copied().max().unwrap_or(0);
        let mut names = vec![ArcIntern::from(""); max_idx + 1];
        for (name, &idx) in &interner {
            names[idx] = name.clone();
        }
        Self { interner, names }
    }

    pub fn generate_follow_ups(&self, needed: &BitVec, available: &BitVec) -> Vec<String> {
        let missing = needed.clone() & !available.clone();
        let mut follow_ups = Vec::new();
        for (name, &idx) in &self.interner {
            if idx < missing.len() && missing[idx] {
                follow_ups.push(format!("Provide {name}"));
            }
        }
        follow_ups.sort();
        follow_ups
    }

    pub fn is_resolved(needed: &BitVec, available: &BitVec) -> bool {
        if needed.count_ones() == 0 {
            return true;
        }
        let missing = needed.clone() & !available.clone();
        missing.count_ones() == 0
    }

    pub fn name_for_index(&self, idx: usize) -> Option<&str> {
        self.names.get(idx).map(AsRef::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
