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
