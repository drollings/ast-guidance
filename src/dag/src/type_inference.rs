//! Ontology type-hierarchy inference via bitvector transitive closure.
//!
//! Each class in an ontology occupies a bit position. For every class `C`
//! the struct stores a `BitVec` of all ancestors reachable through
//! `[child, parent]` subclass edges — including `C` itself (every class
//! is its own ancestor). Once built, `is_subclass_of` is an O(1) bit
//! check on the precomputed ancestor set.
//!
//! # Algorithm
//!
//! - **Initialisation**: each class's ancestor `BitVec` has exactly its
//!   own bit set.
//! - **Fixpoint loop**: iterate all `[child, parent]` edges. For each
//!   edge, merge the parent's ancestor set (and the parent itself) into
//!   the child's ancestor set. Any change to any `BitVec` restarts the
//!   loop. Batched updates avoid borrow-conflicts with the read-only
//!   iteration pass.
//!
//! Worst-case O(N³) for pathological DAGs (N classes, worst-case
//! fixpoint rounds), but ontology hierarchies are typically shallow
//! (tens to low hundreds of classes), so the fixpoint converges quickly.
//!
//! # Example
//!
//! ```
//! use fluent_dag::type_inference::TypeInference;
//!
//! // Animal (1) ← Mammal (2) ← Cat (3)
//! let ti = TypeInference::build(&[1, 2, 3], &[[2, 1], [3, 2]]);
//! assert!(ti.is_subclass_of(3, 1));  // Cat → Animal (transitive)
//! assert!(ti.is_subclass_of(2, 1));  // Mammal → Animal (direct)
//! assert!(!ti.is_subclass_of(1, 3)); // Animal is not a subclass of Cat
//! ```

use bitvec::prelude::*;
use std::collections::HashMap;

/// A precomputed transitive-closure index over an ontology's inheritance
/// graph.
///
/// Construct via [`TypeInference::build`], then query with
/// [`TypeInference::is_subclass_of`]. Once built the struct is
/// read-only — the subclass edges are baked into the bitvectors and
/// cannot be updated without a full rebuild.
///
/// Each class is identified by an `i64` id. Edges are directed
/// `[child, parent]` pairs.
#[derive(Debug)]
pub struct TypeInference {
    /// `class_id → BitVec` where bit *i* is set when class *i* is an
    /// ancestor of `class_id` (including `class_id` itself).
    ancestors: HashMap<i64, BitVec>,
    /// Total number of classes registered during [`build`].
    class_count: usize,
    /// Bijection between class `i64` ids and bit-vector positions.
    id_to_bit: HashMap<i64, usize>,
}

impl TypeInference {
    /// Build the transitive ancestor closure from a set of class ids and
    /// `[child, parent]` subclass edges.
    ///
    /// Every class is its own ancestor. The algorithm iterates edges in a
    /// fixpoint loop: each iteration merges every parent's ancestor set
    /// into its child's ancestor set, restarting whenever any set grows.
    /// Convergence is guaranteed because each class has a fixed bit width
    /// and a bit can only transition from 0 → 1.
    ///
    /// # Panics
    ///
    /// Never panics. `class_ids` and `edges` may reference ids not in
    /// either list — edges with unknown children or parents are silently
    /// skipped. An empty `class_ids` slice produces a valid (empty) graph.
    pub fn build(class_ids: &[i64], edges: &[[i64; 2]]) -> Self {
        let class_count = class_ids.len();
        let mut id_to_bit: HashMap<i64, usize> = HashMap::new();
        for (i, &id) in class_ids.iter().enumerate() {
            id_to_bit.insert(id, i);
        }
        let mut ancestors: HashMap<i64, BitVec> = HashMap::new();
        for &id in class_ids {
            let mut bs = BitVec::repeat(false, class_count);
            if let Some(&bit) = id_to_bit.get(&id) {
                bs.set(bit, true);
            }
            ancestors.insert(id, bs);
        }
        let mut changed = true;
        while changed {
            changed = false;
            let mut updates: Vec<(i64, BitVec)> = Vec::new();
            for &[child, parent] in edges {
                let parent_bit = id_to_bit.get(&parent);
                if let Some(&pb) = parent_bit {
                    if let Some(child_ancestors) = ancestors.get(&child) {
                        if !child_ancestors[pb] || {
                            let parent_ancestors = ancestors.get(&parent);
                            parent_ancestors.is_some_and(|pa| {
                                pa.iter()
                                    .enumerate()
                                    .any(|(i, b)| *b && !child_ancestors[i])
                            })
                        } {
                            let mut new_bits = child_ancestors.clone();
                            new_bits.set(pb, true);
                            if let Some(parent_ancestors) = ancestors.get(&parent) {
                                for (i, bit) in parent_ancestors.iter().enumerate() {
                                    if *bit && !new_bits[i] {
                                        new_bits.set(i, true);
                                        changed = true;
                                    }
                                }
                            }
                            if new_bits != *child_ancestors {
                                changed = true;
                            }
                            updates.push((child, new_bits));
                        }
                    }
                }
            }
            for (id, bits) in updates {
                ancestors.insert(id, bits);
            }
        }
        Self {
            ancestors,
            class_count,
            id_to_bit,
        }
    }

    /// Returns `true` when `child` is a transitive subclass of `parent`.
    ///
    /// Every class is its own subclass — `ti.is_subclass_of(1, 1)` is
    /// always `true` when class `1` was registered. Unregistered ids
    /// return `false` without panicking.
    ///
    /// O(1): a single bit check on the precomputed ancestor `BitVec`.
    pub fn is_subclass_of(&self, child: i64, parent: i64) -> bool {
        if let (Some(_cb), Some(pb)) = (self.id_to_bit.get(&child), self.id_to_bit.get(&parent)) {
            if let Some(child_ancestors) = self.ancestors.get(&child) {
                return child_ancestors[*pb];
            }
        }
        false
    }

    /// Number of classes registered during [`build`].
    pub fn class_count(&self) -> usize {
        self.class_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ontology() {
        let ti = TypeInference::build(&[], &[]);
        assert_eq!(ti.class_count(), 0);
    }

    #[test]
    fn class_is_subclass_of_itself() {
        let ti = TypeInference::build(&[1], &[]);
        assert!(ti.is_subclass_of(1, 1));
    }

    #[test]
    fn direct_subclass() {
        let ti = TypeInference::build(&[1, 2], &[[2, 1]]);
        assert!(ti.is_subclass_of(2, 1));
    }

    #[test]
    fn transitive_subclass() {
        let ti = TypeInference::build(&[1, 2, 3], &[[2, 1], [3, 2]]);
        assert!(ti.is_subclass_of(2, 1));
        assert!(ti.is_subclass_of(3, 2));
        assert!(ti.is_subclass_of(3, 1));
    }

    #[test]
    fn unknown_class_returns_false() {
        let ti = TypeInference::build(&[1, 2], &[[2, 1]]);
        assert!(!ti.is_subclass_of(99, 1));
        assert!(!ti.is_subclass_of(2, 99));
    }
}
