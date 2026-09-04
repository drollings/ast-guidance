use super::*;
use fluent_types::{local_id_of, InterlinguaNamespace};

fn cid(ns: InterlinguaNamespace, h: i64) -> InterlinguaId {
    InterlinguaId::new(ns, local_id_of(h))
}

fn chain_edges() -> Vec<(InterlinguaId, InterlinguaId)> {
    let animal = cid(InterlinguaNamespace::YagoClass, 1);
    let mammal = cid(InterlinguaNamespace::YagoClass, 2);
    let dog = cid(InterlinguaNamespace::YagoClass, 3);
    let poodle = cid(InterlinguaNamespace::YagoClass, 4);
    vec![(mammal, animal), (dog, mammal), (poodle, dog)]
}

#[test]
fn ancestors_walks_the_chain_nearest_first() {
    let h = TaxonomyHierarchy::from_edges(&chain_edges()).expect("edges");
    let poodle = cid(InterlinguaNamespace::YagoClass, 4);
    let ancestors = h.ancestors(poodle);
    assert_eq!(
        ancestors,
        vec![
            cid(InterlinguaNamespace::YagoClass, 3),
            cid(InterlinguaNamespace::YagoClass, 2),
            cid(InterlinguaNamespace::YagoClass, 1),
        ]
    );
}

#[test]
fn is_subclass_transitive_and_identity() {
    let h = TaxonomyHierarchy::from_edges(&chain_edges()).expect("edges");
    let animal = cid(InterlinguaNamespace::YagoClass, 1);
    let poodle = cid(InterlinguaNamespace::YagoClass, 4);
    let unrelated = cid(InterlinguaNamespace::YagoClass, 99);
    assert!(h.is_subclass(poodle, animal));
    assert!(h.is_subclass(poodle, poodle), "identity is a subclass");
    assert!(!h.is_subclass(poodle, unrelated));
    assert!(!h.is_subclass(unrelated, poodle));
}

#[test]
fn cycles_do_not_hang_or_panic() {
    // animal ← mammal and mammal ← animal form a cycle.
    let animal = cid(InterlinguaNamespace::YagoClass, 1);
    let mammal = cid(InterlinguaNamespace::YagoClass, 2);
    let h = TaxonomyHierarchy::from_edges(&[(animal, mammal), (mammal, animal)]).expect("edges");
    // Cycle-resilient DFS returns a partial result rather than looping.
    let _ = h.ancestors(animal);
    assert!(h.contains(animal) && h.contains(mammal));
}

#[test]
fn parent_only_nodes_are_registered() {
    let child = cid(InterlinguaNamespace::YagoClass, 10);
    let parent = cid(InterlinguaNamespace::YagoClass, 11);
    let h = TaxonomyHierarchy::from_edges(&[(child, parent)]).expect("edges");
    assert!(h.contains(parent), "a parent-only node is still queryable");
    assert!(h.is_subclass(child, parent));
    assert!(h.ancestors(parent).is_empty());
}

#[test]
fn unknown_id_has_no_ancestors() {
    let h = TaxonomyHierarchy::from_edges(&chain_edges()).expect("edges");
    assert!(h.ancestors(cid(InterlinguaNamespace::YagoClass, 999)).is_empty());
    assert!(!h.is_subclass(
        cid(InterlinguaNamespace::YagoClass, 999),
        cid(InterlinguaNamespace::YagoClass, 1)
    ));
}
