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
