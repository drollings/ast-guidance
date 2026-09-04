use super::*;
#[test]
fn global_view_has_pos() {
    let v = global_lemma_view();
    assert!(v.pos_count() > 0);
    assert!(v.index_contains("noun", "'hood"));
}
