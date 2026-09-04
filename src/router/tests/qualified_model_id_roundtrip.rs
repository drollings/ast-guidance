use super::*;

#[test]
fn qualified_model_id_roundtrip_basic() {
    for wire in ["bare", "base:qualifier", "a:b", "model:instance-0"] {
        let id = QualifiedModelId::parse(wire);
        assert_eq!(id.as_wire(), wire);
        assert_eq!(format!("{id}"), wire);
    }
}

#[test]
fn qualified_model_id_malformed() {
    // empty qualifier or base should be treated as bare
    assert_eq!(QualifiedModelId::parse(":").as_wire(), ":");
    assert_eq!(QualifiedModelId::parse("base:").as_wire(), "base:");
    assert_eq!(QualifiedModelId::parse(":qual").as_wire(), ":qual");
    // empty string
    assert_eq!(QualifiedModelId::parse("").as_wire(), "");
}

#[test]
fn qualified_model_id_bare_vs_qualified() {
    let bare = QualifiedModelId::bare("base-model");
    assert!(!bare.is_qualified());
    assert_eq!(bare.as_wire(), "base-model");
    let qual = QualifiedModelId::qualified("base", "q");
    assert!(qual.is_qualified());
    assert_eq!(qual.as_wire(), "base:q");
    assert_eq!(QualifiedModelId::parse(&qual.as_wire()), qual);
}

#[test]
fn qualified_model_id_prop_roundtrip() {
    // Property: qualified(b,q).as_wire().split_once(':') == Some((b,q)) is enforced via split_model_key
    for (b, q) in [("a","b"), ("model","inst"), ("x-1","y_2"), ("base","qualifier")] {
        let id = QualifiedModelId::qualified(b,q);
        let wire = id.as_wire();
        let (pb, pq) = crate::config::split_model_key(&wire);
        assert_eq!(pb, b);
        assert_eq!(pq, Some(q));
        assert_eq!(QualifiedModelId::parse(&wire), id);
    }
}

#[test]
fn split_once_literal_guard() {
    // Guard: only two canonical sites should contain split_once(':')
    // This test documents the invariant; CI grep enforces it.
    let wire = "base:qual";
    let (b, q) = crate::config::split_model_key(wire);
    assert_eq!(b, "base");
    assert_eq!(q, Some("qual"));
}
