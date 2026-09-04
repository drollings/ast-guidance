use super::*;

fn fixture() -> CanonicalFormTable {
    CanonicalFormTable::new(
        "1.0",
        [
            ("runs".into(), "run".into()),
            ("ran".into(), "run".into()),
            ("running".into(), "run".into()),
            ("shows".into(), "show".into()),
            ("showed".into(), "show".into()),
        ],
    )
}

#[test]
fn canonical_returns_baked_form_and_identity_for_unknown() {
    let t = fixture();
    assert_eq!(t.canonical("runs"), "run");
    assert_eq!(t.canonical("ran"), "run");
    assert_eq!(t.canonical("showed"), "show");
    // Unknown surface → its own canonical form (identity, never a guess).
    assert_eq!(t.canonical("invented"), "invented");
    assert_eq!(t.canonical(""), "");
}

#[test]
fn first_wins_on_duplicate_surface() {
    let t = CanonicalFormTable::new(
        "1",
        [("run".into(), "run".into()), ("run".into(), "sprint".into())],
    );
    assert_eq!(t.canonical("run"), "run");
}

#[test]
fn from_json_parses_versioned_document() {
    let t = CanonicalFormTable::from_json(
        r#"{"version": "2.0", "map": {"went": "go", "goes": "go", "going": "go"}}"#,
    )
    .expect("parse");
    assert_eq!(t.version(), "2.0");
    assert_eq!(t.canonical("went"), "go");
    assert_eq!(t.canonical("going"), "go");
    assert_eq!(t.len(), 3);
}

#[test]
fn from_json_rejects_malformed_document() {
    assert!(matches!(
        CanonicalFormTable::from_json("not json"),
        Err(CanonicalFormError::Json(_))
    ));
}

#[test]
fn empty_table_is_identity() {
    let t = CanonicalFormTable::empty();
    assert!(t.is_empty());
    assert_eq!(t.canonical("anything"), "anything");
}

#[test]
fn version_and_entries_round_trip() {
    let t = fixture();
    assert_eq!(t.version(), "1.0");
    let entries: Vec<(&str, &str)> = t.entries().collect();
    assert_eq!(entries.len(), 5);
    assert!(entries.contains(&("ran", "run")));
}
