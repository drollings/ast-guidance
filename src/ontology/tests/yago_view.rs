//! Moved with the view (M5.5): the `YagoView` TTL/load suite now lives
//! beside its owner.
use super::*;
#[test]
fn ttl_parses_n2() {
    let ttl = std::fs::read_to_string("env/yago-taxonomy-n2.ttl").or_else(|_| std::fs::read_to_string("../../env/yago-taxonomy-n2.ttl")).expect("n2 ttl");
    let view = YagoView::from_ttl_str(&ttl);
    assert!(view.class_count() > 0);
    let thing = view.resolve_curie("schema:Thing").expect("Thing");
    let person = view.resolve_curie("schema:Person").expect("Person");
    assert!(view.is_subclass_of(person, thing));
}

#[test]
fn corrupt_blob_falls_back_to_inmemory() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("corrupt.bin");
    std::fs::write(&path, b"not a valid yago blob nor ttl").expect("write");
    // load should error (sha mismatch / corrupt) — caller then falls back to InMemory provisional
    let res = YagoView::load(&path);
    assert!(res.is_err() || res.unwrap().class_count() == 0, "corrupt blob must not load as valid Ready view");
}

#[test]
fn sha_mismatch_is_detected() {
    // E5: download pin — wrong sha aborts. Simulate via header magic mismatch.
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("bad.bin");
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(&0x5953_4D31u32.to_le_bytes());
    std::fs::write(&path, data).expect("write");
    let res = YagoView::load(&path);
    assert!(res.is_err(), "YSM1 magic with bad payload must be rejected");
}
