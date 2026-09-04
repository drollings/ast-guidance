use super::*;
use crate::node_store::ContentNodeStore;
use crate::summarization::Summarizer;
use crate::test_stubs::CountingBackend;

fn temp_store() -> Arc<ContentNodeStore> {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-views-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    store
}

fn counting_store() -> (Arc<ContentNodeStore>, Arc<CountingBackend>) {
    let backend = Arc::new(CountingBackend::new("lazy tier text"));
    let summarizer = Summarizer::new(backend.clone(), 20);
    let store = temp_store();
    store.set_summarizer(summarizer);
    (store, backend)
}

#[test]
fn lod_bounds_and_display() {
    assert_eq!(Lod::try_from(0).unwrap().as_u8(), 0);
    assert_eq!(Lod::try_from(5).unwrap().as_u8(), 5);
    assert!(Lod::try_from(6).is_err());
    assert!(Lod::try_from(255).is_err());
    assert_eq!(Lod::LOD2.to_string(), "LOD2");
    assert_eq!(Lod::LOD0, Lod::try_from(0).unwrap());
}

#[test]
fn parallel_views_share_arc_and_derive_lazy_tier_once() {
    let (store, backend) = counting_store();
    let id = store
        .record_request("sess", "r1", "Shared full text for parallel views.")
        .unwrap();

    let v1 =
        ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
    let v2 =
        ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);

    let r1 = v1.render();
    let r2 = v2.render();
    assert_eq!(r1, "lazy tier text");
    assert_eq!(r2, "lazy tier text");
    assert_eq!(
        backend.calls(),
        1,
        "two views over one store derive the shared lazy tier exactly once"
    );
    let _ = id;
}

#[test]
fn with_override_wins_over_default_lod() {
    let store = temp_store();
    let a = store
        .record_request("sess", "r1", "short answer A")
        .unwrap();
    let _b = store
        .record_request("sess", "r2", "full text answer B")
        .unwrap();

    let view = ParallelLedger::for_session(Arc::clone(&store), "sess")
        .with_default_lod(Lod::LOD5)
        .with_override(a, Lod::LOD0);

    // a renders LOD0 (override), b renders LOD5 (label == first sentence).
    assert_eq!(view.lod_for(a), Lod::LOD0);
    assert_eq!(view.lod_for(_b), Lod::LOD5);
    let rendered = view.render();
    assert!(rendered.contains("short answer A"));
    assert!(rendered.contains("full text answer B"));
}

#[test]
fn filtered_ledger_exclusion_is_free_no_derivation() {
    let (store, backend) = counting_store();
    let dead = store
        .record_request("sess", "r1", "Dead end rejected text.")
        .unwrap();
    store
        .record_request("sess", "r2", "Live accepted text.")
        .unwrap();

    let base =
        ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
    let excluded: HashSet<NodeId> = [dead].into_iter().collect();
    let view = FilteredLedger::new(base, excluded);

    let rendered = view.render();
    assert!(!rendered.contains("Dead end"), "excluded node removed");
    assert!(rendered.contains("lazy tier text"), "live node rendered");
    assert_eq!(
        backend.calls(),
        1,
        "only the live node triggers derivation — exclusion is free"
    );
}

#[test]
fn pii_redacted_view_scrubs_email_keeps_clean() {
    let store = temp_store();
    store
        .record_request("sess", "r1", "Contact user@example.com now")
        .unwrap();
    store
        .record_request("sess", "r2", "plain clean text")
        .unwrap();

    let base =
        ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD0);
    let view = pii_redacted(base, HashSet::new());

    let rendered = view.render();
    assert!(
        rendered.contains("[REDACTED:email]"),
        "email-bearing node scrubbed, got: {rendered}"
    );
    assert!(!rendered.contains("user@example.com"));
    assert!(rendered.contains("plain clean text"));
}

#[test]
fn render_degrades_to_lod0_without_summarizer() {
    let store = temp_store();
    store
        .record_request("sess", "r1", "Full text, no summarizer present.")
        .unwrap();

    // LOD2 is un-derivable (no summarizer) → degrade to LOD0, never error.
    let view =
        ParallelLedger::for_session(Arc::clone(&store), "sess").with_default_lod(Lod::LOD2);
    assert_eq!(view.render(), "Full text, no summarizer present.");
}

#[test]
fn render_of_empty_session_is_empty_string() {
    let store = temp_store();
    let view = ParallelLedger::for_session(store, "absent");
    assert_eq!(view.render(), "");
}
