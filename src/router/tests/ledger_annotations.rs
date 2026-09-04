use super::*;

fn claim(key: &str, tier: ProvenanceTier, status: ClaimStatus) -> AnnotationClaim {
    AnnotationClaim {
        claim_key: key.to_string(),
        tier,
        status,
        payload: serde_json::json!({ "key": key }),
        produced_by: "test".to_string(),
        produced_at: 1,
    }
}

fn store() -> AnnotationStore {
    // The table is created by the ledger migration; run it directly here so
    // the store is testable in isolation.
    let db = Arc::new(SqliteStore::open_in_memory().unwrap());
    db.init_schema(ANNOTATION_SCHEMA).unwrap();
    AnnotationStore::new(db)
}

const ANNOTATION_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS ledger_annotations (
    content_hash INTEGER NOT NULL,
    claim_key TEXT NOT NULL,
    claim_id INTEGER NOT NULL,
    tier TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    produced_by TEXT NOT NULL,
    produced_at INTEGER NOT NULL,
    PRIMARY KEY (content_hash, claim_key, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_ledger_annotations_active
    ON ledger_annotations(content_hash, claim_key, claim_id);";

#[test]
fn first_write_establishes_confirmed() {
    let store = store();
    let status = store
        .write(100, &claim("frame:x", ProvenanceTier::Deterministic, ClaimStatus::Confirmed))
        .unwrap();
    assert_eq!(status, ClaimStatus::Confirmed);
    let got = store.read(100, "frame:x").unwrap().expect("present");
    assert_eq!(got.status, ClaimStatus::Confirmed);
    assert_eq!(got.tier, ProvenanceTier::Deterministic);
}

/// DRY guard: the plain snake_case storage text must never drift from the
/// enum's canonical serde wire form (so the match helpers stay in lockstep
/// with `ProvenanceTier`/`ClaimStatus`'s own `rename_all`).
#[test]
fn storage_text_matches_serde_snake_case() {
    for tier in [
        ProvenanceTier::Deterministic,
        ProvenanceTier::LocalModel,
        ProvenanceTier::Frontier,
        ProvenanceTier::HumanReview,
    ] {
        let serde_form: String = serde_json::from_value(serde_json::to_value(tier).unwrap()).unwrap();
        assert_eq!(tier_as_text(tier), serde_form, "{tier:?}");
        assert_eq!(tier_from_text(tier_as_text(tier)).unwrap(), tier);
    }
    for status in [ClaimStatus::Provisional, ClaimStatus::Confirmed, ClaimStatus::Superseded] {
        let serde_form: String = serde_json::from_value(serde_json::to_value(status).unwrap()).unwrap();
        assert_eq!(status_as_text(status), serde_form, "{status:?}");
        assert_eq!(status_from_text(status_as_text(status)).unwrap(), status);
    }
}

#[test]
fn higher_tier_supersedes_and_never_deletes() {
    let store = store();
    // Acceptance criterion #2: LocalModel then Frontier on the same node/claim.
    store
        .write(100, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    let status = store
        .write(100, &claim("frame:x", ProvenanceTier::Frontier, ClaimStatus::Confirmed))
        .unwrap();
    assert_eq!(status, ClaimStatus::Confirmed);

    // The frontier claim is the current, confirmed one.
    let got = store.read(100, "frame:x").unwrap().expect("present");
    assert_eq!(got.tier, ProvenanceTier::Frontier);
    assert_eq!(got.status, ClaimStatus::Confirmed);

    // The prior row was marked superseded, not deleted (row count never drops).
    let history = store.history(100, "frame:x").unwrap();
    assert_eq!(history.len(), 2, "superseded row retained");
    let prior = history.iter().find(|c| c.tier == ProvenanceTier::LocalModel).unwrap();
    assert_eq!(prior.status, ClaimStatus::Superseded);
}

#[test]
fn lower_tier_is_provisional_and_never_overrides() {
    let store = store();
    store
        .write(100, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    // A deterministic (lower) write becomes provisional; the local-model
    // confirmed claim stays authoritative.
    let status = store
        .write(100, &claim("frame:x", ProvenanceTier::Deterministic, ClaimStatus::Confirmed))
        .unwrap();
    assert_eq!(status, ClaimStatus::Provisional);

    let got = store.read(100, "frame:x").unwrap().expect("present");
    assert_eq!(got.tier, ProvenanceTier::LocalModel, "lower tier never overrides");
    assert_eq!(got.status, ClaimStatus::Confirmed);

    // The provisional row is recorded (explicitly not authoritative).
    let active = store.read_active(100).unwrap();
    assert_eq!(active.len(), 1, "only the confirmed claim is active");
    let all = store.history(100, "frame:x").unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].status, ClaimStatus::Confirmed);
    assert_eq!(all[1].status, ClaimStatus::Provisional);
}

#[test]
fn equal_tier_rewrite_is_idempotent() {
    let store = store();
    store
        .write(100, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    // Re-derivation at the same authority does not self-demote.
    let status = store
        .write(100, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    assert_eq!(status, ClaimStatus::Confirmed);
    assert_eq!(store.history(100, "frame:x").unwrap().len(), 1);
}

#[test]
fn hash_change_invalidates_annotations() {
    // Acceptance criterion #3: a node whose content changed (new hash) has
    // no reachable annotations under the old hash — no scheduler needed.
    let store = store();
    store
        .write(111, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    assert!(store.read(111, "frame:x").unwrap().is_some());

    // Same claim key, different (mutated) content hash: unreachable.
    assert!(store.read(222, "frame:x").unwrap().is_none());
    // Re-derivation under the new hash writes fresh.
    store
        .write(222, &claim("frame:x", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    assert_eq!(store.read(222, "frame:x").unwrap().unwrap().tier, ProvenanceTier::LocalModel);
    // The old hash's row remains (audit), reachable by history but not by
    // the current read for the new hash.
    assert!(store.read(111, "frame:x").unwrap().is_some());
}

#[test]
fn distinct_claim_keys_are_independent() {
    let store = store();
    store
        .write(100, &claim("frame:a", ProvenanceTier::Deterministic, ClaimStatus::Confirmed))
        .unwrap();
    store
        .write(100, &claim("frame:b", ProvenanceTier::LocalModel, ClaimStatus::Confirmed))
        .unwrap();
    let active = store.read_active(100).unwrap();
    assert_eq!(active.len(), 2);
}

/// Acceptance criterion #3, through the real node store: mutating a node's
/// content changes its `content_hash` and invalidates prior annotations —
/// without manual scheduling. Also exercises the `ContentNodeStore` writer
/// wiring seam (`write_annotation`).
#[test]
fn node_store_hash_change_invalidates_annotations() {
    use crate::node_store::ContentNodeStore;

    let store = ContentNodeStore::open_in_memory().unwrap();
    let id = store.record_request("sess", "req", "show me the report").unwrap();
    let hash_before = store.snapshot(id).unwrap().content_hash;
    assert_ne!(hash_before, 0, "LOD0 present → non-zero hash");

    // A LocalModel annotation written under the current hash.
    store
        .write_annotation(
            id,
            &claim("frame:report", ProvenanceTier::LocalModel, ClaimStatus::Confirmed),
        )
        .unwrap()
        .expect("durable in-memory store");

    let annotations = AnnotationStore::new(store.shared_sqlite().expect("shared store"));
    assert!(annotations.read(hash_before, "frame:report").unwrap().is_some());

    // Mutating LOD0 changes the hash (record_result → with_node_mut →
    // ensure_lod_eager recomputes).
    store
        .record_result(id, true, Some(0.9), "display the sales report")
        .unwrap();
    let hash_after = store.snapshot(id).unwrap().content_hash;
    assert_ne!(hash_before, hash_after, "content change → new hash");

    // Under the new hash the old claim is unreachable (invalidation is
    // keying, not a scheduler).
    assert!(
        annotations.read(hash_after, "frame:report").unwrap().is_none(),
        "old annotation invalidated by the hash change"
    );
    // The old-hash row remains (audit), and re-derivation under the new
    // hash writes fresh.
    assert!(annotations.read(hash_before, "frame:report").unwrap().is_some());
    store
        .write_annotation(
            id,
            &claim("frame:report", ProvenanceTier::LocalModel, ClaimStatus::Confirmed),
        )
        .unwrap()
        .expect("re-derive");
    assert!(annotations.read(hash_after, "frame:report").unwrap().is_some());
}
