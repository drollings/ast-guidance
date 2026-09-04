use super::*;
use crate::ledger::ContentNodeLedger;

fn store(ledger: &ContentNodeLedger) -> OverlayCandidateStore {
    OverlayCandidateStore::new(
        ledger.node_store().shared_sqlite().expect("shared sqlite"),
    )
}

fn cand(node: NodeId, start: usize, entity: Option<i64>) -> OverlayCandidate {
    let entity_id = entity.map(|e| InterlinguaId::from_u64(e as u64));
    let mut c = OverlayCandidate::entity_link(
        node,
        start,
        start + 3,
        entity_id.unwrap_or(InterlinguaId::from_u64(0x0200_0000_0000_0001)),
        0.9,
        "entity_link",
    );
    c.entity_id = entity_id;
    c
}

#[test]
fn write_is_first_wins() {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = store(&ledger);
    let node = NodeId::from_int(1);
    let a = cand(node, 0, Some(0x0200_0000_0000_0001));
    let b = cand(node, 0, Some(0x0200_0000_0000_0001));
    assert!(s.write_candidate(&a).expect("first insert"), "first wins inserts");
    assert!(
        !s.write_candidate(&b).expect("duplicate"),
        "a duplicate key is ignored (first-wins)"
    );
    let rows = s.for_node(node).expect("query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].span_start, 0);
}

#[test]
fn distinct_spans_or_entities_are_separate_candidates() {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = store(&ledger);
    let node = NodeId::from_int(2);
    s.write_candidate(&cand(node, 0, Some(0x0200_0000_0000_0001)))
        .expect("span 0");
    s.write_candidate(&cand(node, 4, Some(0x0200_0000_0000_0001)))
        .expect("span 4");
    s.write_candidate(&cand(node, 0, Some(0x0200_0000_0000_0002)))
        .expect("entity 2");
    assert_eq!(s.for_node(node).expect("query").len(), 3);
}

#[test]
fn status_transitions_pending_only_and_promotion_is_idempotent() {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = store(&ledger);
    let node = NodeId::from_int(3);
    let c = cand(node, 0, Some(0x0200_0000_0000_0001));
    s.write_candidate(&c).expect("write");
    s.promote(node, &c).expect("promote");
    let rows = s.for_node(node).expect("query");
    assert_eq!(rows[0].status, CandidateStatus::Promoted);
    // A dismissed candidate cannot be promoted (first-wins).
    let c2 = cand(node, 8, Some(0x0200_0000_0000_0002));
    s.write_candidate(&c2).expect("write");
    s.dismiss(node, &c2).expect("dismiss");
    s.promote(node, &c2).expect("promote is a no-op");
    let rows = s.for_node(node).expect("query");
    assert!(rows.iter().any(|r| r.status == CandidateStatus::Dismissed));
    // Promotion of an already-promoted candidate stays promoted.
    s.promote(node, &c).expect("promote again");
    assert_eq!(s.for_node(node).expect("q")[0].status, CandidateStatus::Promoted);
}

#[test]
fn promote_linked_for_node_promotes_matching_entity_candidates() {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = store(&ledger);
    let node = NodeId::from_int(4);
    let linked_id = InterlinguaId::from_u64(0x0200_0000_0000_0001);
    let other_id = InterlinguaId::from_u64(0x0200_0000_0000_0002);
    s.write_candidate(&cand(node, 0, Some(linked_id.as_i64())))
        .expect("write");
    s.write_candidate(&cand(node, 8, Some(other_id.as_i64())))
        .expect("write");
    let n = s
        .promote_linked_for_node(node, &[linked_id])
        .expect("promote");
    assert_eq!(n, 1);
    let rows = s.for_node(node).expect("query");
    let linked = rows.iter().find(|r| r.entity_id == Some(linked_id)).unwrap();
    assert_eq!(linked.status, CandidateStatus::Promoted);
    let other = rows.iter().find(|r| r.entity_id == Some(other_id)).unwrap();
    assert_eq!(other.status, CandidateStatus::Pending);
}

#[test]
fn reconcile_ids_reports_unresolved_entity_ids() {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = store(&ledger);
    let node = NodeId::from_int(5);
    // A non-entity candidate (entity_id 0) never trips id-membership.
    let pii = OverlayCandidate {
        node_id: node,
        span_start: 0,
        span_end: 3,
        kind: ResidualKind::PiiSpan,
        entity_id: None,
        score: Some(1.0),
        source: "pii".into(),
        status: CandidateStatus::Pending,
    };
    s.write_candidate(&pii).expect("write pii");
    assert_eq!(s.reconcile_ids().expect("reconcile"), 0, "no entity ids");
    // An entity id that is not in interlingua_concepts is unresolved.
    s.write_candidate(&cand(node, 4, Some(0x0200_0000_0000_0009)))
        .expect("write entity");
    assert_eq!(s.reconcile_ids().expect("reconcile"), 1);
    // Register the concept → resolved.
    let shared = ledger.node_store().shared_sqlite().expect("shared");
    shared
        .execute(
            "INSERT OR IGNORE INTO interlingua_concepts (id, namespace, canonical_name) \
             VALUES (?1, 512, 'yago:Mystery')",
            params![0x0200_0000_0000_0009_i64],
        )
        .expect("insert concept");
    assert_eq!(s.reconcile_ids().expect("reconcile"), 0);
}

#[test]
fn candidate_serde_round_trip() {
    let c = cand(NodeId::from_int(1), 0, Some(0x0200_0000_0000_0001));
    let json = serde_json::to_string(&c).unwrap();
    let back: OverlayCandidate = serde_json::from_str(&json).unwrap();
    assert_eq!(back, c);
    assert_eq!(serde_json::to_string(&ResidualKind::EntityLink).unwrap(), "\"entity_link\"");
}
