//! Ledger writer for the NLP parse (roadmap §6): one `ContentNode` per
//! request carrying the per-sentence routing signals in `metadata`, alongside
//! the existing request/result nodes. LOD0 (full text) and LOD5 (label) stay
//! eager via the store; LOD1–LOD4 derive lazily from LOD0 as usual.
//!
//! ROADMAP §14.1 extends the metadata with the interlingua frame: per-sentence
//! content-addressed ids, the confidence summary (C1), and the review status.
//! The same ids are mirrored into the durable `interlingua_index` table
//! (migration 5) so the router can query which ids attached to which parse
//! node and index corrections (§14.2/§14.3).

use fluent_types::{ContentNode, InterlinguaId, NodeId};
use rusqlite::params;
use spacy_rs::routing::RoutingSignal;

use crate::ledger::{ContentNodeLedger, LedgerError};
use crate::node_store::new_node;
use crate::pipeline_types::NlpConfidenceSummary;

/// The metadata `kind` discriminator for parse nodes.
pub const NLP_PARSE_KIND: &str = "nlp_parse";

/// Build the interlingua-ids metadata map (`sentence_0`, `sentence_1`, …)
/// from the signals' interlingua frames (§14.1).
fn interlingua_ids_json(signals: &[RoutingSignal]) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = signals
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let frame = s.interlingua.as_ref()?;
            Some((
                format!("sentence_{i}"),
                serde_json::json!({
                    "predicate_id": frame.predicate_id.map(InterlinguaId::as_i64),
                    "subject_id": frame.subject_id.map(InterlinguaId::as_i64),
                    "direct_object_id": frame.direct_object_id.map(InterlinguaId::as_i64),
                    "indirect_object_id": frame.indirect_object_id.map(InterlinguaId::as_i64),
                    "concept_ids": frame.concept_ids.iter().map(|id| id.as_i64()).collect::<Vec<_>>(),
                }),
            ))
        })
        .collect();
    serde_json::Value::Object(map)
}

/// Build the parse `ContentNode` for a request: role `"parse"`, LOD0 = the
/// request text, metadata = the per-sentence [`RoutingSignal`]s (the dep/POS/
/// lemma arrays ride inside each signal's transcript) plus the interlingua
/// frame (ids + confidence + review status, §14.1).
#[must_use]
pub fn parse_node(
    session_id: &str,
    request_id: &str,
    text: &str,
    signals: &[RoutingSignal],
) -> ContentNode {
    parse_node_with_confidence(session_id, request_id, text, signals, None, None)
}

/// [`parse_node`] plus the C1 confidence summary and an eager review status.
///
/// `token_confidence` (optional, serde-default) is the per-token confidence
/// vector the review endpoint reads back so a later review rebuilds the parse
/// with per-token fidelity rather than a flat overall (L3). Additive — old
/// parse nodes without the field behave exactly as before.
#[must_use]
pub fn parse_node_with_confidence(
    session_id: &str,
    request_id: &str,
    text: &str,
    signals: &[RoutingSignal],
    confidence: Option<&NlpConfidenceSummary>,
    token_confidence: Option<&[f64]>,
) -> ContentNode {
    let mut node = new_node(NodeId(0), session_id, request_id, "parse", text, None);
    node.id = None; // the store allocates the id
    let auto_confidence = confidence.as_ref().map_or(1.0, |c| c.overall);
    node.metadata = Some(serde_json::json!({
        "kind": NLP_PARSE_KIND,
        "sentence_count": signals.len(),
        "signals": signals,
        "interlingua_ids": interlingua_ids_json(signals),
        "confidence": confidence,
        "token_confidence": token_confidence,
        "review_status": { "Unreviewed": { "auto_confidence": auto_confidence } },
    }));
    node
}

/// Persist the parse node through the ledger facade (LOD0/LOD5 eager,
/// metadata passed through untouched). Best-effort at the call site.
pub fn record_parse_node(
    ledger: &ContentNodeLedger,
    session_id: &str,
    request_id: &str,
    text: &str,
    signals: &[RoutingSignal],
) -> Result<NodeId, LedgerError> {
    record_parse_node_with_confidence(ledger, session_id, request_id, text, signals, None, None)
}

/// Persist a parse node with its C1 confidence summary + per-token confidence
/// **and** mirror the interlingua ids into `interlingua_index` — the single
/// consolidated write path (ROADMAP_20260828_ORT M1.3, DRY). The node's
/// `review_status` is `unreviewed`; the durable index rows are written in the
/// same ledger call so a live dispatch populates the index exactly as the
/// test path (`record_parse_node`) does. `record_parse_node` is a thin wrapper
/// over this.
pub fn record_parse_node_with_confidence(
    ledger: &ContentNodeLedger,
    session_id: &str,
    request_id: &str,
    text: &str,
    signals: &[RoutingSignal],
    confidence: Option<&NlpConfidenceSummary>,
    token_confidence: Option<&[f64]>,
) -> Result<NodeId, LedgerError> {
    let node = parse_node_with_confidence(session_id, request_id, text, signals, confidence, token_confidence);
    let id = ledger.record_content_node(&node)?;
    populate_interlingua_index(ledger, id, signals, confidence)?;
    Ok(id)
}

/// Populate the durable `interlingua_index` rows for a parse node (ROADMAP
/// §14.3/13.3): one row per role id per sentence, mirroring the metadata.
/// Written in the same transaction as the node itself.
pub fn populate_interlingua_index(
    ledger: &ContentNodeLedger,
    node_id: NodeId,
    signals: &[RoutingSignal],
    confidence: Option<&NlpConfidenceSummary>,
) -> Result<(), LedgerError> {
    let Some(store) = ledger.node_store().shared_sqlite() else {
        return Ok(()); // ephemeral store — no durable index
    };
    let overall = confidence.map_or(1.0, |c| c.overall);
    let is_provisional = confidence.is_some_and(|c| c.semantic_plausibility.is_none());
    let mut rows: Vec<(i64, i64, String, f64)> = Vec::new();
    for s in signals {
        if let Some(il) = &s.interlingua {
            let mut push = |id: Option<InterlinguaId>, role: &str| {
                if let Some(id) = id {
                    rows.push((id.as_i64(), id.local_id(), role.to_string(), overall));
                }
            };
            push(il.predicate_id, "predicate");
            push(il.subject_id, "subject");
            push(il.direct_object_id, "direct_object");
            push(il.indirect_object_id, "indirect_object");
            for c in &il.concept_ids {
                rows.push((c.as_i64(), c.local_id(), "concept".to_string(), overall));
            }
        }
    }
    let status = if is_provisional { "provisional" } else { "unreviewed" };
    store
        .transaction(|tx| {
            for (id, _local, role, conf) in &rows {
                tx.execute(
                    "INSERT OR IGNORE INTO interlingua_index \
                     (node_id, interlingua_id, interlingua_source, role, confidence, review_status, span_key) \
                     VALUES (?1, ?2, 'spacy_lemma', ?3, ?4, ?5, '')",
                    params![node_id.as_int(), id, role, conf, status],
                )?;
            }
            Ok(())
        })
        .map_err(|e| LedgerError::Db(e.to_string()))?;
    Ok(())
}

/// Whether a node is a parse node (read-path helper).
#[must_use]
pub fn is_parse_node(node: &ContentNode) -> bool {
    node.metadata
        .as_ref()
        .and_then(|m| m.get("kind"))
        .and_then(serde_json::Value::as_str)
        == Some(NLP_PARSE_KIND)
}

/// The metadata `kind` discriminator for parse-review nodes (ROADMAP §12.7,
/// C7 — the corrections → workflow-extraction handoff).
pub const PARSE_REVIEW_KIND: &str = "parse_review";

/// Build the `parse_review` ledger node persisted on a review **miss** (an LLM
/// correction actually applied), in the same transaction as the
/// `CorrectionIndex` write (§12.6/§12.7). The metadata carries the pattern key
/// and the prompt/corrections so the chart extractor can fold it into a
/// `ChartAuditStep` (`parse_review_step`, charts/extract.rs). The node carries
/// its origin (`session_id`/`request_id`) so it is indexed under the session
/// it belongs to, never the empty bucket (L4).
#[must_use]
pub fn review_node(
    node_id: NodeId,
    session_id: &str,
    request_id: &str,
    text: &str,
    prompt: &str,
    corrections_json: &str,
    lemma_id: InterlinguaId,
    entity_id: Option<InterlinguaId>,
    review_model: &str,
) -> ContentNode {
    let mut node = new_node(
        NodeId(0),
        session_id,
        request_id,
        "parse_review",
        text,
        None,
    );
    node.id = None; // the store allocates the id
    node.metadata = Some(serde_json::json!({
        "kind": PARSE_REVIEW_KIND,
        "source_node_id": node_id.as_int(),
        "prompt": prompt,
        "corrections_json": corrections_json,
        "lemma_id": lemma_id.as_i64(),
        "entity_id": entity_id.map(InterlinguaId::as_i64),
        "review_model": review_model,
    }));
    node
}

/// Whether a node is a parse-review node (read-path helper).
#[must_use]
pub fn is_parse_review_node(node: &ContentNode) -> bool {
    node.metadata
        .as_ref()
        .and_then(|m| m.get("kind"))
        .and_then(serde_json::Value::as_str)
        == Some(PARSE_REVIEW_KIND)
}
#[cfg(test)]
#[path = "../../tests/ledger_nlp.rs"]
mod tests;
