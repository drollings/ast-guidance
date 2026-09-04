//! `AnnotationStore` — the ledger's tiered annotation store (ROADMAP M4).
//!
//! Every annotation a node carries is keyed to that node's **content hash**
//! (the version identity) and stamped with a producing [`ProvenanceTier`].
//! A write whose tier exceeds the current row's tier supersedes it (never
//! deletes) and installs the new row as authoritative; an equal-or-lower-tier
//! write installs a `Provisional` row that never silently overrides a
//! higher-tier `Confirmed` one. Reads are by `(content_hash, claim_key)`, so a
//! hash change (a node mutation) makes the old rows unreachable **without a
//! staleness scheduler** — invalidation is a consequence of keying, not a
//! separate job.
//!
//! The table lives in the ledger schema (`ledger_migrations`), so the store
//! shares the one `SqliteStore` connection with the node store and the other
//! typed views (`SqliteCorrectionIndex`, `SqliteConceptStore`).

use std::sync::Arc;

use fluent_db::error::DbError;
use fluent_db::store::SqliteStore;
use fluent_types::{AnnotationClaim, ClaimStatus, ProvenanceTier};
use rusqlite::params;

/// The `ledger_annotations` table name (single source of truth for the schema;
/// the DDL itself lives in `crate::ledger::ledger_migrations`).
pub const ANNOTATIONS_TABLE: &str = "ledger_annotations";

/// The tiered annotation store over the shared ledger connection.
#[derive(Debug, Clone)]
pub struct AnnotationStore {
    store: Arc<SqliteStore>,
}

impl AnnotationStore {
    /// An annotation store over the shared connection.
    pub fn new(store: Arc<SqliteStore>) -> Self {
        Self { store }
    }

    /// Write a claim against a node version. The effective [`ClaimStatus`] is
    /// decided here from the tier comparison (not taken from `claim.status`):
    ///
    /// - **no current row** → installs `Confirmed` (the first writer for this
    ///   `(content_hash, claim_key)` establishes the claim),
    /// - **higher tier than the current `Confirmed` row** → marks the prior
    ///   row `Superseded` (never deleted) and installs the new row `Confirmed`,
    /// - **equal tier to a `Confirmed` row** → idempotent: the existing
    ///   confirmed row is kept (re-derivation does not self-demote),
    /// - **lower tier** (or equal against a provisional) → installs a
    ///   `Provisional` row that never overrides a higher-tier `Confirmed` one.
    ///
    /// Returns the effective status of the write.
    pub fn write(
        &self,
        content_hash: u64,
        claim: &AnnotationClaim,
    ) -> Result<ClaimStatus, DbError> {
        self.store.transaction(|tx| {
            let current = current_row(tx, content_hash, &claim.claim_key)?;
            match current {
                None => {
                    let next_id = next_claim_id(tx, content_hash, &claim.claim_key)?;
                    insert_row(
                        tx,
                        content_hash,
                        next_id,
                        claim,
                        ClaimStatus::Confirmed,
                    )?;
                    Ok(ClaimStatus::Confirmed)
                }
                Some(cur) if claim.tier > cur.tier => {
                    // Supersede the prior confirmed row, then confirm the new.
                    tx.execute(
                        "UPDATE ledger_annotations SET status = 'superseded' \
                         WHERE content_hash = ?1 AND claim_key = ?2 AND claim_id = ?3",
                        params![content_hash, claim.claim_key, cur.claim_id],
                    )?;
                    let next_id = next_claim_id(tx, content_hash, &claim.claim_key)?;
                    insert_row(
                        tx,
                        content_hash,
                        next_id,
                        claim,
                        ClaimStatus::Confirmed,
                    )?;
                    Ok(ClaimStatus::Confirmed)
                }
                Some(cur)
                    if claim.tier == cur.tier && cur.status == ClaimStatus::Confirmed =>
                {
                    // Idempotent re-write at the same authority: keep the
                    // confirmed row rather than self-demoting to provisional.
                    Ok(ClaimStatus::Confirmed)
                }
                Some(_) => {
                    // Lower tier (or equal to a provisional): record it as
                    // provisional — it never overrides the higher confirmed row.
                    let next_id = next_claim_id(tx, content_hash, &claim.claim_key)?;
                    insert_row(
                        tx,
                        content_hash,
                        next_id,
                        claim,
                        ClaimStatus::Provisional,
                    )?;
                    Ok(ClaimStatus::Provisional)
                }
            }
        })
    }

    /// Read the current claim for a node version + key — the authoritative
    /// `Confirmed` row when one exists, else the newest non-superseded
    /// (provisional) one. `None` when the hash is unknown — a node whose
    /// content changed (new hash) has no reachable annotations, which is the
    /// invalidation contract.
    pub fn read(
        &self,
        content_hash: u64,
        claim_key: &str,
    ) -> Result<Option<AnnotationClaim>, DbError> {
        self.store.query_row(
            &format!(
                "SELECT claim_key, tier, status, payload, produced_by, produced_at \
                 FROM ledger_annotations \
                 WHERE content_hash = ?1 AND claim_key = ?2 AND status != 'superseded' \
                 ORDER BY {STATUS_PRIORITY}, claim_id DESC LIMIT 1",
            ),
            params![content_hash as i64, claim_key],
            row_to_claim,
        )
    }

    /// Read every **authoritative** (`Confirmed`) claim for a node version.
    /// Pending provisional claims are deliberately excluded — they are not
    /// authoritative.
    pub fn read_active(
        &self,
        content_hash: u64,
    ) -> Result<Vec<AnnotationClaim>, DbError> {
        self.store.query_rows(
            "SELECT claim_key, tier, status, payload, produced_by, produced_at \
             FROM ledger_annotations \
             WHERE content_hash = ?1 AND status = 'confirmed' \
             ORDER BY claim_key, claim_id",
            params![content_hash as i64],
            row_to_claim,
        )
    }

    /// Read the full version history of one claim key (including superseded
    /// rows) — the audit trail behind the "never deleted" invariant.
    pub fn history(
        &self,
        content_hash: u64,
        claim_key: &str,
    ) -> Result<Vec<AnnotationClaim>, DbError> {
        self.store.query_rows(
            "SELECT claim_key, tier, status, payload, produced_by, produced_at \
             FROM ledger_annotations \
             WHERE content_hash = ?1 AND claim_key = ?2 \
             ORDER BY claim_id",
            params![content_hash as i64, claim_key],
            row_to_claim,
        )
    }
}

/// A single ledger-annotations row (with its `claim_id` version) — the subset
/// `current_row` inspects to decide the write's tier comparison.
struct Row {
    claim_id: i64,
    tier: ProvenanceTier,
    status: ClaimStatus,
}

/// SQL ordering so a `Confirmed` row always outranks a `Provisional` one: the
/// authoritative claim is the highest-priority non-superseded row for a key.
const STATUS_PRIORITY: &str = "CASE status \
     WHEN 'confirmed' THEN 0 WHEN 'provisional' THEN 1 ELSE 2 END";

fn row_to_claim(row: &rusqlite::Row<'_>) -> rusqlite::Result<AnnotationClaim> {
    let claim_key: String = row.get(0)?;
    let tier: String = row.get(1)?;
    let status: String = row.get(2)?;
    let payload: String = row.get(3)?;
    let produced_by: String = row.get(4)?;
    let produced_at: i64 = row.get(5)?;
    Ok(AnnotationClaim {
        claim_key,
        tier: tier_from_text(&tier).map_err(conv_err)?,
        status: status_from_text(&status).map_err(conv_err)?,
        payload: serde_json::from_str(&payload).map_err(|e| conv_err(db_err(e)))?,
        produced_by,
        produced_at: produced_at as u64,
    })
}

/// The authoritative (highest-priority, non-superseded) row for
/// `(content_hash, claim_key)` — the row the write's tier comparison is decided
/// against.
///
/// NOTE (M9): transaction-scoped (`&Transaction`), so it uses rusqlite
/// directly — the `db::query` free functions take `&Connection` and there is
/// no pool/transaction variant without an async rewrite. Same statement
/// shape (`prepare` → single-row → `Ok(None)` on no rows).
fn current_row(
    tx: &rusqlite::Transaction<'_>,
    content_hash: u64,
    claim_key: &str,
) -> Result<Option<Row>, DbError> {
    let mut stmt = tx.prepare(
        &format!(
            "SELECT claim_id, tier, status \
             FROM ledger_annotations \
             WHERE content_hash = ?1 AND claim_key = ?2 AND status != 'superseded' \
             ORDER BY {STATUS_PRIORITY}, claim_id DESC LIMIT 1",
        ),
    )?;
    let mut rows = stmt.query(params![content_hash as i64, claim_key])?;
    let row = rows.next()?.map(row_to_row).transpose()?;
    Ok(row)
}

/// The next `claim_id` for `(content_hash, claim_key)` (max + 1, 1-based).
fn next_claim_id(
    tx: &rusqlite::Transaction<'_>,
    content_hash: u64,
    claim_key: &str,
) -> Result<i64, DbError> {
    let next: i64 = tx.query_row(
        "SELECT COALESCE(MAX(claim_id), 0) FROM ledger_annotations \
         WHERE content_hash = ?1 AND claim_key = ?2",
        params![content_hash as i64, claim_key],
        |r| r.get(0),
    )?;
    Ok(next + 1)
}

fn insert_row(
    tx: &rusqlite::Transaction<'_>,
    content_hash: u64,
    claim_id: i64,
    claim: &AnnotationClaim,
    status: ClaimStatus,
) -> Result<(), DbError> {
    tx.execute(
        "INSERT INTO ledger_annotations \
         (content_hash, claim_key, claim_id, tier, status, payload, produced_by, produced_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            content_hash as i64,
            claim.claim_key,
            claim_id,
            tier_as_text(claim.tier),
            status_as_text(status),
            serde_json::to_string(&claim.payload).map_err(db_err)?,
            claim.produced_by,
            claim.produced_at as i64,
        ],
    )?;
    Ok(())
}

fn row_to_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Row> {
    let claim_id: i64 = row.get(0)?;
    let tier: String = row.get(1)?;
    let status: String = row.get(2)?;
    Ok(Row {
        claim_id,
        tier: tier_from_text(&tier).map_err(conv_err)?,
        status: status_from_text(&status).map_err(conv_err)?,
    })
}

/// Plain snake_case storage for [`ProvenanceTier`] (exhaustive over the enum,
/// so a new tier fails to compile here and can never be dropped from parsing).
fn tier_as_text(t: ProvenanceTier) -> &'static str {
    match t {
        ProvenanceTier::Deterministic => "deterministic",
        ProvenanceTier::LocalModel => "local_model",
        ProvenanceTier::Frontier => "frontier",
        ProvenanceTier::HumanReview => "human_review",
    }
}

fn tier_from_text(s: &str) -> Result<ProvenanceTier, DbError> {
    Ok(match s {
        "deterministic" => ProvenanceTier::Deterministic,
        "local_model" => ProvenanceTier::LocalModel,
        "frontier" => ProvenanceTier::Frontier,
        "human_review" => ProvenanceTier::HumanReview,
        other => return Err(DbError::Other(format!("unknown provenance tier: {other}"))),
    })
}

/// Plain snake_case storage for [`ClaimStatus`].
fn status_as_text(s: ClaimStatus) -> &'static str {
    match s {
        ClaimStatus::Provisional => "provisional",
        ClaimStatus::Confirmed => "confirmed",
        ClaimStatus::Superseded => "superseded",
    }
}

fn status_from_text(s: &str) -> Result<ClaimStatus, DbError> {
    Ok(match s {
        "provisional" => ClaimStatus::Provisional,
        "confirmed" => ClaimStatus::Confirmed,
        "superseded" => ClaimStatus::Superseded,
        other => return Err(DbError::Other(format!("unknown claim status: {other}"))),
    })
}

/// Wrap a `DbError` as a SQL column conversion failure (for row readers).
fn conv_err(e: DbError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
}

fn db_err<E: std::fmt::Display>(e: E) -> DbError {
    DbError::Other(e.to_string())
}
#[cfg(test)]
#[path = "../../tests/ledger_annotations.rs"]
mod tests;
