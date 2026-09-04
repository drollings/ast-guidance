//! The async review mechanism (ROADMAP §12 — F4).
//!
//! LLM agents and users review deterministic-parser output: correct dep/head/
//! lemma/POS, link entities through the YaGO taxonomy, mark ambiguities.
//! Review is **async** — it never blocks the hot path — and correction reuse
//! avoids repeat LLM spend (a reviewed `(lemma, entity)` pattern costs zero
//! on the next occurrence).
//!
//! **Boundary (F4):** the [`CorrectionIndex`] persistence seam is a **trait**
//! owned by this crate; the router implements it over its own SQLite table.
//! No router-owned type ever appears here.

use serde::{Deserialize, Serialize};

use fluent_concept::ConceptStoreError;
use crate::llm::{AnnotationRecord, AnnotationResult, AnnotationSource};
use fluent_types::{ConceptMetadata, InterlinguaId};

/// The correction field a reviewer can override on a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionField {
    Dep,
    Head,
    Lemma,
    Pos,
    Ner,
}

/// One token-level correction. `old_value` is reviewer context only — the
/// amend step never reads it — and defaults to empty so the span-scoped
/// refine contract (ROADMAP_20260831_ARCEAGER M2.2, which omits it) parses
/// into the same type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Correction {
    pub token_index: usize,
    pub field: CorrectionField,
    #[serde(default)]
    pub old_value: String,
    pub new_value: String,
}

/// An entity link produced by a review: a token span resolved to a YaGO
/// concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkedEntity {
    pub token_start: usize,
    pub token_end: usize,
    pub entity_type: String,
    pub interlingua_id: InterlinguaId,
    pub confidence: f64,
}

/// The outcome of a review pass over one parse. Also the wire shape of the
/// span-scoped refine reply (ROADMAP_20260831_ARCEAGER M2.2): the refine
/// contract carries `corrections` + `note` only, so every field is defaulted
/// on deserialize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ParseReview {
    /// The corrections actually applied (empty = the parse was accepted).
    #[serde(default)]
    pub corrections: Vec<Correction>,
    /// Entities linked through the taxonomy.
    #[serde(default)]
    pub linked_entities: Vec<LinkedEntity>,
    /// Free-form reviewer note / ambiguity marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ParseReview {
    /// Deserialize a review/refine reply (`{"corrections": [...], "note":
    /// "..."}`). Structural parse only — the amendment and the 7-check gate
    /// are the callers' business.
    pub fn parse_json(json: &str) -> Result<Self, crate::validate::AnnotationError> {
        serde_json::from_str(json)
            .map_err(|e| crate::validate::AnnotationError::Json { source: std::sync::Arc::new(e) })
    }
}

/// The review lifecycle of a parse node. `Reviewed` marks a successful review;
/// `HumanReviewed` the human-overridden variant (stored verbatim for audit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReviewStatus {
    /// Not yet reviewed; `auto_confidence` is the parser's own score.
    Unreviewed { auto_confidence: f64 },
    /// The review model accepted or corrected the parse.
    Reviewed { review: ParseReview },
    /// A human reviewed it (the strongest provenance).
    HumanReviewed { review: ParseReview },
}

impl ReviewStatus {
    /// The ledger provenance tier this review lifecycle maps to (ROADMAP M4):
    /// `HumanReviewed` is the strongest provenance; a model `Reviewed` (or
    /// parser-authored `Unreviewed`) claim is below it.
    #[must_use]
    pub fn tier(&self) -> fluent_types::ProvenanceTier {
        match self {
            ReviewStatus::HumanReviewed { .. } => fluent_types::ProvenanceTier::HumanReview,
            ReviewStatus::Reviewed { .. } => fluent_types::ProvenanceTier::LocalModel,
            ReviewStatus::Unreviewed { .. } => fluent_types::ProvenanceTier::Deterministic,
        }
    }
}

/// The persistence seam for previously-corrected `(lemma, entity)` patterns.
/// Implemented by the router over its `interlingua_index` table (§12.5).
pub trait CorrectionIndex: Send + Sync {
    /// The corrections previously recorded for this pattern, when known.
    fn query_previous_corrections(
        &self,
        lemma_id: InterlinguaId,
        entity_id: Option<InterlinguaId>,
    ) -> Option<Vec<Correction>>;
    /// Persist a correction for this pattern.
    fn record_correction(
        &self,
        lemma_id: InterlinguaId,
        entity_id: Option<InterlinguaId>,
        corrections: &[Correction],
    ) -> Result<(), ConceptStoreError>;
}

/// The taxonomy-grounded review prompt (§12.4): lists the candidate YaGO
/// classes/labels/IRIs and asks the reviewer to resolve entity types to
/// concept ids. **Pure** — unit-testable without a model.
#[must_use]
pub fn review_prompt(text: &str, parse: &AnnotationResult, candidate_concepts: &[ConceptMetadata]) -> String {
    use std::fmt::Write;
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "Review this sentence and its dependency parse. Correct any errors in \
         dep, head, lemma, or POS. For entity tokens, resolve the entity type to \
         one of the candidate concepts below (reply with its id)."
    );
    let _ = writeln!(prompt, "Sentence: {text}");
    let _ = writeln!(prompt, "Parse (token / pos / dep / head):");
    for (i, r) in parse.records().records().iter().enumerate() {
        let _ = writeln!(prompt, "  {i}: {} / {} / {} / {}", r.text, r.pos, r.dep, r.head);
    }
    if candidate_concepts.is_empty() {
        let _ = writeln!(prompt, "Candidate concepts: (none registered)");
    } else {
        let _ = writeln!(prompt, "Candidate concepts:");
        for c in candidate_concepts {
            let _ = writeln!(
                prompt,
                "  id={} name={} label={} iri={}",
                c.id,
                c.canonical_name,
                c.label.as_deref().unwrap_or(""),
                c.yago_iri.as_deref().unwrap_or(""),
            );
        }
    }
    let _ = writeln!(
        prompt,
        "Reply with the corrections as JSON: {{\"corrections\": [{{token_index, field, \
         old_value, new_value}}], \"linked_entities\": [{{token_start, token_end, \
         entity_type, interlingua_id, confidence}}], \"note\": \"...\"}}"
    );
    prompt
}

/// The shared amendment helper: apply field-level edits to token records and
/// return the number of edits actually landed (out-of-range indices are
/// skipped, unparseable `head` values are rejected). One amendment vocabulary
/// for both the review path ([`apply_corrections`]) and the span-scoped
/// refine rungs (ROADMAP_20260831_ARCEAGER M2.2) — DRY.
pub fn apply_edits(records: &mut [AnnotationRecord], edits: &[Correction]) -> usize {
    let mut applied = 0usize;
    for c in edits {
        let Some(rec) = records.get_mut(c.token_index) else {
            continue;
        };
        // R7: guard on old_value — empty = don't-care (refine wire omits it),
        // non-empty mismatch = stale correction, warn and skip.
        let current = match c.field {
            CorrectionField::Dep => rec.dep.clone(),
            CorrectionField::Head => rec.head.to_string(),
            CorrectionField::Lemma => rec.lemma.clone(),
            CorrectionField::Pos => rec.pos.clone(),
            CorrectionField::Ner => rec.ent_type.clone(),
        };
        if !c.old_value.is_empty() && c.old_value != current {
            tracing::warn!(
                token_index = c.token_index,
                field = ?c.field,
                expected_old = %c.old_value,
                actual = %current,
                "stale correction skipped"
            );
            continue;
        }
        let landed = match c.field {
            CorrectionField::Dep => {
                rec.dep.clone_from(&c.new_value);
                true
            }
            CorrectionField::Head => match c.new_value.parse() {
                Ok(h) => {
                    rec.head = h;
                    true
                }
                Err(_) => false,
            },
            CorrectionField::Lemma => {
                rec.lemma.clone_from(&c.new_value);
                true
            }
            CorrectionField::Pos => {
                rec.pos.clone_from(&c.new_value);
                true
            }
            CorrectionField::Ner => {
                rec.ent_type.clone_from(&c.new_value);
                true
            }
        };
        if landed {
            applied += 1;
        }
    }
    applied
}

/// Apply a set of corrections to a parse, producing a corrected
/// [`AnnotationResult`] with provenance [`AnnotationSource::HumanReview`]
/// when any correction was applied (§12.3 step c).
#[must_use]
pub fn apply_corrections(parse: &AnnotationResult, corrections: &[Correction]) -> AnnotationResult {
    let mut records = parse.records().records().to_vec();
    apply_edits(&mut records, corrections);
    let source = if corrections.is_empty() {
        parse.source()
    } else {
        AnnotationSource::HumanReview
    };
    AnnotationResult::new(crate::llm::AnnotationSet(records), source)
        .with_confidence(parse.token_confidence.clone(), parse.parse_confidence.clone())
}

#[cfg(test)]
#[path = "../tests/review.rs"]
mod tests;
