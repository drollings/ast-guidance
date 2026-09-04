//! Frame extraction — spacy-rs as a structural index (ROADMAP M3, G6).
//!
//! Per predicate, a [`Frame`] is a typed argument structure derived
//! deterministically from the attached [`DependencyGraph`] tree (`Doc::children`
//! / heads — the same discipline as validator checks 5/6). A [`FrameAnalysis`]
//! pairs the frames with a typed [`AmbiguityEntry`] list and mints
//! provisional-vs-permanent [`FrameKey`]s:
//!
//! - A **permanent** key is minted only for an ambiguity-free frame.
//! - A **provisional** key is minted for any frame with an open ambiguity
//!   entry. Only permanent keys are persisted to the ledger/graph.
//!
//! Three ambiguity kinds are detected structurally (anaphora and
//! coordination/ellipsis are documented future work, not built):
//!
//! - **Attachment near-tie** — a near-zero oracle margin on the parse
//!   (`ParseConfidence.oracle_margins`), signalling the heuristic parser
//!   doubted an attachment decision.
//! - **Predicate polysemy** — a predicate lemma whose interlingua id resolves
//!   to more than one `ConceptStore` candidate (the store keeps every
//!   canonical under the shared bucket id).
//! - **Negation/modal scope** — a sentence with both a `neg` and a modal `aux`
//!   governing the same predicate.
//!
//! The [`PreferredSenseIndex`] trait is the promotion seam (mirroring
//! `CorrectionIndex`): a resolved `(predicate_lemma_id, ambiguity_kind)` pattern
//! is recorded and reused deterministically, so a repeating ambiguity never
//! re-triggers an LLM call (golden-corpus-style rule genesis applied to senses).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::concept_store::{ConceptStore, ConceptStoreError};
use crate::doc::{Doc, SentStart};
use crate::hash::hash_utf8;
use crate::interlingua::InterlinguaResolver;
use fluent_types::InterlinguaId;

/// A margin at or below this magnitude is treated as an oracle tie (near-tie →
/// [`AmbiguityKind::AttachmentNearTie`]).
pub const TIE_MARGIN_EPSILON: f64 = 1e-6;

/// Dep labels that name the sentence's subject argument.
const SUBJECT_DEPS: &[&str] = &["nsubj", "nsubjpass", "csubj", "csubjpass"];
/// Dep labels that name the direct object.
const DIRECT_OBJECT_DEPS: &[&str] = &["dobj"];
/// Dep labels that name the indirect object.
const INDIRECT_OBJECT_DEPS: &[&str] = &["iobj"];
/// Dep labels that carry modifiers of the head.
const MODIFIER_DEPS: &[&str] = &["compound", "amod", "nummod"];
/// Dep labels that form the prepositional argument frame.
const ARGUMENT_DEPS: &[&str] = &["prep", "pobj", "pcomp"];
/// The dependency label of a sentence root.
const ROOT_DEP: &str = "root";
/// The negation dependency label.
const NEG_DEP: &str = "neg";
/// The auxiliary dependency label.
const AUX_DEP: &str = "aux";
/// Modal auxiliaries conveying necessity.
const NECESSITY_AUX: &[&str] = &["must", "should", "shall", "need"];
/// Modal auxiliaries conveying possibility.
const POSSIBILITY_AUX: &[&str] = &["can", "could", "may", "might", "will", "would"];

/// The syntactic role of a frame's argument slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleType {
    Subject,
    DirectObject,
    IndirectObject,
    /// A prepositional argument (`prep` + its `pobj`).
    Argument,
    Modifier,
}

/// Sentence polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Positive,
    Negative,
}

/// Sentence modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Declarative,
    Interrogative,
    Necessity,
    Possibility,
}

/// The typed ambiguity of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityKind {
    AttachmentNearTie,
    PredicatePolysemy,
    NegationModalScope,
}

/// One ambiguity entry, tied to a token span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityEntry {
    pub kind: AmbiguityKind,
    /// The token span `(start, end)` the ambiguity concerns (the sentence
    /// span, by default).
    pub span: (usize, usize),
    /// A legible reason for the ambiguity (audit-trail friendly).
    pub detail: String,
    /// Candidate interlingua ids when the ambiguity is a sense/attachment
    /// choice (empty otherwise).
    pub candidate_ids: Vec<InterlinguaId>,
}

/// A typed argument structure per predicate: the predicate's interlingua lemma
/// id, the sentence span, the role slots (role + token span + candidate
/// concept ids), and the polarity/modality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub predicate_lemma_id: InterlinguaId,
    pub sentence_span: (usize, usize),
    pub roles: Vec<(RoleType, (usize, usize), Vec<InterlinguaId>)>,
    pub polarity: Polarity,
    pub modality: Modality,
}

/// The output of frame extraction: the frames plus the typed ambiguity list.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameAnalysis {
    pub frames: Vec<Frame>,
    pub ambiguities: Vec<AmbiguityEntry>,
}

impl FrameAnalysis {
    /// Whether the frame at `index` has an open ambiguity entry covering its
    /// sentence span. Only ambiguity-free frames mint permanent keys.
    #[must_use]
    pub fn has_open_ambiguity(&self, index: usize) -> bool {
        let Some(frame) = self.frames.get(index) else {
            return false;
        };
        self.ambiguities
            .iter()
            .any(|a| spans_overlap(a.span, frame.sentence_span))
    }
}

/// Whether two token spans overlap (half-open intervals).
fn spans_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// A content-addressed key for a frame. Permanent keys (ambiguity-free frames)
/// are the ones persisted to the ledger/graph; provisional keys mark frames
/// with an open ambiguity and are never persisted as resolved structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameKey {
    pub predicate_lemma_id: InterlinguaId,
    /// Role slots as `(role, role-token lemma id)`.
    pub roles: Vec<(RoleType, InterlinguaId)>,
    pub polarity: Polarity,
    pub modality: Modality,
    /// `true` for a provisional (ambiguity-open) key.
    pub provisional: bool,
}

impl FrameKey {
    /// A permanent key — minted only for an ambiguity-free frame.
    #[must_use]
    pub fn permanent(
        predicate_lemma_id: InterlinguaId,
        roles: Vec<(RoleType, InterlinguaId)>,
        polarity: Polarity,
        modality: Modality,
    ) -> Self {
        Self {
            predicate_lemma_id,
            roles,
            polarity,
            modality,
            provisional: false,
        }
    }

    /// A provisional key — minted for any frame with an open ambiguity entry.
    #[must_use]
    pub fn provisional(
        predicate_lemma_id: InterlinguaId,
        roles: Vec<(RoleType, InterlinguaId)>,
        polarity: Polarity,
        modality: Modality,
    ) -> Self {
        Self {
            predicate_lemma_id,
            roles,
            polarity,
            modality,
            provisional: true,
        }
    }
}

/// The disambiguation product of a resolved `(predicate, ambiguity_kind)`
/// pattern. Recorded in a [`PreferredSenseIndex`] and replayed deterministically
/// so the next occurrence never re-triggers an LLM call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    /// The chosen sense / attachment / scope id.
    pub chosen_candidate_id: InterlinguaId,
    /// A legible reason (audit-trail friendly).
    pub detail: String,
}

/// The promotion seam for resolved ambiguity patterns (mirrors
/// [`crate::review::CorrectionIndex`]): a `(predicate_lemma_id,
/// ambiguity_kind)` pattern that has been resolved is recorded and replayed
/// deterministically — golden-corpus-style rule genesis applied to senses.
///
/// The router implements this over the existing `interlingua_index`
/// correction-cache rows (the entity-scope column = the ambiguity kind).
pub trait PreferredSenseIndex: Send + Sync {
    /// The previously-recorded resolution for this pattern, when known.
    fn preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
    ) -> Option<Resolution>;
    /// Persist a resolution for this pattern.
    fn record_preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
        resolution: Resolution,
    ) -> Result<(), ConceptStoreError>;
}

/// The deterministic frame extractor: a resolver + concept store wired into
/// the pipeline as the `frame` stage (spacy-rs owns the structure; the router
/// implements the persistence seams).
#[derive(Clone)]
pub struct FrameExtractor {
    resolver: Arc<InterlinguaResolver>,
    concept_store: Arc<dyn ConceptStore>,
}

impl std::fmt::Debug for FrameExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameExtractor")
            .field("resolver", &self.resolver)
            .finish_non_exhaustive()
    }
}

impl FrameExtractor {
    /// An extractor over `resolver` and its shared `concept_store`.
    #[must_use]
    pub fn new(resolver: Arc<InterlinguaResolver>, concept_store: Arc<dyn ConceptStore>) -> Self {
        Self {
            resolver,
            concept_store,
        }
    }

    /// Extract frames + ambiguities for an attached `doc`. `margins` is the
    /// parse's per-oracle-step margins (for [`AmbiguityKind::AttachmentNearTie`]),
    /// threaded from [`crate::llm::AnnotationResult::oracle_margins`].
    #[must_use]
    pub fn extract(&self, doc: &Doc, margins: Option<&[f64]>) -> FrameAnalysis {
        extract_frames(doc, &self.resolver, &*self.concept_store, margins)
    }

    /// Mint the provisional-vs-permanent keys for every frame of `analysis`,
    /// resolving each role slot's lemma id against `doc`.
    /// Permanence also gates on `ConceptStore::state()==Ready` — while `Loading`, every key is provisional.
    #[must_use]
    pub fn keys(&self, doc: &Doc, analysis: &FrameAnalysis) -> Vec<FrameKey> {
        let loading = self.concept_store.state() == crate::concept_store::ConceptStoreState::Loading;
        analysis
            .frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let open = analysis.has_open_ambiguity(i) || loading;
                mint_frame_key(doc, &self.resolver, f, open)
            })
            .collect()
    }
}

/// Deterministically derive a frame analysis from an attached `doc`.
///
/// Sentence boundaries come from the `sent_start` markers (set by the
/// sentencizer and by `set_children_from_heads` at each root). The frame tree
/// reuses `Doc::children`/heads — the same dependency-tree discipline as
/// validator checks 5/6, never a hand-rolled graph walk.
///
/// `margins` (the parse's per-oracle-step margins) drives the
/// [`AmbiguityKind::AttachmentNearTie`] signal; `resolver` supplies the
/// predicate/role interlingua ids; `concept_store` backs polysemy candidate
/// enumeration.
#[must_use]
pub fn extract_frames(
    doc: &Doc,
    resolver: &InterlinguaResolver,
    concept_store: &dyn ConceptStore,
    margins: Option<&[f64]>,
) -> FrameAnalysis {
    if doc.is_empty() {
        return FrameAnalysis::default();
    }
    let len = doc.len();
    let mut starts: Vec<usize> = (0..len)
        .filter(|&i| doc.token(i).sent_start == SentStart::Start)
        .collect();
    if starts.is_empty() || starts[0] != 0 {
        starts.insert(0, 0);
    }
    starts.push(len);

    let attachment_tie = margins.is_some_and(|m| m.iter().any(|&x| x.abs() <= TIE_MARGIN_EPSILON));

    let mut frames = Vec::new();
    let mut ambiguities = Vec::new();

    for w in starts.windows(2) {
        let (s, e) = (w[0], w[1]);
        if s >= e {
            continue;
        }
        let root = (s..e).find(|&i| dep_is(doc.token(i), ROOT_DEP)).unwrap_or(s);
        let predicate_canonical = lemma_of(doc, root);
        let predicate_lemma_id = resolver.lemma_id(&predicate_canonical);

        let polarity = sentence_polarity(doc, s, e);
        let modality = sentence_modality(doc, root, s, e);

        let mut roles = Vec::new();
        for i in s..e {
            if i == root {
                continue;
            }
            let Some(role) = classify_role(doc.token(i)) else {
                continue;
            };
            let lemma_id = resolver.lemma_id(&lemma_of(doc, i));
            let candidates = concept_store.candidates(lemma_id);
            roles.push((
                role,
                (i, i + 1),
                candidates.iter().map(|c| c.id).collect(),
            ));
        }

        // Predicate polysemy: the lemma id resolves to more than one concept.
        let pred_candidates = concept_store.candidates(predicate_lemma_id);
        if pred_candidates.len() > 1 {
            ambiguities.push(AmbiguityEntry {
                kind: AmbiguityKind::PredicatePolysemy,
                span: (s, e),
                detail: format!(
                    "predicate lemma {:?} resolves to {} concepts",
                    predicate_canonical,
                    pred_candidates.len()
                ),
                candidate_ids: pred_candidates.iter().map(|c| c.id).collect(),
            });
        }
        // Negation/modal scope conflict: a neg and a modal aux governing the
        // same predicate overlap in scope.
        if polarity == Polarity::Negative
            && matches!(modality, Modality::Necessity | Modality::Possibility)
        {
            ambiguities.push(AmbiguityEntry {
                kind: AmbiguityKind::NegationModalScope,
                span: (s, e),
                detail: "negation and modal auxiliary scopes overlap".into(),
                candidate_ids: Vec::new(),
            });
        }
        // Attachment near-tie: the heuristic parser doubted an attachment.
        if attachment_tie {
            ambiguities.push(AmbiguityEntry {
                kind: AmbiguityKind::AttachmentNearTie,
                span: (s, e),
                detail: "oracle attachment near-tie (near-zero margin)".into(),
                candidate_ids: Vec::new(),
            });
        }

        frames.push(Frame {
            predicate_lemma_id,
            sentence_span: (s, e),
            roles,
            polarity,
            modality,
        });
    }

    FrameAnalysis { frames, ambiguities }
}

/// Mint the provisional-or-permanent key for a frame. A frame with any open
/// ambiguity entry gets a provisional key; an ambiguity-free frame a permanent
/// one (the only kind persisted to the ledger/graph).
#[must_use]
pub fn mint_frame_key(
    doc: &Doc,
    resolver: &InterlinguaResolver,
    frame: &Frame,
    open_ambiguity: bool,
) -> FrameKey {
    let roles = frame
        .roles
        .iter()
        .map(|(role, (s, _), _)| (*role, resolver.lemma_id(&lemma_of(doc, *s))))
        .collect();
    if open_ambiguity {
        FrameKey::provisional(frame.predicate_lemma_id, roles, frame.polarity, frame.modality)
    } else {
        FrameKey::permanent(frame.predicate_lemma_id, roles, frame.polarity, frame.modality)
    }
}

/// The lemma string for token `i` (through the shared store, falling back to
/// the lowercase surface form).
fn lemma_of(doc: &Doc, i: usize) -> String {
    doc.vocab()
        .strings()
        .get(doc.token(i).lemma)
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| doc.token_text(i).to_ascii_lowercase())
}

/// Whether a token's dep label equals `label` (hash comparison).
fn dep_is(token: &crate::doc::TokenRecord, label: &str) -> bool {
    hash_utf8(label) == token.dep
}

/// Whether a token's dep label is one of `labels` (hash comparison).
fn dep_in(token: &crate::doc::TokenRecord, labels: &[&str]) -> bool {
    labels.iter().any(|l| hash_utf8(l) == token.dep)
}

/// Classify a non-root token's dep into a [`RoleType`], or `None` for a dep
/// that is not a frame slot (det/aux/neg/punct/etc.).
fn classify_role(token: &crate::doc::TokenRecord) -> Option<RoleType> {
    if dep_in(token, SUBJECT_DEPS) {
        Some(RoleType::Subject)
    } else if dep_in(token, DIRECT_OBJECT_DEPS) {
        Some(RoleType::DirectObject)
    } else if dep_in(token, INDIRECT_OBJECT_DEPS) {
        Some(RoleType::IndirectObject)
    } else if dep_in(token, ARGUMENT_DEPS) {
        Some(RoleType::Argument)
    } else if dep_in(token, MODIFIER_DEPS) {
        Some(RoleType::Modifier)
    } else {
        None
    }
}

/// Sentence polarity from the presence of a `neg` dep in `[s, e)`.
fn sentence_polarity(doc: &Doc, s: usize, e: usize) -> Polarity {
    if (s..e).any(|i| dep_is(doc.token(i), NEG_DEP)) {
        Polarity::Negative
    } else {
        Polarity::Positive
    }
}

/// Sentence modality: interrogative from a `?` punct; necessity/possibility
/// from the predicate's modal `aux` children; declarative otherwise.
fn sentence_modality(doc: &Doc, root: usize, s: usize, e: usize) -> Modality {
    // An interrogative `?` terminal wins over the aux classification.
    if (s..e).any(|i| doc.token_text(i) == "?") {
        return Modality::Interrogative;
    }
    for &i in &doc.children(root) {
        if dep_is(doc.token(i), AUX_DEP) {
            let aux = lemma_of(doc, i);
            let lower = aux.to_ascii_lowercase();
            if NECESSITY_AUX.contains(&lower.as_str()) {
                return Modality::Necessity;
            }
            if POSSIBILITY_AUX.contains(&lower.as_str()) {
                return Modality::Possibility;
            }
        }
    }
    Modality::Declarative
}

#[cfg(test)]
#[path = "../tests/frame.rs"]
mod tests;
