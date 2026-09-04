//! The LLM-JSON bridge (walkthrough §10): the annotation JSON contract, its
//! deserialization, and the deterministic **attach** step.
//!
//! # The predict / set_annotations separation
//!
//! spaCy's `TrainablePipe.__call__` is `scores = predict(docs);
//! set_annotations(docs, scores)` (`trainable_pipe.pyx:40`) — the single most
//! important design pattern for the port. Here:
//!
//! - [`AnnotationRecord`] + [`AnnotationSet::parse_json`] are the **predict**
//!   output: the finetuned model's reply as typed data, with **no Doc
//!   mutation**.
//! - [`crate::validate::validate`] is the deterministic gate (§10.2) — the
//!   validator runs before anything is written.
//! - [`apply`] is `set_annotations`: the pure writer that stores the record
//!   fields into the `TokenRecord`s and rebuilds the dependency tree. It
//!   re-runs the gate defensively; [`attach`] is the gate-free writer the
//!   pipeline calls after its own validated ladder.
//!
//! The JSON schema ([`AnnotationRecord::contract`]) is generated from the
//! **same closed label vocabularies** the validator checks against
//! (`Upos::UPOS`, `DepLabelSet::ud_default`), so the prompt the model is
//! trained on and the schema the reply is validated against cannot drift
//! (§10.6).

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::arc_eager::ParseConfidence;
use crate::doc::Doc;
use crate::error::SpacyError;
use crate::hash::hash_utf8;
use crate::labels::{DepLabelSet, EntIoB, Upos};
use crate::validate::{AnnotationError, AnnotationValidator};

/// One token's annotation, mirroring the §10.1 JSON contract. `pos`/`dep`
/// stay strings here (the wire type); the validator resolves them against the
/// closed vocabularies (check 2) and `apply` parses them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnotationRecord {
    /// The surface form as tokenized by the deterministic tokenizer.
    pub text: String,
    /// UPOS value (one of the 17: `adj adp adv aux cconj det intj noun num
    /// part pron propn punct sconj sym verb x`).
    pub pos: String,
    /// Fine-grained tag (e.g. `NNP`, `VBZ`) — optional for the lean port.
    #[serde(default)]
    pub tag: String,
    /// Dependency label from the closed label set (`nsubj`, `dobj`, `prep`,
    /// `pobj`, `compound`, `aux`, `root`, ...). **This is the routing signal**
    /// for Coral Router tool selection (§10.4).
    pub dep: String,
    /// Relative signed offset to the head token: `token.i + head ==
    /// head_index`.
    pub head: i32,
    /// Base form (lowercase by convention).
    #[serde(default)]
    pub lemma: String,
    /// UFEATS morphology string or `""`.
    #[serde(default)]
    pub morph: String,
    /// BILUO entity marker or `""`/`"O"` for outside.
    #[serde(default)]
    pub ent_iob: String,
    /// Entity type (required where `ent_iob != O`).
    #[serde(default)]
    pub ent_type: String,
}

impl AnnotationRecord {
    /// The JSON schema for the annotation contract, generated from the same
    /// closed vocabularies the validator checks (§10.6 — the single source of
    /// truth for the LLM prompt and the validation gate).
    #[must_use]
    pub fn contract() -> serde_json::Value {
        let pos: Vec<String> = Upos::UPOS.iter().map(ToString::to_string).collect();
        let dep = DepLabelSet::ud_default();
        json!({
            "type": "array",
            "description": "spaCy-compatible annotation for the tokenized sentence",
            "items": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "surface form from the deterministic tokenizer" },
                    "pos": { "type": "string", "enum": pos, "description": "UPOS tag from the 17-tag set" },
                    "tag": { "type": "string", "description": "fine-grained tag (optional)" },
                    "dep": { "type": "string", "enum": dep.to_sorted_vec(), "description": "dependency relation from the accepted set" },
                    "head": { "type": "integer", "description": "relative signed offset; token.i + head == head_index" },
                    "lemma": { "type": "string", "description": "base form (lowercase by convention)" },
                    "morph": { "type": "string", "description": "UFEATS morphology string or empty" },
                    "ent_iob": { "type": "string", "enum": ["", "B", "I", "L", "U", "O"], "description": "BILUO entity marker" },
                    "ent_type": { "type": "string", "description": "entity type (required where ent_iob != O)" }
                },
                "required": ["text", "pos", "dep", "head"]
            }
        })
    }

    /// The canonical annotation system prompt for a token list: asks the model
    /// to reply with ONLY the §10.1 JSON array matching [`Self::contract`].
    /// This is the single source of truth for the LLM annotation rung — the
    /// live-ai test and the Coral Router's `NlpStage` fetch both build their
    /// prompt here, so the wire contract cannot drift between them.
    #[must_use]
    pub fn prompt(tokens: &[String]) -> String {
        format!(
            "Annotate each token with spaCy-compatible JSON. Reply with ONLY a JSON array \
             matching this schema:\n{}\nTokens (in order): {:?}",
            Self::contract(),
            tokens,
        )
    }
}

/// The span-scoped refinement prompt contract (ROADMAP_20260831_ARCEAGER
/// M2.2): given the deterministic base parse and the focused token indices,
/// the model reconsiders **only** those tokens and replies with a corrections
/// object — the same shape the review contract already uses, minus
/// `old_value`. Like [`AnnotationRecord::prompt`], this is the single source
/// of truth for the refine wire format: the live-ai refine test and the
/// router's refine fetch both build their prompt here.
pub struct LlmRefinePrompt;

impl LlmRefinePrompt {
    /// The JSON schema of the refine reply, generated from the same closed
    /// field vocabulary ([`CorrectionField`]) the amendment step parses
    /// (§10.6's no-drift rule applied to the refine contract).
    #[must_use]
    pub fn contract() -> serde_json::Value {
        use crate::review::CorrectionField;
        let field: Vec<String> = [
            CorrectionField::Dep,
            CorrectionField::Head,
            CorrectionField::Lemma,
            CorrectionField::Pos,
        ]
        .iter()
        .filter_map(|f| {
            serde_json::to_value(f)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
        })
        .collect();
        json!({
            "type": "object",
            "description": "token-level corrections for the focused tokens only",
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "token_index": { "type": "integer", "description": "index into the token list" },
                            "field": { "type": "string", "enum": field, "description": "record field to amend" },
                            "new_value": { "type": "string", "description": "replacement value (head as an integer string)" }
                        },
                        "required": ["token_index", "field", "new_value"]
                    }
                },
                "note": { "type": "string", "description": "optional free-form remark" }
            },
            "required": ["corrections"]
        })
    }

    /// The refine system prompt: shows the base parse line by line, marks the
    /// focus tokens, and asks for corrections limited to those indices.
    /// Corrections outside the focus are ignored by the refiner, so the
    /// deterministic base can never be silently rewritten off-focus.
    #[must_use]
    pub fn prompt(tokens: &[String], base: &AnnotationResult, focus: &[usize]) -> String {
        use std::fmt::Write;
        let mut prompt = String::new();
        let _ = writeln!(
            prompt,
            "The deterministic parser produced this base parse. Reconsider ONLY the \
             tokens marked FOCUS below and reply with corrections for them; omit any \
             token that is already right. Reply with ONLY a JSON object matching this \
             schema:\n{}",
            Self::contract(),
        );
        let _ = writeln!(prompt, "Base parse (index / token / pos / dep / head / lemma):");
        for (i, r) in base.records().records().iter().enumerate() {
            let mark = if focus.contains(&i) { " FOCUS" } else { "" };
            let _ = writeln!(prompt, "  {i}: {} / {} / {} / {} / {}{mark}", r.text, r.pos, r.dep, r.head, r.lemma);
        }
        let _ = writeln!(prompt, "Focus token indices: {focus:?}");
        let _ = writeln!(prompt, "Tokens (in order): {tokens:?}");
        prompt
    }
}

/// An ordered set of token annotations, one per token of a doc.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AnnotationSet(pub Vec<AnnotationRecord>);

impl AnnotationSet {
    /// Deserialize the LLM reply (the §10.1 token array). Structural parse
    /// only — the closed-vocabulary / tree checks are the validator's job.
    pub fn parse_json(json: &str) -> Result<Self, AnnotationError> {
        serde_json::from_str(json).map_err(|e| AnnotationError::Json { source: std::sync::Arc::new(e) })
    }

    /// The records.
    #[must_use]
    pub fn records(&self) -> &[AnnotationRecord] {
        &self.0
    }

    /// Token count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Run the 7-check gate for `doc` with `validator`. Pure.
    pub fn validate(
        &self,
        validator: &AnnotationValidator,
        doc: &Doc,
    ) -> Result<(), AnnotationError> {
        validator.validate(doc, self)
    }
}

/// Which rung of the annotation ladder produced a parse (ROADMAP §9.1). The
/// wire `AnnotationSet` is provenance-blind; this rides alongside it through
/// the typed channel so downstream routing can see how trustworthy a parse is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationSource {
    /// The LLM rung (full UD deps).
    Llm,
    /// The deterministic transition parser (heuristic ArcEager).
    ArcEager,
    /// The terminal deterministic rule rung (star parse).
    RuleRung,
    /// A parse that was corrected by human/reviewer feedback.
    HumanReview,
    /// A trained-encoder rung (ROADMAP_20260827_ORT §2.7/§4): hidden states →
    /// small heads produce the annotations. Confidence rides per-token like
    /// ArcEager, so a low encoder confidence is treated the same way.
    Encoder,
    /// A frontier/remote provider produced the parse — a future substitution
    /// point (no task adapter is built now). Carried so the exhaustive
    /// `match`es stay honest (the no-wildcard rule): every existing
    /// provenance dispatch must account for it.
    Frontier,
}

impl AnnotationSource {
    /// The ledger provenance tier this producing rung maps to (ROADMAP M4).
    /// Deterministic rungs → `Deterministic`, local-model rungs →
    /// `LocalModel`, human review → `HumanReview`, frontier → `Frontier`.
    /// Exhaustive by design: a new source is a compile error, never a silent
    /// default.
    #[must_use]
    pub fn tier(self) -> fluent_types::ProvenanceTier {
        match self {
            AnnotationSource::ArcEager | AnnotationSource::RuleRung => {
                fluent_types::ProvenanceTier::Deterministic
            }
            AnnotationSource::Llm | AnnotationSource::Encoder => {
                fluent_types::ProvenanceTier::LocalModel
            }
            AnnotationSource::HumanReview => fluent_types::ProvenanceTier::HumanReview,
            AnnotationSource::Frontier => fluent_types::ProvenanceTier::Frontier,
        }
    }

    /// Whether this rung produces a parse confidence routing should gate on
    /// (ArcEager/encoder do; LLM/rule/human-review report 1.0 by convention).
    ///
    /// The match is exhaustive by design — adding a variant breaks the build
    /// instead of silently compiling a `_ => false` wildcard that would treat
    /// a future confidence-bearing rung as fail-open (the bug class that
    /// already bit once, ROADMAP_20260829 §1.5).
    #[must_use]
    pub fn is_confidence_bearing(self) -> bool {
        match self {
            AnnotationSource::Llm
            | AnnotationSource::RuleRung
            | AnnotationSource::HumanReview
            | AnnotationSource::Frontier => false,
            AnnotationSource::ArcEager | AnnotationSource::Encoder => true,
        }
    }
}

/// The ladder handoff: an [`AnnotationSet`] plus provenance and confidence
/// (ROADMAP §9.1 — F7 keeps low-confidence parses flowing forward; confidence
/// gates downstream *routing*, never rung fallthrough).
#[derive(Debug, Clone)]
pub struct AnnotationResult {
    pub records: AnnotationSet,
    pub source: AnnotationSource,
    /// Per-token confidence (ArcEager fills it; Llm/Rule set `None`).
    pub token_confidence: Option<Vec<f64>>,
    /// Parse-level confidence (ArcEager); `None` otherwise.
    pub parse_confidence: Option<ParseConfidence>,
    /// Per-oracle-step margins (ArcEager); `None` otherwise. A near-zero
    /// margin is an attachment near-tie — the frame stage's
    /// `AmbiguityKind::AttachmentNearTie` signal (ROADMAP M3).
    pub oracle_margins: Option<Vec<f64>>,
    /// Interlingua collisions surfaced by the resolve step (G9): the number of
    /// `CollisionNote::Collision` entries `InterlinguaResolver::resolve_doc`
    /// produced for this parse. `0` when no resolver is wired (or no
    /// collision). Consumed by the router's `NlpConfidenceSummary`.
    pub collision_count: usize,
}

impl AnnotationResult {
    /// A result with no confidence (the LLM and rule rungs) and no collisions.
    #[must_use]
    pub fn new(set: AnnotationSet, source: AnnotationSource) -> Self {
        Self {
            records: set,
            source,
            token_confidence: None,
            parse_confidence: None,
            oracle_margins: None,
            collision_count: 0,
        }
    }

    /// Attach confidence to an already-built result.
    #[must_use]
    pub fn with_confidence(
        mut self,
        token: Option<Vec<f64>>,
        parse: Option<ParseConfidence>,
    ) -> Self {
        self.token_confidence = token;
        self.parse_confidence = parse;
        self
    }

    /// The annotation set (the wire records).
    #[must_use]
    pub fn records(&self) -> &AnnotationSet {
        &self.records
    }

    /// The per-token confidence vector, when the producing rung filled it.
    #[must_use]
    pub fn token_confidence(&self) -> Option<&[f64]> {
        self.token_confidence.as_deref()
    }

    /// The producing rung.
    #[must_use]
    pub const fn source(&self) -> AnnotationSource {
        self.source
    }

    pub fn parse_confidence_mut(&mut self) -> Option<&mut ParseConfidence> {
        self.parse_confidence.as_mut()
    }
}

/// Validate (with the default UD label set) then write `set` into `doc`.
/// The convenient gate+attach path for callers using the canonical labels.
pub fn apply(doc: &mut Doc, set: &AnnotationSet) -> Result<(), AnnotationError> {
    apply_with(doc, set, &AnnotationValidator::new())
}

/// Validate (with a custom validator) then write `set` into `doc`.
pub fn apply_with(
    doc: &mut Doc,
    set: &AnnotationSet,
    validator: &AnnotationValidator,
) -> Result<(), AnnotationError> {
    validator.validate(doc, set)?;
    attach(doc, set)
}

/// Write `set` into `doc` **without re-validating**. Callers must have run
/// the gate first (the pipeline does: its ladder validates each rung before
/// accepting it). This is the `set_annotations` mutate step — the only place
/// token records are written from annotations.
pub fn attach(doc: &mut Doc, set: &AnnotationSet) -> Result<(), AnnotationError> {
    if set.0.len() != doc.len() {
        return Err(AnnotationError::Apply(format!(
            "record count {} does not match token count {}",
            set.0.len(),
            doc.len()
        )));
    }
    // The shared store: dep/lemma strings are interned (spaCy's
    // `vocab.strings.add` in the setter paths) so their hashes resolve back to
    // strings — the routing-signal extraction (`routing.rs`) needs the reverse
    // mapping. Cloned to avoid a borrow conflict with the `token_mut` below.
    let strings = Arc::clone(doc.vocab().strings());
    for (i, rec) in set.0.iter().enumerate() {
        let morph_key = if rec.morph.is_empty() {
            0
        } else {
            doc.vocab().morphology().add(&rec.morph)
        };
        let lemma = if rec.lemma.is_empty() {
            rec.text.to_ascii_lowercase()
        } else {
            rec.lemma.clone()
        };
        strings.add(&rec.dep);
        strings.add(&lemma);
        let token = doc.token_mut(i);
        token.pos = rec
            .pos
            .parse::<Upos>()
            .map_err(|_e: SpacyError| AnnotationError::UnknownPos(rec.pos.clone()))?;
        token.tag = hash_utf8(&rec.tag);
        token.dep = hash_utf8(&rec.dep);
        token.head = rec.head;
        token.lemma = hash_utf8(&lemma);
        token.morph = morph_key;
        token.ent_iob = match rec.ent_iob.trim().to_ascii_uppercase().as_str() {
            "B" | "U" => EntIoB::Begin,
            "I" | "L" => EntIoB::Inside,
            "O" | "" => EntIoB::Outside,
            other => {
                return Err(AnnotationError::Apply(format!(
                    "invalid BILUO marker {other:?} for token {i}"
                )))
            }
        };
        token.ent_type = if rec.ent_type.is_empty() {
            0
        } else {
            hash_utf8(&rec.ent_type)
        };
    }
    doc.set_children_from_heads()
        .map_err(|e| AnnotationError::Apply(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/llm.rs"]
mod tests;
