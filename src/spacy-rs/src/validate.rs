//! The deterministic annotation validator gate (walkthrough §10.2).
//!
//! After `serde_json` deserializes the LLM's token-array reply into
//! [`crate::llm::AnnotationRecord`]s, this gate runs **before anything is
//! written to the Doc**. It enforces the section-9 tree/entity invariants
//! generalized from `Doc.from_array`'s validation (`doc.pyx:1115-1157`):
//!
//! 1. Count + text match — one record per token, `text` equals the
//!    deterministic tokenizer's orth.
//! 2. Closed vocabularies — `pos` in the 17 `Upos::UPOS`; `dep` in the
//!    configured [`DepLabelSet`] (default `ud_default()`, so UD labels like
//!    `compound`/`case` pass, garbage fails).
//! 3. Head bounds; no self-loop except ROOT.
//! 4. At least one ROOT (`dep == "root"` and `head == 0`) — one per sentence
//!    for multi-sentence docs, never zero.
//! 5. Connectivity — every token reaches the ROOT via heads.
//! 6. Acyclicity — no head cycles.
//! 7. BILUO entity spans well-formed; `ent_type` present where IOB ≠ O.
//!
//! On any failure the validator returns a typed [`AnnotationError`] (never a
//! partial application) and the fallback ladder engages (§10.3). The gate is
//! a **pure function** of `(doc, records)` — no Doc mutation — so it is the
//! `predict` half of the `predict`/`set_annotations` separation; the mutate
//! step (`crate::llm::apply`) runs only after this gate passes.
//!
//! Projectivity (§10.2 check 8, optional) is gated behind
//! [`AnnotationValidator::require_projectivity`], off by default.

use thiserror::Error;

use crate::doc::Doc;
use crate::labels::{DepLabelSet, Upos};
use crate::llm::{AnnotationRecord, AnnotationSet};

/// Errors from the deterministic annotation gate. Each variant corresponds to
/// one of the §10.2 checks, so a caller can discriminate the invalid class
/// without string matching.
#[derive(Debug, Error)]
pub enum AnnotationError {
    /// Check 1: the reply has a different number of records than tokens.
    #[error("annotation count mismatch: got {got} records for {expected} tokens")]
    CountMismatch { expected: usize, got: usize },
    /// Check 1: a record's `text` does not equal the token's orth.
    #[error("token {index} text mismatch: expected {expected:?}, got {got:?}")]
    TextMismatch {
        index: usize,
        expected: String,
        got: String,
    },
    /// Check 2: `pos` is not one of the 17 UPOS tags.
    #[error("unknown POS tag: {0:?}")]
    UnknownPos(String),
    /// Check 2: `dep` is not in the accepted label set.
    #[error("unknown dependency label: {0:?}")]
    UnknownDep(String),
    /// Check 3: `i + head` falls outside `0..len`.
    #[error("invalid head: token {token} has relative head {head}, outside 0..{len}")]
    HeadOutOfBounds { token: usize, head: i32, len: usize },
    /// Check 3: a non-ROOT token points at itself (`head == 0`).
    #[error("token {token} is a non-ROOT self-loop (head == 0)")]
    SelfLoop { token: usize },
    /// Check 4: a token declares `dep == "root"` but `head != 0`.
    #[error("token {token} has dep ROOT but head {head} != 0")]
    RootHeadMismatch { token: usize, head: i32 },
    /// Check 4: not exactly one ROOT.
    #[error("expected at least one ROOT (one per sentence), found {found}")]
    RootCount { expected: usize, found: usize },
    /// Check 5: tokens not reachable from the ROOT.
    #[error("tokens not reachable from ROOT: {0:?}")]
    Unreachable(Vec<usize>),
    /// Check 6: a cycle in the head structure (the cycle path).
    #[error("dependency head cycle: {0:?}")]
    Cycle(Vec<usize>),
    /// Check 7: a malformed BILUO entity span.
    #[error("malformed BILUO entity at token {index}: {detail}")]
    MalformedBiluo { index: usize, detail: String },
    /// Check 8 (optional): the parse is non-projective.
    #[error(
        "non-projective dependency edge: token {token} heads {head} across tokens {blocked:?}"
    )]
    NonProjective {
        token: usize,
        head: usize,
        blocked: Vec<usize>,
    },
    /// The reply is not parseable as the annotation JSON contract.
    #[error("annotation JSON parse failure: {source}")]
    Json {
        #[source]
        source: std::sync::Arc<serde_json::Error>,
    },
    /// Applying validated records to a Doc failed.
    #[error("annotation apply failure: {0}")]
    Apply(String),
    /// The doc has no tokens; nothing to annotate.
    #[error("cannot annotate an empty document")]
    EmptyDocument,
}

impl Clone for AnnotationError {
    fn clone(&self) -> Self {
        match self {
            Self::CountMismatch { expected, got } => Self::CountMismatch { expected: *expected, got: *got },
            Self::TextMismatch { index, expected, got } => Self::TextMismatch { index: *index, expected: expected.clone(), got: got.clone() },
            Self::UnknownPos(s) => Self::UnknownPos(s.clone()),
            Self::UnknownDep(s) => Self::UnknownDep(s.clone()),
            Self::HeadOutOfBounds { token, head, len } => Self::HeadOutOfBounds { token: *token, head: *head, len: *len },
            Self::SelfLoop { token } => Self::SelfLoop { token: *token },
            Self::RootHeadMismatch { token, head } => Self::RootHeadMismatch { token: *token, head: *head },
            Self::RootCount { expected, found } => Self::RootCount { expected: *expected, found: *found },
            Self::Unreachable(v) => Self::Unreachable(v.clone()),
            Self::Cycle(v) => Self::Cycle(v.clone()),
            Self::MalformedBiluo { index, detail } => Self::MalformedBiluo { index: *index, detail: detail.clone() },
            Self::NonProjective { token, head, blocked } => Self::NonProjective { token: *token, head: *head, blocked: blocked.clone() },
            Self::Json { source } => Self::Json { source: std::sync::Arc::clone(source) },
            Self::Apply(s) => Self::Apply(s.clone()),
            Self::EmptyDocument => Self::EmptyDocument,
        }
    }
}

impl PartialEq for AnnotationError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::CountMismatch { expected: a1, got: b1 }, Self::CountMismatch { expected: a2, got: b2 }) => a1 == a2 && b1 == b2,
            (Self::TextMismatch { index: i1, expected: e1, got: g1 }, Self::TextMismatch { index: i2, expected: e2, got: g2 }) => i1 == i2 && e1 == e2 && g1 == g2,
            (Self::UnknownPos(a), Self::UnknownPos(b)) => a == b,
            (Self::UnknownDep(a), Self::UnknownDep(b)) => a == b,
            (Self::HeadOutOfBounds { token: t1, head: h1, len: l1 }, Self::HeadOutOfBounds { token: t2, head: h2, len: l2 }) => t1 == t2 && h1 == h2 && l1 == l2,
            (Self::SelfLoop { token: t1 }, Self::SelfLoop { token: t2 }) => t1 == t2,
            (Self::RootHeadMismatch { token: t1, head: h1 }, Self::RootHeadMismatch { token: t2, head: h2 }) => t1 == t2 && h1 == h2,
            (Self::RootCount { expected: e1, found: f1 }, Self::RootCount { expected: e2, found: f2 }) => e1 == e2 && f1 == f2,
            (Self::Unreachable(a), Self::Unreachable(b)) => a == b,
            (Self::Cycle(a), Self::Cycle(b)) => a == b,
            (Self::MalformedBiluo { index: i1, detail: d1 }, Self::MalformedBiluo { index: i2, detail: d2 }) => i1 == i2 && d1 == d2,
            (Self::NonProjective { token: t1, head: h1, blocked: b1 }, Self::NonProjective { token: t2, head: h2, blocked: b2 }) => t1 == t2 && h1 == h2 && b1 == b2,
            (Self::Json { source: s1 }, Self::Json { source: s2 }) => s1.to_string() == s2.to_string(),
            (Self::Apply(a), Self::Apply(b)) => a == b,
            (Self::EmptyDocument, Self::EmptyDocument) => true,
            _ => false,
        }
    }
}

impl Eq for AnnotationError {}

/// The deterministic gate. Holds the accepted dependency-label vocabulary
/// (the `DepLabelSet`, serde-able so a model's `label_data` can override it)
/// and the optional projectivity requirement.
#[derive(Debug, Clone)]
pub struct AnnotationValidator {
    dep_labels: DepLabelSet,
    require_projectivity: bool,
}

impl Default for AnnotationValidator {
    fn default() -> Self {
        Self {
            dep_labels: DepLabelSet::ud_default(),
            require_projectivity: false,
        }
    }
}

impl AnnotationValidator {
    /// A validator with the canonical UD label set and no projectivity check.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A validator with a custom accepted dependency-label set.
    #[must_use]
    pub fn with_dep_labels(dep_labels: DepLabelSet) -> Self {
        Self {
            dep_labels,
            require_projectivity: false,
        }
    }

    /// Require a projective dependency tree (check 8, §10.2 "optionally").
    #[must_use]
    pub fn require_projectivity(mut self, require: bool) -> Self {
        self.require_projectivity = require;
        self
    }

    /// The accepted dependency-label set.
    #[must_use]
    pub fn dep_labels(&self) -> &DepLabelSet {
        &self.dep_labels
    }

    /// Run the full 7-check gate (plus the optional projectivity check) over
    /// `records` for `doc`. Pure: never mutates `doc`.
    pub fn validate(&self, doc: &Doc, set: &AnnotationSet) -> Result<(), AnnotationError> {
        validate(doc, set, &self.dep_labels, self.require_projectivity)
    }
}

/// Free-function form of the gate with an explicit label set.
pub fn validate(
    doc: &Doc,
    set: &AnnotationSet,
    dep_labels: &DepLabelSet,
    require_projectivity: bool,
) -> Result<(), AnnotationError> {
    if doc.is_empty() {
        return Err(AnnotationError::EmptyDocument);
    }

    // ── Check 1: count + text ──
    let records = &set.0;
    if records.len() != doc.len() {
        return Err(AnnotationError::CountMismatch {
            expected: doc.len(),
            got: records.len(),
        });
    }
    for (i, rec) in records.iter().enumerate() {
        let orth = doc.token_text(i);
        if rec.text != orth {
            return Err(AnnotationError::TextMismatch {
                index: i,
                expected: orth,
                got: rec.text.clone(),
            });
        }
    }

    // ── Check 2: closed vocabularies ──
    for rec in records {
        let pos: Upos = rec
            .pos
            .parse()
            .map_err(|_e| AnnotationError::UnknownPos(rec.pos.clone()))?;
        if !Upos::UPOS.contains(&pos) {
            return Err(AnnotationError::UnknownPos(rec.pos.clone()));
        }
        if !dep_labels.contains(&rec.dep) {
            return Err(AnnotationError::UnknownDep(rec.dep.clone()));
        }
    }

    // ── Check 3: head bounds; no self-loop except ROOT ──
    let len = records.len();
    for (i, rec) in records.iter().enumerate() {
        let abs = i as i64 + i64::from(rec.head);
        if abs < 0 || abs >= len as i64 {
            return Err(AnnotationError::HeadOutOfBounds {
                token: i,
                head: rec.head,
                len,
            });
        }
        let is_root = rec.dep.eq_ignore_ascii_case("root");
        if rec.head == 0 && !is_root {
            return Err(AnnotationError::SelfLoop { token: i });
        }
        if is_root && rec.head != 0 {
            return Err(AnnotationError::RootHeadMismatch {
                token: i,
                head: rec.head,
            });
        }
    }

    // ── Check 4: at least one ROOT ──
    // One `dep == "root"` (and `head == 0`) token per sentence; a multi-sentence
    // doc legitimately has several. Zero roots is always invalid.
    let roots: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.dep.eq_ignore_ascii_case("root"))
        .map(|(i, _)| i)
        .collect();
    if roots.is_empty() {
        return Err(AnnotationError::RootCount {
            expected: 1,
            found: 0,
        });
    }

    // ── Checks 5 + 6: connectivity and acyclicity (dag/SKILL prime
    // directive — compose DependencyGraph, never hand-roll graph algorithms) ──
    // Every token is registered as a node that self-provides and *depends on
    // its head* (a root token has no deps). `topo_sort` detects head cycles
    // (Kahn's algorithm → GraphError::Cycle); `dependents_of(root)` walks the
    // transitive dependent closure for connectivity. Because every non-root
    // chain must terminate at a no-dep node (a ROOT, guaranteed by checks 3–4)
    // and cycles are rejected, each token is reachable from some root.
    let mut graph = fluent_dag::dep_graph::DependencyGraph::<usize>::new();
    for (i, rec) in records.iter().enumerate() {
        let is_root = rec.dep.eq_ignore_ascii_case("root");
        let deps: Vec<usize> = if is_root {
            Vec::new()
        } else {
            vec![(i as i64 + i64::from(rec.head)) as usize]
        };
        graph
            .register(&i, &deps, &[i])
            .map_err(|_e| AnnotationError::Cycle(Vec::new()))?;
    }
    if let Err(_cycle) = graph.topo_sort() {
        return Err(AnnotationError::Cycle(Vec::new()));
    }
    // Connectivity: every token must be reachable from some ROOT.
    let mut reachable: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, rec) in records.iter().enumerate() {
        if rec.dep.eq_ignore_ascii_case("root") {
            reachable.insert(i);
            reachable.extend(graph.dependents_of(&i));
        }
    }
    let unreached: Vec<usize> = (0..len).filter(|i| !reachable.contains(i)).collect();
    if !unreached.is_empty() {
        return Err(AnnotationError::Unreachable(unreached));
    }

    // ── Check 7: BILUO entity spans ──
    let mut open: Option<(String, bool)> = None; // (type, span_closed)
    for (i, rec) in records.iter().enumerate() {
        let iob = rec.ent_iob.trim().to_ascii_uppercase();
        let ty = rec.ent_type.trim();
        match iob.as_str() {
            "" | "O" => {
                if matches!(open, Some((_, false))) {
                    return Err(AnnotationError::MalformedBiluo {
                        index: i,
                        detail: "unterminated span closed by outside token".into(),
                    });
                }
                open = None;
            }
            "B" | "U" => {
                if matches!(open, Some((_, false))) {
                    return Err(AnnotationError::MalformedBiluo {
                        index: i,
                        detail: "new span begins before the previous span is closed with L".into(),
                    });
                }
                if ty.is_empty() {
                    return Err(AnnotationError::MalformedBiluo {
                        index: i,
                        detail: format!("{iob} requires ent_type"),
                    });
                }
                open = Some((ty.to_string(), iob == "U"));
            }
            "I" | "L" => match &open {
                Some((pty, closed)) if !*closed && pty == ty => {
                    open = Some((ty.to_string(), iob == "L"));
                }
                Some((pty, _)) => {
                    return Err(AnnotationError::MalformedBiluo {
                        index: i,
                        detail: format!(
                            "{iob} must continue a B/I of the same type (open {pty:?})"
                        ),
                    });
                }
                None => {
                    return Err(AnnotationError::MalformedBiluo {
                        index: i,
                        detail: format!("{iob} without a preceding B of the same type"),
                    });
                }
            },
            other => {
                return Err(AnnotationError::MalformedBiluo {
                    index: i,
                    detail: format!("unknown IOB marker {other:?}"),
                });
            }
        }
    }
    if matches!(open, Some((_, false))) {
        return Err(AnnotationError::MalformedBiluo {
            index: records.len() - 1,
            detail: "span ends without L".into(),
        });
    }

    // ── Check 8 (optional): projectivity ──
    if require_projectivity {
        check_projectivity(records)?;
    }

    Ok(())
}

/// For every edge `token → head`, every token strictly between them must be a
/// descendant of `head` (a non-projective edge crosses a non-descendant).
fn check_projectivity(records: &[AnnotationRecord]) -> Result<(), AnnotationError> {
    for (i, rec) in records.iter().enumerate() {
        if rec.head == 0 {
            continue;
        }
        let h = (i as i64 + i64::from(rec.head)) as usize;
        let (lo, hi) = if i < h { (i, h) } else { (h, i) };
        for j in (lo + 1)..hi {
            if !is_descendant(records, j, h) {
                return Err(AnnotationError::NonProjective {
                    token: i,
                    head: h,
                    blocked: vec![j],
                });
            }
        }
    }
    Ok(())
}

/// Whether `j` is a descendant of `h` (walking `j`'s ancestors up to `h`).
fn is_descendant(records: &[AnnotationRecord], j: usize, h: usize) -> bool {
    let mut cur = j;
    let mut guard = 0;
    while guard <= records.len() {
        let head = (cur as i64 + i64::from(records[cur].head)) as usize;
        if head == cur {
            return false; // reached a root without passing h
        }
        if head == h {
            return true;
        }
        cur = head;
        guard += 1;
    }
    false
}

#[cfg(test)]
#[path = "../tests/validate.rs"]
mod tests;
