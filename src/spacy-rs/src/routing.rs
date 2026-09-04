//! DEP-as-routing-signal extraction (walkthrough §10.4; roadmap §6).
//!
//! Given an **attached** [`Doc`] (deps/heads set by the LLM rung or the rule
//! rung), each sentence is reduced to a [`RoutingSignal`]: the predicate
//! (root lemma), the argument frame (`nsubj`/`dobj`/`iobj` pick the tool and
//! its arguments; `compound`/`amod`/`nummod` carry modifiers; `aux`/`neg`/
//! `advmod` qualify intent; `prep` + `pobj` carry the argument frame), plus a
//! per-token transcript (tokens/lemmas/pos/deps/heads) for the ledger
//! `ContentNode` metadata.
//!
//! Role classification compares the token's dep hash against the closed role
//! vocabulary (`hash_utf8` of the label) — no `StringStore` round-trip on the
//! hot classification path. The transcript strings (lemmas/deps) resolve
//! through the store, which [`crate::llm::attach`] populates by interning the
//! strings it hashes.
//!
//! The deterministic rule rung produces a **star parse** (one `root` per
//! sentence, everything else a flat `dep`), so a doc annotated without the LLM
//! degrades gracefully: the signal carries the predicate and the residual
//! `dependents`, with the role slots empty. The extraction itself is
//! dep-complete — a full UD parse fills every slot.

use serde::{Deserialize, Serialize};

use crate::doc::{Doc, SentStart, TokenRecord};
use crate::hash::hash_utf8;
use fluent_types::InterlinguaId;

/// Dep labels that name the sentence's subject argument.
const SUBJECT_DEPS: &[&str] = &["nsubj", "nsubjpass", "csubj", "csubjpass"];
/// Dep labels that name the direct object.
const DIRECT_OBJECT_DEPS: &[&str] = &["dobj"];
/// Dep labels that name the indirect object.
const INDIRECT_OBJECT_DEPS: &[&str] = &["iobj"];
/// Dep labels that carry modifiers of the head.
const MODIFIER_DEPS: &[&str] = &["compound", "amod", "nummod"];
/// Dep labels that qualify intent (auxiliaries, negation, manner adverbials).
const QUALIFIER_DEPS: &[&str] = &["aux", "auxpass", "neg", "advmod"];
/// Dep labels whose `pobj` children form the argument frame.
const PREP_DEPS: &[&str] = &["prep"];
/// The objects of a `prep` that complete an argument frame.
const POBJ_DEPS: &[&str] = &["pobj", "pcomp"];
/// The dependency label of a sentence root.
const ROOT_DEP: &str = "root";

/// The interlingua frame of one sentence (ROADMAP §11.6): the content-
/// addressed ids of the routing-relevant tokens. This is what lets Coral
/// Router dispatch deterministically on the parse — the same phrasing
/// collapses to the same predicate/object ids regardless of surface words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterlinguaSignal {
    /// The predicate (root) token's lemma id.
    pub predicate_id: Option<InterlinguaId>,
    /// The subject token's lemma id.
    pub subject_id: Option<InterlinguaId>,
    /// The direct-object token's lemma id.
    pub direct_object_id: Option<InterlinguaId>,
    /// The indirect-object token's lemma id.
    pub indirect_object_id: Option<InterlinguaId>,
    /// PROPN tokens with a YaGO match (their `interlingua_entity_id`), for
    /// concept-aware routing and review.
    pub concept_ids: Vec<InterlinguaId>,
    /// Per-token lemma ids, aligned with the `tokens` transcript. A token
    /// whose lemma was never resolved carries `InterlinguaId(0)` (the
    /// RESERVED namespace id — the documented none-sentinel).
    pub token_ids: Vec<InterlinguaId>,
    /// Per-sentence parse confidence (the mean of the sentence tokens'
    /// confidence; `None` when the parse carried none). The `confidence_min`
    /// gate on `match_interlingua` filters reads this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// The routing frame of one sentence, plus its per-token transcript.
///
/// Serde-serializable so it can ride in `StageDecision.metadata` and be stored
/// in a ledger `ContentNode`'s metadata verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingSignal {
    /// The sentence text (token orths joined with spaces).
    pub sentence: String,
    /// The predicate — the root token's lemma (a verb, or the nominal head of
    /// a verbless sentence).
    pub predicate: String,
    /// The `nsubj`/`nsubjpass`/`csubj` subject's lemma.
    pub subject: Option<String>,
    /// The `dobj` direct object's lemma.
    pub direct_object: Option<String>,
    /// The `iobj` indirect object's lemma.
    pub indirect_object: Option<String>,
    /// `compound`/`amod`/`nummod` modifier lemmas.
    pub modifiers: Vec<String>,
    /// `aux`/`auxpass`/`neg`/`advmod` intent qualifiers.
    pub qualifiers: Vec<String>,
    /// `prep` + `pobj` argument frames: `(preposition lemma, object lemma)`.
    pub arguments: Vec<(String, String)>,
    /// Residual lemmas whose dep is not one of the role labels (the flat
    /// `dep` children of the star parse land here).
    pub dependents: Vec<String>,
    /// Per-token surface forms.
    pub tokens: Vec<String>,
    /// Per-token lemmas.
    pub lemmas: Vec<String>,
    /// Per-token UPOS names (lowercase).
    pub pos: Vec<String>,
    /// Per-token dep labels.
    pub deps: Vec<String>,
    /// Per-token relative head offsets.
    pub heads: Vec<i32>,
    /// The interlingua frame (ids), when the doc carries resolved ids
    /// (ROADMAP §11.6). `None` for an unresolveable sentence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interlingua: Option<InterlinguaSignal>,
}

/// Extract one [`RoutingSignal`] per sentence of an attached `doc`.
///
/// Sentence boundaries come from the `sent_start` markers (set by the
/// sentencizer and by `set_children_from_heads` at each root). A doc with no
/// markers is treated as a single sentence. Requires an attached doc (deps and
/// heads set); a raw tokenizer output yields degenerate signals (first token
/// as predicate, everything else a dependent).
#[must_use]
pub fn extract_routing_signals(doc: &Doc) -> Vec<RoutingSignal> {
    if doc.is_empty() {
        return Vec::new();
    }
    let len = doc.len();
    let mut starts: Vec<usize> = (0..len)
        .filter(|&i| doc.token(i).sent_start == SentStart::Start)
        .collect();
    if starts.is_empty() || starts[0] != 0 {
        starts.insert(0, 0);
    }
    starts.push(len);
    starts
        .windows(2)
        .filter(|w| w[0] < w[1])
        .map(|w| signal_for_sentence(doc, w[0], w[1]))
        .collect()
}

/// Build the signal for the token span `[start, end)`.
fn signal_for_sentence(doc: &Doc, start: usize, end: usize) -> RoutingSignal {
    let root = (start..end)
        .find(|&i| dep_is(doc.token(i), ROOT_DEP))
        .unwrap_or(start);

    let strings = doc.vocab().strings();
    let lemma_of = |i: usize| -> String {
        strings
            .get(doc.token(i).lemma)
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| doc.token_text(i).to_ascii_lowercase())
    };
    let dep_of = |i: usize| -> String {
        strings
            .get(doc.token(i).dep)
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    let mut subject = None;
    let mut subject_idx = None;
    let mut direct_object = None;
    let mut direct_object_idx = None;
    let mut indirect_object = None;
    let mut indirect_object_idx = None;
    let mut modifiers = Vec::new();
    let mut qualifiers = Vec::new();
    let mut arguments = Vec::new();
    let mut dependents = Vec::new();
    let mut concept_ids = Vec::new();

    for i in start..end {
        if i == root {
            continue;
        }
        let token = doc.token(i);
        if dep_in(token, SUBJECT_DEPS) {
            subject = Some(lemma_of(i));
            subject_idx = Some(i);
        } else if dep_in(token, DIRECT_OBJECT_DEPS) {
            direct_object = Some(lemma_of(i));
            direct_object_idx = Some(i);
        } else if dep_in(token, INDIRECT_OBJECT_DEPS) {
            indirect_object = Some(lemma_of(i));
            indirect_object_idx = Some(i);
        } else if dep_in(token, MODIFIER_DEPS) {
            modifiers.push(lemma_of(i));
        } else if dep_in(token, QUALIFIER_DEPS) {
            qualifiers.push(lemma_of(i));
        } else if dep_in(token, PREP_DEPS) {
            let prep = lemma_of(i);
            for &c in &doc.children(i) {
                if dep_in(doc.token(c), POBJ_DEPS) {
                    arguments.push((prep.clone(), lemma_of(c)));
                }
            }
        } else if dep_in(token, POBJ_DEPS) {
            // A `pobj` is captured under its `prep`'s argument frame above;
            // not a standalone role.
        } else {
            dependents.push(lemma_of(i));
        }
        if let Some(ent) = token.interlingua_entity_id {
            if ent.is_yago() {
                concept_ids.push(ent);
            }
        }
    }

    let tokens: Vec<String> = (start..end).map(|i| doc.token_text(i)).collect();
    let sentence = tokens.join(" ");
    let lemmas = (start..end).map(lemma_of).collect();
    let pos = (start..end).map(|i| doc.token(i).pos.to_string()).collect();
    let deps = (start..end).map(dep_of).collect();
    let heads = (start..end).map(|i| doc.token(i).head).collect();

    // Sentence confidence: the mean of the tokens' confidence (DRY — reads
    // `TokenRecord.confidence`, never recomputes a second copy). `None` when
    // the parse carried no per-token confidence (LLM/rule rungs).
    let sentence_confidence = {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for i in start..end {
            if let Some(c) = doc.token(i).confidence {
                sum += c;
                count += 1;
            }
        }
        if count == 0 { None } else { Some(sum / count as f64) }
    };

    // The interlingua frame: content-addressed ids of the routing-relevant
    // roles plus the aligned per-token lemma ids (`InterlinguaId(0)` = none).
    let none_id = InterlinguaId::from_u64(0);
    let id_of = |i: usize| doc.token(i).interlingua_lemma_id.unwrap_or(none_id);
    let interlingua = Some(InterlinguaSignal {
        predicate_id: doc.token(root).interlingua_lemma_id,
        subject_id: subject_idx.map(id_of),
        direct_object_id: direct_object_idx.map(id_of),
        indirect_object_id: indirect_object_idx.map(id_of),
        concept_ids,
        token_ids: (start..end).map(id_of).collect(),
        confidence: sentence_confidence,
    });

    RoutingSignal {
        sentence,
        predicate: lemma_of(root),
        subject,
        direct_object,
        indirect_object,
        modifiers,
        qualifiers,
        arguments,
        dependents,
        tokens,
        lemmas,
        pos,
        deps,
        heads,
        interlingua,
    }
}

/// Whether a token's dep label is one of the role labels (hash comparison).
fn dep_in(token: &TokenRecord, labels: &[&str]) -> bool {
    let hash = token.dep;
    labels.iter().any(|l| hash_utf8(l) == hash)
}

/// Whether a token's dep label equals `label` (hash comparison).
fn dep_is(token: &TokenRecord, label: &str) -> bool {
    hash_utf8(label) == token.dep
}

#[cfg(test)]
#[path = "../tests/routing.rs"]
mod tests;
