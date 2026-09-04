//! The deterministic transition parser (ROADMAP §8 — F6, F8).
//!
//! # Honest framing
//!
//! This is a **deterministic, model-free transition parser** whose oracle is a
//! hand-coded heuristic — *not* spaCy's `ArcEager`, whose oracle is a trained
//! transition model. It does **not** claim dependency parity with spaCy. It
//! exists to lift the fallback ladder's output from a flat star parse to a
//! shallow `nsubj`/`dobj`/`iobj`/`compound`/`prep`+`pobj` structure so routing
//! signals are usable even with no LLM (VISION: deterministic before
//! probabilistic; the LLM stays the primary rung, this parser the middle
//! fallback, `RuleAnnotator` the terminal infallible rung).
//!
//! # Head convention (F8)
//!
//! - **Internal** `heads[i]` holds the **absolute** index of `i`'s head, `-1`
//!   while unset. The virtual root is represented by the dep label `root`,
//!   **not** by any head value — so the ROOT test is
//!   `labels[i] == hash_utf8("root")`, never `heads[i] == 0` (which would
//!   misread token 0's head).
//! - **Output** converts to spaCy's **relative signed offset** convention on
//!   `to_annotation_set`: non-root `head = abs_head - i`, root `head = 0`.
//!
//! # POS heuristics (F6)
//!
//! [`infer_pos`] derives UPOS from lexeme flags plus a closed English
//! function-word map. PROPN fires on `is_upper()` **only** — never
//! `is_title()`, which matches sentence-initial common nouns. This asymmetry
//! is inherent to lexeme-only POS and is exactly why the LLM rung remains
//! primary; the golden corpus asserts both the negative case (Title Case
//! non-entity → NOT PROPN) and the positive case (ALL-CAPS → PROPN).

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::doc::Doc;
use crate::labels::{DepLabelSet, Upos};
use crate::lemmatizer::Lemmatizer;
use crate::lexeme::LexemeFlags;
use crate::llm::{AnnotationRecord, AnnotationResult, AnnotationSet, AnnotationSource};
use crate::pipeline::{AnnotateError, AnnotationRung};
use crate::sentencizer::Sentencizer;
use crate::strings::StringStore;
use crate::validate::{AnnotationError, AnnotationValidator};
use crate::vocab::Vocab;

// ─────────────────────────────────────────────────────────────────────────
// POS heuristics (§8.2)
// ─────────────────────────────────────────────────────────────────────────

/// The closed English function-word POS map — the only way a lexeme-only
/// heuristic gets DET/ADP/AUX/CCONJ/SCONJ. Honest about its limits: it is a
/// fixed list, not a trained tagger.
fn closed_funcword_pos(text: &str) -> Option<Upos> {
    let w = text.to_ascii_lowercase();
    let det = [
        "the", "a", "an", "this", "that", "these", "those", "every", "each", "some", "any", "no",
        "my", "your", "his", "her", "its", "our", "their",
    ];
    let adp = [
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into", "about", "after",
        "before", "under", "over", "through", "between", "among", "during", "within", "without",
        "against", "across", "behind", "above", "below", "near", "off", "out", "up", "down",
        "toward", "towards", "upon", "along", "around", "beside", "beyond",
    ];
    let aux = [
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have",
        "has", "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "ought", "get", "got",
    ];
    let cconj = ["and", "or", "but", "nor", "yet", "so"];
    let sconj = [
        "if", "because", "although", "while", "when", "since", "unless", "whereas", "though",
        "until", "once",
    ];
    let pron = [
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them", "us", "my",
        "mine", "your", "yours", "his", "hers", "ours", "theirs", "who", "whom", "what", "which",
    ];
    if det.contains(&w.as_str()) {
        Some(Upos::Det)
    } else if adp.contains(&w.as_str()) {
        Some(Upos::Adp)
    } else if aux.contains(&w.as_str()) {
        Some(Upos::Aux)
    } else if cconj.contains(&w.as_str()) {
        Some(Upos::Cconj)
    } else if sconj.contains(&w.as_str()) {
        Some(Upos::Sconj)
    } else if pron.contains(&w.as_str()) {
        Some(Upos::Pron)
    } else {
        None
    }
}

/// A closed set of common English verbs so the heuristic parser can detect
/// the predicate it needs for `nsubj`/`dobj` extraction. Verbs are an open
/// class, so this list is deliberately finite and honest: verbs outside it
/// fall through to NOUN (the documented false-negative class that keeps the
/// LLM rung primary, §8.1).
fn is_closed_verb(text: &str) -> bool {
    matches!(
        text.to_ascii_lowercase().as_str(),
        "be" | "am" | "are" | "is" | "was" | "were" | "been" | "being" | "have" | "has"
            | "had" | "do" | "does" | "did" | "go" | "goes" | "went" | "gone" | "make"
            | "makes" | "made" | "take" | "takes" | "took" | "taken" | "see" | "sees" | "saw"
            | "seen" | "come" | "comes" | "came" | "get" | "gets" | "got" | "give" | "gives"
            | "gave" | "given" | "use" | "uses" | "used" | "find" | "finds" | "found" | "want"
            | "wants" | "wanted" | "look" | "looks" | "looked" | "put" | "puts" | "run" | "runs"
            | "ran" | "sat" | "sits" | "sit" | "launch" | "launches" | "launched" | "show"
            | "shows" | "showed" | "shown" | "display" | "displays" | "displayed" | "report"
            | "reports" | "reported" | "buy" | "buys" | "bought" | "sell" | "sells" | "sold"
            | "read" | "reads" | "write" | "writes" | "wrote" | "written" | "call" | "calls"
            | "called" | "need" | "needs" | "needed" | "know" | "knows" | "knew" | "known"
            | "think" | "thinks" | "thought" | "say" | "says" | "said" | "tell" | "tells"
            | "told" | "ask" | "asks" | "asked" | "work" | "works" | "worked" | "play" | "plays"
            | "played" | "move" | "moves" | "moved" | "live" | "lives" | "lived" | "believe"
            | "believes" | "believed" | "hold" | "holds" | "held" | "bring" | "brings" | "brought"
            | "happen" | "happens" | "happened" | "bark" | "barks" | "barked" | "eat" | "eats"
            | "ate" | "eaten" | "drink" | "drinks" | "drank" | "drunk" | "walk" | "walks"
            | "walked" | "create" | "creates" | "created" | "build" | "builds" | "built"
    )
}

/// Hosts that govern a bare infinitive: do-support and the modals, plus the
/// `n't`-split stubs the tokenizer emits (`wo`/`n't`, `ca`/`n't`). Checked
/// case-insensitively without allocating (`eq_ignore_ascii_case`).
fn is_bare_infinitive_host(text: &str) -> bool {
    const HOSTS: &[&str] = &[
        "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "ca", "wo",
    ];
    HOSTS.iter().any(|h| text.eq_ignore_ascii_case(h))
}

/// Whether `text` is a negator hosted by an auxiliary (`n't`, `not`).
fn is_aux_negator(text: &str) -> bool {
    text.eq_ignore_ascii_case("n't") || text.eq_ignore_ascii_case("not")
}

/// Contextual verb detection for the bare infinitive after a do-modal
/// auxiliary (`Do help them`, `Don't help them`, `She won't answer calls`).
/// The tokenizer splits `n't` off its host, so the infinitive — outside the
/// closed verb list — falls through to NOUN and the clause loses its root.
///
/// Upgrades an alpha-fallback NOUN to VERB only when the previous token is a
/// do-modal host, or a negator (`n't`/`not`) hosted by one. Clitic forms
/// (`'s`, `'re`, `'ll`, …) are deliberately excluded: possessive `'s`
/// (`Bell's theorem`) must not trigger. Lexical verbs (`need help`), nouns
/// (`go today`), and determiners never match, so true nominals are untouched.
fn refine_pos_bare_infinitive(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let prev = texts[i - 1].as_str();
        let governed = if is_bare_infinitive_host(prev) {
            true
        } else if is_aux_negator(prev) && i >= 2 && is_bare_infinitive_host(texts[i - 2].as_str()) {
            true
        } else {
            false
        };
        if governed {
            pos[i] = Upos::Verb;
        }
    }
}

/// Contracted-be disambiguation (`It's raining`, `You're late` vs. `Bell's
/// theorem`). Runs after the lexeme-only tags so it can read neighbor POS.
/// Two guarded upgrades, in one ascending pass so the participle rule sees
/// the freshly classified `'s`:
///
/// - `'s` → AUX only when its host is a pronoun (`It`/`You`/…). After a noun
///   it is the possessive case marker (UD: PART/case) and is left untouched.
/// - an `-ing` word → VERB only when its host is an aux-classified be-clitic
///   (`'s`/`'re`/`'m`) — the progressive participle. Full be-forms (`were
///   surprising`) are excluded: participial adjectives after full be belong
///   to copular handling, and firing there would mistag them.
fn refine_pos_contracted_be(texts: &[String], pos: &mut [Upos]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::X {
            continue;
        }
        if texts[i].eq_ignore_ascii_case("'s") && i >= 1 && pos[i - 1] == Upos::Pron {
            pos[i] = Upos::Aux;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let prev = texts[i - 1].as_str();
        let clitic_aux = pos[i - 1] == Upos::Aux
            && (prev.eq_ignore_ascii_case("'s")
                || prev.eq_ignore_ascii_case("'re")
                || prev.eq_ignore_ascii_case("'m"));
        if clitic_aux && texts[i].len() > 4 && texts[i].to_ascii_lowercase().ends_with("ing") {
            pos[i] = Upos::Verb;
        }
    }
}

/// Derive UPOS from lexeme flags + the closed function-word map. PROPN fires
/// on `is_upper()` **only** (never `is_title()`, which matches sentence-initial
/// common nouns); Title-Case proper nouns (Google, Paris, Tuesday) fall
/// through to `NOUN` — the documented false-negative class (§8.2).
#[must_use]
pub fn infer_pos(flags: LexemeFlags, text: &str) -> Upos {
    if flags.is_punct() {
        return Upos::Punct;
    }
    if flags.is_digit() {
        return Upos::Num;
    }
    if flags.is_space() {
        return Upos::Space;
    }
    // Closed function-word map first, so "the"/"is"/"of" resolve before the
    // alpha checks (and stay DET/AUX/ADP even when title-cased at sentence
    // start).
    if let Some(pos) = closed_funcword_pos(text) {
        return pos;
    }
    // Contraction splinters the tokenizer emits: `n't` is a particle (UD:
    // PART), and the `n't`-split stubs `wo`/`ca` plus the unambiguous
    // are-clitic `'re` are auxiliaries. Possessive/clitic `'s` is genuinely
    // ambiguous (It's vs. Bell's) and is resolved contextually in
    // `refine_pos_contracted_be`, never here.
    if text.eq_ignore_ascii_case("n't") {
        return Upos::Part;
    }
    if text.eq_ignore_ascii_case("wo")
        || text.eq_ignore_ascii_case("ca")
        || text.eq_ignore_ascii_case("'re")
    {
        return Upos::Aux;
    }
    // A closed set of common verbs gives the parser a predicate to govern
    // nsubj/dobj around. Verbs outside the list are an honest NOUN false
    // negative (open class; the LLM rung is the primary POS source, §8.1).
    if is_closed_verb(text) {
        return Upos::Verb;
    }
    // is_upper() ONLY — never is_title(). Placed before is_alpha so an
    // all-caps token carrying a digit (HTML5) still counts as PROPN, exactly
    // as spaCy's IS_UPPER flag does.
    if flags.is_upper() {
        return Upos::Propn;
    }
    if flags.is_alpha() {
        return Upos::Noun;
    }
    Upos::X
}

// ─────────────────────────────────────────────────────────────────────────
// The transition system (§8.3)
// ─────────────────────────────────────────────────────────────────────────

/// A parser action: a move plus (for LEFT/RIGHT) the dependency label hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArcEagerAction {
    pub move_type: ArcEagerMove,
    pub label: u64,
}

/// The five ArcEager moves (`arc_eager.pyx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcEagerMove {
    /// Push the buffer head onto the stack.
    Shift,
    /// Pop the stack top (right-complete).
    Reduce,
    /// Stack top → dependent of the buffer head.
    Left,
    /// Buffer head → dependent of the stack top.
    Right,
    /// Mark a sentence boundary and clear the stack.
    Break,
}

/// The parser state (F8): **absolute** head indices internally, `-1` unset.
#[derive(Debug, Clone)]
pub struct ArcEagerState {
    pub stack: Vec<usize>,
    pub buffer: VecDeque<usize>,
    /// Absolute head indices during parsing; -1 = unset (F8). Not relative.
    pub heads: Vec<i32>,
    pub labels: Vec<u64>,
    pub left_children: Vec<Vec<usize>>,
    pub right_children: Vec<Vec<usize>>,
    pub n_tokens: usize,
    pub sent_start: usize,
}

impl ArcEagerState {
    /// A fresh full-doc state with every head unset.
    #[must_use]
    pub fn new(n_tokens: usize, sent_start: usize) -> Self {
        Self {
            stack: Vec::new(),
            buffer: VecDeque::new(),
            heads: vec![-1; n_tokens],
            labels: vec![0; n_tokens],
            left_children: vec![Vec::new(); n_tokens],
            right_children: vec![Vec::new(); n_tokens],
            n_tokens,
            sent_start,
        }
    }

    /// Reset the state for a fresh sentence spanning `[s, e)` with root `r`.
    /// `r` is pre-designated: its head stays `-1` and its label is `root`
    /// (never re-attached by the oracle — exactly one ROOT per sentence).
    pub fn reset_for_sentence(&mut self, s: usize, e: usize, root: usize, root_label: u64) {
        self.stack.clear();
        self.buffer.clear();
        self.buffer.extend(s..e);
        self.sent_start = s;
        self.heads[root] = -1;
        self.labels[root] = root_label;
    }

    /// Terminal: buffer and stack both empty.
    #[must_use]
    pub fn is_final(&self) -> bool {
        self.buffer.is_empty() && self.stack.is_empty()
    }

    /// The stack top index, if any.
    #[must_use]
    pub fn stack_top(&self) -> Option<usize> {
        self.stack.last().copied()
    }

    /// The buffer head index, if any.
    #[must_use]
    pub fn buffer_head(&self) -> Option<usize> {
        self.buffer.front().copied()
    }

    /// The set of candidate actions for the current state, given per-token
    /// POS. Only label variants plausible for the current `(stack_top, buffer
    /// head)` POS pair are offered, so the oracle rarely ties.
    pub fn candidate_actions(&self, pos: &[Upos], labels: &DepLabels) -> Vec<ArcEagerAction> {
        let mut out = Vec::new();
        let Some(b) = self.buffer_head() else {
            // Drain: Reduce until the stack is empty.
            if !self.stack.is_empty() {
                out.push(ArcEagerAction {
                    move_type: ArcEagerMove::Reduce,
                    label: 0,
                });
            }
            return out;
        };
        let s = self.stack_top();

        // SHIFT is always viable when the buffer is non-empty.
        out.push(ArcEagerAction {
            move_type: ArcEagerMove::Shift,
            label: 0,
        });

        let pb = pos[b];
        if pb == Upos::Punct {
            if s.is_some() {
                // Attach the punctuation to the most recent head.
                out.push(ArcEagerAction {
                    move_type: ArcEagerMove::Right,
                    label: labels.punct,
                });
            }
            // BREAK at a sentence boundary (punct is the sentence terminator).
            out.push(ArcEagerAction {
                move_type: ArcEagerMove::Break,
                label: 0,
            });
            return out;
        }

        let Some(s) = s else {
            // Nothing to attach to yet — only SHIFT (already pushed).
            return out;
        };
        let ps = pos[s];

        // The stack top may only be re-headed if it is still unset AND is not
        // the sentence root (the root's head stays -1 forever — attach at most
        // once → acyclic, see module docs / property test 9.10).
        let s_free = self.heads[s] == -1 && self.labels[s] != labels.root;
        let b_free = self.heads[b] == -1 && self.labels[b] != labels.root;

        if s_free {
            match (ps, pb) {
                (Upos::Noun | Upos::Propn | Upos::Pron, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                }
                (Upos::Det, Upos::Noun | Upos::Propn) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.det));
                }
                (Upos::Adj, Upos::Noun | Upos::Propn) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.amod));
                }
                (Upos::Noun | Upos::Propn, Upos::Noun | Upos::Propn) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.compound));
                }
                (Upos::Aux, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.aux));
                }
                (Upos::Adv, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.advmod));
                }
                (Upos::Cconj, _) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.cc));
                }
                // The negator depends on the verb it negates (Don't help:
                // n't → neg → help). Without this arm the PART splinter sits
                // on the stack and every pre-verbal token falls to repair-dep.
                (Upos::Part, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.neg));
                }
                _ => {}
            }
        }

        if b_free {
            match (ps, pb) {
                (Upos::Verb, Upos::Noun | Upos::Propn | Upos::Pron) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.dobj));
                    out.push(Self::act(ArcEagerMove::Right, labels.nsubj));
                }
                (Upos::Verb, Upos::Adv) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.advmod));
                }
                (Upos::Verb, Upos::Adj) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.acomp));
                }
                (Upos::Verb, Upos::Aux) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.aux));
                }
                (Upos::Verb | Upos::Noun | Upos::Adj, Upos::Adp) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.prep));
                }
                // The object of a preposition follows it on the buffer and
                // DEPENDS ON the preposition (already on the stack): RIGHT.
                (Upos::Adp, Upos::Noun | Upos::Propn | Upos::Pron) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.pobj));
                }
                _ => {}
            }
        }

        // Reduce is viable when the stack top is right-complete — except an
        // Adp, which must stay to attract its object (Right(prep) leaves the
        // preposition on the stack so the following noun becomes its pobj).
        // Likewise never pop while an aux/PART is still on the buffer: the
        // stack top may be that aux's subject or host (We did n't see), and
        // Reduce outbids Shift (2.0 > 1.0), stranding them for repair-dep.
        if self.right_children[s].is_empty()
            && ps != Upos::Adp
            && !matches!(pb, Upos::Aux | Upos::Part)
        {
            out.push(ArcEagerAction {
                move_type: ArcEagerMove::Reduce,
                label: 0,
            });
        }

        out
    }

    fn act(move_type: ArcEagerMove, label: u64) -> ArcEagerAction {
        ArcEagerAction { move_type, label }
    }

    /// Apply `action`, mutating the head/label/child state (absolute heads,
    /// F8). `label` is the dep hash (the `root` label is set by the caller at
    /// `reset_for_sentence` and never re-attached).
    pub fn apply(&mut self, action: &ArcEagerAction) {
        let b = self.buffer.front().copied();
        let s = self.stack_top();
        match action.move_type {
            ArcEagerMove::Shift => {
                if let Some(b) = b {
                    self.stack.push(b);
                    self.buffer.pop_front();
                }
            }
            ArcEagerMove::Reduce => {
                if self.stack_top().is_some() {
                    self.stack.pop();
                }
            }
            ArcEagerMove::Left => {
                if let (Some(s), Some(b)) = (s, b) {
                    if self.heads[s] == -1 {
                        self.heads[s] = b as i32;
                        self.labels[s] = action.label;
                        self.left_children[b].push(s);
                    }
                    self.stack.pop();
                }
            }
            ArcEagerMove::Right => {
                if let (Some(s), Some(b)) = (s, b) {
                    if self.heads[b] == -1 {
                        self.heads[b] = s as i32;
                        self.labels[b] = action.label;
                        self.right_children[s].push(b);
                    }
                    self.stack.push(b);
                    self.buffer.pop_front();
                }
            }
            ArcEagerMove::Break => {
                // Close the current sentence: clear the stack and consume the
                // boundary token so the move always progresses (a Break on a
                // lone leading punctuation would otherwise loop forever).
                self.stack.clear();
                if let Some(b) = b {
                    self.buffer.pop_front();
                    self.sent_start = b;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The heuristic oracle (§8.5)
// ─────────────────────────────────────────────────────────────────────────

/// The deterministic heuristic oracle. Scores each candidate action for the
/// current `(stack_top, buffer head)` POS pair; `best_with_margin` picks the
/// winner and reports how much it beat the runner-up by (margin 0 on ties —
/// the load-bearing input to [`ParseConfidence`], §8.5/§9.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicOracle;

impl DeterministicOracle {
    /// A confidence score in `(0, 100]` for a candidate action. Default 1.0
    /// for SHIFT/Reduce (routine bookkeeping); the label-specific heuristics
    /// reward the structures routing actually consumes.
    pub fn score(
        &self,
        state: &ArcEagerState,
        action: &ArcEagerAction,
        pos: &[Upos],
        labels: &DepLabels,
    ) -> f64 {
        let b = state.buffer_head();
        let s = state.stack_top();
        match action.move_type {
            ArcEagerMove::Shift => 1.0,
            ArcEagerMove::Reduce => 2.0,
            ArcEagerMove::Break => 5.0,
            ArcEagerMove::Left => {
                let (Some(s), Some(b)) = (s, b) else {
                    return -1000.0;
                };
                match action.label {
                    l if l == labels.nsubj && pos[b] == Upos::Verb => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn | Upos::Pron) {
                            100.0
                        } else {
                            10.0
                        }
                    }
                    l if l == labels.det && matches!(pos[b], Upos::Noun | Upos::Propn) => {
                        if pos[s] == Upos::Det {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.compound => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn)
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            60.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.amod => {
                        if pos[s] == Upos::Adj && matches!(pos[b], Upos::Noun | Upos::Propn) {
                            85.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.aux => {
                        if pos[s] == Upos::Aux && pos[b] == Upos::Verb {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.advmod => {
                        if pos[s] == Upos::Adv && pos[b] == Upos::Verb {
                            80.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.cc => {
                        if pos[s] == Upos::Cconj {
                            55.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.neg => {
                        if pos[s] == Upos::Part && pos[b] == Upos::Verb {
                            95.0
                        } else {
                            5.0
                        }
                    }
                    _ => 1.0,
                }
            }
            ArcEagerMove::Right => {
                let (Some(s), Some(b)) = (s, b) else {
                    return -1000.0;
                };
                match action.label {
                    l if l == labels.dobj && pos[s] == Upos::Verb => {
                        if matches!(pos[b], Upos::Noun | Upos::Propn | Upos::Pron) {
                            100.0
                        } else {
                            10.0
                        }
                    }
                    l if l == labels.nsubj && pos[s] == Upos::Verb => {
                        if matches!(pos[b], Upos::Noun | Upos::Propn | Upos::Pron) {
                            60.0
                        } else {
                            10.0
                        }
                    }
                    l if l == labels.prep => {
                        if pos[b] == Upos::Adp
                            && matches!(pos[s], Upos::Verb | Upos::Noun | Upos::Adj)
                        {
                            95.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.pobj => {
                        if pos[s] == Upos::Adp
                            && matches!(pos[b], Upos::Noun | Upos::Propn | Upos::Pron)
                        {
                            95.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.punct => 98.0,
                    l if l == labels.advmod => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Adv {
                            75.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.acomp => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Adj {
                            80.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.aux => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Aux {
                            85.0
                        } else {
                            5.0
                        }
                    }
                    _ => 1.0,
                }
            }
        }
    }

    /// Pick the best-scoring action and its margin over the runner-up. Margin
    /// 0.0 iff the best is tied (≥2 actions share the max) or the candidate
    /// set is empty. A single candidate is scored as fully certain (margin 1).
    pub fn best_with_margin(
        &self,
        state: &ArcEagerState,
        actions: &[ArcEagerAction],
        pos: &[Upos],
        labels: &DepLabels,
    ) -> Option<(ArcEagerAction, f64)> {
        if actions.is_empty() {
            return None;
        }
        let mut scored: Vec<(ArcEagerAction, f64)> = actions
            .iter()
            .map(|a| (*a, self.score(state, a, pos, labels)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = scored[0];
        if scored.len() == 1 {
            return Some((best.0, 1.0));
        }
        let second = scored[1];
        let margin = best.1 - second.1;
        Some((best.0, margin))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The dependency label hash set
// ─────────────────────────────────────────────────────────────────────────

/// The dep-label hashes the parser emits. Interned into the vocab's
/// `StringStore` at construction so reverse lookup (hash → string) works for
/// the output records.
#[derive(Debug, Clone)]
pub struct DepLabels {
    pub root: u64,
    pub nsubj: u64,
    pub dobj: u64,
    pub iobj: u64,
    pub prep: u64,
    pub pobj: u64,
    pub compound: u64,
    pub det: u64,
    pub aux: u64,
    pub punct: u64,
    pub amod: u64,
    pub advmod: u64,
    pub cc: u64,
    pub dep: u64,
    pub neg: u64,
    pub acomp: u64,
    pub xcomp: u64,
}

impl DepLabels {
    /// Compute the hashes, interning each label into `strings`.
    #[must_use]
    pub fn new(strings: &StringStore) -> Self {
        let intern = |label: &str| strings.add(label);
        Self {
            root: intern("root"),
            nsubj: intern("nsubj"),
            dobj: intern("dobj"),
            iobj: intern("iobj"),
            prep: intern("prep"),
            pobj: intern("pobj"),
            compound: intern("compound"),
            det: intern("det"),
            aux: intern("aux"),
            punct: intern("punct"),
            amod: intern("amod"),
            advmod: intern("advmod"),
            cc: intern("cc"),
            dep: intern("dep"),
            neg: intern("neg"),
            acomp: intern("acomp"),
            xcomp: intern("xcomp"),
        }
    }

    /// The label string for a hash, via the shared store.
    #[must_use]
    pub fn to_string(&self, label: u64, strings: &StringStore) -> String {
        strings
            .get(label)
            .map_or_else(|| "dep".to_string(), |s| s.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// ParseConfidence (§9.3)
// ─────────────────────────────────────────────────────────────────────────

/// Margin-aware parse confidence. `overall = max(0, mean(token_scores) -
/// 0.05 * oracle_tie_count)`; ROOT detection uses the `root` dep label (F8),
/// never `heads[i] == 0`; `role_coverage` is the fraction of `{nsubj, dobj}`
/// argument slots filled.
///
/// Honest about what it measures: oracle tie-count, role coverage, and
/// PROPN-not-in-registry. It does **not** reveal *which* actions were tied —
/// margin tells you *that* the oracle was uncertain, not *why*. Sufficient
/// for routing (F7).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParseConfidence {
    pub overall: f64,
    pub token_scores: Vec<f64>,
    pub role_coverage: f64,
    pub oracle_tie_count: usize,
    /// The per-oracle-step margins (ROADMAP M3): a near-zero margin is an
    /// attachment near-tie — the frame extractor's
    /// `AmbiguityKind::AttachmentNearTie` signal. Empty for rungs that carry
    /// no margins.
    pub oracle_margins: Vec<f64>,
    /// Separate YaGO type-plausibility (Alt C) — never blended into `oracle_margins`, per roadmap E7.
    pub semantic_plausibility: Option<f64>,
}

impl ParseConfidence {
    /// Compute confidence from the per-step oracle margins and per-token
    /// scores. Ties (`margin == 0`) reduce `overall` by 5% each.
    #[must_use]
    pub fn compute(token_scores: &[f64], margins: &[f64], role_coverage: f64) -> Self {
        let oracle_tie_count = margins.iter().filter(|&&m| m == 0.0).count();
        let mean = if token_scores.is_empty() {
            0.0
        } else {
            token_scores.iter().sum::<f64>() / token_scores.len() as f64
        };
        let overall = (mean - 0.05 * oracle_tie_count as f64).max(0.0);
        Self {
            overall,
            token_scores: token_scores.to_vec(),
            role_coverage,
            oracle_tie_count,
            oracle_margins: margins.to_vec(),
            semantic_plausibility: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The annotator (§8.6)
// ─────────────────────────────────────────────────────────────────────────

/// The deterministic transition annotator over a tokenized doc. Always returns
/// a set (even a flat `dep`-star fallback for degenerate input); confidence
/// carries the oracle's uncertainty. `Ok(None)` is reserved for genuine
/// structural failure (empty doc) — the only case that falls through to the
/// rule rung (F7).
pub struct ArcEagerAnnotator {
    vocab: Arc<Vocab>,
    dep_labels: DepLabelSet,
    labels: DepLabels,
    sentencizer: Sentencizer,
    lemmatizer: Lemmatizer,
}

impl std::fmt::Debug for ArcEagerAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcEagerAnnotator")
            .field("dep_labels", &self.dep_labels)
            .field("sentencizer", &self.sentencizer)
            .finish_non_exhaustive()
    }
}

impl ArcEagerAnnotator {
    /// The English-default annotator: shared `vocab`, UD label set, default
    /// sentencizer and rule lemmatizer.
    #[must_use]
    pub fn en_default(vocab: Arc<Vocab>) -> Self {
        Self::new(vocab, DepLabelSet::ud_default(), Sentencizer::new(), Lemmatizer::english_rule())
    }

    /// An annotator over `vocab`, the accepted `dep_labels`, `sentencizer`
    /// (sentence boundaries) and `lemmatizer` (output lemmas).
    #[must_use]
    pub fn new(
        vocab: Arc<Vocab>,
        dep_labels: DepLabelSet,
        sentencizer: Sentencizer,
        lemmatizer: Lemmatizer,
    ) -> Self {
        let labels = DepLabels::new(vocab.strings());
        Self {
            vocab,
            dep_labels,
            labels,
            sentencizer,
            lemmatizer,
        }
    }

    /// The vocab backing the annotator.
    #[must_use]
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// The accepted dependency-label set.
    #[must_use]
    pub fn dep_labels(&self) -> &DepLabelSet {
        &self.dep_labels
    }

    /// The sentencizer providing sentence boundaries.
    #[must_use]
    pub fn sentencizer(&self) -> &Sentencizer {
        &self.sentencizer
    }

    /// The rule lemmatizer.
    #[must_use]
    pub fn lemmatizer(&self) -> &Lemmatizer {
        &self.lemmatizer
    }

    /// Deterministic parse with confidence. Always returns a set (even a flat
    /// `dep`-star fallback for degenerate input); confidence carries the
    /// oracle's uncertainty. `Err(EmptyDocument)` on an empty doc.
    pub fn annotate_with_confidence(
        &self,
        doc: &Doc,
    ) -> Result<(AnnotationResult, ParseConfidence), AnnotationError> {
        if doc.is_empty() {
            return Err(AnnotationError::EmptyDocument);
        }

        let texts: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
        let mut pos: Vec<Upos> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| infer_pos(doc.token(i).lexeme.flags, text))
            .collect();
        // Contextual pass over the lexeme-only tags: bare infinitive after a
        // do-modal host. Runs before root-picking so the oracle sees verbs.
        refine_pos_bare_infinitive(&texts, &mut pos);
        // Contracted-be pass: pronoun-hosted 's → AUX, then clitic-hosted
        // -ing participle → VERB. Sequenced after the infinitive pass; the
        // two govern disjoint contexts (aux-host vs. clitic-host).
        refine_pos_contracted_be(&texts, &mut pos);

        // Sentence boundaries from the sentencizer.
        let starts = self.sentencizer.predict(doc);
        let sentences = partition_sentences(&starts, doc.len());

        let mut state = ArcEagerState::new(doc.len(), 0);
        let oracle = DeterministicOracle;
        let mut margins = Vec::new();
        let mut sentence_roots = Vec::new();

        for (s, e) in sentences {
            let root = pick_root(&pos, s, e);
            sentence_roots.push(root);
            state.reset_for_sentence(s, e, root, self.labels.root);
            while !state.is_final() {
                let actions = state.candidate_actions(&pos, &self.labels);
                let Some((best, margin)) = oracle.best_with_margin(&state, &actions, &pos, &self.labels)
                else {
                    break;
                };
                margins.push(margin);
                state.apply(&best);
            }
            // Repair: any non-root token left unheaded attaches to the root
            // with `dep` (the honest flat fallback) — guarantees connectivity
            // and exactly one ROOT per sentence.
            for i in s..e {
                if state.heads[i] == -1 && i != root {
                    state.heads[i] = root as i32;
                    state.labels[i] = self.labels.dep;
                }
            }
        }

        // Build the output records (relative heads; root = 0, else abs - i).
        let strings = Arc::clone(self.vocab.strings());
        let mut records = Vec::with_capacity(doc.len());
        let mut token_scores = Vec::with_capacity(doc.len());
        for (i, &p) in pos.iter().enumerate() {
            let is_root = state.labels[i] == self.labels.root;
            let head = if is_root { 0 } else { state.heads[i] - i as i32 };
            let text = doc.token_text(i);
            let lemma = self.lemmatizer
                .lemmatize(&text, p, 0)
                .first()
                .cloned()
                .unwrap_or_else(|| text.to_ascii_lowercase());
            records.push(AnnotationRecord {
                text,
                pos: p.to_string(),
                tag: String::new(),
                dep: self.labels.to_string(state.labels[i], &strings),
                head,
                lemma,
                morph: String::new(),
                ent_iob: String::new(),
                ent_type: String::new(),
            });
            token_scores.push(token_score(state.labels[i], &self.labels));
        }

        let role_coverage = self.role_coverage(&state, &pos);
        let confidence = ParseConfidence::compute(&token_scores, &margins, role_coverage);
        let mut result = AnnotationResult::new(AnnotationSet(records), AnnotationSource::ArcEager)
            .with_confidence(Some(token_scores), Some(confidence.clone()));
        // Thread the oracle margins for the frame stage's attachment near-tie
        // signal (ROADMAP M3). `margins` is still owned here — `compute` took
        // it by reference.
        result.oracle_margins = Some(margins);
        Ok((result, confidence))
    }

    /// Fraction of `{nsubj, dobj}` argument slots filled, per sentence with a
    /// verb. A transitive sentence with both roles → 1.0; intransitive with
    /// just a subject → 1.0; a verb with no arguments → 0.0.
    fn role_coverage(&self, state: &ArcEagerState, pos: &[Upos]) -> f64 {
        let roots: Vec<usize> = (0..state.n_tokens)
            .filter(|&i| state.labels[i] == self.labels.root)
            .collect();
        if roots.is_empty() {
            return 0.0;
        }
        let mut filled = 0.0;
        let mut expected = 0.0;
        for &r in &roots {
            let mut has_nsubj = false;
            let mut has_dobj = false;
            let mut sentence_pos = Vec::new();
            for (i, &p) in pos.iter().enumerate().take(state.n_tokens) {
                // Reach the root via absolute heads (children of r, direct).
                if state.heads[i] == r as i32 {
                    if state.labels[i] == self.labels.nsubj {
                        has_nsubj = true;
                    }
                    if state.labels[i] == self.labels.dobj {
                        has_dobj = true;
                    }
                }
                if i == r {
                    sentence_pos.push(p);
                }
            }
            if has_nsubj {
                filled += 1.0;
            }
            expected += 1.0; // nsubj slot
            if matches!(sentence_pos.first(), Some(Upos::Verb)) {
                expected += 1.0; // dobj slot for a verbal sentence
                if has_dobj {
                    filled += 1.0;
                }
            }
        }
        if expected == 0.0 {
            1.0
        } else {
            let coverage: f64 = filled / expected;
            coverage.clamp(0.0, 1.0)
        }
    }
}

/// The per-token confidence for a label: high for the routing-relevant
/// structures, lower for heuristic-only or repair labels.
fn token_score(label: u64, labels: &DepLabels) -> f64 {
    if matches!(
        label,
        l if l == labels.root
            || l == labels.nsubj
            || l == labels.dobj
            || l == labels.prep
            || l == labels.pobj
            || l == labels.det
            || l == labels.compound
            || l == labels.aux
            || l == labels.punct
    ) {
        1.0
    } else if matches!(
        label,
        l if l == labels.amod || l == labels.advmod || l == labels.acomp || l == labels.cc
    ) {
        0.7
    } else {
        0.5
    }
}

/// Partition `[0, len)` into `(start, end)` sentence ranges from the
/// sentencizer's `start` boolean vector.
fn partition_sentences(starts: &[bool], len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    for (i, &is_start) in starts.iter().enumerate() {
        if is_start && i != 0 {
            out.push((cur, i));
            cur = i;
        }
    }
    if cur < len {
        out.push((cur, len));
    }
    if out.is_empty() && len > 0 {
        out.push((0, len));
    }
    out
}

/// The sentence root: the leftmost verb; else the leftmost NOUN/PROPN/PRON;
/// else the first token (the minimal star fallback for degenerate sentences).
fn pick_root(pos: &[Upos], s: usize, e: usize) -> usize {
    (s..e)
        .find(|&i| pos[i] == Upos::Verb)
        .or_else(|| (s..e).find(|&i| matches!(pos[i], Upos::Noun | Upos::Propn | Upos::Pron)))
        .unwrap_or(s)
}

// ─────────────────────────────────────────────────────────────────────────
// The rung (§8.6, F7)
// ─────────────────────────────────────────────────────────────────────────

/// The ladder rung wrapping the deterministic parser. **Always returns** its
/// parse — confidence rides in `AnnotationResult` and gates *downstream*
/// routing, never rung fallthrough (F7). `Ok(None)` is reserved for genuine
/// structural failure (empty doc) — the only case that falls through to
/// `RuleRung`.
pub struct ArcEagerRung {
    annotator: Arc<ArcEagerAnnotator>,
    validator: Arc<AnnotationValidator>,
}

impl std::fmt::Debug for ArcEagerRung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcEagerRung").finish_non_exhaustive()
    }
}

impl ArcEagerRung {
    /// A rung parsing with `annotator` and gating with `validator`.
    #[must_use]
    pub fn new(annotator: Arc<ArcEagerAnnotator>, validator: Arc<AnnotationValidator>) -> Self {
        Self {
            annotator,
            validator,
        }
    }
}

impl AnnotationRung for ArcEagerRung {
    fn run<'a>(
        self: Box<Self>,
        doc: &'a Doc,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a>>
    {
        Box::pin(async move {
            let (result, _parse_conf) = match self.annotator.annotate_with_confidence(doc) {
                Ok(out) => out,
                // Empty doc — genuine structural failure, falls to RuleRung.
                Err(AnnotationError::EmptyDocument) => return Ok(None),
                Err(e) => return Err(AnnotateError::Rejected(e)),
            };
            self.validator
                .validate(doc, result.records())
                .map_err(AnnotateError::Rejected)?;
            Ok(Some(result))
        })
    }
}

#[cfg(test)]
#[path = "../tests/arc_eager.rs"]
mod tests;
