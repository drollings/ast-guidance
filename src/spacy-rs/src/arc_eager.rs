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
        "everybody",
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

/// Determiner-led nominal colliding with the closed verb list (`the
/// report`, `a saw`). The closed verb check in [`infer_pos`] fires before
/// any nominal guard, so a determiner-led noun that shares a form with the
/// list tags VERB, steals root, and strands the true predicate. A
/// determiner never governs a finite verb, so DET + closed-verb-form reads
/// nominal. Downgrades VERB to NOUN only directly after a DET, sequenced
/// FIRST among the refines so only [`infer_pos`] verbs (the closed list)
/// are candidates — every VERB-upgrade pass below reads the corrected
/// tags, and no upgrade output is ever touched. Guards: bare verbs (`Dogs
/// bark`, `Close your books` — no determiner) never match. Known
/// boundary: determiner-shaped relativizers (`the book that reports…`)
/// have no corpus instance — `that`-headed verb disambiguation is its own
/// rule.
fn refine_pos_det_closed_verb(_texts: &[String], pos: &mut [Upos]) {
    for i in 1..pos.len() {
        if pos[i] != Upos::Verb || pos[i - 1] != Upos::Det {
            continue;
        }
        pos[i] = Upos::Noun;
    }
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

/// Sentence-initial directive verb (`Close your books`, `Solve this
/// equation`, `Search the web`). Directives outside the closed verb list
/// fall through to NOUN and the clause loses its root. Upgrades an
/// alpha-fallback NOUN to VERB only at a sentence start followed by a
/// DET+NOUN object phrase. Strictly scoped: a NOUN second (`Anna finished`,
/// `Translate hello`, `Explain Bell's`) or a conjunction (`Dogs and cats`),
/// pronoun (`Help them`), or verb (`Dogs bark`) second never fires, so
/// subjects and bare/proper objects are untouched.
fn refine_pos_directive_initial(texts: &[String], pos: &mut [Upos]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let at_start = i == 0
            || matches!(
                texts[i - 1].as_str(),
                "." | "!" | "?" | ";" | ":" | "—" | "--"
            );
        if !at_start || i + 2 >= texts.len() {
            continue;
        }
        if pos[i + 1] == Upos::Det && pos[i + 2] == Upos::Noun {
            pos[i] = Upos::Verb;
        }
    }
}

/// Whether `text` is a nominative pronoun surface (finite-clause subject).
/// Object/possessive forms (`me`, `them`, `mine`) are excluded even though
/// the closed map tags them PRON.
fn is_nominative_subject(text: &str) -> bool {
    const NOMINATIVE: &[&str] = &["i", "you", "he", "she", "it", "we", "they", "who"];
    NOMINATIVE.iter().any(|h| text.eq_ignore_ascii_case(h))
}

/// Finite verb after a nominative pronoun subject (`We stayed`, `it snowed`,
/// `who called`, `you clean`). The closed verb list misses most of these, so
/// matrix and SCONJ-embedded clauses alike lose their predicate — including
/// after dual-class markers (`after she scored`) that lex as ADP, since the
/// rule keys on the subject, never the marker. Upgrades an alpha-fallback
/// NOUN to VERB only after a nominative pronoun. Guards: object pronouns
/// (`Call me later`), determiners (`her keys`), AUX hosts (`are coming`), and
/// noun subjects (`Anna finished`) never match. Known boundary: vocative
/// collectives (`you guys`) would over-fire — no corpus instance; left for
/// later work.
fn refine_pos_pronoun_subject_verb(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Pron {
            continue;
        }
        if is_nominative_subject(texts[i - 1].as_str()) {
            pos[i] = Upos::Verb;
        }
    }
}

/// Sentence-initial comment `As` (`As you know, …`, `As everybody knows,
/// …`). The closed map lexes every `as` as ADP, so comment clauses read as
/// prepositional phrases: the Sconj-keyed mark arm never fires and the
/// subject misattaches as `pobj`. Upgrades a sentence-initial `as` to
/// SCONJ only with a nominal next (the comment subject). Guards: medial
/// `as` (`Paris, as always, …`) never matches — comment position is
/// initial by construction. Same O(n) frame-scan shape.
fn refine_pos_comment_as(texts: &[String], pos: &mut [Upos]) {
    if texts.len() < 2 || pos[0] != Upos::Adp || !texts[0].eq_ignore_ascii_case("as") {
        return;
    }
    if matches!(pos[1], Upos::Noun | Upos::Propn | Upos::Pron) {
        pos[0] = Upos::Sconj;
    }
}

/// Whether `text` is a nominal relativizer with corpus evidence (`who`,
/// `that`, `where`). Only forms attested in the bench; `which`/`whom` have
/// no instance and stay out until one appears.
fn is_relative_marker(text: &str) -> bool {
    text.eq_ignore_ascii_case("who")
        || text.eq_ignore_ascii_case("that")
        || text.eq_ignore_ascii_case("where")
}

/// Shifted DET+NOUN predicate after a clause boundary (`Truthfully, the
/// plan failed`, `Sadly, the trip ended early`). The initial-noun frame
/// only sees positions 0-2, shielded here by leading adverbial material —
/// but the same DET+NOUN+predicate shape recurs past the boundary comma.
/// Upgrades the third NOUN to VERB only for a clause-final predicate
/// (only ADV/PUNCT may follow: `failed.` / `ended early.`), which excludes
/// appositive nominals with more clause to come (`an old brick, still…`).
/// Guards (mirroring the initial-noun rule): lowercase target, and the
/// comma must sit directly before the determiner (comma-free triples like
/// `the sales report` never match). Same O(n) frame-scan shape.
fn refine_pos_shifted_det_noun_verb(texts: &[String], pos: &mut [Upos]) {
    for i in 3..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        if texts[i - 3] != "," || pos[i - 2] != Upos::Det {
            continue;
        }
        if !matches!(pos[i - 1], Upos::Noun | Upos::Propn) {
            continue;
        }
        if !texts[i].chars().next().is_some_and(|c| c.is_lowercase()) {
            continue;
        }
        if pos[i + 1..]
            .iter()
            .all(|p| matches!(p, Upos::Punct | Upos::Adv))
        {
            pos[i] = Upos::Verb;
        }
    }
}

/// Matrix finite verb after a relative clause (`The man who called left`,
/// `The book that I bought vanished`). The closed verb list misses the
/// matrix predicate, so it falls through to NOUN and the relcl verb steals
/// root. Upgrades a sentence-final alpha-fallback NOUN to VERB only when it
/// directly follows the FIRST verb after a nominal-headed who/that/where
/// (the relcl predicate — closed-list or pronoun-rule-upgraded). The
/// first-verb scope is load-bearing: a second verb after the marker is the
/// matrix verb itself, so a nominal following it is its adverbial/object
/// (`work hard`: hard stays NOUN, its dobj→work head was already right).
/// Guards: complementizer `that` headed by a verb (`I know that they play
/// soccer`) never matches; title-case finals (`called Anna`, the §8.2
/// proper-noun class) never match; interrogative initials (`Who called
/// earlier?`) have no nominal head. Known boundary: medial matrix verbs
/// with a trailing predicate nominal/adverbial (`stands empty`, `improve
/// fast`) and DET-headed relcl verbs (`that cried`) are out of scope — they
/// need verb-capability or relcl-subject knowledge, not this frame.
fn refine_pos_relative_matrix_verb(texts: &[String], pos: &mut [Upos]) {
    // Nominal-headed relativizer positions (the relative-frame gate).
    let markers: Vec<usize> = (1..texts.len())
        .filter(|&r| {
            is_relative_marker(texts[r].as_str()) && matches!(pos[r - 1], Upos::Noun | Upos::Propn)
        })
        .collect();
    if markers.is_empty() {
        return;
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Verb {
            continue;
        }
        // A finite matrix verb is never capitalized.
        if !texts[i].chars().next().is_some_and(|c| c.is_lowercase()) {
            continue;
        }
        // Sentence-final: only trailing punctuation may follow.
        if !texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        }) {
            continue;
        }
        // The host verb must be the first verb after some marker: no other
        // VERB may stand between that marker and the host.
        let closes_relcl = markers.iter().any(|&r| {
            r < i - 1 && (r + 1..i - 1).all(|j| pos[j] != Upos::Verb)
        });
        if closes_relcl {
            pos[i] = Upos::Verb;
        }
    }
}

/// Comma-framed `-ly` adverbial (`Sadly, …`, `…, frankly, …`). The tagger
/// has no ADV path, so sentence/manner adverbials fall through to NOUN —
/// and sentence-initial ones even steal root. Upgrades an alpha-fallback
/// NOUN to ADV only for an `-ly` form (allocation-free suffix check, same
/// shape as the be-predicate `-ing` check) immediately followed by a comma
/// and hosted by a clause edge (sentence start or a preceding comma).
/// Guards: temporal `-ly` nominals after a preposition (`In July, …`) and
/// determiner-headed ones (`The family …`) never match. Known boundary:
/// non-`ly` adverbials (`early`, `still`, `always`, `daily`, `hard`) need an
/// adverb lexicon, not a suffix frame.
fn refine_pos_comma_adverbial(texts: &[String], pos: &mut [Upos]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() <= 4
            || !word
                .get(word.len() - 2..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ly"))
        {
            continue;
        }
        if !texts.get(i + 1).is_some_and(|next| next == ",") {
            continue;
        }
        if i > 0 && texts[i - 1] != "," {
            continue;
        }
        pos[i] = Upos::Adv;
    }
}

/// Relativizer `that` and its clause verb (`The baby that cried slept`,
/// `The dog that barked ran off`). The closed map lexes every `that` as
/// DET, so a nominal-headed relativizer never reads as PRON — and its
/// clause verb (`cried`) strands as NOUN, unseen by every subject rule. Two
/// guarded upgrades in one frame pass, same shape as the contracted-be
/// rule: nominal-headed `that` (DET-tagged only) → PRON, then an
/// alpha-fallback NOUN directly after that PRON-`that` → VERB (the clause
/// predicate, mirroring the pronoun-subject rule for a marker the closed
/// list never pronounces). Guards: verb-headed `that` (complementizer: `I
/// know that they play`) and headless `that` (demonstrative: `That book`)
/// never match; the verb upgrade keys on the word `that`, so bare DET+NOUN
/// pairs (`the sales report`) and other pronouns (`he left`) never match.
/// Known boundary: `fact`-complements (`The fact that he left`) tag `that`
/// PRON where UD reads SCONJ — complementizer disambiguation is its own
/// rule.
fn refine_pos_that_relative(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] == Upos::Det
            && texts[i].eq_ignore_ascii_case("that")
            && matches!(pos[i - 1], Upos::Noun | Upos::Propn)
        {
            pos[i] = Upos::Pron;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Pron {
            continue;
        }
        if texts[i - 1].eq_ignore_ascii_case("that") {
            pos[i] = Upos::Verb;
        }
    }
}

/// Locative relativizer `where` (`The store where we met closed`). The
/// tagger leaves it NOUN, so the Sconj-keyed mark arm — which fires for
/// `when`/`because` — never sees it, and the anchor compounds onto it.
/// Upgrades a NOUN `where` to SCONJ only with a nominal head (the relative
/// anchor). Guards: sentence-initial interrogatives (`Where is my bag?`)
/// and verb-hosted uses never match. Known divergence: refs pin ADP for
/// `where` while the sibling `as`-frame refs pin SCONJ; the rule follows
/// the functional (mark-taking) reading, so UPOS stays divergent either
/// way and only the attachment is claimed. Disjoint from the `that` frame
/// above (different word, different target tag).
fn refine_pos_where_marker(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || !texts[i].eq_ignore_ascii_case("where") {
            continue;
        }
        if matches!(pos[i - 1], Upos::Noun | Upos::Propn) {
            pos[i] = Upos::Sconj;
        }
    }
}

/// Dual-class `after` (`The game ended after she scored` vs `after
/// lunch`). The closed map lexes every `after` as ADP, so a clausal
/// complement reads as a prepositional phrase: the Sconj-keyed mark arm
/// never fires and the subject misattaches as `pobj`. Upgrades an ADP
/// `after` to SCONJ only with a clausal complement ahead — a nominal
/// subject (PRON/NOUN/PROPN, or DET + NOUN/PROPN) directly followed by a
/// VERB. Guards: nominal complements (`after lunch`, `after the meeting`
/// — no verb ahead) never match. Sequenced after the pronoun-subject
/// pass so clause verbs upgraded there (`spoke`, `scored`) are visible
/// as the VERB host. Known boundary: `before` is the same dual class
/// with no corpus instance — left out until one appears.
fn refine_pos_clausal_after(texts: &[String], pos: &mut [Upos]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Adp || !texts[i].eq_ignore_ascii_case("after") {
            continue;
        }
        // Pattern A: nominal subject + verb (`after he spoke`).
        let bare = i + 2 < texts.len()
            && matches!(pos[i + 1], Upos::Pron | Upos::Noun | Upos::Propn)
            && pos[i + 2] == Upos::Verb;
        // Pattern B: determiner-led subject + verb (`after the meeting ended`).
        let det_led = i + 3 < texts.len()
            && pos[i + 1] == Upos::Det
            && matches!(pos[i + 2], Upos::Noun | Upos::Propn)
            && pos[i + 3] == Upos::Verb;
        if bare || det_led {
            pos[i] = Upos::Sconj;
        }
    }
}

/// Closed-class time/manner adverbials in sentence-final position (`The
/// dog barks loudly`, `She reads books daily`, `Work hard, play fair`).
/// The tagger has no adverb lexicon, so these fall through to NOUN — and
/// with the verb directly on the stack the existing Right-advmod arm then
/// misfires as dobj. Upgrades an alpha-fallback NOUN to ADV for a curated
/// set (every form bench-attested with an ADV ref) plus `-ly` forms, in
/// final position or before a comma. Guards: determiner hosts (`a daily`)
/// and conjunction hosts (`red and fast`, coordination frames) never match;
/// `late` additionally excludes AUX hosts (copular predicates: `was late`,
/// `'re late` stay ADJ). Deliberately outside the set: `today` (frozen refs
/// irreconcilable — NOUN/advmod vs ADV), `later`/`earlier` (golden-pinned
/// NOUN), `still` (pre-verbal frame only — predicative `be still` must not
/// match), WH-initials (`Why`/`Where`, interrogative frame), and
/// coordination-`or` slots (`Run daily or quit` — CC-disambiguation is its
/// own rule). Same O(n) frame-scan shape as the other refines.
fn refine_pos_final_adverbial(texts: &[String], pos: &mut [Upos]) {
    const ADVERBS: &[&str] = &[
        "now", "early", "again", "always", "hard", "fast", "well", "fair", "here", "daily", "much",
        "yet", "late",
    ];
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let word = texts[i].as_str();
        let in_set = ADVERBS.iter().any(|a| word.eq_ignore_ascii_case(a));
        let ly = !in_set
            && word.len() > 4
            && word
                .get(word.len() - 2..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ly"));
        if !in_set && !ly {
            continue;
        }
        // Copular `late` is adjectival, never adverbial.
        if word.eq_ignore_ascii_case("late") && i > 0 && pos[i - 1] == Upos::Aux {
            continue;
        }
        if i > 0 && matches!(pos[i - 1], Upos::Det | Upos::Cconj) {
            continue;
        }
        let trailing_punct = texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        });
        // `-ly` forms upgrade only sentence-finally (the comma frame belongs
        // to the comma-adverbial pass, whose clause-edge host guard keeps
        // `July`-class nominals out); set members also fire before a comma
        // (`as always,`, `Work hard,`), but never sentence-initially there.
        let comma_next = !ly && texts.get(i + 1).is_some_and(|next| next == ",") && i > 0;
        if trailing_punct || comma_next {
            pos[i] = Upos::Adv;
        }
    }
}

/// Sensory linking frame (`Dinner smells great`, `The soup tastes salty`,
/// `The movie sounds boring`). Bare-initial sensory verbs fall outside
/// every subject rule, and their predicate complements strand as nominal
/// objects. Two guarded upgrades in one frame pass (contracted-be shape):
/// a NOUN sensory word (taste/sound/smell, base + -s only — corpus
/// evidence; look is closed-listed already, feel/seem unattested) with a
/// nominal following it → VERB, then an alpha-fallback NOUN directly after
/// a sensory VERB → ADJ (the predicate complement, mirroring the
/// be-predicate rule for a host the closed list never pronounces).
/// Guards: determiner-led objects (`Taste the soup`) never match either
/// upgrade — transitivity reads through the determiner. Known boundaries:
/// bare transitives (`smells smoke`) and attributive sensory nouns (`taste
/// buds`) are positionally identical with no bench instance — lexical
/// subcategorization is its own rule.
fn refine_pos_linking_predicate(texts: &[String], pos: &mut [Upos]) {
    const SENSORY: &[&str] = &[
        "taste", "tastes", "sound", "sounds", "smell", "smells",
    ];
    let is_sensory = |w: &str| SENSORY.iter().any(|s| w.eq_ignore_ascii_case(s));
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun || !is_sensory(texts[i].as_str()) {
            continue;
        }
        if texts.len() > i + 1 && matches!(pos[i + 1], Upos::Noun | Upos::Propn) {
            pos[i] = Upos::Verb;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Verb {
            continue;
        }
        if is_sensory(texts[i - 1].as_str()) {
            pos[i] = Upos::Adj;
        }
    }
}

/// Clause-initial verb after a comma boundary (`The test, honestly,
/// scared us`, `Her car, red and fast, won`). Matrix verbs opening after a
/// parenthetical-closing comma fall through to NOUN — no subject rule sees
/// a comma-adjacent predicate — and strand their objects into repair-dep.
/// Upgrades a NOUN directly after a comma to VERB only when an earlier
/// comma directly follows a nominal (the parenthetical opener) — i.e. the
/// target opens a new clause after parenthetical material, not a
/// subordinate clause after its predicate (`If it snows, schools…`, whose
/// comma follows a verb) and not an imperative conjunct (`Study hard,
/// rest…`, no nominal opener). Guards: `-ing` participles (their own
/// clause-role gap — upgrading one would crown it root), CCONJ-next
/// (coordinated predicate adjectives: `red and fast`), and VERB-next
/// (infinitival/complement adjacency: `still works`). Same O(n)
/// frame-scan shape as the other refines.
fn refine_pos_post_comma_verb(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || texts[i - 1] != "," {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() > 4
            && word
                .get(word.len() - 3..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
        {
            continue;
        }
        // Coordinated predicate adjectives (`red and fast`) and
        // infinitival/complement adjacency (`still works`) are not clause
        // openings.
        if let Some(next_pos) = pos.get(i + 1) {
            if matches!(next_pos, Upos::Cconj | Upos::Verb) {
                continue;
            }
        }
        // The comma must close parenthetical material opened right after a
        // nominal — otherwise the target is a subordinate subject
        // (`schools`, `lunch`), an imperative conjunct (`rest`), or a
        // complement (`stay`).
        let opened = (1..i - 1).any(|k| {
            texts[k] == "," && matches!(pos[k - 1], Upos::Noun | Upos::Propn | Upos::Pron)
        });
        if !opened {
            continue;
        }
        pos[i] = Upos::Verb;
    }
}
/// subject position (`The store where …`). Only forms with corpus evidence;
/// `when`/`that`/`who` lex out of NOUN already and need no guard.
fn is_noun_subject_verb_blocker(text: &str) -> bool {
    text.eq_ignore_ascii_case("where")
}

/// Finite verb after an initial determiner-led noun subject (`The game
/// ended`, `The bus arrived`). The closed verb list misses these, so the
/// sentence loses its root. Upgrades an alpha-fallback NOUN to VERB only in
/// strict sentence-initial DET+NOUN+NOUN position (0–2). Guards:
/// auxiliaries (`is/was`), punctuation (`,`, parentheticals), relativizers
/// (`who/that/where` — PRON/DET/WH-blocked), and conjunctions never match;
/// medial DET+NOUN+NOUN (`the sales report`) is out of scope by position.
/// Deliberately NOT covered: bare NOUN+NOUN initials (`Rain fell`,
/// `Translate hello`) — subject–verb and verb–object readings are
/// POS-identical there (`Define photosynthesis.` proves it), so that frame
/// needs verb-capability knowledge, not another positional guard.
fn refine_pos_initial_noun_verb(texts: &[String], pos: &mut [Upos]) {
    if texts.len() < 3 || pos[2] != Upos::Noun {
        return;
    }
    if pos[0] != Upos::Det || pos[1] != Upos::Noun {
        return;
    }
    if is_noun_subject_verb_blocker(texts[2].as_str()) {
        return;
    }
    // A finite verb in third position is never capitalized.
    if !texts[2].chars().next().is_some_and(|c| c.is_lowercase()) {
        return;
    }
    pos[2] = Upos::Verb;
}

/// Finite verb of a conjoined clause (`floods stayed`, `spirits rose`,
/// `cats nap`, `rice fill`, `coffee helps`). The closed verb list misses
/// these, so the second clause loses its predicate. Upgrades an
/// alpha-fallback NOUN to VERB only in CC + NOUN + NOUN position — the
/// middle noun is the post-conjunction subject, the third its verb. Guards:
/// the third token must be lowercase (title-case is the §8.2 proper-noun
/// class); clause-final CC + NOUN (`and eggs`, conjoined objects) has no
/// third token and never fires. Known boundary: clause-final conjoined
/// verbs (`but failed`, `or quit`) share their shape with conjoined objects
/// and need clause-subject tracking — left for later work.
fn refine_pos_conjoined_clause_verb(texts: &[String], pos: &mut [Upos]) {
    for i in 2..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Noun || pos[i - 2] != Upos::Cconj {
            continue;
        }
        if texts[i].chars().next().is_some_and(|c| c.is_lowercase()) {
            pos[i] = Upos::Verb;
        }
    }
}

/// Whether `text` is a finite `be` form (copula host). Sensory and
/// resultative copulas (`taste`, `become`) are their own rules.
fn is_be_form(text: &str) -> bool {
    const BE: &[&str] = &[
        "is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re", "'m",
    ];
    BE.iter().any(|h| text.eq_ignore_ascii_case(h))
}

/// Inverted copular predicate (`Is lunch ready?`, `is the sky blue`). The
/// be-predicate rule only sees AUX-adjacent complements, and its guard
/// rightly protects a directly-following nominal subject (`Is lunch…`
/// keeps `lunch`) — but when an overt subject stands between be and the
/// complement, the predicate strands as a nominal object. Upgrades an
/// alpha-fallback NOUN to ADJ only after the nearest preceding AUX-tagged
/// be-form with ≥1 nominal strictly between (the overt subject). Guards
/// (mirroring be-predicate/initial-noun): `-ing` participles, capitalized
/// targets, and subject-less complements (`is a doctor`, `is the station`
/// — bare determiners never match). Sequenced just before be-predicate;
/// disjoint from it by construction (its direct/bridged shapes have no
/// nominal between be and target).
fn refine_pos_inverted_copular(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() > 4
            && word
                .get(word.len() - 3..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
        {
            continue;
        }
        if !word.chars().next().is_some_and(|c| c.is_lowercase()) {
            continue;
        }
        let Some(j) = (0..i).rfind(|&j| pos[j] == Upos::Aux && is_be_form(texts[j].as_str()))
        else {
            continue;
        };
        if ((j + 1)..i).any(|k| matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)) {
            pos[i] = Upos::Adj;
        }
    }
}

/// Copular predicate adjective (`fee is low`, `isn't ready`). There is no
/// adjective detection elsewhere, so be-predicates fall through to NOUN and
/// the copula has nothing to attach to. Upgrades an alpha-fallback NOUN to
/// ADJ only directly after an AUX-tagged be-form, or after an n't/not
/// negator hosted by one (`isn't ready`). Guards: the be must not be
/// sentence-initial (interrogative AUX — `Is lunch ready` keeps its nominal
/// subject); possessive `'s` is X-tagged, never AUX, so `Bell's theorem` is
/// untouched; `-ing` forms are participles, not predicates (`are coming`,
/// `were surprising` stay for participle handling); determiners and verbs
/// never match. Sensory copulas (`tastes salty`) are a separate rule.
fn refine_pos_be_predicate(texts: &[String], pos: &mut [Upos]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        // Participles end in -ing (allocation-free suffix check); they are
        // verbs or participial adjectives, never copular predicates.
        let word = texts[i].as_str();
        let is_participle = word.len() > 4
            && word
                .get(word.len() - 3..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"));
        if is_participle {
            continue;
        }
        let direct = pos[i - 1] == Upos::Aux
            && is_be_form(texts[i - 1].as_str())
            && i - 1 >= 1;
        let bridged = i >= 2
            && (texts[i - 1].eq_ignore_ascii_case("n't") || texts[i - 1].eq_ignore_ascii_case("not"))
            && pos[i - 2] == Upos::Aux
            && is_be_form(texts[i - 2].as_str())
            && i - 2 >= 1;
        if direct || bridged {
            pos[i] = Upos::Adj;
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
    pub fn candidate_actions(
        &self,
        pos: &[Upos],
        texts: &[String],
        labels: &DepLabels,
    ) -> Vec<ArcEagerAction> {
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
                    // An object relativizer with an overt subject (the book
                    // that I bought): a nominative pronoun visibly between
                    // marker and verb proves s is the object, not the
                    // subject. Subject frames (no intervening pronoun) never
                    // offer it, so true subjects never compete.
                    if ps == Upos::Pron
                        && ((s + 1)..b).any(|m| is_nominative_subject(texts[m].as_str()))
                    {
                        out.push(Self::act(ArcEagerMove::Left, labels.obj));
                    }
                }
                // The subject of a predicate adjective (fee is low: fee →
                // nsubj → low). The only ADJ the tagger emits are be-rule
                // predicates, so this fires exactly on copular subjects.
                (Upos::Noun | Upos::Propn | Upos::Pron, Upos::Adj) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                }
                (Upos::Det, Upos::Noun | Upos::Propn) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.det));
                }
                (Upos::Adj, Upos::Noun | Upos::Propn) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.amod));
                }
                (Upos::Noun | Upos::Propn, Upos::Noun | Upos::Propn) => {
                    // Bare nominals compound (sales report). Comma-delimited
                    // ones are apposition — handled by the Right arm below
                    // (the appositive follows its anchor, so it attaches
                    // rightward); offering compound here too would tie.
                    // Conjoined ones (cats and dogs) are coordination —
                    // handled by the conj arm below; a conjunction between
                    // the pair is never compounding, so this arm stands
                    // down there too and the two never tie.
                    if ((s + 1)..b).all(|k| pos[k] != Upos::Punct && pos[k] != Upos::Cconj) {
                        out.push(Self::act(ArcEagerMove::Left, labels.compound));
                    }
                }
                (Upos::Aux, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.aux));
                }
                // The copula depends on its predicate (fee is low: is → cop
                // → low). Pairs with the be-predicate tagger rule above.
                (Upos::Aux, Upos::Adj) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.cop));
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
                // The subordinator depends on its clause verb (because it
                // snowed: because → mark → snowed). Needs the SCONJ Reduce
                // wait below: the marker must survive its subject first.
                // Never across a clause boundary (punctuation): in "Although
                // hungry, he shared", the marker must not reach the matrix
                // verb — it belongs to its own clause.
                (Upos::Sconj, Upos::Verb) => {
                    if ((s + 1)..b).all(|k| pos[k] != Upos::Punct) {
                        out.push(Self::act(ArcEagerMove::Left, labels.mark));
                    }
                }
                // A subordinator facing a nominal with no finite verb before
                // the clause boundary (Although tired, …): the marker's
                // attachment is underdetermined — no predicate exists to
                // host `mark`. Offer a `dep` rival at Shift parity so
                // `best_with_margin` records a near-tie (→
                // `RefineReason::Confidence(Ties)` → `AttachmentNearTie`
                // downstream) instead of a confident Shift. Clauses with an
                // overt verb (because it snowed) never match, so clean
                // subordinates stay tie-free. Shift still wins ties by
                // stable order, so heads/labels are unchanged — only the
                // margin drops (Track B: flag, don't guess).
                (Upos::Sconj, Upos::Noun | Upos::Propn | Upos::Pron) => {
                    let verbless = !(b..texts.len())
                        .take_while(|&j| pos[j] != Upos::Punct)
                        .any(|j| pos[j] == Upos::Verb);
                    if verbless {
                        out.push(Self::act(ArcEagerMove::Left, labels.dep));
                    }
                }
                _ => {}
            }
        }

        if b_free {
            match (ps, pb) {
                // A verb governs its nominal argument rightward — but never
                // across a clause boundary (He cooks, she cleans): a nominal
                // past punctuation is the next clause's subject, not this
                // verb's object. Withholding both arcs lets it shift into
                // Left-nsubj from its own predicate; the parataxis/appos
                // arms above own the punctuated pairs, so nothing else
                // competes. Same boundary idiom throughout.
                (Upos::Verb, Upos::Noun | Upos::Propn | Upos::Pron) => {
                    if ((s + 1)..b).all(|k| pos[k] != Upos::Punct) {
                        out.push(Self::act(ArcEagerMove::Right, labels.dobj));
                        out.push(Self::act(ArcEagerMove::Right, labels.nsubj));
                    }
                }
                // A second finite verb across a punctuation + nominal boundary
                // (She texted; he called) is parataxis, not a complement:
                // both a clause boundary and an intervening subject must
                // separate the pair. Adjacent verbs (called left, relative
                // clause), bare participles (smiling, took), and complement
                // clauses (said he left, no punctuation) never match. A
                // subordinator between the pair owns its own arm below, so a
                // punctuated subordinate (he left, because she cried) keeps
                // the incumbent parataxis reading.
                (Upos::Verb, Upos::Verb) => {
                    let boundary = ((s + 1)..b).any(|k| pos[k] == Upos::Punct);
                    let subject = ((s + 1)..b).any(|k| {
                        matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                    });
                    if boundary && subject {
                        out.push(Self::act(ArcEagerMove::Right, labels.parataxis));
                    } else if let Some(marker) =
                        (s + 1..b).rfind(|&m| pos[m] == Upos::Sconj)
                    {
                        // A subordinate clause verb governed by the matrix
                        // verb (stayed because it snowed: snowed → ccomp →
                        // stayed). The innermost (nearest-b) subordinator
                        // decides the label, answering to the UD refs:
                        // `because` complements (ccomp), `when`/`if`/`after`
                        // adjoin (advcl); `as`/`although` (comment/verbless
                        // frames) keep their own dynamics. An overt nominal
                        // subject must stand between marker and verb —
                        // subject-less fragments never match. Needs the Verb
                        // Sconj wait below so the matrix verb survives until
                        // the clause verb arrives.
                        let word = texts[marker].as_str();
                        let has_subject = ((marker + 1)..b).any(|k| {
                            matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                        });
                        if has_subject {
                            if word.eq_ignore_ascii_case("because") {
                                out.push(Self::act(ArcEagerMove::Right, labels.ccomp));
                            } else if word.eq_ignore_ascii_case("when")
                                || word.eq_ignore_ascii_case("if")
                                || word.eq_ignore_ascii_case("after")
                            {
                                out.push(Self::act(ArcEagerMove::Right, labels.advcl));
                            }
                        }
                    }
                }
                // A comma-delimited nominal follows its anchor (My brother, a
                // doctor): the appositive attaches rightward. The Left
                // compound arm above stands down for punctuated pairs, so the
                // two never compete. Needs the NOUN Reduce wait below so the
                // anchor survives the appositive's determiner. A conjoined
                // nominal (cats and dogs: dogs → conj → cats) attaches the
                // same way, gated on a conjunction strictly between the
                // pair — bare compounds never compete (the compound arm
                // stands down for conjoined pairs), and a pair with both a
                // comma and a conjunction keeps the incumbent appos reading
                // (appos is offered first). The anchor must not itself be
                // comma-adjacent (red in "car, red and fast"): comma-opened
                // nominals are appositive material owned by the appos frame
                // — coordination inside apposition is its own rule. Needs
                // the nominal CCONJ wait below so the anchor survives the
                // marker. Scoped to nominal pairs — conjoined verbs need
                // conjunct-agreement POS knowledge (their second conjunct
                // tags NOUN today), not this attachment arm.
                (Upos::Noun | Upos::Propn, Upos::Noun | Upos::Propn) => {
                    if ((s + 1)..b).any(|k| pos[k] == Upos::Punct) {
                        out.push(Self::act(ArcEagerMove::Right, labels.appos));
                    }
                    if s == 0 || texts[s - 1] != "," {
                        if ((s + 1)..b).any(|k| pos[k] == Upos::Cconj) {
                            out.push(Self::act(ArcEagerMove::Right, labels.conj));
                        }
                    }
                }
                // The relcl predicate depends on its nominal anchor (called →
                // relcl → man). Gated on a nominal-headed who/that/where
                // strictly between the pair, so true subjects (Dogs bark,
                // no marker) never compete: the Left nsubj arm above simply
                // isn't rivaled there. Needs the marker Reduce wait below so
                // the anchor survives its relativizer (Reduce outbids Shift).
                (Upos::Noun | Upos::Propn, Upos::Verb) => {
                    if ((s + 1)..b).any(|m| {
                        m > 0
                            && is_relative_marker(texts[m].as_str())
                            && matches!(pos[m - 1], Upos::Noun | Upos::Propn)
                    }) {
                        out.push(Self::act(ArcEagerMove::Right, labels.relcl));
                    }
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
        // Same for an unheaded verb with a determiner on the buffer (Close
        // your books): its object phrase is arriving — popping strands the
        // verb and the object falls to repair-dep instead of dobj. Scoped to
        // non-closed verbs (the directive upgrades): closed verbs keep the
        // old dynamics their fallback paths rely on (Find the cheapest
        // flight, where cheapest must stay free to compound onto flight).
        // And a subordinator waits for its clause (subject NP, then verb):
        // popping on the subject ([because], it) strands the marker in
        // repair-dep instead of mark. AUX/PART buffers keep the old
        // exclusion above — copular fragments are out of scope.
        // And a nominal anchor waits for its determiner (My brother, a
        // doctor): popping on the DET ([brother], a) strands the anchor
        // before the appositive arrives. Scoped to bare nouns — verbs keep
        // their own waits, pronouns their greedy Right arms.
        // And a nominal anchor waits for its relativizer (The man [who]):
        // popping on the marker strands the anchor before the clause verb
        // arrives, and the relcl arm above never meets its pair. Scoped to
        // nominal-headed markers — complementizer that (verb-headed) and
        // bare interrogatives keep the old dynamics. The wait spans the
        // embedded subject too (The book that [I]): object relatives strand
        // the anchor a second time on the subject pronoun, so a PRON
        // directly after a nominal-headed marker extends the same wait —
        // the pronoun itself still reduces normally (only nominal stack
        // tops wait), and complement frames never match (verb-headed that).
        // And an auxiliary waits for its clause to form (Can [you]): with
        // no arc pairing (Aux, nominal), Reduce pops the aux on its subject
        // and it strands into repair-dep — while the Left-aux arm below
        // would have attached it had the pair met. Scoped to nominal-ish
        // buffers (subject/determiner arriving); verbs, adjectives, and
        // auxiliaries keep the old dynamics (cop/neg/progressive paths).
        // And an object relativizer waits for the embedded subject (that
        // [I]): popping on the subject strands the marker before the clause
        // verb arrives, and the obj arm above never meets its pair. Scoped
        // to pronouns directly after a nominal-headed marker — subject
        // markers attach immediately, complement frames never match.
        // And a matrix verb waits for its subordinate clause (stayed
        // [because]): popping on the subordinator strands the verb before
        // the clause verb arrives, and the ccomp/advcl arm above never
        // meets its pair. Scoped to subordinator buffers — complementizer
        // `that` is DET (never matches), bare complements (`said he left`,
        // no marker) and comment/verbless frames (`as`, `although`) keep
        // the old dynamics.
        // And a nominal anchor waits for its conjunct (cats [and]): popping
        // on the conjunction strands the first conjunct before the second
        // arrives, and the conj arm above never meets its pair. Scoped to
        // BARE nominal stack tops (no dependents yet — a nominal already
        // governing dependents is phrasally saturated, e.g. the clause
        // predicate in "Grades dropped yet spirits rose", which must reduce
        // so the second clause can form) with conjunction buffers — verbs
        // keep their own waits, pronouns their greedy Right arms.
        if self.right_children[s].is_empty()
            && ps != Upos::Adp
            && !matches!(pb, Upos::Aux | Upos::Part)
            && !(self.heads[s] == -1
                && ps == Upos::Verb
                && pb == Upos::Det
                && !is_closed_verb(&texts[s]))
            && !(ps == Upos::Sconj && matches!(pb, Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det))
            && !(ps == Upos::Noun && pb == Upos::Det)
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && b > 0
                && is_relative_marker(texts[b].as_str())
                && matches!(pos[b - 1], Upos::Noun | Upos::Propn))
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && pb == Upos::Pron
                && b >= 2
                && is_relative_marker(texts[b - 1].as_str())
                && matches!(pos[b - 2], Upos::Noun | Upos::Propn))
            && !(ps == Upos::Aux
                && matches!(pb, Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det))
            && !(ps == Upos::Pron
                && pb == Upos::Pron
                && b >= 2
                && is_relative_marker(texts[b - 1].as_str())
                && matches!(pos[b - 2], Upos::Noun | Upos::Propn))
            && !(ps == Upos::Verb && pb == Upos::Sconj)
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && pb == Upos::Cconj
                && self.left_children[s].is_empty()
                && self.right_children[s].is_empty())
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
                    // Predicate-adjective subject (fee is low): same label,
                    // same confidence as verbal nsubj.
                    l if l == labels.nsubj && pos[b] == Upos::Adj => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn | Upos::Pron) {
                            100.0
                        } else {
                            10.0
                        }
                    }
                    // The object relativizer outbids the subject-Left (100):
                    // inside a gated frame the overt subject between marker
                    // and verb proves s is the object, and attaching s as
                    // subject would both mishead it and pop it before the
                    // anchor meets the verb. Ungated here by design — the
                    // candidate arm above is the gate, so this weight only
                    // ever ranks intervening-subject pairs.
                    l if l == labels.obj && pos[b] == Upos::Verb => {
                        if pos[s] == Upos::Pron {
                            105.0
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
                    l if l == labels.mark => {
                        if pos[s] == Upos::Sconj && pos[b] == Upos::Verb {
                            95.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.cop => {
                        if pos[s] == Upos::Aux && pos[b] == Upos::Adj {
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
                    l if l == labels.parataxis => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Verb {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    // The subordinate clause verb depends on its matrix verb
                    // (snowed → ccomp → stayed, works → advcl → sings).
                    // Parataxis-level confidence; the candidate arm above is
                    // the gate, so these weights only ever rank
                    // marker-framed pairs.
                    l if l == labels.ccomp => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Verb {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.advcl => {
                        if pos[s] == Upos::Verb && pos[b] == Upos::Verb {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    l if l == labels.appos => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn)
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            60.0
                        } else {
                            5.0
                        }
                    }
                    // The conjunct depends on its anchor (dogs → conj →
                    // cats): coordination-level confidence, matching the
                    // compound/appos pair it disjoins from by gate. Ungated
                    // here by design — the candidate arm above is the gate.
                    l if l == labels.conj => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn)
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            60.0
                        } else {
                            5.0
                        }
                    }
                    // The relcl arc outbids the subject-Left (100): inside a
                    // marker frame the anchor is never the clause verb's
                    // subject (the relativizer or its pronoun is), and
                    // attaching the anchor Left would both mishead it and pop
                    // it before the matrix verb arrives. Ungated here by
                    // design — the candidate arm above is the gate, so this
                    // weight only ever ranks marker-framed pairs.
                    l if l == labels.relcl => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn) && pos[b] == Upos::Verb {
                            105.0
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
    pub mark: u64,
    pub cop: u64,
    pub appos: u64,
    pub parataxis: u64,
    pub acomp: u64,
    pub xcomp: u64,
    pub relcl: u64,
    pub obj: u64,
    pub ccomp: u64,
    pub advcl: u64,
    pub conj: u64,
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
            mark: intern("mark"),
            cop: intern("cop"),
            appos: intern("appos"),
            parataxis: intern("parataxis"),
            acomp: intern("acomp"),
            xcomp: intern("xcomp"),
            relcl: intern("relcl"),
            obj: intern("obj"),
            ccomp: intern("ccomp"),
            advcl: intern("advcl"),
            conj: intern("conj"),
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
        // Contextual pass over the lexeme-only tags: determiner-led nominals
        // colliding with the closed verb list. Runs first so only infer_pos
        // verbs are candidates and every VERB-upgrade below reads corrected
        // tags.
        refine_pos_det_closed_verb(&texts, &mut pos);
        // Contextual pass over the lexeme-only tags: bare infinitive after a
        // do-modal host. Runs before root-picking so the oracle sees verbs.
        refine_pos_bare_infinitive(&texts, &mut pos);
        // Contracted-be pass: pronoun-hosted 's → AUX, then clitic-hosted
        // -ing participle → VERB. Sequenced after the infinitive pass; the
        // two govern disjoint contexts (aux-host vs. clitic-host).
        refine_pos_contracted_be(&texts, &mut pos);
        // Directive pass: sentence-initial NOUN + DET + NOUN → VERB.
        // Sequenced last; disjoint from the aux/clitic triggers above.
        refine_pos_directive_initial(&texts, &mut pos);
        // Pronoun-subject pass: finite verb after a nominative pronoun.
        // Disjoint from all above (PRON prev never matches aux/clitic/
        // initial triggers).
        refine_pos_pronoun_subject_verb(&texts, &mut pos);
        // That-relative pass: nominal-headed that → PRON, then the NOUN
        // after that- PRON → VERB. Sequenced after the pronoun pass (whose
        // nominative list never pronounces that) and before the
        // relative-matrix pass, whose VERB-host gate reads the clause verbs
        // upgraded here (cried → slept). Disjoint targets from both.
        refine_pos_that_relative(&texts, &mut pos);
        // Where-marker pass: nominal-headed where → SCONJ so the existing
        // mark arm fires. Sequenced with the other frame passes; targets
        // (NOUN-where) are disjoint from every verb/ADJ upgrade, and no
        // refine reads SCONJ positionally (the initial-noun blocker keys on
        // the word, order-free).
        refine_pos_where_marker(&texts, &mut pos);
        // Clausal-after pass: ADP after + subject + verb → SCONJ so the
        // existing mark arm fires. Sequenced after the pronoun-subject pass
        // (clause verbs upgraded there are the VERB host) with the other
        // SCONJ-frame passes; nominal complements never match, so prep/pobj
        // frames are untouched.
        refine_pos_clausal_after(&texts, &mut pos);
        // Final-adverbial pass: closed time/manner set + -ly finals →
        // ADV. Targets are disjoint from the comma-adverbial pass (which
        // owns comma -ly with its clause-edge host guard; this pass skips
        // comma -ly), and ADV outputs feed no verb/ADJ upgrade.
        refine_pos_final_adverbial(&texts, &mut pos);
        // Linking-predicate pass: bare-initial sensory verb → VERB, then
        // the NOUN after a sensory VERB → ADJ. Sequenced with the frame
        // passes; targets (sensory words, their complements) are disjoint
        // from every relativizer/adverbial trigger, and ADJ outputs feed
        // nothing upstream of the be-predicate pass.
        refine_pos_linking_predicate(&texts, &mut pos);
        // Post-comma pass: clause-initial NOUN after a parenthetical
        // boundary → VERB. Sequenced after the linking pass (sensory verbs
        // read first where both could apply — disjoint in practice: no
        // bench sensory verb sits post-comma) and before the
        // relative-matrix pass (disjoint: matrix needs a marker frame).
        refine_pos_post_comma_verb(&texts, &mut pos);
        // Shifted-initial pass: comma + DET + NOUN + clause-final NOUN →
        // VERB. Sequenced after the post-comma pass (disjoint: that pass
        // needs comma-adjacent targets, this one comma-distant) and before
        // the relative-matrix pass (disjoint: matrix needs a marker
        // frame). The ADV trailer reads final-adverbial outputs.
        refine_pos_shifted_det_noun_verb(&texts, &mut pos);
        // Comment-As pass: sentence-initial as + nominal → SCONJ so the
        // existing mark arm fires. First-token only, disjoint from every
        // medial frame; SCONJ outputs feed no refine (waits and arms read
        // them at transition time).
        refine_pos_comment_as(&texts, &mut pos);
        // Relative-matrix pass: sentence-final NOUN after a relcl VERB with
        // a nominal-headed who/that/where earlier. Sequenced after the
        // pronoun pass so relcl verbs upgraded there (wait, study, sang)
        // are visible as the VERB host; disjoint targets (sentence-final
        // only) from the initial-noun positions.
        refine_pos_relative_matrix_verb(&texts, &mut pos);
        // Adverbial pass: comma-framed -ly NOUN → ADV. Sequenced after the
        // verb passes; targets (NOUN) are disjoint from every verb/ADJ
        // upgrade above, and comma-framed adverbials never feed those
        // triggers (no DET+NOUN initials, no PRON hosts, no be hosts).
        refine_pos_comma_adverbial(&texts, &mut pos);
        // Initial noun-subject pass: DET+NOUN+NOUN / NOUN+NOUN at the start.
        // Disjoint (targets positions the earlier passes leave as NOUN).
        refine_pos_initial_noun_verb(&texts, &mut pos);
        // Conjoined-clause pass: CC + NOUN + NOUN → VERB. Disjoint (CC prev
        // matches none of the above triggers).
        refine_pos_conjoined_clause_verb(&texts, &mut pos);
        // Inverted-copular pass: be-AUX + subject + predicate-NOUN → ADJ.
        // Sequenced just before be-predicate; disjoint from it (its
        // direct/bridged shapes have no nominal between be and target) and
        // from every verb trigger above (ADJ targets).
        refine_pos_inverted_copular(&texts, &mut pos);
        // Copular-predicate pass: be + NOUN → ADJ. Disjoint (AUX prev with
        // NOUN target matches none of the verb triggers above).
        refine_pos_be_predicate(&texts, &mut pos);

        // Sentence boundaries from the sentencizer.
        let starts = self.sentencizer.predict(doc);
        let sentences = partition_sentences(&starts, doc.len());

        let mut state = ArcEagerState::new(doc.len(), 0);
        let oracle = DeterministicOracle;
        let mut margins = Vec::new();
        let mut sentence_roots = Vec::new();

        for (s, e) in sentences {
            let root = pick_root(&pos, &texts, s, e);
            sentence_roots.push(root);
            state.reset_for_sentence(s, e, root, self.labels.root);
            while !state.is_final() {
                let actions = state.candidate_actions(&pos, &texts, &self.labels);
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

/// The sentence root: the leftmost verb that does not close a relative
/// clause; else the leftmost adjective (copular predicates head their
/// clause); else the leftmost NOUN/PROPN/PRON; else the first token (the
/// minimal star fallback for degenerate sentences).
///
/// The first rung is a first-accept walk with a skip predicate, not a bare
/// leftmost pick: a verb pending after a nominal-headed who/that/where is
/// the relcl predicate, never the matrix root (`The man who called left`
/// roots at `left`). The skip resets at every verb, so only the clause the
/// marker opens is skipped — a matrix-first verb (`I know the man who
/// called`) still wins, and complementizer `that` (verb-headed) or
/// sentence-initial interrogatives (no nominal head) never arm the skip.
/// Load-bearing for the relcl arm: a pre-designated root is excluded from
/// re-attachment, so the label arm can only fire once the crown moves.
fn pick_root(pos: &[Upos], texts: &[String], s: usize, e: usize) -> usize {
    let mut rel_pending = false;
    for i in s..e {
        if pos[i] == Upos::Verb {
            if !rel_pending {
                return i;
            }
            rel_pending = false;
            continue;
        }
        if i > 0
            && is_relative_marker(texts[i].as_str())
            && matches!(pos[i - 1], Upos::Noun | Upos::Propn)
        {
            rel_pending = true;
        }
    }
    (s..e)
        .find(|&i| pos[i] == Upos::Adj)
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
