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

/// The closed function-word POS map — the only way a lexeme-only heuristic
/// gets DET/ADP/AUX/CCONJ/SCONJ. Reads per-language
/// [`LexemeFlags`] bits (populated from `LexiconConfig::function_words`)
/// instead of hard-coded spellings; priority order (DET > ADP > AUX >
/// CCONJ > SCONJ > PRON) decides multi-category forms (`after` is ADP,
/// `that`/`her` are DET). Honest about its limits: it is a finite
/// data table, not a trained tagger.
fn closed_funcword_pos(flags: LexemeFlags) -> Option<Upos> {
    if flags.is_det_word() {
        Some(Upos::Det)
    } else if flags.is_adp_word() {
        Some(Upos::Adp)
    } else if flags.is_aux_word() {
        Some(Upos::Aux)
    } else if flags.is_cconj_word() {
        Some(Upos::Cconj)
    } else if flags.is_sconj_word() {
        Some(Upos::Sconj)
    } else if flags.is_pron_word() {
        Some(Upos::Pron)
    } else {
        None
    }
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
fn refine_pos_det_closed_verb(_texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 1..pos.len() {
        if pos[i] != Upos::Verb || pos[i - 1] != Upos::Det {
            continue;
        }
        pos[i] = Upos::Noun;
    }
}

/// Hosts that govern a bare infinitive: do-support and the modals, plus the
/// `n't`-split stubs the tokenizer emits (`wo`/`n't`, `ca`/`n't`). Matched
/// as a lexeme bit, not a spelling list.
fn is_bare_infinitive_host(flags: LexemeFlags) -> bool {
    flags.is_bare_inf_host()
}

/// Whether the lexeme is a negator hosted by an auxiliary (`n't`, `not`).
fn is_aux_negator(flags: LexemeFlags) -> bool {
    flags.is_negator()
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
/// A finite verb later in the clause withholds the upgrade: the post-host
/// nominal is then the inverted subject, not the infinitive (`Does
/// photosynthesis work` — `work` crowns, `photosynthesis` subjects — while
/// `Do help them`, with no verb ahead, keeps the infinitive reading).
/// Clause-final s-forms don't count as the later verb: they are plural
/// object nouns (`She won't answer calls`), owned by the bare-object rule
/// below, so `answer` still upgrades.
fn refine_pos_bare_infinitive(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let governed = if is_bare_infinitive_host(flags[i - 1]) {
            true
        } else if is_aux_negator(flags[i - 1]) && i >= 2 && is_bare_infinitive_host(flags[i - 2]) {
            true
        } else {
            false
        };
        if !governed {
            continue;
        }
        let later_verb = (i + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .any(|k| pos[k] == Upos::Verb && !is_sform_final(texts, k));
        if !later_verb {
            pos[i] = Upos::Verb;
        }
    }
}

/// A verb-form token reading as a plural object noun: an s-form with nothing
/// but punctuation after it (`calls` in `She won't answer calls`). Shared by
/// the bare-object rule below and the bare-infinitive gate above (which must
/// not mistake such objects for the clause predicate).
fn is_sform_final(texts: &[String], i: usize) -> bool {
    let word = texts[i].as_str();
    word.len() > 2
        && word
            .get(word.len() - 1..)
            .is_some_and(|sfx| sfx.eq_ignore_ascii_case("s"))
        && texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        })
}

/// Bare object noun after a verb (`She won't answer calls`). A closed-verb
/// s-form directly after a VERB at sentence end is a plural object noun:
/// English morphosyntax forbids a finite s-form after a bare verb
/// (modals and causatives govern bare forms), so the s-form reads
/// nominal. Downgrades VERB → NOUN only finally (nothing but punctuation
/// follows) with a VERB host. Guards: pronoun hosts (`she calls` —
/// finite 3sg), non-s-forms (`called left` — matrix root), and DET hosts
/// (the determiner rule owns those) never match.
fn refine_pos_bare_object_noun(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Verb || pos[i - 1] != Upos::Verb {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() <= 2
            || !word
                .get(word.len() - 1..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("s"))
        {
            continue;
        }
        if !texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        }) {
            continue;
        }
        pos[i] = Upos::Noun;
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
fn refine_pos_directive_initial(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
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

/// Demonstrative object pronoun (`Translate this to French`, `Summarize
/// this?`, `Elaborate on this`). The closed map lexes every `this`/`these`/
/// `those` as DET, but a determiner obligatorily heads a nominal on its
/// right — with an ADP, clause boundary, or nothing after it, there is no
/// head, so the demonstrative IS the object (UD: PRON). Upgrades DET to
/// PRON only when the next token is an ADP or sentence-final punctuation —
/// a nominal (`this equation`), verb (`this works`), or auxiliary (`This
/// is`) after it keeps the determiner reading. `that` is excluded: it
/// relativizes (`Dogs that bark bite`), and the that-relative pass owns it.
/// Sequenced before the imperative pass so the retagged pronoun feeds its
/// pronoun-object frame (`Translate this` crowns via the PRON second, the
/// same dynamics as `Remind me`).
fn refine_pos_demonstrative_object(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Det {
            continue;
        }
        if !flags[i].is_demonstrative() {
            continue;
        }
        let bare = match texts.get(i + 1) {
            None => true,
            Some(next) => {
                matches!(
                    next.as_str(),
                    "." | "!" | "?" | ";" | ":" | "—" | "--" | "," | "(" | ")" | "..."
                ) || flags.get(i + 1).is_some_and(|f| f.is_adp_word())
            }
        };
        if bare {
            pos[i] = Upos::Pron;
        }
    }
}

/// Discourse-initial imperative (`Please confirm the date`, `Never mind
/// that`, `Always check twice`, `Just send it now`, `Kindly review the
/// draft`). A politeness/temporal marker before a bare verb strands the
/// verb as NOUN (no subject rule sees a marker-adjacent predicate) and the
/// marker itself roots as NOUN, corrupting the whole clause. Upgrades the
/// second-position NOUN to VERB and retags the marker (`please` → INTJ, a
/// tag the parser never otherwise produces and nothing else reads, so it
/// strands honestly via repair-dep; `never`/`always`/`just`/`kindly` → ADV,
/// taking the existing (Adv, Verb) advmod arm) only in a verbless clause
/// (the same gate as the imperative pass — `Dogs bark` never matches) with
/// a DET/PRON/ADV or word-gated `twice` third (`confirm the`, `mind that`,
/// `send it`, `check twice`). Fragments with nominal thirds (`Just good
/// friends`, `Never a dull moment` — DET/NOUN thirds) and mid-sentence
/// markers (`as always,`, `... please.`) never match. Known boundary:
/// `twice`/`now`-class temporal complements keep their own (mis)tagging —
/// the arm crowns the verb, the adjunct stays residual.
fn refine_pos_discourse_initial_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    if !pos.iter().any(|&p| p == Upos::Noun) {
        return;
    }
    if pos.iter().any(|&p| p == Upos::Verb || p == Upos::Aux) {
        return;
    }
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun || !flags[i].is_discourse_marker() {
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
        if pos[i + 1] != Upos::Noun {
            continue;
        }
        let third_ok = matches!(pos[i + 2], Upos::Det | Upos::Pron | Upos::Adv)
            || flags[i + 2].is_twice_word();
        if !third_ok {
            continue;
        }
        pos[i + 1] = Upos::Verb;
        pos[i] = if flags[i].is_please_word() {
            Upos::Intj
        } else {
            Upos::Adv
        };
    }
}

/// Attributive `-ly` modifier (`quarterly sales report`). Words ending
/// in -ly are adverbial by default (the final-adverbial and
/// comma-adverbial passes own those shapes), but directly before a nominal
/// head they are attributive adjectives — English has a productive class
/// of them (quarterly, monthly, friendly, only). Upgrades an
/// alpha-fallback NOUN ending in -ly (length > 4, so `July`-class shorts
/// never match) to ADJ only directly before a NOUN/PROPN. Guards: anything
/// already tagged ADV (`widely`, `kindly` after the discourse pass,
/// `Sadly,` after the comma pass) never matches — the rule only sees
/// NOUNs — and conjunctions/punctuation next keep the nominal reading
/// (`Run daily or quit` stays put). Sequenced after the discourse pass so
/// `Kindly review` (retagged ADV there) is never re-read as attributive.
fn refine_pos_attributive_ly(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
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
        if i + 1 < texts.len() && matches!(pos[i + 1], Upos::Noun | Upos::Propn) {
            pos[i] = Upos::Adj;
        }
    }
}

/// Sentence-initial imperative with a non-determiner complement (`Remind
/// me at noon`, `Translate hello to French`, `Explain Bell's theorem`,
/// `Help them win`). Directives outside the closed verb list fall through
/// to NOUN and the clause loses its root; the directive pass only covers
/// DET-led objects, leaving pronoun, bare-nominal, and proper-name objects
/// stranded. Upgrades a sentence-initial alpha-fallback NOUN to VERB only
/// in a verbless clause (no VERB/AUX anywhere — the same gate as the
/// fallback-root tie, so `Dogs bark`, `Anna finished`, and `Cats and dogs
/// play` never match) with one of three complement frames: an object
/// pronoun (`me`, `them` — nominatives like `who` head relative clauses
/// and never match); a bare nominal object plus a prepositional
/// adjunct later in the clause (`hello … to French`); or a nominal object
/// plus a possessive `'s` (`Bell's theorem`). Strictly scoped: determiner
/// seconds (the directive pass owns those), verb/conjunction seconds, and
/// bare-final nominal pairs (`Define photosynthesis` — the Track B
/// verbless-fragment tie owns those) never fire, so subjects and NP
/// fragments are untouched. Known boundary: verbless headlines with a PP
/// adjunct (`Markets rally in Asia`) read imperative — no corpus instance.
fn refine_pos_imperative_non_det_object(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    if !pos.iter().any(|&p| p == Upos::Noun) {
        return;
    }
    if pos.iter().any(|&p| p == Upos::Verb || p == Upos::Aux) {
        return;
    }
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let at_start = i == 0
            || matches!(
                texts[i - 1].as_str(),
                "." | "!" | "?" | ";" | ":" | "—" | "--"
            );
        if !at_start || i + 1 >= texts.len() {
            continue;
        }
        // Pronoun object (`Remind me`, `Help them`): only object forms —
        // nominative pronouns (`who`, `you`, `they`) head relative and
        // finite clauses (`People who wait`, the mirror of the
        // pronoun-subject pass, which upgrades after exactly these forms),
        // so they never read as imperative objects.
        if pos[i + 1] == Upos::Pron && !is_nominative_subject(flags[i + 1]) {
            pos[i] = Upos::Verb;
            continue;
        }
        // Bare nominal object with a prepositional adjunct later in the
        // clause (`Translate hello to French`) or a possessive `'s`
        // (`Explain Bell's theorem`). Nominal pairs with neither
        // (`Dogs chase red cars`) keep the compound dynamics.
        if matches!(pos[i + 1], Upos::Noun | Upos::Propn) {
            let rest = &texts[i + 2..];
            let has_pp = (0..rest.len())
                .take_while(|&k| {
                    !matches!(
                        rest[k].as_str(),
                        "." | "!" | "?" | ";" | ":" | "," | "—" | "--"
                    )
                })
                .any(|k| flags[i + 2 + k].is_adp_word());
            let has_possessive = flags
                .get(i + 2)
                .is_some_and(|f| f.is_be_clitic_s());
            if has_pp || has_possessive {
                pos[i] = Upos::Verb;
            }
        }
    }
}

/// Dual-class `get`/`got` imperative (`Get me a coffee`). The closed map
/// lexes both as AUX (passive/causative hosts), so a lexical main-verb
/// `Get` strands as an aux-dependent and its object pronouns root the
/// sentence. Upgrades a sentence-initial AUX-tagged get-word to VERB with
/// a nominal/pronoun complement ahead in the clause — the ditransitive
/// frame (`me a coffee`) then lands through the existing iobj/dobj arms.
/// Guards: non-initial uses and complement-less frames (passives,
/// causatives) never match. Known boundary: `get` + adjectival complement
/// (`get dressed`, no corpus instance) stays approximate. Same O(n)
/// frame-scan shape as the other refines.
fn refine_pos_get_imperative(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    if texts.is_empty() || pos[0] != Upos::Aux || !flags[0].is_get_word() {
        return;
    }
    let complement = (1..texts.len())
        .take_while(|&k| pos[k] != Upos::Punct)
        .any(|k| matches!(pos[k], Upos::Pron | Upos::Det | Upos::Noun | Upos::Propn));
    if complement {
        pos[0] = Upos::Verb;
    }
}

/// Bare past-tense transitive (`John opened the door`, `Anna finished her
/// lunch`). An -ed word with a bare-initial nominal subject and a
/// determiner-led object is a past-tense predicate: -ed adjectives live in
/// attributive (DET-led) or predicative-after-linking/be (AUX-prev)
/// position, never here.
fn refine_pos_bare_ed_transitive(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    if texts.len() < 4
        || !matches!(pos[0], Upos::Noun | Upos::Propn)
        || pos[1] != Upos::Noun
        || pos[2] != Upos::Det
        || !matches!(pos[3], Upos::Noun | Upos::Propn)
    {
        return;
    }
    let word = texts[1].as_str();
    if word.len() <= 3
        || !word
            .get(word.len() - 2..)
            .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ed"))
        || !word.chars().next().is_some_and(|c| c.is_lowercase())
    {
        return;
    }
    pos[1] = Upos::Verb;
}

/// Whether the lexeme is a possessive determiner (obligatorily heads a nominal
/// on its right: my/your/her snack). Articles route through the existing
/// det dynamics; complementizer `that` (verb-headed) is excluded by category
/// at the call site; object/possessive pronouns (`me`, `mine`) lex as
/// PRON, never DET.
fn is_possessive_determiner(flags: LexemeFlags) -> bool {
    flags.is_possessive()
}

/// Whether the lexeme is a nominative pronoun surface (finite-clause subject).
/// Object/possessive forms (`me`, `them`, `mine`) are excluded even though
/// the closed map tags them PRON.
fn is_nominative_subject(flags: LexemeFlags) -> bool {
    flags.is_nominative()
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
fn refine_pos_pronoun_subject_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Pron {
            continue;
        }
        if is_nominative_subject(flags[i - 1]) {
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
fn refine_pos_comment_as(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    if texts.len() < 2 || pos[0] != Upos::Adp || !flags[0].is_as_word() {
        return;
    }
    if matches!(pos[1], Upos::Noun | Upos::Propn | Upos::Pron) {
        pos[0] = Upos::Sconj;
    }
}

/// Whether the lexeme is a nominal relativizer with corpus evidence (`who`,
/// `that`, `where`). Only forms attested in the bench; `which`/`whom` have
/// no instance and stay out until one appears.
fn is_relative_marker(flags: LexemeFlags) -> bool {
    flags.is_relativizer()
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
fn refine_pos_shifted_det_noun_verb(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
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
fn refine_pos_relative_matrix_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    // Nominal-headed relativizer positions (the relative-frame gate).
    let markers: Vec<usize> = (1..texts.len())
        .filter(|&r| {
            is_relative_marker(flags[r]) && matches!(pos[r - 1], Upos::Noun | Upos::Propn)
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

/// Matrix predicate after a relative clause with a complement ahead
/// (`stands empty`, `improve fast`, `ducks out …`). The existing
/// relative-matrix pass owns the sentence-final shape (`barked ran off`);
/// here the matrix verb carries material after it, so that pass's final
/// gate never sees it and the verb strands as a nominal object — the
/// relcl verb keeps the crown and the whole clause misattaches.
/// Two licensed readings sharing one frame gate (a nominal-headed
/// relativizer earlier whose clause verb is the immediately preceding
/// VERB — the same `closes_relcl` shape as the final pass, so fragments
/// and bare coordinations never match):
///
/// - an `-s` form (`stands`, `ducks`) is verbal by morphology — English
///   has no nominal `-s` reading after a finite verb — and additionally
///   licenses a following preposition (`ducks out on weekends`);
/// - a bare form (`improve`) needs an overt nominal/adjectival/adverbial
///   complement or determiner directly after it (`improve fast`).
///
/// Guards (mirroring the final pass): finite verbs are never capitalized,
/// and pre-verbal coordinators/complementizers never match. Known
/// boundaries: markerless verb–verb sequences (`eat accumulates` — no
/// relativizer, reads as verb+object without lexicon knowledge) and
/// bare-form + PP-adjunct (`run out of time`, no corpus instance) stay
/// approximate. Same O(n) frame-scan shape as the other refines.
fn refine_pos_relcl_matrix_complement(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    // Nominal-headed relativizer positions (the relative-frame gate).
    let markers: Vec<usize> = (1..texts.len())
        .filter(|&r| {
            is_relative_marker(flags[r]) && matches!(pos[r - 1], Upos::Noun | Upos::Propn)
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
        // Sentence-final targets belong to the existing matrix pass.
        if texts[i + 1..].iter().all(|t| {
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
        if !closes_relcl {
            continue;
        }
        let word = texts[i].as_str();
        let s_form = word.len() > 3
            && word
                .get(word.len() - 1..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("s"));
        let next = pos[i + 1];
        // Bare forms need a nominal/adjectival/adverbial complement or a
        // determiner; `-s` morphology additionally licenses a following
        // preposition (the adjunct PP of `ducks out on weekends`).
        let licensed = matches!(
            next,
            Upos::Noun | Upos::Propn | Upos::Pron | Upos::Adj | Upos::Adv | Upos::Det
        ) || (s_form && next == Upos::Adp);
        if licensed {
            pos[i] = Upos::Verb;
        }
    }
}
/// has no ADV path, so sentence/manner adverbials fall through to NOUN —
/// and sentence-initial ones even steal root. Upgrades an alpha-fallback
/// NOUN to ADV only for an `-ly` form (allocation-free suffix check, same
/// shape as the be-predicate `-ing` check) immediately followed by a comma
/// and hosted by a clause edge (sentence start or a preceding comma).
/// Guards: temporal `-ly` nominals after a preposition (`In July, …`) and
/// determiner-headed ones (`The family …`) never match. Known boundary:
/// non-`ly` adverbials (`early`, `still`, `always`, `daily`, `hard`) need an
/// adverb lexicon, not a suffix frame.
fn refine_pos_comma_adverbial(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
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
fn refine_pos_that_relative(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] == Upos::Det
            && flags[i].is_that_word()
            && matches!(pos[i - 1], Upos::Noun | Upos::Propn)
        {
            pos[i] = Upos::Pron;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Pron {
            continue;
        }
        if flags[i - 1].is_that_word() {
            pos[i] = Upos::Verb;
        }
    }
}

/// Interrogative WH-adverbial (`Where is my bag`, `Why is the sky blue`,
/// `How does photosynthesis work`, `When does the store close`). The tagger
/// leaves where/why/how as NOUN and `when` as SCONJ, so interrogative
/// adverbials read as subjects and markers — yet the refs pin ADV/advmod
/// for all four unanimously. Upgrades NOUN where/why/how → ADV when no
/// nominal or determiner precedes (sentence-initial or after a boundary —
/// the exact mirror of the where-marker gate below, disjoint by
/// construction), and SCONJ `when` → ADV only before an AUX (the inversion
/// frame; fronted subordinates like `When it rains` and medial uses keep
/// SCONJ). Sequenced immediately before the where-marker pass. Guards:
/// nominal- or determiner-headed uses (`The store where`, `the where`)
/// keep their incumbent NOUN (→ SCONJ via the marker rule); `who`/`what`
/// (true nominals) never match — the bit covers only the adverbial four.
/// Known divergence kept: medial non-headed `where` stays the pinned NOUN
/// residual.
fn refine_pos_interrogative_wh_adverbial(
    texts: &[String],
    pos: &mut [Upos],
    flags: &[LexemeFlags],
) {
    for i in 0..texts.len() {
        if !flags[i].is_wh_adverbial() {
            continue;
        }
        if pos[i] == Upos::Noun {
            if i > 0 && matches!(pos[i - 1], Upos::Noun | Upos::Propn | Upos::Det) {
                continue;
            }
            pos[i] = Upos::Adv;
        } else if pos[i] == Upos::Sconj
            && i + 1 < texts.len()
            && pos[i + 1] == Upos::Aux
        {
            pos[i] = Upos::Adv;
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
fn refine_pos_where_marker(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || !flags[i].is_where_word() {
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
fn refine_pos_clausal_after(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Adp || !flags[i].is_after_word() {
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
/// match), WH-initials (`Why`/`Where`/`How` — upgraded to ADV by the
/// interrogative-WH pass before this runs, so they never reach it), and
/// coordination-`or` slots (`Run daily or quit` — CC-disambiguation is its
/// own rule). Same O(n) frame-scan shape as the other refines.
fn refine_pos_final_adverbial(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let word = texts[i].as_str();
        let in_set = flags[i].is_adverb_word();
        let ly = !in_set
            && word.len() > 4
            && word
                .get(word.len() - 2..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ly"));
        if !in_set && !ly {
            continue;
        }
        // Predicative position after AUX is adjectival, never adverbial
        // (`was late`; the category bit covers every closed adverbial, so
        // the old `late`-only spelling check generalizes to the category).
        if in_set && i > 0 && pos[i - 1] == Upos::Aux {
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

/// Temporal `yet` (`She isn't ready yet`). The closed map lexes every
/// `yet` as CCONJ, but sentence-final `yet` after a predicate adjective
/// is the aspectual adverb — a coordinator always has a second finite
/// clause after it. Upgrades CCONJ → ADV only finally (nothing but
/// punctuation follows), with an ADJ host, and with no finite verb ahead
/// before any punctuation. Guards: clausal frames (`rose yet wages
/// stalled`: verb ahead) never match. The (Adj, Adv) arm below lands it.
fn refine_pos_temporal_yet(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Cconj || !flags[i].is_yet_word() {
            continue;
        }
        if pos[i - 1] != Upos::Adj {
            continue;
        }
        if !texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        }) {
            continue;
        }
        let verb_ahead = (i + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .any(|k| pos[k] == Upos::Verb);
        if verb_ahead {
            continue;
        }
        pos[i] = Upos::Adv;
    }
}

/// Sensory linking frame (`Dinner smells great`, `The soup tastes salty`,
/// `The movie sounds boring`) plus the epistemic linking frame (`Something
/// feels wrong`, `Nothing seems right`, `Everyone appears ready`, `This
/// remains uncertain`). Bare-initial linking verbs fall outside every
/// subject rule, and their predicate complements strand as nominal objects.
/// Two guarded upgrades in one frame pass (contracted-be shape): a NOUN
/// linking word (sensory: taste/sound/smell; epistemic: feel/seem/remain/
/// appear; base + -s only — corpus evidence) with a nominal following it →
/// VERB, then an alpha-fallback NOUN directly after a linking VERB → ADJ
/// (the predicate complement, mirroring the be-predicate rule for a host
/// the closed list never pronounces). The epistemic set additionally fires
/// before ADV complements (`seems unlikely` — no nominal exists to anchor
/// the sensory shape) and before VERB complements (`remains uncertain`,
/// where the initial-noun rule already crowned the complement: the pair
/// then ties honestly in the oracle instead of crowning silently). Guards:
/// determiner-led objects (`Taste the soup`, `feel the fabric`) never match
/// either upgrade — transitivity reads through the determiner; plural-noun
/// `remains` (`The remains were buried`, AUX-next) and noun `feel` (`a feel
/// for music`, ADP-next) never match the verb step. Known boundaries: bare
/// transitives (`smells smoke`), attributive sensory nouns (`taste buds`),
/// and seem-to-VP infinitives (`It seems to work`, ADP-next) are
/// positionally identical with no bench instance — lexical
/// subcategorization is its own rule.
fn refine_pos_linking_predicate(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun || !flags[i].is_sensory_verb() {
            continue;
        }
        if texts.len() > i + 1 && matches!(pos[i + 1], Upos::Noun | Upos::Propn) {
            pos[i] = Upos::Verb;
        }
    }
    for i in 0..texts.len() {
        if pos[i] != Upos::Noun || !flags[i].is_epistemic_verb() {
            continue;
        }
        // Epistemic linkers take adjectival complements, so ADV- and
        // VERB-next count alongside nominals (sensory linkers keep the
        // nominal-only shape above — their complements are always nominal
        // in corpus).
        if texts.len() > i + 1
            && matches!(
                pos[i + 1],
                Upos::Noun | Upos::Propn | Upos::Adv | Upos::Verb
            )
        {
            pos[i] = Upos::Verb;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Verb {
            continue;
        }
        if flags[i - 1].is_sensory_verb() || flags[i - 1].is_epistemic_verb() {
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
fn refine_pos_post_comma_verb(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
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

/// Comma-framed participial modifier (`The CEO, smiling, took questions`,
/// `The baby, giggling, grabbed toys`). An -ing word set off by commas is
/// a reduced-relative modifier, not an appositive nominal. Upgrades NOUN
/// → VERB only for -ing forms (allocation-free suffix check, mirroring
/// the copular rule) directly framed by commas on both sides. Guards:
/// bare -ing nominals (`the building`, `helps mornings` — no comma
/// frame), AUX-governed progressives (`are coming` — AUX, not comma,
/// precedes), and coordinated predicate adjectives (`red and fast` — no
/// -ing) never match.
fn refine_pos_comma_participle(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || texts[i - 1] != "," {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() <= 4
            || !word
                .get(word.len() - 3..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
        {
            continue;
        }
        if texts.get(i + 1).is_none_or(|t| t != ",") {
            continue;
        }
        pos[i] = Upos::Verb;
    }
}
/// subject position (`The store where …`). Only forms with corpus evidence;
/// `when`/`that`/`who` lex out of NOUN already and need no guard.
fn is_noun_subject_verb_blocker(flags: LexemeFlags) -> bool {
    flags.is_where_word()
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
fn refine_pos_initial_noun_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    if texts.len() < 3 || pos[2] != Upos::Noun {
        return;
    }
    if pos[0] != Upos::Det || pos[1] != Upos::Noun {
        return;
    }
    if is_noun_subject_verb_blocker(flags[2]) {
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
fn refine_pos_conjoined_clause_verb(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 2..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Noun || pos[i - 2] != Upos::Cconj {
            continue;
        }
        if texts[i].chars().next().is_some_and(|c| c.is_lowercase()) {
            pos[i] = Upos::Verb;
        }
    }
}

/// Whether the lexeme is a finite `be` form (copula host). Matched as a
/// lexeme bit, not a spelling list; sensory and resultative copulas
/// (`taste`, `become`) are their own rules.
fn is_be_form(flags: LexemeFlags) -> bool {
    flags.is_be_verb()
}

/// Inverted copular predicate (`Is lunch ready?`, `is the sky blue`). The
/// be-predicate rule only sees AUX-adjacent complements, and its guard
/// rightly protects a directly-following nominal subject (`Is lunch…`
/// keeps `lunch`) — but when an overt subject stands between be and the
/// complement, the predicate strands as a nominal object. Upgrades an
/// alpha-fallback NOUN to ADJ only after the nearest preceding AUX-tagged
/// be-form with ≥1 nominal strictly between (the overt subject). Guards
/// (mirroring be-predicate/initial-noun): `-ing` participles, capitalized
/// targets, temporal `today` (a bare time adjunct — `are coming today` —
/// never a predicate adjective), and subject-less complements (`is a
/// doctor`, `is the station` — bare determiners never match). Sequenced
/// just before be-predicate; disjoint from it by construction (its
/// direct/bridged shapes have no nominal between be and target).
fn refine_pos_inverted_copular(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
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
        if flags[i].is_today_word() {
            continue;
        }
        let Some(j) = (0..i).rfind(|&j| pos[j] == Upos::Aux && is_be_form(flags[j]))
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
fn refine_pos_be_predicate(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
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
        let direct =
            pos[i - 1] == Upos::Aux && is_be_form(flags[i - 1]) && i - 1 >= 1;
        let bridged = i >= 2
            && flags[i - 1].is_negator()
            && pos[i - 2] == Upos::Aux
            && is_be_form(flags[i - 2])
            && i - 2 >= 1;
        if direct || bridged {
            pos[i] = Upos::Adj;
        }
    }
}

/// Progressive vs. participial-adjective `-ing` after full be (`are coming
/// today`, `were surprising`, `is raining`). The contracted-be rule owns
/// clitic hosts; full be-forms reach here as NOUN (the tagger has no `-ing`
/// verb), and both copular passes above deliberately skip participles (`are
/// coming`, `were surprising` stay for participle handling) — so post-be
/// `-ing` is a dead zone: always NOUN, always confidently wrong
/// (progressives lose their predicate, participial adjectives lose their
/// tree). This rule is that handling. Two licensed readings, split by what
/// follows the participle:
///
/// - An overt complement or adjunct ahead (ADJ/ADV/nominal/DET before any
///   punctuation: `feeling sad`, `raining hard`, `coming today`,
///   `becoming the leader`) licenses the progressive → VERB. The
///   negator bridge mirrors the copular pass (`aren't coming`), and
///   bare `-ing` nominals (`the building`), gerund subjects (`Swimming is
///   fun`), SCONJ/comma-framed participles (`While waiting`, `smiling,
///   took`), and non-be AUX governors never match — no be-AUX immediately
///   before, so the gate never sees them.
/// - Clause-final position (punctuation or end next: `were surprising`,
///   `is raining`) is genuinely ambiguous — a bare progressive and a
///   participial adjective are POS-identical there without lexicon
///   knowledge — so it reads ADJ: the tree goes right (the cop lands, the
///   subject attaches, the participle crowns) and the `-ing` cop-rival in
///   the oracle records the near-tie (→ `RefineReason::Confidence(Ties)` →
///   `AttachmentNearTie`) instead of a confident mistag. The standing
///   `full_be_participial_adjective_stays_non_verb` guard (`assert_ne`
///   verb) keeps holding.
///
/// Known boundaries: `going to + VERB` futures (ADP next — no corpus
/// instance), lexical `-ing` nouns (`It is morning/evening`), and
/// attributive `-ing` + bare nominal (`surprising news`, no corpus
/// instance) stay approximate. Same O(n) frame-scan shape as the other
/// refines.
fn refine_pos_progressive_ing(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let word = texts[i].as_str();
        if word.len() <= 4
            || !word
                .get(word.len() - 3..)
                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
        {
            continue;
        }
        let direct = pos[i - 1] == Upos::Aux && is_be_form(flags[i - 1]);
        let bridged = i >= 2
            && flags[i - 1].is_negator()
            && pos[i - 2] == Upos::Aux
            && is_be_form(flags[i - 2]);
        if !(direct || bridged) {
            continue;
        }
        match pos.get(i + 1) {
            // Clause-final: genuinely ambiguous → ADJ (tree-right) + the
            // oracle tie below flags the POS doubt.
            None | Some(Upos::Punct) => {
                pos[i] = Upos::Adj;
            }
            // Overt complement/adjunct: licensed progressive → VERB.
            Some(n)
                if matches!(
                    n,
                    Upos::Adj | Upos::Adv | Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det
                ) =>
            {
                pos[i] = Upos::Verb;
            }
            // Anything else (going-to futures, auxiliaries, …): leave NOUN.
            _ => {}
        }
    }
}

/// Question-inversion verb (`Did the report arrive`). A NOUN after a
/// do-modal-hosted DET-led nominal subject is the finite verb of an
/// inverted clause. Sequenced after the copular passes (ADJ outputs never
/// match) and keyed on do-modal hosts (be-hosts never match).
fn refine_pos_inversion_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 3..texts.len() {
        if pos[i] != Upos::Noun
            || !matches!(pos[i - 1], Upos::Noun | Upos::Propn)
            || pos[i - 2] != Upos::Det
            || !is_bare_infinitive_host(flags[i - 3])
        {
            continue;
        }
        pos[i] = Upos::Verb;
    }
}

/// Bare verb of a modal-question inversion (`Will this work?`). The sibling
/// inversion pass owns the DET-led-subject shape (`Did the report arrive`);
/// here the subject is a bare nominal or pronoun with no determiner, so
/// that pass's DET anchor never matches and the predicate strands as the
/// sentence root's nominal object. Upgrades a clause-final alpha-fallback
/// NOUN to VERB after a modal/do AUX + nominal subject. Guards: be-hosts
/// (`Is lunch ready` — copular inversion keeps its own dynamics) never
/// match the bare-infinitive-host key, and verbs found by earlier passes
/// (`Can we leave`, `Do you play` via the pronoun-subject pass) never
/// match. A demonstrative subject in this frame (`this`) is pronominal —
/// retagged alongside, since the demonstrative-object pass only fires on
/// bare (complement-less) position. Known boundary: bare inversions with a
/// full-NP subject (`Did John arrive`, no corpus instance) need subject-NP
/// tracking, not this DET-less shape.
fn refine_pos_modal_question_verb(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 2..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        if !texts[i].chars().next().is_some_and(|c| c.is_lowercase()) {
            continue;
        }
        // Clause-final predicate: only trailing punctuation may follow.
        if !texts[i + 1..].iter().all(|t| {
            let w = t.as_str();
            matches!(w, "." | "!" | "?" | ";" | ":" | "," | "—" | "--")
        }) {
            continue;
        }
        if !matches!(
            pos[i - 1],
            Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det
        ) || pos[i - 2] != Upos::Aux
            || !is_bare_infinitive_host(flags[i - 2])
        {
            continue;
        }
        pos[i] = Upos::Verb;
        if pos[i - 1] == Upos::Det && flags[i - 1].is_demonstrative() {
            pos[i - 1] = Upos::Pron;
        }
    }
}

/// First predicate of a clausal coordination (`Prices rose yet wages
/// stalled`). Coordination joins likes: a CCONJ-headed second clause with
/// an overt VERB predicate proves the first predicate verbal too.
fn refine_pos_clausal_first_predicate(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun || !matches!(pos[i - 1], Upos::Noun | Upos::Propn) {
            continue;
        }
        // The CCONJ opening the second clause, before any punctuation.
        let Some(j) = (i + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .find(|&k| pos[k] == Upos::Cconj)
        else {
            continue;
        };
        // Its overt nominal subject, before any punctuation.
        let subject = (j + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .any(|k| matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron));
        // Its overt VERB predicate, before any punctuation.
        let verb = (j + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .any(|k| pos[k] == Upos::Verb);
        if subject && verb {
            pos[i] = Upos::Verb;
        }
    }
}

/// Elliptical second predicate after a conjunction (`ran but fell`,
/// `laughed and cried`). Coordination joins likes: the first conjunct's
/// category decides the second's — but only for a predicate, never an
/// overt subject (`but floods stayed`: floods stays nominal). The two are
/// told apart by what follows: an elliptical predicate is clause-final
/// (no finite verb ahead before any punctuation), while an overt subject
/// is followed by its predicate.
fn refine_pos_conjoined_predicate_agreement(texts: &[String], pos: &mut [Upos], __flags: &[LexemeFlags]) {
    for i in 2..texts.len() {
        if pos[i] != Upos::Noun || pos[i - 1] != Upos::Cconj || pos[i - 2] != Upos::Verb {
            continue;
        }
        let verb_ahead = (i + 1..texts.len())
            .take_while(|&k| pos[k] != Upos::Punct)
            .any(|k| pos[k] == Upos::Verb);
        if verb_ahead {
            continue;
        }
        pos[i] = Upos::Verb;
    }
}

/// Contracted-be disambiguation (`It's raining`, `You're late`, `I'm
/// feeling` vs. `Bell's theorem`). Runs after the lexeme-only tags so it can
/// read neighbor POS. Two guarded upgrades, in one ascending pass so the
/// participle rule sees the freshly classified clitic:
///
/// - a be-clitic (`'s`/`'re`/`'m`) → AUX only when its host is a pronoun
///   (`It`/`You`/`I`/…). After a noun it is the possessive case marker (UD:
///   PART/case) and is left untouched. `'re` already rides the closed-map
///   AUX bit, so in practice this promotes `'s` and `'m`.
/// - an `-ing` word → VERB only when its host is an aux-classified be-clitic
///   (`'s`/`'re`/`'m`) — the progressive participle. Full be-forms (`were
///   surprising`) are excluded: participial adjectives after full be belong
///   to copular handling, and firing there would mistag them.
fn refine_pos_contracted_be(texts: &[String], pos: &mut [Upos], flags: &[LexemeFlags]) {
    for i in 0..texts.len() {
        if pos[i] != Upos::X {
            continue;
        }
        if flags[i].is_be_clitic() && i >= 1 && pos[i - 1] == Upos::Pron {
            pos[i] = Upos::Aux;
        }
    }
    for i in 1..texts.len() {
        if pos[i] != Upos::Noun {
            continue;
        }
        let clitic_aux = pos[i - 1] == Upos::Aux && flags[i - 1].is_be_clitic();
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
pub fn infer_pos(flags: LexemeFlags) -> Upos {
    if flags.is_punct() {
        return Upos::Punct;
    }
    if flags.is_digit() {
        return Upos::Num;
    }
    if flags.is_space() {
        return Upos::Space;
    }
    // Closed function-word categories first, so "the"/"is"/"of" resolve
    // before the alpha checks (and stay DET/AUX/ADP even when title-cased
    // at sentence start). Priority order lives in `closed_funcword_pos`.
    if let Some(pos) = closed_funcword_pos(flags) {
        return pos;
    }
    // Contraction splinters the tokenizer emits: negators are particles
    // (UD: PART), and bare-infinitive hosts double as auxiliaries here —
    // the negator `not` has no nominal or verbal reading, so tagging it
    // here (rather than letting it fall to NOUN and get stolen by the
    // bare-infinitive upgrade: `She did not call` rooted `not`) needs no
    // collision audit. Every host form is AUX in the closed map already
    // except the `n't`-split stubs (`wo`/`ca`), which this line covers
    // (`'re` rides the AUX bit). Possessive/clitic `'s` is genuinely
    // ambiguous (It's vs. Bell's) and is resolved contextually in
    // `refine_pos_contracted_be`, never here.
    if flags.is_negator() {
        return Upos::Part;
    }
    if flags.is_bare_inf_host() {
        return Upos::Aux;
    }
    // A closed set of common verbs gives the parser a predicate to govern
    // nsubj/dobj around. Verbs outside the set are an honest NOUN false
    // negative (open class; the LLM rung is the primary POS source, §8.1).
    if flags.is_verb_word() {
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

    /// Copular-predicate subject gate: `b` is a nominal already governing
    /// its copula (attached just before, since be sits between subject and
    /// predicate — order matters, the cop-Left arm below runs first), and
    /// `s` is a non-existential nominal-or-determiner that can only be its
    /// subject. Shared by the det/compound/nsubj pushes below so the gate
    /// cannot drift between them. Existential `there`/`here` are excluded
    /// (`there` owns the expl arm; `here` keeps repair dynamics as a
    /// documented residual).
    fn copular_subject_frame(
        &self,
        flags: &[LexemeFlags],
        s: usize,
        b: usize,
        pos: &[Upos],
        labels: &DepLabels,
    ) -> bool {
        matches!(
            pos[s],
            Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det
        ) && matches!(pos[b], Upos::Noun | Upos::Propn)
            && !flags[s].is_locative()
            && self.left_children[b]
                .iter()
                .any(|&c| self.labels[c] == labels.cop)
    }

    /// The set of candidate actions for the current state, given per-token
    /// POS. Only label variants plausible for the current `(stack_top, buffer
    /// head)` POS pair are offered, so the oracle rarely ties.
    pub fn candidate_actions(
        &self,
        pos: &[Upos],
        texts: &[String],
        flags: &[LexemeFlags],
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

        // A verbless fallback root is genuinely ambiguous (Define
        // photosynthesis: imperative verb–object vs. NP fragment — the two
        // are POS-identical without lexicon knowledge, see
        // refine_pos_directive_initial). Shared by the rival below and the
        // Reduce block: a nominal root with no VERB or AUX anywhere in the
        // sentence (copular frames and verb-anchored clauses are determined
        // and never match).
        let fallback_root_tie = self.labels[s] == labels.root
            && matches!(ps, Upos::Noun | Upos::Propn)
            && !pos
                .iter()
                .any(|&p| p == Upos::Verb || p == Upos::Aux);
        // A clausal coordinator defers to its predicate (rose yet wages
        // stalled: yet → cc → stalled, NOT wages). Shared by the cc arm
        // below and the Reduce block: when the stack below already holds a
        // complete clause (a verb with its nsubj attached) and a second
        // finite verb follows the buffer, the marker belongs to the
        // upcoming predicate — withholding cc lets the nominal shift into
        // Left-nsubj from its own verb first, and withholding Reduce keeps
        // the marker alive until that verb arrives (Reduce outbids Shift,
        // so without this the deferred marker pops and the pair never
        // meets). Both conditions are required: elliptical frames (ran but
        // fell — no verb ahead) and subject-less verbs (Buy milk and eggs
        // — no nsubj below) keep the immediate reading, as do all nominal
        // coordinations (no verb below at all).
        let clausal_cc_defer = ps == Upos::Cconj
            && self.stack.iter().any(|&t| {
                pos[t] == Upos::Verb
                    && self.left_children[t]
                        .iter()
                        .any(|&c| self.labels[c] == labels.nsubj)
            })
            && (b + 1..texts.len())
                .take_while(|&k| pos[k] != Upos::Punct)
                .any(|k| pos[k] == Upos::Verb);

        // A verbless concessive marker facing a non-nominal complement
        // (Although in pain: SCONJ + ADP with no finite verb before the
        // clause boundary) is underdetermined — the same Track B dynamics
        // as the nominal arm below. Shared by the rival and the Reduce
        // block. The nominal (NOUN/PRON) shape keeps its own inline guard
        // below; this covers only the ADJ/ADP shapes the directive names.
        // Clauses with an overt verb never match.
        let sconj_verbless_tie = ps == Upos::Sconj
            && matches!(pb, Upos::Adj | Upos::Adp)
            && !(b..texts.len())
                .take_while(|&j| pos[j] != Upos::Punct)
                .any(|j| pos[j] == Upos::Verb);

        // The stack top may only be re-headed if it is still unset AND is not
        // the sentence root (the root's head stays -1 forever — attach at most
        // once → acyclic, see module docs / property test 9.10).
        let s_free = self.heads[s] == -1 && self.labels[s] != labels.root;
        let b_free = self.heads[b] == -1 && self.labels[b] != labels.root;

        if s_free {
            match (ps, pb) {
                (Upos::Noun | Upos::Propn | Upos::Pron, Upos::Verb) => {
                    // A comma-framed participle is a modifier, never the
                    // predicate its anchor subjects (`The CEO, smiling`:
                    // CEO must not become nsubj of smiling — the amod-Right
                    // arm below owns that pair). Participles end in -ing
                    // (allocation-free suffix check, mirroring the tagger
                    // rules) directly after a comma; finite verbs and
                    // unframed progressives (`It's raining`) never match.
                    let word = texts[b].as_str();
                    let framed_participle = word.len() > 4
                        && word
                            .get(word.len() - 3..)
                            .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
                        && b > 0
                        && texts[b - 1] == ",";
                    if !framed_participle {
                        out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                    }
                    // An object relativizer with an overt subject (the book
                    // that I bought): a nominative pronoun visibly between
                    // marker and verb proves s is the object, not the
                    // subject. Subject frames (no intervening pronoun) never
                    // offer it, so true subjects never compete.
                    if ps == Upos::Pron
                        && ((s + 1)..b).any(|m| is_nominative_subject(flags[m]))
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
                    // A stranded demonstrative subject of a copular
                    // predicate (This is a Rust project): the det offer
                    // above loses to the gated nsubj weight below once the
                    // copula is attached. Ungated push by design — without
                    // a copula on b it scores 1.0 and loses outright, so
                    // clean determiners never tie.
                    if self.copular_subject_frame(flags, s, b, pos, labels) {
                        out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                    }
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
                    // Existential `there` (There is a problem): the expletive
                    // depends on the copular predicate. Word-gated (only
                    // `there`) and frame-gated on the copula already
                    // attached — locative nominals without one keep the
                    // compound dynamics above.
                    if flags[s].is_there_word()
                        && self.left_children[b]
                            .iter()
                            .any(|&c| self.labels[c] == labels.cop)
                    {
                        out.push(Self::act(ArcEagerMove::Left, labels.expl));
                    }
                    // A nominal subject of a copular predicate (Dogs is a
                    // nominal? no — She is a doctor after be/cop attach):
                    // the compound offer above loses to the gated nsubj
                    // weight below once the copula is attached. Ungated
                    // push by design — without a copula on b it scores 1.0
                    // and loses outright, so `Dogs chase red cars` never
                    // ties (the Track B flagship keeps its dynamics: chase
                    // holds no copula).
                    if self.copular_subject_frame(flags, s, b, pos, labels) {
                        out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                    }
                }
                // The pronominal subject of a copular predicate (She is a
                // doctor, Who is the president): no general (Pron, Noun)
                // arm exists, so this gated offer is the only one. The
                // push itself is gated (unlike the det/compound pushes, it
                // has no higher-scoring sibling to lose to) so clean
                // pronoun-nominal meetings never see a tie.
                (Upos::Pron, Upos::Noun | Upos::Propn) => {
                    if self.copular_subject_frame(flags, s, b, pos, labels) {
                        out.push(Self::act(ArcEagerMove::Left, labels.nsubj));
                    }
                }
                // A pre-nominal designator depends on its head (204 →
                // nummod → status). Pairs with the Right-nummod withhold in
                // the b-free block (without it the number attaches to the
                // preceding nominal before its head arrives) and the
                // head-final root rung (the head must hold the crown, or
                // repair owns the number). Lives in this s-free block
                // because the head may be the pre-designated root, for
                // which the b-free block is skipped. Gated on a
                // punct-free pair, mirroring the compound arm.
                (Upos::Num, Upos::Noun | Upos::Propn) => {
                    if ((s + 1)..b).all(|k| pos[k] != Upos::Punct) {
                        out.push(Self::act(ArcEagerMove::Left, labels.nummod));
                    }
                }
                (Upos::Aux, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.aux));
                }
                // The copula depends on its predicate (fee is low: is → cop
                // → low). Pairs with the be-predicate tagger rule above.
                (Upos::Aux, Upos::Adj) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.cop));
                    // A copular complement with a DET-led nominal between
                    // be and the predicate (`is the water cycle`, `is a
                    // Rust project`) is information-theoretically
                    // ambiguous: the same shape reads as an overt-subject
                    // predicate adjective (`is the sky blue`) or a modified
                    // predicate nominal — and only lexicon knowledge (which
                    // adjective-hood needs) splits them. Offer a `dep`
                    // rival at cop parity (95) so `best_with_margin`
                    // records a near-tie (→ `RefineReason::Confidence(Ties)`
                    // → `AttachmentNearTie` downstream) instead of a
                    // confident cop. The cop arm above is unconditional, so
                    // the rival can only tie, never outbid (pushed after,
                    // loses ties by stable order) — heads/labels unchanged,
                    // only the margin drops (Track B: flag, don't guess).
                    // Bare-subject frames (`Is lunch ready`, no DET between)
                    // and direct complements (`Your fee is low`) never
                    // match, so clean copulars stay tie-free.
                    if is_be_form(flags[s])
                        && (s + 1..b).any(|k| pos[k] == Upos::Det)
                        && (s + 1..b).any(|k| {
                            matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                        })
                    {
                        out.push(Self::act(ArcEagerMove::Left, labels.dep));
                    }
                    // A clause-final `-ing` participle after be (`were
                    // surprising`, `is raining`) is information-theoretically
                    // ambiguous: the same shape reads as a participial
                    // adjective or a bare progressive — and only lexicon
                    // knowledge (which verbhood needs) splits them. Offer a
                    // `dep` rival at cop parity (95, scored by the weight
                    // arm's ungated (Aux, Adj) branch) so `best_with_margin`
                    // records a near-tie (→ `RefineReason::Confidence(Ties)`
                    // → `AttachmentNearTie`) instead of a confident cop. Same
                    // Track B dynamics as the DET-tie above: pushed after,
                    // loses ties by stable order — heads/labels unchanged,
                    // only the margin drops. Gated on a be-form stack top
                    // and the `-ing` suffix, so direct complements (`Your
                    // fee is low`), inverted frames (`Is lunch ready`), and
                    // non-participial predicates never match — clean
                    // copulars stay tie-free.
                    if is_be_form(flags[s]) {
                        let word = texts[b].as_str();
                        if word.len() > 4
                            && word
                                .get(word.len() - 3..)
                                .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
                        {
                            out.push(Self::act(ArcEagerMove::Left, labels.dep));
                        }
                    }
                }
                // The copula depends on a nominal predicate (She is a
                // doctor: is → cop → doctor). Pairs with the pick_root
                // predicate-nominal rule (the predicate must hold the
                // crown, or repair owns the heads). Gated on a be-form
                // word with a determiner between — possessive `has a car`,
                // causative `let the dead`, and progressive `is raining`
                // never match (no DET or no be-form) — and never in
                // Where-initial interrogatives (question-02/08 pin
                // be-as-root; the Where convention decision owns those).
                // A predicative adjective later in the clause (`Is the
                // report ready`) withholds the arm: there the nominal is
                // the subject, and the ADJ crowns — offering cop would
                // mishead be onto the subject (polar uas regression).
                (Upos::Aux, Upos::Noun | Upos::Propn) => {
                    if is_be_form(flags[s])
                        && ((s + 1)..b).any(|k| pos[k] == Upos::Det)
                        && !(b + 1..texts.len())
                            .take_while(|&k| pos[k] != Upos::Punct)
                            .any(|k| pos[k] == Upos::Adj)
                        && !flags
                            .get(self.sent_start)
                            .is_some_and(|f| f.is_where_word())
                    {
                        out.push(Self::act(ArcEagerMove::Left, labels.cop));
                    }
                }
                // A comment clause depends on its matrix predicate (As you
                // know, your fee is low: know → parataxis → low). The
                // clause faces its predicate leftward (the predicate
                // follows), so unlike the (Verb, Verb) juxtaposition arm
                // this attaches Left. Gated on a clause boundary plus an
                // intervening nominal subject — bare (Verb, Adj) pairs
                // (sensory complements: smells great, no boundary) never
                // compete. Needs the subordinate root-skip (the predicate
                // must hold the crown, or repair owns the heads).
                (Upos::Verb, Upos::Adj) => {
                    let boundary = ((s + 1)..b).any(|k| pos[k] == Upos::Punct);
                    let subject = ((s + 1)..b).any(|k| {
                        matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                    });
                    if boundary && subject {
                        out.push(Self::act(ArcEagerMove::Left, labels.parataxis));
                    }
                }
                (Upos::Adv, Upos::Verb) => {
                    out.push(Self::act(ArcEagerMove::Left, labels.advmod));
                }
                // An interrogative adverbial on a crowned be-predicate
                // (`Where is the station`: Where → advmod → is). The generic
                // arm above only sees verbal predicates; this one is gated
                // on the WH bit, a be-form head, and the head holding the
                // sentence crown — non-root be (`is` in `Why is the sky
                // blue`, `does` in `How does photosynthesis work`) keeps
                // repair dynamics, and the predicate adjective or verb owns
                // those attachments. Sole offer on its pairs (no nsubj/det
                // arm matches ADV tops), so it wins outright at fallthrough
                // weight with no tie.
                (Upos::Adv, Upos::Aux) => {
                    if flags[s].is_wh_adverbial()
                        && is_be_form(flags[b])
                        && self.labels[b] == labels.root
                    {
                        out.push(Self::act(ArcEagerMove::Left, labels.advmod));
                    }
                }
                // An interrogative adverbial on a predicate adjective
                // (`Why is the sky blue`: Why → advmod → blue). WH-gated so
                // plain adverbials keep repair dynamics; sole offer, no tie.
                (Upos::Adv, Upos::Adj) => {
                    if flags[s].is_wh_adverbial() {
                        out.push(Self::act(ArcEagerMove::Left, labels.advmod));
                    }
                }
                (Upos::Cconj, _) => {
                    // Defers under `clausal_cc_defer` above (withholds the
                    // arc; the Reduce block withholds the pop).
                    if !clausal_cc_defer {
                        out.push(Self::act(ArcEagerMove::Left, labels.cc));
                    }
                }
                // The negator depends on the verb it negates (Don't help:
                // n't → neg → help). Without this arm the PART splinter sits
                // on the stack and every pre-verbal token falls to repair-dep.
                // Negators equally negate predicate adjectives (isn't
                // ready): same arm, same head-local shape.
                (Upos::Part, Upos::Verb | Upos::Adj) => {
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
                // A subordinator facing a non-nominal with no finite verb
                // before the clause boundary (Although in pain, …): the
                // marker's attachment is underdetermined — no predicate
                // exists to host `mark`. Same Track B dynamics as the
                // nominal arm above: offer a `dep` rival at Shift parity so
                // `best_with_margin` records a near-tie (→
                // `RefineReason::Confidence(Ties)` → `AttachmentNearTie`
                // downstream) instead of a confident Shift/Reduce. Clauses
                // with an overt verb never match. Shift still wins ties by
                // stable order, so heads/labels are unchanged — only the
                // margin drops (Track B: flag, don't guess). Needs the
                // verbless SCONJ Reduce wait below (shared condition above;
                // Reduce outbids Shift, so without it the tie never forms).
                (Upos::Sconj, Upos::Adj | Upos::Adp) => {
                    if sconj_verbless_tie {
                        out.push(Self::act(ArcEagerMove::Left, labels.dep));
                    }
                }
                _ => {}
            }
        }

        if b_free {
            // A verbless fallback root is genuinely ambiguous (Define
            // photosynthesis: imperative verb–object vs. NP fragment — the
            // two are POS-identical without lexicon knowledge, see
            // refine_pos_directive_initial). Offer a `dep` rival at Shift
            // parity so best_with_margin records a near-tie (→
            // RefineReason::Confidence(Ties) → AttachmentNearTie downstream)
            // instead of a confident Shift. Gated by `fallback_root_tie`
            // above (nominal root, no VERB or AUX in the sentence — copular
            // frames and verb-anchored clauses never match); the Reduce
            // block withholds the pop under the same condition (Reduce
            // outbids Shift, so without this the tie never forms). Shift
            // still wins ties by stable order, so heads/labels are
            // unchanged — only the margin drops (Track B: flag, don't
            // guess).
            if fallback_root_tie {
                out.push(Self::act(ArcEagerMove::Right, labels.dep));
            }
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
                        // Temporal "today" heads an adverbial modifier, never
                        // a direct object (Send the invoice today) — the
                        // advmod offer here owns it. Word-gated like the yet
                        // rule; daily/later shapes never match.
                        if flags[b].is_today_word() {
                            out.push(Self::act(ArcEagerMove::Right, labels.advmod));
                        } else {
                            out.push(Self::act(ArcEagerMove::Right, labels.dobj));
                        }
                        out.push(Self::act(ArcEagerMove::Right, labels.nsubj));
                        // A ditransitive recipient (Give me the report, Show
                        // me the sales report): a bare pronoun directly after
                        // the verb with a determiner-led nominal after it is
                        // the indirect object — the DET-led nominal takes
                        // dobj below. This arm carries its own frame
                        // condition (dobj above matches the pair too, so the
                        // block gate is not enough); the 105 weight below
                        // outbids dobj (100) only inside it. Pronouns before
                        // prepositions (Remind me at noon), bare nominals
                        // (Help them win), and adverbs (Call me later) keep
                        // the incumbent dobj.
                        if pb == Upos::Pron
                            && b + 2 < texts.len()
                            && pos[b + 1] == Upos::Det
                            && matches!(pos[b + 2], Upos::Noun | Upos::Propn)
                        {
                            out.push(Self::act(ArcEagerMove::Right, labels.iobj));
                        }
                    }
                }
                // A be-AUX governs its nominal complement rightward in a
                // Where-initial be-question (Where is the station: station →
                // dobj → is, per question-02/08). The (Verb, nominal) arm
                // above never sees the pair (be is AUX, not VERB), so the
                // complement strands into repair-dep with the right head
                // and the wrong label. Gated on the clause opening with
                // Where (scan back to the previous punctuation boundary) and
                // a be-form stack top, so copular declaratives (`She is a
                // doctor`), inverted predicates (`Is lunch ready`), and
                // modal AUX frames (`Can you help`) keep incumbent
                // dynamics; pronouns stay out (subjects, never objects).
                // Needs the Where-be root rung above (the AUX must hold the
                // crown, or repair owns the heads).
                (Upos::Aux, Upos::Noun | Upos::Propn) => {
                    let clause_start = (0..s)
                        .rev()
                        .find(|&k| pos[k] == Upos::Punct)
                        .map_or(0, |k| k + 1);
                    if flags
                        .get(clause_start)
                        .is_some_and(|f| f.is_where_word())
                        && is_be_form(flags[s])
                        && ((s + 1)..b).all(|k| pos[k] != Upos::Punct)
                    {
                        out.push(Self::act(ArcEagerMove::Right, labels.dobj));
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
                        let has_subject = ((marker + 1)..b).any(|k| {
                            matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                        });
                        if has_subject {
                            if flags[marker].is_subord_complement() {
                                out.push(Self::act(ArcEagerMove::Right, labels.ccomp));
                            } else if flags[marker].is_subord_adverbial() {
                                out.push(Self::act(ArcEagerMove::Right, labels.advcl));
                            }
                        }
                    } else if let Some(marker) =
                        (s + 1..b).rfind(|&m| pos[m] == Upos::Cconj)
                    {
                        // A coordinated clause verb governed by the first
                        // predicate (passed but floods stayed: stayed → conj
                        // → passed). Same shape as the subordinate arm with
                        // the coordinator class instead: the innermost
                        // conjunction decides, and an overt nominal subject
                        // must stand between marker and verb — or, failing
                        // that, the matrix verb must visibly govern its own
                        // subject (ran but fell: no overt subject, but ran
                        // already governs She — the classic elliptical
                        // coordination sharing its subject). Needs the Verb
                        // CCONJ wait below (the matrix verb must survive the
                        // conjunction) and the cc-deferral above (the marker
                        // must survive its nominal to reach this verb).
                        // Punctuated pairs keep the incumbent parataxis
                        // reading (first branch); SCONJ pairs the ccomp/advcl
                        // reading (second branch).
                        let has_subject = ((marker + 1)..b).any(|k| {
                            matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                        });
                        let shared_subject = !has_subject
                            && self.left_children[s]
                                .iter()
                                .any(|&c| self.labels[c] == labels.nsubj);
                        if has_subject || shared_subject {
                            out.push(Self::act(ArcEagerMove::Right, labels.conj));
                        }
                    } else if ((s + 1)..b).any(|k| texts[k] == ",")
                        && !((s + 1)..b).any(|k| {
                            matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron)
                        })
                    {
                        // Asyndetic coordination (Work hard, play fair):
                        // juxtaposed predicates sharing a null subject
                        // conjoin. An overt nominal between makes it
                        // parataxis (first branch above — "He cooks, she
                        // cleans"); semicolons and periods are sentence
                        // boundaries owned by parataxis and Break, so only
                        // commas qualify.
                        out.push(Self::act(ArcEagerMove::Right, labels.conj));
                    } else if self.labels[s] == labels.root
                        && ((s + 1)..b).all(|k| {
                            pos[k] != Upos::Punct
                                && pos[k] != Upos::Sconj
                                && pos[k] != Upos::Cconj
                        })
                    {
                        // A root-governed bare complement (Let the dead go:
                        // go → ccomp → Let). Permissive/causative crowns take
                        // bare infinitives with no marker, boundary, or
                        // conjunction between — any of those owns its own
                        // branch above (parataxis, ccomp/advcl, conj), and
                        // non-root matrices (called left: called never
                        // roots) keep the incumbent dynamics.
                        out.push(Self::act(ArcEagerMove::Right, labels.ccomp));
                    } else {
                        // An unanchored verb–verb embedding (I think she
                        // knows he left: left has no marker, no boundary,
                        // and a non-root matrix): the second verb's
                        // attachment is underdetermined — the licensed arms
                        // above own every determined shape (parataxis,
                        // marked ccomp/advcl, coordination, root bare
                        // complement), so reaching here means no signal
                        // licenses the pair. Offer a `dep` rival at Shift
                        // parity so `best_with_margin` records a near-tie
                        // (→ `RefineReason::Confidence(Ties)` →
                        // `AttachmentNearTie` downstream) instead of a
                        // confident Shift. Scores 1.0 like Shift, so it can
                        // never outbid a licensed arc or Reduce — Shift
                        // still wins ties by stable order, so heads/labels
                        // are unchanged; only the margin drops (Track B:
                        // flag, don't guess).
                        out.push(Self::act(ArcEagerMove::Right, labels.dep));
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
                            && is_relative_marker(flags[m])
                            && matches!(pos[m - 1], Upos::Noun | Upos::Propn)
                    }) {
                        out.push(Self::act(ArcEagerMove::Right, labels.relcl));
                    }
                    // A comma-framed participle modifies its anchor
                    // (smiling → amod → CEO), same post-nominal-modifier
                    // shape as relcl. Gated on the -ing form directly after
                    // a comma (the tagger pass admits only comma-framed
                    // -ing, and the Left nsubj arm above stands down for
                    // exactly that shape), so true subjects and unframed
                    // progressives never compete. Un-gated on weight by
                    // design — the candidate gate above is the gate.
                    let word = texts[b].as_str();
                    if b > 0
                        && texts[b - 1] == ","
                        && word.len() > 4
                        && word
                            .get(word.len() - 3..)
                            .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
                    {
                        out.push(Self::act(ArcEagerMove::Right, labels.amod));
                    }
                }
                // A numeric modifier follows its nominal head on the buffer
                // (invoice 1001, flight 204: number → nummod → invoice).
                // Numbers never govern leftward, so no Left arm competes;
                // the bare-noun fallback (repair-dep with the right head)
                // is what this replaces, deterministically. A pre-nominal
                // designator (flight 204 status) belongs to the FOLLOWING
                // head instead — withhold nummod when a nominal stands
                // directly after the number (no punctuation between), so
                // the number shifts into the Left-nummod arm in the s-free
                // block below (Left arms must live there: the head may be
                // the pre-designated root, for which the b-free block is
                // skipped); number-final frames (invoice 1001) fire as before.
                (Upos::Noun | Upos::Propn, Upos::Num) => {
                    let prenominal = b + 1 < texts.len()
                        && matches!(pos[b + 1], Upos::Noun | Upos::Propn | Upos::Pron)
                        && (s + 1..b + 2).all(|k| pos[k] != Upos::Punct);
                    if !prenominal {
                        out.push(Self::act(ArcEagerMove::Right, labels.nummod));
                    }
                }
                (Upos::Verb, Upos::Adv) => {
                    out.push(Self::act(ArcEagerMove::Right, labels.advmod));
                }
                // A copular predicate takes its trailing adverbial rightward
                // (late again: again → advmod → late; ready yet, once `yet`
                // reads ADV). No arm offers (Adj, Adv) today, so they strand
                // in repair-dep with the right head.
                (Upos::Adj, Upos::Adv) => {
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
        // (No comma wait is needed or possible here: comma buffers take the
        // Punct early-return above, so a comma-buffered Reduce never reaches
        // this block. The parenthetical anchor strands one step later — see
        // the punct-child Reduce below.)
        //
        // Reduce normally needs a childless top (popping would orphan the
        // children). Trailing punctuation is exempt: an attached (headed,
        // non-root) top whose right children are ALL punctuation may still
        // pop — the punctuation is a trailing ornament, not an open phrase.
        // Without this, an appositive carrying its closing comma (doctor in
        // "My brother, a doctor, lives here") can never Reduce: it buries
        // the anchor, the Left-nsubj arm never meets its pair, and the
        // anchor strands in repair-dep. Scoped to headed non-root tops with
        // punct-only right children — unheaded tops (open phrases like the
        // subordinate predicate in "If it rains, stay") and lexical children
        // keep the incumbent protection.
        if (self.right_children[s].is_empty()
            || (self.heads[s] != -1
                && self.labels[s] != labels.root
                && self.right_children[s]
                    .iter()
                    .all(|&c| pos[c] == Upos::Punct)))
            && ps != Upos::Adp
            && !matches!(pb, Upos::Aux | Upos::Part)
            && !(self.heads[s] == -1
                && ps == Upos::Verb
                && pb == Upos::Det
                && !flags[s].is_verb_word()
            )
            && !(ps == Upos::Sconj && matches!(pb, Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det))
            // And a verbless concessive marker facing a non-nominal holds
            // for its tie (Although [in]: popping the marker strands it
            // before the tie pair forms — Reduce outbids the Shift/dep
            // parity, so without this the margin never drops. Shared
            // condition above; verbless-gated, so clauses with an overt
            // verb keep the old dynamics).
            && !sconj_verbless_tie
            && !(ps == Upos::Noun && pb == Upos::Det)
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && b > 0
                && is_relative_marker(flags[b])
                && matches!(pos[b - 1], Upos::Noun | Upos::Propn))
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && pb == Upos::Pron
                && b >= 2
                && is_relative_marker(flags[b - 1])
                && matches!(pos[b - 2], Upos::Noun | Upos::Propn))
            && !(ps == Upos::Aux
                && matches!(pb, Upos::Pron | Upos::Noun | Upos::Propn | Upos::Det))
            && !(ps == Upos::Pron
                && pb == Upos::Pron
                && b >= 2
                && is_relative_marker(flags[b - 1])
                && matches!(pos[b - 2], Upos::Noun | Upos::Propn))
            && !(ps == Upos::Verb && pb == Upos::Sconj)
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && pb == Upos::Cconj
                && self.left_children[s].is_empty()
                && self.right_children[s].is_empty())
            // And a matrix verb waits for its coordinated clause (passed
            // [but]): popping on the conjunction strands the first
            // predicate before the clause verb arrives, and the conj arm
            // above never meets its pair. Scoped by lookahead to genuine
            // second clauses (a finite verb follows the conjunction before
            // any punctuation) — elliptical frames (ran but fell: no verb
            // ahead) keep the old dynamics and the honest dep fallback.
            && !(ps == Upos::Verb
                && pb == Upos::Cconj
                && (b + 1..texts.len())
                    .take_while(|&k| pos[k] != Upos::Punct)
                    .any(|k| pos[k] == Upos::Verb))
            // And a deferred clausal coordinator waits for its predicate
            // (rose yet [wages]: popping the marker strands it before the
            // clause verb arrives — shared condition above).
            && !clausal_cc_defer
            // And a matrix verb waits out a possessive determiner (ate [my]:
            // popping on the determiner strands the verb before its object
            // arrives — the determiner heads its noun and vacates, so the
            // dobj arm meets its pair. Scoped to possessives (which
            // obligatorily head nominals); articles keep the existing
            // dynamics, complementizer `that` and pronouns never match.
            && !(ps == Upos::Verb
                && pb == Upos::Det
                && is_possessive_determiner(flags[b])
            )
            // And a verbless fallback root holds for its tie (Define
            // [photosynthesis]: popping the root strands it before the tie
            // pair forms — Reduce outbids the Shift/dep parity, so without
            // this the margin never drops. Shared condition above).
            && !fallback_root_tie
            // And a nominal head waits out a pre-nominal designator
            // (flight [204]: popping on the number strands the head before
            // the designator's own head arrives — the withhold above leaves
            // Shift as the only arc, but Reduce outbids Shift, so without
            // this the head falls to repair-dep instead of compound.
            // Shared shape with the withhold; number-final frames (invoice
            // [1001]) keep the old dynamics).
            && !(matches!(ps, Upos::Noun | Upos::Propn)
                && pb == Upos::Num
                && b + 1 < texts.len()
                && matches!(pos[b + 1], Upos::Noun | Upos::Propn | Upos::Pron)
                && (s + 1..b + 2).all(|k| pos[k] != Upos::Punct))
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
                        if pos[s] == Upos::Adj
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
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
                        if pos[s] == Upos::Part
                            && matches!(pos[b], Upos::Verb | Upos::Adj)
                        {
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
                        // Adjectival predicates (fee is low) and nominal
                        // predicates with the copula attached (is → cop →
                        // doctor): b already governs its copula in the
                        // latter case, so the nominal can only be the
                        // predicate. Ungated here by design — the candidate
                        // arms above are the gates (mirrors obj-Left 105).
                        if pos[s] == Upos::Aux
                            && (pos[b] == Upos::Adj
                                || (matches!(pos[b], Upos::Noun | Upos::Propn)
                                    && state.left_children[b]
                                        .iter()
                                        .any(|&c| state.labels[c] == labels.cop)))
                        {
                            95.0
                        } else {
                            5.0
                        }
                    }
                    // The nominal-predicate subject (She → nsubj → doctor):
                    // b already governs its copula, so the nominal on the
                    // stack can only be its subject. Ungated here by design
                    // — the candidate arms above are the gates. Without a
                    // copula on b this scores 1.0 (the old fallthrough), so
                    // clean pairs never tie.
                    l if l == labels.nsubj
                        && matches!(pos[b], Upos::Noun | Upos::Propn)
                        && matches!(
                            pos[s],
                            Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det
                        )
                        && state.left_children[b]
                            .iter()
                            .any(|&c| state.labels[c] == labels.cop) =>
                    {
                        100.0
                    }
                    // Existential `there` (There → expl → problem): same
                    // gate shape as the subject above, expletive weight.
                    l if l == labels.expl
                        && matches!(pos[s], Upos::Noun | Upos::Propn)
                        && matches!(pos[b], Upos::Noun | Upos::Propn)
                        && state.left_children[b]
                            .iter()
                            .any(|&c| state.labels[c] == labels.cop) =>
                    {
                        95.0
                    }
                    // Comment-clause parataxis (know → parataxis → low).
                    // Ungated here by design — only the arm above offers
                    // Left parataxis, gated on boundary + subject.
                    l if l == labels.parataxis
                        && pos[s] == Upos::Verb
                        && pos[b] == Upos::Adj =>
                    {
                        90.0
                    }
                    // Pre-nominal designator (204 → nummod → status): same
                    // determiner-level confidence as post-nominal nummod.
                    // Ungated here by design — the Left arm in the s-free
                    // block is the only offerer.
                    l if l == labels.nummod => {
                        if pos[s] == Upos::Num
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            90.0
                        } else {
                            5.0
                        }
                    }
                    // The copular-complement category tie (is → dep →
                    // cycle, at cop parity): the DET-led-nominal frame is
                    // genuinely ambiguous (overt subject + predicate
                    // adjective vs. modified predicate nominal), so the
                    // rival must score exactly the incumbent cop weight —
                    // never above (it would flip the head) and never at
                    // Shift parity (it would never tie). Ungated here by
                    // design — the gated arm above is the only (Aux, Adj)
                    // dep offerer; every other Left-dep rival keeps its
                    // Shift-parity 1.0 fallthrough below.
                    l if l == labels.dep => {
                        if pos[s] == Upos::Aux && pos[b] == Upos::Adj {
                            95.0
                        } else {
                            1.0
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
                    // The be-question complement (station → dobj → is):
                    // aux-family confidence, below verbal dobj (100) so a
                    // verb on the stack always keeps its own dynamics.
                    // Ungated here by design — the Where-be candidate arm
                    // above is the only offerer, so this weight only ever
                    // ranks Where-be pairs.
                    l if l == labels.dobj && pos[s] == Upos::Aux => {
                        if matches!(pos[b], Upos::Noun | Upos::Propn) {
                            90.0
                        } else {
                            10.0
                        }
                    }
                    // The ditransitive recipient outbids the direct object
                    // (100): inside the gated frame above, the pronoun
                    // between a verb and a DET-led nominal is the indirect
                    // object, and attaching it as dobj would mislabel it
                    // while stranding the true direct object. Ungated here
                    // by design — the candidate arm above is the gate, so
                    // this weight only ever ranks ditransitive pairs
                    // (mirrors the obj-Left 105).
                    l if l == labels.iobj && pos[s] == Upos::Verb => {
                        if pos[b] == Upos::Pron {
                            105.0
                        } else {
                            10.0
                        }
                    }
                    // A numeric modifier depends on its nominal head
                    // (1001 → nummod → invoice). Determiner-level
                    // confidence; the candidate arm above is the only
                    // offerer, so this weight only ranks numeral pairs.
                    l if l == labels.nummod => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn) && pos[b] == Upos::Num {
                            90.0
                        } else {
                            5.0
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
                        } else if pos[s] == Upos::Verb
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            // Temporal-noun adverbial (today → advmod →
                            // Send). Ungated here by design — only the
                            // today arm above offers (verb, nominal)
                            // advmod.
                            75.0
                        } else if pos[s] == Upos::Adj && pos[b] == Upos::Adv {
                            // Trailing adverbial of a copular predicate
                            // (again → advmod → late). Ungated here by
                            // design — only the arm above offers (Adj, Adv).
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
                    // Post-nominal participial modifier (smiling → amod →
                    // CEO). Ungated here by design — only the
                    // comma-participle arm above offers (nominal, verb)
                    // amod, and the Left-nsubj arm stands down for exactly
                    // that shape.
                    l if l == labels.amod
                        && matches!(pos[s], Upos::Noun | Upos::Propn)
                        && pos[b] == Upos::Verb =>
                    {
                        85.0
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
                    // compound/appos pair it disjoins from by gate — and the
                    // coordinated clause verb on its matrix predicate
                    // (stayed → conj → passed), gated by the CCONJ arm
                    // above. Ungated here by design — the candidate arms
                    // above are the gates.
                    l if l == labels.conj => {
                        if matches!(pos[s], Upos::Noun | Upos::Propn)
                            && matches!(pos[b], Upos::Noun | Upos::Propn)
                        {
                            60.0
                        } else if pos[s] == Upos::Verb && pos[b] == Upos::Verb {
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
    pub nummod: u64,
    pub expl: u64,
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
            nummod: intern("nummod"),
            expl: intern("expl"),
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
        let flags: Vec<LexemeFlags> = (0..doc.len()).map(|i| doc.token(i).lexeme.flags).collect();
        let mut pos: Vec<Upos> = flags.iter().map(|&f| infer_pos(f)).collect();
        // Contextual pass over the lexeme-only tags: determiner-led nominals
        // colliding with the closed verb list. Runs first so only infer_pos
        // verbs are candidates and every VERB-upgrade below reads corrected
        // tags.
        refine_pos_det_closed_verb(&texts, &mut pos, &flags);
        // Contextual pass over the lexeme-only tags: bare infinitive after a
        // do-modal host. Runs before root-picking so the oracle sees verbs.
        refine_pos_bare_infinitive(&texts, &mut pos, &flags);
        // Bare-object pass: closed-verb s-form after a VERB at sentence end
        // (`answer calls`) is a plural object noun — English morphosyntax
        // forbids a finite s-form after a bare verb. Sequenced after the
        // infinitive pass (the VERB host must be visible); the existing
        // dobj arm lands it. Guards: pronoun hosts (`she calls` — finite),
        // non-s-forms (`called left` — matrix root), and non-final
        // positions never match.
        refine_pos_bare_object_noun(&texts, &mut pos, &flags);
        // Contracted-be pass: pronoun-hosted 's → AUX, then clitic-hosted
        // -ing participle → VERB. Sequenced after the infinitive pass; the
        // two govern disjoint contexts (aux-host vs. clitic-host).
        refine_pos_contracted_be(&texts, &mut pos, &flags);
        // Directive pass: sentence-initial NOUN + DET + NOUN → VERB.
        // Sequenced last; disjoint from the aux/clitic triggers above.
        refine_pos_directive_initial(&texts, &mut pos, &flags);
        // Bare-ed-transitive pass: initial nominal + -ed word + DET-led
        // object (`John opened the door`, `Anna finished her lunch`) is a
        // past-tense transitive clause. Past morphology plus the
        // transitive frame identifies it: -ed adjectives are attributive
        // (DET-led) or predicative after linking/be (AUX prev) — never
        // bare-initial-subject position. Sequenced after the directive
        // pass (disjoint: that pass needs DET+NOUN at 1-2, this one an
        // -ed word); the pre-existing Verb–Det wait (which only excludes
        // closed-list verbs) holds the object slot, and dobj lands it.
        // Guards: conjunctions (`Dogs and cats`), pronouns (`Who
        // called`), non- -ed forms (`Birds sing`, `Translate hello`),
        // non-determiner thirds (`Grades dropped yet`, `NASA launched
        // HTML5`), and titlecase verbs never match.
        refine_pos_bare_ed_transitive(&texts, &mut pos, &flags);
        // Imperative pass with a non-determiner complement (`Remind me`,
        // `Translate hello to French`, `Explain Bell's theorem`): the
        // closed-list gap the directive pass (DET-led objects only) leaves
        // behind. Sequenced after the bare-ed pass so its VERB output feeds
        // the verbless-clause gate (`Anna finished` is verbal by now and
        // never matches), and after the demonstrative pass so retagged
        // object pronouns (`Translate this`) take the pronoun frame;
        // targets (sentence-initial NOUN) are disjoint from both, and the
        // DET-second shape stays with the directive pass.
        refine_pos_demonstrative_object(&texts, &mut pos, &flags);
        refine_pos_imperative_non_det_object(&texts, &mut pos, &flags);
        // Discourse-initial imperative (`Please confirm`, `Never mind`):
        // the marker frame the imperative pass cannot see (its seconds are
        // verbs, never pronouns/PPs). Sequenced after it (disjoint:
        // marker-initial words never match the bare-noun frames) so the
        // crowned verb feeds the standard dobj dynamics below.
        refine_pos_discourse_initial_verb(&texts, &mut pos, &flags);
        // Get-imperative pass: sentence-initial AUX-tagged get/got with a
        // nominal complement → VERB (`Get me a coffee`). Sequenced after
        // the discourse pass (disjoint: AUX-initial targets match none of
        // the bare-noun frames); the crowned verb feeds the standard
        // ditransitive dynamics below.
        refine_pos_get_imperative(&texts, &mut pos, &flags);
        // Attributive -ly (`quarterly sales`): adverbial by default, but
        // adjectival directly before a nominal head. Sequenced after the
        // discourse pass (Kindly is ADV by now) and after every ADV
        // tagger, so only stranded NOUNs are candidates.
        refine_pos_attributive_ly(&texts, &mut pos, &flags);
        // Pronoun-subject pass: finite verb after a nominative pronoun.
        // Disjoint from all above (PRON prev never matches aux/clitic/
        // initial triggers).
        refine_pos_pronoun_subject_verb(&texts, &mut pos, &flags);
        // That-relative pass: nominal-headed that → PRON, then the NOUN
        // after that- PRON → VERB. Sequenced after the pronoun pass (whose
        // nominative list never pronounces that) and before the
        // relative-matrix pass, whose VERB-host gate reads the clause verbs
        // upgraded here (cried → slept). Disjoint targets from both.
        refine_pos_that_relative(&texts, &mut pos, &flags);
        // Interrogative WH-adverbial pass: clause-initial where/why/how
        // (and AUX-led `when`) → ADV. Sequenced immediately before the
        // where-marker pass; the two gates mirror each other (nominal-prev
        // vs. not) so no token can match both.
        refine_pos_interrogative_wh_adverbial(&texts, &mut pos, &flags);
        // Where-marker pass: nominal-headed where → SCONJ so the existing
        // mark arm fires. Sequenced with the other frame passes; targets
        // (NOUN-where) are disjoint from every verb/ADJ upgrade, and no
        // refine reads SCONJ positionally (the initial-noun blocker keys on
        // the word, order-free).
        refine_pos_where_marker(&texts, &mut pos, &flags);
        // Clausal-after pass: ADP after + subject + verb → SCONJ so the
        // existing mark arm fires. Sequenced after the pronoun-subject pass
        // (clause verbs upgraded there are the VERB host) with the other
        // SCONJ-frame passes; nominal complements never match, so prep/pobj
        // frames are untouched.
        refine_pos_clausal_after(&texts, &mut pos, &flags);
        // Final-adverbial pass: closed time/manner set + -ly finals →
        // ADV. Targets are disjoint from the comma-adverbial pass (which
        // owns comma -ly with its clause-edge host guard; this pass skips
        // comma -ly), and ADV outputs feed no verb/ADJ upgrade.
        refine_pos_final_adverbial(&texts, &mut pos, &flags);
        // Linking-predicate pass: bare-initial sensory verb → VERB, then
        // the NOUN after a sensory VERB → ADJ. Sequenced with the frame
        // passes; targets (sensory words, their complements) are disjoint
        // from every relativizer/adverbial trigger, and ADJ outputs feed
        // nothing upstream of the be-predicate pass.
        refine_pos_linking_predicate(&texts, &mut pos, &flags);
        // Post-comma pass: clause-initial NOUN after a parenthetical
        // boundary → VERB. Sequenced after the linking pass (sensory verbs
        // read first where both could apply — disjoint in practice: no
        // bench sensory verb sits post-comma) and before the
        // relative-matrix pass (disjoint: matrix needs a marker frame).
        refine_pos_post_comma_verb(&texts, &mut pos, &flags);
        // Participial-modifier pass: comma-framed -ing NOUN → VERB (The
        // CEO, smiling, took questions). The -ing morphology plus the
        // comma frame identifies reduced-relative modifiers; bare -ing
        // nouns (building, morning) and AUX-governed progressives (are
        // coming) never match. Sequenced after the post-comma pass
        // (disjoint: that pass excludes -ing) with the other frame
        // passes; the amod-Right arm below and the guarded Left-nsubj
        // do the rest.
        refine_pos_comma_participle(&texts, &mut pos, &flags);
        // Shifted-initial pass: comma + DET + NOUN + clause-final NOUN →
        // VERB. Sequenced after the post-comma pass (disjoint: that pass
        // needs comma-adjacent targets, this one comma-distant) and before
        // the relative-matrix pass (disjoint: matrix needs a marker
        // frame). The ADV trailer reads final-adverbial outputs.
        refine_pos_shifted_det_noun_verb(&texts, &mut pos, &flags);
        // Comment-As pass: sentence-initial as + nominal → SCONJ so the
        // existing mark arm fires. First-token only, disjoint from every
        // medial frame; SCONJ outputs feed no refine (waits and arms read
        // them at transition time).
        refine_pos_comment_as(&texts, &mut pos, &flags);
        // Relative-matrix pass: sentence-final NOUN after a relcl VERB with
        // a nominal-headed who/that/where earlier. Sequenced after the
        // pronoun pass so relcl verbs upgraded there (wait, study, sang)
        // are visible as the VERB host; disjoint targets (sentence-final
        // only) from the initial-noun positions.
        refine_pos_relative_matrix_verb(&texts, &mut pos, &flags);
        // Relative-matrix complement pass: matrix NOUN after a relcl VERB
        // with a complement ahead (`stands empty`, `improve fast`, `ducks
        // out …`). Sequenced right after the final matrix pass (whose
        // sentence-final gate leaves exactly these targets); disjoint from
        // it by construction (non-final targets only). VERB outputs feed
        // the standard verbal dynamics below.
        refine_pos_relcl_matrix_complement(&texts, &mut pos, &flags);
        // Adverbial pass: comma-framed -ly NOUN → ADV. Sequenced after the
        // verb passes; targets (NOUN) are disjoint from every verb/ADJ
        // upgrade above, and comma-framed adverbials never feed those
        // triggers (no DET+NOUN initials, no PRON hosts, no be hosts).
        refine_pos_comma_adverbial(&texts, &mut pos, &flags);
        // Initial noun-subject pass: DET+NOUN+NOUN / NOUN+NOUN at the start.
        // Disjoint (targets positions the earlier passes leave as NOUN).
        refine_pos_initial_noun_verb(&texts, &mut pos, &flags);
        // Conjoined-clause pass: CC + NOUN + NOUN → VERB. Disjoint (CC prev
        // matches none of the above triggers).
        refine_pos_conjoined_clause_verb(&texts, &mut pos, &flags);
        // Inverted-copular pass: be-AUX + subject + predicate-NOUN → ADJ.
        // Sequenced just before be-predicate; disjoint from it (its
        // direct/bridged shapes have no nominal between be and target) and
        // from every verb trigger above (ADJ targets).
        refine_pos_inverted_copular(&texts, &mut pos, &flags);
        // Copular-predicate pass: be + NOUN → ADJ. Disjoint (AUX prev with
        // NOUN target matches none of the verb triggers above).
        refine_pos_be_predicate(&texts, &mut pos, &flags);
        // Progressive-participle pass: post-be -ing NOUN → VERB with an
        // overt complement ahead, ADJ when clause-final (genuinely
        // ambiguous — the oracle tie below flags it). Sequenced right
        // after be-predicate (whose -ing skip leaves exactly these
        // targets); ADJ outputs feed the cop dynamics, VERB outputs the
        // standard verbal ones, and the inversion-verb pass below never
        // matches (targets are never NOUN by then, and be-hosts never
        // match its do-modal key).
        refine_pos_progressive_ing(&texts, &mut pos, &flags);
        // Inversion-verb pass: do-modal host + DET + nominal + NOUN → VERB
        // (`Did the report arrive`: arrive is the finite verb of a
        // question-inverted clause). Sequenced LAST so copular predicates
        // upgraded above (`Is the sky blue`: blue is already ADJ) never
        // match, and keyed on do-modal hosts (reusing
        // `is_bare_infinitive_host`) so be-hosts never match either —
        // copular inversion keeps its own dynamics. Guards: the verb must
        // still be NOUN-tagged (verbs found by earlier passes — `Can we
        // leave`, `Do you play` via the pronoun-subject pass — never
        // match), with a DET-led nominal subject directly before it.
        // Known boundary: bare inversions without a determiner (`Did John
        // arrive`, no corpus instance) need subject-NP tracking, not this
        // DET-anchored shape.
        refine_pos_inversion_verb(&texts, &mut pos, &flags);
        // Modal-question pass: modal/do AUX + bare nominal subject +
        // clause-final NOUN → VERB (`Will this work`). Sequenced right
        // after the DET-anchored inversion pass (disjoint: AUX at i-2 vs.
        // DET at i-2); VERB outputs feed the standard verbal dynamics, and
        // the coordination-agreement passes below never match (no CCONJ).
        refine_pos_modal_question_verb(&texts, &mut pos, &flags);
        // Clausal-coordination predicate agreement: a CCONJ-headed second
        // clause with an overt VERB predicate proves the first predicate
        // verbal (`Prices rose yet wages stalled`: stalled is VERB, so rose
        // is VERB). Sequenced after every VERB-upgrade pass so the
        // second-clause verb is visible; the existing nsubj arm and root
        // selection do the rest. Guards: overt nominal subject directly
        // before (DET-led nominals are arguments, conjunction-led words
        // are second conjuncts — neither upgrades), and the full
        // CCONJ + subject + VERB frame ahead before any punctuation.
        // Known boundary: verb-capability without a clausal frame (`Dogs
        // chase red cars`) needs lexicon knowledge — Track B, out of scope.
        refine_pos_clausal_first_predicate(&texts, &mut pos, &flags);
        // Elliptical second predicate after a conjunction (`ran but fell`).
        // The first conjunct's category decides (coordination joins likes):
        // a VERB two back proves a verbal second conjunct, a nominal two
        // back a nominal one (`milk and eggs` stays nominal). Sequenced
        // after every VERB-upgrade so the first conjunct is visible; the
        // (Verb, Verb) conj arm (overt- or shared-subject shape) does the
        // rest. Guards: determiner-led nominals (DET, not CCONJ, precedes)
        // and adverbial-shielded frames (`daily or quit` — the pinned-NOUN
        // adverbial sits between) never match.
        refine_pos_conjoined_predicate_agreement(&texts, &mut pos, &flags);
        // Temporal-`yet` pass: sentence-final `yet` after a predicate
        // adjective (`She isn't ready yet`) is the aspectual adverb, not
        // the adversative coordinator — coordinators always head a second
        // clause (a finite verb follows before any punctuation). Sequenced
        // LAST (after the copular passes, so the ADJ host is visible);
        // gated on the closed-map CCONJ tag, final position, and an ADJ
        // host, so clausal frames (`Prices rose yet wages stalled`: verb
        // ahead) never match. ADV outputs feed nothing downstream. The
        // (Adj, Adv) arm lands it.
        refine_pos_temporal_yet(&texts, &mut pos, &flags);

        // Sentence boundaries from the sentencizer.
        let starts = self.sentencizer.predict(doc);
        let sentences = partition_sentences(&starts, doc.len());

        let mut state = ArcEagerState::new(doc.len(), 0);
        let oracle = DeterministicOracle;
        let mut margins = Vec::new();
        let mut sentence_roots = Vec::new();

        for (s, e) in sentences {
            let root = pick_root(&pos, &texts, &flags, s, e);
            sentence_roots.push(root);
            state.reset_for_sentence(s, e, root, self.labels.root);
            while !state.is_final() {
                let actions = state.candidate_actions(&pos, &texts, &flags, &self.labels);
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
/// Comma-framed participles (`The CEO, smiling, took questions`) skip the
/// same way: a modifier is never the matrix root, by the same logic as
/// the relcl predicate.
/// A subordinate-clause verb (SCONJ + nominal subject frame before it, as
/// in `As you know, ...`) skips the same way — but only with a well-formed
/// matrix predicate (VERB or ADJ) later in the sentence. Without one (`If
/// it rains, stay home`: stay tags NOUN), skipping would orphan the only
/// predicate, so the subordinate verb keeps the crown exactly as before.
fn pick_root(pos: &[Upos], texts: &[String], flags: &[LexemeFlags], s: usize, e: usize) -> usize {
    let mut rel_pending = false;
    for i in s..e {
        if pos[i] == Upos::Verb {
            if !rel_pending
                && !is_comma_framed_participle(texts, pos, i)
                && !is_subordinate_predicate(pos, s, e, i)
            {
                return i;
            }
            rel_pending = false;
            continue;
        }
        if i > 0
            && is_relative_marker(flags[i])
            && matches!(pos[i - 1], Upos::Noun | Upos::Propn)
        {
            rel_pending = true;
        }
    }
    // Where-initial be-question (`Where is the station`, `Where is my
    // bag`): the WH-word is pinned NOUN by the adverb-lexicon gap, so no
    // VERB or ADJ crowns — and the refs root the be-AUX itself
    // (question-02/08: is → root, station/bag → dobj → is). Crown the
    // leftmost be-form AUX when a nominal follows it in the sentence.
    // Guards: any VERB (the first rung already won those, e.g. `Where did
    // she go` → go) or any ADJ (copular predicates head, e.g. `Is lunch
    // ready` → ready) keeps its crown; non-Where sentences (`She is a
    // doctor`) and What/Why-initial questions (`What is 2+2`) keep
    // incumbent dynamics. Purely categorial — no lexicon beyond the
    // closed be-forms and the interrogative word itself.
    if flags.get(s).is_some_and(|f| f.is_where_word())
        && !(s..e).any(|k| pos[k] == Upos::Verb || pos[k] == Upos::Adj)
    {
        let aux = (s..e).find(|&i| {
            pos[i] == Upos::Aux
                && is_be_form(flags[i])
                && (i + 1..e).any(|k| matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron))
        });
        if let Some(aux) = aux {
            return aux;
        }
    }
    // A copular predicate nominal (She is a doctor, Who is the president,
    // There is a problem, This is a Rust project): be + DET + nominals
    // with the nominal span running to the sentence end crowns the LAST
    // nominal of the span (project, not Rust). Runs after the verb scan
    // (finite verbs still win) and before the ADJ fallback — and only when
    // no ADJ stands after be in the sentence, so `Is the report ready`
    // and `Is the sky blue` keep their predicate-adjective crowns.
    // Guards: the DET must sit immediately after a be-form word (`is on
    // the table`, `is raining`, `is not correct` never match — ADP, VERB,
    // PART after be), and any VERB/AUX/ADJ/ADP/SCONJ/CCONJ/PRON inside the
    // span aborts it (`Will this work`, `What does the API return`,
    // `Is lunch on the table` keep incumbent crowns). Where-initial
    // interrogatives are excluded throughout the package: question-02/08
    // pin be-as-root, and reconciling those two conventions is its own
    // design decision — `Where is the station` keeps incumbent dynamics.
    if !flags
        .get(s)
        .is_some_and(|f| f.is_where_word())
    {
        if let Some(pred) = copular_predicate_nominal(pos, texts, flags, s, e) {
            return pred;
        }
    }
    (s..e)
        .find(|&i| pos[i] == Upos::Adj)
        .or_else(|| compound_head_nominal(pos, s, e))
        .or_else(|| (s..e).find(|&i| matches!(pos[i], Upos::Noun | Upos::Propn | Upos::Pron)))
        .unwrap_or(s)
}

/// The head-final crown of a bare nominal chain (`checker` in `Rust
/// borrow checker`, `status` in `flight 204 status`): the last nominal of
/// a sentence whose every content token is nominal (see `pick_root`).
/// `None` otherwise.
fn compound_head_nominal(pos: &[Upos], s: usize, e: usize) -> Option<usize> {
    let mut content = (s..e).filter(|&i| pos[i] != Upos::Punct);
    // Two-word fragments (`Define photosynthesis`) keep the first-crown +
    // Track B tie dynamics — the imperative reading is still live there.
    if content.clone().count() < 3 {
        return None;
    }
    if !content.all(|i| matches!(pos[i], Upos::Noun | Upos::Propn | Upos::Num)) {
        return None;
    }
    (s..e).rfind(|&i| matches!(pos[i], Upos::Noun | Upos::Propn))
}

/// The last nominal of a be + DET-led span running to the sentence end
/// (`doctor` in `She is a doctor`, `project` in `This is a Rust project`),
/// if the span is a clean copular predicate (see `pick_root`). `None`
/// otherwise.
fn copular_predicate_nominal(
    pos: &[Upos],
    _texts: &[String],
    flags: &[LexemeFlags],
    s: usize,
    e: usize,
) -> Option<usize> {
    for i in s..e {
        if pos[i] != Upos::Aux || !is_be_form(flags[i]) {
            continue;
        }
        if i + 1 >= e || pos[i + 1] != Upos::Det {
            continue;
        }
        let mut idx = None;
        let mut clean = true;
        for (k, &p) in pos[i + 2..e].iter().enumerate() {
            match p {
                Upos::Noun | Upos::Propn => idx = Some(i + 2 + k),
                Upos::Punct | Upos::Det => {}
                _ => {
                    clean = false;
                    break;
                }
            }
        }
        if clean {
            if let Some(j) = idx {
                // No predicative adjective later in the sentence: `Is the
                // report ready` keeps `ready`.
                let adj_later = pos[j + 1..e].iter().any(|&p| p == Upos::Adj);
                if !adj_later {
                    return Some(j);
                }
            }
        }
    }
    None
}

/// A subordinate-clause predicate for root purposes: a VERB with an
/// SCONJ + nominal/pronoun frame before it in the sentence, while a VERB
/// or ADJ predicate stands later (the matrix it depends on).
fn is_subordinate_predicate(pos: &[Upos], s: usize, e: usize, i: usize) -> bool {
    if pos[i] != Upos::Verb {
        return false;
    }
    let framed = (s..i).any(|m| {
        pos[m] == Upos::Sconj
            && (m + 1..i).any(|k| matches!(pos[k], Upos::Noun | Upos::Propn | Upos::Pron))
    });
    if !framed {
        return false;
    }
    (i + 1..e).any(|k| pos[k] == Upos::Verb || pos[k] == Upos::Adj)
}

/// A comma-framed -ing verb (`smiling` in `The CEO, smiling, took`): a
/// participial modifier, never a root candidate. Same allocation-free
/// suffix check as the tagger rules.
fn is_comma_framed_participle(texts: &[String], pos: &[Upos], i: usize) -> bool {
    if pos[i] != Upos::Verb || i == 0 || i + 1 >= texts.len() {
        return false;
    }
    if texts[i - 1] != "," || texts[i + 1] != "," {
        return false;
    }
    let word = texts[i].as_str();
    word.len() > 4
        && word
            .get(word.len() - 3..)
            .is_some_and(|sfx| sfx.eq_ignore_ascii_case("ing"))
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
