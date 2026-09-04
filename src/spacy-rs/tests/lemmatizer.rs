use super::*;
use crate::morph::Morphology;
use std::sync::Arc;

fn en() -> Lemmatizer {
    let strings = Arc::new(StringStore::new());
    rule_lemmatizer_with_strings(strings)
}

#[test]
fn plural_nouns_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("cats", Upos::Noun, 0)[0], "cat");
    assert_eq!(l.lemmatize("boxes", Upos::Noun, 0)[0], "box");
    assert_eq!(l.lemmatize("wives", Upos::Noun, 0)[0], "wife");
    assert_eq!(l.lemmatize("children", Upos::Noun, 0)[0], "child", "exception");
}

#[test]
fn verb_inflections_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("running", Upos::Verb, 0)[0], "run");
    assert_eq!(l.lemmatize("studies", Upos::Verb, 0)[0], "study");
    assert_eq!(l.lemmatize("went", Upos::Verb, 0)[0], "go", "exception");
}

#[test]
fn aux_be_forms_resolve_to_be() {
    // UD + spaCy lookup parity: auxiliary be-forms resolve to the lemma
    // "be" (8 bench pins, unanimous where pinned; all other be-forms carry
    // empty refs and auto-credit either way). Closed irregular class, same
    // shape as the ca/wo/n't splinter map below.
    let l = en();
    for surface in ["is", "are", "was", "were", "am", "be", "been", "being"] {
        assert_eq!(l.lemmatize(surface, Upos::Aux, 0)[0], "be", "{surface}");
    }
}

#[test]
fn verb_fallback_lowercases_sentence_case() {
    // Sentence case carries no lexical information on verbs (never proper
    // nouns): a capitalized VERB with no applicable rule falls back to the
    // lowercased form, not the surface. 16 bench pins, unanimous.
    let l = en();
    for surface in ["Let", "Close", "Send", "Buy", "Run", "Work"] {
        assert_eq!(
            l.lemmatize(surface, Upos::Verb, 0)[0],
            surface.to_lowercase(),
            "{surface}"
        );
    }
}

#[test]
fn acronym_fallback_lowercases() {
    // Must-NOT-fire (scope): all-caps tokens are acronyms, never
    // title-case names — "CEO" lowercases while "French" keeps surface.
    let l = en();
    assert_eq!(l.lemmatize("CEO", Upos::Noun, 0)[0], "ceo");
}

#[test]
fn titlecase_nominal_fallback_keeps_surface() {
    // Must-NOT-fire: title-case nominals may be proper (`French`, `John`)
    // — the fallback keeps surface, per the proper-noun convention pinned
    // by refs and `proper_nouns_keep_case`. (Common title-initials like
    // `Study` stay a documented gap: proper/common needs §8.2 evidence.)
    let l = en();
    assert_eq!(l.lemmatize("French", Upos::Noun, 0)[0], "French");
}

#[test]
fn be_lemma_verb_path_uses_tables() {
    // Invariance: a VERB-tagged "was" already resolved to "be" through the
    // blob verb exceptions before the Aux map existed — both routes agree,
    // so the Aux gate changes no VERB behavior. Pins the table route
    // against future blob regressions.
    let l = en();
    assert_eq!(l.lemmatize("was", Upos::Verb, 0)[0], "be");
    assert_eq!(l.lemmatize("was", Upos::Aux, 0)[0], "be");
}

#[test]
fn adjective_comparatives_reduce() {
    let l = en();
    assert_eq!(l.lemmatize("faster", Upos::Adj, 0)[0], "fast");
    // spaCy's exception table for "better" is ["good", "well"]; exceptions
    // are inserted at position 0 in order, so the LAST lemma wins.
    assert_eq!(l.lemmatize("better", Upos::Adj, 0)[0], "well", "exception");
}

#[test]
fn proper_nouns_keep_case() {
    let l = en();
    assert_eq!(l.lemmatize("Apple", Upos::Propn, 0)[0], "Apple");
}

#[test]
fn unknown_pos_lowercases() {
    let l = en();
    assert_eq!(l.lemmatize("THE", Upos::Det, 0)[0], "the");
    assert_eq!(l.lemmatize("123", Upos::Num, 0)[0], "123");
}

#[test]
fn base_form_skips_lemmatization() {
    let strings = Arc::new(StringStore::new());
    let m = Arc::new(Morphology::new(strings));
    let l = en().with_morphology(m);
    // Number=Sing noun is a base form
    let sing = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("Number=Sing");
    assert!(l.is_base_form(Upos::Noun, sing));
    assert_eq!(l.lemmatize("cat", Upos::Noun, sing)[0], "cat");
    // VerbForm=Inf is a base form
    let inf = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("VerbForm=Inf");
    assert!(l.is_base_form(Upos::Verb, inf));
    // Plural is not
    let plur = l
        .morphology
        .as_ref()
        .expect("morphology")
        .add("Number=Plur");
    assert!(!l.is_base_form(Upos::Noun, plur));
    assert_eq!(l.lemmatize("cats", Upos::Noun, plur)[0], "cat");
}

#[test]
fn cache_reuses_analyses() {
    let l = en();
    assert_eq!(l.cache_len(), 0);
    let a = l.lemmatize("cats", Upos::Noun, 0);
    assert_eq!(a, vec!["cat"]);
    assert_eq!(l.cache_len(), 1);
    let b = l.lemmatize("cats", Upos::Noun, 0);
    assert_eq!(a, b);
    assert_eq!(l.cache_len(), 1);
}

#[test]
fn lookup_mode() {
    let table = HashMap::from([
        ("went".to_string(), "go".to_string()),
        ("cats".to_string(), "cat".to_string()),
    ]);
    let l = Lemmatizer::lookup(table);
    assert_eq!(l.lemmatize("went", Upos::Verb, 0)[0], "go");
    assert_eq!(l.lemmatize("cats", Upos::Noun, 0)[0], "cat");
    assert_eq!(l.lemmatize("unknown", Upos::Noun, 0)[0], "unknown");
}

#[test]
fn english_rule_data_is_loaded() {
    let l = en();
    let blob = l.blob.as_ref().expect("rule mode carries a blob");
    assert_eq!(blob.pos_count(), 5);
    assert!(!blob.rules("noun").is_empty());
    assert!(blob.index_contains("noun", "aardvark"));
    assert!(blob.exc_for("verb", "went").is_some());
    assert_eq!(l.mode(), LemmatizerMode::Rule);
}

/// Side-by-side probe (diagnostic only — asserts no score): for every bench
/// token whose deterministic UPOS misses the ref, compare lemma-support
/// under the predicted vs gold POS. Support ladder (higher wins):
/// 3 = closed/exception hit (`contraction_lemma`, blob `exc`),
/// 2 = base form, rule-derived in-index form, or PROPN surface,
/// 1 = rule-derived OOV form or no-tables guess,
/// 0 = surface fallback (no rule fired).
/// The provenance mirrors `rule_lemmatize` exactly (same order semantics:
/// in-index `insert(0)`, exceptions prepended so the last wins); tokens
/// where the probe cannot reproduce the pipeline's own pred lemma are
/// excluded as fidelity errors rather than counted. No production code is
/// touched — this quantifies the ceiling for lemma-driven POS rerank, it
/// does not implement it.
#[test]
fn lemma_support_pos_ceiling() {
    use std::path::PathBuf;
    use std::str::FromStr;

    #[derive(Debug, serde::Deserialize)]
    struct Item {
        id: String,
        text: String,
    }
    #[derive(Debug, serde::Deserialize)]
    struct Dataset {
        items: Vec<Item>,
    }
    #[derive(Debug, serde::Deserialize)]
    struct RefRec {
        text: String,
        pos: String,
    }
    #[derive(Debug, serde::Deserialize)]
    struct Refs {
        refs: std::collections::BTreeMap<String, Vec<RefRec>>,
    }

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
    }

    /// (support level, primary lemma), mirroring `rule_lemmatize`.
    fn support(l: &Lemmatizer, orth: &str, pos: Upos) -> (u8, String) {
        if let Some(lemma) = super::contraction_lemma(orth, pos) {
            return (3, lemma.to_string());
        }
        let lower = orth.to_lowercase();
        let univ_pos = pos.to_string();
        if l.is_base_form(pos, 0) {
            return (2, lower);
        }
        let blob = l.blob.as_ref().expect("rule mode carries a blob");
        if blob.pos_keys().all(|k| k != univ_pos) {
            if pos == Upos::Propn {
                return (2, orth.to_string());
            }
            return (1, lower);
        }
        let rules = blob.rules(&univ_pos);
        let mut forms: Vec<String> = Vec::new();
        let mut oov_forms: Vec<String> = Vec::new();
        for (old, new) in rules {
            if lower.ends_with(old) {
                let form = format!("{}{}", &lower[..lower.len() - old.len()], new);
                if form.is_empty() {
                    continue;
                }
                if blob.index_contains(&univ_pos, &form) || !lex_attrs::is_alpha(&form) {
                    if blob.index_contains(&univ_pos, &form) {
                        forms.insert(0, form);
                    } else {
                        forms.push(form);
                    }
                } else {
                    oov_forms.push(form);
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        forms.retain(|f| seen.insert(f.clone()));
        let mut exc_winner: Option<String> = None;
        if let Some(lemmas) = blob.exc_for(&univ_pos, lower.as_str()) {
            for lemma in lemmas.split(|&b| b == 0) {
                if lemma.is_empty() {
                    continue;
                }
                let lemma = std::str::from_utf8(lemma).expect("lemma blob is UTF-8");
                if seen.insert(lemma.to_string()) {
                    forms.insert(0, lemma.to_string());
                    exc_winner = Some(lemma.to_string());
                }
            }
        }
        if forms.is_empty() {
            forms.extend(oov_forms);
        }
        if forms.is_empty() {
            return (0, orth.to_string());
        }
        let winner = forms.remove(0);
        // Exception winners score 3 even when also in-index.
        if exc_winner.as_deref() == Some(winner.as_str()) {
            return (3, winner);
        }
        // Distinguish rule-derived in-index (2) from OOV (1): recompute the
        // winner's provenance by rule order is overkill — in-index lookup
        // on the winner is exact for the strip-to-known-word case, and
        // OOV winners are never in-index by construction.
        if blob.index_contains(&univ_pos, &winner) || !lex_attrs::is_alpha(&winner) {
            (2, winner)
        } else {
            // Winner is either an OOV rule form or an exception that is not
            // in-index (e.g. multiword splits): exceptions still count as
            // closed support.
            if blob.exc_for(&univ_pos, lower.as_str()).is_some() {
                (3, winner)
            } else {
                (1, winner)
            }
        }
    }

    let raw = std::fs::read_to_string(data_dir().join("parse_bench.json")).expect("dataset readable");
    let dataset: Dataset = serde_json::from_str(&raw).expect("dataset parses");
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.refs.json")).expect("refs readable");
    let refs: Refs = serde_json::from_str(&raw).expect("refs parse");
    assert!(!dataset.items.is_empty(), "dataset is non-empty");

    let l = en();
    let pipe = crate::pipeline::NlpPipeline::en_default().expect("en pipeline");
    let (mut errors, mut flip, mut abstain, mut adversarial, mut fidelity) = (0, 0, 0, 0, 0);
    let (mut flip_list, mut adv_list) = (Vec::new(), Vec::new());
    for item in &dataset.items {
        let Some(gold) = refs.refs.get(&item.id) else {
            continue;
        };
        let Ok((_doc, result)) = pipe.process_sync_with_confidence(
            &item.text,
            None,
            None,
            crate::pipeline::RefinePolicy::default(),
        ) else {
            continue;
        };
        let set = result.records;
        if set.0.len() != gold.len()
            || !set.0.iter().zip(gold.iter()).all(|(p, g)| p.text == g.text)
        {
            continue;
        }
        for (pred, g) in set.0.iter().zip(gold.iter()) {
            if pred.pos == g.pos {
                continue;
            }
            errors += 1;
            let Ok(gold_pos) = Upos::from_str(&g.pos) else {
                continue;
            };
            let Ok(pred_pos) = Upos::from_str(&pred.pos) else {
                continue;
            };
            let (psup, plem) = support(&l, &pred.text, pred_pos);
            if plem != pred.lemma {
                fidelity += 1;
                continue;
            }
            let (gsup, glem) = support(&l, &pred.text, gold_pos);
            let _ = glem;
            if gsup > psup {
                flip += 1;
                flip_list.push(format!(
                    "{} {}: {:?} pred({},{}) gold({},{})",
                    item.id, pred.text, g.pos, pred.pos, psup, g.pos, gsup
                ));
            } else if gsup == psup {
                abstain += 1;
            } else {
                adversarial += 1;
                adv_list.push(format!(
                    "{} {}: pred({},{}) gold({},{})",
                    item.id, pred.text, pred.pos, psup, g.pos, gsup
                ));
            }
        }
    }
    eprintln!("\nlemma-support ceiling over {errors} UPOS errors (fidelity-excluded: {fidelity})");
    eprintln!("  flippable (gold>pred): {flip}");
    for f in &flip_list {
        eprintln!("    FLIP {f}");
    }
    eprintln!("  abstain (tie): {abstain}");
    eprintln!("  adversarial (gold<pred): {adversarial}");
    for a in &adv_list {
        eprintln!("    ADVERSARIAL {a}");
    }
}
