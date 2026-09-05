//! Hermetic parse-accuracy bench (fully offline — no LLM, no network).
//!
//! Scores the deterministic ladder (`NlpPipeline::en_default` +
//! `process_sync(text, None)`, i.e. the exact path the router uses before any
//! refiner) against pinned references in `tests/data/parse_bench.refs.json`.
//! References are authored once — hand-written seed plus reviewed LLM output
//! from the live `parse_bench_live` generator — then frozen; this target only
//! ever reads them.
//!
//! Metrics per item: UPOS accuracy, UAS (head attachment), LAS (head+label),
//! lemma accuracy. Reported overall and per dataset category. The ratchet:
//! `parse_bench.floors.json` pins the current scores at full precision (the
//! scoreboard prints 3 decimals for humans) plus the scored-item count — any
//! rule PR must clear every floor. Raise the floors (re-pin actuals) when a
//! rule lands; never lower them to make red green.
//!
//! Run the scoreboard with:
//! `make spacy-parse-benchmark`
//! (`cargo test -p spacy-rs --test parse_bench -- --nocapture`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

#[derive(Debug, Deserialize)]
struct Item {
    id: String,
    category: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Dataset {
    items: Vec<Item>,
}

/// One reference record: the ladder's own `AnnotationRecord` JSON shape
/// (`text/pos/dep/head/lemma`), Universal Dependencies analysis expressed in
/// the parser's closed label vocabulary.
#[derive(Debug, Deserialize)]
struct RefRec {
    text: String,
    pos: String,
    dep: String,
    head: i32,
    #[serde(default)]
    lemma: String,
}

#[derive(Debug, Deserialize)]
struct Refs {
    refs: BTreeMap<String, Vec<RefRec>>,
}

#[derive(Debug, Deserialize, Clone, Copy, Default)]
struct MetricFloor {
    upos: f64,
    uas: f64,
    las: f64,
    lemma: f64,
}

#[derive(Debug, Deserialize)]
struct Floors {
    overall: MetricFloor,
    categories: BTreeMap<String, MetricFloor>,
    scored_items: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct Accum {
    tokens: usize,
    upos: usize,
    uas: usize,
    las: usize,
    lemma: usize,
}

impl Accum {
    fn add(&mut self, pred_pos: &str, pred_dep: &str, pred_head: i32, pred_lemma: &str, gold: &RefRec) {
        self.tokens += 1;
        if pred_pos == gold.pos {
            self.upos += 1;
        }
        if pred_head == gold.head {
            self.uas += 1;
        }
        if pred_head == gold.head && pred_dep == gold.dep {
            self.las += 1;
        }
        // Lemma counts when the ref pins one and the parser agrees; refs that
        // omit the lemma (uncertain gold) auto-credit.
        if gold.lemma.is_empty() || pred_lemma == gold.lemma {
            self.lemma += 1;
        }
    }

    fn scores(&self) -> MetricFloor {
        let n = self.tokens.max(1) as f64;
        MetricFloor {
            upos: self.upos as f64 / n,
            uas: self.uas as f64 / n,
            las: self.las as f64 / n,
            lemma: self.lemma as f64 / n,
        }
    }
}

/// Rubric completeness: every reference token pins `pos`, `dep`, `head`
/// AND `lemma`. A missing lemma auto-credits (`Accum::add`), silently
/// inflating the lemma metric — an incomplete rubric is a dishonest meter.
#[test]
fn refs_are_fully_annotated() {
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.refs.json")).expect("refs readable");
    let refs: Refs = serde_json::from_str(&raw).expect("refs parse");
    let mut gaps: Vec<String> = Vec::new();
    for (id, recs) in &refs.refs {
        for r in recs {
            if r.pos.is_empty() || r.dep.is_empty() {
                gaps.push(format!("{id}: {:?} missing pos/dep", r.text));
            }
            if r.lemma.is_empty() {
                gaps.push(format!("{id}: {:?} missing lemma", r.text));
            }
        }
    }
    assert!(
        gaps.is_empty(),
        "rubric has {} unannotated tokens:\n{}",
        gaps.len(),
        gaps.join("\n")
    );
}

#[test]
fn dataset_is_stratified() {
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.json")).expect("dataset readable");
    let dataset: Dataset = serde_json::from_str(&raw).expect("dataset parses");
    assert!(!dataset.items.is_empty(), "dataset is non-empty");
    let mut per_category: BTreeMap<&str, usize> = BTreeMap::new();
    for item in &dataset.items {
        *per_category.entry(item.category.as_str()).or_default() += 1;
    }
    for (category, count) in &per_category {
        assert!(*count >= 3, "category '{category}' needs ≥3 items, has {count}");
    }
    eprintln!("bench dataset: {} items across {} categories", dataset.items.len(), per_category.len());
}

#[test]
fn parse_bench_accuracy_floors() {
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.json")).expect("dataset readable");
    let dataset: Dataset = serde_json::from_str(&raw).expect("dataset parses");
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.refs.json")).expect("refs readable");
    let refs: Refs = serde_json::from_str(&raw).expect("refs parse");
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.floors.json")).expect("floors readable");
    let floors: Floors = serde_json::from_str(&raw).expect("floors parse");

    let pipe = spacy_rs::NlpPipeline::en_default().expect("en pipeline");
    let mut overall = Accum::default();
    let mut per_category: BTreeMap<String, Accum> = BTreeMap::new();
    let mut scored = 0usize;
    // (dataset category, item id + reason) — grouped by the true category
    // below. Reconstructing the category from the id (`split('-')`) mangles
    // multi-word categories (`command-target-*` landed under `command`,
    // `rare-verb-*` and `rare-lex-*` lumped as `rare`).
    let mut unscored: Vec<(String, String)> = Vec::new();

    for item in &dataset.items {
        let Some(gold) = refs.refs.get(&item.id) else {
            unscored.push((item.category.clone(), item.id.clone()));
            continue;
        };
        let (doc, result) = pipe
            .process_sync_with_confidence(
                &item.text,
                None,
                None,
                spacy_rs::RefinePolicy::default(),
            )
            .expect("deterministic parse");
        let _ = doc;
        let set = result.records;
        if set.0.len() != gold.len()
            || !set.0.iter().zip(gold.iter()).all(|(p, g)| p.text == g.text)
        {
            unscored.push((item.category.clone(), format!("{} [drift]", item.id)));
            continue;
        }
        scored += 1;
        let accum = per_category.entry(item.category.clone()).or_default();
        for (pred, g) in set.0.iter().zip(gold.iter()) {
            overall.add(&pred.pos, &pred.dep, pred.head, &pred.lemma, g);
            accum.add(&pred.pos, &pred.dep, pred.head, &pred.lemma, g);
        }
    }

    // Scoreboard (visible with `-- --nocapture`, i.e. `make spacy-parse-benchmark`).
    let overall_scores = overall.scores();
    eprintln!(
        "\nparse-bench: {scored} scored / {} items ({} unscored)",
        dataset.items.len(),
        unscored.len()
    );
    eprintln!(
        "overall : upos={:.3} uas={:.3} las={:.3} lemma={:.3} ({} tokens) [{}/{}/{}/{}]",
        overall_scores.upos, overall_scores.uas, overall_scores.las, overall_scores.lemma,
        overall.tokens, overall.upos, overall.uas, overall.las, overall.lemma
    );
    for (category, accum) in &per_category {
        let s = accum.scores();
        eprintln!(
            "  {:<14} upos={:.3} uas={:.3} las={:.3} lemma={:.3} ({} tokens) [{}/{}/{}/{}]",
            category, s.upos, s.uas, s.las, s.lemma, accum.tokens,
            accum.upos, accum.uas, accum.las, accum.lemma
        );
    }
    if !unscored.is_empty() {
        // One line per category, not per item — the per-item list used to
        // bury the scoreboard under 50+ lines.
        let mut by_category: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (category, u) in &unscored {
            by_category.entry(category.clone()).or_default().push(u.clone());
        }
        eprintln!("  unscored: {} (generate via live ref-gen)", unscored.len());
        for (category, ids) in &by_category {
            eprintln!("    {category}: {} missing ({})", ids.len(), ids.join(", "));
        }
    }

    // Ratchet: coverage never shrinks silently, no metric regresses.
    assert!(scored > 0, "bench measured nothing — pin seed refs first");
    assert!(
        scored >= floors.scored_items,
        "scored-item coverage shrank: {scored} < pinned {}",
        floors.scored_items
    );
    assert_no_regress("overall", &overall_scores, &floors.overall);
    for (category, accum) in &per_category {
        if let Some(floor) = floors.categories.get(category) {
            assert_no_regress(category, &accum.scores(), floor);
        }
    }
}

fn assert_no_regress(name: &str, actual: &MetricFloor, floor: &MetricFloor) {
    const EPS: f64 = 1e-9;
    assert!(actual.upos + EPS >= floor.upos, "{name}: upos regressed {:.3} < {:.3}", actual.upos, floor.upos);
    assert!(actual.uas + EPS >= floor.uas, "{name}: uas regressed {:.3} < {:.3}", actual.uas, floor.uas);
    assert!(actual.las + EPS >= floor.las, "{name}: las regressed {:.3} < {:.3}", actual.las, floor.las);
    assert!(actual.lemma + EPS >= floor.lemma, "{name}: lemma regressed {:.3} < {:.3}", actual.lemma, floor.lemma);
}
