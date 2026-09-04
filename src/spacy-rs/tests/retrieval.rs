use super::*;
use crate::doc::Doc;
use crate::lang::en::lexicon_config;
use crate::vocab::Vocab;

fn doc_with(texts: &[(&str, f64, Option<InterlinguaId>)]) -> Doc {
    // Build a small parsed doc: each entry is (token_text, per-token
    // confidence, interlingua_lemma_id). Lemma == lowercase token text.
    let vocab = Arc::new(Vocab::new(lexicon_config()));
    let mut doc = Doc::new(vocab.clone());
    for (i, (text, conf, il_id)) in texts.iter().enumerate() {
        let n = doc.push_back(text, i + 1 < texts.len()).expect("push");
        let _ = n;
        let last = doc.len() - 1;
        {
            let tokens = doc.tokens_mut();
            let tok = &mut tokens[last];
            tok.lemma = vocab.strings().add(&text.to_lowercase());
            tok.confidence = Some(*conf);
            tok.interlingua_lemma_id = *il_id;
        }
    }
    doc
}

#[test]
fn lemma_grep_returns_hits_with_confidence_and_lemma_id() {
    let doc = doc_with(&[
        ("Show", 0.9, Some(InterlinguaId::from_u64(7))),
        ("me", 0.8, None),
        ("the", 0.7, None),
        ("report", 0.6, Some(InterlinguaId::from_u64(9))),
    ]);
    let hits = lemma_grep(&doc, "show");
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.lemma, "show");
    assert_eq!(hit.lemma_id, Some(InterlinguaId::from_u64(7)));
    assert_eq!(hit.parse_confidence, 0.9);
    // Byte span of the first token in "Show me the report".
    assert_eq!(hit.span, Span { start: 0, end: 4 });
}

#[test]
fn lemma_grep_is_case_insensitive_and_skips_unmatched() {
    let doc = doc_with(&[("show", 0.9, None), ("display", 0.5, None)]);
    assert_eq!(lemma_grep(&doc, "SHOW").len(), 1);
    assert_eq!(lemma_grep(&doc, "list").len(), 0);
}

#[test]
fn lemma_grep_skips_tokens_without_a_resolved_lemma() {
    let vocab = Arc::new(Vocab::new(lexicon_config()));
    let mut doc = Doc::new(vocab.clone());
    doc.push_back("show", false).expect("push");
    // lemma hash left 0 → strings.get(0) resolves to "", which cannot equal
    // the query, so the token is skipped rather than falsely matched.
    assert!(lemma_grep(&doc, "show").is_empty());
}

// ── Fuzzy retrieval (hermetic) ──────────────────────────────────────

/// A deterministic synonym-aware provider: function words embed by
/// identity, and the synonym table maps paraphrase equivalents onto the
/// same dimension — the exact paraphrase axis lemma-grep cannot cover.
struct SynonymProvider;
impl EmbeddingProvider for SynonymProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut v = vec![0.0f32; 16];
        for tok in text.split_whitespace() {
            let t = tok.to_lowercase();
            let dim = match t.as_str() {
                "show" | "display" | "get" | "list" => 0usize,
                "me" => 1,
                "the" => 2,
                "report" | "sales" => 3,
                "table" | "chart" => 4,
                other => (crate::hash::hash_utf8(other) % 16) as usize,
            };
            v[dim] += 1.0;
        }
        Some(v)
    }
}

fn fuzzy_index() -> InMemoryFuzzyIndex {
    let mut idx = InMemoryFuzzyIndex::new(Arc::new(SynonymProvider));
    idx.insert(Span { start: 0, end: 18 }, "show me the report");
    idx.insert(Span { start: 20, end: 40 }, "delete the old file");
    idx
}

#[test]
fn fuzzy_retrieval_covers_the_paraphrase_gap() {
    let idx = fuzzy_index();
    let hits = idx.search("display the report", 2);
    assert!(!hits.is_empty(), "paraphrase-matched region found");
    assert_eq!(
        hits[0].span,
        Span { start: 0, end: 18 },
        "paraphrase region ranks first"
    );
    assert!(hits[0].score > 0.8, "high paraphrase similarity");
}

#[test]
fn fuzzy_retrieval_returns_top_k_by_score() {
    let idx = fuzzy_index();
    let hits = idx.search("sales table", 2);
    assert!(!hits.is_empty());
    let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "sorted desc");
}

#[test]
fn cosine_zero_on_length_mismatch() {
    assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_parity_with_canonical() {
    // Characterization (M3a): locks current `cosine` outputs verbatim so the
    // P1 migration (body → shared primitive) must preserve them byte-for-byte.
    // Table: equal / orthogonal / opposite / zero / empty / mismatch / NaN.
    let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
        (vec![1.0, 0.0], vec![1.0, 0.0]),
        (vec![1.0, 0.0], vec![0.0, 1.0]),
        (vec![1.0, 0.0], vec![-1.0, 0.0]),
        (vec![0.0, 0.0], vec![0.0, 0.0]),
        (vec![0.0, 0.0], vec![1.0, 0.0]),
        (vec![], vec![]),
        (vec![], vec![1.0]),
        (vec![1.0], vec![1.0, 2.0]),
        (vec![f32::NAN], vec![1.0]),
        (vec![1.0, 2.0], vec![2.0, 4.0]),
    ];
    let expected: Vec<Option<f32>> = vec![
        Some(1.0),
        Some(0.0),
        Some(-1.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        Some(0.0),
        None, // NaN in → NaN out
        Some(1.0),
    ];
    for ((a, b), want) in cases.iter().zip(expected.iter()) {
        let got = cosine(a, b);
        match want {
            Some(w) => assert!(
                (got - w).abs() < 1e-6,
                "cosine({a:?}, {b:?}) = {got}, want {w}"
            ),
            None => assert!(got.is_nan(), "cosine({a:?}, {b:?}) = {got}, want NaN"),
        }
        // And the shared-primitive candidate must agree. Strictly this is
        // semantic equality (`==` + NaN-is-NaN), not bitwise identity: on
        // empty-vs-empty the bespoke body yields `-0.0` (0.0/1e-9 with a
        // negative-zero numerator from the empty sum) where the canonical
        // yields `0.0`. `-0.0 == 0.0` and the only caller filters `> 0.0`
        // (false for both), so the delta is filter-invisible — recorded
        // here, migration stays safe.
        let canon = common_core::vector_math::cosine_similarity_f32(a, b);
        assert!(
            (got == canon) || (got.is_nan() && canon.is_nan()),
            "parity: cosine({a:?},{b:?})={got} vs canonical={canon}"
        );
    }
}

/// Fixed-embedding provider: query embeds to `q`, regions to listed vectors.
struct FixedProvider {
    q: Option<Vec<f32>>,
    region_embs: Vec<Vec<f32>>,
}
impl EmbeddingProvider for FixedProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        if text == "QUERY" {
            return self.q.clone();
        }
        text.strip_prefix("REGION")
            .and_then(|n| n.parse::<usize>().ok())
            .and_then(|i| self.region_embs.get(i).cloned())
    }
}

#[test]
fn fuzzy_search_filters_nonpositive() {
    // Characterization (M3a): strict-positive filter + take(k) order.
    // q=(1,0); regions: r0=(1,0) sim 1.0, r1=(-1,0) sim -1.0, r2=(0,0) sim 0.0.
    let provider = Arc::new(FixedProvider {
        q: Some(vec![1.0, 0.0]),
        region_embs: vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 0.0]],
    });
    let mut idx = InMemoryFuzzyIndex::new(provider);
    for (i, start) in [0usize, 10, 20].iter().enumerate() {
        idx.insert(
            Span { start: *start, end: *start + 5 },
            &format!("REGION{i}"),
        );
    }
    // Negative and zero sims excluded; only the positive region survives.
    let hits = idx.search("QUERY", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span, Span { start: 0, end: 5 });
    // k == 0 → empty.
    assert!(idx.search("QUERY", 0).is_empty());
    // k larger than region count → capped at the positive survivors.
    assert_eq!(idx.search("QUERY", 100).len(), 1);
    // Unembeddable query → empty.
    let dead = InMemoryFuzzyIndex::new(Arc::new(FixedProvider {
        q: None,
        region_embs: vec![],
    }));
    assert!(dead.search("QUERY", 5).is_empty());
}

// ── Cross-check ─────────────────────────────────────────────────────

#[test]
fn cross_check_surfaces_both_axes_on_material_disagreement() {
    // Lemma-grep "show" → low-confidence deterministic hit (ArcEager near
    // tie). Fuzzy "display the report" → high paraphrase score on the same
    // region. Material disagreement → BOTH surfaced, never deduped.
    let doc = doc_with(&[
        ("show", 0.3, Some(InterlinguaId::from_u64(7))),
        ("me", 0.8, None),
        ("the", 0.7, None),
        ("report", 0.6, None),
    ]);
    let lemma_hits = lemma_grep(&doc, "show");
    let fuzzy_hits = fuzzy_index().search("display the report", 2);

    let report = cross_check(&lemma_hits, &fuzzy_hits, 0.1);
    assert!(report.hits.len() >= 2, "lemma + fuzzy both surfaced");
    let sources: Vec<RetrievalSource> = report.hits.iter().map(|h| h.source).collect();
    assert!(sources.contains(&RetrievalSource::LemmaGrep));
    assert!(sources.contains(&RetrievalSource::Fuzzy));

    // The region covering both carries a disagreement verdict.
    let disagreed: Vec<&RegionVerdict> = report
        .regions
        .iter()
        .filter(|r| r.lemma_confidence.is_some() && r.fuzzy_score.is_some())
        .collect();
    assert!(!disagreed.is_empty(), "a region is covered by both axes");
    assert!(
        disagreed.iter().any(|r| r.disagreed),
        "material confidence difference is surfaced, not collapsed"
    );
}

#[test]
fn cross_check_lemma_only_regions_have_no_conflict() {
    let doc = doc_with(&[("show", 0.9, None), ("zzz", 0.5, None)]);
    let lemma_hits = lemma_grep(&doc, "show");
    let report = cross_check(&lemma_hits, &[], 0.1);
    assert_eq!(report.hits.len(), 1);
    let v = &report.regions[0];
    assert_eq!(v.lemma_confidence, Some(0.9));
    assert_eq!(v.fuzzy_score, None);
    assert!(!v.disagreed);
}
