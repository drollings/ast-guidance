//! Chart selection — deterministic capability match → HNSW retrieval → LLM
//! adjudication.
//!
//! Given a raw request, the selector picks the best chart, cheapest step
//! first (VISION: deterministic before probabilistic):
//!
//! 1. **Deterministic capability match** — if the request already names a
//!    chart or one of its provides assets, select it with `score = 1.0`.
//!    No LLM call.
//! 2. **HNSW retrieval** — embed the *raw user request* (never a
//!    classifier-authored summary, review R1) and query the `workflow_library`
//!    index built at boot in `ChartStore`; candidates below `cfg.min_score`
//!    are dropped.
//! 3. **LLM adjudication** — one `ChatBackend` call over the candidate list
//!    returns the single best chart (or none). The LLM decides *which* chart;
//!    the deterministic entity binding decides *whether it is executable*
//!    (`Exact` vs `Partial` with interview gaps).
//!
//! Everything here is pure data-in / data-out — no orchestration state.

use std::fmt::Write as _;
use std::sync::Arc;

use common_core::contains_ident_word;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;

use crate::charts::binding::{bind_chart, AmbiguousDep, Bindings, Entity};
use crate::charts::store::ChartStore;
use crate::charts::{ChartDef, ChartError};
use crate::config::ChartsConfig;

/// Trait for ColBERT-based candidate re-ranking. Defined here so
/// `ChartSelector` can hold an optional reranker without depending on
/// `fluent-onnx` directly. The onnx-gated implementation lives in `ort.rs`.
pub trait ColbertRerank: Send + Sync {
    /// Re-rank candidates by ColBERT MaxSim relevance to the query.
    /// Returns candidates reordered by descending ColBERT score, carrying
    /// the original HNSW scores (the reranker re-ranks, not fabricates).
    fn rerank_candidates(
        &self,
        query: &str,
        candidates: &[(String, f64)],
        doc_texts: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, f64)>;
}

/// How well a request fits a chart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChartFit {
    /// The chart is fully bound by the provided entities — executable now.
    Exact,
    /// The chart is the best fit but some required inputs are missing. The
    /// gap names drive the interview loop.
    Partial { gaps: Vec<String> },
    /// No chart fits the request — fall through to fresh planning.
    Mismatch,
}

/// The result of selecting a chart for a request.
#[derive(Debug, Clone)]
pub struct ChartMatch {
    /// Selected chart name. Empty for `ChartFit::Mismatch`.
    pub chart: String,
    /// Selection confidence in `[0, 1]`: 1.0 for a deterministic match, else
    /// the top HNSW cosine similarity.
    pub score: f64,
    /// Executability of the selected chart given the bound entities.
    pub fit: ChartFit,
    /// Deterministic binding of the selected chart's deps against the
    /// entities (`None` only for a mismatch).
    pub bindings: Option<Bindings>,
}

impl ChartMatch {
    /// The "no chart fits" sentinel.
    fn mismatch() -> Self {
        Self {
            chart: String::new(),
            score: 0.0,
            fit: ChartFit::Mismatch,
            bindings: None,
        }
    }
}

/// Three-step chart selector. Cheap steps always run first and short-circuit.
pub struct ChartSelector {
    store: Arc<ChartStore>,
    /// Adjudicator backend (mock-injectable). `None` skips the LLM step and
    /// falls back to the top HNSW candidate.
    client: Option<Arc<dyn ChatBackend>>,
    /// Reranker backend (mock-injectable). `None` skips the rerank step.
    reranker: Option<Arc<dyn ChatBackend>>,
    /// ColBERT late-interaction reranker (mock-injectable). When present and
    /// the LLM reranker is absent, uses MaxSim scoring for candidate reranking.
    colbert: Option<Arc<dyn ColbertRerank>>,
    cfg: ChartsConfig,
}

impl ChartSelector {
    /// Build a selector over the boot-loaded chart store.
    pub fn new(
        store: Arc<ChartStore>,
        client: Option<Arc<dyn ChatBackend>>,
        cfg: ChartsConfig,
    ) -> Self {
        Self {
            store,
            client,
            reranker: None,
            colbert: None,
            cfg,
        }
    }

    /// Attach a reranker backend for Step 2.5 (candidate re-ranking).
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn ChatBackend>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Attach a ColBERT reranker for Step 2.5 (MaxSim-based candidate
    /// re-ranking). Used when the LLM reranker is absent.
    #[must_use]
    pub fn with_colbert_reranker(mut self, colbert: Arc<dyn ColbertRerank>) -> Self {
        self.colbert = Some(colbert);
        self
    }

    /// Select the best chart for `request`, or `ChartFit::Mismatch`.
    ///
    /// `entities` are the bound context entities used to decide whether a
    /// selected chart is `Exact` (executable) or `Partial` (needs interview).
    pub fn select(&self, request: &str, entities: &[Entity]) -> Result<ChartMatch, ChartError> {
        // Step 1: deterministic capability match — cheapest, no LLM.
        if let Some(chart) = self.deterministic_hit(request) {
            tracing::debug!(
                target: "router.charts.select",
                chart = %chart.name,
                "deterministic capability match"
            );
            return Ok(self.build_match(&chart, 1.0, entities, &[]));
        }

        // Step 2: HNSW retrieval over the raw request.
        let candidates = self.retrieve(request)?;
        if candidates.is_empty() {
            return Ok(ChartMatch::mismatch());
        }

        // Step 2.5: LLM re-ranking of the HNSW candidates (cheap before
        // expensive — a cross-encoder reranker narrows what the adjudicator
        // must judge). Failure degrades to the HNSW order.
        let candidates = self.rerank(request, candidates);

        // Step 3: LLM adjudication over the candidate list.
        if let Some(client) = &self.client {
            match self.adjudicate(client, request, &candidates) {
                Ok(Some((name, gaps))) => {
                    let score = candidates
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map_or(0.0, |(_, s)| *s);
                    let Some(chart) = self.store.get(&name) else {
                        return Ok(ChartMatch::mismatch());
                    };
                    return Ok(self.build_match(&chart, score, entities, &gaps));
                }
                Ok(None) => {
                    // The LLM judged no candidate fits — honor that.
                    return Ok(ChartMatch::mismatch());
                }
                Err(e) => {
                    tracing::warn!(
                        target: "router.charts.select",
                        error = %e,
                        "adjudicator call failed — using top HNSW candidate"
                    );
                }
            }
        }

        // No adjudicator (or it failed): pick the top candidate deterministically.
        match self.store.get(&candidates[0].0) {
            Some(chart) => Ok(self.build_match(&chart, candidates[0].1, entities, &[])),
            None => Ok(ChartMatch::mismatch()),
        }
    }

    /// Step 1: a request that names a chart or one of its provides assets.
    ///
    /// Iterates in a stable (name-sorted) order so selection is reproducible.
    /// Name matches take priority over provides-asset matches; both are
    /// whole-identifier matches (`contains_ident_word` treats `_` as a
    /// boundary, so `bug_triage` is nameable as `bug_triage`).
    fn deterministic_hit(&self, request: &str) -> Option<Arc<ChartDef>> {
        let sorted = self.store.charts_sorted();
        for chart in &sorted {
            if contains_ident_word(request, &chart.name) {
                return Some(chart.clone());
            }
        }
        sorted
            .iter()
            .find(|chart| {
                chart
                    .targets
                    .iter()
                    .any(|t| t.provides.iter().any(|p| contains_ident_word(request, p)))
            })
            .cloned()
    }

    /// Step 2: top-k charts by similarity to the embedded raw request,
    /// filtered by `cfg.min_score`.
    fn retrieve(&self, request: &str) -> Result<Vec<(String, f64)>, ChartError> {
        if !self.store.is_index_built() {
            return Ok(Vec::new());
        }
        let k = self.cfg.max_candidates;
        let hits = self.store.search(request, k)?;
        Ok(hits
            .into_iter()
            .filter(|(_, s)| f64::from(*s) >= self.cfg.min_score)
            .map(|(n, s)| (n, f64::from(s)))
            .collect())
    }

    /// Step 2.5: one LLM call that re-orders the HNSW candidates by relevance
    /// to the raw request. Cross-encoder rerankers are cheap relative to
    /// full adjudication, so this runs before Step 3 (VISION: cheap before
    /// expensive).
    ///
    /// When the LLM reranker is absent but a ColBERT reranker is present,
    /// MaxSim scoring re-ranks the candidates instead.
    ///
    /// Returns the re-ordered candidate list carrying *original* HNSW scores
    /// (the reranker only re-ranks; it cannot fabricate a chart outside the
    /// candidate set). On a missing / failed / unparseable / invalid rerank
    /// call the HNSW order is preserved (Step 3 still adjudicates correctly).
    fn rerank(&self, request: &str, candidates: Vec<(String, f64)>) -> Vec<(String, f64)> {
        // ColBERT reranking: when the LLM reranker is absent, try MaxSim.
        if self.reranker.is_none() {
            if let Some(colbert) = &self.colbert {
                let doc_texts = |name: &str| -> Option<String> {
                    self.store.get(name).map(|c| {
                        let mut text = c.description.clone();
                        text.push(' ');
                        text.push_str(&c.name);
                        for t in &c.targets {
                            for p in &t.provides {
                                text.push(' ');
                                text.push_str(p);
                            }
                        }
                        text
                    })
                };
                let reranked = colbert.rerank_candidates(request, &candidates, &doc_texts);
                if !reranked.is_empty() {
                    tracing::debug!(
                        target: "router.charts.select",
                        candidates = reranked.len(),
                        "colbert MaxSim rerank"
                    );
                    return reranked;
                }
                tracing::warn!(
                    target: "router.charts.select",
                    "colbert rerank returned empty — falling back to HNSW order"
                );
            }
            return candidates;
        }

        let Some(client) = self.reranker.as_ref() else {
            return candidates;
        };
        let prompt = build_rerank_prompt(request, &candidates, &self.store);
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: request.to_string(),
            },
        ];
        tracing::debug!(
            target: "router.charts.select",
            candidates = candidates.len(),
            prompt_len = prompt.len(),
            "reranker LLM request"
        );
        let Ok(response) = client.chat_complete(&messages) else {
            tracing::warn!(
                target: "router.charts.select",
                "reranker call failed — using HNSW candidate order"
            );
            return candidates;
        };
        let Some(names) = parse_rerank_output(&response) else {
            tracing::warn!(
                target: "router.charts.select",
                "reranker response was not parseable JSON — using HNSW candidate order"
            );
            return candidates;
        };
        let score_of = |name: &str| candidates.iter().find(|(n, _)| n == name).map(|(_, s)| *s);
        let mut reordered: Vec<(String, f64)> = Vec::with_capacity(names.len());
        for name in names {
            if let Some(score) = score_of(&name) {
                reordered.push((name, score));
            } else {
                tracing::warn!(
                    target: "router.charts.select",
                    chart = name,
                    "reranker named a chart outside the candidate list"
                );
            }
        }
        // Preserve any candidate the reranker omitted by appending the
        // un-named HNSW candidates back in original order, so the adjudicator
        // still sees the full candidate set.
        for (name, score) in &candidates {
            if !reordered.iter().any(|(n, _)| n == name) {
                reordered.push((name.clone(), *score));
            }
        }
        if reordered.is_empty() {
            candidates
        } else {
            reordered
        }
    }

    /// Step 3: one LLM call over the candidate list → the single best chart
    /// name (plus the LLM's flagged gaps) or `None` (mismatch). The returned
    /// name is validated against the candidate list so a hallucinated chart is
    /// treated as a mismatch.
    fn adjudicate(
        &self,
        client: &Arc<dyn ChatBackend>,
        request: &str,
        candidates: &[(String, f64)],
    ) -> Result<Option<(String, Vec<String>)>, ChartError> {
        let prompt = build_adjudicator_prompt(request, candidates, &self.store);
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: request.to_string(),
            },
        ];
        tracing::debug!(
            target: "router.charts.select",
            candidates = candidates.len(),
            prompt_len = prompt.len(),
            "adjudicator LLM request"
        );
        let response = client
            .chat_complete(&messages)
            .map_err(|e| ChartError::Selection {
                reason: format!("adjudicator call failed: {e}"),
            })?;
        let Some(out) = parse_adjudicator_output(&response) else {
            tracing::warn!(
                target: "router.charts.select",
                "adjudicator response was not parseable JSON"
            );
            return Ok(None);
        };
        if matches!(out.fit, AdjudicatorFit::Mismatch) {
            return Ok(None);
        }
        let Some(name) = out.chart.as_deref() else {
            return Ok(None);
        };
        if !candidates.iter().any(|(n, _)| n == name) {
            tracing::warn!(
                target: "router.charts.select",
                chart = name,
                "adjudicator named a chart outside the candidate list"
            );
            return Ok(None);
        }
        Ok(Some((name.to_string(), out.gaps)))
    }

    /// Build a `ChartMatch` for a chosen chart, resolving any ambiguous
    /// deps first (LLM pick with a deterministic tie-break fallback).
    ///
    /// The fit is the union of the deterministic binding's gaps (unmatched
    /// required deps) and the adjudicator's flagged gaps. The binding is the
    /// authority on executability; the LLM's gaps capture semantic
    /// incompleteness the binding cannot see (interview material).
    fn build_match(
        &self,
        chart: &ChartDef,
        score: f64,
        entities: &[Entity],
        llm_gaps: &[String],
    ) -> ChartMatch {
        let bindings = self.resolve_ambiguity(&bind_chart(chart, entities));
        let mut gaps = bindings.unmatched.clone();
        gaps.extend(llm_gaps.iter().cloned());
        gaps.sort_unstable();
        gaps.dedup();
        let fit = if gaps.is_empty() {
            ChartFit::Exact
        } else {
            ChartFit::Partial { gaps }
        };
        ChartMatch {
            chart: chart.name.clone(),
            score,
            fit,
            bindings: Some(bindings),
        }
    }

    /// Resolve every ambiguous dep to a single bound entity.
    ///
    /// For each `AmbiguousDep`, one LLM call picks a candidate (validated
    /// against the candidate list). A failed/unparseable call — or no LLM
    /// backend at all — falls back to a deterministic tie-break (first, then
    /// lexicographic by id). Other deps' bindings are left untouched, so
    /// resolving one dep never disturbs another's satisfied assets.
    fn resolve_ambiguity(&self, bindings: &Bindings) -> Bindings {
        if bindings.ambiguous.is_empty() {
            return bindings.clone();
        }
        let mut resolved = bindings.clone();
        for amb in &bindings.ambiguous {
            let pick = self
                .pick_ambiguous_candidate(amb)
                .unwrap_or_else(|| deterministic_pick(&amb.candidates));
            resolved
                .satisfied
                .insert(crate::charts::binding::asset_key(&pick));
            resolved
                .entity_map
                .entry(amb.dep.clone())
                .or_default()
                .push(pick);
        }
        resolved.ambiguous.clear();
        resolved
    }

    /// One LLM call over an ambiguous dep's candidate list → the chosen
    /// entity, or `None` when the pick is unparseable / invalid (caller
    /// falls back to the deterministic tie-break).
    fn pick_ambiguous_candidate(&self, amb: &AmbiguousDep) -> Option<Entity> {
        let client = self.client.as_ref()?;
        let prompt = build_ambiguity_prompt(amb);
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: format!("dep: {}\n", amb.dep),
            },
        ];
        tracing::debug!(
            target: "router.charts.select",
            dep = %amb.dep,
            candidates = amb.candidates.len(),
            prompt_len = prompt.len(),
            "ambiguity adjudicator LLM request"
        );
        let response = client.chat_complete(&messages).ok()?;
        let entity_id = parse_ambiguity_output(&response)?;
        let picked = amb.candidates.iter().find(|e| e.id == entity_id).cloned();
        if picked.is_none() {
            tracing::warn!(
                target: "router.charts.select",
                dep = %amb.dep,
                entity_id = %entity_id,
                "ambiguity adjudicator named a non-candidate — deterministic fallback"
            );
        }
        picked
    }
}

/// Deterministic tie-break for an ambiguous dep: first, then lexicographic
/// by id. Candidates are never empty (ambiguity implies >= 2 matches).
fn deterministic_pick(candidates: &[Entity]) -> Entity {
    candidates
        .iter()
        .min_by(|a, b| a.id.cmp(&b.id))
        .expect("ambiguous dep has at least one candidate")
        .clone()
}

// ── Adjudicator prompt + strict-JSON parsing ─────────────────────────────

/// The candidate's `fit` as judged by the LLM. A hint: final `ChartFit` is
/// derived from entity binding so a hallucinated "exact" never overrides a
/// demonstrably unbound dep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdjudicatorFit {
    Exact,
    Partial,
    Mismatch,
}

/// Parsed adjudicator output.
#[derive(Debug, Clone)]
struct AdjudicatorOutput {
    chart: Option<String>,
    fit: AdjudicatorFit,
    gaps: Vec<String>,
}

/// Build the adjudicator system prompt from the candidate list. Mirrors
/// `ClassifierStage::build_system_prompt`: the candidate list is
/// auto-generated from the store (no hand-maintained copy) and the output
/// schema is spelled out so small models can comply.
fn build_adjudicator_prompt(
    request: &str,
    candidates: &[(String, f64)],
    store: &ChartStore,
) -> String {
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "You are a chart selector. Given a user request and candidate charts, \
         choose the single best chart, or none."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "User request: {request}");
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Candidate charts:");
    for (i, (name, score)) in candidates.iter().enumerate() {
        let Some(chart) = store.get(name) else {
            continue;
        };
        let provides: Vec<&str> = chart
            .targets
            .iter()
            .flat_map(|t| t.provides.iter().map(String::as_str))
            .collect();
        let _ = writeln!(prompt, "{}. name: \"{name}\"", i + 1);
        let _ = writeln!(prompt, "   description: \"{}\"", chart.description);
        let _ = writeln!(prompt, "   provides: [{}]", provides.join(", "));
        let _ = writeln!(prompt, "   similarity: {score:.2}");
    }
    let _ = writeln!(prompt);
    let _ = writeln!(
        prompt,
        "Output ONLY this JSON object:\n\
         {{\"chart\": \"<candidate name> or null\", \
         \"fit\": \"exact\" | \"partial\" | \"mismatch\", \
         \"gaps\": [\"<missing input>\"]}}"
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Rules:");
    let _ = writeln!(prompt, "- exact: the chart fully covers the request.");
    let _ = writeln!(
        prompt,
        "- partial: the chart is the best fit but some inputs are missing; list them in gaps."
    );
    let _ = writeln!(prompt, "- mismatch: no chart fits the request.");
    let _ = writeln!(prompt, "- Only output the JSON object, no other text.");
    prompt
}

/// Parse an adjudicator response into a `ChartMatch`-shaped verdict.
///
/// Tolerant by design (mirrors `parse_classifier_response`): the shared
/// `fluent_llm::parse_typed` codec strips markdown code fences, fast-paths
/// a direct parse, then extracts the first `{...}` object; missing fields are
/// sanitized to defaults.
fn parse_adjudicator_output(raw: &str) -> Option<AdjudicatorOutput> {
    let value =
        fluent_llm::parse_typed::<serde_json::Value>(raw, &serde_json::Value::Null, |_| {}).ok()?;
    sanitize_adjudicator_output(&value)
}

/// Fill missing adjudicator fields with defaults (sanitize philosophy).
fn sanitize_adjudicator_output(value: &serde_json::Value) -> Option<AdjudicatorOutput> {
    let obj = value.as_object()?;
    let chart: Option<String> = obj
        .get("chart")
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let fit = match obj.get("fit").and_then(|f| f.as_str()) {
        Some("exact") => AdjudicatorFit::Exact,
        Some("partial") => AdjudicatorFit::Partial,
        Some("mismatch" | "none") => AdjudicatorFit::Mismatch,
        _ => {
            if chart.is_some() {
                AdjudicatorFit::Exact
            } else {
                AdjudicatorFit::Mismatch
            }
        }
    };
    let gaps: Vec<String> = obj
        .get("gaps")
        .and_then(|g| g.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Some(AdjudicatorOutput { chart, fit, gaps })
}

// ── Reranker (Step 2.5): LLM candidate re-ordering ──────────────────────

/// Build the reranker system prompt from the candidate list. The reranker
/// re-orders candidates by relevance (it does not pick a winner — that is the
/// adjudicator's job), and its output is an array of candidate names.
fn build_rerank_prompt(request: &str, candidates: &[(String, f64)], store: &ChartStore) -> String {
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "You are a chart reranker. Given a user request and candidate charts, \
         rank the candidates by relevance to the request, most relevant first."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "User request: {request}");
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Candidate charts:");
    for (i, (name, score)) in candidates.iter().enumerate() {
        let Some(chart) = store.get(name) else {
            continue;
        };
        let provides: Vec<&str> = chart
            .targets
            .iter()
            .flat_map(|t| t.provides.iter().map(String::as_str))
            .collect();
        let _ = writeln!(prompt, "{}. name: \"{name}\"", i + 1);
        let _ = writeln!(prompt, "   description: \"{}\"", chart.description);
        let _ = writeln!(prompt, "   provides: [{}]", provides.join(", "));
        let _ = writeln!(prompt, "   similarity: {score:.2}");
    }
    let _ = writeln!(prompt);
    let _ = writeln!(
        prompt,
        "Output ONLY this JSON array of candidate names, most relevant first:\n\
         [\"<candidate name>\", ...]"
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Rules:");
    let _ = writeln!(prompt, "- Rank every candidate exactly once.");
    let _ = writeln!(prompt, "- Only output the JSON array, no other text.");
    prompt
}

/// Parse a reranker response into an ordered list of candidate names.
///
/// Tolerant by design (mirrors `parse_adjudicator_output`): the shared
/// `fluent_llm::parse_typed` codec strips fences, accepts a bare array or
/// a `{"ranking": [...]}` object, and extracts the first balanced JSON value.
/// Non-string entries are dropped. `None` means "not parseable" — the caller
/// keeps the HNSW order.
fn parse_rerank_output(raw: &str) -> Option<Vec<String>> {
    let value =
        fluent_llm::parse_typed::<serde_json::Value>(raw, &serde_json::Value::Null, |_| {}).ok()?;
    let names: Option<Vec<&str>> = match &value {
        serde_json::Value::Array(arr) => Some(arr.iter().filter_map(|v| v.as_str()).collect()),
        serde_json::Value::Object(obj) => obj
            .get("ranking")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect()),
        _ => None,
    };
    let names = names?;
    if names.is_empty() {
        return None;
    }
    Some(names.into_iter().map(str::to_string).collect())
}

// ── Ambiguity adjudicator: LLM pick + deterministic tie-break ───────

/// Build the ambiguity-adjudicator system prompt for a single dep.
///
/// The candidate entities are summarized (id, kind, and the value as JSON)
/// and the output schema is spelled out so small models can comply.
fn build_ambiguity_prompt(amb: &AmbiguousDep) -> String {
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "You are a resolver. A workflow dependency matched multiple context \
         entities; choose the single best one."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Dependency: {}", amb.dep);
    if amb.description != amb.dep {
        let _ = writeln!(prompt, "Description: {}", amb.description);
    }
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Candidate entities:");
    for (i, e) in amb.candidates.iter().enumerate() {
        let _ = writeln!(prompt, "{}. id: \"{}\"", i + 1, e.id);
        let _ = writeln!(prompt, "   kind: \"{}\"", e.kind);
        let _ = writeln!(
            prompt,
            "   value: {}",
            serde_json::to_string(&e.value).unwrap_or_default()
        );
    }
    let _ = writeln!(prompt);
    let _ = writeln!(
        prompt,
        "Output ONLY this JSON object:\n{{\"entity_id\": \"<candidate id>\"}}"
    );
    let _ = writeln!(prompt, "Rules:");
    let _ = writeln!(
        prompt,
        "- entity_id must be one of the candidate ids above."
    );
    let _ = writeln!(prompt, "- Only output the JSON object, no other text.");
    prompt
}

/// Parse an ambiguity-adjudicator response into a candidate entity id.
///
/// Tolerant by design: the shared `fluent_llm::parse_typed` codec strips
/// code fences and extracts the first `{...}` object. Returns `None` when the
/// id is missing or empty — the caller falls back to the deterministic
/// tie-break.
fn parse_ambiguity_output(raw: &str) -> Option<String> {
    let value =
        fluent_llm::parse_typed::<serde_json::Value>(raw, &serde_json::Value::Null, |_| {}).ok()?;
    let id = value
        .as_object()?
        .get("entity_id")
        .and_then(|v| v.as_str())?
        .trim();
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}
#[cfg(test)]
#[path = "../../tests/charts_select.rs"]
mod tests;
