//! Chart auto-extraction from dispatch audit transcripts.
//!
//! A successful high-capability (frontier/local) solution is distilled into a
//! named chart — each LLM call / transform becomes a `ChartTarget` with a
//! template capturing the prompt shape and `depends`/`provides` edges from the
//! actual data flow. Extraction is a best-effort *deterministic*
//! reconstruction (no LLM): the chart is written as a **draft** that only
//! becomes selectable after a rubric-validated run.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::WorkflowExtractionMode;

use super::store::{ChartStore, UpsertOutcome};
use super::{ChartDef, ChartError, ChartTarget, DepSpec};

/// A single step in a solved dispatch chain — one LLM call or deterministic
/// transform. This is the audit-shaped record consumed by
/// [`extract_chart_from_audit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAuditStep {
    /// Stable step id (becomes the chart-target name).
    pub id: String,
    /// Human-readable purpose label.
    pub purpose: String,
    /// The prompt actually sent to the model.
    pub prompt: String,
    /// The model's response.
    pub response: String,
    /// Step ids this step read from (data-flow edges).
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Asset names this step produced.
    #[serde(default)]
    pub provides: Vec<String>,
}

/// The solved chain: the original query plus its discrete steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAuditTranscript {
    /// The user's original request.
    pub query: String,
    /// Model key that authored the solution (staleness tracking).
    #[serde(default)]
    pub author_model: String,
    /// The discrete steps, in execution order.
    pub steps: Vec<ChartAuditStep>,
}

/// ROADMAP §12.7 (C7): adapt a `parse_review` ledger node into a
/// [`ChartAuditStep`], so a session whose transcript chains `parse → review →
/// dispatch` folds the review into [`ChartAuditTranscript`] and
/// [`extract_chart_from_audit`] distils the whole chain as a draft chart.
/// Returns `None` for any non-`parse_review` node.
pub fn parse_review_step(node: &fluent_types::ContentNode) -> Option<ChartAuditStep> {
    let meta = node.metadata.as_ref()?;
    if meta.get("kind")?.as_str() != Some(crate::ledger::nlp::PARSE_REVIEW_KIND) {
        return None;
    }
    let node_id = node.id?.as_int();
    let source_node_id = meta.get("source_node_id")?.as_i64()?;
    Some(ChartAuditStep {
        id: format!("review:{node_id}"),
        purpose: "parse review".into(),
        prompt: meta.get("prompt")?.as_str()?.to_string(),
        response: meta.get("corrections_json")?.as_str()?.to_string(),
        depends_on: vec![format!("parse:{source_node_id}")],
        provides: vec!["reviewed_parse".into()],
    })
}

/// Best-effort deterministic reconstruction of a `ChartDef` from an audit
/// transcript.
///
/// - Chart name is a slug of the query (truncated to
///   `CHART_EXTRACTED_NAME_MAX_CHARS`); a degenerate slug is a hard
///   `ChartError::Invalid` — an unnameable chart must not be written.
/// - Each step becomes a `ChartTarget`: the step's `prompt` is turned into a
///   minijinja template (the original query text is replaced with
///   `{{ request }}` where found, else the request is appended), `depends_on`
///   maps to `Capability` deps on the upstream step ids, and `provides`
///   carries the step's assets (plus the step id, the self-provide
///   convention).
/// - The chart is a *draft*: it self-provides `draft_output` only when no
///   step already provides an asset, and `author_model` records the authoring
///   model for the staleness policy.
pub fn extract_chart_from_audit(transcript: &ChartAuditTranscript) -> Result<ChartDef, ChartError> {
    let name = slugify_chart_name(&transcript.query);
    if name.is_empty() {
        return Err(ChartError::Invalid {
            reason: "cannot auto-extract a chart: query produced an empty name".into(),
        });
    }

    if transcript.steps.is_empty() {
        return Err(ChartError::Invalid {
            reason: format!("cannot auto-extract chart '{name}': transcript has no steps"),
        });
    }

    let mut targets: Vec<ChartTarget> = Vec::with_capacity(transcript.steps.len());
    for step in &transcript.steps {
        let mut provides = step.provides.clone();
        if !provides.contains(&step.id) {
            provides.push(step.id.clone());
        }
        let depends: Vec<DepSpec> = step
            .depends_on
            .iter()
            .map(|dep| DepSpec::Capability { name: dep.clone() })
            .collect();
        targets.push(ChartTarget {
            name: step.id.clone(),
            provides,
            depends,
            template: template_from_prompt(&step.prompt, &transcript.query),
            essential: true,
            rubric: None,
        });
    }

    let chart = ChartDef {
        name,
        description: format!(
            "Auto-extracted from a solved dispatch: {}",
            transcript.query
        ),
        schema_version: super::CHART_SCHEMA_VERSION,
        author_model: if transcript.author_model.is_empty() {
            "frontier".to_string()
        } else {
            transcript.author_model.clone()
        },
        targets,
        rubric: None,
    };

    // Validate the draft: an auto-extracted chart that fails the content
    // model is discarded (a broken draft must not be persisted).
    chart.validate().map_err(|e| ChartError::Invalid {
        reason: format!(
            "auto-extracted chart '{}' failed validation: {e}",
            chart.name
        ),
    })?;
    Ok(chart)
}
/// Turn a step's concrete prompt into a reusable minijinja template.
///
/// **First-occurrence only**: if the query appears in the prompt, only the
/// **first** occurrence is replaced with `{{ request }}` — a prompt that
/// repeats the query leaves later occurrences literal. Otherwise the request
/// is appended as a final line.
///
/// **Draft fidelity**: the extracted chart is a draft, gated by a
/// rubric-validated run before promotion, so a lower-fidelity template is
/// acceptable.
///
/// This is a deterministic best-effort reconstruction — it never calls an
/// LLM. Entity/dep placeholders are intentionally left as-is (the operator
/// or a later extraction pass can tighten them).
pub(crate) fn template_from_prompt(prompt: &str, query: &str) -> String {
    if !query.is_empty() {
        if let Some(pos) = prompt.find(query) {
            let mut t = String::with_capacity(prompt.len() + 16);
            t.push_str(&prompt[..pos]);
            t.push_str("{{ request }}");
            t.push_str(&prompt[pos + query.len()..]);
            return t;
        }
    }
    format!("{prompt}\nUser request: {{{{ request }}}}")
}

/// Slug options reproducing [`slugify_chart_name`] byte-for-byte (P4).
/// Domain constant: stays router-side, composed over
/// `common_core::string::slugify_with`.
pub const CHART_SLUG_OPTIONS: common_core::string::SlugOptions =
    common_core::string::SlugOptions {
        ascii_only: true,
        separator: '_',
        collapse_runs: true,
        trim_leading: false,
        trim_trailing: true,
        max_chars: Some(super::CHART_EXTRACTED_NAME_MAX_CHARS),
    };

/// Slugify a query into a valid chart name: lowercase alphanumerics and
/// `_`/`-` only, truncated to `CHART_EXTRACTED_NAME_MAX_CHARS`.
///
/// Thin wrapper over `common_core::string::slugify_with` with
/// [`CHART_SLUG_OPTIONS`] (parity locked by
/// `slugify_chart_name_characterization_table`).
pub fn slugify_chart_name(query: &str) -> String {
    common_core::string::slugify_with(query, &CHART_SLUG_OPTIONS)
}

/// Build the audit-shaped transcript for a single successful dispatch
/// (the current dispatch chain is one buffered LLM call, so the faithful
/// decomposition is one step). The `prompt` is the **actual** prompt sent
/// to the model (LOD0 fidelity) — it captures the real LOD0 prompt
/// shape so `template_from_prompt` produces a faithful template, instead of
/// synthesizing a "Solve the following request…" wrapper.
pub fn transcript_from_dispatch(
    query: &str,
    prompt: &str,
    author_model: &str,
    response: &str,
) -> ChartAuditTranscript {
    ChartAuditTranscript {
        query: query.to_string(),
        author_model: author_model.to_string(),
        steps: vec![ChartAuditStep {
            id: "solve".into(),
            purpose: "solve the request".into(),
            prompt: prompt.to_string(),
            response: response.to_string(),
            depends_on: vec![],
            provides: vec!["solution".into()],
        }],
    }
}

/// The dispatch post-processing hook: distills successful dispatches into
/// draft charts in the shared `ChartStore`.
///
/// Best-effort by design (VISION §"Post-processing: audit + workflow
/// extraction"):
///
/// - Disabled unless `enabled(true)` (the operator's
///   `post_process.workflow_extraction` flag) — then it is a no-op that never
///   fails a request.
/// - Extraction is a *deterministic* reconstruction (no LLM); the written
///   chart is a **draft**, excluded from selection until a rubric-validated
///   run promotes it.
/// - Writes are idempotent: a near-neighbor chart in the `workflow_library`
///   index subsumes the new draft instead of duplicating it (VISION's rule),
///   and the subsumed chart's `author_model` is refreshed — a newer, stronger
///   model re-authors a stale chart.
pub struct WorkflowExtractor {
    store: Arc<ChartStore>,
    enabled: bool,
    near_threshold: f32,
    mode: WorkflowExtractionMode,
}

impl WorkflowExtractor {
    pub fn new(store: Arc<ChartStore>) -> Self {
        Self {
            store,
            enabled: false,
            near_threshold: super::store::CHART_SUBSUME_THRESHOLD,
            mode: WorkflowExtractionMode::default(),
        }
    }

    /// Opt in to extraction. Off by default — no-op until enabled.
    #[must_use]
    pub fn enabled(mut self, on: bool) -> Self {
        self.enabled = on;
        self
    }

    /// Override the near-neighbor subsumption threshold.
    #[must_use]
    pub fn with_near_threshold(mut self, threshold: f32) -> Self {
        self.near_threshold = threshold;
        self
    }

    /// Set the extraction scope: `Frontier` (default) distills only
    /// frontier-assisted (escalated/fallback) dispatches; `All` restores the
    /// blanket behavior.
    #[must_use]
    pub fn with_extraction_mode(mut self, mode: WorkflowExtractionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Distill `transcript` into a draft chart and upsert it idempotently.
    ///
    /// Returns `Ok(None)` when extraction is disabled (the common case) and
    /// `Ok(Some(outcome))` on a successful write. A best-effort extraction
    /// that fails (e.g. an unnameable query) is logged and swallowed — the
    /// learning loop must never break the request path.
    pub fn extract_from_transcript(
        &self,
        transcript: &ChartAuditTranscript,
    ) -> Result<Option<UpsertOutcome>, ChartError> {
        if !self.enabled {
            return Ok(None);
        }
        let chart = extract_chart_from_audit(transcript)?;
        let outcome = self.store.upsert_idempotent(chart, self.near_threshold)?;
        crate::audit::emit(
            "chart_extract",
            serde_json::json!({
                "outcome": outcome,
                "author_model": transcript.author_model,
            }),
        );
        Ok(Some(outcome))
    }

    /// Record a successful dispatch: builds the transcript and extracts.
    /// Never fails — errors are logged, not propagated (the request already
    /// succeeded; extraction must not turn it into a failure).
    ///
    /// `is_fallback` reports whether the successful response came from an
    /// escalated/fallback target (an index > 0 in the dispatch chain). In
    /// `Frontier` mode, a primary-target success (the common case) is not
    /// distilled into a chart — the VISION learning loop trends frontier-call
    /// frequency *down*, so it learns from frontier-assisted solutions only.
    pub fn record_success(
        &self,
        query: &str,
        prompt: &str,
        author_model: &str,
        response: &str,
        is_fallback: bool,
    ) {
        if !self.enabled {
            return;
        }
        if self.mode == WorkflowExtractionMode::Frontier && !is_fallback {
            return;
        }
        let transcript = transcript_from_dispatch(query, prompt, author_model, response);
        match self.extract_from_transcript(&transcript) {
            Ok(_) => {}
            Err(e) => crate::audit::emit(
                "chart_extract",
                serde_json::json!({ "error": e.to_string() }),
            ),
        }
    }
}
#[cfg(test)]
#[path = "../../tests/charts_extract.rs"]
mod tests;
