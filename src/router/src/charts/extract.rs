//! Chart auto-extraction from dispatch audit transcripts.
//!
//! A successful high-capability (frontier/local) solution is distilled into a
//! named chart — each LLM call / transform becomes a `ChartTarget` with a
//! template capturing the prompt shape and `depends`/`provides` edges from the
//! actual data flow. Extraction is a best-effort *deterministic*
//! reconstruction (no LLM): the chart is written as a **draft** that only
//! becomes selectable after a rubric-validated run (M9 gate).

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
    /// Model key that authored the solution (staleness tracking, M10).
    #[serde(default)]
    pub author_model: String,
    /// The discrete steps, in execution order.
    pub steps: Vec<ChartAuditStep>,
}

/// Best-effort deterministic reconstruction of a `ChartDef` from an audit
/// transcript (M10).
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
///   model for the M10 staleness policy.
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

/// Slugify a query into a valid chart name: lowercase alphanumerics and
/// `_`/`-` only, truncated to `CHART_EXTRACTED_NAME_MAX_CHARS`.
pub fn slugify_chart_name(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    for c in query.chars().map(|c| c.to_ascii_lowercase()) {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out.truncate(super::CHART_EXTRACTED_NAME_MAX_CHARS);
    out
}

/// Build the audit-shaped transcript for a single successful dispatch
/// (the current dispatch chain is one buffered LLM call, so the faithful
/// decomposition is one step). The `prompt` is the **actual** prompt sent
/// to the model (M10a LOD0 fidelity) — it captures the real LOD0 prompt
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

/// The M10 dispatch post-processing hook: distills successful dispatches into
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
///   run promotes it (M9 gate).
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

    /// Set the extraction scope (M10b): `Frontier` (default) distills only
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
mod tests {
    use super::*;
    use crate::charts::store::ChartStore;

    fn transcript() -> ChartAuditTranscript {
        ChartAuditTranscript {
            query: "Draft a release plan for the v2 API".into(),
            author_model: "claude-4".into(),
            steps: vec![
                ChartAuditStep {
                    id: "plan".into(),
                    purpose: "outline the release steps".into(),
                    prompt: "Draft a release plan for the v2 API: list phases and owners.".into(),
                    response: "Phase 1: ...".into(),
                    depends_on: vec![],
                    provides: vec!["release_plan".into()],
                },
                ChartAuditStep {
                    id: "verify".into(),
                    purpose: "check the plan is complete".into(),
                    prompt: "Given the release plan, verify it covers rollback.".into(),
                    response: "Add rollback step.".into(),
                    depends_on: vec!["plan".into()],
                    provides: vec!["verified_plan".into()],
                },
            ],
        }
    }

    #[test]
    fn extracts_valid_chart_from_transcript() {
        let chart = extract_chart_from_audit(&transcript()).expect("extracts");
        chart.validate().expect("draft validates");
        assert_eq!(chart.author_model, "claude-4");
        assert_eq!(chart.targets.len(), 2);
        assert_eq!(chart.targets[0].name, "plan");
        assert_eq!(chart.targets[1].name, "verify");
        // depends_on edge becomes a Capability dep on the upstream step id.
        match &chart.targets[1].depends[0] {
            DepSpec::Capability { name } => assert_eq!(name, "plan"),
            other @ DepSpec::EntityMatch { .. } => panic!("expected capability dep, got {other:?}"),
        }
        // The query text in the prompt is replaced with {{ request }}.
        assert!(chart.targets[0].template.contains("{{ request }}"));
        assert!(!chart.targets[0].template.contains("v2 API"));
    }

    #[test]
    fn empty_query_is_rejected() {
        let mut t = transcript();
        t.query = "!!".into();
        let err = extract_chart_from_audit(&t).unwrap_err();
        assert!(matches!(err, ChartError::Invalid { .. }));
    }

    #[test]
    fn no_steps_is_rejected() {
        let mut t = transcript();
        t.steps.clear();
        let err = extract_chart_from_audit(&t).unwrap_err();
        assert!(matches!(err, ChartError::Invalid { .. }));
    }

    #[test]
    fn prompt_without_query_appends_request() {
        let t = template_from_prompt("Produce a checklist.", "write a checklist");
        assert!(t.contains("{{ request }}"));
        assert!(t.contains("Produce a checklist."));
    }

    #[test]
    fn prompt_repeating_the_query_replaces_only_first_occurrence() {
        // Only the *first* occurrence is substituted; a prompt that repeats
        // the query leaves later occurrences literal.
        let t = template_from_prompt(
            "write a checklist for write a checklist",
            "write a checklist",
        );
        assert_eq!(t, "{{ request }} for write a checklist");
    }

    #[test]
    fn slugify_normalizes_and_truncates() {
        assert_eq!(slugify_chart_name("Hello, World!"), "hello_world");
        assert_eq!(
            slugify_chart_name("  multiple   spaces  "),
            "multiple_spaces"
        );
        assert_eq!(slugify_chart_name("already-kebab"), "already-kebab");
        assert!(
            slugify_chart_name(&"a".repeat(200)).len()
                <= super::super::CHART_EXTRACTED_NAME_MAX_CHARS
        );
    }

    #[test]
    fn every_target_self_provides_its_id() {
        let mut t = transcript();
        for step in &mut t.steps {
            step.provides.clear();
        }
        let chart = extract_chart_from_audit(&t).expect("extracts");
        // The DependencySession self-provide convention keeps the draft
        // selectable even when the transcript records no explicit provides.
        for target in &chart.targets {
            assert!(
                target.provides.iter().any(|p| p == &target.name),
                "target '{}' must self-provide its id: {:?}",
                target.name,
                target.provides
            );
        }
    }

    // ── M10: WorkflowExtractor (dispatch post-processing hook) ───────────

    #[test]
    fn transcript_from_dispatch_produces_single_step() {
        let t = transcript_from_dispatch(
            "write a release plan",
            "system: You are a planner.\nuser: write a release plan",
            "claude-4",
            "Phase 1: ...",
        );
        assert_eq!(t.query, "write a release plan");
        assert_eq!(t.author_model, "claude-4");
        assert_eq!(t.steps.len(), 1);
        assert_eq!(t.steps[0].id, "solve");
        // M10a LOD0 fidelity: the step prompt is the real prompt that was
        // sent to the model — no synthesized "Solve the following request…"
        // wrapper.
        assert_eq!(
            t.steps[0].prompt,
            "system: You are a planner.\nuser: write a release plan"
        );
        assert!(!t.steps[0].prompt.contains("Solve the following request"));
        let chart = extract_chart_from_audit(&t).expect("extracts");
        assert!(
            chart.targets[0].template.contains("{{ request }}"),
            "query text must become a request placeholder"
        );
        // The template captures the real LOD0 prompt shape: the system line
        // is preserved verbatim, only the query is substituted.
        assert_eq!(
            chart.targets[0].template,
            "system: You are a planner.\nuser: {{ request }}"
        );
    }

    #[test]
    fn extractor_disabled_is_a_noop() {
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone());
        let outcome = extractor
            .extract_from_transcript(&transcript())
            .expect("disabled extraction never fails");
        assert!(outcome.is_none());
        assert!(store.is_empty(), "disabled extractor must not write charts");
    }

    #[test]
    fn extractor_enabled_writes_draft_chart() {
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
        let outcome = extractor
            .extract_from_transcript(&transcript())
            .expect("extracts")
            .expect("enabled extractor writes");
        assert_eq!(outcome, UpsertOutcome::Inserted);
        assert_eq!(store.len(), 1);
        let name = slugify_chart_name(&transcript().query);
        assert!(
            store.get(&name).is_some(),
            "draft chart stored under its slug"
        );
        assert!(store.is_draft(&name), "extracted chart is a draft");
    }

    #[test]
    fn extractor_record_success_swallows_extraction_failure() {
        // A query that slugs to nothing must not panic or propagate — the
        // request already succeeded and the learning loop is best-effort.
        // `is_fallback = true` so the extraction is attempted under the
        // default `Frontier` mode (the swallow path is what we test).
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
        extractor.record_success("!!!", "user: !!!", "claude-4", "some answer", true);
        assert!(store.is_empty());
    }

    // ── M10b: extraction scope (frontier-assisted only by default) ────────

    #[test]
    fn frontier_mode_skips_primary_dispatch() {
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
        extractor.record_success(
            "write a release plan",
            "user: write a release plan",
            "claude-4",
            "Phase 1: ...",
            false,
        );
        assert!(
            store.is_empty(),
            "a primary-target success must not be distilled under Frontier mode"
        );
    }

    #[test]
    fn frontier_mode_extracts_fallback_dispatch() {
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
        extractor.record_success(
            "write a release plan",
            "user: write a release plan",
            "claude-4",
            "Phase 1: ...",
            true,
        );
        assert_eq!(store.len(), 1, "a fallback dispatch is distilled");
    }

    #[test]
    fn all_mode_preserves_blanket_extraction() {
        let store = Arc::new(ChartStore::new(None));
        let extractor = WorkflowExtractor::new(store.clone())
            .enabled(true)
            .with_extraction_mode(WorkflowExtractionMode::All);
        extractor.record_success(
            "write a release plan",
            "user: write a release plan",
            "claude-4",
            "Phase 1: ...",
            false,
        );
        assert_eq!(
            store.len(),
            1,
            "mode \"all\" keeps distilling primary-target successes"
        );
    }
}
