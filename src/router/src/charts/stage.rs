//! ChartPromptStage — a `ClassifierStage`-shaped component that renders a
//! chart target's minijinja template at execution time and makes one LLM call
//! through the injected `ChatBackend`.
//!
//! It mirrors `ClassifierStage` (same shape: `name`, `Arc<dyn ChatBackend>`,
//! `Arc<Limiter>`, `depends`/`provides` as `ArcIntern<str>`,
//! `impl_component!`) but keeps `execute` synchronous and non-blocking —
//! timeout/retry/cancellation belong to the Zone supervisor (M9).

use std::collections::HashMap;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;
use guidance_llm::ChatMessage;

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

use super::binding::{bind_entities, parse_entities_from_ctx};
use super::render::{render, BoundEntity, RenderContext};
use super::DepSpec;

/// Metadata key under which the stage records the raw LLM text (mirrors the
/// classifier's `response` handoff key).
pub const CHART_RESPONSE_META_KEY: &str = "response";
/// Metadata key under which the stage records the parsed structured output —
/// the `provides` payload read by upstream targets via `stage.{id}.output`.
pub const CHART_OUTPUT_META_KEY: &str = "output";
/// Metadata key under which the stage records the chart name (provenance).
pub const CHART_NAME_META_KEY: &str = "chart_name";
/// Metadata key under which the stage records the chart-target name.
pub const CHART_TARGET_META_KEY: &str = "chart_target";

/// A single chart-target prompt stage.
pub struct ChartPromptStage {
    /// Stage id == chart-target name (the self-provided asset).
    name: ArcIntern<str>,
    /// Chart name (provenance).
    chart_name: String,
    /// Injected LLM backend (mock-injectable).
    client: Arc<dyn ChatBackend>,
    /// Bounds concurrent chart LLM calls.
    limiter: Arc<Limiter>,
    /// The target's minijinja template.
    template: String,
    /// The target's structured dependency specs — re-bound at execution time
    /// so the rendered prompt carries the right entity preamble.
    depends_specs: Vec<DepSpec>,
    /// Upstream chart-target stage ids whose `output` this target reads via
    /// the `stage.{id}.output` metadata mirror.
    upstream_ids: Vec<String>,
    /// Concrete asset names this stage depends on (the capability deps the
    /// upstream targets provide) — drives `PipelineGraph` topo-order.
    depends: Vec<ArcIntern<str>>,
    /// Concrete asset names this stage provides (target `provides` + the
    /// self-provided target name, deduplicated).
    provides: Vec<ArcIntern<str>>,
}

impl ChartPromptStage {
    /// Construct a stage for one chart target.
    ///
    /// `target` is the chart-target name (== stage id == self-provided
    /// asset). `depends` must be the concrete asset names the upstream
    /// targets provide so `PipelineGraph` topo-sorts correctly; `provides`
    /// must be the target's `provides` list plus the target name itself.
    pub fn new(
        client: Arc<dyn ChatBackend>,
        limiter: Arc<Limiter>,
        target: impl Into<String>,
        chart_name: impl Into<String>,
        template: impl Into<String>,
        depends_specs: Vec<DepSpec>,
        upstream_ids: Vec<String>,
        depends: Vec<ArcIntern<str>>,
        provides: Vec<ArcIntern<str>>,
    ) -> Self {
        Self {
            name: ArcIntern::from(target.into()),
            chart_name: chart_name.into(),
            client,
            limiter,
            template: template.into(),
            depends_specs,
            upstream_ids,
            depends,
            provides,
        }
    }

    /// The chart-target name (== stage id).
    pub fn target(&self) -> &str {
        &self.name
    }

    /// The chart name (provenance).
    pub fn chart_name(&self) -> &str {
        &self.chart_name
    }
}

impl WorkUnit for ChartPromptStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let request = extract_user_message(ctx)?;
        let entities = parse_entities_from_ctx(ctx);
        let bindings = bind_entities(&self.depends_specs, &entities);

        // Compile (M5) already guarantees a fully-bound chart, but execution
        // re-binds in case the ctx entities differ from compile time. Failing
        // closed here beats rendering a prompt with missing inputs.
        if !bindings.unmatched.is_empty() {
            return Err(WorkError::Execution(format!(
                "chart target '{}' has unmatched required deps: {:?}",
                self.name, bindings.unmatched
            )));
        }
        if !bindings.ambiguous.is_empty() {
            let names: Vec<&str> = bindings.ambiguous.iter().map(|a| a.dep.as_str()).collect();
            return Err(WorkError::Execution(format!(
                "chart target '{}' has ambiguous deps: {:?}",
                self.name, names
            )));
        }

        let deps: HashMap<String, Vec<BoundEntity>> = bindings
            .entity_map
            .into_iter()
            .map(|(dep, entities)| {
                let bound = entities
                    .into_iter()
                    .map(|e| BoundEntity {
                        id: e.id,
                        kind: e.kind,
                        value: e.value,
                    })
                    .collect();
                (dep, bound)
            })
            .collect();

        let render_ctx = RenderContext {
            request: request.clone(),
            deps,
            upstream: read_metadata_json(ctx, &self.upstream_ids),
            chart: self.chart_name.clone(),
        };

        let system_prompt = render(&self.template, &render_ctx)
            .map_err(|e| WorkError::Execution(e.to_string()))?;

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system_prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: request,
            },
        ];

        tracing::info!(
            target: "router.charts.stage",
            chart = %self.chart_name,
            target = %self.name,
            prompt_len = system_prompt.len(),
            "chart target LLM request"
        );

        let response = self
            .limiter
            .run_sync(|| async { self.client.chat_complete(&messages) })
            .map_err(|e| {
                WorkError::Execution(format!(
                    "chart target '{}' LLM call failed: {e}",
                    self.name
                ))
            })?;

        let output = parse_output(&response);

        tracing::info!(
            target: "router.charts.stage",
            chart = %self.chart_name,
            target = %self.name,
            response_len = response.len(),
            "chart target LLM call succeeded"
        );

        let decision = StageDecision {
            stage: PipelineStage::Classifier,
            verdict: StageVerdict::Passed,
            score: None,
            reason: format!("chart target '{}' completed", self.name),
            latency_ms: 0,
            metadata: serde_json::json!({
                CHART_RESPONSE_META_KEY: response,
                CHART_OUTPUT_META_KEY: output,
                CHART_TARGET_META_KEY: self.name(),
                CHART_NAME_META_KEY: self.chart_name,
            }),
        };

        WorkOutput::typed("chart_target_completed", &decision)
    }
}

impl FieldAccess for ChartPromptStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "ChartPromptStage has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "ChartPromptStage has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for ChartPromptStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "chart_name": self.chart_name,
                "target": self.name(),
                "depends": self.depends.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "provides": self.provides.iter().map(ToString::to_string).collect::<Vec<_>>(),
            },
            "required": []
        })
    }
}

impl_component!(ChartPromptStage);

// ── helpers ──────────────────────────────────────────────────────────────

/// Read prior chart-target outputs from the `stage.{id}.output` metadata
/// mirror. `PipelineGraph` promotes each chart stage's decision metadata
/// under `stage.{stage_id}.` and serializes non-scalar values (like the
/// `output` object) to JSON strings so they survive the scalar-only
/// `MetadataValue` boundary.
///
/// The returned map wraps each output under an `output` key so templates
/// access it as `upstream.<stage_id>.output` (the documented contract in
/// `render.rs` and the Appendix A seed charts).
fn read_metadata_json(
    ctx: &WorkContext,
    upstream_ids: &[String],
) -> HashMap<String, serde_json::Value> {
    let mut upstream = HashMap::new();
    for id in upstream_ids {
        let key = format!("stage.{id}.{CHART_OUTPUT_META_KEY}");
        if let Some(MetadataValue::String(raw)) = ctx.metadata.get(&key) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
                upstream.insert(id.clone(), serde_json::json!({ CHART_OUTPUT_META_KEY: v }));
            }
        }
    }
    upstream
}

/// Parse a chart-target LLM response into a structured `output` value.
///
/// Sanitization policy: trim, strip common markdown code fences
/// (```` ```json ```` / ```` ``` ````), then parse as JSON. Unparseable
/// responses fall back to a string leaf so the raw text still flows to
/// upstream targets via `stage.{id}.output`.
fn parse_output(response: &str) -> serde_json::Value {
    let trimmed = response.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(str::trim)
        .and_then(|s| s.strip_suffix("```").map(str::trim))
        .unwrap_or(trimmed);
    match serde_json::from_str::<serde_json::Value>(stripped) {
        Ok(v) => v,
        Err(_) => serde_json::Value::String(trimmed.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::binding::Entity;
    use crate::test_stubs::StubChatBackend;

    fn make_ctx(user_text: &str, entities: &[Entity]) -> WorkContext {
        let request_json = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": user_text}]
        });
        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(request_json.to_string()),
        );
        if !entities.is_empty() {
            ctx.metadata.insert(
                super::super::binding::ENTITIES_META_KEY.into(),
                MetadataValue::String(serde_json::to_string(entities).unwrap()),
            );
        }
        ctx
    }

    fn stage_with(
        backend: StubChatBackend,
        template: &str,
        upstream_ids: Vec<String>,
    ) -> ChartPromptStage {
        ChartPromptStage::new(
            Arc::new(backend),
            Arc::new(Limiter::new(4)),
            "reproduce",
            "bug_triage",
            template,
            vec![],
            upstream_ids,
            vec![],
            vec![ArcIntern::from("repro_plan"), ArcIntern::from("reproduce")],
        )
    }

    #[test]
    fn execute_emits_chart_provenance_metadata() {
        let backend = StubChatBackend::always(r#"{"plan": "minimal repro steps"}"#);
        let stage = stage_with(
            backend,
            "Given the bug report {{ request }}, write a reproduction plan.",
            vec![],
        );
        let ctx = make_ctx("crash on startup", &[]);
        let output = stage.execute(&ctx).unwrap();
        let decision: StageDecision = output.data_take().unwrap();
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(
            decision.metadata[CHART_NAME_META_KEY],
            serde_json::json!("bug_triage")
        );
        assert_eq!(
            decision.metadata[CHART_TARGET_META_KEY],
            serde_json::json!("reproduce")
        );
        assert_eq!(
            decision.metadata[CHART_OUTPUT_META_KEY]["plan"],
            serde_json::json!("minimal repro steps")
        );
        assert!(decision.metadata[CHART_RESPONSE_META_KEY].is_string());
    }

    #[test]
    fn execute_renders_entities_and_upstream() {
        // First stage output promoted under `stage.reproduce.output`.
        let mut ctx = make_ctx(
            "crash on startup",
            &[Entity {
                id: "issue-42".into(),
                kind: "report".into(),
                value: serde_json::json!({"title": "Segfault on load"}),
            }],
        );
        ctx.metadata.insert(
            "stage.reproduce.output".into(),
            MetadataValue::String(serde_json::json!({"plan": "minimal repro"}).to_string()),
        );

        let backend = StubChatBackend::always(r#"{"cause": "null pointer deref"}"#);
        let stage = ChartPromptStage::new(
            Arc::new(backend),
            Arc::new(Limiter::new(4)),
            "root_cause",
            "bug_triage",
            "Using the plan {{ upstream.reproduce.output.plan }}, find the cause of {{ request }}.\n{% for e in deps.report %}Report: {{ e.value.title }}\n{% endfor %}",
            vec![DepSpec::EntityMatch {
                name: "report".into(),
                description: "the bug report".into(),
                predicate: Some(super::super::EntityPredicate {
                    fields: vec![super::super::FieldRule {
                        path: "title".into(),
                        ty: super::super::FieldType::String,
                        required: true,
                        min: None,
                        max: None,
                        pattern: None,
                    }],
                    any_of: vec![],
                }),
                required: true,
            }],
            vec!["reproduce".to_string()],
            vec![ArcIntern::from("repro_plan")],
            vec![ArcIntern::from("root_cause")],
        );

        let output = stage.execute(&ctx).unwrap();
        let decision: StageDecision = output.data_take().unwrap();
        assert_eq!(
            decision.metadata[CHART_OUTPUT_META_KEY]["cause"],
            serde_json::json!("null pointer deref")
        );
    }

    #[test]
    fn unmatched_required_dep_fails_closed() {
        // No entities provided; the report dep is required → error.
        let backend = StubChatBackend::always(r#"{"x": 1}"#);
        let stage = ChartPromptStage::new(
            Arc::new(backend),
            Arc::new(Limiter::new(4)),
            "fix_plan",
            "bug_triage",
            "fix {{ request }}",
            vec![DepSpec::EntityMatch {
                name: "report".into(),
                description: "the bug report".into(),
                predicate: Some(super::super::EntityPredicate {
                    fields: vec![super::super::FieldRule {
                        path: "title".into(),
                        ty: super::super::FieldType::String,
                        required: true,
                        min: None,
                        max: None,
                        pattern: None,
                    }],
                    any_of: vec![],
                }),
                required: true,
            }],
            vec![],
            vec![],
            vec![ArcIntern::from("fix_plan")],
        );
        let ctx = make_ctx("help", &[]);
        let err = stage.execute(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("unmatched required deps"),
            "expected unmatched-deps error, got: {err}"
        );
    }

    #[test]
    fn parse_output_strips_fences_and_falls_back() {
        assert_eq!(
            parse_output("```json\n{\"a\": 1}\n```"),
            serde_json::json!({"a": 1})
        );
        assert_eq!(
            parse_output("plain text answer"),
            serde_json::json!("plain text answer")
        );
        assert_eq!(parse_output("42"), serde_json::json!(42));
    }
}
