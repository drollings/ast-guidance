use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use fluent_wvr::prelude::*;
use fluent_wvr_testutil::StubComponent;
use serde::{Deserialize, Serialize};

use crate::pipeline::{PipelineOrchestrator, PipelineResult};
use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;
use crate::stages::deterministic::DeterministicPreFilter;
use crate::stages::router::{RouterStage, RoutingPolicy};
use crate::types::RouterRequest;

/// Fixture data for the mock router. Each map is keyed by the user message
/// text (the input string) and contains the pre-generated response that the
/// corresponding stage would produce.
///
/// All fields are optional; missing fixtures cause the corresponding stage
/// to produce a default "passed" result.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockFixtures {
    /// Quality gate responses: input text → serialized StageDecision JSON.
    #[serde(default)]
    pub quality_gate: HashMap<String, String>,

    /// Guardrail responses: input text → serialized StageDecision JSON.
    #[serde(default)]
    pub guardrail: HashMap<String, String>,

    /// Planning refinement responses: input text → serialized StageDecision JSON.
    #[serde(default)]
    pub planning: HashMap<String, String>,

    /// Agent responses: compound key "agent_id||input_text" → response string.
    #[serde(default)]
    pub agent_responses: HashMap<String, String>,

    /// Frontier responses: compound key "provider||input_text" → response string.
    #[serde(default)]
    pub frontier_responses: HashMap<String, String>,
}

impl MockFixtures {
    pub fn new() -> Self {
        Self {
            quality_gate: HashMap::new(),
            guardrail: HashMap::new(),
            planning: HashMap::new(),
            agent_responses: HashMap::new(),
            frontier_responses: HashMap::new(),
        }
    }

    /// Load fixtures from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Helper: create a "passed" StageDecision for a pipeline stage.
    fn passed_decision(stage: PipelineStage, reason: &str) -> StageDecision {
        StageDecision {
            stage,
            verdict: StageVerdict::Passed,
            score: Some(1.0),
            reason: reason.into(),
            latency_ms: 0,
            metadata: serde_json::json!({}),
        }
    }

}

impl Default for MockFixtures {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing a `PipelineOrchestrator` with mock stage
/// substitutions. The deterministic pre-filter and router stages are real;
/// LLM-dependent stages are replaced with `StubComponent` handlers that
/// read from the fixture map.
fn build_mock_pipeline(fixtures: &Arc<MockFixtures>) -> PipelineOrchestrator {
    let stages: Vec<Arc<dyn Component>> = vec![
        // Stage 1: real deterministic pre-filter (no LLM)
        Arc::new(DeterministicPreFilter::new()),
        // Stage 2: mock quality gate (StubComponent with fixture handler)
        Arc::new(
            StubComponent::new("pipeline.stage2.quality_gate")
                .with_dep("pipeline.stage1.output")
                .with_provides("pipeline.stage2.output")
                .with_handler({
                    let fixtures = Arc::clone(fixtures);
                    move |ctx| {
                        let input = extract_user_message(ctx)?;
                        let decision = fixtures
                            .quality_gate
                            .get(&input)
                            .and_then(|json| serde_json::from_str::<StageDecision>(json).ok())
                            .unwrap_or_else(|| {
                                MockFixtures::passed_decision(
                                    PipelineStage::QualityGate,
                                    "mock: quality gate passed",
                                )
                            });
                        WorkOutput::typed("classified", &decision)
                    }
                }),
        ),
        // Stage 3: mock planning refinement (StubComponent with fixture handler)
        Arc::new(
            StubComponent::new("pipeline.stage3.planning")
                .with_dep("pipeline.stage2.output")
                .with_provides("pipeline.stage3.output")
                .with_handler({
                    let fixtures = Arc::clone(fixtures);
                    move |ctx| {
                        let input = extract_user_message(ctx)?;
                        let decision = fixtures
                            .planning
                            .get(&input)
                            .and_then(|json| serde_json::from_str::<StageDecision>(json).ok())
                            .unwrap_or_else(|| {
                                MockFixtures::passed_decision(
                                    PipelineStage::PlanningRefinement,
                                    "mock: planning passed",
                                )
                            });
                        WorkOutput::typed("planned", &decision)
                    }
                }),
        ),
        // Stage 4: mock guardrail check (StubComponent with fixture handler)
        Arc::new(
            StubComponent::new("pipeline.stage4.guardrail")
                .with_dep("pipeline.stage3.output")
                .with_provides("pipeline.stage4.output")
                .with_handler({
                    let fixtures = Arc::clone(fixtures);
                    move |ctx| {
                        let input = extract_user_message(ctx)?;
                        let decision = fixtures
                            .guardrail
                            .get(&input)
                            .and_then(|json| serde_json::from_str::<StageDecision>(json).ok())
                            .unwrap_or_else(|| {
                                MockFixtures::passed_decision(
                                    PipelineStage::GuardrailCheck,
                                    "mock: guardrail passed",
                                )
                            });
                        WorkOutput::typed("checked", &decision)
                    }
                }),
        ),
        // Stage 5: real router stage (no LLM — policy-based decision)
        Arc::new(RouterStage::new(RoutingPolicy::LocalFirst)),
    ];

    PipelineOrchestrator::new(stages)
}

// ── MockRouter ──────────────────────────────────────────────────────────

/// Full mock mode: no LLM inference. The pipeline runs against fixture data.
/// Use for fast, deterministic testing of the full pipeline flow.
pub struct MockRouter {
    pipeline: PipelineOrchestrator,
    fixtures: Arc<MockFixtures>,
}

impl MockRouter {
    /// Create a new mock router with the given fixtures.
    pub fn new(fixtures: MockFixtures) -> Self {
        let fixtures = Arc::new(fixtures);
        let pipeline = build_mock_pipeline(&fixtures);
        Self { pipeline, fixtures }
    }

    /// Load fixtures from a JSON file and create a mock router.
    pub fn from_fixture_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let fixtures = MockFixtures::from_file(path)?;
        Ok(Self::new(fixtures))
    }

    /// Run the full pipeline against mock fixtures.
    ///
    /// Returns the pipeline result on success or a router error on pipeline failure.
    pub fn route(&self, request: &RouterRequest) -> Result<PipelineResult, WorkError> {
        let messages_json = serde_json::to_value(&request.messages)
            .map_err(|e| WorkError::Execution(format!("serialization error: {e}")))?;

        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(messages_json.to_string()),
        );

        let output = self.pipeline.execute(&ctx)?;
        output.data_take().map_err(|e| WorkError::Execution(e.to_string()))
    }

    /// Borrow the pipeline (for introspection in tests).
    pub fn pipeline(&self) -> &PipelineOrchestrator {
        &self.pipeline
    }

    /// Borrow the fixtures.
    pub fn fixtures(&self) -> &MockFixtures {
        &self.fixtures
    }
}

// ── RouterOnlyMock ──────────────────────────────────────────────────────

/// Router-only mock mode: quality gate + routing use real policy decisions
/// (not LLM calls), but all downstream agent/frontier responses are fixtures.
///
/// In this mode, the deterministic pre-filter and router stages are real.
/// The quality gate, planning, and guardrail stages are replaced with mock
/// stages that return fixture-provided results.
pub struct RouterOnlyMock {
    pipeline: PipelineOrchestrator,
    #[allow(dead_code)]
    fixtures: Arc<MockFixtures>,
}

impl RouterOnlyMock {
    pub fn new(fixtures: MockFixtures) -> Self {
        let fixtures = Arc::new(fixtures);
        let pipeline = build_mock_pipeline(&fixtures);
        Self { pipeline, fixtures }
    }

    pub fn from_fixture_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let fixtures = MockFixtures::from_file(path)?;
        Ok(Self::new(fixtures))
    }

    /// Run the pipeline. Same as MockRouter::route — identical dispatch
    /// path; the differentiation is in how the mock stages are constructed
    /// (always fixtures, never real LLM).
    pub fn route(&self, request: &RouterRequest) -> Result<PipelineResult, WorkError> {
        let messages_json = serde_json::to_value(&request.messages)
            .map_err(|e| WorkError::Execution(format!("serialization error: {e}")))?;

        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(messages_json.to_string()),
        );

        let output = self.pipeline.execute(&ctx)?;
        output.data_take().map_err(|e| WorkError::Execution(e.to_string()))
    }
}
