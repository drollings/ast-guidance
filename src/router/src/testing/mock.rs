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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockFixtures {
    #[serde(default)]
    pub quality_gate: HashMap<String, String>,

    #[serde(default)]
    pub guardrail: HashMap<String, String>,

    #[serde(default)]
    pub planning: HashMap<String, String>,

    #[serde(default)]
    pub classifier: HashMap<String, String>,

    #[serde(default)]
    pub agent_responses: HashMap<String, String>,

    #[serde(default)]
    pub frontier_responses: HashMap<String, String>,
}

impl MockFixtures {
    pub fn new() -> Self {
        Self {
            quality_gate: HashMap::new(),
            guardrail: HashMap::new(),
            planning: HashMap::new(),
            classifier: HashMap::new(),
            agent_responses: HashMap::new(),
            frontier_responses: HashMap::new(),
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Ok(serde_json::from_str(&content)?)
    }

    fn passed_classifier() -> StageDecision {
        StageDecision {
            stage: PipelineStage::Classifier,
            verdict: StageVerdict::Passed,
            score: Some(0.95),
            reason: "intent=question, action=route, coherence=0.95".into(),
            latency_ms: 0,
            metadata: serde_json::json!({
                "coherence_score": 0.95,
                "safety_score": 0.9,
                "intent": "question",
                "action": "route",
                "reason": "well-formed factual query",
                "routing_target": {
                    "url": "http://localhost:8080/v1",
                    "model": "fast",
                    "target_name": "fast"
                }
            }),
        }
    }
}

impl Default for MockFixtures {
    fn default() -> Self {
        Self::new()
    }
}

fn build_mock_pipeline(fixtures: &Arc<MockFixtures>) -> PipelineOrchestrator {
    let stages: Vec<Arc<dyn Component>> = vec![
        Arc::new(DeterministicPreFilter::new()),
        Arc::new(
            StubComponent::new("pipeline.stage2.classifier")
                .with_dep("pipeline.stage1.output")
                .with_provides("pipeline.stage2.output")
                .with_handler({
                    let fixtures = Arc::clone(fixtures);
                    move |ctx| {
                        let input = extract_user_message(ctx)?;
                        let decision = fixtures
                            .classifier
                            .get(&input)
                            .and_then(|json| serde_json::from_str::<StageDecision>(json).ok())
                            .unwrap_or_else(|| {
                                MockFixtures::passed_classifier()
                            });
                        WorkOutput::typed("classified", &decision)
                    }
                }),
        ),
        Arc::new(RouterStage::new(RoutingPolicy::LocalFirst)),
    ];

    PipelineOrchestrator::new(stages)
}

pub struct MockRouter {
    pipeline: PipelineOrchestrator,
    fixtures: Arc<MockFixtures>,
}

impl MockRouter {
    pub fn new(fixtures: MockFixtures) -> Self {
        let fixtures = Arc::new(fixtures);
        let pipeline = build_mock_pipeline(&fixtures);
        Self { pipeline, fixtures }
    }

    pub fn from_fixture_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let fixtures = MockFixtures::from_file(path)?;
        Ok(Self::new(fixtures))
    }

    pub fn route(&self, request: &RouterRequest) -> Result<PipelineResult, WorkError> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| WorkError::Execution(format!("serialization error: {e}")))?;

        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(request_json),
        );

        let output = self.pipeline.execute(&ctx)?;
        output.data_take().map_err(|e| WorkError::Execution(e.to_string()))
    }

    pub fn pipeline(&self) -> &PipelineOrchestrator {
        &self.pipeline
    }

    pub fn fixtures(&self) -> &MockFixtures {
        &self.fixtures
    }
}

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

    pub fn route(&self, request: &RouterRequest) -> Result<PipelineResult, WorkError> {
        let request_json = serde_json::to_string(request)
            .map_err(|e| WorkError::Execution(format!("serialization error: {e}")))?;

        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(request_json),
        );

        let output = self.pipeline.execute(&ctx)?;
        output.data_take().map_err(|e| WorkError::Execution(e.to_string()))
    }
}
