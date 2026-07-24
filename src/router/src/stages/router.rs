//! Stage 5: RouterStage — selects destination (local agent or frontier) and
//! transform strategy. Emits a `RoutingDecision` into the stage output.

use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingPolicy {
    CostMinimizing,
    LocalFirst,
    FrontierOnly,
    AutoRouting { classifier_model: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub destination: RoutingDestination,
    pub transform: TransformStrategy,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingDestination {
    LocalAgent {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        adapter: Option<String>,
        session_id: String,
    },
    Frontier {
        provider: String,
        model: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformStrategy {
    None,
    PiiAnonymize,
    DecomposeToAnonymizedHypothetical,
    DecomposeToSubtasks,
}

pub struct RouterStage {
    name: ArcIntern<str>,
    routing_policy: RoutingPolicy,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl RouterStage {
    pub fn new(routing_policy: RoutingPolicy) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage5.router"),
            routing_policy,
            depends: vec![ArcIntern::from("pipeline.stage4.output")],
            provides: vec![ArcIntern::from("pipeline.stage5.output")],
        }
    }

    fn make_routing_decision(&self) -> RoutingDecision {
        match &self.routing_policy {
            RoutingPolicy::LocalFirst | RoutingPolicy::CostMinimizing => RoutingDecision {
                destination: RoutingDestination::LocalAgent {
                    model: "default-agent".into(),
                    adapter: None,
                    session_id: "default".into(),
                },
                transform: TransformStrategy::None,
                confidence: 0.8,
                reason: "routing to local agent (LocalFirst policy)".into(),
            },
            RoutingPolicy::FrontierOnly => RoutingDecision {
                destination: RoutingDestination::Frontier {
                    provider: "openai".into(),
                    model: "gpt-4o".into(),
                },
                transform: TransformStrategy::None,
                confidence: 1.0,
                reason: "routing to frontier (FrontierOnly policy)".into(),
            },
            RoutingPolicy::AutoRouting { .. } => RoutingDecision {
                destination: RoutingDestination::LocalAgent {
                    model: "default-agent".into(),
                    adapter: None,
                    session_id: "default".into(),
                },
                transform: TransformStrategy::None,
                confidence: 0.7,
                reason: "auto-routing to local agent (default)".into(),
            },
        }
    }
}

impl WorkUnit for RouterStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let routing_decision = self.make_routing_decision();

        tracing::info!(target: "router.pipeline.stage5",
            policy = ?self.routing_policy,
            destination = ?routing_decision.destination,
            transform = ?routing_decision.transform,
            confidence = %routing_decision.confidence,
            reason = %routing_decision.reason,
            "router decision"
        );

        WorkOutput::typed(
            "routed",
            &StageDecision {
                stage: PipelineStage::Router,
                verdict: StageVerdict::Passed,
                score: Some(routing_decision.confidence),
                reason: routing_decision.reason.clone(),
                latency_ms: 0,
                metadata: serde_json::json!({
                    "routing_decision": routing_decision,
                }),
            },
        )
    }
}

impl FieldAccess for RouterStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "RouterStage has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "RouterStage has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for RouterStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(RouterStage);