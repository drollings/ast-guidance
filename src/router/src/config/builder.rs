//! Pipeline builder — constructs pipeline stages from `RouterConfig`.
//! Separated from `config.rs` to keep the configuration types focused
//! on data definition rather than orchestration.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_wvr::prelude::Component;
use guidance_llm::client::ChatBackend;
use guidance_llm::{LlmClient, LlmConfig};

use super::{default_true, RejectPatterns, RouterConfig};
use crate::pipeline::PipelineOrchestrator;
use crate::score_matrix::ScoreMatrix;

/// Named pipeline parameters. Pipelines are stored as a map keyed by name.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct PipelineParams {
    #[serde(default = "default_true")]
    pub deterministic_prefilter: bool,
    #[serde(default = "default_true")]
    pub classifier: bool,
    #[serde(default = "default_true")]
    pub router: bool,
    #[serde(default = "default_coherence_threshold")]
    pub coherence_threshold: f64,
    #[serde(default)]
    pub classifier_model: Option<String>,
    #[serde(default)]
    pub blacklist: Option<String>,
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            classifier: true,
            router: true,
            coherence_threshold: default_coherence_threshold(),
            classifier_model: None,
            blacklist: None,
            score_matrix: None,
        }
    }
}

fn default_coherence_threshold() -> f64 {
    0.70
}

impl RouterConfig {
    pub fn load_reject_patterns(path: &str) -> RejectPatterns {
        load_json_or_default::<RejectPatterns>(Path::new(path))
    }

    pub fn routing_config(&self) -> super::RoutingConfig {
        super::RoutingConfig {
            routes: self.routes.clone(),
            models: self.models.clone(),
            model_groups: self.model_groups.clone(),
            system_prompt: self.system_prompt.clone(),
            safety_threshold: self.safety_threshold,
            default_route: self.default_route.clone(),
            score_matrix: self.score_matrix.clone(),
        }
    }

    pub fn build_named_pipeline(&self, name: &str) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend(name, None)
    }

    pub fn build_named_pipeline_with_backend(
        &self,
        name: &str,
        classifier_backend: Option<Arc<dyn ChatBackend>>,
    ) -> Option<PipelineOrchestrator> {
        let params = self.pipelines.get(name)?;
        let mut stages: Vec<Arc<dyn Component>> = Vec::new();

        if params.deterministic_prefilter {
            if let Some(ref blacklist_path) = params.blacklist {
                let reject_patterns = Self::load_reject_patterns(blacklist_path);
                stages.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::from_config(
                        &reject_patterns,
                    ),
                ));
            } else {
                stages.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::new(),
                ));
            }
        }

        if params.classifier {
            let routing_config = self.routing_config();
            let client: Arc<dyn ChatBackend> = if let Some(backend) = classifier_backend {
                tracing::info!(target: "router.config", pipeline = %name, backend = "mock/transcript", "classifier using injected backend");
                backend
            } else {
                let client = build_classifier_client(self, name, params)?;
                tracing::info!(target: "router.config", pipeline = %name, "classifier using real LLM client");
                client
            };
            stages.push(Arc::new(
                crate::stages::classifier::ClassifierStage::new(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
                ),
            ));
        } else if classifier_backend.is_some() {
            tracing::warn!(
                target: "router.config",
                pipeline = %name,
                "classifier backend was provided but classifier is disabled for this pipeline"
            );
        }

        Some(PipelineOrchestrator::new(stages))
    }

    pub fn build_all_pipelines(&self) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend(None)
    }

    pub fn build_all_pipelines_with_backend(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        let mut map = HashMap::new();
        let pipeline_count = self.pipelines.len();
        let has_mock = classifier_backend.is_some();
        tracing::info!(target: "router.config", pipeline_count = pipeline_count, mock_backend = has_mock, classifier_model = ?self.classifier_model, default_route = %self.default_route, "building pipelines");
        for name in self.pipelines.keys() {
            let backend_for_pipeline = classifier_backend.cloned();
            if let Some(pipeline) =
                self.build_named_pipeline_with_backend(name, backend_for_pipeline)
            {
                map.insert(name.clone(), Arc::new(pipeline));
            }
        }
        tracing::info!(target: "router.config", built = map.len(), "pipelines built");
        map
    }

    pub fn route_pipeline_names(&self, model_name: &str) -> Vec<String> {
        self.routes
            .get(model_name)
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    }
}

/// Build a classifier LLM client from the model config.
///
/// # DIP note
/// This factory is the **only** place in the crate that constructs a concrete
/// `LlmClient`.  The rest of the pipeline receives `Arc<dyn ChatBackend>` and
/// is oblivious to the concrete implementation.  There is exactly one
/// `ChatBackend` implementation today (`LlmClient`); if a second appears,
/// the factory can inject it without touching pipeline code.
fn build_classifier_client(
    config: &RouterConfig,
    name: &str,
    params: &PipelineParams,
) -> Option<Arc<dyn ChatBackend>> {
    let classifier_key = params
        .classifier_model
        .as_ref()
        .or(config.classifier_model.as_ref())
        .or_else(|| {
            config.model_groups
                .get("fast")
                .and_then(|names| names.first())
        });
    let classifier_entry = classifier_key.and_then(|k| config.models.get(k));
    let (entry, model_key) = if let Some(e) = classifier_entry {
        let key = classifier_key.unwrap();
        (e, key.as_str())
    } else {
        tracing::error!(target: "router.config", pipeline = %name, pipeline_model = ?params.classifier_model, root_model = ?config.classifier_model, "no classifier model found in config");
        return None;
    };
    let model_name_for_llm = entry.name.as_deref().unwrap_or(model_key);
    let classifier_config = LlmConfig::new()
        .api_url(entry.endpoint.clone())
        .model(model_name_for_llm.to_string())
        .timeout_ms(entry.total_timeout_ms)
        .maybe_extra_body_params(entry.params.clone())
        .build();
    Some(Arc::new(LlmClient::with_config(classifier_config)))
}
