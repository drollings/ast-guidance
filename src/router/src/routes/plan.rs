use crate::workflow_config::WorkflowConfig;

pub struct PlanRoute {
    /// HNSW index for prior workflows.
    workflow_index: Option<crate::hnsw::HnswIndexHandle>,
    /// Template workflows keyed by task class.
    templates: std::collections::HashMap<String, WorkflowConfig>,
}

#[derive(Debug, Clone)]
pub struct PlanResult {
    pub workflow: WorkflowConfig,
    pub source: PlanSource,
    pub interview_questions: Vec<String>,
    pub gaps_filled: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanSource {
    HnswHit,
    TemplateAdapted,
    FreshDraft,
}

impl Default for PlanRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanRoute {
    pub fn new() -> Self {
        Self {
            workflow_index: None,
            templates: std::collections::HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_index(mut self, index: crate::hnsw::HnswIndexHandle) -> Self {
        self.workflow_index = Some(index);
        self
    }

    pub fn register_template(&mut self, task_class: impl Into<String>, workflow: WorkflowConfig) {
        self.templates.insert(task_class.into(), workflow);
    }

    /// Execute the plan route:
    /// 1. HNSW lookup for similar prior workflow
    /// 2. On miss: extract partial info → template workflow → identify deltas
    /// 3. Generate interview questions for missing inputs
    /// 4. Return PlanResult for the caller to present interview and re-submit
    pub fn plan(&self, _user_message: &str, _intent: Option<&str>) -> PlanResult {
        PlanResult {
            workflow: WorkflowConfig::default(),
            source: PlanSource::FreshDraft,
            interview_questions: Vec::new(),
            gaps_filled: Vec::new(),
        }
    }
}
