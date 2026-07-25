//! Workflow DAG configuration types — deserialized from JSON.
//!
//! These types represent the declarative workflow definition (the
//! "graph config") that gets compiled into `PipelineGraph` instances
//! at boot time.  Each stage variant maps to one or more `Component`
//! implementations.
//!
//! # Example JSON
//!
//! ```jsonc
//! {
//!   "workflows": {
//!     "default": {
//!       "system_prompt": "env/prompts/classifier.md",
//!       "stages": [
//!         { "id": "prefilter", "type": "deterministic", "patterns": "env/pii-patterns.json" },
//!         { "id": "classifier", "type": "classify", "model": "classifier",
//!           "retry": { "max_attempts": 3, "prompts": ["env/prompts/retry1.md"] } },
//!         { "id": "router", "type": "switch", "field": "intent",
//!           "branches": {
//!             "code": { "type": "pipeline_ref", "name": "code_router" },
//!             "question": { "type": "dispatch", "model": "fast" },
//!             "*": { "type": "dispatch", "model": "fast" }
//!           }
//!         }
//!       ]
//!     },
//!     "code_router": {
//!       "stages": [
//!         { "id": "code_classifier", "type": "classify", "model": "classifier",
//!           "system_prompt": "env/prompts/code_complexity.md" },
//!         { "id": "code_dispatch", "type": "dispatch", "model": "code" }
//!       ]
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Top-level config ─────────────────────────────────────────────────────

/// Top-level workflow configuration loaded from a JSON file.
/// Lives alongside `RouterConfig` and can be merged in or referenced
/// via a path field.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowConfig {
    /// Named workflow definitions.  The `"default"` workflow is used when no
    /// explicit workflow name is requested.
    #[serde(default)]
    pub workflows: HashMap<String, WorkflowDef>,
}

// ── Workflow definition ──────────────────────────────────────────────────

/// A named pipeline graph definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDef {
    /// Default system prompt for classifier stages that don't set their own.
    /// Can be an inline string or a path to a file (e.g. `"env/prompts/classifier.md"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Ordered stages.  Execution order is derived from each stage's
    /// `depends_on` declarations (topological sort), not from array position.
    #[serde(default)]
    pub stages: Vec<WorkflowStage>,
}

// ── Stage variants ───────────────────────────────────────────────────────

/// A single stage in a workflow graph.
///
/// Discriminated by the `"type"` field.  The `id` is used for dependency
/// declarations (`depends_on`) and for metadata promotion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkflowStage {
    /// Deterministic regex-based pre-filter.  Maps to `DeterministicPreFilter`.
    #[serde(rename = "deterministic")]
    DeterministicFilter {
        id: String,
        /// Optional path to a `RejectPatterns` JSON file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        patterns: Option<String>,
        /// Stage IDs this stage depends on (must execute after these).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },

    /// LLM classifier stage.  Maps to `ClassifierStage`, optionally wrapped
    /// in `RetryClassifier` when `retry` is set.
    #[serde(rename = "classify")]
    Classify {
        id: String,
        /// Override the model used for classification.  When absent, uses the
        /// root-level `classifier_model` or the first model in the `"fast"` group.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Path or inline system prompt override for this classifier instance.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        system_prompt: Option<String>,
        /// Retry configuration for JSON parse failures.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry: Option<RetryConfig>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },

    /// Conditional branching stage.  Maps to `SwitchStage` with inline
    /// sub-pipeline definitions or `PipelineRefStage` references.
    #[serde(rename = "switch")]
    Switch {
        id: String,
        /// Metadata key to switch on (e.g. `"intent"`, `"complexity"`).
        field: String,
        /// Branch definitions.  The key is the metadata value to match
        /// against (string comparison).  The special key `"*"` is a wildcard.
        /// Each value is a list of stages to execute for that branch.
        branches: HashMap<String, Vec<WorkflowStage>>,
        /// Default branch when no exact or wildcard match is found.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Vec<WorkflowStage>>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },

    /// Reference to another named workflow.  Maps to `PipelineRefStage`.
    #[serde(rename = "pipeline_ref")]
    PipelineRef {
        id: String,
        /// Name of the target workflow (must exist in `workflows`).
        name: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },

    /// Direct dispatch to a named model (no further stages).  Maps to
    /// a `RoutingTarget` embedded in a `StageDecision` with `Rerouted` verdict.
    #[serde(rename = "dispatch")]
    Dispatch {
        id: String,
        /// Model name to dispatch to.
        model: String,
        /// Optional override of the dispatch endpoint URL.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        endpoint: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },

    /// Range-based numeric switch.  Maps to `SwitchStage` with numeric
    /// thresholds instead of exact string matches.
    #[serde(rename = "range_switch")]
    RangeSwitch {
        id: String,
        /// Metadata key to evaluate (expected to contain a numeric value).
        field: String,
        /// Ordered range thresholds.  Evaluated from first to last; the
        /// first matching range wins.  Each range has an optional `min`/`max`
        /// bound and a list of stages to execute.
        ranges: Vec<RangeBranch>,
        /// Default branch when no range matches.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Vec<WorkflowStage>>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        depends_on: Vec<String>,
    },
}

// ── Supporting types ─────────────────────────────────────────────────────

/// Retry strategy for classifier parse failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (total calls = 1 + max_attempts).
    pub max_attempts: usize,
    /// Escalating system prompt overrides.  `prompts[0]` is injected on the
    /// first retry, `prompts[1]` on the second, and so on.  When there are
    /// more retries than prompts, the last prompt is reused.
    ///
    /// Each entry can be an inline string or a path (e.g. `"env/prompts/retry1.md"`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<String>,
}

/// A single range branch for `RangeSwitch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeBranch {
    /// Minimum value (inclusive).  When absent, the range starts at -infinity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum value (exclusive).  When absent, the range extends to +infinity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Stages to execute when the field falls within this range.
    pub stages: Vec<WorkflowStage>,
}

// ── Helpers ──────────────────────────────────────────────────────────────

impl WorkflowConfig {
    /// Load a `WorkflowConfig` from a JSON file, returning `Self::default()`
    /// on any error (file-not-found, parse error, etc.).
    pub fn load_or_default(path: &std::path::Path) -> Self {
        common_core::config::load_json_or_default::<Self>(path)
    }

    /// Returns the `WorkflowDef` for the given name, or `None`.
    pub fn get(&self, name: &str) -> Option<&WorkflowDef> {
        self.workflows.get(name)
    }

    /// Returns the default workflow (`"default"` key), or `None`.
    pub fn default_workflow(&self) -> Option<&WorkflowDef> {
        self.workflows.get("default")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_workflow_config() {
        let json = r#"{
            "workflows": {
                "default": {
                    "stages": [
                        { "id": "prefilter", "type": "deterministic" },
                        { "id": "classifier", "type": "classify" }
                    ]
                }
            }
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(json).unwrap();
        let def = cfg.default_workflow().unwrap();
        assert_eq!(def.stages.len(), 2);
    }

    #[test]
    fn parse_switch_with_branches() {
        let json = r#"{
            "workflows": {
                "default": {
                    "stages": [
                        { "id": "classifier", "type": "classify" },
                        { "id": "router", "type": "switch", "field": "intent",
                          "branches": {
                            "code": [
                              { "id": "code_dispatch", "type": "dispatch", "model": "code" }
                            ],
                            "*": [
                              { "id": "default_dispatch", "type": "dispatch", "model": "fast" }
                            ]
                          }
                        }
                    ]
                }
            }
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(json).unwrap();
        let def = cfg.default_workflow().unwrap();
        assert_eq!(def.stages.len(), 2);
        // Verify the switch stage
        match &def.stages[1] {
            WorkflowStage::Switch { field, branches, .. } => {
                assert_eq!(field, "intent");
                assert_eq!(branches.len(), 2);
                assert!(branches.contains_key("code"));
                assert!(branches.contains_key("*"));
            }
            _ => panic!("expected Switch stage"),
        }
    }

    #[test]
    fn parse_pipeline_ref() {
        let json = r#"{
            "workflows": {
                "default": {
                    "stages": [
                        { "id": "delegate", "type": "pipeline_ref", "name": "code_router" }
                    ]
                },
                "code_router": {
                    "stages": [
                        { "id": "dispatch", "type": "dispatch", "model": "code" }
                    ]
                }
            }
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.workflows.len(), 2);
        assert!(cfg.workflows.contains_key("code_router"));
    }

    #[test]
    fn parse_classify_with_retry() {
        let json = r#"{
            "workflows": {
                "default": {
                    "stages": [
                        { "id": "classifier", "type": "classify",
                          "retry": { "max_attempts": 3, "prompts": ["p1", "p2"] } }
                    ]
                }
            }
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(json).unwrap();
        let def = cfg.default_workflow().unwrap();
        match &def.stages[0] {
            WorkflowStage::Classify { retry, .. } => {
                let r = retry.as_ref().unwrap();
                assert_eq!(r.max_attempts, 3);
                assert_eq!(r.prompts.len(), 2);
            }
            _ => panic!("expected Classify stage"),
        }
    }

    #[test]
    fn parse_range_switch() {
        let json = r#"{
            "workflows": {
                "default": {
                    "stages": [
                        { "id": "rs", "type": "range_switch", "field": "complexity",
                          "ranges": [
                            { "max": 3, "stages": [
                                { "id": "d1", "type": "dispatch", "model": "code" }
                            ]},
                            { "min": 3, "max": 6, "stages": [
                                { "id": "d2", "type": "dispatch", "model": "code" }
                            ]}
                          ],
                          "default": [
                            { "id": "df", "type": "dispatch", "model": "frontier" }
                          ]
                        }
                    ]
                }
            }
        }"#;
        let cfg: WorkflowConfig = serde_json::from_str(json).unwrap();
        let def = cfg.default_workflow().unwrap();
        match &def.stages[0] {
            WorkflowStage::RangeSwitch { field, ranges, default, .. } => {
                assert_eq!(field, "complexity");
                assert_eq!(ranges.len(), 2);
                assert!(default.is_some());
            }
            _ => panic!("expected RangeSwitch stage"),
        }
    }
}
