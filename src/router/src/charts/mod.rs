//! Chart content model — a library of *cached, templated dependency graphs*.
//!
//! A **chart** is a stored, reusable DAG of targets. Each target carries a
//! Jinja2-style prompt template plus structured dependencies. A cheap
//! classifier *selects* the right chart for a new request, *binds* context
//! entities to the chart's abstract dependencies (duck-typing), *renders*
//! each target's template into a prompt, and executes the DAG in topological
//! order — so a low-capability model can reliably repeat work that was
//! originally solved by a high-capability one.
//!
//! This module defines the declarative chart schema (`ChartDef`/`ChartTarget`/
//! `DepSpec`/`EntityPredicate`) and its validation. No I/O, no rendering,
//! no LLM.

use std::collections::HashSet;

use fluent_dag::dep_graph::DependencyGraph;
use serde::{Deserialize, Serialize};

pub mod binding;
pub mod compile;
pub mod execute;
pub mod extract;
pub mod render;
pub mod rubric;
pub mod select;
pub mod stage;
pub mod store;

use common_core::constants::default_true;

// ── Module-local constants ───────────────────────────────────────────────

/// Maximum length of a chart description (chars).
pub const CHART_DESCRIPTION_MAX_CHARS: usize = 240;
/// Current chart schema version. Charts with a different version are rejected.
pub const CHART_SCHEMA_VERSION: u32 = 1;
/// Maximum length of a single target template (chars).
pub const CHART_TEMPLATE_MAX_CHARS: usize = 16_384;
/// Maximum number of targeted interview questions rendered for a `Partial`
/// fit. Fixed and small — the interview is one round, never open-ended.
pub const CHART_MAX_INTERVIEW_QUESTIONS: usize = 3;
/// Default LLM-judge acceptance threshold for a chart rubric gate.
pub const DEFAULT_RUBRIC_MIN_SCORE: f64 = 0.7;
/// Consecutive rubric-gate failures after which a chart is demoted (no longer
/// selected) and flagged in the audit log (staleness policy).
pub const CHART_STALE_FAILS: usize = 3;
/// Maximum length of an auto-extracted chart name (derived from a query).
pub const CHART_EXTRACTED_NAME_MAX_CHARS: usize = 64;

// ── Errors ───────────────────────────────────────────────────────────────

/// Errors produced by chart parsing, validation, binding, rendering, and
/// selection. All chart error surfaces funnel through this enum.
#[derive(Debug, thiserror::Error)]
pub enum ChartError {
    #[error("chart parse error in {path}: {source}")]
    Parse {
        #[source]
        source: serde_json::Error,
        path: String,
    },
    #[error("unsupported chart schema version: {0}")]
    UnsupportedVersion(u32),
    #[error("invalid chart: {reason}")]
    Invalid { reason: String },
    #[error("duplicate chart entity name: {0}")]
    DuplicateName(String),
    #[error("unresolved chart dependency: {0}")]
    UnresolvedDependency(String),
    #[error("template render failed for target {target}: {detail}")]
    Render { target: String, detail: String },
    #[error("entity binding failed for target {target}, dep {dep}: {detail}")]
    Binding {
        target: String,
        dep: String,
        detail: String,
    },
    #[error("chart compile failed: {reason}")]
    Compile { reason: String },
    #[error("chart selection failed: {reason}")]
    Selection { reason: String },
    #[error("chart index error: {reason}")]
    Index { reason: String },
    #[error("chart I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A non-fatal validation notice. Chart `validate()` returns these alongside
/// a valid chart — callers decide whether to log, warn, or promote to an
/// error. Today the only warning kind is a capability dependency that no
/// chart target provides (satisfiable only by bound entities at runtime).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChartValidationWarning {
    /// The capability asset that only a bound entity can satisfy.
    pub dep: String,
    /// Human-readable explanation.
    pub message: String,
}

// ── Chart content types ──────────────────────────────────────────────────

/// A stored, reusable DAG chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDef {
    /// Unique key; also the provides asset of the chart.
    pub name: String,
    /// Human-readable description, `<= CHART_DESCRIPTION_MAX_CHARS`.
    pub description: String,
    /// Gate: charts with unsupported versions are rejected at load.
    pub schema_version: u32,
    /// "human" or a model key (staleness tracking).
    pub author_model: String,
    /// DAG nodes.
    pub targets: Vec<ChartTarget>,
    /// Chart-level acceptance rubric. Gates the final chart output (the last
    /// completed target's output) before a run is reported successful.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<ChartRubric>,
}

impl ChartDef {
    /// Validate the chart against the content-model invariants.
    ///
    /// Returns a list of non-fatal warnings (e.g. capability dependencies
    /// satisfiable only by bound entities at runtime) on success. Fatal
    /// problems return `Err(ChartError)`.
    pub fn validate(&self) -> Result<Vec<ChartValidationWarning>, ChartError> {
        // name
        if self.name.is_empty() {
            return Err(ChartError::Invalid {
                reason: "chart name must be non-empty".into(),
            });
        }
        if !self
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(ChartError::Invalid {
                reason: format!(
                    "chart name '{name}' must contain only alphanumerics, '_' or '-'",
                    name = self.name
                ),
            });
        }

        // description
        if self.description.chars().count() > CHART_DESCRIPTION_MAX_CHARS {
            return Err(ChartError::Invalid {
                reason: format!("chart description exceeds {CHART_DESCRIPTION_MAX_CHARS} chars"),
            });
        }

        // schema version
        if self.schema_version != CHART_SCHEMA_VERSION {
            return Err(ChartError::UnsupportedVersion(self.schema_version));
        }

        // targets
        if self.targets.is_empty() {
            return Err(ChartError::Invalid {
                reason: "chart has no targets".into(),
            });
        }

        let mut target_names: HashSet<&str> = HashSet::new();
        let mut provides_seen: HashSet<&str> = HashSet::new();
        let mut any_provides = false;

        if let Some(rubric) = &self.rubric {
            validate_rubric(rubric, "chart")?;
        }

        for target in &self.targets {
            if target.name.is_empty() {
                return Err(ChartError::Invalid {
                    reason: "target name must be non-empty".into(),
                });
            }
            if !target_names.insert(target.name.as_str()) {
                return Err(ChartError::DuplicateName(target.name.clone()));
            }
            if target.template.is_empty() {
                return Err(ChartError::Invalid {
                    reason: format!("target '{}' has an empty template", target.name),
                });
            }
            if target.template.chars().count() > CHART_TEMPLATE_MAX_CHARS {
                return Err(ChartError::Invalid {
                    reason: format!(
                        "target '{}' template exceeds {} chars",
                        target.name, CHART_TEMPLATE_MAX_CHARS
                    ),
                });
            }
            if let Some(rubric) = &target.rubric {
                validate_rubric(rubric, &format!("target '{}'", target.name))?;
            }
            if !target.provides.is_empty() {
                any_provides = true;
            }
            for provides in &target.provides {
                if provides.is_empty() {
                    return Err(ChartError::Invalid {
                        reason: format!("target '{}' has an empty provides entry", target.name),
                    });
                }
                if !provides_seen.insert(provides.as_str()) {
                    return Err(ChartError::DuplicateName(provides.clone()));
                }
            }
        }

        if !any_provides {
            return Err(ChartError::Invalid {
                reason: "chart has no target that provides an asset; it can never be selected"
                    .into(),
            });
        }

        // Dependency-graph check: every Capability dep must be satisfiable
        // either by another target's provides (in-graph) or by a bound
        // entity at runtime (out-of-graph, allowed → warning).
        let warnings = self.unresolved_capability_deps();

        Ok(warnings)
    }

    /// Build a `DependencyGraph<String>` from the chart's capability deps
    /// and return the assets that are depended on but provided by no chart
    /// target. These are the deps that binding must satisfy at runtime.
    fn unresolved_capability_deps(&self) -> Vec<ChartValidationWarning> {
        let mut graph: DependencyGraph<String> = DependencyGraph::new();
        for target in &self.targets {
            let deps: Vec<String> = target
                .depends
                .iter()
                .filter_map(|d| match d {
                    DepSpec::Capability { name } => Some(name.clone()),
                    DepSpec::EntityMatch { .. } => None,
                })
                .collect();
            // Mirror the DependencySession convention: every target provides
            // its own name as an asset, so capability deps can reference a
            // target by name as well as by its explicit provides list.
            let mut provides = target.provides.clone();
            provides.push(target.name.clone());
            let _ = graph.register(&target.name, &deps, &provides);
        }

        graph
            .unresolved_deps()
            .into_iter()
            .map(|dep| ChartValidationWarning {
                dep,
                message: "capability dependency not provided by any chart target; \
                     expected to be satisfied by a bound entity"
                    .to_string(),
            })
            .collect()
    }
}

/// Validate an acceptance rubric: field paths non-empty, min_score in `[0,1]`.
fn validate_rubric(rubric: &ChartRubric, owner: &str) -> Result<(), ChartError> {
    for path in &rubric.require_fields {
        if path.trim().is_empty() {
            return Err(ChartError::Invalid {
                reason: format!("{owner} rubric has an empty require_fields path"),
            });
        }
    }
    if !(0.0..=1.0).contains(&rubric.min_score) {
        return Err(ChartError::Invalid {
            reason: format!(
                "{owner} rubric min_score {} outside [0,1]",
                rubric.min_score
            ),
        });
    }
    Ok(())
}

/// A single node in the chart DAG. Executing it renders `template` and makes
/// one LLM call whose structured output provides `provides`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartTarget {
    /// Unique target name within the chart; also the target's stage id and
    /// self-provided asset (the DependencySession convention).
    pub name: String,
    /// Concrete asset names (exact, dep_graph model).
    #[serde(default)]
    pub provides: Vec<String>,
    /// Structured dependencies.
    #[serde(default)]
    pub depends: Vec<DepSpec>,
    /// minijinja source.
    pub template: String,
    #[serde(default)]
    pub essential: bool,
    /// Acceptance rubric gating this target's output before it is promoted to
    /// `provides`. Absent → output accepted on successful execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<ChartRubric>,
}

/// An acceptance rubric for a chart target's (or chart's) output.
///
/// Before a target's output is promoted to `provides`, the rubric gate
/// runs: a cheap deterministic field-presence rule first, and an optional
/// LLM judge only when `judge_model` is set.  An absent rubric accepts any
/// successfully-executed output.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChartRubric {
    /// Dotted field paths that must be present and non-null in the output.
    #[serde(default)]
    pub require_fields: Vec<String>,
    /// When set, an LLM judge evaluates the output before promotion. The value
    /// is a model key / label identifying the judge backend (the caller maps
    /// it to an injected `ChatBackend`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<String>,
    /// Judge acceptance threshold in `[0, 1]`. Consulted only when
    /// `judge_model` is set and a judge backend is available.
    #[serde(default = "default_rubric_min_score")]
    pub min_score: f64,
}

fn default_rubric_min_score() -> f64 {
    DEFAULT_RUBRIC_MIN_SCORE
}

/// A structured dependency of a chart target.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DepSpec {
    /// Exact asset name — satisfied by a bound entity or another target's
    /// provides.
    Capability { name: String },
    /// Abstract dependency — satisfied by a context entity matching
    /// `predicate`.
    EntityMatch {
        name: String,
        /// Human/LLM label.
        description: String,
        /// Deterministic rule; LLM fallback if `None`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        predicate: Option<EntityPredicate>,
        #[serde(default = "default_true")]
        required: bool,
    },
}

impl DepSpec {
    /// The dependency's name (capability asset or entity-match label).
    pub fn name(&self) -> &str {
        match self {
            DepSpec::Capability { name } | DepSpec::EntityMatch { name, .. } => name,
        }
    }
}

/// A deterministic JSON-Schema subset. Field semantics mirror `FieldSchema`:
/// substring `pattern`, numeric `min`/`max`, typed paths into a
/// `serde_json::Value`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityPredicate {
    #[serde(default)]
    pub fields: Vec<FieldRule>,
    /// Nested alternatives (OR).
    #[serde(default)]
    pub any_of: Vec<EntityPredicate>,
}

/// A single typed rule on a dotted path into an entity's `value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldRule {
    /// Dotted path, e.g. "user.id"; "." = root.
    pub path: String,
    #[serde(default)]
    pub ty: FieldType,
    /// Path must exist.
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// SUBSTRING match (repo convention, not regex).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

/// The JSON value type a `FieldRule` constrains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Number,
    Bool,
    Array,
    #[default]
    Any,
}

/// Convenience for tests and seed data: build a chart JSON string from a
/// `ChartDef`-shaped value is a plain `serde_json::to_string_pretty` call —
/// kept here to document the round-trip contract rather than hide it.
#[doc(hidden)]
pub fn chart_to_json(chart: &ChartDef) -> Result<String, ChartError> {
    serde_json::to_string_pretty(chart).map_err(|e| ChartError::Parse {
        source: e,
        path: "<serialize>".into(),
    })
}

/// Consolidated parse helper: deserialize **and** validate a `ChartDef`
/// from JSON. Parsing a chart always enforces the content model. The
/// canonical home is `store::chart_from_str`; this re-export exposes it at
/// the `charts` module root so in-module consumers don't reach into
/// `store` for a pure content-model operation.
pub use store::chart_from_str;
#[cfg(test)]
#[path = "../../tests/charts_mod.rs"]
mod tests;
