//! Classification-tree engine
//!
//! Evaluates a `ClassificationTree` recursively:
//!
//! - `filter` nodes short-circuit deterministically (`hard_reject` /
//!   `soft_redirect` / `output_filter`),
//! - `classifier` nodes auto-build their prompt from their children (key +
//!   description) and the three-axis JSON schema, call the injected backend,
//!   enforce coherence/safety thresholds, and pick a child,
//! - `terminal` nodes resolve a `RoutingTarget` through
//!   `RoutingConfig::resolve_route` (complexity-based model selection),
//! - `fallback` children are evaluated when a classifier picks no named child
//!   or its LLM call fails.
//!
//! Every visited node emits a `StageDecision` (the final one carries the
//! `routing_target` / rejection for the pipeline handoff) and a durable audit
//! record via `audit::emit` with `kind = "tree_node"`.
//!
//! The module is split: [`engine`] (the recursive walk + [`ClassificationEngine`]),
//! [`verdict`] (the three-axis verdict + `parse_tree_verdict`), and [`decisions`]
//! (the `TreeOutcome`/`TreeEvaluation` types and the `StageDecision` builders).
//! The classifier-node prompt builders live in `crate::config::classification`
//! (`ClassificationNode::build_prompt`).

pub mod decisions;
pub mod engine;
pub mod verdict;

pub use decisions::{final_decision, TreeEvaluation, TreeHandoff, TreeOutcome};
pub use engine::{cost, ClassificationEngine};
pub use verdict::{parse_tree_verdict, TreeClassifierVerdict};
#[cfg(test)]
#[path = "../../../tests/stages_tree_mod.rs"]
mod tests;
