//! Classification-tree configuration
//!
//! `RouterConfig.classification = Some(tree)` switches the classifier stage
//! into tree-driven mode: instead of the flat single-LLM-call
//! prompt/score-matrix path, the stage evaluates a nested tree of
//! [`ClassificationNode`]s recursively. Classifier nodes auto-build their
//! prompt from their children's keys and descriptions (per `doc/router/VISION.md`
//! §"The Classification Tree"), so adding a route updates the prompt with no
//! manual maintenance.
//!
//! The `pipelines` stage table, `models`, and `model_groups` are unchanged
//! and still load; the flat *views* the rest of the server expects
//! (route→pipeline mapping, system prompt) are derived from the tree (M3c:
//! flat `routes` / `system_prompt` / root `score_matrix` are gone).

use std::fmt::Write;

use fluent_types::InterlinguaId;
use serde::{Deserialize, Serialize};

use crate::config::filters::FilterOutcome;

/// The interlingua match for a `Filter` node (ROADMAP §14.6, C6): a
/// deterministic dispatch on the request's parsed predicate/subject/object
/// ids. Same phrasing → same ids → same route, zero tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InterlinguaMatch {
    /// A predicate (root) lemma id; `None` = don't check.
    pub predicate_id: Option<InterlinguaId>,
    /// A subject lemma id; `None` = don't check.
    pub subject_id: Option<InterlinguaId>,
    /// A direct-object lemma id; `None` = don't check.
    pub object_id: Option<InterlinguaId>,
    /// Minimum sentence confidence for a match (default: any).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_min: Option<f64>,
}

/// The top-level classification tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationTree {
    /// Root node — typically a `classifier` node that branches on the request's
    /// domain/coherence/safety/complexity.
    pub root: ClassificationNode,
}

impl ClassificationTree {
    /// The model key of the root node when it is a `classifier` — the natural
    /// default for the classifier stage's backend when no flat
    /// `classifier_model` is configured.
    pub fn root_classifier_model(&self) -> Option<&str> {
        match &self.root {
            ClassificationNode::Classifier { model, .. } => Some(model),
            _ => None,
        }
    }

    /// Every `classifier` model key referenced anywhere in the tree, deduplicated
    /// (root first). The pipeline builder uses this to construct per-node
    /// backends so a sub-classifier on a different model dispatches to its own
    /// endpoint (real mode only).
    pub fn classifier_model_keys(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut keys = Vec::new();
        self.root.collect_classifier_models(&mut seen, &mut keys);
        keys
    }

    /// `(route, group, description)` for every `terminal` node in the tree —
    /// the source for the derived flat `routes` view.
    pub fn terminal_views(&self) -> Vec<(String, Option<String>, String)> {
        let mut out = Vec::new();
        self.root.collect_terminals(&mut out);
        out
    }

    /// Whether the terminal for `route` forces dispatch (`always_route`).
    /// `false` for unknown routes — the caller falls back to the default.
    pub fn terminal_always_route(&self, route: &str) -> bool {
        self.root.find_terminal(route).unwrap_or(false)
    }

    /// Auto-generate the classifier system prompt from the root node's
    /// children and descriptions — the derived `system_prompt` view for
    /// tree configs `None` when the root is not a classifier or has no
    /// routeable children.
    pub fn derive_system_prompt(&self) -> Option<String> {
        let (coherence, safety) = match &self.root {
            ClassificationNode::Classifier {
                coherence_threshold,
                safety_threshold,
                ..
            } => (
                coherence_threshold.unwrap_or(default_coherence_threshold()),
                safety_threshold.unwrap_or(default_safety_threshold()),
            ),
            _ => (default_coherence_threshold(), default_safety_threshold()),
        };
        self.root.build_prompt(coherence, safety)
    }
}

fn default_coherence_threshold() -> f64 {
    0.70
}

fn default_safety_threshold() -> f64 {
    0.5
}

/// A named branch of a classifier node. The LLM picks exactly one `key`; the
/// tree engine then evaluates that child's `node` recursively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationChild {
    pub key: String,
    #[serde(default)]
    pub description: String,
    pub node: ClassificationNode,
}

/// One node in the classification tree.
///
/// | `type` | Role | LLM call? |
/// |--------|------|-----------|
/// | `classifier` | LLM call that picks one child branch; prompt auto-built from children | Yes |
/// | `terminal` | Dispatch target; resolves a model via `RoutingConfig::resolve_route` | No |
/// | `filter` | Deterministic regex check that short-circuits (`hard_reject` / `soft_redirect` / `output_filter`) | No |
/// | `fallback` | Child evaluated when a classifier picks no named child or its LLM call fails | Only if the wrapped node is a classifier |
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum ClassificationNode {
    Classifier {
        description: String,
        /// Model key (from `models`) used for this classifier's LLM call.
        model: String,
        /// Per-node coherence threshold; defaults to the pipeline's.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        coherence_threshold: Option<f64>,
        /// Per-node safety threshold; defaults to `safety_threshold` in config.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        safety_threshold: Option<f64>,
        #[serde(default)]
        children: Vec<ClassificationChild>,
    },
    /// A dispatch target. `route` names the routed intent; `group` selects
    /// the dispatch model group for the derived flat view. `always_route`
    /// (default `false`) forces dispatch even when the classifier answers
    /// directly — the tree-carried copy of the former flat flag (M3c).
    Terminal {
        route: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group: Option<String>,
        #[serde(default)]
        description: String,
        /// Never let the classifier answer requests on this route directly.
        #[serde(default)]
        always_route: bool,
    },
    /// A deterministic regex check over the user message. Short-circuits the
    /// enclosing classifier when a pattern matches. When `match_interlingua`
    /// is set, it is evaluated **instead of** `patterns` — a Filter node is
    /// regex **or** interlingua, never both (§14.6, C6).
    Filter {
        #[serde(default)]
        description: String,
        #[serde(default)]
        patterns: Vec<String>,
        #[serde(default)]
        outcome: FilterOutcome,
        /// `soft_redirect` target: a sibling child key of the enclosing
        /// classifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redirect_to: Option<String>,
        /// Deterministic dispatch on the request's interlingua ids (C6).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        match_interlingua: Option<InterlinguaMatch>,
    },
    /// A child of a classifier node, evaluated when the LLM picks no named
    /// child or the LLM call itself fails.
    Fallback {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        node: Box<ClassificationNode>,
    },
}

impl ClassificationNode {
    /// Auto-construct the classifier system prompt from the node's description
    /// and its children (key + description), plus the three-axis output schema
    /// (`route`/`coherence`/`safety`/`complexity`/`reason`) and the node's
    /// threshold guidance. `None` when the node is not a classifier or has no
    /// routeable children.
    pub fn build_prompt(&self, coherence_threshold: f64, safety_threshold: f64) -> Option<String> {
        let ClassificationNode::Classifier {
            description,
            children,
            ..
        } = self
        else {
            return None;
        };

        let routeable: Vec<&ClassificationChild> = children
            .iter()
            .filter(|c| {
                matches!(
                    c.node,
                    ClassificationNode::Classifier { .. } | ClassificationNode::Terminal { .. }
                )
            })
            .collect();
        if routeable.is_empty() {
            return None;
        }

        let mut prompt = String::new();
        let _ = writeln!(prompt, "You are a {description}.");
        let _ = writeln!(prompt);
        let _ = writeln!(prompt, "Available routes:");
        for child in &routeable {
            if child.description.is_empty() {
                let _ = writeln!(prompt, "- {}", child.key);
            } else {
                let _ = writeln!(prompt, "- {}: {}", child.key, child.description);
            }
        }
        let _ = writeln!(prompt);
        let keys: Vec<&str> = routeable.iter().map(|c| c.key.as_str()).collect();
        let _ = writeln!(prompt, "You must output exactly one JSON object with:");
        let _ = writeln!(
            prompt,
            "  \"route\": \"<exactly one of: {}>\"",
            keys.join(", ")
        );
        let _ = writeln!(
            prompt,
            "  \"coherence\": 0.0-1.0 (how well-formed and coherent the query is)"
        );
        let _ = writeln!(
            prompt,
            "  \"safety\": 0.0-1.0 (1.0 = completely safe, 0.0 = policy violation)"
        );
        let _ = writeln!(
            prompt,
            "  \"complexity\": 0-10 (0 = trivial, 10 = requires most capable model)"
        );
        let _ = writeln!(
            prompt,
            "  \"reason\": \"brief explanation for the routing decision\""
        );
        let _ = writeln!(prompt);
        let _ = writeln!(
            prompt,
            "If the query is incoherent (coherence < {coherence_threshold:.2}) or unsafe \
             (safety < {safety_threshold:.2}), route to the fallback branch or output an empty route."
        );
        let _ = writeln!(prompt, "Only output JSON, no other text.");
        Some(prompt)
    }

    fn collect_classifier_models(
        &self,
        seen: &mut std::collections::HashSet<String>,
        out: &mut Vec<String>,
    ) {
        match self {
            ClassificationNode::Classifier {
                model, children, ..
            } => {
                if seen.insert(model.clone()) {
                    out.push(model.clone());
                }
                for child in children {
                    child.node.collect_classifier_models(seen, out);
                }
            }
            ClassificationNode::Fallback { node, .. } => {
                node.collect_classifier_models(seen, out);
            }
            _ => {}
        }
    }

    fn collect_terminals(&self, out: &mut Vec<(String, Option<String>, String)>) {
        match self {
            ClassificationNode::Terminal {
                route,
                group,
                description,
                ..
            } => out.push((route.clone(), group.clone(), description.clone())),
            ClassificationNode::Classifier { children, .. } => {
                for child in children {
                    child.node.collect_terminals(out);
                }
            }
            ClassificationNode::Fallback { node, .. } => node.collect_terminals(out),
            ClassificationNode::Filter { .. } => {}
        }
    }

    /// The `always_route` flag of the terminal named `route`, or `None`
    /// when no terminal bears that name.
    fn find_terminal(&self, route: &str) -> Option<bool> {
        match self {
            ClassificationNode::Terminal {
                route: name,
                always_route,
                ..
            } if name == route => Some(*always_route),
            ClassificationNode::Classifier { children, .. } => {
                children.iter().find_map(|c| c.node.find_terminal(route))
            }
            ClassificationNode::Fallback { node, .. } => node.find_terminal(route),
            _ => None,
        }
    }
}
// M13a stub
#[allow(dead_code)] pub fn prompt_from(_tree: &ClassificationTree) -> String { String::new() }
#[cfg(test)]
#[path = "../../tests/config_classification.rs"]
mod tests;
