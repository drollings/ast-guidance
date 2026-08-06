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
//! The flat sections (`pipelines`, `routes`, `system_prompt`, `score_matrix`,
//! `models`, `model_groups`) are unchanged and still load; when a tree is
//! present the flat *views* the rest of the server expects (route→pipeline
//! mapping, system prompt) are derived from it (M4.4).

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::config::filters::FilterOutcome;

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
    /// `classifier_model` is configured (M4.1 boot).
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
    /// the source for the derived flat `routes` view (M4.4).
    pub fn terminal_views(&self) -> Vec<(String, Option<String>, String)> {
        let mut out = Vec::new();
        self.root.collect_terminals(&mut out);
        out
    }

    /// Auto-generate the classifier system prompt from the root node's children
    /// and descriptions — the derived `system_prompt` view for tree configs
    /// (M4.4). `None` when the root is not a classifier or has no routeable
    /// children.
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
    /// A dispatch target. `route` names an entry in the flat `routes` map;
    /// when the tree config has no such entry, `group` supplies one for the
    /// derived flat view.
    Terminal {
        route: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group: Option<String>,
        #[serde(default)]
        description: String,
    },
    /// A deterministic regex check over the user message. Short-circuits the
    /// enclosing classifier when a pattern matches.
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
        let _ = writeln!(prompt, "  \"route\": \"<exactly one of: {}>\"", keys.join(", "));
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
        let _ = writeln!(prompt, "  \"reason\": \"brief explanation for the routing decision\"");
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
            ClassificationNode::Classifier { model, children, .. } => {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfig;

    fn tree_json() -> &'static str {
        r#"{
            "root": {
                "type": "classifier",
                "description": "request router",
                "model": "fast",
                "coherence_threshold": 0.6,
                "safety_threshold": 0.4,
                "children": [
                    {
                        "key": "code",
                        "description": "programming and implementation",
                        "node": { "type": "terminal", "route": "code", "group": "code" }
                    },
                    {
                        "key": "translation",
                        "description": "translation between languages",
                        "node": { "type": "terminal", "route": "translation", "group": "translation" }
                    },
                    {
                        "key": "blocked",
                        "description": "known-bad content",
                        "node": {
                            "type": "filter",
                            "description": "blocks banned topics",
                            "patterns": ["harmful pattern"],
                            "outcome": "hard_reject"
                        }
                    },
                    {
                        "key": "general",
                        "description": "everything else",
                        "node": {
                            "type": "fallback",
                            "description": "default branch",
                            "node": { "type": "terminal", "route": "local", "group": "question" }
                        }
                    }
                ]
            }
        }"#
    }

    #[test]
    fn tree_json_parses_all_node_types() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        let ClassificationNode::Classifier {
            description,
            model,
            coherence_threshold,
            safety_threshold,
            children,
        } = &tree.root
        else {
            panic!("root should be a classifier")
        };
        assert_eq!(description, "request router");
        assert_eq!(model, "fast");
        assert_eq!(*coherence_threshold, Some(0.6));
        assert_eq!(*safety_threshold, Some(0.4));
        assert_eq!(children.len(), 4);
        assert!(matches!(children[0].node, ClassificationNode::Terminal { .. }));
        assert!(matches!(children[1].node, ClassificationNode::Terminal { .. }));
        assert!(matches!(children[2].node, ClassificationNode::Filter { .. }));
        assert!(matches!(children[3].node, ClassificationNode::Fallback { .. }));
    }

    #[test]
    fn tree_round_trips() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        let serialized = serde_json::to_string(&tree).unwrap();
        let back: ClassificationTree = serde_json::from_str(&serialized).unwrap();
        match (&tree.root, &back.root) {
            (ClassificationNode::Classifier { children: a, .. }, ClassificationNode::Classifier { children: b, .. }) => {
                assert_eq!(a.len(), b.len());
                assert_eq!(a[0].key, b[0].key);
                assert_eq!(&a[3].node, &ClassificationNode::Fallback { description: Some("default branch".into()), node: Box::new(ClassificationNode::Terminal { route: "local".into(), group: Some("question".into()), description: String::new() }) });
            }
            _ => panic!("round-trip changed the root type"),
        }
    }

    #[test]
    fn router_config_parses_classification_section() {
        let json = format!(
            r#"{{ "classification": {tree}, "models": {{}}, "model_groups": {{}} }}"#,
            tree = tree_json()
        );
        let cfg: RouterConfig = serde_json::from_str(&json).unwrap();
        assert!(cfg.classification.is_some());
    }

    #[test]
    fn flat_config_without_classification_is_none() {
        let cfg: RouterConfig =
            serde_json::from_str(r#"{"server": {"bind_addr": "127.0.0.1:0"}}"#).unwrap();
        assert!(cfg.classification.is_none());
    }

    #[test]
    fn root_classifier_model_resolved() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        assert_eq!(tree.root_classifier_model(), Some("fast"));
    }

    #[test]
    fn classifier_model_keys_dedup_across_depth() {
        let json = r#"{
            "root": {
                "type": "classifier",
                "description": "root",
                "model": "fast",
                "children": [
                    {
                        "key": "sub",
                        "description": "sub",
                        "node": {
                            "type": "classifier",
                            "description": "subdomain",
                            "model": "small",
                            "children": [
                                {
                                    "key": "code",
                                    "description": "code",
                                    "node": { "type": "terminal", "route": "code" }
                                }
                            ]
                        }
                    },
                    {
                        "key": "again",
                        "description": "again",
                        "node": {
                            "type": "classifier",
                            "description": "second small",
                            "model": "small",
                            "children": []
                        }
                    }
                ]
            }
        }"#;
        let tree: ClassificationTree = serde_json::from_str(json).unwrap();
        assert_eq!(tree.classifier_model_keys(), vec!["fast".to_string(), "small".to_string()]);
    }

    #[test]
    fn terminal_views_collect_routes_and_groups() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        let mut views = tree.terminal_views();
        views.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            views,
            vec![
                ("code".into(), Some("code".into()), String::new()),
                ("local".into(), Some("question".into()), String::new()),
                ("translation".into(), Some("translation".into()), String::new()),
            ]
        );
    }

    #[test]
    fn build_prompt_lists_only_routeable_children() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        let prompt = tree.root.build_prompt(0.6, 0.4).expect("prompt");
        assert!(prompt.contains("You are a request router."));
        assert!(prompt.contains("- code: programming and implementation"));
        assert!(prompt.contains("- translation: translation between languages"));
        // The filter and fallback children are NOT LLM-choosable branches.
        assert!(!prompt.contains("known-bad content"));
        assert!(!prompt.contains("default branch"));
        assert!(prompt.contains("\"route\": \"<exactly one of: code, translation>\""));
        assert!(prompt.contains("\"coherence\": 0.0-1.0"));
        assert!(prompt.contains("\"complexity\": 0-10"));
        assert!(prompt.contains("coherence < 0.60"));
        assert!(prompt.contains("safety < 0.40"));
    }

    #[test]
    fn derive_system_prompt_uses_root_thresholds() {
        let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
        let prompt = tree.derive_system_prompt().unwrap();
        assert!(prompt.contains("You are a request router."));
        assert!(prompt.contains("coherence < 0.60"));
    }

    #[test]
    fn derive_system_prompt_none_for_non_classifier_root() {
        let tree: ClassificationTree =
            serde_json::from_str(r#"{"root": {"type": "terminal", "route": "fast"}}"#).unwrap();
        assert!(tree.derive_system_prompt().is_none());
    }

    #[test]
    fn filter_node_parses_outcome_variants() {
        let node: ClassificationNode =
            serde_json::from_str(r#"{"type": "filter", "description": "d", "patterns": ["x"], "outcome": "soft_redirect", "redirect_to": "code"}"#).unwrap();
        let ClassificationNode::Filter {
            outcome,
            redirect_to,
            ..
        } = node
        else {
            panic!("expected filter")
        };
        assert_eq!(outcome, FilterOutcome::SoftRedirect);
        assert_eq!(redirect_to.as_deref(), Some("code"));
    }

    #[test]
    fn fallback_node_parses_and_wraps() {
        let node: ClassificationNode = serde_json::from_str(
            r#"{"type": "fallback", "node": {"type": "terminal", "route": "local"}}"#,
        )
        .unwrap();
        assert!(matches!(node, ClassificationNode::Fallback { .. }));
    }
}
