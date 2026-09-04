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
    assert!(matches!(
        children[0].node,
        ClassificationNode::Terminal { .. }
    ));
    assert!(matches!(
        children[1].node,
        ClassificationNode::Terminal { .. }
    ));
    assert!(matches!(
        children[2].node,
        ClassificationNode::Filter { .. }
    ));
    assert!(matches!(
        children[3].node,
        ClassificationNode::Fallback { .. }
    ));
}

#[test]
fn tree_round_trips() {
    let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
    let serialized = serde_json::to_string(&tree).unwrap();
    let back: ClassificationTree = serde_json::from_str(&serialized).unwrap();
    match (&tree.root, &back.root) {
        (
            ClassificationNode::Classifier { children: a, .. },
            ClassificationNode::Classifier { children: b, .. },
        ) => {
            assert_eq!(a.len(), b.len());
            assert_eq!(a[0].key, b[0].key);
            assert_eq!(
                &a[3].node,
                &ClassificationNode::Fallback {
                    description: Some("default branch".into()),
                    node: Box::new(ClassificationNode::Terminal {
                        route: "local".into(),
                        group: Some("question".into()),
                        description: String::new(),
                        always_route: false,
                    })
                }
            );
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
    assert_eq!(
        tree.classifier_model_keys(),
        vec!["fast".to_string(), "small".to_string()]
    );
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
            (
                "translation".into(),
                Some("translation".into()),
                String::new()
            ),
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
        match_interlingua,
        ..
    } = node
    else {
        panic!("expected filter")
    };
    assert_eq!(outcome, FilterOutcome::SoftRedirect);
    assert_eq!(redirect_to.as_deref(), Some("code"));
    assert!(match_interlingua.is_none(), "regex filter has no interlingua match");
}

#[test]
fn filter_node_parses_match_interlingua() {
    let node: ClassificationNode = serde_json::from_str(
        r#"{"type": "filter", "description": "report requests", "match_interlingua": {"predicate_id": 2251799813685260, "object_id": 2251799813685262, "confidence_min": 0.5}, "outcome": "soft_redirect", "redirect_to": "report"}"#,
    )
    .unwrap();
    let ClassificationNode::Filter {
        patterns,
        match_interlingua,
        ..
    } = node
    else {
        panic!("expected filter")
    };
    assert!(patterns.is_empty(), "interlingua filter ignores patterns");
    let m = match_interlingua.expect("match");
    assert_eq!(m.predicate_id, Some(InterlinguaId::from_u64(2251799813685260)));
    assert_eq!(m.object_id, Some(InterlinguaId::from_u64(2251799813685262)));
    assert_eq!(m.confidence_min, Some(0.5));
    assert!(m.subject_id.is_none());
}

#[test]
fn existing_configs_deserialize_unchanged() {
    // A pre-C6 filter (no match_interlingua field) still deserializes.
    let node: ClassificationNode = serde_json::from_str(
        r#"{"type": "filter", "description": "d", "patterns": ["harmful"], "outcome": "hard_reject"}"#,
    )
    .unwrap();
    assert!(matches!(node, ClassificationNode::Filter { .. }));
    let tree: ClassificationTree = serde_json::from_str(tree_json()).unwrap();
    let _ = tree;
}

#[test]
fn fallback_node_parses_and_wraps() {
    let node: ClassificationNode = serde_json::from_str(
        r#"{"type": "fallback", "node": {"type": "terminal", "route": "local"}}"#,
    )
    .unwrap();
    assert!(matches!(node, ClassificationNode::Fallback { .. }));
}
