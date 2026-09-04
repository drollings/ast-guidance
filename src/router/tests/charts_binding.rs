use super::*;

fn entity(id: &str, kind: &str, value: serde_json::Value) -> Entity {
    Entity {
        id: id.into(),
        kind: kind.into(),
        value,
    }
}

fn entity_match(name: &str, predicate: EntityPredicate, required: bool) -> DepSpec {
    DepSpec::EntityMatch {
        name: name.into(),
        description: "d".into(),
        predicate: Some(predicate),
        required,
    }
}

fn capability(name: &str) -> DepSpec {
    DepSpec::Capability { name: name.into() }
}

#[test]
fn exact_single_match() {
    let dep = entity_match(
        "report",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "title".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: None,
            }],
            any_of: vec![],
        },
        true,
    );
    let entities = vec![entity(
        "issue-1",
        "report",
        serde_json::json!({"title": "Boom"}),
    )];
    let b = bind_entities(&[dep], &entities);
    assert!(b.satisfied.contains("entity:report:issue-1"));
    assert!(b.unmatched.is_empty());
    assert!(b.ambiguous.is_empty());
    assert_eq!(b.entity_map["report"][0].id, "issue-1");
}

#[test]
fn nested_path_resolution() {
    assert_eq!(
        resolve_path(&serde_json::json!({"user": {"id": 7}}), "user.id"),
        Some(&serde_json::json!(7))
    );
    assert_eq!(
        resolve_path(
            &serde_json::json!({"items": [{"name": "a"}, {"name": "b"}]}),
            "items[1].name"
        ),
        Some(&serde_json::json!("b"))
    );
    assert_eq!(
        resolve_path(&serde_json::json!({"x": 1}), ".").map(serde_json::Value::is_object),
        Some(true)
    );
    assert!(resolve_path(&serde_json::json!({"a": 1}), "a.b").is_none());
}

#[test]
fn numeric_coercion() {
    let dep = entity_match(
        "port",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "port".into(),
                ty: FieldType::Number,
                required: true,
                min: Some(1.0),
                max: Some(65535.0),
                pattern: None,
            }],
            any_of: vec![],
        },
        true,
    );
    // Numeric string counts as Number.
    let entities = vec![entity(
        "svc",
        "service",
        serde_json::json!({"port": "8080"}),
    )];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.unmatched.len(), 0, "numeric string must coerce to Number");
    assert_eq!(b.satisfied.len(), 1);
}

#[test]
fn numeric_min_max_violation() {
    let dep = entity_match(
        "port",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "port".into(),
                ty: FieldType::Number,
                required: true,
                min: Some(1000.0),
                max: None,
                pattern: None,
            }],
            any_of: vec![],
        },
        true,
    );
    let entities = vec![entity("svc", "service", serde_json::json!({"port": 80}))];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.unmatched.len(), 1, "below-min value must not match");
}

#[test]
fn substring_pattern_matches() {
    let dep = entity_match(
        "email",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "address".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: Some("@example.com".into()),
            }],
            any_of: vec![],
        },
        true,
    );
    let entities = vec![
        entity(
            "u1",
            "user",
            serde_json::json!({"address": "a@example.com"}),
        ),
        entity("u2", "user", serde_json::json!({"address": "b@other.org"})),
    ];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.entity_map["email"][0].id, "u1");
    assert_eq!(b.unmatched.len(), 0);
}

#[test]
fn missing_required_goes_unmatched() {
    let dep = entity_match(
        "report",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "title".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: None,
            }],
            any_of: vec![],
        },
        true,
    );
    let entities = vec![entity("e1", "report", serde_json::json!({}))];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.unmatched, vec!["report".to_string()]);
    assert!(b.satisfied.is_empty());
}

#[test]
fn two_matches_are_ambiguous() {
    let dep = entity_match(
        "report",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "kind".into(),
                ty: FieldType::String,
                required: false,
                min: None,
                max: None,
                pattern: None,
            }],
            any_of: vec![],
        },
        true,
    );
    let entities = vec![
        entity("e1", "report", serde_json::json!({"kind": "bug"})),
        entity("e2", "report", serde_json::json!({"kind": "bug"})),
    ];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.ambiguous.len(), 1);
    assert_eq!(b.ambiguous[0].candidates.len(), 2);
    assert!(b.satisfied.is_empty());
}

#[test]
fn optional_missing_is_skipped() {
    let dep = entity_match(
        "report",
        EntityPredicate {
            fields: vec![FieldRule {
                path: "title".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: None,
            }],
            any_of: vec![],
        },
        false,
    );
    let entities = vec![entity("e1", "report", serde_json::json!({}))];
    let b = bind_entities(&[dep], &entities);
    assert!(b.unmatched.is_empty(), "optional dep must be skipped");
    assert!(b.satisfied.is_empty());
}

#[test]
fn capability_bound_by_kind() {
    let entities = vec![
        entity("f1", "file", serde_json::json!({"name": "a.rs"})),
        entity("f2", "file", serde_json::json!({"name": "b.rs"})),
    ];
    let b = bind_entities(&[capability("file")], &entities);
    assert_eq!(b.ambiguous.len(), 1, "two entities of kind=file");
    assert_eq!(b.ambiguous[0].dep, "file");

    let single = vec![entity("f1", "file", serde_json::json!({}))];
    let b = bind_entities(&[capability("file")], &single);
    assert_eq!(b.satisfied, HashSet::from(["entity:file:f1".to_string()]));
}

#[test]
fn capability_bound_by_provides_array() {
    let entities = vec![entity(
        "agent1",
        "agent",
        serde_json::json!({"provides": ["spell_check", "lint"]}),
    )];
    let b = bind_entities(&[capability("spell_check")], &entities);
    assert!(b.satisfied.contains("entity:agent:agent1"));
}

#[test]
fn zero_match_capability_is_unmatched() {
    // A capability dep with no matching entity surfaces as unmatched
    // instead of silently binding to nothing.
    let b = bind_entities(&[capability("spell_check")], &[]);
    assert_eq!(b.unmatched, vec!["spell_check".to_string()]);
    assert!(b.satisfied.is_empty());
    assert!(b.ambiguous.is_empty());
}

#[test]
fn matched_capability_leaves_unmatched_empty() {
    let entities = vec![entity(
        "agent1",
        "agent",
        serde_json::json!({"provides": ["spell_check"]}),
    )];
    let b = bind_entities(&[capability("spell_check")], &entities);
    assert!(b.unmatched.is_empty());
    assert!(b.satisfied.contains("entity:agent:agent1"));
}

#[test]
fn bind_chart_filters_in_graph_capabilities_out_of_unmatched() {
    // Chart-aware binding: a capability dep provided by another chart
    // target in-graph is bound by the graph, not by context entities —
    // it must not surface as a selection gap.
    let chart_json = r#"{
        "name": "internal_chain",
        "description": "in-graph capability chain",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            { "name": "a", "provides": ["a_out"], "depends": [],
              "template": "a {{ request }}", "essential": true },
            { "name": "b", "provides": ["b_out"], "depends": [
                { "kind": "capability", "name": "a_out" }
              ], "template": "b {{ upstream.a.output }}", "essential": true },
            { "name": "c", "provides": ["c_out"], "depends": [
                { "kind": "capability", "name": "b_out" },
                { "kind": "capability", "name": "external_data" }
              ], "template": "c {{ upstream.b.output }}", "essential": true }
        ]
    }"#;
    let chart: super::super::ChartDef = serde_json::from_str(chart_json).unwrap();
    // No entities at all: `a_out`/`b_out` are in-graph-provided and must
    // be filtered out; `external_data` has no provider and no entity, so
    // it remains the only gap.
    let b = bind_chart(&chart, &[]);
    assert_eq!(b.unmatched, vec!["external_data".to_string()]);
}

#[test]
fn entity_match_without_predicate_unmatched_if_required() {
    let dep = DepSpec::EntityMatch {
        name: "who".into(),
        description: "d".into(),
        predicate: None,
        required: true,
    };
    let entities = vec![entity("e1", "user", serde_json::json!({}))];
    let b = bind_entities(&[dep], &entities);
    assert_eq!(b.unmatched, vec!["who".to_string()]);
}

#[test]
fn any_of_alternative_matches() {
    let pred = EntityPredicate {
        fields: vec![FieldRule {
            path: "kind".into(),
            ty: FieldType::String,
            required: true,
            min: None,
            max: None,
            pattern: None,
        }],
        any_of: vec![EntityPredicate {
            fields: vec![FieldRule {
                path: "status".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: Some("blocked".into()),
            }],
            any_of: vec![],
        }],
    };
    // Top-level `fields` requires `kind`, but value has `status` only →
    // fields fail, whole predicate fails. With an empty top-level fields
    // the any_of alone decides.
    let pred2 = EntityPredicate {
        fields: vec![],
        any_of: vec![EntityPredicate {
            fields: vec![FieldRule {
                path: "status".into(),
                ty: FieldType::String,
                required: true,
                min: None,
                max: None,
                pattern: Some("blocked".into()),
            }],
            any_of: vec![],
        }],
    };
    let v = serde_json::json!({"status": "blocked"});
    assert!(!evaluate_predicate(&pred, &v));
    assert!(evaluate_predicate(&pred2, &v));
    assert!(!evaluate_predicate(
        &pred2,
        &serde_json::json!({"status": "open"})
    ));
}

#[test]
fn parse_entities_from_ctx_reads_metadata() {
    let mut ctx = WorkContext::default();
    assert!(parse_entities_from_ctx(&ctx).is_empty());
    let entities = vec![entity("e1", "report", serde_json::json!({"title": "T"}))];
    ctx.set_structured(ENTITIES_META_KEY, &entities);
    let parsed = parse_entities_from_ctx(&ctx);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].id, "e1");
}

#[test]
fn parse_entities_from_ctx_ignores_garbage() {
    let mut ctx = WorkContext::default();
    ctx.structured
        .insert(ENTITIES_META_KEY.into(), serde_json::json!([{"id": 1}]));
    assert!(parse_entities_from_ctx(&ctx).is_empty());
}

#[test]
fn three_target_chart_orders_via_topo_sort() {
    use fluent_dag::dep_graph::DependencyGraph;

    // Target A provides "a_out" (satisfied by a bound entity's provides
    // array). B depends on capability "a_out" and provides "b_out".
    // C depends on capability "b_out" and on entity-match "report".
    let deps_a = [];
    let deps_b = [capability("a_out")];
    let deps_c = vec![
        capability("b_out"),
        entity_match(
            "report",
            EntityPredicate {
                fields: vec![FieldRule {
                    path: "title".into(),
                    ty: FieldType::String,
                    required: true,
                    min: None,
                    max: None,
                    pattern: None,
                }],
                any_of: vec![],
            },
            true,
        ),
    ];

    // Bind target C's deps: b_out is not entity-satisfiable (it's a
    // chart-internal asset), report is.
    let entities = vec![entity("r1", "report", serde_json::json!({"title": "T"}))];
    let c_bindings = bind_entities(&deps_c, &entities);
    assert!(c_bindings.satisfied.contains("entity:report:r1"));
    // The zero-match capability `b_out` is now self-described as
    // unmatched at the per-target binding layer — the chart-level
    // `bind_chart` filters in-graph-provided capabilities, but plain
    // `bind_entities` has no chart context.
    assert_eq!(c_bindings.unmatched, vec!["b_out".to_string()]);

    // Build the chart graph (node = target name; deps = capability dep
    // names; provides = provides list + self-name).
    let mut graph: DependencyGraph<String> = DependencyGraph::new();
    graph
        .register(
            &"A".into(),
            &deps_a.iter().map(dep_name).collect::<Vec<_>>(),
            &["a_out".into(), "A".into()],
        )
        .unwrap();
    graph
        .register(
            &"B".into(),
            &deps_b.iter().map(dep_name).collect::<Vec<_>>(),
            &["b_out".into(), "B".into()],
        )
        .unwrap();
    graph
        .register(
            &"C".into(),
            &deps_c.iter().map(dep_name).collect::<Vec<_>>(),
            &["c_out".into(), "C".into()],
        )
        .unwrap();

    let order = graph.topo_sort().unwrap();
    assert_eq!(order, vec!["A", "B", "C"]);
}

fn dep_name(d: &DepSpec) -> String {
    d.name().to_string()
}
