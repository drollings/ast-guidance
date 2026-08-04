//! Entity binding layer — the deterministic "duck-typing" resolver.
//!
//! Binds context entities to a chart target's `DepSpec`s using structural
//! JSON-Schema-style predicates. This is the "first layer" of the dual-layer
//! resolver: it produces concrete asset keys and the caller (M5 stage) drives
//! the existing `DependencyGraph` (`topo_sort`, `ready_nodes`, `is_ready`).
//! Nothing in `dep_graph.rs` or `resolver.rs` is modified — this module is
//! pure data in, pure data out.
//!
//! LLM fallback (for `Unmatched`/`Ambiguous`) is M8.

use std::collections::{HashMap, HashSet};

use fluent_wvr::prelude::WorkContext;
use serde::{Deserialize, Serialize};

use super::{DepSpec, EntityPredicate, FieldRule, FieldType};

/// Structured-metadata key under which the HTTP handler / upstream stage
/// places the entity list (as `serde_json::Value`, read via `set_structured`).
pub const ENTITIES_META_KEY: &str = "entities";

/// A candidate context entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub kind: String, // "file" | "message" | "record" | ...
    pub value: serde_json::Value,
}

/// The result of binding a target's `DepSpec`s against a set of entities.
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    /// Concrete asset keys satisfied by bound entities: `"entity:{kind}:{id}"`.
    pub satisfied: HashSet<String>,
    /// dep name → bound entities (for the renderer context).
    pub entity_map: HashMap<String, Vec<Entity>>,
    /// Dep names that matched nothing and are required.
    pub unmatched: Vec<String>,
    /// Dep names with multiple structural matches (need LLM adjudication).
    pub ambiguous: Vec<AmbiguousDep>,
}

/// A dep that matched more than one entity — adjudication engages (M8).
#[derive(Debug, Clone)]
pub struct AmbiguousDep {
    pub dep: String,
    /// Human/LLM label from the dep spec (or the dep name when absent).
    pub description: String,
    pub candidates: Vec<Entity>,
}

/// Bind a target's dependency specs against candidate entities.
///
/// Deterministic path only:
/// - `EntityMatch` with a `predicate`: 0 matches → unmatched/skipped (per
///   `required`); 1 match → bound; >1 matches → ambiguous.
/// - `EntityMatch` without a `predicate`: cannot be evaluated
///   deterministically — treated as no match (required → unmatched, M8 LLM
///   fallback engages).
/// - `Capability { name }`: satisfied when an entity's `kind == name` or its
///   `value.provides` array contains `name`; a zero-match capability is
///   pushed into `unmatched` (D1) so capability-gapped charts classify
///   `Partial` instead of failing only at compile time.
///
/// This per-target form has no chart context, so it cannot know which
/// capability assets the chart's own targets provide in-graph. Use
/// [`bind_chart`] (which filters in-graph capabilities out of `unmatched`)
/// for chart-level selection decisions.
pub fn bind_entities(deps: &[DepSpec], entities: &[Entity]) -> Bindings {
    bind_entities_impl(deps, entities, &HashSet::new())
}

/// Chart-aware binding: capability assets provided by the chart's own
/// targets are bound by the graph (never a context-entity gap), so they are
/// excluded from `unmatched` (D1). `EntityMatch` semantics are unchanged.
fn bind_entities_impl(deps: &[DepSpec], entities: &[Entity], in_graph: &HashSet<&str>) -> Bindings {
    let mut bindings = Bindings::default();

    for dep in deps {
        match dep {
            DepSpec::Capability { name } => {
                if in_graph.contains(name.as_str()) {
                    // Satisfied by another chart target (its `provides` or
                    // self-provided name); compile resolves the edge via
                    // `provider_of`. Never an entity-binding gap.
                    continue;
                }
                let matches: Vec<&Entity> = entities
                    .iter()
                    .filter(|e| {
                        e.kind == *name
                            || e.value
                                .get("provides")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|p| {
                                    p.iter().any(|x| x.as_str() == Some(name.as_str()))
                                })
                    })
                    .collect();
                bind_matches(
                    &mut bindings,
                    dep.name(),
                    matches.into_iter().cloned().collect(),
                );
            }
            DepSpec::EntityMatch {
                name,
                description,
                predicate,
                required,
            } => {
                let Some(pred) = predicate else {
                    // No deterministic rule → cannot bind structurally. The
                    // LLM fallback (M8) adjudicates; surface as unmatched if
                    // required.
                    if *required {
                        bindings.unmatched.push(name.clone());
                    }
                    continue;
                };
                let matches: Vec<&Entity> = entities
                    .iter()
                    .filter(|e| evaluate_predicate(pred, &e.value))
                    .collect();
                if matches.is_empty() {
                    if *required {
                        bindings.unmatched.push(name.clone());
                    }
                } else if matches.len() == 1 {
                    bind_one(&mut bindings, dep.name(), matches[0]);
                } else {
                    bindings.ambiguous.push(AmbiguousDep {
                        dep: name.clone(),
                        description: description.clone(),
                        candidates: matches.into_iter().cloned().collect(),
                    });
                }
            }
        }
    }

    bindings
}

/// Bind every target in a chart against candidate entities and aggregate the
/// results into a single chart-level `Bindings`.
///
/// The aggregation is order-independent: satisfied assets and entity maps are
/// unioned, unmatched deps are deduplicated (in first-seen order), and
/// ambiguous deps accumulate. Used by chart selection (M7) to decide whether
/// a chosen chart is `Exact` or `Partial` (interview gaps).
///
/// Chart-aware (D1): a `Capability` dep that another chart target provides
/// in-graph is bound by the graph, not by context entities — it is excluded
/// from `unmatched`. Only capabilities with **no in-graph provider and no
/// matching entity** surface as gaps, driving the M8 interview instead of an
/// `Exact`-then-`ChartError::Compile`.
pub fn bind_chart(chart: &super::ChartDef, entities: &[Entity]) -> Bindings {
    let mut agg = Bindings::default();
    let in_graph = in_graph_assets(chart);
    for target in &chart.targets {
        let per_target = bind_entities_impl(&target.depends, entities, &in_graph);
        agg.satisfied.extend(per_target.satisfied);
        agg.entity_map.extend(per_target.entity_map);
        for dep in per_target.unmatched {
            if !agg.unmatched.contains(&dep) {
                agg.unmatched.push(dep);
            }
        }
        agg.ambiguous.extend(per_target.ambiguous);
    }
    agg
}

/// The set of capability assets the chart's own targets provide in-graph:
/// every target self-provides its name in addition to its explicit
/// `provides` list (the DependencySession convention).
fn in_graph_assets(chart: &super::ChartDef) -> HashSet<&str> {
    let mut set = HashSet::new();
    for target in &chart.targets {
        set.insert(target.name.as_str());
        for provides in &target.provides {
            set.insert(provides.as_str());
        }
    }
    set
}

/// Bind exactly one matched entity to a dep.
fn bind_one(bindings: &mut Bindings, dep: &str, entity: &Entity) {
    bindings.satisfied.insert(asset_key(entity));
    bindings
        .entity_map
        .entry(dep.to_string())
        .or_default()
        .push(entity.clone());
}

/// Apply the shared 0/1/>1 semantics to a match list (used by Capability).
///
/// A zero-match `Capability` dep is pushed into `unmatched` (D1/option a):
/// the chart then classifies `ChartFit::Partial { gaps }` (drives the M8
/// interview) instead of `Exact`-then-`ChartError::Compile`, and `compile.rs`
/// still treats an unbound capability as a hard error. `Capability` has no
/// `required` field, so a zero-match capability is *always* unmatched.
fn bind_matches(bindings: &mut Bindings, dep: &str, matches: Vec<Entity>) {
    if matches.is_empty() {
        bindings.unmatched.push(dep.to_string());
        return;
    }
    if matches.len() == 1 {
        bind_one(bindings, dep, &matches[0]);
    } else {
        bindings.ambiguous.push(AmbiguousDep {
            dep: dep.into(),
            description: dep.to_string(),
            candidates: matches,
        });
    }
}

/// Concrete asset key for a bound entity.
pub fn asset_key(entity: &Entity) -> String {
    format!("entity:{}:{}", entity.kind, entity.id)
}

/// Evaluate a predicate against a structured value.
///
/// Semantics: all `fields` rules must hold (AND), and if `any_of` is
/// non-empty, at least one sub-predicate must hold (OR). A predicate with
/// neither `fields` nor `any_of` matches anything.
pub fn evaluate_predicate(pred: &EntityPredicate, v: &serde_json::Value) -> bool {
    let fields_ok = pred.fields.iter().all(|rule| evaluate_field(rule, v));
    if !fields_ok {
        return false;
    }
    if pred.any_of.is_empty() {
        return true;
    }
    pred.any_of.iter().any(|sub| evaluate_predicate(sub, v))
}

/// Evaluate a single field rule against a value.
fn evaluate_field(rule: &FieldRule, v: &serde_json::Value) -> bool {
    let Some(field) = resolve_path(v, &rule.path) else {
        // Path must exist when required.
        return !rule.required;
    };

    if !type_matches(rule.ty, field) {
        return false;
    }

    // Numeric min/max — applies when the field is (or coerces to) a number.
    if rule.min.is_some() || rule.max.is_some() {
        match as_f64(field) {
            Some(n) => {
                if let Some(min) = rule.min {
                    if n < min {
                        return false;
                    }
                }
                if let Some(max) = rule.max {
                    if n > max {
                        return false;
                    }
                }
            }
            None => return false,
        }
    }

    // Substring pattern (repo convention — not regex). The string form of
    // the field must contain the pattern.
    if let Some(pattern) = &rule.pattern {
        match as_string(field) {
            Some(s) if s.contains(pattern.as_str()) => {}
            _ => return false,
        }
    }

    true
}

/// Coercive type check — numeric strings count as `Number` (matches the
/// classifier's sanitization tolerance).
fn type_matches(ty: FieldType, v: &serde_json::Value) -> bool {
    match ty {
        FieldType::Any => true,
        FieldType::String => v.is_string(),
        FieldType::Number => v.is_number() || v.as_str().is_some_and(|s| s.parse::<f64>().is_ok()),
        FieldType::Bool => v.is_boolean() || matches!(v.as_str(), Some("true" | "false")),
        FieldType::Array => v.is_array(),
    }
}

/// Coerce a value to `f64` (numbers pass through; numeric strings parse).
fn as_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.parse::<f64>().ok())
}

/// Coerce a scalar value to its string form (numbers/bools stringify).
fn as_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Resolve a dotted path (e.g. `"user.id"`, `"."` = root) into a value.
///
/// Navigation: object fields by name, array elements by `[n]` index
/// (`"items[0].name"`). Returns `None` when any segment is missing or
/// malformed.
pub fn resolve_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    if path.trim() == "." {
        return Some(v);
    }
    let mut current = v;
    for segment in path.split('.') {
        current = resolve_segment(current, segment)?;
    }
    Some(current)
}

/// Resolve a single path segment which may be `name` or `name[0]`.
fn resolve_segment<'a>(v: &'a serde_json::Value, segment: &str) -> Option<&'a serde_json::Value> {
    if let Some(open) = segment.find('[') {
        if !segment.ends_with(']') {
            return None;
        }
        let name = &segment[..open];
        let index: usize = segment[open + 1..segment.len() - 1].parse().ok()?;
        let obj = v.get(name)?;
        return obj.as_array()?.get(index);
    }
    v.get(segment)
}

/// Parse the entity list from a `WorkContext`'s structured channel.
///
/// The HTTP handler or an upstream stage places the entities as a typed
/// `serde_json::Value` under `ctx.structured["entities"]`. Absence or a
/// deserialization failure yields `[]` — never fails the request.
pub fn parse_entities_from_ctx(ctx: &WorkContext) -> Vec<Entity> {
    ctx.structured(ENTITIES_META_KEY).unwrap_or_default()
}

#[cfg(test)]
mod tests {
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
        // D1: a capability dep with no matching entity surfaces as unmatched
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
        // it must not surface as a selection gap (D1).
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
        // D1: the zero-match capability `b_out` is now self-described as
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
}
