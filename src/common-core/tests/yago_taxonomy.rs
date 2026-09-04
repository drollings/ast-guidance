//! Parity with the deprecated `src/ontology/tools/*.py` shims on a
//! checked-in fixture TTL (`fixtures/yago_mini.ttl`).
//!
//! Name normalization unifies on the Rust semantics
//! (`common_core::yago_normalize`): the `_UXXXX` decode is the documented
//! delta vs Python (Python leaves `u0028`, Rust decodes to `(`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use common_core::yago_taxonomy::{
    build_json, collect_edges, compute_kept_iris, normalize_curie, prune_ttl, taxonomy_from_ttl,
    to_flat, to_json_string, ClassEntry,
};

fn fixture_ttl() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("yago_mini.ttl");
    std::fs::read_to_string(&path).expect("fixture TTL readable")
}

fn entry(parents: &[&str], ancestors: &[&str]) -> ClassEntry {
    ClassEntry {
        parents: parents.iter().map(ToString::to_string).collect(),
        ancestors: ancestors.iter().map(ToString::to_string).collect(),
    }
}

#[test]
fn yago_python_parity_on_fixture() {
    // Mirrors the pytest cases in parse_yago_taxonomy_to_json.py:
    // simple hierarchy, multiple inheritance, yago-only filter, Q-suffix
    // strip + lowercase + underscores→spaces.
    let map = taxonomy_from_ttl(&fixture_ttl(), true, true);

    let mut expected = BTreeMap::new();
    expected.insert(
        "yago:adult video game".to_string(),
        entry(&["yago:video game"], &["yago:software", "yago:video game"]),
    );
    expected.insert("yago:a".to_string(), entry(&["yago:root"], &["yago:root"]));
    expected.insert("yago:b".to_string(), entry(&["yago:root"], &["yago:root"]));
    expected.insert(
        "yago:c".to_string(),
        entry(&["yago:a", "yago:b"], &["yago:a", "yago:b", "yago:root"]),
    );
    expected.insert(
        "yago:city".to_string(),
        entry(
            &["yago:place"],
            &["rdfs:class", "yago:entity", "yago:place"],
        ),
    );
    expected.insert(
        "yago:creativework".to_string(),
        entry(&[], &[]),
    );
    expected.insert(
        "yago:entity".to_string(),
        entry(&["rdfs:class"], &["rdfs:class"]),
    );
    expected.insert(
        "yago:place".to_string(),
        entry(&["yago:entity"], &["rdfs:class", "yago:entity"]),
    );
    // Documented Rust delta: `_UXXXX` decodes where Python left `u0028`.
    expected.insert(
        "yago:remix (work)".to_string(),
        entry(&["yago:creativework"], &["yago:creativework"]),
    );
    expected.insert("yago:root".to_string(), entry(&[], &[]));
    expected.insert("yago:software".to_string(), entry(&[], &[]));
    expected.insert(
        "yago:video game".to_string(),
        entry(&["yago:software"], &["yago:software"]),
    );

    assert_eq!(map, expected, "fixture JSON under Rust semantics");
    // The yago-only filter drops the schema: classes as keys.
    assert!(
        !map.keys().any(|k| k.starts_with("schema:")),
        "yago-only filter, keys: {:?}",
        map.keys().collect::<Vec<_>>()
    );
    // ... but their edges never leak into yago entries either.
    for (k, v) in &map {
        assert!(
            v.parents.iter().chain(&v.ancestors).all(|n| !n.starts_with("schema:")),
            "no schema: names in {k}"
        );
    }
}

#[test]
fn parity_include_all_keeps_schema_classes() {
    // Mirrors `test_yago_only_filter --include-all`: schema:City appears.
    let map = taxonomy_from_ttl(&fixture_ttl(), false, true);
    assert!(map.contains_key("yago:city"), "yago keys stay");
    assert!(map.contains_key("schema:city"), "include-all keeps schema keys");
    assert_eq!(
        map["schema:city"].parents,
        vec!["schema:place".to_string()]
    );
}

#[test]
fn parity_flat_mode_lists_ancestors() {
    let map = taxonomy_from_ttl(&fixture_ttl(), true, true);
    let flat = to_flat(&map);
    assert_eq!(
        flat["yago:city"],
        vec![
            "rdfs:class".to_string(),
            "yago:entity".to_string(),
            "yago:place".to_string()
        ]
    );
}

#[test]
fn parity_json_string_is_sorted_and_newline_terminated() {
    let map = taxonomy_from_ttl(&fixture_ttl(), true, true);
    let text = to_json_string(&map);
    assert!(text.ends_with('\n'), "trailing newline like the shim");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    let keys: Vec<&str> = parsed
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(keys, sorted, "sorted keys");
    // Round-trip: the emitted text parses back to the same map.
    let back: BTreeMap<String, ClassEntry> =
        serde_json::from_str(&text).expect("round-trip");
    assert_eq!(back, map);
}

#[test]
fn parity_collect_edges_expands_curies() {
    let (prefixes, edges) = collect_edges(&fixture_ttl());
    assert_eq!(
        prefixes.get("yago").map(String::as_str),
        Some("http://yago-knowledge.org/resource/")
    );
    assert!(
        edges.contains(&(
            "http://yago-knowledge.org/resource/City_Q515".to_string(),
            "http://yago-knowledge.org/resource/Place".to_string()
        )),
        "CURIEs expand to full IRIs"
    );
}

#[test]
fn parity_build_json_direct_edges() {
    // `build_json` on hand-built edges (mirrors the multiple-inheritance
    // pytest): C inherits A + B, both inherit Root.
    let edges = vec![
        ("http://a/C".to_string(), "http://a/A".to_string()),
        ("http://a/C".to_string(), "http://a/B".to_string()),
        ("http://a/A".to_string(), "http://a/Root".to_string()),
        ("http://a/B".to_string(), "http://a/Root".to_string()),
    ];
    let prefixes = BTreeMap::new();
    let map = build_json(&edges, &prefixes, false, false);
    let c = &map["http://a/c"];
    assert_eq!(c.parents, vec!["http://a/a", "http://a/b"]);
    assert_eq!(
        c.ancestors,
        vec!["http://a/a", "http://a/b", "http://a/root"]
    );
}

const PRUNE_TTL: &str = "@prefix yago: <http://yago-knowledge.org/resource/> .\n\
@prefix schema: <http://schema.org/> .\n\
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
yago:Entity rdfs:subClassOf rdfs:Class .\n\
yago:Agent rdfs:subClassOf yago:Entity .\n\
yago:Place rdfs:subClassOf yago:Entity .\n\
yago:City rdfs:subClassOf yago:Place .\n\
yago:Dog rdfs:subClassOf yago:Mammal .\n\
yago:Mammal rdfs:subClassOf yago:Entity .\n";

#[test]
fn prune_tier_one_keeps_immediate_children() {
    // Mirrors `test_tier_one_keeps_immediate_children` in
    // prune_yago_taxonomy.py: depth-2 classes are pruned.
    let out = prune_ttl(PRUNE_TTL, 1);
    assert!(out.contains("yago:Agent"), "tier-1 child kept");
    assert!(out.contains("yago:Place"), "tier-1 child kept");
    assert!(out.contains("yago:Mammal"), "tier-1 child kept");
    assert!(!out.contains("yago:City"), "depth-2 pruned");
    assert!(!out.contains("yago:Dog"), "depth-2 pruned");
}

#[test]
fn prune_tier_two_keeps_grandchildren() {
    let out = prune_ttl(PRUNE_TTL, 2);
    assert!(out.contains("yago:City"), "tier-2 grandchild kept");
    assert!(out.contains("yago:Dog"), "tier-2 grandchild kept");
}

#[test]
fn prune_prefixes_always_preserved() {
    let out = prune_ttl(PRUNE_TTL, 0);
    assert!(out.contains("@prefix yago:"), "headers survive");
    assert!(!out.contains("yago:Agent"), "tier-0 keeps roots only");
}

#[test]
fn prune_multiple_inheritance_keeps_shallow_path() {
    // Mirrors `test_compute_kept_iris_multiple_inheritance`: C is depth 2
    // via both parents — out at tier 1, in at tier 2.
    let edges = vec![
        ("http://a/C".to_string(), "http://a/A".to_string()),
        ("http://a/C".to_string(), "http://a/B".to_string()),
        ("http://a/A".to_string(), "http://a/Root".to_string()),
        ("http://a/B".to_string(), "http://a/Root".to_string()),
    ];
    let kept = compute_kept_iris(&edges, 1);
    assert!(kept.contains("http://a/Root"));
    assert!(kept.contains("http://a/A"));
    assert!(kept.contains("http://a/B"));
    assert!(!kept.contains("http://a/C"));
    let kept2 = compute_kept_iris(&edges, 2);
    assert!(kept2.contains("http://a/C"));
}

#[test]
fn normalize_curie_preserves_lowercased_prefix() {
    assert_eq!(normalize_curie("yago:City_Q515"), "yago:city");
    assert_eq!(normalize_curie("YAGO:City"), "yago:city");
    assert_eq!(
        normalize_curie("<http://yago-knowledge.org/resource/City_Q515>"),
        "<http://yago-knowledge.org/resource/city>"
    );
}
