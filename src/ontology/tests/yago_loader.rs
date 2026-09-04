use super::*;
use fluent_types::local_id_of;

fn loader() -> (YaGoLoader, LoadStats) {
    let mut l = YaGoLoader::new();
    let stats = l.load_embedded().expect("embedded registry");
    (l, stats)
}

#[test]
fn embedded_registry_loads_all_entries() {
    let (_l, stats) = loader();
    assert!(stats.classes >= 7, "at least the reference classes");
    assert!(stats.edges > 0);
}

#[test]
fn ids_are_deterministic_and_truncated() {
    let (l, _) = loader();
    let concepts = &l.into_concepts();
    let person = concepts
        .iter()
        .find(|c| c.canonical_name == "schema:Person")
        .expect("Person");
    // local is the 48-bit truncation; node_id the full 64-bit hash.
    assert_eq!(person.id, yago_class_id("http://schema.org/Person"));
    assert_eq!(person.id.local_id(), local_id_of(hash_iri("http://schema.org/Person")));
    let node_id = person.node_id.expect("node_id set");
    assert_eq!(node_id.as_int(), hash_iri("http://schema.org/Person"));
    assert_ne!(node_id.as_int(), person.id.local_id(), "F5: stored, never derived");
}

#[test]
fn schema_person_resolves_by_canonical_name() {
    let (l, _) = loader();
    let concepts = l.into_concepts();
    let person = concepts
        .iter()
        .find(|c| c.canonical_name == "schema:Person")
        .expect("Person");
    assert_eq!(person.label.as_deref(), Some("Person"));
    assert!(person.id.is_yago());
}

#[test]
fn subclass_edges_are_present() {
    let (l, _) = loader();
    let edges = l.subclass_edges();
    assert!(
        edges
            .iter()
            .any(|(child, parent)| {
                child == &yago_class_id("http://yago-knowledge.org/resource/Dog")
                    && parent == &yago_class_id("http://yago-knowledge.org/resource/Mammal")
            }),
        "Dog ← Mammal edge present"
    );
}

#[test]
fn load_class_labels_overrides() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let labels = dir.path().join("labels.json");
    std::fs::write(
        &labels,
        r#"[{"iri":"http://schema.org/Person","label":"human person"}]"#,
    )
    .expect("write");
    let (mut l, _) = loader();
    let applied = l.load_class_labels(&labels).expect("labels");
    assert_eq!(applied, 1);
    let concepts = l.into_concepts();
    let person = concepts
        .iter()
        .find(|c| c.canonical_name == "schema:Person")
        .expect("Person");
    assert_eq!(person.label.as_deref(), Some("human person"));
}

#[test]
fn to_metadata_parent_class_id_matches_subclass_edges() {
    // DRY (red-team M2): `parent_class_id` and `subclass_edges` must derive
    // from the SAME `ClassEntry.superclass` field — never two sources of
    // edges. For every concept with a parent, the metadata's parent id
    // appears as the parent of that exact child in `subclass_edges`.
    let (l, _) = loader();
    let edges = l.subclass_edges().to_vec();
    let concepts = l.into_concepts();
    assert!(concepts.iter().any(|c| c.parent_class_id.is_some()));
    for c in &concepts {
        if let Some(parent) = c.parent_class_id {
            assert!(
                edges.iter().any(|(child, p)| child == &c.id && p == &parent),
                "{} parent {parent} must appear in subclass_edges",
                c.canonical_name
            );
        }
    }
}

#[test]
fn load_taxonomy_from_file() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("classes.json");
    std::fs::write(
        &path,
        r#"[{"iri":"http://yago-knowledge.org/resource/Foo","label":"Foo","superclass":"http://yago-knowledge.org/resource/Entity"}]"#,
    )
    .expect("write");
    let mut l = YaGoLoader::new();
    let stats = l.load_taxonomy(&path).expect("load");
    assert_eq!(stats.classes, 1);
    assert_eq!(stats.edges, 1);
}
