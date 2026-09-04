use super::*;
use fluent_types::InterlinguaNamespace;

#[test]
fn test_class_lookup() {
    let cls = lookup_class("http://schema.org/Person");
    assert!(cls.is_some());
    assert_eq!(cls.unwrap().label, "Person");
}

#[test]
fn test_class_lookup_unknown() {
    assert!(lookup_class("http://unknown.example/").is_none());
}

#[test]
fn test_property_lookup() {
    let prop = lookup_property("http://www.w3.org/2000/01/rdf-schema#label");
    assert!(prop.is_some());
    assert_eq!(prop.unwrap().lod_target, Some(4));
}

#[test]
fn test_property_lookup_type() {
    let prop = lookup_property("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    assert!(prop.is_some());
    assert_eq!(prop.unwrap().lod_target, None);
}

#[test]
fn test_superclass_chain_person() {
    let chain = superclass_chain("http://schema.org/Person");
    assert!(chain.len() >= 2, "chain length: {}", chain.len());
    assert_eq!(chain[0], "http://schema.org/Person");
    assert_eq!(chain[1], "http://yago-knowledge.org/resource/Entity");
}

#[test]
fn test_domain_validation() {
    let prop = lookup_property("http://yago-knowledge.org/resource/hasGender");
    assert!(prop.is_some());
    assert_eq!(prop.unwrap().domain, Some("http://schema.org/Person"));
}

#[test]
fn test_transitive_property() {
    let prop = lookup_property("http://www.w3.org/2000/01/rdf-schema#subClassOf");
    assert!(prop.unwrap().transitive);
}

#[test]
fn test_is_subclass_of_identity() {
    assert!(is_subclass_of(
        "http://schema.org/Person",
        "http://schema.org/Person"
    ));
}

#[test]
fn test_is_subclass_of_direct() {
    assert!(is_subclass_of(
        "http://schema.org/Person",
        "http://yago-knowledge.org/resource/Entity"
    ));
}

#[test]
fn test_is_subclass_of_unrelated() {
    assert!(!is_subclass_of(
        "http://schema.org/Person",
        "http://schema.org/Product"
    ));
}

#[test]
fn test_is_subclass_of_unknown_child() {
    assert!(!is_subclass_of(
        "http://unknown/Foo",
        "http://yago-knowledge.org/resource/Entity"
    ));
}

#[test]
fn test_whitelist() {
    assert!(is_whitelisted("http://schema.org/Person"));
    assert!(is_whitelisted("http://yago-knowledge.org/resource/Entity"));
    assert!(!is_whitelisted("http://unknown/Foo"));
}

#[test]
fn test_all_classes_count() {
    assert_eq!(ALL_CLASSES.len(), 7);
}

#[test]
fn test_all_properties_count() {
    assert_eq!(ALL_PROPERTIES.len(), 11);
}

#[test]
fn property_ids_are_deterministic() {
    let p = property_interlingua_id(&PROP_SUBCLASS);
    assert_eq!(p.namespace(), InterlinguaNamespace::RdfProperty);
    assert_eq!(
        p.local_id(),
        local_id_of(hash_iri("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
    );
    // Stable across calls.
    assert_eq!(p, property_interlingua_id(&PROP_SUBCLASS));
    assert_eq!(subclass_property_id(), p);
    assert_eq!(
        property_interlingua_id_by_iri("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
        Some(p)
    );
    assert_eq!(property_interlingua_id_by_iri("http://unknown/prop"), None);
}

#[test]
fn whitelist_id_is_namespace_aware_and_truncation_correct() {
    let person = InterlinguaId::new(
        InterlinguaNamespace::YagoClass,
        local_id_of(hash_iri("http://schema.org/Person")),
    );
    assert!(is_whitelisted_id(person));
    // A YagoClass id that is NOT a whitelist class is not whitelisted.
    let other = InterlinguaId::new(
        InterlinguaNamespace::YagoClass,
        local_id_of(hash_iri("http://yago-knowledge.org/resource/NotWhitelisted")),
    );
    assert!(!is_whitelisted_id(other));
    // A non-Yago namespace never matches, even with a colliding local.
    let lemma = InterlinguaId::new(InterlinguaNamespace::SpacyLemma, person.local_id());
    assert!(!is_whitelisted_id(lemma));
}

#[test]
fn whitelist_hash_matches_truncated_ids() {
    // A full-width hash whose 48-bit local matches a whitelisted class.
    let person_hash = hash_iri("http://schema.org/Person");
    assert!(is_whitelisted_hash(person_hash));
    assert!(!is_whitelisted_hash(hash_iri("http://unknown/Foo")));
    // Truncation-agreement: the truncated id's local is whitelisted.
    assert!(is_whitelisted_id(InterlinguaId::new(
        InterlinguaNamespace::YagoClass,
        local_id_of(person_hash),
    )));
}
