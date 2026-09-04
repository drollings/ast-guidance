use super::*;

fn build_triple(s: &str, p: &str, o: &str) -> Triple {
    Triple {
        subject: Term::Iri(s.to_string()),
        predicate: Term::Iri(p.to_string()),
        object: Term::Iri(o.to_string()),
    }
}

const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";

#[test]
fn test_inference_empty() {
    let engine = InferenceEngine::new();
    let result = engine.infer(&[]).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_rule_addition() {
    let mut engine = InferenceEngine::new();
    engine.add_rule(InferenceRule {
        rule_type: RuleType::SubclassTransitivity,
        trigger_predicate: RDFS_SUBCLASS_OF.to_string(),
    });
    assert_eq!(engine.rules.len(), 1);
}

#[test]
fn test_subclass_transitivity_ab_bc_gives_ac() {
    let ab = build_triple("Scientist", RDFS_SUBCLASS_OF, "Person");
    let bc = build_triple("Person", RDFS_SUBCLASS_OF, "Agent");

    let mut engine = InferenceEngine::new();
    engine.add_rule(InferenceRule {
        rule_type: RuleType::SubclassTransitivity,
        trigger_predicate: RDFS_SUBCLASS_OF.to_string(),
    });

    let derived = engine.infer(&[ab, bc]).unwrap();
    assert_eq!(derived.len(), 1);
    assert_eq!(derived[0].subject, Term::Iri("Scientist".to_string()));
    assert_eq!(derived[0].object, Term::Iri("Agent".to_string()));
}

#[test]
fn test_subclass_transitivity_longer_chain() {
    let ab = build_triple("Developer", RDFS_SUBCLASS_OF, "Programmer");
    let bc = build_triple("Programmer", RDFS_SUBCLASS_OF, "Person");
    let cd = build_triple("Person", RDFS_SUBCLASS_OF, "Agent");

    let mut engine = InferenceEngine::new();
    engine.add_rule(InferenceRule {
        rule_type: RuleType::SubclassTransitivity,
        trigger_predicate: RDFS_SUBCLASS_OF.to_string(),
    });

    let derived = engine.infer(&[ab, bc, cd]).unwrap();
    assert_eq!(derived.len(), 3);
}

#[test]
fn test_subclass_transitivity_no_new_edges() {
    let ab = build_triple("Cat", RDFS_SUBCLASS_OF, "Animal");

    let mut engine = InferenceEngine::new();
    engine.add_rule(InferenceRule {
        rule_type: RuleType::SubclassTransitivity,
        trigger_predicate: RDFS_SUBCLASS_OF.to_string(),
    });

    let derived = engine.infer(&[ab]).unwrap();
    assert_eq!(derived.len(), 0);
}

#[test]
fn test_capability_inference_direct() {
    let mut ci = CapabilityInference::new();
    ci.register_capability("Person", "has_birth_date");
    assert!(ci.duck_type("Person", "has_birth_date"));
    assert!(!ci.duck_type("Person", "has_altitude"));
}

#[test]
fn test_capability_inference_inherited() {
    let mut ci = CapabilityInference::new();
    let triple = build_triple("Scientist", RDFS_SUBCLASS_OF, "Person");
    ci.load_hierarchy(&[triple], RDFS_SUBCLASS_OF);
    ci.register_capability("Person", "has_birth_date");
    assert!(ci.duck_type("Scientist", "has_birth_date"));
}

#[test]
fn test_capability_inference_transitive() {
    let mut ci = CapabilityInference::new();
    let ab = build_triple("Developer", RDFS_SUBCLASS_OF, "Person");
    let bc = build_triple("Person", RDFS_SUBCLASS_OF, "Agent");
    ci.load_hierarchy(&[ab, bc], RDFS_SUBCLASS_OF);
    ci.register_capability("Agent", "has_id");
    assert!(ci.duck_type("Developer", "has_id"));
}

#[test]
fn test_capability_inference_cache_invalidation() {
    let mut ci = CapabilityInference::new();
    let triple = build_triple("Cat", RDFS_SUBCLASS_OF, "Animal");
    ci.load_hierarchy(&[triple], RDFS_SUBCLASS_OF);
    assert!(!ci.duck_type("Cat", "can_purr"));
    ci.register_capability("Animal", "can_breathe");
    assert!(ci.duck_type("Cat", "can_breathe"));
}

#[test]
fn test_capability_inference_cycle_safe() {
    let mut ci = CapabilityInference::new();
    let ab = build_triple("CycleA", RDFS_SUBCLASS_OF, "CycleB");
    let ba = build_triple("CycleB", RDFS_SUBCLASS_OF, "CycleA");
    ci.load_hierarchy(&[ab, ba], RDFS_SUBCLASS_OF);
    ci.register_capability("CycleA", "cycle_cap");
    assert!(ci.duck_type("CycleB", "cycle_cap"));
    let caps = ci.infer_capabilities("CycleA");
    assert!(caps.contains("cycle_cap"));
}

#[test]
fn test_capability_inference_traverses_chain() {
    let mut ci = CapabilityInference::new();
    let ep = build_triple("Engineer", RDFS_SUBCLASS_OF, "Person");
    let pa = build_triple("Person", RDFS_SUBCLASS_OF, "Agent");
    ci.load_hierarchy(&[ep, pa], RDFS_SUBCLASS_OF);
    ci.register_capability("Agent", "has_id");
    ci.register_capability("Person", "has_name");
    let caps = ci.infer_capabilities("Engineer");
    assert!(caps.contains("has_id"));
    assert!(caps.contains("has_name"));
    assert!(!caps.contains("has_altitude"));
}
