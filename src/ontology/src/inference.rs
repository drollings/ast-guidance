use std::collections::{HashMap, HashSet};

use guidance_rdf::parser::Term;
use guidance_rdf::parser::Triple;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuleType {
    SubclassTransitivity,
    SubpropertyTransitivity,
    DomainRange,
    InverseOf,
}

#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub rule_type: RuleType,
    pub trigger_predicate: String,
}

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("no triples provided")]
    EmptyInput,
}

pub struct InferenceEngine {
    rules: Vec<InferenceRule>,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    pub fn infer(&self, triples: &[Triple]) -> Result<Vec<Triple>, InferenceError> {
        let mut derived: Vec<Triple> = Vec::new();

        for rule in &self.rules {
            if rule.rule_type == RuleType::SubclassTransitivity {
                infer_subclass_transitivity(triples, &mut derived, &rule.trigger_predicate);
            }
        }

        Ok(derived)
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn infer_subclass_transitivity(base: &[Triple], derived: &mut Vec<Triple>, predicate_iri: &str) {
    let mut known: HashSet<(String, String)> = HashSet::new();

    for t in base {
        if is_subclass_triple(t, predicate_iri) {
            if let (Some(s), Some(o)) = (triple_subject_iri(t), triple_object_iri(t)) {
                known.insert((s, o));
            }
        }
    }
    for t in derived.iter() {
        if is_subclass_triple(t, predicate_iri) {
            if let (Some(s), Some(o)) = (triple_subject_iri(t), triple_object_iri(t)) {
                known.insert((s, o));
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        let edges: Vec<(String, String)> = known.iter().cloned().collect();

        for (sub_a, obj_a) in &edges {
            for (sub_b, obj_b) in &edges {
                if obj_a != sub_b {
                    continue;
                }
                let new_edge = (sub_a.clone(), obj_b.clone());
                if known.contains(&new_edge) {
                    continue;
                }
                let triple = build_subclass_triple(&new_edge.0, predicate_iri, &new_edge.1);
                derived.push(triple);
                known.insert(new_edge);
                changed = true;
            }
        }
    }
}

fn is_subclass_triple(t: &Triple, predicate_iri: &str) -> bool {
    matches!(&t.predicate, Term::Iri(s) if s == predicate_iri)
}

fn triple_subject_iri(t: &Triple) -> Option<String> {
    match &t.subject {
        Term::Iri(s) => Some(s.clone()),
        _ => None,
    }
}

fn triple_object_iri(t: &Triple) -> Option<String> {
    match &t.object {
        Term::Iri(s) => Some(s.clone()),
        _ => None,
    }
}

fn build_subclass_triple(subject_iri: &str, predicate_iri: &str, object_iri: &str) -> Triple {
    Triple {
        subject: Term::Iri(subject_iri.to_string()),
        predicate: Term::Iri(predicate_iri.to_string()),
        object: Term::Iri(object_iri.to_string()),
    }
}

pub struct CapabilityInference {
    hierarchy: HashMap<String, Vec<String>>,
    direct_capabilities: HashMap<String, HashSet<String>>,
    inferred_cache: HashMap<String, HashSet<String>>,
}

impl CapabilityInference {
    pub fn new() -> Self {
        Self {
            hierarchy: HashMap::new(),
            direct_capabilities: HashMap::new(),
            inferred_cache: HashMap::new(),
        }
    }

    pub fn load_hierarchy(&mut self, triples: &[Triple], predicate_iri: &str) {
        for t in triples {
            if !is_subclass_triple(t, predicate_iri) {
                continue;
            }
            let Some(child) = triple_subject_iri(t) else {
                continue;
            };
            let Some(parent) = triple_object_iri(t) else {
                continue;
            };
            self.hierarchy.entry(child).or_default().push(parent);
        }
        self.inferred_cache.clear();
    }

    pub fn add_subclass_edge(&mut self, child_iri: &str, parent_iri: &str) {
        self.hierarchy
            .entry(child_iri.to_string())
            .or_default()
            .push(parent_iri.to_string());
        self.inferred_cache.clear();
    }

    pub fn register_capability(&mut self, class_iri: &str, capability_name: &str) {
        self.direct_capabilities
            .entry(class_iri.to_string())
            .or_default()
            .insert(capability_name.to_string());
        self.inferred_cache.clear();
    }

    pub fn invalidate(&mut self, _class_iri: &str) {
        self.inferred_cache.clear();
    }

    pub fn infer_capabilities(&mut self, class_iri: &str) -> &HashSet<String> {
        if self.inferred_cache.contains_key(class_iri) {
            return &self.inferred_cache[class_iri];
        }

        let mut merged = HashSet::new();

        if let Some(direct) = self.direct_capabilities.get(class_iri) {
            merged.extend(direct.iter().cloned());
        }

        let mut visited = HashSet::new();
        self.collect_ancestor_caps(class_iri, &mut merged, &mut visited);

        self.inferred_cache.insert(class_iri.to_string(), merged);
        &self.inferred_cache[class_iri]
    }

    fn collect_ancestor_caps(
        &self,
        class_iri: &str,
        out: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(class_iri.to_string()) {
            return;
        }

        let parents = match self.hierarchy.get(class_iri) {
            Some(p) => p.clone(),
            None => return,
        };

        for parent in &parents {
            if let Some(direct) = self.direct_capabilities.get(parent) {
                out.extend(direct.iter().cloned());
            }
            self.collect_ancestor_caps(parent, out, visited);
        }
    }

    pub fn duck_type(&mut self, class_iri: &str, capability_name: &str) -> bool {
        let caps = self.infer_capabilities(class_iri);
        caps.contains(capability_name)
    }
}

impl Default for CapabilityInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../tests/inference.rs"]
mod tests;
