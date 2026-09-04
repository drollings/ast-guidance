pub const NS_YAGO: &str = "http://yago-knowledge.org/resource/";
pub const NS_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const NS_RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
pub const NS_OWL: &str = "http://www.w3.org/2002/07/owl#";
pub const NS_XSD: &str = "http://www.w3.org/2001/XMLSchema#";
pub const NS_SCHEMA: &str = "http://schema.org/";
pub const NS_SKOS: &str = "http://www.w3.org/2004/02/skos/core#";

use fluent_types::{local_id_of, property_id_for_iri, InterlinguaId};
use guidance_rdf::normalize::hash_iri;

pub const YAGO_VERSION: &str = "4.5";

#[derive(Debug, Clone, Copy)]
pub struct OntologyClass {
    pub iri: &'static str,
    pub label: &'static str,
    pub superclass: Option<&'static str>,
    pub properties: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropertyRange {
    Iri,
    String,
    LangString,
    Integer,
    Decimal,
    Boolean,
    DateTime,
    Any,
}

#[derive(Debug, Clone, Copy)]
pub struct OntologyProperty {
    pub iri: &'static str,
    pub label: &'static str,
    pub domain: Option<&'static str>,
    pub range: PropertyRange,
    pub transitive: bool,
    pub symmetric: bool,
    pub lod_target: Option<usize>,
}

const YAGO_ENTITY_PROPS: &[&str] = &[
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://schema.org/description",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
];

const YAGO_PERSON_PROPS: &[&str] = &[
    "http://yago-knowledge.org/resource/hasGender",
    "http://yago-knowledge.org/resource/hasNationality",
    "http://yago-knowledge.org/resource/bornIn",
    "http://yago-knowledge.org/resource/diedIn",
    "http://yago-knowledge.org/resource/hasWikipediaArticle",
];

const YAGO_ORG_PROPS: &[&str] = &["http://yago-knowledge.org/resource/hasWikipediaArticle"];

pub const CLASS_ENTITY: OntologyClass = OntologyClass {
    iri: "http://yago-knowledge.org/resource/Entity",
    label: "Entity",
    superclass: None,
    properties: YAGO_ENTITY_PROPS,
};

pub const CLASS_PERSON: OntologyClass = OntologyClass {
    iri: "http://schema.org/Person",
    label: "Person",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: YAGO_PERSON_PROPS,
};

pub const CLASS_ORGANIZATION: OntologyClass = OntologyClass {
    iri: "http://schema.org/Organization",
    label: "Organization",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: YAGO_ORG_PROPS,
};

pub const CLASS_LOCATION: OntologyClass = OntologyClass {
    iri: "http://schema.org/Place",
    label: "Location",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: &[],
};

pub const CLASS_EVENT: OntologyClass = OntologyClass {
    iri: "http://schema.org/Event",
    label: "Event",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: &[],
};

pub const CLASS_ARTIFACT: OntologyClass = OntologyClass {
    iri: "http://yago-knowledge.org/resource/Artifact",
    label: "Artifact",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: &[],
};

pub const CLASS_CONCEPT: OntologyClass = OntologyClass {
    iri: "http://yago-knowledge.org/resource/Concept",
    label: "Concept",
    superclass: Some("http://yago-knowledge.org/resource/Entity"),
    properties: &[],
};

pub const ALL_CLASSES: &[&OntologyClass] = &[
    &CLASS_ENTITY,
    &CLASS_PERSON,
    &CLASS_ORGANIZATION,
    &CLASS_LOCATION,
    &CLASS_EVENT,
    &CLASS_ARTIFACT,
    &CLASS_CONCEPT,
];

pub fn lookup_class(iri: &str) -> Option<&'static OntologyClass> {
    ALL_CLASSES.iter().copied().find(|cls| cls.iri == iri)
}

pub fn superclass_chain(iri: &str) -> Vec<String> {
    let mut chain: Vec<String> = Vec::new();
    let mut current: Option<&str> = Some(iri);
    while let Some(cur) = current {
        chain.push(cur.to_string());
        match lookup_class(cur) {
            Some(cls) => current = cls.superclass,
            None => break,
        }
    }
    chain
}

pub fn is_subclass_of(child_iri: &str, parent_iri: &str) -> bool {
    if child_iri == parent_iri {
        return true;
    }
    let mut current: Option<&str> = Some(child_iri);
    while let Some(cur) = current {
        match lookup_class(cur) {
            Some(cls) => match cls.superclass {
                Some(superclass) if superclass == parent_iri => return true,
                Some(superclass) => current = Some(superclass),
                None => break,
            },
            None => break,
        }
    }
    false
}

pub const WHITELIST_IRIS: &[&str] = &[
    "http://yago-knowledge.org/resource/Entity",
    "http://schema.org/Person",
    "http://schema.org/Organization",
    "http://schema.org/Place",
    "http://schema.org/Event",
    "http://yago-knowledge.org/resource/Artifact",
    "http://yago-knowledge.org/resource/Concept",
];

pub fn is_whitelisted(iri: &str) -> bool {
    WHITELIST_IRIS.contains(&iri)
}

/// Whether an [`InterlinguaId`] names a whitelisted YaGO class (ROADMAP
/// §13.6 — namespace-aware, truncation-correct). Compares the **48-bit local**
/// against the truncated whitelist hashes.
pub fn is_whitelisted_id(id: InterlinguaId) -> bool {
    id.is_yago()
        && WHITELIST_IRIS
            .iter()
            .any(|iri| id.local_id() == local_id_of(hash_iri(iri)))
}

/// Truncation-aware hash whitelist check. Legacy callers pass a full-width
/// `hash_iri` value; comparing full hashes silently misses a *truncated*
/// interlingua id, so both sides are reduced to the 48-bit local (DRY — the
/// same truncation `is_whitelisted_id` uses).
pub fn is_whitelisted_hash(hash: i64) -> bool {
    let local = local_id_of(hash);
    WHITELIST_IRIS.iter().any(|iri| local == local_id_of(hash_iri(iri)))
}

/// The deterministic `RDF_PROPERTY` interlingua id for an ontology property
/// (ROADMAP §13.4). Lets the ledger reference typed relations by id.
pub fn property_interlingua_id(prop: &OntologyProperty) -> InterlinguaId {
    property_id_for_iri(prop.iri)
}

/// The `RDF_PROPERTY` id for a property IRI, when it is one of the known
/// [`OntologyProperty`] constants.
pub fn property_interlingua_id_by_iri(iri: &str) -> Option<InterlinguaId> {
    lookup_property(iri).map(property_interlingua_id)
}

/// The `RDF_PROPERTY` id for the `rdfs:subClassOf` relation.
pub fn subclass_property_id() -> InterlinguaId {
    property_interlingua_id(&PROP_SUBCLASS)
}

pub const PROP_LABEL: OntologyProperty = OntologyProperty {
    iri: "http://www.w3.org/2000/01/rdf-schema#label",
    label: "label",
    domain: None,
    range: PropertyRange::LangString,
    transitive: false,
    symmetric: false,
    lod_target: Some(4),
};

pub const PROP_COMMENT: OntologyProperty = OntologyProperty {
    iri: "http://www.w3.org/2000/01/rdf-schema#comment",
    label: "comment",
    domain: None,
    range: PropertyRange::LangString,
    transitive: false,
    symmetric: false,
    lod_target: Some(0),
};

pub const PROP_DESCRIPTION: OntologyProperty = OntologyProperty {
    iri: "http://schema.org/description",
    label: "description",
    domain: None,
    range: PropertyRange::LangString,
    transitive: false,
    symmetric: false,
    lod_target: Some(1),
};

pub const PROP_PREF_LABEL: OntologyProperty = OntologyProperty {
    iri: "http://www.w3.org/2004/02/skos/core#prefLabel",
    label: "prefLabel",
    domain: None,
    range: PropertyRange::LangString,
    transitive: false,
    symmetric: false,
    lod_target: Some(4),
};

pub const PROP_TYPE: OntologyProperty = OntologyProperty {
    iri: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    label: "type",
    domain: None,
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const PROP_SUBCLASS: OntologyProperty = OntologyProperty {
    iri: "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    label: "subClassOf",
    domain: None,
    range: PropertyRange::Iri,
    transitive: true,
    symmetric: false,
    lod_target: None,
};

pub const PROP_HAS_GENDER: OntologyProperty = OntologyProperty {
    iri: "http://yago-knowledge.org/resource/hasGender",
    label: "hasGender",
    domain: Some("http://schema.org/Person"),
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const PROP_HAS_NATIONALITY: OntologyProperty = OntologyProperty {
    iri: "http://yago-knowledge.org/resource/hasNationality",
    label: "hasNationality",
    domain: Some("http://schema.org/Person"),
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const PROP_BORN_IN: OntologyProperty = OntologyProperty {
    iri: "http://yago-knowledge.org/resource/bornIn",
    label: "bornIn",
    domain: Some("http://schema.org/Person"),
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const PROP_DIED_IN: OntologyProperty = OntologyProperty {
    iri: "http://yago-knowledge.org/resource/diedIn",
    label: "diedIn",
    domain: Some("http://schema.org/Person"),
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const PROP_WIKIPEDIA: OntologyProperty = OntologyProperty {
    iri: "http://yago-knowledge.org/resource/hasWikipediaArticle",
    label: "hasWikipediaArticle",
    domain: None,
    range: PropertyRange::Iri,
    transitive: false,
    symmetric: false,
    lod_target: None,
};

pub const ALL_PROPERTIES: &[&OntologyProperty] = &[
    &PROP_LABEL,
    &PROP_COMMENT,
    &PROP_DESCRIPTION,
    &PROP_PREF_LABEL,
    &PROP_TYPE,
    &PROP_SUBCLASS,
    &PROP_HAS_GENDER,
    &PROP_HAS_NATIONALITY,
    &PROP_BORN_IN,
    &PROP_DIED_IN,
    &PROP_WIKIPEDIA,
];

pub fn lookup_property(iri: &str) -> Option<&'static OntologyProperty> {
    ALL_PROPERTIES.iter().copied().find(|prop| prop.iri == iri)
}

#[cfg(test)]
#[path = "../tests/yago.rs"]
mod tests;
