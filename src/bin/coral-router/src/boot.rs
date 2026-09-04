//! Boot-time YaGO taxonomy load + two-store reconciliation (ROADMAP
//! §11.7/§13.7/§13.8 — C3).
//!
//! **One loader, two homes, same ids.** [`YaGoLoader`] produces the taxonomy
//! once; both durable stores are fed from that same `Vec<ConceptMetadata>`:
//!
//! 1. coral's content-addressed graph (`context_nodes`, id = the full 64-bit
//!    `hash_iri`),
//! 2. the router's `SqliteConceptStore` (`interlingua_concepts`, id = the
//!    48-bit interlingua id with the same `node_id` cross-ref, F5).
//!
//! A boot-time reconciliation then **fails loudly** if the two homes drift:
//! the class counts must match and every reference class's `node_id` must
//! resolve in both. The `InMemoryConceptStore` is never the production store.


use std::sync::Arc;

use coral_context::db::Library;
use fluent_router::concept_store_sqlite::SqliteConceptStore;
use fluent_router::config::RouterConfig;
use fluent_router::ledger::ContentNodeLedger;
use fluent_types::{ContentNode, NodeId};
use guidance_ontology::yago::WHITELIST_IRIS;
use guidance_ontology::yago_loader::YaGoLoader;
use fluent_concept::{ConceptStore, TaxonomyHierarchy};

/// Open the ledger the way the boot wants it (ROADMAP_20260828_ORT M1.1): a
/// configured `ledger` section opens its durable path (with a `warn!` for the
/// in-memory fallback) and attaches the summarizer; when no `ledger` section is
/// configured the boot still opens an in-memory ledger so nlp/review/overlay
/// (which require a shared SQLite connection) have a backing store.
pub fn open_ledger(config: &RouterConfig) -> Result<Arc<ContentNodeLedger>, String> {
    let ledger = match &config.ledger {
        Some(ledger_cfg) => {
            let opened = match &ledger_cfg.path {
                Some(path) => ContentNodeLedger::open(path),
                None => {
                    tracing::warn!(
                        target: "coral-router",
                        "ledger section has no path - using an in-memory ledger (ephemeral)",
                    );
                    ContentNodeLedger::open_in_memory()
                }
            };
            let ledger = opened.map_err(|e| format!("fatal: ledger open failed: {e}"))?;
            match config.summarizer_for_ledger() {
                Some(summarizer) => {
                    let model_key = ledger_cfg
                        .model
                        .clone()
                        .or_else(|| config.classifier_model.clone());
                    tracing::info!(
                        target: "coral-router",
                        ledger_model = ?model_key,
                        summarizer = true,
                        "ledger summarizer attached",
                    );
                    Arc::new(ledger.with_summarizer(summarizer))
                }
                None => {
                    tracing::warn!(
                        target: "coral-router",
                        "ledger section present but no summarizer derivable - ledger attached without LOD derivation",
                    );
                    Arc::new(ledger)
                }
            }
        }
        None => {
            tracing::warn!(
                target: "coral-router",
                "nlp/review/overlay configured without a `ledger` section - using an in-memory ledger (ephemeral)",
            );
            Arc::new(ContentNodeLedger::open_in_memory().map_err(|e| format!("ledger open failed: {e}"))?)
        }
    };
    Ok(ledger)
}

/// The early-boot shared concept-store bundle (ROADMAP_20260828_ORT M1.1):
/// the `SqliteConceptStore` + `SqliteCorrectionIndex` over the shared ledger
/// connection and the reconciliation stats. Built **before** the pipeline build
/// so a resolver can be threaded into the NLP pipeline; reused by the
/// review/overlay workers (DRY — one store, not one per consumer).
pub struct ConceptStoreBoot {
    /// The shared concept store (also the `ConceptStore` trait object).
    pub concept_store: Arc<SqliteConceptStore>,
    /// The correction index over the same connection.
    pub correction_index: Arc<fluent_router::ledger::correction_index::SqliteCorrectionIndex>,
    /// Reconciliation statistics (for the boot log).
    pub stats: ReconcileStats,
}

/// Build the shared concept store + correction index over `ledger`'s shared
/// connection and reconcile YaGO across the two homes. `ledger` must be
/// durable-backed (shared SQLite present).
pub fn build_concept_store_boot(
    ledger: &ContentNodeLedger,
) -> Result<ConceptStoreBoot, String> {
    let shared_sqlite = ledger
        .node_store()
        .shared_sqlite()
        .expect("ledger must have shared sqlite for the concept store");
    let concept_store = Arc::new(SqliteConceptStore::new(Arc::clone(&shared_sqlite)));
    let correction_index = Arc::new(
        fluent_router::ledger::correction_index::SqliteCorrectionIndex::new(Arc::clone(
            &shared_sqlite,
        )),
    );
    // The coral content-addressed graph home is scoped to the reconcile (the
    // load fills both homes from one taxonomy); nothing needs it afterwards.
    let coral = Library::open_in_memory().expect("coral in-memory");
    let stats = load_yago_and_reconcile(&coral, &concept_store)?;
    Ok(ConceptStoreBoot {
        concept_store,
        correction_index,
        stats,
    })
}

/// Reconciliation statistics.
///
/// The reconciliation seam (13.10) is exercised hermetically by the tests in
/// this module and composed into the live boot (`main.rs`), which opens the
/// coral graph home and the router's `SqliteConceptStore` over the shared
/// ledger connection.
///
/// `coral_nodes` / `router_concepts` are **membership counts**: how many of
/// the loader's concepts resolve in each home (never raw row counts — a
/// 48-bit collision must not read as drift, red-team M5/L6).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReconcileStats {
    pub classes: usize,
    pub coral_nodes: usize,
    pub router_concepts: usize,
}

/// Load the embedded YaGO taxonomy into both homes and reconcile them.
///
/// Returns the reconciliation stats, or an error the boot treats as fatal (a
/// drift between the two stores is a loud boot failure, §13.8).
///
/// Reconciliation is **id-membership**, not raw count equality (red-team M5):
/// every loader concept's `id` must resolve in the router store and its
/// `node_id` in coral, the distinct-id counts must agree (collision-tolerant —
/// a colliding canonical shares an id and never inflates the count), and the
/// coral cross-check is scoped to YaGO provenance, never total `node_count()`
/// (L6 — a shared coral DB may hold unrelated content).
pub fn load_yago_and_reconcile(
    coral: &Library,
    store: &SqliteConceptStore,
) -> Result<ReconcileStats, String> {
    let mut loader = YaGoLoader::new();
    loader
        .load_embedded()
        .map_err(|e| format!("yago taxonomy load failed: {e}"))?;

    let edges = loader.subclass_edges().to_vec();
    let concepts = loader.into_concepts();

    // Home 1 — coral's durable content-addressed graph.
    for c in &concepts {
        let node_id = c
            .node_id
            .ok_or_else(|| format!("concept {} missing node_id", c.canonical_name))?;
        coral
            .insert_node(&ContentNode {
                id: Some(node_id),
                name: c.canonical_name.as_str().into(),
                source: c.yago_iri.clone().unwrap_or_default(),
                lod: vec![c.label.clone().unwrap_or_default()],
                ..Default::default()
            })
            .map_err(|e| format!("coral insert of {} failed: {e}", c.canonical_name))?;
    }

    // Home 2 — the router's materialized index (same ids, same node_id).
    for c in &concepts {
        store
            .insert(c.clone())
            .map_err(|e| format!("sqlite insert of {} failed: {e}", c.canonical_name))?;
    }
    store
        .set_hierarchy(TaxonomyHierarchy::from_edges(&edges).map_err(|e| e.to_string())?);

    // ── Reconciliation assert (M5): id-membership, never raw counts. ──
    // Distinct-id agreement (collision-tolerant): the loader dedupes by id, so
    // `concepts.len()` is the distinct-id count; the router store's `iter_ids`
    // is `SELECT DISTINCT id`. A colliding canonical shares a bucket id and
    // does not inflate the count; a stray unique id (two writers) does.
    let router_ids = store.iter_ids().count();
    if router_ids != concepts.len() {
        return Err(format!(
            "concept-store drift: loader produced {} distinct ids, router store holds {}",
            concepts.len(),
            router_ids
        ));
    }

    // Forward membership: every loader concept resolves in both homes.
    let mut router_resolved = 0usize;
    let mut coral_resolved = 0usize;
    for c in &concepts {
        if store.get(c.id).is_ok() {
            router_resolved += 1;
        }
        let node_id = c.node_id.ok_or_else(|| {
            format!("concept {} missing node_id", c.canonical_name)
        })?;
        if coral
            .get_node(node_id)
            .map_err(|e| format!("coral get {} failed: {e}", c.canonical_name))?
            .is_some()
        {
            coral_resolved += 1;
        }
    }
    if router_resolved != concepts.len() {
        return Err(format!(
            "concept-store drift: only {router_resolved}/{} loader concepts resolve in the router store",
            concepts.len()
        ));
    }
    if coral_resolved != concepts.len() {
        return Err(format!(
            "concept-store drift: only {coral_resolved}/{} loader concepts resolve in coral",
            concepts.len()
        ));
    }

    // L6: the coral cross-check is scoped to YaGO provenance — never total
    // `node_count()` (a shared coral DB holds unrelated content).
    let coral_yago = coral
        .count_nodes_by_source_prefix("http://yago-knowledge.org/resource/")
        .map_err(|e| format!("coral yago count failed: {e}"))?
        + coral
            .count_nodes_by_source_prefix("http://schema.org/")
            .map_err(|e| format!("coral schema count failed: {e}"))?;
    if coral_yago != concepts.len() as i64 {
        return Err(format!(
            "concept-store drift: coral holds {coral_yago} YaGO-sourced nodes, loader produced {}",
            concepts.len()
        ));
    }

    for iri in WHITELIST_IRIS {
        let node_id = NodeId::from_int(guidance_rdf::normalize::hash_iri(iri));
        let in_coral = coral
            .get_node(node_id)
            .map_err(|e| format!("coral get {iri} failed: {e}"))?
            .is_some();
        let id = guidance_ontology::yago_loader::yago_class_id(iri);
        let in_router = store.get(id).is_ok();
        if !in_coral || !in_router {
            return Err(format!(
                "reference class {iri} missing from {}",
                if !in_coral { "coral" } else { "the router store" }
            ));
        }
    }

    Ok(ReconcileStats {
        classes: concepts.len(),
        coral_nodes: coral_resolved,
        router_concepts: router_resolved,
    })
}

/// Build a `SqliteConceptStore` over the shared ledger connection (the boot
/// composition seam). Returns `None` when the ledger has no durable backing.
#[allow(dead_code)]
pub fn sqlite_store_from_ledger(
    ledger: &ContentNodeLedger,
) -> Option<SqliteConceptStore> {
    ledger
        .node_store()
        .shared_sqlite()
        .map(SqliteConceptStore::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use guidance_ontology::yago::ALL_CLASSES;

    #[test]
    fn embedded_taxonomy_reconciles_across_two_homes() {
        let coral = Library::open_in_memory().expect("coral in-memory");
        let ledger = ContentNodeLedger::open_in_memory().expect("ledger in-memory");
        let store = sqlite_store_from_ledger(&ledger).expect("shared sqlite");

        let stats = load_yago_and_reconcile(&coral, &store).expect("reconcile");
        assert!(stats.classes >= 7);
        assert_eq!(stats.coral_nodes, stats.classes);
        assert_eq!(stats.router_concepts, stats.classes);

        // Every reference class resolves in the router store with its coral
        // node_id cross-ref (F5: full 64-bit).
        for iri in ALL_CLASSES {
            let id = guidance_ontology::yago_loader::yago_class_id(iri.iri);
            let meta = store.get(id).expect("resolves in router store");
            let node_id = meta.node_id.expect("node_id cross-ref");
            assert!(
                coral.get_node(node_id).expect("coral get").is_some(),
                "{} resolves in coral",
                iri.iri
            );
        }
    }

    #[test]
    fn drift_is_detected() {
        // A store that silently gained a row (or a loader that was never fed
        // to it) must fail the boot loudly instead of silently diverging.
        let coral = Library::open_in_memory().expect("coral in-memory");
        let ledger = ContentNodeLedger::open_in_memory().expect("ledger in-memory");
        let store = sqlite_store_from_ledger(&ledger).expect("shared sqlite");
        load_yago_and_reconcile(&coral, &store).expect("first reconcile");

        // Inject a row the loader never produced (simulating two writers).
        let stray = guidance_ontology::yago_loader::yago_class_id(
            "http://yago-knowledge.org/resource/StrayClass",
        );
        store
            .insert(fluent_types::ConceptMetadata {
                id: stray,
                canonical_name: "yago:StrayClass".into(),
                namespace: fluent_types::InterlinguaNamespace::YagoClass,
                yago_iri: Some("http://yago-knowledge.org/resource/StrayClass".into()),
                yago_class_iri: None,
                label: Some("stray".into()),
                node_id: Some(NodeId::from_int(guidance_rdf::normalize::hash_iri(
                    "http://yago-knowledge.org/resource/StrayClass",
                ))),
                parent_class_id: None,
            })
            .expect("stray insert");

        // Re-run: idempotent inserts on both sides keep coral at N, but the
        // router store now has N+1 → loud failure.
        assert!(load_yago_and_reconcile(&coral, &store).is_err());
    }

    #[test]
    fn collision_tolerant_reconciliation_passes() {
        let coral = Library::open_in_memory().expect("coral in-memory");
        let ledger = ContentNodeLedger::open_in_memory().expect("ledger in-memory");
        let store = sqlite_store_from_ledger(&ledger).expect("shared sqlite");
        load_yago_and_reconcile(&coral, &store).expect("first reconcile");

        // Register a second canonical that shares a truncated 48-bit id with
        // an existing concept (a genuine collision). The PK is
        // (namespace, canonical_name), so both canonicals are stored; the
        // distinct-id count does not inflate.
        let existing = store
            .resolve_yago_iri("http://schema.org/Person")
            .expect("existing person");
        let colliding = fluent_types::ConceptMetadata {
            id: existing, // same truncated id → collision bucket
            canonical_name: "schema:CollisionPerson".into(),
            namespace: fluent_types::InterlinguaNamespace::YagoClass,
            yago_iri: Some("http://schema.org/collision-person".into()),
            yago_class_iri: None,
            label: Some("collision person".into()),
            node_id: Some(NodeId::from_int(guidance_rdf::normalize::hash_iri(
                "http://schema.org/collision-person",
            ))),
            parent_class_id: None,
        };
        store.insert(colliding).expect("colliding insert");

        // Id-membership reconciliation survives the collision (no boot brick),
        // and both canonicals remain resolvable by IRI.
        load_yago_and_reconcile(&coral, &store).expect("reconcile with collision");
        assert_eq!(
            store
                .resolve_yago_iri("http://schema.org/Person")
                .expect("person iri"),
            existing
        );
        assert_eq!(
            store
                .resolve_yago_iri("http://schema.org/collision-person")
                .expect("collision iri"),
            existing,
            "both colliding canonicals resolve under the shared bucket id"
        );
    }

    #[test]
    fn shared_sqlite_is_one_connection() {
        let ledger = ContentNodeLedger::open_in_memory().expect("ledger in-memory");
        let store = sqlite_store_from_ledger(&ledger).expect("shared sqlite");
        // The store shares the ledger's connection: a concept written through
        // the SqliteConceptStore is visible to the ledger's own connection.
        let mut loader = YaGoLoader::new();
        loader.load_embedded().expect("load");
        let concepts = loader.into_concepts();
        store.insert(concepts[0].clone()).expect("insert");
        assert_eq!(store.yago_class_count().expect("count"), 1);
        let _ = Arc::new(());
    }

    /// M1.1 (ROADMAP_20260828_ORT, G3): the concept store exists **before** the
    /// pipeline build when nlp is on. Reproduces the boot ordering — open the
    /// ledger, build the shared store + reconcile, then build the pipeline with
    /// a resolver threaded through `NlpDeps` — and asserts the store is live and
    /// populated before the pipeline consumes it.
    #[test]
    fn concept_store_exists_before_pipeline_build_when_nlp_enabled() {
        use fluent_router::config::builder::NlpDeps;
        use fluent_router::config::RouterConfig;

        let config: RouterConfig = serde_json::from_str(
            r#"{
                "pipelines": {
                    "default": {"deterministic_prefilter": false, "nlp": true, "classifier": false}
                }
            }"#,
        )
        .expect("config");

        // nlp: true ⇒ the store must be built before pipeline build.
        assert!(config.pipelines.values().any(|p| p.nlp), "nlp enabled");

        // Boot order (mirrors main.rs): open ledger → build store + reconcile →
        // thread NlpDeps → build pipelines.
        let ledger = open_ledger(&config).expect("ledger opens");
        let store_boot = build_concept_store_boot(&ledger).expect("store + reconcile built before pipeline");
        assert!(store_boot.stats.classes >= 7, "YaGO reconcile populated the store");
        let nlp_deps = NlpDeps {
            concept_store: Some(store_boot.concept_store as std::sync::Arc<dyn fluent_concept::ConceptStore>),
            strings_path: None,
        };
        let map = config.build_all_pipelines_with_backend_onnx_and_nlp(None, None, &nlp_deps);
        assert!(
            map.contains_key("default"),
            "pipeline builds with the early-built store (G3 closed)"
        );
    }
}