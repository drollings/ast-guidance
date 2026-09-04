use super::*;
use crate::config::ModelEntry;

fn plain_entry() -> ModelEntry {
    ModelEntry {
        name: None,
        endpoint: "http://127.0.0.1:1/v1/chat/completions".into(),
        intelligence: 1,
        cost_input: 0.0,
        cost_output: 0.0,
        cost_cached_read: 0.0,
        speed: 1,
        total_timeout_ms: 0,
        idle_timeout_ms: 0,
        stream: true,
        filter_thinking: false,
        retry_count: 0,
        retry_base_interval_s: 0,
        params: None,
        instances: None,
        weights: None,
        hf_repo: None,
        hf_file: None,
            api_key: None,
    }
}

fn role_config(model_path: &str) -> fluent_onnx::OnnxRoleConfig {
    fluent_onnx::OnnxRoleConfig {
        pinned: false,
        no_sleep: false,
        sleep_idle_seconds: None,
        total_timeout_ms: 0,
        idle_timeout_ms: 0,
        params: None,
        instances: None,
        model: fluent_onnx::OnnxConfig::new()
            .model_path(model_path)
            .tokenizer_path("/models/tokenizer.json")
            // Unloadable so registration does not attempt a real load.
            .resident(false)
            .build(),
    }
}

#[test]
fn is_managed_covers_only_llama_declarations() {
    assert!(!plain_entry().is_managed());
    let mut weights = plain_entry();
    weights.weights = Some("model.gguf".into());
    assert!(weights.is_managed());
    let mut hf = plain_entry();
    hf.hf_repo = Some("author/model".into());
    assert!(hf.is_managed());
    let mut inst = plain_entry();
    inst.instances = Some(Default::default());
    assert!(inst.is_managed());
}

#[test]
fn no_onnx_config_yields_empty_registry() {
    let config = RouterConfig::default();
    #[cfg(feature = "onnx")]
    {
        let registry = build_onnx_registry(&config).expect("build");
        assert!(registry.is_none());
    }
    #[cfg(not(feature = "onnx"))]
    {
        let registry = build_onnx_registry(&config).expect("build");
        assert!(registry.is_none());
    }
}

#[cfg(feature = "onnx")]
#[test]
fn role_fleet_registers_under_role_keys_with_implied_tasks() {
let mut config = RouterConfig::default();
config.onnx = Some(fluent_onnx::OnnxFleetConfig {
    encoder: Some(role_config("/models/encoder.onnx")),
    pii: None,
    router: None,
    policy: None,
    colbert: Some(role_config("/models/colbert.onnx")),
    llm: None,
});
let registry = build_onnx_registry(&config).expect("build").expect("fleet");
let encoder_key = fluent_onnx::OnnxRole::Encoder.registry_key();
let colbert_key = fluent_onnx::OnnxRole::Colbert.registry_key();
assert!(registry.config(encoder_key).is_some());
assert!(registry.config(colbert_key).is_some());
assert_eq!(
    registry.config(encoder_key).unwrap().task,
    fluent_onnx::OnnxTask::FillMask
);
assert_eq!(
    registry.config(colbert_key).unwrap().task,
    fluent_onnx::OnnxTask::LateInteraction
);
// Unconfigured roles are not registered.
assert!(!registry.config(fluent_onnx::OnnxRole::Pii.registry_key()).is_some());
}

#[cfg(feature = "onnx")]
#[test]
fn llm_role_registers_as_causal_lm_with_lifecycle() {
let mut config = RouterConfig::default();
config.onnx = Some(fluent_onnx::OnnxFleetConfig {
    llm: Some(fluent_onnx::OnnxRoleConfig {
        pinned: true,
        no_sleep: false,
        sleep_idle_seconds: Some(30),
        total_timeout_ms: 120000,
        idle_timeout_ms: 0,
        params: None,
        instances: None,
        model: fluent_onnx::OnnxConfig::new()
            .model_path("/models/llm.onnx")
            .tokenizer_path("/models/llm/tokenizer.json")
            // Unloadable so registration does not attempt a real load.
            .resident(false)
            .task(fluent_onnx::OnnxTask::CausalLm)
            .max_gen_tokens(512)
            .build(),
    }),
    ..Default::default()
});
let registry = build_onnx_registry(&config).expect("build").expect("fleet");
let key = fluent_onnx::OnnxRole::Llm.registry_key();
let cfg = registry.config(key).expect("llm registered");
assert_eq!(cfg.task, fluent_onnx::OnnxTask::CausalLm);
assert_eq!(cfg.max_gen_tokens, 512);
assert!(registry.is_pinned(key), "role pinned flag reaches the registry");
assert_eq!(registry.sleep_idle_seconds(key), Some(30));
assert!(!registry.refuses_unload(key), "unloadable role is not Always");
}

#[cfg(feature = "onnx")]
#[test]
fn build_onnx_residency_yields_loop_only_when_fleet_registered() {
let mut config = RouterConfig::default();
// No fleet → no loop.
assert!(build_onnx_residency(&config, &Arc::new(fluent_onnx::OrtSessionRegistry::new(
    Arc::new(fluent_onnx::OrtSessionLoader),
)))
.is_none());

config.onnx = Some(fluent_onnx::OnnxFleetConfig {
    llm: Some(role_config("/models/llm.onnx")),
    ..Default::default()
});
let registry = build_onnx_registry(&config).expect("build").expect("fleet");
let loop_ = build_onnx_residency(&config, &registry).expect("loop built");
let _ = Arc::into_inner(loop_).expect("unique Arc for the test");
}

#[test]
fn fleet_round_trip_keeps_roles() {
    let mut config = RouterConfig::default();
    config.onnx = Some(fluent_onnx::OnnxFleetConfig {
        encoder: Some(role_config("/models/encoder.onnx")),
        ..Default::default()
    });
    let json = serde_json::to_string(&config).unwrap();
    let back: RouterConfig = serde_json::from_str(&json).unwrap();
    let fleet = back.onnx.expect("onnx fleet");
    assert!(fleet.has(fluent_onnx::OnnxRole::Encoder));
    assert!(!fleet.has(fluent_onnx::OnnxRole::Colbert));
}

#[test]
#[cfg(feature = "onnx")]
fn nlp_encoder_fetch_returns_none_for_missing_model() {
    let registry = Arc::new(fluent_onnx::OrtSessionRegistry::new(Arc::new(
        fluent_onnx::OrtSessionLoader,
    )));
    let result = nlp_encoder_fetch(&registry, "nonexistent").expect("no ort error");
    assert!(result.is_none(), "missing model yields no encoder rung");
}

/// A doc over the classic "The cat sat." sentence (4 tokens).
fn doc_for_cat_sat() -> spacy_rs::Doc {
    let vocab = std::sync::Arc::new(spacy_rs::vocab::Vocab::new(
        spacy_rs::lang::en::lexicon_config(),
    ));
    let mut doc = spacy_rs::Doc::new(vocab);
    for (text, sp) in [("The", true), ("cat", true), ("sat", true), (".", false)] {
        doc.push_back(text, sp).expect("push");
    }
    doc
}

#[cfg(feature = "onnx")]
mod map_annotations_tests {
    use super::*;
    use fluent_onnx::TokenAnnotation;
    use spacy_rs::pipeline::AnnotateError;
    use spacy_rs::AnnotationValidator;

    fn valid_annotations() -> Vec<Option<TokenAnnotation>> {
        vec![
            Some(TokenAnnotation { pos: "det".into(), dep: "det".into(), head_abs: Some(1) }),
            Some(TokenAnnotation { pos: "noun".into(), dep: "nsubj".into(), head_abs: Some(2) }),
            Some(TokenAnnotation { pos: "verb".into(), dep: "root".into(), head_abs: Some(2) }),
            Some(TokenAnnotation { pos: "punct".into(), dep: "punct".into(), head_abs: Some(2) }),
        ]
    }

    #[test]
    fn maps_to_annotation_set_that_passes_the_seven_check_gate() {
        let doc = doc_for_cat_sat();
        let lemmatizer = spacy_rs::Lemmatizer::english_rule();
        let set = map_annotations(&doc, &lemmatizer, &valid_annotations()).expect("map");
        // Record count matches token count; text is the spacy orth by construction.
        assert_eq!(set.len(), doc.len());
        assert_eq!(set.records()[0].text, "The");
        assert_eq!(set.records()[3].text, ".");
        // F8 relative heads: head_abs − i.
        assert_eq!(set.records()[0].head, 1); // The → cat
        assert_eq!(set.records()[1].head, 1); // cat → sat
        assert_eq!(set.records()[2].head, 0); // sat → self (root)
        assert_eq!(set.records()[3].head, -1); // . → sat
        // The 7-check gate accepts the mapped set (valid tree, closed vocab).
        set.validate(&AnnotationValidator::new(), &doc).expect("7-check gate passes");
    }

    #[test]
    fn none_aligned_token_errors_fall_back() {
        let doc = doc_for_cat_sat();
        let lemmatizer = spacy_rs::Lemmatizer::english_rule();
        let mut annotations = valid_annotations();
        annotations[1] = None; // spacy token 1 had no covering LFM subword
        let err = map_annotations(&doc, &lemmatizer, &annotations).expect_err("None token");
        assert!(matches!(err, AnnotateError::Encoder(_)), "got {err:?}");
    }

    #[test]
    fn unknown_label_errors_fall_back() {
        let doc = doc_for_cat_sat();
        let lemmatizer = spacy_rs::Lemmatizer::english_rule();
        let mut annotations = valid_annotations();
        annotations[0] = Some(TokenAnnotation {
            pos: "not_a_real_pos".into(),
            dep: "det".into(),
            head_abs: Some(1),
        });
        let err = map_annotations(&doc, &lemmatizer, &annotations).expect_err("unknown label");
        assert!(matches!(err, AnnotateError::Encoder(_)), "got {err:?}");
    }

    #[test]
    fn none_head_is_decoded_as_root() {
        let doc = doc_for_cat_sat();
        let lemmatizer = spacy_rs::Lemmatizer::english_rule();
        // The root token's head maps to a special token → None → ROOT.
        let mut annotations = valid_annotations();
        annotations[2] = Some(TokenAnnotation {
            pos: "verb".into(),
            dep: "root".into(),
            head_abs: None,
        });
        let set = map_annotations(&doc, &lemmatizer, &annotations).expect("map");
        assert_eq!(set.records()[2].dep, "root");
        assert_eq!(set.records()[2].head, 0);
    }
}

#[test]
#[cfg(not(feature = "onnx"))]
fn nlp_encoder_fetch_noop_without_onnx_feature() {
    let registry = Arc::new(fluent_onnx::OrtSessionRegistry::new(Arc::new(
        fluent_onnx::OrtSessionLoader,
    )));
    let result = nlp_encoder_fetch(&registry, "encoder").expect("no ort error");
    assert!(result.is_none());
}

/// The full entity-link loop (ROADMAP_20260828_ORT_FIXES M3.2): the
/// scorer's canonical→id mapping (baked index + store) feeds an
/// `EntityLinkWorker`, and a PROPN-span candidate lands on
/// `overlay_candidates` (never a doc-id write). The ColBERT encode step is
/// model-dependent, so the scorer is built from the same pure
/// `EntitySimilarityIndex` + `score_span` path the adapter wires, with
/// synthetic query tokens.
#[cfg(feature = "onnx")]
#[tokio::test]
async fn entity_link_full_loop_scorer_to_overlay_candidates() {
    use crate::server::entity_link::{EntityLinkJob, EntityLinkWorker, EntityLinkScorer};
    use fluent_concurrency::tokio_runtime;
    use fluent_types::{ConceptMetadata, InterlinguaId, InterlinguaNamespace, NodeId};
    use spacy_rs::concept_store_mem::InMemoryConceptStore;
    use spacy_rs::ConceptStore;

    // YaGO `Entity` reference class + a child entity the scorer resolves to.
    let concepts = InMemoryConceptStore::new();
    let root = InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x100);
    concepts
        .insert(ConceptMetadata {
            id: root,
            canonical_name: "yago:Entity".into(),
            namespace: InterlinguaNamespace::YagoClass,
            yago_iri: None,
            yago_class_iri: None,
            label: None,
            node_id: None,
            parent_class_id: None,
        })
        .expect("root");
    let paris = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x001);
    concepts
        .insert(ConceptMetadata {
            id: paris,
            canonical_name: "yago:Paris".into(),
            namespace: InterlinguaNamespace::YagoEntity,
            yago_iri: None,
            yago_class_iri: None,
            label: Some("Paris".into()),
            node_id: None,
            parent_class_id: Some(root),
        })
        .expect("paris");
    let concepts: Arc<dyn ConceptStore> = Arc::new(concepts);

    // A baked index exactly as `bake_entity_index` produces (data-time).
    let index = fluent_onnx::EntitySimilarityIndex::new(
        vec![fluent_onnx::ConceptEncoding {
            namespace: "YagoEntity".into(),
            canonical: "yago:Paris".into(),
            token_embeddings: vec![vec![1.0, 0.0]],
        }],
        0.5,
    );
    // The scorer's encode step is model-dependent; use synthetic query tokens
    // through the same pure `score_span` mapping the adapter wires.
    let concepts_h = Arc::clone(&concepts);
    let scorer: EntityLinkScorer = Arc::new(move |_text| {
        score_span(&[vec![0.99, 0.14]], &index, concepts_h.as_ref())
    });

    let ledger = crate::ledger::ContentNodeLedger::open_in_memory().expect("ledger");
    let candidates = crate::ledger::overlay::OverlayCandidateStore::new(
        ledger.node_store().shared_sqlite().expect("shared"),
    );
    let node = NodeId::from_int(7);
    let worker = Arc::new(EntityLinkWorker::new(
        &candidates,
        &concepts,
        &scorer,
        0.5,
        root,
        8,
        4,
        tokio_runtime(),
    ));
    worker
        .submit(EntityLinkJob {
            node_id: node,
            span_start: 0,
            span_end: 5,
            text: "Paris".into(),
        })
        .await
        .expect("submit");
    for _ in 0..100 {
        if candidates.for_node(node).map(|r| r.len()).unwrap_or(0) >= 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    worker.drain().await;

    let rows = candidates.for_node(node).expect("query");
    assert_eq!(rows.len(), 1, "one entity-link candidate lands");
    assert_eq!(rows[0].entity_id, Some(paris), "resolved canonical → id");
    assert_eq!(rows[0].source, "entity_link");
}

#[cfg(feature = "onnx")]
#[test]
fn score_span_resolves_canonical_and_respects_threshold() {
    use fluent_types::{InterlinguaId, InterlinguaNamespace};
    use spacy_rs::concept_store_mem::InMemoryConceptStore;
    use spacy_rs::ConceptStore;

    let concepts = InMemoryConceptStore::new();
    let known = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x002);
    concepts
        .insert(fluent_types::ConceptMetadata {
            id: known,
            canonical_name: "yago:Paris".into(),
            namespace: InterlinguaNamespace::YagoEntity,
            yago_iri: None,
            yago_class_iri: None,
            label: Some("Paris".into()),
            node_id: None,
            parent_class_id: None,
        })
        .expect("insert");
    // The baked entry's canonical matches; a hit resolves to the id.
    let index = fluent_onnx::EntitySimilarityIndex::new(
        vec![fluent_onnx::ConceptEncoding {
            namespace: "YagoEntity".into(),
            canonical: "yago:Paris".into(),
            token_embeddings: vec![vec![1.0, 0.0]],
        }],
        0.9,
    );
    let scored = score_span(&[vec![0.99, 0.14]], &index, &concepts);
    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].0, known);
    assert!(scored[0].1 >= 0.9);

    // A below-threshold query (query far from the baked token) yields nothing.
    let cold = score_span(&[vec![0.0, 1.0]], &index, &concepts);
    assert!(cold.is_empty(), "below threshold → no candidates");
}

/// ROADMAP M2: the onnx `ChatBackend` seam — grammar wiring + fail-open.
#[cfg(feature = "onnx")]
mod onnx_backend_tests {
    use super::*;
    use fluent_llm::client::ChatBackend;
    use fluent_llm::{ChatMessage, LlmError};
    use fluent_onnx::LlmParams;
    use std::sync::Mutex;

    /// A fixed token-id → text map (mirrors the hermetic grammar tests).
    struct TestVocab {
        tokens: Vec<String>,
    }
    impl TestVocab {
        fn from_list(tokens: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                tokens: tokens.iter().map(|s| s.to_string()).collect(),
            })
        }
    }
    impl fluent_onnx::TokenVocab for TestVocab {
        fn token_text(&self, id: u32) -> Option<String> {
            self.tokens.get(id as usize).cloned()
        }
    }

    /// A fake decode runner that records whether each call was grammar-
    /// constrained (the injected-fake-decoder seam for hermetic tests).
    struct FakeRunner {
        calls: Mutex<Vec<bool>>,
        output: String,
    }
    impl OnnxLlmRunner for FakeRunner {
        fn complete(
            &self,
            _messages: &[ChatMessage],
            grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
            _max_tokens: Option<usize>,
            _params: LlmParams,
        ) -> Result<String, LlmError> {
            self.calls.lock().unwrap().push(grammar.is_some());
            Ok(self.output.clone())
        }
    }

    fn one_message() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }]
    }

    #[test]
    fn free_text_call_passes_no_grammar() {
        let vocab = TestVocab::from_list(&["{", "}", ":", "\"action\"", "\"x\""]);
        let runner = Arc::new(FakeRunner {
            calls: Mutex::new(Vec::new()),
            output: "free text".into(),
        });
        let backend = OnnxChatBackend::new(runner.clone(), vocab, LlmParams::default());
        let out = backend.chat_complete(&one_message()).expect("free text");
        assert_eq!(out, "free text");
        assert_eq!(runner.calls.lock().unwrap()[0], false, "no grammar for free text");
    }

    #[test]
    fn constrained_call_feeds_schema_into_a_grammar() {
        let vocab = TestVocab::from_list(&["{", "}", ":", ",", "\"action\"", "\"x\""]);
        let runner = Arc::new(FakeRunner {
            calls: Mutex::new(Vec::new()),
            output: "{\"action\":\"x\"}".into(),
        });
        let backend = OnnxChatBackend::new(runner.clone(), vocab, LlmParams::default());
        // The llama-fork `response_format.schema` vocabulary a constrained
        // caller (classifier / annotation) sends.
        let extras = serde_json::json!({
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"]
                }
            }
        });
        backend
            .chat_complete_with_extras(&one_message(), &extras)
            .expect("constrained call");
        assert_eq!(
            runner.calls.lock().unwrap()[0],
            true,
            "a response_format.schema must be fed into a grammar"
        );
    }

    #[test]
    fn unrepresentable_schema_degrades_to_free_text() {
        let vocab = TestVocab::from_list(&["{", "}", ":", "\"action\"", "\"x\""]);
        let runner = Arc::new(FakeRunner {
            calls: Mutex::new(Vec::new()),
            output: "whatever".into(),
        });
        let backend = OnnxChatBackend::new(runner.clone(), vocab, LlmParams::default());
        // A schema the structural grammar cannot represent (array-of-objects
        // fields, e.g. a review schema) → no grammar → free text (fail-open;
        // the caller's post-hoc validator is the backstop).
        let extras = serde_json::json!({
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"corrections": {"type": "array", "items": {"type": "object"}}},
                    "required": ["corrections"]
                }
            }
        });
        backend
            .chat_complete_with_extras(&one_message(), &extras)
            .expect("degrades to free text");
        assert_eq!(runner.calls.lock().unwrap()[0], false, "unrepresentable → no grammar");
    }

    #[test]
    fn onnx_chat_backend_returns_none_for_unregistered_and_non_causal_lm() {
        let config = RouterConfig::default();
        let registry = build_onnx_registry(&config).expect("build");
        assert!(registry.is_none(), "no fleet → no registry");

        // A registry with only an encoder (FillMask) role.
        let mut cfg = RouterConfig::default();
        cfg.onnx = Some(fluent_onnx::OnnxFleetConfig {
            encoder: Some(role_config("/models/encoder.onnx")),
            ..Default::default()
        });
        let registry = build_onnx_registry(&cfg).expect("build").expect("fleet");
        // Unregistered key → None (no warn path hit).
        let backend = onnx_chat_backend(&registry, "nonexistent").expect("no ort error");
        assert!(backend.is_none(), "unregistered key → None");
        // A registered but non-CausalLm key → None (loud warn).
        let encoder_key = fluent_onnx::OnnxRole::Encoder.registry_key();
        assert!(registry.is_registered(encoder_key));
        let backend = onnx_chat_backend(&registry, encoder_key).expect("no ort error");
        assert!(backend.is_none(), "non-CausalLm key → None (fail-open)");
    }

/// A stub `ChatBackend` for the resolver-routing tests.
struct StubBackend(&'static str);

impl fluent_llm::client::ChatBackend for StubBackend {
    fn chat_complete(
        &self,
        _messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        Ok(self.0.to_string())
    }
    fn chat_complete_with_extras(
        &self,
        _messages: &[fluent_llm::ChatMessage],
        _extras: &serde_json::Value,
    ) -> Result<String, fluent_llm::LlmError> {
        Ok(self.0.to_string())
    }
}

/// Stub onnx loader (no ort, no model) for the registry-level lazy-residency
/// tests below — the `OnnxWeights` lifecycle (load/release) is exercised
/// without constructing a real pool (which needs a real session).
#[derive(Default)]
struct StubLoader;

impl fluent_onnx::SessionLoader for StubLoader {
    fn load(
        &self,
        _config: &fluent_onnx::OnnxConfig,
        _model_key: &str,
    ) -> Result<fluent_onnx::SessionHandle, fluent_onnx::OrtError> {
        Ok(fluent_onnx::SessionHandle::new("stub"))
    }
}

/// An Unloadable, unpinned `llm` role config (generative — `CausalLm`), the
/// shape a lazy onnx role registers with (stays unloaded at boot).
fn lazy_llm_role() -> fluent_onnx::OnnxRoleConfig {
    fluent_onnx::OnnxRoleConfig {
        pinned: false,
        no_sleep: false,
        sleep_idle_seconds: Some(1),
        total_timeout_ms: 0,
        idle_timeout_ms: 0,
        params: None,
        instances: None,
        model: fluent_onnx::OnnxConfig::new()
            .model_path("/models/llm.onnx")
            .tokenizer_path("/models/llm/tokenizer.json")
            .resident(false)
            .task(fluent_onnx::OnnxTask::CausalLm)
            .build(),
    }
}

fn stub_registry() -> Arc<OrtSessionRegistry> {
    Arc::new(OrtSessionRegistry::new(Arc::new(StubLoader::default())))
}

/// ROADMAP M6: an `Unloadable`+unpinned onnx role stays unloaded at boot
/// and loads on first `ensure_loaded` (the onnx lazy-residency rule);
/// `unload` releases it. Registry-level (no real pool/session needed).
#[tokio::test]
async fn onnx_weights_lazy_role_loads_on_first_use_and_releases() {
    use fluent_llm::runtime::LlmWeights;
    let reg = stub_registry();
    let key = fluent_onnx::OnnxRole::Llm.registry_key();
    reg.register_with_lifecycle(
        key,
        lazy_llm_role().to_onnx_config(fluent_onnx::OnnxRole::Llm),
        fluent_onnx::ResidencyPolicy::Unloadable {
            weights: true,
            context: true,
        },
        false,
        Some(1),
    )
    .expect("register");
    let weights = OnnxWeights::new(key.to_string(), Arc::clone(&reg), lazy_llm_role());
    assert!(!weights.is_loaded(), "lazy role stays unloaded at boot");

    // First resolution (through the trait surface) loads it.
    weights.ensure_loaded().await.expect("lazy load");
    assert!(weights.is_loaded(), "first use loads the session");

    // A pinned/Always role would refuse; this Unloadable one unloads.
    weights.unload().await.expect("unload");
    assert!(!weights.is_loaded(), "release returns the lazy role to unloaded");
}

/// ROADMAP M6: `local_backend_for_instance` routes an onnx key to the
/// resolver's context-bound backend (the onnx analogue of `<base>:<instance>`
/// dispatch), while `local_backend` keeps the role's default.
#[test]
fn local_backend_for_instance_onnx_branch_routes_to_resolver() {

    let mut config = RouterConfig::default();
    let key = fluent_onnx::OnnxRole::Llm.registry_key();
    config.install_onnx_resolver(move |k, instance| {
        if k != key {
            return None;
        }
        match instance {
            Some(name) if name == "swarm" => {
                Some(Arc::new(StubBackend("onnx-swarm")))
            }
            _ => Some(Arc::new(StubBackend("onnx-default"))),
        }
    });

    // A named onnx context resolves through `local_backend_for_instance`.
    let instance_backend = config.local_backend_for_instance(key, "swarm");
    assert!(instance_backend.is_some(), "onnx instance branch resolves");
    assert_eq!(
        instance_backend.unwrap().chat_complete(&[]).unwrap(),
        "onnx-swarm"
    );

    // The role's default (no instance) still resolves through `local_backend`.
    let default_backend = config.local_backend(key);
    assert!(default_backend.is_some());
    assert_eq!(default_backend.unwrap().chat_complete(&[]).unwrap(), "onnx-default");
}

/// ROADMAP M6: `onnx_pool_context` mirrors the llama `pool_qualifier` rule
/// for an onnx role's `instances` block (largest non-default group wins).
#[test]
fn onnx_pool_context_uses_largest_non_default_group() {

    let mut role = lazy_llm_role();
    role.instances = Some(
        serde_json::from_str(r#"{
            "swarm":  { "num_ctx": 16384, "count": 3, "group": "swarm" },
            "ledger": { "num_ctx": 131072, "pinned": true, "default": true }
        }"#)
        .expect("instances json"),
    );
    assert_eq!(
        onnx_pool_context(&role).as_deref(),
        Some("swarm"),
        "largest non-default group is the pool context"
    );

    let mut single = lazy_llm_role();
    single.instances = Some(
        serde_json::from_str(r#"{ "ledger": { "num_ctx": 8192 } }"#).expect("instances"),
    );
    assert_eq!(
        onnx_pool_context(&single).as_deref(),
        Some("ledger"),
        "single shared group is the pool context"
    );

    let none = lazy_llm_role();
    assert_eq!(onnx_pool_context(&none), None, "no instances → no pool context");
    let _ = RouterConfig::default();
}

// ─── M12-S1 characterization: `pii_prefilter` cfg-pair contract ────────
// Both cfg definitions (onnx classifier vs no-ort regex fallback) must agree
// on these observable legs: no model + no auto-enqueue → `None`; no model +
// auto-enqueue → the regex baseline (detects a plain email); an unregistered
// model key degrades to the same two outcomes (fail-open, never an error).
// A future move of this wiring into `fluent-onnx`/`fluent-llm` must keep
// this table byte-identical.
#[test]
fn pii_prefilter_no_model_no_auto_enqueue_returns_none() {
    let filter = pii_prefilter(None, None, false).expect("no ort error");
    assert!(filter.is_none(), "nothing configured and nothing required → no filter");
}

#[test]
fn pii_prefilter_no_model_auto_enqueue_returns_regex_baseline() {
    let filter = pii_prefilter(None, None, true).expect("no ort error");
    let detector = filter.expect("auto-enqueue requires the regex baseline");
    let spans = detector
        .detect("contact alice@example.com for details")
        .expect("regex detect");
    assert!(!spans.is_empty(), "baseline must catch a plain email");
}

#[test]
fn pii_prefilter_unregistered_model_falls_back_without_error() {
    // Unregistered key + auto-enqueue → regex baseline (fail-open).
    let filter = pii_prefilter(None, Some("no-such-pii-model"), true).expect("no ort error");
    assert!(filter.is_some(), "unregistered model falls back to the baseline");
    // Unregistered key + no auto-enqueue → None.
    let filter = pii_prefilter(None, Some("no-such-pii-model"), false).expect("no ort error");
    assert!(filter.is_none());
}
}
