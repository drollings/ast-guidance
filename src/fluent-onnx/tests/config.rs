use super::*;

#[test]
fn serde_round_trip_all_variants() {
    for task in [
        OnnxTask::FillMask,
        OnnxTask::ZeroShotRouting,
        OnnxTask::ZeroShotTokenMatching,
        OnnxTask::TokenClassification,
        OnnxTask::LateInteraction,
        OnnxTask::MaskedDiffusion,
        OnnxTask::CausalLm,
    ] {
        for quant in [Quant::Fp32, Quant::Fp16, Quant::Fp8, Quant::Q8, Quant::Q4] {
            let cfg = OnnxConfig::new()
                .model_path("/models/encoder/onnx/model_q8.onnx")
                .tokenizer_path("/models/encoder/tokenizer.json")
                .task(task)
                .quantization(quant)
                .build();
            let json = serde_json::to_string(&cfg).unwrap();
            let back: OnnxConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(cfg, back);
        }
    }
}

#[test]
fn task_serde_names() {
    assert_eq!(
        serde_json::to_string(&OnnxTask::ZeroShotTokenMatching).unwrap(),
        "\"zero_shot_token_matching\""
    );
    assert_eq!(serde_json::to_string(&Quant::Q8).unwrap(), "\"q8\"");
}

#[test]
fn defaults_applied_on_deserialize() {
    let json = r#"{"model_path":"m.onnx","tokenizer_path":"t.json","task":"fill_mask"}"#;
    let cfg: OnnxConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.execution_provider, "cpu");
    assert_eq!(cfg.intra_threads, 1);
    assert_eq!(cfg.optimization_level, "all");
    assert_eq!(cfg.quantization, Quant::Q8);
    assert!(cfg.resident);
    assert_eq!(cfg.max_seq_len, 1024);
    assert_eq!(cfg.dims, None);
    assert_eq!(cfg.label_source, None);
    assert_eq!(cfg.annotation_heads, None);
    assert_eq!(cfg.max_gen_tokens, 512);
    assert_eq!(cfg.resident_bytes, None);
    assert_eq!(cfg.llm_io, None);
    assert_eq!(cfg.max_ctx, None);
    assert!(matches!(cfg.policy(), ResidencyPolicy::Always));
}

#[test]
fn resident_knob_derives_policy() {
    assert!(ResidencyPolicy::from_resident(true).is_always());
    let unloadable = ResidencyPolicy::from_resident(false);
    assert!(matches!(
        unloadable,
        ResidencyPolicy::Unloadable {
            weights: true,
            context: true
        }
    ));
    assert!(!unloadable.is_always());
}

#[test]
fn suggested_quantization_is_fp32_for_binary_gates() {
    assert_eq!(
        Quant::suggested_for(OnnxTask::ZeroShotTokenMatching),
        Quant::Fp32
    );
    assert_eq!(
        Quant::suggested_for(OnnxTask::TokenClassification),
        Quant::Fp32
    );
    assert_eq!(Quant::suggested_for(OnnxTask::FillMask), Quant::Q8);
    assert_eq!(Quant::suggested_for(OnnxTask::ZeroShotRouting), Quant::Q8);
    assert!(Quant::Fp32.is_fp32());
    assert!(!Quant::Q8.is_fp32());
}

#[test]
fn policy_labels_round_trips_and_defaults_to_none() {
    let cfg = OnnxConfig::new()
        .model_path("/models/policy.onnx")
        .tokenizer_path("/models/tokenizer.json")
        .task(OnnxTask::ZeroShotTokenMatching)
        .maybe_policy_labels(Some(std::path::PathBuf::from("/models/policy_labels.json")))
        .build();
    assert_eq!(
        cfg.policy_labels.as_deref(),
        Some(std::path::Path::new("/models/policy_labels.json"))
    );
    let json = serde_json::to_string(&cfg).unwrap();
    let back: OnnxConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.policy_labels, cfg.policy_labels);

    let plain: OnnxConfig = serde_json::from_str(
        r#"{"model_path":"m.onnx","tokenizer_path":"t.json","task":"zero_shot_token_matching"}"#,
    )
    .unwrap();
    assert_eq!(plain.policy_labels, None);
}

#[test]
fn residency_policy_serde_round_trip() {
    for policy in [
        ResidencyPolicy::Always,
        ResidencyPolicy::Unloadable {
            weights: true,
            context: false,
        },
    ] {
        let json = serde_json::to_string(&policy).unwrap();
        let back: ResidencyPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(policy, back);
    }
}

#[test]
fn validate_accepts_matching_architecture() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("encoder");
    std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
    std::fs::write(model_dir.join("config.json"), r#"{
        "architectures": ["MaskedLmGraph"],
        "onnx": {"inputs": ["input_ids","attention_mask"], "outputs": ["logits","last_hidden_state"]}
    }"#)
    .unwrap();
    let cfg = OnnxConfig::new()
        .model_path(model_dir.join("onnx/model_q8.onnx"))
        .tokenizer_path(model_dir.join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .build();
    cfg.validate().unwrap();
}

#[test]
fn validate_rejects_mismatched_architecture() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("router");
    std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
    std::fs::write(model_dir.join("config.json"), r#"{
        "architectures": ["TwoTowerGraph"],
        "onnx": {"inputs": ["input_ids","attention_mask"], "outputs": ["token_proj","rule_proj"]}
    }"#)
    .unwrap();
    let cfg = OnnxConfig::new()
        .model_path(model_dir.join("onnx/model_q8.onnx"))
        .tokenizer_path(model_dir.join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .build();
    assert!(matches!(cfg.validate(), Err(OrtError::TaskMismatch { .. })));
}

#[test]
fn validate_tolerates_missing_config_json() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .build();
    cfg.validate().unwrap();
}

// ── M2.1: the annotation-heads contract ──

fn heads_config(dir: &tempfile::TempDir) -> OnnxConfig {
    OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .annotation_heads(
            AnnotationHeads::new()
                .labels(dir.path().join("annotation_labels.json"))
                .build(),
        )
        .build()
}

fn write_labels(dir: &tempfile::TempDir) {
    std::fs::write(
        dir.path().join("annotation_labels.json"),
        r#"{"upos":["NOUN","VERB","ADJ"],"dep":["nsubj","obj","det"]}"#,
    )
    .unwrap();
}

#[test]
fn annotation_heads_serde_round_trip_with_defaults() {
    let heads = AnnotationHeads::new()
        .labels("/models/annotation_labels.json")
        .build();
    assert_eq!(heads.pos_output, "upos_logits");
    assert_eq!(heads.dep_output, "dep_logits");
    assert_eq!(heads.head_output, "head_logits");
    let json = serde_json::to_string(&heads).unwrap();
    // A labels-only onnx block round-trips with the default head names.
    let from: AnnotationHeads =
        serde_json::from_str(&format!(r#"{{"labels":"/l.json"}}"#)).unwrap();
    assert_eq!(from.pos_output, "upos_logits");
    let back: AnnotationHeads = serde_json::from_str(&json).unwrap();
    assert_eq!(back, heads);
}

#[test]
fn validate_accepts_matching_annotation_heads() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["MaskedLmGraph"],
            "onnx":{"outputs":["last_hidden_state","upos_logits","dep_logits","head_logits"]}}"#,
    )
    .unwrap();
    write_labels(&dir);
    heads_config(&dir).validate().unwrap();
}

#[test]
fn validate_rejects_missing_head_output_loudly() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    // The config.json declares the base output but omits `head_logits`.
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["MaskedLmGraph"],
            "onnx":{"outputs":["last_hidden_state","upos_logits","dep_logits"]}}"#,
    )
    .unwrap();
    write_labels(&dir);
    let err = heads_config(&dir).validate().expect_err("missing head");
    assert!(
        matches!(err, OrtError::OutputMismatch { ref missing, .. } if missing == "head_logits"),
        "expected a loud head-output mismatch, got {err:?}"
    );
}

#[test]
fn validate_rejects_missing_labels_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["MaskedLmGraph"],
            "onnx":{"outputs":["last_hidden_state","upos_logits","dep_logits","head_logits"]}}"#,
    )
    .unwrap();
    // No labels file written → loud ConfigRead error.
    let err = heads_config(&dir).validate().expect_err("missing labels");
    assert!(matches!(err, OrtError::ConfigRead { .. }), "got {err:?}");
}

#[test]
fn validate_rejects_empty_label_vocabularies() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["MaskedLmGraph"],
            "onnx":{"outputs":["last_hidden_state","upos_logits","dep_logits","head_logits"]}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("annotation_labels.json"),
        r#"{"upos":[],"dep":[]}"#,
    )
    .unwrap();
    let err = heads_config(&dir).validate().expect_err("empty labels");
    assert!(matches!(err, OrtError::ConfigParse { .. }), "got {err:?}");
}

#[test]
fn annotation_labels_load_parses() {
    let dir = tempfile::tempdir().unwrap();
    write_labels(&dir);
    let labels = AnnotationLabels::load(&dir.path().join("annotation_labels.json")).unwrap();
    assert_eq!(labels.upos, vec!["NOUN", "VERB", "ADJ"]);
    assert_eq!(labels.dep, vec!["nsubj", "obj", "det"]);
}

// ── Role-based fleet declarations ──

#[test]
fn role_maps_to_task_and_registry_key() {
    assert_eq!(OnnxRole::Encoder.task(), OnnxTask::FillMask);
    assert_eq!(OnnxRole::Pii.task(), OnnxTask::TokenClassification);
    assert_eq!(OnnxRole::Router.task(), OnnxTask::ZeroShotRouting);
    assert_eq!(OnnxRole::Policy.task(), OnnxTask::ZeroShotTokenMatching);
    assert_eq!(OnnxRole::Colbert.task(), OnnxTask::LateInteraction);
    assert_eq!(OnnxRole::Llm.task(), OnnxTask::CausalLm);
    assert_eq!(OnnxRole::Encoder.registry_key(), "onnx/encoder");
    assert_eq!(OnnxRole::Colbert.registry_key(), "onnx/colbert");
    assert_eq!(OnnxRole::Llm.registry_key(), "onnx/llm");
    // Binary gates recommend fp32.
    assert!(OnnxRole::Pii.suggested_quantization().is_fp32());
    assert!(OnnxRole::Policy.suggested_quantization().is_fp32());
    assert!(!OnnxRole::Encoder.suggested_quantization().is_fp32());
    assert!(!OnnxRole::Llm.suggested_quantization().is_fp32());
}

#[test]
fn role_config_task_is_injected_from_role() {
    let cfg = OnnxRoleConfig {
        pinned: true,
        no_sleep: true,
        sleep_idle_seconds: Some(30),
        total_timeout_ms: 90000,
        idle_timeout_ms: 0,
        params: None,
        instances: None,
        model: OnnxConfig::new()
            .model_path("/models/encoder.onnx")
            .tokenizer_path("/models/tokenizer.json")
            .build(),
    };
    let model = cfg.clone().to_onnx_config(OnnxRole::Encoder);
    assert_eq!(model.task, OnnxTask::FillMask);
    // The role config itself defaults the task (serde), then the role overrides.
    assert_eq!(cfg.model.task, OnnxTask::FillMask);
}

#[test]
fn role_serde_round_trip_defaults_task_and_lifecycle() {
    let json = r#"{
        "model_path": "/models/pii.onnx",
        "tokenizer_path": "/models/pii/tokenizer.json",
        "pinned": true,
        "total_timeout_ms": 120000
    }"#;
    let cfg: OnnxRoleConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.model.task, OnnxTask::FillMask, "task defaults when omitted");
    assert!(cfg.pinned);
    assert_eq!(cfg.total_timeout_ms, 120000);
    assert!(!cfg.no_sleep);
    assert_eq!(cfg.sleep_idle_seconds, None);
    let back: OnnxRoleConfig = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert_eq!(back, cfg);
}

#[test]
fn fleet_iter_yields_only_configured_roles() {
    let fleet = OnnxFleetConfig {
        encoder: Some(OnnxRoleConfig {
            pinned: true,
            no_sleep: false,
            sleep_idle_seconds: None,
            total_timeout_ms: 0,
            idle_timeout_ms: 0,
            params: None,
            instances: None,
            model: OnnxConfig::new()
                .model_path("/models/encoder.onnx")
                .tokenizer_path("/models/tokenizer.json")
                .build(),
        }),
        pii: None,
        router: None,
        policy: None,
        colbert: Some(OnnxRoleConfig {
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: None,
            total_timeout_ms: 0,
            idle_timeout_ms: 0,
            params: None,
            instances: None,
            model: OnnxConfig::new()
                .model_path("/models/colbert.onnx")
                .tokenizer_path("/models/colbert/tokenizer.json")
                .build(),
        }),
        llm: Some(OnnxRoleConfig {
            pinned: true,
            no_sleep: false,
            sleep_idle_seconds: Some(30),
            total_timeout_ms: 120000,
            idle_timeout_ms: 0,
            params: None,
            instances: None,
            model: OnnxConfig::new()
                .model_path("/models/llm.onnx")
                .tokenizer_path("/models/llm/tokenizer.json")
                .task(OnnxTask::CausalLm)
                .build(),
        }),
    };
    assert!(!fleet.is_empty());
    let keys: Vec<OnnxRole> = fleet.iter().map(|(r, _)| r).collect();
    assert_eq!(
        keys,
        vec![OnnxRole::Encoder, OnnxRole::Colbert, OnnxRole::Llm]
    );
    assert!(fleet.has(OnnxRole::Encoder));
    assert!(!fleet.has(OnnxRole::Pii));
    assert!(fleet.has(OnnxRole::Llm));

    let empty = OnnxFleetConfig::default();
    assert!(empty.is_empty());
    assert_eq!(empty.iter().count(), 0);
}

#[test]
fn fleet_defaults_every_role_absent() {
    let fleet = OnnxFleetConfig::default();
    for role in OnnxRole::all() {
        assert!(!fleet.has(role));
    }
}

// ── M0: the generative `CausalLm`/`Llm` surface ──

#[test]
fn causal_lm_task_contract() {
    assert_eq!(
        OnnxTask::CausalLm.expected_architecture_families(),
        &["Lfm2ForCausalLM"]
    );
    assert_eq!(OnnxTask::CausalLm.required_outputs(), &["logits"]);
    assert_eq!(
        serde_json::to_string(&OnnxTask::CausalLm).unwrap(),
        "\"causal_lm\""
    );
    // The generative task is not a binary gate — q8 (or q4 for the
    // checkpoint) is acceptable.
    assert_eq!(Quant::suggested_for(OnnxTask::CausalLm), Quant::Q8);
}

#[test]
fn llm_io_defaults_match_transformers_js_export() {
    let io = LlmIo::new().build();
    assert_eq!(io.input_ids, "input_ids");
    assert_eq!(io.attention_mask, "attention_mask");
    assert_eq!(io.position_ids, "position_ids");
    assert_eq!(io.past_key_values, "past_key_values");
    assert_eq!(io.conv_state, "past_conv");
    assert_eq!(io.present, "present");
    assert_eq!(io.present_conv, "present_conv");
    assert_eq!(io.logits, "logits");
    assert_eq!(io.kv_cache_dtype, "float32");
    assert_eq!(io.num_key_value_heads, 8);
    assert_eq!(io.head_dim, 64);
    assert_eq!(io.hidden_size, 2048);
    assert_eq!(io.conv_l_cache, 3);
    assert_eq!(io.attention_layers, vec![2, 5, 9, 13, 17, 21, 24, 27]);
    assert_eq!(io.conv_layers.len(), 22);
    // The two layer lists are disjoint and together cover every layer 0..30.
    let mut all: Vec<usize> = io.attention_layers.iter().chain(io.conv_layers.iter()).copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all, (0..30).collect::<Vec<usize>>());
    assert_eq!(all.len(), 30);

    // A bare `{}` onnx llm_io block round-trips with the defaults.
    let from: LlmIo = serde_json::from_str("{}").unwrap();
    assert_eq!(from, io);
    let json = serde_json::to_string(&from).unwrap();
    let back: LlmIo = serde_json::from_str(&json).unwrap();
    assert_eq!(back, io);

    // An explicit override survives the round-trip.
    let custom = LlmIo::new()
        .logits("final_logits".to_string())
        .kv_cache_dtype("float16".to_string())
        .conv_state("past_conv".to_string())
        .build();
    let back: LlmIo = serde_json::from_str(&serde_json::to_string(&custom).unwrap()).unwrap();
    assert_eq!(back.logits, "final_logits");
    assert_eq!(back.kv_cache_dtype, "float16");
    assert_eq!(back.conv_state, "past_conv");
}

#[test]
fn onnx_config_generative_knobs_round_trip() {
    let cfg = OnnxConfig::new()
        .model_path("/models/llm.onnx")
        .tokenizer_path("/models/tokenizer.json")
        .task(OnnxTask::CausalLm)
        .max_gen_tokens(256)
        .maybe_resident_bytes(Some(2_000_000_000))
        .maybe_llm_io(Some(LlmIo::new().build()))
        .build();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: OnnxConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back, cfg);
    assert_eq!(back.max_gen_tokens, 256);
    assert_eq!(back.resident_bytes, Some(2_000_000_000));
    assert_eq!(back.llm_io, Some(LlmIo::new().build()));
}

#[test]
fn validate_accepts_causal_lm_config() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("llm");
    std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
    // The LFM2.5-2.6B export declares `Lfm2ForCausalLM` and carries no
    // `onnx.outputs` block — the architecture check is the loud surface.
    std::fs::write(model_dir.join("config.json"), r#"{
        "architectures": ["Lfm2ForCausalLM"],
        "use_cache": true,
        "vocab_size": 128000
    }"#)
    .unwrap();
    let cfg = OnnxConfig::new()
        .model_path(model_dir.join("onnx/model_q4.onnx"))
        .tokenizer_path(model_dir.join("tokenizer.json"))
        .task(OnnxTask::CausalLm)
        .quantization(Quant::Q4)
        .max_gen_tokens(512)
        .build();
    cfg.validate().unwrap();

    // A mismatched architecture is still a loud boot error for CausalLm.
    std::fs::write(model_dir.join("config.json"), r#"{
        "architectures": ["TwoTowerGraph"],
        "onnx": {"outputs": ["token_proj", "rule_proj"]}
    }"#)
    .unwrap();
    assert!(matches!(
        cfg.validate(),
        Err(OrtError::TaskMismatch { .. })
    ));
}

// ── M3: onnx context-declaration vocabulary (`instances` + `max_ctx`) ──

#[test]
fn onnx_config_max_ctx_defaults_absent_and_round_trips() {
    // Absent `max_ctx` (the default) → `None`; byte-identical to today.
    let plain: OnnxConfig =
        serde_json::from_str(r#"{"model_path":"m.onnx","tokenizer_path":"t.json"}"#).unwrap();
    assert_eq!(plain.max_ctx, None);

    let with_cap: OnnxConfig = serde_json::from_str(r#"{
        "model_path": "m.onnx",
        "tokenizer_path": "t.json",
        "max_ctx": 8192
    }"#)
    .unwrap();
    assert_eq!(with_cap.max_ctx, Some(8192));

    let back: OnnxConfig =
        serde_json::from_str(&serde_json::to_string(&with_cap).unwrap()).unwrap();
    assert_eq!(back, with_cap);
}

#[test]
fn onnx_role_config_instances_absent_defaults_to_none() {
    // An `OnnxRoleConfig` without an `instances` block keeps today's
    // single-shot decode (None) — byte-identical for existing configs.
    let cfg: OnnxRoleConfig = serde_json::from_str(r#"{
        "model_path": "/models/pii.onnx",
        "tokenizer_path": "/models/pii/tokenizer.json",
        "pinned": true
    }"#)
    .unwrap();
    assert!(cfg.instances.is_none());
}

#[test]
fn onnx_role_config_instances_round_trip() {
    // A declared `instances` block uses the same profile vocabulary as the
    // llama `ModelEntry.instances`, and round-trips exactly.
    let json = r#"{
        "model_path": "/models/llm.onnx",
        "tokenizer_path": "/models/llm/tokenizer.json",
        "pinned": true,
        "instances": {
            "swarm": { "num_ctx": 16384, "count": 3, "group": "swarm", "max_ctx": 8192 },
            "ledger": { "num_ctx": 131072, "pinned": true, "default": true, "resume": true }
        }
    }"#;
    let cfg: OnnxRoleConfig = serde_json::from_str(json).unwrap();
    let instances = cfg.instances.as_ref().expect("instances block parsed");
    assert_eq!(instances.len(), 2);
    let swarm = &instances["swarm"];
    assert_eq!(swarm.num_ctx, 16384);
    assert_eq!(swarm.count, 3);
    assert_eq!(swarm.group.as_deref(), Some("swarm"));
    assert_eq!(swarm.max_ctx, Some(8192));
    let ledger = &instances["ledger"];
    assert!(ledger.pinned && ledger.default && ledger.resume);
    assert!(ledger.max_ctx.is_none());

    // The `warm` alias maps to `no_sleep` exactly as in the llama grammar.
    let warm: OnnxInstanceProfile =
        serde_json::from_str(r#"{"num_ctx": 8192, "warm": true}"#).unwrap();
    assert!(warm.no_sleep);

    let back: OnnxRoleConfig = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert_eq!(back, cfg);
}

#[test]
fn onnx_instance_profile_max_ctx_clamps() {
    // A cap below `num_ctx` clamps the window at materialization; the
    // clamp is a no-op when `max_ctx` is absent.
    let mut clamped: OnnxInstanceProfile =
        serde_json::from_str(r#"{"num_ctx": 65536, "max_ctx": 8192}"#).unwrap();
    clamped.apply_max_ctx();
    assert_eq!(clamped.num_ctx, 8192);

    let mut above: OnnxInstanceProfile =
        serde_json::from_str(r#"{"num_ctx": 4096, "max_ctx": 8192}"#).unwrap();
    above.apply_max_ctx();
    assert_eq!(above.num_ctx, 4096, "cap above num_ctx is a no-op");

    let mut absent: OnnxInstanceProfile =
        serde_json::from_str(r#"{"num_ctx": 16384}"#).unwrap();
    absent.apply_max_ctx();
    assert_eq!(absent.num_ctx, 16384, "absent max_ctx is a no-op");
}
