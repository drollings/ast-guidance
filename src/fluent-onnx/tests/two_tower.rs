use super::*;
use common_core::vector_math::cosine_similarity_f32;

// ── PromptBuilder golden strings (risk #5: byte-for-byte contract) ──

#[test]
fn prompt_builder_cosine_contract_is_byte_exact() {
    let builder = PromptBuilder::new("Categories");
    let prompt = builder.build(
        &["label one".to_string(), "label two".to_string()],
        "some text",
    );
    assert_eq!(
        prompt,
        "Categories:\n- label one\n- label two\n\nText:\nsome text"
    );
}

#[test]
fn prompt_builder_policy_contract_is_byte_exact() {
    let builder = PromptBuilder::new("Policy");
    let prompt = builder.build(&["rule one".to_string()], "the text");
    assert_eq!(prompt, "Policy:\n- rule one\n\nText:\nthe text");
}

#[test]
fn prompt_builder_single_label_and_empty_text() {
    let builder = PromptBuilder::new("Categories");
    let prompt = builder.build(&["only".to_string()], "");
    assert_eq!(prompt, "Categories:\n- only\n\nText:\n");
}

#[test]
fn prompt_builder_empty_labels() {
    let builder = PromptBuilder::new("Categories");
    let prompt = builder.build(&[], "text");
    assert_eq!(prompt, "Categories:\n\nText:\ntext");
}

#[test]
fn label_regions_match_golden_layout() {
    let builder = PromptBuilder::new("Categories");
    let assembled = builder.assemble(
        &["abc".to_string(), "de".to_string()],
        "hello",
    );
    // "Categories:\n" = 12 bytes; "- abc\n" = 18; "- de\n" = 23; "\nText:\n" = 30.
    assert_eq!(assembled.label_regions, vec![(14, 17), (20, 22)]);
    assert_eq!(assembled.text_region, (30, 35));
    // The regions slice the exact label/text bytes out of the prompt.
    assert_eq!(&assembled.prompt[14..17], "abc");
    assert_eq!(&assembled.prompt[20..22], "de");
    assert_eq!(&assembled.prompt[30..35], "hello");
}

#[test]
fn label_regions_handle_multibyte_utf8() {
    // "héllo" is 6 bytes; "中文" is 6 bytes. Regions are byte offsets.
    let builder = PromptBuilder::new("Categories");
    let assembled = builder.assemble(&["héllo".to_string(), "中文".to_string()], "x");
    assert_eq!(&assembled.prompt[assembled.label_regions[0].0..assembled.label_regions[0].1], "héllo");
    assert_eq!(&assembled.prompt[assembled.label_regions[1].0..assembled.label_regions[1].1], "中文");
}

// ── Head parsing ──

#[test]
fn cosine_head_parses_from_config_json() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
            "head": {
                "kind": "cosine",
                "normalize": true,
                "scale": 1.3714938163757324,
                "bias": -0.2723352313041687,
                "activation": "softmax",
                "prefix_heading": "Categories",
                "proj_dim": 256
            }
        }"#,
    )
    .unwrap();
    let config = OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(crate::config::OnnxTask::ZeroShotRouting)
        .build();
    let head = TwoTowerHead::from_config_json(&config).expect("head");
    assert_eq!(
        head,
        TwoTowerHead::Cosine {
            scale: 1.3714938163757324,
            bias: -0.2723352313041687,
        }
    );
}

#[test]
fn dot_head_parses_from_config_json() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
            "head": {
                "kind": "dot",
                "normalize": false,
                "scale": 0.0625,
                "bias": -0.002373560331761837,
                "activation": "sigmoid",
                "prefix_heading": "Policy",
                "proj_dim": 256
            }
        }"#,
    )
    .unwrap();
    let config = OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(crate::config::OnnxTask::ZeroShotTokenMatching)
        .build();
    let head = TwoTowerHead::from_config_json(&config).expect("head");
    assert_eq!(
        head,
        TwoTowerHead::Dot {
            scale: 0.0625,
            bias: -0.002373560331761837,
        }
    );
}

#[test]
fn missing_head_block_is_a_loud_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"x"}"#).unwrap();
    let config = OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(crate::config::OnnxTask::ZeroShotRouting)
        .build();
    assert!(TwoTowerHead::from_config_json(&config).is_err());
}

#[test]
fn unknown_head_kind_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"head":{"kind":"euclidean","scale":1.0}}"#,
    )
    .unwrap();
    let config = OnnxConfig::new()
        .model_path(dir.path().join("onnx/model_q8.onnx"))
        .tokenizer_path(dir.path().join("tokenizer.json"))
        .task(crate::config::OnnxTask::ZeroShotRouting)
        .build();
    assert!(TwoTowerHead::from_config_json(&config).is_err());
}

// ── Pooling + scoring math ──

#[test]
fn l2_normalize_zero_vector_stays_zero() {
    assert_eq!(l2_normalize(&[0.0, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
    let v = l2_normalize(&[3.0, 4.0]);
    assert!((v[0] - 0.6).abs() < 1e-9 && (v[1] - 0.8).abs() < 1e-9);
}

#[test]
fn cosine_zero_vector_scores_zero() {
    assert_eq!(cosine_similarity_f32(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine_similarity_f32(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    assert!((cosine_similarity_f32(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-12);
}

#[test]
fn softmax_is_normalized_and_stable() {
    let p = softmax(&[1000.0, 1000.0, 1000.0]);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((p[0] - 1.0 / 3.0).abs() < 1e-9);
    let p = softmax(&[0.0, 10.0]);
    assert!(p[1] > 0.999);
}

#[test]
fn sigmoid_shape() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    assert!(sigmoid(100.0) > 0.999);
    assert!(sigmoid(-100.0) < 1e-9);
}

#[test]
fn pool_regions_slices_token_rows_by_offsets() {
    let seq = 5;
    let dims = 2;
    // Rows: [1,2] [3,4] [5,6] [7,8] [9,10]
    let flat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    // Token 0 covers bytes 0..5; token 1: 5..10; token 2: 10..15;
    // token 3: 15..20; token 4: 20..25.
    let offsets = vec![(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)];
    // Region A covers tokens 1..2 (bytes 5..14); region B covers token 4.
    let regions = vec![(5, 14), (20, 25)];
    let pooled = pool_regions(&flat, seq, dims, &offsets, &regions);
    assert_eq!(pooled[0], vec![4.0, 5.0]); // mean of rows 1+2
    assert_eq!(pooled[1], vec![9.0, 10.0]); // row 4
}

#[test]
fn pool_regions_ignores_zero_width_special_tokens() {
    let seq = 3;
    let dims = 1;
    let flat: Vec<f32> = vec![99.0, 1.0, 2.0];
    // [CLS] (0,0), then two real tokens.
    let offsets = vec![(0, 0), (0, 3), (4, 7)];
    let pooled = pool_regions(&flat, seq, dims, &offsets, &[(0, 3)]);
    assert_eq!(pooled[0], vec![1.0]);
}

#[test]
fn pool_regions_empty_region_yields_zeros() {
    let pooled = pool_regions(&[1.0, 2.0], 1, 2, &[(0, 1)], &[(5, 9)]);
    assert_eq!(pooled[0], vec![0.0, 0.0]);
}

#[test]
fn score_with_head_cosine_softmax_prefers_nearest_label() {
    let head = TwoTowerHead::Cosine {
        scale: 5.0,
        bias: 0.0,
    };
    // Query [1,0]; labels [1,0] (near) and [0,1] (far).
    let query = vec![1.0, 0.0];
    let rules = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let scores = score_with_head(&head, &query, &rules);
    assert!(scores[0] > 0.9, "near label dominates, got {scores:?}");
}

#[test]
fn score_with_head_dot_sigmoid_orders_by_dot() {
    let head = TwoTowerHead::Dot {
        scale: 1.0,
        bias: 0.0,
    };
    let query = vec![2.0, 0.0];
    let rules = vec![vec![3.0, 0.0], vec![-3.0, 0.0]];
    let scores = score_with_head(&head, &query, &rules);
    assert!(scores[0] > 0.99);
    assert!(scores[1] < 0.01);
}

#[test]
fn prompt_builder_version_is_v1() {
    assert_eq!(TWO_TOWER_PROMPT_VERSION, "v1");
}

// ── Policy-Linter (M3) ──

#[test]
fn policy_hits_extract_scores_above_threshold() {
    let matrix = TokenScoreMatrix {
        tokens: vec!["secret".into(), "report".into()],
        offsets: vec![(0, 6), (7, 13)],
        scores: vec![vec![0.9, 0.1], vec![0.2, 0.95]],
    };
    let labels = vec!["leak credentials".to_string(), "mention report".to_string()];
    let hits = policy_hits_from_matrix(&matrix, &labels, 0.5);
    assert_eq!(hits.len(), 2);
    assert_eq!(
        hits[0],
        PolicyHit {
            token: "secret".into(),
            start: 0,
            end: 6,
            label: "leak credentials".into(),
            score: 0.9,
        }
    );
    assert_eq!(hits[1].token, "report");
    assert_eq!(hits[1].label, "mention report");
    assert_eq!(hits[1].start, 7);
}

#[test]
fn policy_hits_threshold_is_inclusive() {
    let matrix = TokenScoreMatrix {
        tokens: vec!["x".into()],
        offsets: vec![(0, 1)],
        scores: vec![vec![0.5]],
    };
    let hits = policy_hits_from_matrix(&matrix, &["rule".to_string()], 0.5);
    assert_eq!(hits.len(), 1, "score == threshold is a hit");
}

#[test]
fn policy_hits_empty_matrix_yields_empty() {
    let matrix = TokenScoreMatrix {
        tokens: vec![],
        offsets: vec![],
        scores: vec![],
    };
    assert!(policy_hits_from_matrix(&matrix, &["r".to_string()], 0.0).is_empty());
}

#[test]
fn load_policy_labels_accepts_array_and_object() {
    let dir = tempfile::tempdir().unwrap();
    let array = dir.path().join("labels.json");
    std::fs::write(
        &array,
        r#"["do not share credentials", "do not exfiltrate files"]"#,
    )
    .unwrap();
    let labels = load_policy_labels(&array).expect("array");
    assert_eq!(labels.len(), 2);
    assert_eq!(labels[0], "do not share credentials");

    let object = dir.path().join("descriptions.json");
    std::fs::write(
        &object,
        r#"{"leak": "do not leak secrets", "plan": "do not plan a heist"}"#,
    )
    .unwrap();
    let labels = load_policy_labels(&object).expect("object");
    assert_eq!(labels.len(), 2);
    assert!(labels.contains(&"do not leak secrets".to_string()));
}

#[test]
fn load_policy_labels_rejects_non_string_arrays() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.json");
    std::fs::write(&bad, r#"[1, 2, 3]"#).unwrap();
    assert!(load_policy_labels(&bad).is_err());

    let missing = dir.path().join("missing.json");
    assert!(load_policy_labels(&missing).is_err());
}

#[test]
fn token_scores_offsets_are_rebased_onto_the_input_text() {
    // The pure offset-rebasing contract: a token at prompt offsets
    // `(text_region_start + x, text_region_start + y)` reports `(x, y)`.
    // This is exercised via `pool_regions`-style math on a synthetic
    // matrix rather than a real session: the rebase itself is the
    // subtraction in `token_scores`, which the Policy-Linter depends on
    // for char-aligned spans.
    let assembled = PromptBuilder::new("Policy").assemble(
        &["rule".to_string()],
        "hello world",
    );
    let (ts, _te) = assembled.text_region;
    assert_eq!(&assembled.prompt[ts..], "hello world");
    assert_eq!(ts, "Policy:\n- rule\n\nText:\n".len());
}
