use super::*;

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

#[test]
fn chat_template_renders_roles_and_generation_prompt() {
    let text = apply_chat_template(&[
        msg("system", "You are a router."),
        msg("user", "What is 2+2?"),
    ]);
    assert_eq!(
        text,
        "<|im_start|>system\nYou are a router.<|im_end|>\n\
         <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

#[test]
fn chat_template_empty_messages_still_opens_assistant() {
    assert_eq!(apply_chat_template(&[]), "<|im_start|>assistant\n");
}

#[test]
fn params_determinism() {
    assert!(LlmParams::default().is_deterministic());
    assert!(LlmParams {
        temperature: 0.0,
        do_sample: true,
    }
    .is_deterministic());
    assert!(!LlmParams {
        temperature: 0.8,
        do_sample: true,
    }
    .is_deterministic());
}

#[test]
fn argmax_selects_highest_logit() {
    let logits = vec![0.1f32, 0.9, 0.5, -1.0, 3.0, 0.0];
    assert_eq!(sample_next_token(&logits, None, LlmParams::default()), 4);
}

#[test]
fn argmax_respects_allowed_set() {
    let logits = vec![0.1f32, 0.9, 0.5, -1.0, 3.0, 0.0];
    let allowed: HashSet<u32> = [0u32, 1, 2].into_iter().collect();
    assert_eq!(
        sample_next_token(&logits, Some(&allowed), LlmParams::default()),
        1
    );
    let allowed: HashSet<u32> = [0u32, 3].into_iter().collect();
    assert_eq!(
        sample_next_token(&logits, Some(&allowed), LlmParams::default()),
        0
    );
}

#[test]
fn sample_is_within_allowed_or_global() {
    let logits = vec![0.1f32, 0.9, 0.5, -1.0, 3.0, 0.0];
    let params = LlmParams {
        temperature: 0.8,
        do_sample: true,
    };
    let picked = sample_next_token(&logits, None, params);
    assert!(picked < logits.len() as u32);
}

#[test]
fn temperature_sample_is_reproducible() {
    let logits = vec![0.1f32, 0.9, 0.5, -1.0, 3.0, 0.0];
    let a = temperature_sample(&logits, 0.7);
    let b = temperature_sample(&logits, 0.7);
    assert_eq!(a, b, "fixed-seed sample must be reproducible");
}

#[test]
fn bos_and_eos_ids_match_generation_config() {
    assert_eq!(BOS_TOKEN_ID, 124894);
    assert_eq!(EOS_TOKEN_ID, 124900);
    assert_eq!(VOCAB_SIZE, 128000);
}

#[test]
fn validate_io_accepts_matching_graph() {
    let io = LlmIo::new().build();
    let mut inputs = vec![
        "input_ids".to_string(),
        "attention_mask".to_string(),
        "position_ids".to_string(),
    ];
    for l in io.conv_layers.clone() {
        inputs.push(format!("past_conv.{l}"));
    }
    for l in io.attention_layers.clone() {
        inputs.push(format!("past_key_values.{l}.key"));
        inputs.push(format!("past_key_values.{l}.value"));
    }
    let outputs = vec!["logits".to_string()];
    validate_io(&io, &inputs, &outputs).expect("matching graph validates");
}

#[test]
fn validate_io_rejects_missing_input_loudly() {
    let io = LlmIo::new().build();
    let inputs = vec!["input_ids".to_string()];
    let outputs = vec!["logits".to_string()];
    assert!(validate_io(&io, &inputs, &outputs).is_err());
}

#[test]
fn validate_io_rejects_missing_output_loudly() {
    let io = LlmIo::new().build();
    let mut inputs = vec![
        "input_ids".to_string(),
        "attention_mask".to_string(),
        "position_ids".to_string(),
    ];
    for l in io.conv_layers.clone() {
        inputs.push(format!("past_conv.{l}"));
    }
    for l in io.attention_layers.clone() {
        inputs.push(format!("past_key_values.{l}.key"));
        inputs.push(format!("past_key_values.{l}.value"));
    }
    assert!(validate_io(&io, &inputs, &[]).is_err());
}

// ── ROADMAP M2: the hermetic session stub + throwaway-generate regression ──

/// A hermetic session stub (no ort graph): echoes the input KV back as
/// `present_*` outputs — with per-position markers derived from the prompt
/// / decode tokens — and emits `logits` whose argmax is a deterministic
/// function of the input ids. This exercises the context/pool decode
/// mechanics (context interleaving, the rolling window, throwaway
/// `generate`) without a model. Shared with the pool tests (`pub(crate)`).
pub(crate) struct StubRun {
    io: LlmIo,
    /// The token the prefill's last-position logits select (the first
    /// decoded token).
    pub first_token: u32,
}

impl StubRun {
    pub(crate) fn new(io: LlmIo) -> Self {
        Self {
            io,
            first_token: 42,
        }
    }

    /// The next-token target for a run: a decode (single id) advances by
    /// one; a prefill (multi-id) selects `first_token`.
    fn target_for(&self, ids: &[i64]) -> i64 {
        if ids.len() == 1 {
            ids[0] + 1
        } else {
            i64::from(self.first_token)
        }
    }
}

impl OnnxRun for StubRun {
    fn run(
        &self,
        inputs: Vec<(String, Value)>,
    ) -> Result<Vec<(String, Value)>, OrtError> {
        let mut ids: Vec<i64> = Vec::new();
        let mut mask_len: usize = 0;
        let mut past_kv: BTreeMap<usize, (Vec<f32>, Vec<f32>)> = BTreeMap::new();
        let mut conv_past: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for (name, value) in &inputs {
            if name == &self.io.input_ids {
                if let Ok((_, data)) = value.try_extract_tensor::<i64>() {
                    ids = data.to_vec();
                }
            } else if name == &self.io.attention_mask {
                if let Ok((_, data)) = value.try_extract_tensor::<i64>() {
                    mask_len = data.len();
                }
            } else if let Some(rest) = name.strip_prefix(&self.io.past_key_values) {
                let parts: Vec<&str> = rest.split('.').collect();
                if parts.len() == 3 {
                    if let Ok(layer) = parts[1].parse::<usize>() {
                        if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                            let entry =
                                past_kv.entry(layer).or_insert((Vec::new(), Vec::new()));
                            if parts[2] == "key" {
                                entry.0 = data.to_vec();
                            } else if parts[2] == "value" {
                                entry.1 = data.to_vec();
                            }
                        }
                    }
                }
            } else if let Some(rest) = name.strip_prefix(&self.io.conv_state) {
                if let Ok(layer) = rest.parse::<usize>() {
                    if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                        conv_past.insert(layer, data.to_vec());
                    }
                }
            }
        }
        if ids.is_empty() {
            return Err(OrtError::Other("stub: no input_ids in run".into()));
        }
        let mask_len = if mask_len == 0 { ids.len() } else { mask_len };
        let heads = self.io.num_key_value_heads;
        let head_dim = self.io.head_dim;
        let target = self.target_for(&ids);

        let mut outputs: Vec<(String, Value)> = Vec::new();
        // logits: [1, ids.len(), vocab] with the target hot at the last
        // position — the session's logits length is the *input* sequence
        // length, and `extract_last_logits` derives the vocab window from
        // `seq`.
        let seq = ids.len().max(1);
        let mut logits = vec![f32::NEG_INFINITY; seq * VOCAB_SIZE];
        let last = (seq - 1) * VOCAB_SIZE;
        logits[last + target as usize] = 1.0;
        outputs.push((
            self.io.logits.clone(),
            f32tensor([1, seq, VOCAB_SIZE], logits)?,
        ));

        // present.{L}.key/.value: echo the input past and append the new
        // positions (marker per position: the prompt token on prefill, the
        // decode token on decode). Row-major `[head][position][head_dim]`
        // — the same layout a real session emits, so `PastState::truncate`
        // and `build_inputs` see the shapes they assume. Each head's past
        // positions are kept and its new position(s) appended in place.
        for &layer in &self.io.attention_layers {
            let (past_key, past_value) = past_kv.get(&layer).cloned().unwrap_or_default();
            let past_positions = if past_key.is_empty() {
                0
            } else {
                past_key.len() / (heads * head_dim)
            };
            let append = mask_len.saturating_sub(past_positions);
            let mut new_key = Vec::with_capacity(heads * mask_len * head_dim);
            let mut new_value = Vec::with_capacity(heads * mask_len * head_dim);
            for h in 0..heads {
                let start = h * past_positions * head_dim;
                new_key.extend_from_slice(&past_key[start..start + past_positions * head_dim]);
                new_value
                    .extend_from_slice(&past_value[start..start + past_positions * head_dim]);
                for p in 0..append {
                    let marker = if past_positions == 0 {
                        ids[p] as f32
                    } else {
                        ids[0] as f32
                    };
                    new_key.extend_from_slice(&vec![marker; head_dim]);
                    new_value.extend_from_slice(&vec![marker * 2.0; head_dim]);
                }
            }
            outputs.push((
                format!("{}.{layer}.key", self.io.present),
                f32tensor([1, heads, mask_len, head_dim], new_key)?,
            ));
            outputs.push((
                format!("{}.{layer}.value", self.io.present),
                f32tensor([1, heads, mask_len, head_dim], new_value)?,
            ));
        }
        // present_conv.{L}: echo (a fixed-size sliding window).
        for &layer in &self.io.conv_layers {
            let data = conv_past.get(&layer).cloned().unwrap_or_else(|| {
                vec![0.0; self.io.hidden_size * self.io.conv_l_cache]
            });
            outputs.push((
                format!("{}.{layer}", self.io.present_conv),
                f32tensor([1, self.io.hidden_size, self.io.conv_l_cache], data)?,
            ));
        }
        Ok(outputs)
    }
}

/// A minimal whitespace-split `tokenizer.json` (mirrors `tokenizer.rs`'s
/// fixture) so a session can be built hermetically.
pub(crate) fn test_tokenizer() -> Arc<LfmTokenizer> {
    let dir = tempfile::tempdir().unwrap();
    let json = r###"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "[PAD]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
            {"id": 1, "content": "[UNK]", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": {"type": "Wordpiece", "prefix": "##", "cleanup": true, "handle_chinese_chars": true},
        "model": {
            "type": "WordPiece",
            "vocab": {
                "[PAD]": 0, "[UNK]": 1,
                "hello": 2, "world": 3, "the": 4, "cat": 5
            },
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100
        }
    }"###;
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, json).unwrap();
    LfmTokenizer::from_file(&path, 512).unwrap()
}

/// A session over the stub runner + the default `LlmIo` contract.
pub(crate) fn session_with_stub(first_token: u32) -> Arc<OrtLlmSession> {
    let io = LlmIo::new().build();
    let mut stub = StubRun::new(io.clone());
    stub.first_token = first_token;
    Arc::new(OrtLlmSession::new_with_runner(
        Arc::new(stub),
        test_tokenizer(),
        io,
        512,
    ))
}

#[test]
fn generate_throwaway_context_matches_sampler_sequence() {
    // The throwaway-context `generate` must run exactly the pre-refactor
    // loop shape: prefill → sampler → per-token decode, so the produced
    // token sequence is the one the (unchanged, pure) sampler dictates for
    // the stub's deterministic logits. Prefill selects `first_token` (42);
    // each decode step advances by one.
    let session = session_with_stub(42);
    let prompt = vec![100i64, 200, 300];
    let tokens = session
        .generate(&prompt, None, 5, &[EOS_TOKEN_ID], LlmParams::default())
        .expect("generate");
    assert_eq!(tokens, vec![42, 43, 44, 45, 46]);
}

#[test]
fn complete_on_context_persists_kv_across_calls_and_is_independent() {
    // The M6 context-bound chat entry: `complete_on_context` runs the same
    // template→decode flow on a *named* context, so its KV advances across
    // calls and does not leak into a sibling context.
    let session = session_with_stub(42);
    let cc = OnnxChatCompletion::new(session.clone());
    let msg = vec![ChatMessage {
        role: "user".into(),
        content: "hi".into(),
    }];
    let a = OnnxContext::new(
        "a".into(),
        OnnxContextProfile {
            group: "g".into(),
            n_ctx: 64,
            max_ctx: None,
            pinned: false,
            resume: false,
        },
    );
    let b = OnnxContext::new(
        "b".into(),
        OnnxContextProfile {
            group: "g".into(),
            n_ctx: 64,
            max_ctx: None,
            pinned: false,
            resume: false,
        },
    );
    cc.complete_on_context(&a, &msg, None, Some(2), LlmParams::default())
        .expect("a call 1");
    cc.complete_on_context(&a, &msg, None, Some(2), LlmParams::default())
        .expect("a call 2");
    cc.complete_on_context(&b, &msg, None, Some(1), LlmParams::default())
        .expect("b call 1");

    let a_seq = a.past().expect("a has KV").seq_len;
    let b_seq = b.past().expect("b has KV").seq_len;
    // Two calls on `a` (each prefill + 2 decode) beat one call on `b`
    // (prefill + 1 decode), and the contexts never share KV.
    assert!(a_seq > b_seq, "a's KV advances across calls (a={a_seq}, b={b_seq})");
    assert!(b_seq >= 2, "b holds only its own call (b={b_seq})");
    assert_ne!(a_seq, b_seq, "contexts do not share KV");
}

#[test]
fn generate_stops_on_eos() {
    let session = session_with_stub(u32::try_from(EOS_TOKEN_ID - 1).expect("fits u32"));
    let prompt = vec![1i64, 2];
    let tokens = session
        .generate(&prompt, None, 100, &[EOS_TOKEN_ID], LlmParams::default())
        .expect("generate");
    // 124899 → decode → 124900 (EOS) → break before pushing the EOS.
    assert_eq!(tokens, vec![(EOS_TOKEN_ID - 1) as u32]);
}

#[test]
fn decode_step_before_prefill_is_a_loud_error() {
    let session = session_with_stub(42);
    let ctx = OnnxContext::new(
        "a".into(),
        OnnxContextProfile {
            group: "g".into(),
            n_ctx: 64,
            max_ctx: None,
            pinned: false,
            resume: false,
        },
    );
    let err = session.decode_step(&ctx, 7).expect_err("no prefill yet");
    assert!(err.to_string().contains("before any prefill"), "got {err}");
}

#[test]
fn prefill_stores_kv_into_the_context() {
    let session = session_with_stub(42);
    let ctx = OnnxContext::new(
        "a".into(),
        OnnxContextProfile {
            group: "g".into(),
            n_ctx: 64,
            max_ctx: None,
            pinned: false,
            resume: false,
        },
    );
    let logits = session.prefill(&ctx, &[1, 2, 3]).expect("prefill");
    assert_eq!(logits.len(), VOCAB_SIZE, "last-position logits");
    let past = ctx.past().expect("prefill stored KV");
    assert_eq!(past.seq_len, 3);
    // The stub marks each KV position with the prompt token id (head 0,
    // first head_dim element of the row-major [head][position] layout).
    let keys: Vec<f32> = (0..past.seq_len).map(|p| past.kv[&2].0[p * 64]).collect();
    assert_eq!(keys, vec![1.0, 2.0, 3.0]);
    assert!(ctx.last_used() > 0, "prefill touches the context");
}
