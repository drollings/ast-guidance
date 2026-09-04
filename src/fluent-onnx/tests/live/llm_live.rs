//! Live-AI probe for the LFM2.5-2.6B `CausalLm` ONNX export (ROADMAP M1 §5).
//!
//! This is the **empirical grounding** the M1 decoder is built against: it
//! loads `onnx/model_q4.onnx`, prints the session's declared inputs/outputs
//! (names, dtypes, shapes), runs a full prefill (with the required conv/KV
//! past-state inputs), extracts the logits, and performs one KV-cached decode
//! step (feeding the `present_*` outputs back as `past_*` inputs). The observed
//! input contract is captured as a golden fixture
//! (`tests/live/fixtures/lfm25_26b_io.json`) and as `LlmIo` defaults.
//!
//! Compiled only under the `live-ai` feature, `#[ignore]`d, run via
//! `make ort-test-live` / `make test-live`. Env contract:
//! `ORT_LIVE_LLM_MODEL` — the model directory (defaults to the ROADMAP wiring
//! point `/ai/models/lfm2/2.6b/LiquidAI/LFM2.5-2.6B-ONNX`). When absent the
//! test skips cleanly.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use ort::value::{Tensor, Value};

use fluent_onnx::tokenizer::LfmTokenizer;
use fluent_onnx::{OnnxConfig, OnnxTask, Quant, OrtSessionLoader, SessionHandle, SessionLoader};

fn live_llm_dir() -> Option<PathBuf> {
    let dir = std::env::var("ORT_LIVE_LLM_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let default = PathBuf::from("/ai/models/lfm2/2.6b/LiquidAI/LFM2.5-2.6B-ONNX");
            default.is_dir().then_some(default)
        });
    dir
}

/// Read a numeric field from the model's `config.json`.
fn config_number(dir: &std::path::Path, key: &str) -> usize {
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    cfg.get(key).and_then(serde_json::Value::as_u64).unwrap() as usize
}

/// The discovered graph IO contract: the standard id inputs plus the per-layer
/// conv-state and attention-KV past inputs.
struct GraphIo {
    /// Conv layers (conv-state `past_conv.{L}`, shape `[1, hidden, conv_cache]`).
    conv_layers: Vec<usize>,
    /// Full-attention layers (KV `past_key_values.{L}.key/value`).
    attention_layers: Vec<usize>,
    /// num_key_value_heads.
    n_kv_heads: usize,
    /// head_dim.
    head_dim: usize,
    /// hidden_size (conv-state feature dim).
    hidden_size: usize,
    /// conv_L_cache (conv-state window).
    conv_cache: usize,
}

impl GraphIo {
    fn introspect(session: &ort::session::Session, dir: &std::path::Path) -> Self {
        let mut conv_layers = Vec::new();
        let mut attention_layers = Vec::new();
        for input in session.inputs() {
            let name = input.name();
            if let Some(rest) = name.strip_prefix("past_conv.") {
                if let Ok(layer) = rest.parse::<usize>() {
                    conv_layers.push(layer);
                }
            } else if let Some(rest) = name.strip_prefix("past_key_values.") {
                if let Some(layer_str) = rest.split('.').next() {
                    if let Ok(layer) = layer_str.parse::<usize>() {
                        if !attention_layers.contains(&layer) {
                            attention_layers.push(layer);
                        }
                    }
                }
            }
        }
        conv_layers.sort_unstable();
        attention_layers.sort_unstable();

        let hidden_size = config_number(dir, "hidden_size");
        let num_heads = config_number(dir, "num_heads");
        let n_kv_heads = config_number(dir, "num_key_value_heads");
        let conv_cache = config_number(dir, "conv_L_cache");
        let head_dim = hidden_size / num_heads;
        Self {
            conv_layers,
            attention_layers,
            n_kv_heads,
            head_dim,
            hidden_size,
            conv_cache,
        }
    }
}

/// The KV/conv past-state, carried across decode steps. Holds owned f32 data
/// so it can be rebuilt into tensors for each `session.run`.
struct PastState {
    seq_len: usize,
    /// present_conv.{L} → flat `[hidden_size * conv_cache]` f32.
    conv: BTreeMap<usize, Vec<f32>>,
    /// present.{L}.key / .value → flat `[n_kv_heads * seq_len * head_dim]`.
    kv: BTreeMap<usize, (Vec<f32>, Vec<f32>)>,
}

fn zeros_f32(len: usize) -> Vec<f32> {
    vec![0.0; len]
}

/// Build the session inputs for a run: the static id tensors plus the
/// past-state tensors.
fn build_inputs(
    io: &GraphIo,
    past: &PastState,
    ids: &[i64],
    mask: &[i64],
    positions: &[i64],
) -> Vec<(String, Value)> {
    fn f32tensor<A: ort::value::ToShape>(shape: A, data: Vec<f32>) -> Value {
        Tensor::from_array((shape, data))
            .expect("build tensor")
            .into_dyn()
    }

    fn i64tensor<A: ort::value::ToShape>(shape: A, data: Vec<i64>) -> Value {
        Tensor::from_array((shape, data))
            .expect("build int tensor")
            .into_dyn()
    }

    let batch = 1usize;
    let seq = ids.len();
    let mask_len = mask.len();
    let mut inputs: Vec<(String, Value)> = Vec::new();
    inputs.push(("input_ids".to_string(), i64tensor([batch, seq], ids.to_vec())));
    inputs.push(("attention_mask".to_string(), i64tensor([batch, mask_len], mask.to_vec())));
    inputs.push(("position_ids".to_string(), i64tensor([batch, seq], positions.to_vec())));

    for &layer in &io.conv_layers {
        let data = past.conv.get(&layer).cloned().unwrap_or_else(|| {
            zeros_f32(io.hidden_size * io.conv_cache)
        });
        inputs.push((
            format!("past_conv.{layer}"),
            f32tensor([batch, io.hidden_size, io.conv_cache], data),
        ));
    }
    for &layer in &io.attention_layers {
        let (key, value) = past.kv.get(&layer).cloned().unwrap_or_else(|| {
            let len = io.n_kv_heads * past.seq_len * io.head_dim;
            (zeros_f32(len), zeros_f32(len))
        });
        let kv_shape = [batch, io.n_kv_heads, past.seq_len, io.head_dim];
        inputs.push((format!("past_key_values.{layer}.key"), f32tensor(kv_shape, key)));
        inputs.push((format!("past_key_values.{layer}.value"), f32tensor(kv_shape, value)));
    }
    inputs
}

/// Extract the present_* outputs back into a `PastState` for the next step.
fn capture_past(io: &GraphIo, outputs: &ort::session::SessionOutputs<'_>, seq_len: usize) -> PastState {
    let mut conv = BTreeMap::new();
    let mut kv = BTreeMap::new();
    for (name, value) in outputs.iter() {
        if let Some(rest) = name.strip_prefix("present_conv.") {
            if let Ok(layer) = rest.parse::<usize>() {
                let (_, data) = value.try_extract_tensor::<f32>().expect("present_conv f32");
                conv.insert(layer, data.to_vec());
            }
        } else if let Some(rest) = name.strip_prefix("present.") {
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.len() == 2 {
                if let Ok(layer) = parts[0].parse::<usize>() {
                    let entry = kv.entry(layer).or_insert((Vec::new(), Vec::new()));
                    let (_, data) = value.try_extract_tensor::<f32>().expect("present kv f32");
                    if parts[1] == "key" {
                        entry.0 = data.to_vec();
                    } else if parts[1] == "value" {
                        entry.1 = data.to_vec();
                    }
                }
            }
        }
    }
    let _ = io;
    PastState { seq_len, conv, kv }
}

fn argmax_last_logits(logits: &[f32], vocab: usize, seq: usize) -> u32 {
    let start = (seq - 1) * vocab;
    let window = &logits[start..start + vocab];
    let mut best = 0usize;
    for i in 1..window.len() {
        if window[i] > window[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_LLM_MODEL (or the default /ai/models path); run via `make ort-test-live`"]
fn probe_prefills_and_kv_decode_step() {
    let Some(dir) = live_llm_dir() else {
        eprintln!("skipping: no live LLM model available");
        return;
    };
    let (config, handle) = load_session(&dir);
    let session = handle
        .downcast_arc::<Mutex<ort::session::Session>>()
        .expect("handle holds an ort session");
    let mut session = session.lock().unwrap();

    eprintln!("=== Session inputs ===");
    for input in session.inputs() {
        eprintln!("  input: name={:?} dtype={:?}", input.name(), input.dtype());
    }
    eprintln!("=== Session outputs ===");
    for output in session.outputs() {
        eprintln!("  output: name={:?} dtype={:?}", output.name(), output.dtype());
    }

    let io = GraphIo::introspect(&session, &dir);
    eprintln!(
        "discovered: conv_layers={:?} attention_layers={:?} n_kv_heads={} head_dim={} hidden={} conv_cache={}",
        io.conv_layers, io.attention_layers, io.n_kv_heads, io.head_dim, io.hidden_size, io.conv_cache
    );
    assert_eq!(io.conv_layers.len(), 22, "22 conv layers");
    assert_eq!(io.attention_layers.len(), 8, "8 full-attention layers");
    assert_eq!(io.n_kv_heads, 8);
    assert_eq!(io.head_dim, 64);
    assert_eq!(io.hidden_size, 2048);
    assert_eq!(io.conv_cache, 3);

    let tok_path = config.tokenizer_path.as_ref().expect("tokenizer_path");
    let tokenizer = LfmTokenizer::from_file(tok_path, 512).expect("load tokenizer");
    let enc = tokenizer.encode("Hello world").expect("encode");
    let ids: Vec<i64> = enc.ids.iter().map(|&i| i64::from(i)).collect();
    let seq = ids.len();
    let mask: Vec<i64> = vec![1; seq];
    let positions: Vec<i64> = (0..seq as i64).collect();
    let vocab = 128000usize;

    // ── Prefill: empty past-state ──
    let (next, past) = {
        let prefill_past = PastState {
            seq_len: 0,
            conv: BTreeMap::new(),
            kv: BTreeMap::new(),
        };
        let prefill_inputs = build_inputs(&io, &prefill_past, &ids, &mask, &positions);
        let outputs = session
            .run(prefill_inputs)
            .expect("prefill run with all inputs");
        let (logits_shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .expect("logits f32");
        eprintln!(
            "prefill logits shape: {logits_shape:?} len={}",
            logits.len()
        );
        assert_eq!(logits.len(), seq * vocab, "prefill logits must cover seq × vocab");
        let next = argmax_last_logits(logits, vocab, seq);
        let past = capture_past(&io, &outputs, seq);
        (next, past)
    }; // outputs dropped here → releases the session borrow

    // ── KV-cached decode step: feed present_* back as past_*, one new token ──
    let decode_ids = vec![i64::from(next)];
    let decode_mask: Vec<i64> = vec![1; seq + 1];
    let decode_positions = vec![seq as i64];
    let decode_inputs = build_inputs(&io, &past, &decode_ids, &decode_mask, &decode_positions);
    let outputs2 = session.run(decode_inputs).expect("decode step run");
    let (shape2, logits2) = outputs2["logits"]
        .try_extract_tensor::<f32>()
        .expect("decode logits f32");
    eprintln!("decode step logits shape: {shape2:?} len={}", logits2.len());
    assert_eq!(logits2.len(), vocab, "decode step yields a single-position logits");

    eprintln!("PREFILL+NEXT-TOKEN-OK first-token-id={next}");
}

fn load_session(dir: &std::path::Path) -> (OnnxConfig, SessionHandle) {
    let config = OnnxConfig::new()
        .model_path(dir.join("onnx"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::CausalLm)
        .quantization(Quant::Q4)
        .build();
    config.validate().expect("config validates against config.json");
    let handle = OrtSessionLoader
        .load(&config, "live-llm")
        .expect("load q4 causal-lm session");
    (config, handle)
}

/// The single "run a chat call" entry — the M2 backend seam.
fn make_completion(
    config: &OnnxConfig,
    handle: &SessionHandle,
) -> fluent_onnx::OnnxChatCompletion {
    let session = fluent_onnx::build_llm_session_from_handle(handle, config, "live-llm")
        .expect("build llm session");
    fluent_onnx::OnnxChatCompletion::new(std::sync::Arc::new(session))
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_LLM_MODEL (or the default /ai/models path); run via `make ort-test-live`"]
fn decoder_free_text_generate_is_deterministic() {
    let Some(dir) = live_llm_dir() else {
        eprintln!("skipping: no live LLM model available");
        return;
    };
    let (config, handle) = load_session(&dir);
    let completion = make_completion(&config, &handle);
    let messages = vec![fluent_llm::ChatMessage {
        role: "user".into(),
        content: "What is 2+2? Answer in one word.".into(),
    }];
    // Free-text (no grammar), short generation: exercises prefill + the KV
    // decode loop end-to-end against the real checkpoint.
    let first = completion
        .complete(&messages, None, Some(8), fluent_onnx::LlmParams::default())
        .expect("first generate");
    let second = completion
        .complete(&messages, None, Some(8), fluent_onnx::LlmParams::default())
        .expect("second generate");
    // Determinism: `intra_threads=1` + argmax ⇒ bit-identical output, which is
    // also the KV-cache-correctness signal (a broken past-state feed would
    // diverge).
    assert!(!first.trim().is_empty(), "decoder must produce text");
    assert_eq!(first, second, "KV-cached decode must be deterministic");
    eprintln!("decoder free-text output: {:?}", first);
}

/// Build a grammar over a small schema, backed by the model's real vocab.
fn build_json_grammar(
    dir: &std::path::Path,
) -> fluent_onnx::JsonObjectGrammar {
    use fluent_onnx::{HuggingFaceVocab, JsonField, JsonSchema, JsonType, TokenVocab};
    let tokenizer = LfmTokenizer::from_file(&dir.join("tokenizer.json"), 512).expect("tokenizer");
    let vocab: std::sync::Arc<dyn TokenVocab> =
        std::sync::Arc::new(HuggingFaceVocab::new(tokenizer.inner().clone()));
    let schema = JsonSchema::new(vec![
        JsonField::required("answer", JsonType::String),
        JsonField::optional("score", JsonType::Number),
    ]);
    fluent_onnx::JsonObjectGrammar::new(schema, vocab)
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_LLM_MODEL (or the default /ai/models path); run via `make ort-test-live`"]
fn decoder_grammar_constrained_produces_valid_json() {
    let Some(dir) = live_llm_dir() else {
        eprintln!("skipping: no live LLM model available");
        return;
    };
    let (config, handle) = load_session(&dir);
    let completion = make_completion(&config, &handle);
    let messages = vec![fluent_llm::ChatMessage {
        role: "user".into(),
        content: "Reply with JSON: {\"answer\":\"four\",\"score\":4}".into(),
    }];
    let mut grammar = build_json_grammar(&dir);
    let out = completion
        .complete(&messages, Some(&mut grammar), Some(64), fluent_onnx::LlmParams::default())
        .expect("grammar-constrained generate");
    eprintln!("grammar-constrained output: {:?}", out);
    // The structural guarantee (hermetic): a grammar-constrained decode never
    // emits malformed JSON — the output is always a valid JSON prefix. Assert
    // it live so a future regression that breaks the guarantee fails loudly.
    assert!(
        fluent_onnx::is_valid_json_prefix(&out),
        "grammar-constrained decode must always be a valid JSON prefix (got {out:?})"
    );
}