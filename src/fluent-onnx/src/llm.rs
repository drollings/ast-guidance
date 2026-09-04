//! The generative `CausalLm` decoder — `OrtLlmSession` (ROADMAP M1/M2, feature
//! `onnx`).
//!
//! This is the one non-encoder path in `fluent-onnx`: an autoregressive
//! decoder over the LFM2.5-2.6B `Lfm2ForCausalLM` export, with KV + conv-state
//! caching and grammar-constrained sampling. It is `spacy-rs`/`guidance`/
//! `coral`-free: it consumes `&[ChatMessage]`-shaped strings and a `Grammar`,
//! and the router wires it behind `fluent_llm::client::ChatBackend` (M2).
//!
//! The IO contract (input/output tensor names, KV + conv-state layout) lives in
//! [`crate::config::LlmIo`], which defaults to the empirically-discovered golden
//! fixture (`tests/live/fixtures/lfm25_26b_io.json`). The decoder introspects
//! the graph at load when `LlmIo` is `None`; a provided `LlmIo` is validated
//! against the graph and a mismatch is a loud load error.
//!
//! ## Decode
//!
//! 1. `apply_chat_template` renders the chat turns (LFM2.5 `chat_template.jinja`
//!    shape), then tokenization produces `prompt_ids` with a leading `bos`.
//! 2. **Prefill** ([`OrtLlmSession::prefill`]): `input_ids`/`attention_mask`/
//!    `position_ids` over the whole prompt, plus zero `past_conv.{L}` and empty
//!    `past_key_values.{L}` → `logits [1, S, vocab]` + `present_*` outputs. The
//!    resulting KV is stored into the caller's [`crate::context::OnnxContext`].
//! 3. **Per-token decode** ([`OrtLlmSession::decode_step`]): feed the previous
//!    `present_*` back as `past_*` from the context, mask the last-position
//!    logits with `Grammar::allowed_ids`, argmax (or temperature-sample) among
//!    the allowed ids, `grammar.advance`, stop on EOS/max-tokens. Returns the
//!    decoded token ids.
//!
//! [`OrtLlmSession::generate`] is a thin wrapper over the two entry points that
//! allocates a throwaway context — byte-identical output for every existing
//! caller, with the KV now owned by the context rather than a per-call local.
//! The KV lives per-context (ROADMAP M2), so named contexts interleave on one
//! loaded session without sharing state.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use fluent_llm::ChatMessage;
use ort::value::{Tensor, Value};

use crate::config::{LlmIo, OnnxConfig, OnnxTask};
use crate::context::{OnnxContext, OnnxContextProfile, PastState};
use crate::error::OrtError;
use crate::grammar::Grammar;
use crate::session::{OrtSessionRegistry, SessionHandle};
use crate::tokenizer::LfmTokenizer;

/// The LFM2.5 vocabulary size.
pub const VOCAB_SIZE: usize = 128000;

/// The LFM2.5 `bos` token id (from `generation_config.json`).
pub const BOS_TOKEN_ID: i64 = 124894;

/// The LFM2.5 `eos` token id (from `generation_config.json`).
pub const EOS_TOKEN_ID: i64 = 124900;

/// Sampling parameters for a decode run. `do_sample=false` (or temperature 0)
/// → greedy argmax (the determinism default); otherwise temperature sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlmParams {
    /// Sampling temperature (default `0.1`; `0.0` → argmax).
    pub temperature: f32,
    /// Whether to sample from the softmax rather than argmax.
    pub do_sample: bool,
}

impl Default for LlmParams {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            do_sample: false,
        }
    }
}

impl LlmParams {
    /// The effective deterministic flag: `!do_sample || temperature == 0.0`.
    pub fn is_deterministic(self) -> bool {
        !self.do_sample || self.temperature == 0.0
    }
}

/// The minimal "run the graph" seam (ROADMAP M2). The real impl wraps an
/// `ort::session::Session`; hermetic tests inject a stub that returns canned
/// owned outputs, so the context/pool decode mechanics (context interleaving,
/// the rolling window) are testable without a model. Inputs/outputs are owned
/// `ort::value::Value` (the type-erased `DynValue`) — the same shape a
/// `session.run` accepts and returns.
pub(crate) trait OnnxRun: Send + Sync {
    fn run(
        &self,
        inputs: Vec<(String, Value)>,
    ) -> Result<Vec<(String, Value)>, OrtError>;
}

/// Real session-backed runner: serializes every run on the shared
/// `Mutex<Session>` (one loaded weights instance, N contexts).
struct RealSessionRun {
    session: Arc<Mutex<ort::session::Session>>,
}

impl OnnxRun for RealSessionRun {
    fn run(
        &self,
        inputs: Vec<(String, Value)>,
    ) -> Result<Vec<(String, Value)>, OrtError> {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = session
            .run(inputs)
            .map_err(|e| OrtError::Other(format!("session run: {e}")))?;
        Ok(outputs
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect())
    }
}

/// The generative decoder: an ort session run seam, a tokenizer, and the
/// resolved [`LlmIo`] contract. The KV state lives per-context (ROADMAP M2) —
/// `prefill`/`decode_step` read and advance a caller's [`OnnxContext`].
pub struct OrtLlmSession {
    runner: Arc<dyn OnnxRun>,
    tokenizer: Arc<LfmTokenizer>,
    io: LlmIo,
    /// Default max generated tokens (the config's `max_gen_tokens`).
    max_gen_tokens: usize,
}

/// Tensor-builder helpers (monomorphized).
fn f32tensor<A: ort::value::ToShape>(shape: A, data: Vec<f32>) -> Result<Value, OrtError> {
    Tensor::from_array((shape, data))
        .map(Value::into_dyn)
        .map_err(|e| OrtError::Other(format!("build tensor: {e}")))
}

fn i64tensor<A: ort::value::ToShape>(shape: A, data: Vec<i64>) -> Result<Value, OrtError> {
    Tensor::from_array((shape, data))
        .map(Value::into_dyn)
        .map_err(|e| OrtError::Other(format!("build int tensor: {e}")))
}

impl OrtLlmSession {
    /// Build the session inputs for a single `run`: the id tensors plus the
    /// past-state tensors.
    fn build_inputs(
        &self,
        past: &PastState,
        ids: &[i64],
        mask: &[i64],
        positions: &[i64],
    ) -> Result<Vec<(String, Value)>, OrtError> {
        let batch = 1usize;
        let seq = ids.len();
        let mask_len = mask.len();
        let io = &self.io;
        let mut inputs = Vec::with_capacity(3 + io.conv_layers.len() + io.attention_layers.len() * 2);
        inputs.push((io.input_ids.clone(), i64tensor([batch, seq], ids.to_vec())?));
        inputs.push((
            io.attention_mask.clone(),
            i64tensor([batch, mask_len], mask.to_vec())?,
        ));
        inputs.push((io.position_ids.clone(), i64tensor([batch, seq], positions.to_vec())?));

        for &layer in &io.conv_layers {
            let data = past
                .conv
                .get(&layer)
                .cloned()
                .unwrap_or_else(|| vec![0.0; io.hidden_size * io.conv_l_cache]);
            inputs.push((
                format!("{}.{layer}", io.conv_state),
                f32tensor([batch, io.hidden_size, io.conv_l_cache], data)?,
            ));
        }
        for &layer in &io.attention_layers {
            let (key, value) = past.kv.get(&layer).cloned().unwrap_or_else(|| {
                let len = io.num_key_value_heads * past.seq_len * io.head_dim;
                (vec![0.0; len], vec![0.0; len])
            });
            let kv_shape = [batch, io.num_key_value_heads, past.seq_len, io.head_dim];
            inputs.push((
                format!("{}.{layer}.key", io.past_key_values),
                f32tensor(kv_shape, key)?,
            ));
            inputs.push((
                format!("{}.{layer}.value", io.past_key_values),
                f32tensor(kv_shape, value)?,
            ));
        }
        Ok(inputs)
    }

    /// Extract the `present_*` outputs back into a `PastState` for the next
    /// step.
    fn capture_past(&self, outputs: &[(String, Value)], seq_len: usize) -> PastState {
        let io = &self.io;
        let mut conv = BTreeMap::new();
        let mut kv = BTreeMap::new();
        for (name, value) in outputs {
            if let Some(rest) = name.strip_prefix(&io.present_conv) {
                if let Some(rest) = rest.strip_prefix('.') {
                    if let Ok(layer) = rest.parse::<usize>() {
                        if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                            conv.insert(layer, data.to_vec());
                        }
                    }
                }
            } else if let Some(rest) = name.strip_prefix(&io.present) {
                // `present.{L}.key` / `present.{L}.value`
                let parts: Vec<&str> = rest.split('.').collect();
                if parts.len() == 3 {
                    if let Ok(layer) = parts[1].parse::<usize>() {
                        if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                            let entry = kv.entry(layer).or_insert((Vec::new(), Vec::new()));
                            if parts[2] == "key" {
                                entry.0 = data.to_vec();
                            } else if parts[2] == "value" {
                                entry.1 = data.to_vec();
                            }
                        }
                    }
                }
            }
        }
        PastState { seq_len, conv, kv }
    }

    /// The last-position logits (vocab-length) from a run's `logits` output.
    fn extract_last_logits(
        &self,
        outputs: &[(String, Value)],
        seq: usize,
    ) -> Result<Vec<f32>, OrtError> {
        let (shape, logits) = outputs
            .iter()
            .find(|(name, _)| name == &self.io.logits)
            .map(|(_, v)| v.try_extract_tensor::<f32>())
            .transpose()
            .map_err(|e| OrtError::Other(format!("extract logits: {e}")))?
            .ok_or_else(|| {
                OrtError::Other(format!("no `{}` output in session run", self.io.logits))
            })?;
        let vocab = shape.num_elements() / seq.max(1);
        let start = (seq - 1) * vocab;
        let window = &logits[start..start + vocab];
        Ok(window.to_vec())
    }

    /// Tokenize a chat render into prompt ids (with the leading `bos`).
    fn tokenize_prompt(&self, text: &str) -> Result<Vec<i64>, OrtError> {
        let enc = self.tokenizer.encode(text)?;
        let mut ids: Vec<i64> = enc.ids.iter().map(|&i| i64::from(i)).collect();
        ids.insert(0, BOS_TOKEN_ID);
        Ok(ids)
    }

    /// Prefill a context: run the prompt through the graph once and store the
    /// resulting KV into `ctx`. Returns the last-position logits (the first
    /// decode step's `current_logits`). A prompt longer than the context window
    /// drops its oldest KV positions (rolling window) so the invariant
    /// `seq_len <= n_ctx` holds everywhere; the truncation is logged.
    pub fn prefill(&self, ctx: &OnnxContext, prompt_ids: &[i64]) -> Result<Vec<f32>, OrtError> {
        let seq = prompt_ids.len().max(1);
        let mask: Vec<i64> = vec![1; seq];
        let positions: Vec<i64> = (0..seq as i64).collect();
        let prefill_past = PastState {
            seq_len: 0,
            conv: BTreeMap::new(),
            kv: BTreeMap::new(),
        };
        let inputs = self.build_inputs(&prefill_past, prompt_ids, &mask, &positions)?;
        let outputs = self.runner.run(inputs)?;
        let logits = self.extract_last_logits(&outputs, seq)?;
        let mut past = self.capture_past(&outputs, seq);
        let n_ctx = ctx.n_ctx() as usize;
        if past.seq_len > n_ctx {
            let dropped =
                past.truncate(n_ctx, self.io.num_key_value_heads, self.io.head_dim);
            tracing::warn!(
                target: "fluent-onnx",
                context = %ctx.name(),
                dropped,
                "prefill exceeded n_ctx - truncating oldest KV (rolling window)",
            );
        }
        ctx.store_past(past);
        ctx.touch();
        Ok(logits)
    }

    /// One single-token decode step on `ctx`: feed the previous token's id and
    /// the context's stored KV, advance the KV in the context, and return the
    /// next position's logits. A step that would grow the KV past `n_ctx`
    /// truncates the oldest positions first (rolling window, logged). The
    /// context must hold a prefill (or an earlier decode step) — a step before
    /// any prefill is a loud error.
    pub fn decode_step(&self, ctx: &OnnxContext, next: u32) -> Result<Vec<f32>, OrtError> {
        let mut past = ctx.past().ok_or_else(|| {
            OrtError::Other(format!(
                "decode_step on context {} before any prefill",
                ctx.name()
            ))
        })?;
        let n_ctx = ctx.n_ctx() as usize;
        // A decode that would exceed the window drops the oldest KV positions
        // so the window stays at n_ctx (rolling window).
        if past.seq_len.saturating_add(1) > n_ctx {
            let dropped = past.truncate(
                n_ctx.saturating_sub(1),
                self.io.num_key_value_heads,
                self.io.head_dim,
            );
            tracing::warn!(
                target: "fluent-onnx",
                context = %ctx.name(),
                dropped,
                "decode reached n_ctx - truncating oldest KV (rolling window)",
            );
        }
        let total = past.seq_len + 1;
        let ids = vec![i64::from(next)];
        let mask: Vec<i64> = vec![1; total];
        let positions = vec![(total - 1) as i64];
        let inputs = self.build_inputs(&past, &ids, &mask, &positions)?;
        let outputs = self.runner.run(inputs)?;
        let logits = self.extract_last_logits(&outputs, 1)?;
        let new_past = self.capture_past(&outputs, total);
        ctx.store_past(new_past);
        ctx.touch();
        Ok(logits)
    }

    /// Decode a full generation: chat-template render → tokenize → prefill →
    /// grammar-constrained decode loop. Returns the generated token ids.
    ///
    /// A thin wrapper over [`OrtLlmSession::prefill`]/[`OrtLlmSession::decode_step`]
    /// with a **throwaway context** — the KV is owned by the context rather
    /// than a per-call local, but the context is dropped on return, so the
    /// output is byte-identical to the pre-context loop for every caller.
    pub fn generate(
        &self,
        prompt_ids: &[i64],
        grammar: Option<&mut (dyn Grammar + 'static)>,
        max_tokens: usize,
        stop_ids: &[i64],
        params: LlmParams,
    ) -> Result<Vec<u32>, OrtError> {
        // Throwaway-context decode: a fresh context that is dropped at the end
        // (the single-shot path — byte-identical to the pre-M6 `generate`).
        let ctx = OnnxContext::new(
            "__one_shot__".into(),
            OnnxContextProfile {
                group: "scratch".into(),
                n_ctx: u64::MAX,
                max_ctx: None,
                pinned: false,
                resume: false,
            },
        );
        self.generate_on_context(&ctx, prompt_ids, grammar, max_tokens, stop_ids, params)
    }

    /// Decode onto a **named** context (ROADMAP M6): the same token-selection
    /// loop as [`Self::generate`], but the KV lives in `ctx` and persists
    /// between calls — the "one weights load, N named context windows" decode
    /// the onnx fleet gains. The context's `n_ctx` rolling window is honored.
    pub fn generate_on_context(
        &self,
        ctx: &OnnxContext,
        prompt_ids: &[i64],
        mut grammar: Option<&mut (dyn Grammar + 'static)>,
        max_tokens: usize,
        stop_ids: &[i64],
        params: LlmParams,
    ) -> Result<Vec<u32>, OrtError> {
        let mut current_logits = self.prefill(ctx, prompt_ids)?;

        let mut tokens: Vec<u32> = Vec::with_capacity(max_tokens);
        loop {
            if tokens.len() >= max_tokens {
                break;
            }
            let allowed: Option<Vec<u32>> = grammar.as_deref().map(|g| g.allowed_ids(VOCAB_SIZE));
            if let Some(ref allowed) = allowed {
                if allowed.is_empty() {
                    break;
                }
            }
            let allowed_set = allowed.map(|a| a.into_iter().collect::<HashSet<u32>>());
            let next = sample_next_token(&current_logits, allowed_set.as_ref(), params);
            if stop_ids.contains(&i64::from(next)) {
                break;
            }
            tokens.push(next);
            if let Some(g) = grammar.as_deref_mut() {
                g.advance(next);
            }
            current_logits = self.decode_step(ctx, next)?;
        }
        Ok(tokens)
    }

    /// Decode the generated token ids back into text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, OrtError> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        self.tokenizer.decode(tokens)
    }

    /// The resolved IO contract (introspected or validated).
    pub fn io(&self) -> &LlmIo {
        &self.io
    }

    /// The underlying tokenizer (for vocab introspection — the router builds a
    /// [`crate::grammar::TokenVocab`] from it so grammar-constrained decodes use
    /// the same vocabulary the session was built with).
    pub fn tokenizer(&self) -> &LfmTokenizer {
        &self.tokenizer
    }

    /// The config's default max generated tokens.
    pub fn max_gen_tokens(&self) -> usize {
        self.max_gen_tokens
    }
}

/// Render a chat turn list into the LFM2.5 chat-template text (the
/// `chat_template.jinja` shape, without the `bos` token which is prepended at
/// tokenization). The generation prompt is `<|im_start|>assistant\n`.
///
/// This is a pure, hermetic function — the LFM2.5 template's message loop is
/// the only part the decoder needs (no tools, no thinking blocks for the
/// structured-generation path).
pub fn apply_chat_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Pick the next token from a vocab-length logits slice: argmax, or a
/// temperature sample among the grammar-allowed ids. Pure + hermetic.
pub fn sample_next_token(
    logits: &[f32],
    allowed: Option<&HashSet<u32, std::collections::hash_map::RandomState>>,
    params: LlmParams,
) -> u32 {
    if let Some(allowed) = allowed {
        let mut best_logit = f32::NEG_INFINITY;
        let mut best = 0u32;
        for &id in allowed {
            let id = id as usize;
            if id < logits.len() && logits[id] > best_logit {
                best_logit = logits[id];
                best = id as u32;
            }
        }
        if best_logit.is_finite() {
            return best;
        }
    }
    if params.is_deterministic() {
        argmax(logits)
    } else {
        temperature_sample(logits, params.temperature)
    }
}

/// Global argmax over the logits.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Temperature softmax over the finite logits, then a fixed-seed sample (so a
/// fixed temperature is reproducible in hermetic tests).
pub fn temperature_sample(logits: &[f32], temperature: f32) -> u32 {
    let inv = 1.0 / temperature.max(1e-5);
    let scaled: Vec<f64> = logits.iter().map(|&l| f64::from(l * inv)).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = scaled.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return argmax(logits);
    }
    for w in &mut weights {
        *w /= total;
    }
    // Fixed-seed LCG for reproducibility.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r: f64 = ((state >> 33) as f64) / (1u64 << 31) as f64;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += *w;
        if r <= acc {
            return i as u32;
        }
    }
    (weights.len() - 1) as u32
}

/// The single "run a chat call" entry the router's backend wraps (M2): render
/// the chat template, tokenize, then grammar-constrained decode to text.
pub struct OnnxChatCompletion {
    session: Arc<OrtLlmSession>,
}

impl OnnxChatCompletion {
    /// Wrap a decoder session.
    pub fn new(session: Arc<OrtLlmSession>) -> Self {
        Self { session }
    }

    /// Run a chat call: `complete(messages, grammar, max_tokens, params)`.
    /// `grammar` is a `'static` trait object (built fresh per call by the
    /// router's backend from a `response_format.schema`); `None` is free text.
    pub fn complete(
        &self,
        messages: &[ChatMessage],
        grammar: Option<&mut (dyn Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: LlmParams,
    ) -> Result<String, OrtError> {
        self.complete_inner(None, messages, grammar, max_tokens, params)
    }

    /// Run a chat call onto a **named** context (ROADMAP M6): the same
    /// template → tokenize → decode flow as [`Self::complete`], but the KV
    /// lives in and advances `ctx`, so a follow-up call on the same context
    /// continues from where the previous one stopped. `ctx` must belong to the
    /// session's pool.
    pub fn complete_on_context(
        &self,
        ctx: &OnnxContext,
        messages: &[ChatMessage],
        grammar: Option<&mut (dyn Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: LlmParams,
    ) -> Result<String, OrtError> {
        self.complete_inner(Some(ctx), messages, grammar, max_tokens, params)
    }

    fn complete_inner(
        &self,
        ctx: Option<&OnnxContext>,
        messages: &[ChatMessage],
        grammar: Option<&mut (dyn Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: LlmParams,
    ) -> Result<String, OrtError> {
        let prompt_text = apply_chat_template(messages);
        let prompt_ids = self.session.tokenize_prompt(&prompt_text)?;
        let max_tokens = max_tokens.unwrap_or(self.session.max_gen_tokens());
        let tokens = match ctx {
            Some(ctx) => self.session.generate_on_context(
                ctx,
                &prompt_ids,
                grammar,
                max_tokens,
                &[EOS_TOKEN_ID],
                params,
            )?,
            None => self.session.generate(
                &prompt_ids,
                grammar,
                max_tokens,
                &[EOS_TOKEN_ID],
                params,
            )?,
        };
        self.session.decode(&tokens)
    }
}

/// Introspect the graph's declared inputs/outputs into an [`LlmIo`] contract.
/// Layer lists are derived from the input names; the fixed dims come from the
/// model's `config.json`. Called when a config declares `llm_io: None`.
fn introspect_io(session: &ort::session::Session, config: &OnnxConfig) -> Result<LlmIo, OrtError> {
    let mut conv_layers = Vec::new();
    let mut attention_layers = Vec::new();
    let mut input_names = Vec::new();
    let mut output_names = Vec::new();
    for input in session.inputs() {
        let name = input.name();
        input_names.push(name.to_string());
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
    for output in session.outputs() {
        output_names.push(output.name().to_string());
    }
    conv_layers.sort_unstable();
    attention_layers.sort_unstable();

    let (hidden_size, num_heads, n_kv_heads, conv_cache) = config_json_dims(config)?;
    let head_dim = hidden_size.checked_div(num_heads).unwrap_or(64);

    let mut io = LlmIo::new()
        .hidden_size(hidden_size)
        .conv_l_cache(conv_cache)
        .num_key_value_heads(n_kv_heads)
        .head_dim(head_dim)
        .attention_layers(attention_layers)
        .conv_layers(conv_layers)
        .build();

    if input_names.iter().any(|n| n == "input_ids") {
        io.input_ids = "input_ids".to_string();
    }
    if input_names.iter().any(|n| n == "attention_mask") {
        io.attention_mask = "attention_mask".to_string();
    }
    if input_names.iter().any(|n| n == "position_ids") {
        io.position_ids = "position_ids".to_string();
    }
    if output_names.iter().any(|n| n == "logits") {
        io.logits = "logits".to_string();
    }
    validate_io(&io, &input_names, &output_names)?;
    Ok(io)
}

/// Validate a provided [`LlmIo`] against the graph's declared inputs/outputs —
/// a mismatch is a loud load error.
fn validate_io(io: &LlmIo, input_names: &[String], output_names: &[String]) -> Result<(), OrtError> {
    let expect = |name: &str| -> Result<(), OrtError> {
        if input_names.iter().any(|n| n == name) {
            Ok(())
        } else {
            Err(OrtError::Other(format!(
                "LlmIo declares input \"{name}\" but the graph does not (inputs: {input_names:?})"
            )))
        }
    };
    expect(&io.input_ids)?;
    expect(&io.attention_mask)?;
    expect(&io.position_ids)?;
    for &layer in &io.attention_layers {
        expect(&format!("{}.{layer}.key", io.past_key_values))?;
        expect(&format!("{}.{layer}.value", io.past_key_values))?;
    }
    for &layer in &io.conv_layers {
        expect(&format!("{}.{layer}", io.conv_state))?;
    }
    if !output_names.iter().any(|n| n == io.logits.as_str()) {
        return Err(OrtError::Other(format!(
            "LlmIo declares output \"{}\" but the graph does not (outputs: {output_names:?})",
            io.logits
        )));
    }
    Ok(())
}

/// Read the fixed model dims from the model's `config.json` for introspection.
fn config_json_dims(config: &OnnxConfig) -> Result<(usize, usize, usize, usize), OrtError> {
    let Some(path) = config.config_json_path() else {
        return Ok((2048, 32, 8, 3));
    };
    let raw = std::fs::read_to_string(&path).map_err(|e| OrtError::ConfigRead {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let num = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).map(|x| x as usize);
    Ok((
        num("hidden_size").unwrap_or(2048),
        num("num_heads").unwrap_or(32),
        num("num_key_value_heads").unwrap_or(8),
        num("conv_L_cache").unwrap_or(3),
    ))
}

/// Build an `OrtLlmSession` over an already-loaded registry session handle,
/// resolving/validating the `LlmIo` contract.
pub fn build_llm_session_from_handle(
    handle: &SessionHandle,
    config: &OnnxConfig,
    model_key: &str,
) -> Result<OrtLlmSession, OrtError> {
    if config.task != OnnxTask::CausalLm {
        return Err(OrtError::Other(format!(
            "onnx model {model_key} is not a CausalLm session (task={:?})",
            config.task
        )));
    }
    let session = handle
        .downcast_arc::<Mutex<ort::session::Session>>()
        .ok_or_else(|| {
            OrtError::Other(format!("session handle for {model_key} does not hold an ort session"))
        })?;
    let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
        OrtError::Other(format!("onnx llm role '{model_key}' missing tokenizer_path"))
    })?;
    let tokenizer = LfmTokenizer::from_file(tokenizer_path, config.max_seq_len)?;

    let io = if let Some(io) = &config.llm_io {
        let guard = session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let input_names: Vec<String> = guard.inputs().iter().map(|i| i.name().to_string()).collect();
        let output_names: Vec<String> =
            guard.outputs().iter().map(|o| o.name().to_string()).collect();
        validate_io(io, &input_names, &output_names)?;
        io.clone()
    } else {
        let guard = session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        introspect_io(&guard, config)?
    };

    Ok(OrtLlmSession {
        runner: Arc::new(RealSessionRun { session }),
        tokenizer,
        io,
        max_gen_tokens: config.max_gen_tokens,
    })
}

impl OrtLlmSession {
    /// Build a session over a pre-built run seam + io contract. The router
    /// reaches the real seam through [`build_llm_session_from_handle`];
    /// `#[cfg(test)]` because only the hermetic tests construct a session over
    /// a stub runner directly.
    #[cfg(test)]
    pub(crate) fn new_with_runner(
        runner: Arc<dyn OnnxRun>,
        tokenizer: Arc<LfmTokenizer>,
        io: LlmIo,
        max_gen_tokens: usize,
    ) -> Self {
        Self {
            runner,
            tokenizer,
            io,
            max_gen_tokens,
        }
    }
}

/// Build an `OrtLlmSession` from a registry session for `model_key`, if the
/// model is registered and its task is `CausalLm`. Returns `None` for an
/// unregistered or non-CausalLm key (fail-open), a loud error for a broken
/// handle.
pub fn build_llm_session(
    registry: &OrtSessionRegistry,
    model_key: &str,
) -> Result<Option<OrtLlmSession>, OrtError> {
    let Some(config) = registry.config(model_key) else {
        return Ok(None);
    };
    if config.task != OnnxTask::CausalLm {
        return Ok(None);
    }
    let Some(handle) = registry.ensure_loaded(model_key)? else {
        return Ok(None);
    };
    build_llm_session_from_handle(&handle, &config, model_key).map(Some)
}

#[cfg(test)]
#[path = "../tests/llm.rs"]
pub(crate) mod tests;
