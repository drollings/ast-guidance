//! The per-context KV surface (ROADMAP M2): `PastState`, `OnnxContext`, and the
//! RAM `OnnxKVCache` — the onnx half of the shared `LlmContext`/`LlmKVCache`
//! contracts in `fluent_llm::runtime`.
//!
//! One weights load (an [`crate::llm::OrtLlmSession`]) serves N named context
//! windows; each context owns its own `PastState` (KV + conv state) behind a
//! `Mutex`, plus the lifecycle knobs the llama fork already has (`n_ctx`,
//! `max_ctx`, `pinned`, `resume`, `last_used`). The decode loop reads and
//! advances a context's KV between `prefill`/`decode_step` calls, so contexts
//! interleave on the shared session without sharing KV.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_llm::runtime::{LlmContext, LlmKVCache, LlmRuntimeError, SnapshotMeta};

/// The default context window (tokens) for a named onnx context whose profile
/// leaves `n_ctx` at 0 — mirrors the llama default (`default_num_ctx`).
pub const DEFAULT_ONNX_CONTEXT_TOKENS: u64 = 16384;

/// The conv/KV past-state carried across decode steps. Holds owned f32 data so
/// it can be rebuilt into tensors for each `session.run`. A context's KV lives
/// as one `Arc<Mutex<Option<PastState>>>` — `None` until the first `prefill`.
#[derive(Debug, Clone, Default)]
pub struct PastState {
    /// The total sequence length processed so far (past tokens).
    pub seq_len: usize,
    /// present_conv.{L} → flat `[hidden_size * conv_l_cache]` f32.
    pub conv: BTreeMap<usize, Vec<f32>>,
    /// present.{L}.key / .value → flat `[n_kv_heads * seq_len * head_dim]`.
    pub kv: BTreeMap<usize, (Vec<f32>, Vec<f32>)>,
}

impl PastState {
    /// Keep only the most recent `max_seq` KV positions (rolling window). The
    /// conv state is a fixed-size sliding window (`conv_l_cache`) and never
    /// grows with the sequence, so only the full-attention KV is truncated.
    /// `n_kv_heads`/`head_dim` come from the session's [`crate::config::LlmIo`].
    /// Returns the number of positions dropped.
    pub fn truncate(&mut self, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> usize {
        if self.seq_len <= max_seq {
            return 0;
        }
        let drop = self.seq_len - max_seq;
        let keep = max_seq;
        for (key, value) in self.kv.values_mut() {
            let expected = n_kv_heads * self.seq_len * head_dim;
            if key.len() != expected || value.len() != expected {
                continue; // a mismatched layer is left alone (never a panic)
            }
            let mut new_key = Vec::with_capacity(n_kv_heads * keep * head_dim);
            let mut new_value = Vec::with_capacity(n_kv_heads * keep * head_dim);
            for head in 0..n_kv_heads {
                let base = head * self.seq_len * head_dim;
                for pos in drop..self.seq_len {
                    let start = base + pos * head_dim;
                    new_key.extend_from_slice(&key[start..start + head_dim]);
                    new_value.extend_from_slice(&value[start..start + head_dim]);
                }
            }
            *key = new_key;
            *value = new_value;
        }
        self.seq_len = keep;
        drop
    }

    /// Approximate resident bytes of the stored tensors (f32 × 4).
    pub fn bytes(&self) -> u64 {
        let mut bytes = 0u64;
        for (k, v) in self.kv.values() {
            bytes = bytes
                .saturating_add((k.len() * 4) as u64)
                .saturating_add((v.len() * 4) as u64);
        }
        for c in self.conv.values() {
            bytes = bytes.saturating_add((c.len() * 4) as u64);
        }
        bytes
    }
}

/// The declarative shape of a named onnx context (ROADMAP M2). M3 maps the
/// llama `InstanceProfile` vocabulary onto this; until then the pool's
/// `ensure_context` uses it directly.
#[derive(Debug, Clone)]
pub struct OnnxContextProfile {
    /// The context's group (mirrors a llama instance's group).
    pub group: String,
    /// The allocated context window in tokens (`0` → the pool default).
    pub n_ctx: u64,
    /// The context-size cap (`None` = no cap). `resize` refuses growth past it.
    pub max_ctx: Option<u64>,
    /// Whether the context is pinned (never evicted).
    pub pinned: bool,
    /// Whether the context is resume-marked (KV preserved across eviction).
    pub resume: bool,
}

impl Default for OnnxContextProfile {
    fn default() -> Self {
        Self {
            group: "default".into(),
            n_ctx: DEFAULT_ONNX_CONTEXT_TOKENS,
            max_ctx: None,
            pinned: false,
            resume: false,
        }
    }
}

/// The "save to RAM" half of [`LlmKVCache`] (ROADMAP M2): saves/restores a
/// context's `PastState` by cloning it into/from a named in-process map. The
/// llama fork persists snapshots through the fork API; onnx keeps them in RAM
/// (the roadmap's per-runtime snapshot choice — a real "save to RAM").
pub struct OnnxKVCache {
    /// The context's live KV slot (shared with the owning [`OnnxContext`]).
    state: Arc<Mutex<Option<PastState>>>,
    /// The named snapshot index.
    snapshots: Arc<Mutex<HashMap<String, PastState>>>,
    /// The owning context's name (snapshot diagnostics).
    context_name: String,
}

impl OnnxKVCache {
    /// Build the adapter over a context's shared KV slot + snapshot index. The
    /// Arcs are shared with the owning context, so the adapter and the context
    /// mutate the same KV with no type-level reference cycle.
    fn new(
        state: Arc<Mutex<Option<PastState>>>,
        snapshots: Arc<Mutex<HashMap<String, PastState>>>,
        context_name: String,
    ) -> Arc<Self> {
        Arc::new(Self {
            state,
            snapshots,
            context_name,
        })
    }

    /// Synchronous "save to RAM" (ROADMAP M6): the sync chat-decode path
    /// snapshots the context's KV before it can be evicted, without an await.
    pub fn save_sync(&self, name: &str) -> Result<(), LlmRuntimeError> {
        let past = lock(&self.state).clone();
        let Some(past) = past else {
            return Err(LlmRuntimeError::NotLoaded(format!(
                "context {} has no KV to snapshot",
                self.context_name
            )));
        };
        lock(&self.snapshots).insert(name.to_string(), past);
        Ok(())
    }

    /// Synchronous "restore from RAM" (ROADMAP M6): rehydrate the context's KV
    /// from a named snapshot before a continued decode.
    pub fn restore_sync(&self, name: &str) -> Result<(), LlmRuntimeError> {
        let past = lock(&self.snapshots).get(name).cloned().ok_or_else(|| {
            LlmRuntimeError::NotLoaded(format!(
                "snapshot {name} not found for context {}",
                self.context_name
            ))
        })?;
        *lock(&self.state) = Some(past);
        Ok(())
    }
}

#[async_trait::async_trait]
impl LlmKVCache for OnnxKVCache {
    async fn save(&self, name: &str) -> Result<(), LlmRuntimeError> {
        OnnxKVCache::save_sync(self, name)
    }

    async fn restore(&self, name: &str) -> Result<(), LlmRuntimeError> {
        OnnxKVCache::restore_sync(self, name)
    }

    async fn list(&self) -> Vec<SnapshotMeta> {
        lock(&self.snapshots)
            .iter()
            .map(|(name, past)| SnapshotMeta {
                name: name.clone(),
                size: past.bytes(),
                mtime: 0,
                n_ctx_seq: past.seq_len as u64,
            })
            .collect()
    }

    async fn delete(&self, name: &str) -> Result<(), LlmRuntimeError> {
        lock(&self.snapshots).remove(name);
        Ok(())
    }
}

/// One named context window with its own KV cache (ROADMAP M2), allocated from
/// a shared [`crate::llm::OrtLlmSession`]. Owns the KV slot + lifecycle knobs;
/// the decode loop (`prefill`/`decode_step`) reads and advances it, so contexts
/// interleave on the shared session without sharing KV.
pub struct OnnxContext {
    name: String,
    group: String,
    n_ctx: AtomicU64,
    max_ctx: Option<u64>,
    pinned: bool,
    resume: Arc<AtomicBool>,
    last_used: AtomicI64,
    state: Arc<Mutex<Option<PastState>>>,
    kv: Arc<OnnxKVCache>,
}

impl OnnxContext {
    /// Allocate a named context from the given profile. A profile with
    /// `n_ctx == 0` inherits the pool default. The pool wraps it in an
    /// `Arc` (the shared registry handle).
    pub fn new(name: String, profile: OnnxContextProfile) -> Self {
        let n_ctx = if profile.n_ctx == 0 {
            DEFAULT_ONNX_CONTEXT_TOKENS
        } else {
            profile.n_ctx
        };
        let state = Arc::new(Mutex::new(None));
        let snapshots = Arc::new(Mutex::new(HashMap::new()));
        let kv = OnnxKVCache::new(
            Arc::clone(&state),
            snapshots,
            name.clone(),
        );
        Self {
            name,
            group: profile.group,
            n_ctx: AtomicU64::new(n_ctx),
            max_ctx: profile.max_ctx,
            pinned: profile.pinned,
            resume: Arc::new(AtomicBool::new(profile.resume)),
            last_used: AtomicI64::new(0),
            state,
            kv,
        }
    }

    /// The bare context name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The context's group.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// The allocated context window in tokens.
    pub fn n_ctx(&self) -> u64 {
        self.n_ctx.load(Ordering::Relaxed)
    }

    /// The context-size cap (`None` = no cap). `resize` refuses growth past it.
    pub fn max_ctx(&self) -> Option<u64> {
        self.max_ctx
    }

    /// Whether the context is pinned (never evicted).
    pub fn pinned(&self) -> bool {
        self.pinned
    }

    /// Whether the context is resume-marked (KV preserved across eviction).
    pub fn resume(&self) -> bool {
        self.resume.load(Ordering::Relaxed)
    }

    /// Set/clear the resume flag.
    pub fn set_resume(&self, enabled: bool) {
        self.resume.store(enabled, Ordering::Relaxed);
    }

    /// Record dispatch recency (the onnx clock: unix-ms).
    pub fn touch(&self) {
        self.last_used.store(now_unix_ms(), Ordering::Relaxed);
    }

    /// Pool-native last use (unix-ms; `0` = never used).
    pub fn last_used(&self) -> i64 {
        self.last_used.load(Ordering::Relaxed)
    }

    /// A RAM-resident context owns no VRAM.
    pub fn vram_bytes(&self) -> u64 {
        0
    }

    /// Store the context's KV (set by `prefill`/`decode_step`).
    pub fn store_past(&self, past: PastState) {
        *lock(&self.state) = Some(past);
    }

    /// The context's current KV (`None` until the first `prefill`).
    pub fn past(&self) -> Option<PastState> {
        lock(&self.state).clone()
    }

    /// Clear the context's KV (freeing it). Snapshot rows are kept — the
    /// resume contract preserves them across destroy.
    pub fn clear(&self) {
        *lock(&self.state) = None;
    }

    /// Current KV occupancy in tokens.
    pub fn seq_len(&self) -> u64 {
        lock(&self.state)
            .as_ref()
            .map_or(0, |p| p.seq_len as u64)
    }

    /// Resident KV bytes at current occupancy.
    pub fn kv_bytes(&self) -> u64 {
        lock(&self.state).as_ref().map_or(0, PastState::bytes)
    }

    /// Resize the context window. Growth past `max_ctx` is refused; a shrink
    /// below the current KV occupancy takes effect at the next decode step
    /// (the rolling window). In-place — unlike the llama fork's destroy +
    /// re-create.
    pub fn resize(&self, n_ctx: u64) -> Result<(), LlmRuntimeError> {
        if let Some(cap) = self.max_ctx {
            if n_ctx > cap {
                return Err(LlmRuntimeError::Other(format!(
                    "context {} resize to {n_ctx} exceeds max_ctx {cap}",
                    self.name
                )));
            }
        }
        self.n_ctx.store(n_ctx, Ordering::Relaxed);
        Ok(())
    }

    /// The KV-cache adapter (snapshot/restore/list/delete over this context's
    /// KV).
    pub fn kv_cache(&self) -> Arc<OnnxKVCache> {
        Arc::clone(&self.kv)
    }
}

#[async_trait::async_trait]
impl LlmContext for OnnxContext {
    fn name(&self) -> &str {
        OnnxContext::name(self)
    }
    fn group(&self) -> &str {
        OnnxContext::group(self)
    }
    fn n_ctx(&self) -> u64 {
        OnnxContext::n_ctx(self)
    }
    fn max_ctx(&self) -> Option<u64> {
        OnnxContext::max_ctx(self)
    }
    async fn resize(&self, n_ctx: u64) -> Result<(), LlmRuntimeError> {
        OnnxContext::resize(self, n_ctx)
    }
    fn pinned(&self) -> bool {
        OnnxContext::pinned(self)
    }
    fn resume(&self) -> bool {
        OnnxContext::resume(self)
    }
    fn set_resume(&self, enabled: bool) {
        OnnxContext::set_resume(self, enabled);
    }
    fn touch(&self) {
        OnnxContext::touch(self);
    }
    fn last_used(&self) -> i64 {
        OnnxContext::last_used(self)
    }
    fn vram_bytes(&self) -> u64 {
        0
    }
    async fn destroy(&self) -> Result<(), LlmRuntimeError> {
        OnnxContext::clear(self);
        Ok(())
    }
    fn kv_cache(&self) -> Arc<dyn LlmKVCache> {
        OnnxContext::kv_cache(self) as Arc<dyn LlmKVCache>
    }
}

/// Current system time as unix-milliseconds.
fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

#[cfg(test)]
#[path = "../tests/context.rs"]
mod tests;
