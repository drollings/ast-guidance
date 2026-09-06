use super::*;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// A stub weights instance for the hermetic engine tests. Tracks load /
/// unload / touch calls and exposes canned residency rows.
struct StubWeights {
    key: &'static str,
    policy: EvictionPolicy,
    weights_bytes: u64,
    loaded: AtomicBool,
    pinned: bool,
    refuse_unload: bool,
    sleep_idle_seconds: Option<i32>,
    last_used: AtomicI64,
    rows: Vec<LlmResidencyRow>,
    unloaded: Mutex<usize>,
    contexts: Mutex<Vec<StubContext>>,
    in_flight: AtomicUsize,
}

impl StubWeights {
    fn llama(
        key: &'static str,
        weights_bytes: u64,
        loaded: bool,
        rows: Vec<LlmResidencyRow>,
    ) -> Self {
        Self {
            key,
            policy: EvictionPolicy::FootprintColdness,
            weights_bytes,
            loaded: AtomicBool::new(loaded),
            pinned: false,
            refuse_unload: false,
            sleep_idle_seconds: None,
            last_used: AtomicI64::new(-1),
            rows,
            unloaded: Mutex::new(0),
            contexts: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
        }
    }

    fn onnx(
        key: &'static str,
        weights_bytes: u64,
        loaded: bool,
        sleep_idle_seconds: Option<i32>,
        last_used: i64,
    ) -> Self {
        Self {
            key,
            policy: EvictionPolicy::LruLargest,
            weights_bytes,
            loaded: AtomicBool::new(loaded),
            pinned: false,
            refuse_unload: false,
            sleep_idle_seconds,
            last_used: AtomicI64::new(last_used),
            rows: Vec::new(),
            unloaded: Mutex::new(0),
            contexts: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
        }
    }

    fn with_refuse_unload(mut self) -> Self {
        self.refuse_unload = true;
        self
    }

    fn with_last_used(mut self, last_used: i64) -> Self {
        self.last_used = AtomicI64::new(last_used);
        self
    }

    fn set_in_flight(&self, n: usize) {
        self.in_flight.store(n, Ordering::Relaxed);
    }

    fn unload_count(&self) -> usize {
        *self.unloaded.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl LlmWeights for StubWeights {
    fn model_key(&self) -> &str {
        self.key
    }
    fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }
    fn pinned(&self) -> bool {
        self.pinned
    }
    fn refuse_unload(&self) -> bool {
        self.refuse_unload
    }
    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Relaxed)
    }
    fn sleep_idle_seconds(&self) -> Option<i32> {
        self.sleep_idle_seconds
    }
    async fn ensure_loaded(&self) -> Result<(), LlmRuntimeError> {
        self.loaded.store(true, Ordering::Relaxed);
        Ok(())
    }
    async fn unload(&self) -> Result<(), LlmRuntimeError> {
        if self.refuse_unload {
            return Err(LlmRuntimeError::UnloadRefused(self.key.to_string()));
        }
        self.loaded.store(false, Ordering::Relaxed);
        *self.unloaded.lock().unwrap() += 1;
        Ok(())
    }
    fn touch(&self) {
        self.last_used.store(0, Ordering::Relaxed);
    }
    fn last_used(&self) -> i64 {
        self.last_used.load(Ordering::Relaxed)
    }
    async fn residency_rows(&self) -> Vec<LlmResidencyRow> {
        self.rows.clone()
    }
    fn context(&self, name: &str) -> Option<Arc<dyn LlmContext>> {
        self.contexts
            .lock()
            .unwrap()
            .iter()
            .find(|c| c.name == name)
            .map(|c| Arc::new(c.clone()) as Arc<dyn LlmContext>)
    }
    async fn ensure_context(&self, _name: &str) -> Result<Arc<dyn LlmContext>, LlmRuntimeError> {
        Err(LlmRuntimeError::NotLoaded(self.key.to_string()))
    }
    fn eviction_policy(&self) -> EvictionPolicy {
        self.policy
    }
    fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

/// A stub context that records destroys. A resume-marked stub models the
/// save-before-evict contract in its `evict` override (snapshot event
/// first, then the destroy), so the engine test below pins that the pass
/// drives eviction through `evict()` in order.
#[derive(Clone)]
struct StubContext {
    name: &'static str,
    destroyed: Arc<Mutex<usize>>,
    resume: bool,
    events: Option<Arc<Mutex<Vec<String>>>>,
}

impl StubContext {
    fn new(name: &'static str, destroyed: Arc<Mutex<usize>>) -> Self {
        Self { name, destroyed, resume: false, events: None }
    }

    fn resume_marked(
        name: &'static str,
        destroyed: Arc<Mutex<usize>>,
        events: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        Self { name, destroyed, resume: true, events: Some(events) }
    }
}

#[async_trait::async_trait]
impl LlmContext for StubContext {
    fn name(&self) -> &str {
        self.name
    }
    fn group(&self) -> &str {
        self.name
    }
    fn n_ctx(&self) -> u64 {
        16384
    }
    fn max_ctx(&self) -> Option<u64> {
        None
    }
    async fn resize(&self, _n_ctx: u64) -> Result<(), LlmRuntimeError> {
        Ok(())
    }
    fn pinned(&self) -> bool {
        false
    }
    fn resume(&self) -> bool {
        false
    }
    fn set_resume(&self, _enabled: bool) {}
    fn touch(&self) {}
    fn last_used(&self) -> i64 {
        0
    }
    fn vram_bytes(&self) -> u64 {
        0
    }
    async fn destroy(&self) -> Result<(), LlmRuntimeError> {
        *self.destroyed.lock().unwrap() += 1;
        if let Some(events) = &self.events {
            events.lock().unwrap().push(format!("destroy:{}", self.name));
        }
        Ok(())
    }
    fn kv_cache(&self) -> Arc<dyn LlmKVCache> {
        Arc::new(StubKvCache)
    }

    async fn evict(&self) -> Result<(), LlmRuntimeError> {
        // The resume-marked contract: snapshot first, destroy second. The
        // engine must drive eviction through this method (never `destroy`
        // directly) so adapters can preserve state before the drop.
        if self.resume {
            if let Some(events) = &self.events {
                events.lock().unwrap().push(format!("snapshot:{}", self.name));
            }
        }
        self.destroy().await
    }
}

struct StubKvCache;

#[async_trait::async_trait]
impl LlmKVCache for StubKvCache {
    async fn save(&self, _name: &str) -> Result<(), LlmRuntimeError> {
        Ok(())
    }
    async fn restore(&self, _name: &str) -> Result<(), LlmRuntimeError> {
        Ok(())
    }
    async fn list(&self) -> Vec<SnapshotMeta> {
        Vec::new()
    }
    async fn delete(&self, _name: &str) -> Result<(), LlmRuntimeError> {
        Ok(())
    }
}

fn row(key: &'static str, name: &'static str, pinned: bool, last_used: i64) -> LlmResidencyRow {
    LlmResidencyRow {
        context_key: format!("{key}:{name}"),
        group: name.to_string(),
        n_ctx: 16384,
        parallel: 1,
        pinned,
        resume: false,
        state: "loaded".into(),
        runtime: LlmRuntime::Llama,
        model_bytes: 0,
        context_bytes: 0,
        compute_bytes: 0,
        total_bytes: 0,
        vram_bytes: 1000,
        last_used,
    }
}

#[tokio::test]
async fn idle_release_fires_for_stub_whose_idle_exceeds_threshold() {
    // A 1s-idle onnx session, clock 10s ahead of its last_used.
    let w = Arc::new(StubWeights::onnx(
        "onnx/lazy",
        100,
        true,
        Some(1),
        9_000_000_000i64 - 10_000,
    ));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[w.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");
    assert_eq!(w.unload_count(), 1, "idle release unloads the session");
}

#[tokio::test]
async fn idle_release_skips_fresh_and_refusing_weights() {
    // Fresh: 0ms idle → never released.
    let fresh = Arc::new(StubWeights::onnx("onnx/fresh", 100, true, Some(30), 9_000_000_000i64));
    // Refusing (`Always`/pinned-equivalent): idle but never released.
    let refusing =
        Arc::new(StubWeights::onnx("onnx/refusing", 100, true, Some(1), 0).with_refuse_unload());
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine
        .residency_cycle(&[fresh.clone() as Arc<dyn LlmWeights>, refusing.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(fresh.unload_count(), 0, "fresh session stays");
    assert_eq!(refusing.unload_count(), 0, "Always/pinned session never released");
}

#[tokio::test]
async fn budget_eviction_footprint_coldness_matches_eviction_order() {
    // Two unpinned llama contexts with distinct coldness; the engine must
    // pick the same first candidate the shared `eviction_order` primitive
    // would for the identical input.
    let mut a = row("base", "a", false, 100);
    a.vram_bytes = 2000;
    let mut b = row("base", "b", false, 200);
    b.vram_bytes = 1000;
    let pinned = row("base", "pinned", true, 0);
    let w = Arc::new(StubWeights::llama(
        "base",
        5000,
        true,
        vec![a.clone(), b.clone(), pinned.clone()],
    ));
    // Register stub contexts so the engine can resolve + destroy them.
    // Each context owns its own destroy counter (the evicted-context
    // assertion must distinguish which one was destroyed).
    let destroyed_a = Arc::new(Mutex::new(0usize));
    let destroyed_b = Arc::new(Mutex::new(0usize));
    let destroyed_pinned = Arc::new(Mutex::new(0usize));
    *w.contexts.lock().unwrap() = vec![
        StubContext::new("a", Arc::clone(&destroyed_a)),
        StubContext::new("b", Arc::clone(&destroyed_b)),
        StubContext::new("pinned", Arc::clone(&destroyed_pinned)),
    ];

    // Crank the budget so the pass evicts (mirror the llama over-budget);
    // used = max(model_bytes) 0 + vram 2000+1000+1000 = 4000 > 500 (VRAM
    // accounting: fork weights + VRAM context bytes, never file size or
    // all-memory totals).
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        Some(500),
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[w.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");

    // The shared primitive, fed the identical candidate inputs, must agree
    // on the first eviction candidate.
    let now = 9_000_000_000i64;
    // `eviction_order` takes seconds (the llama pool); derive from the ms clock.
    let now_secs = now / 1000;
    let candidates: Vec<Candidate> = vec![
        Candidate {
            kind: CandidateKind::Context { weights: Arc::clone(&w) as Arc<dyn LlmWeights>, name: "a".into() },
            freed_bytes: 2000,
            last_used: 100,
        },
        Candidate {
            kind: CandidateKind::Context { weights: Arc::clone(&w) as Arc<dyn LlmWeights>, name: "b".into() },
            freed_bytes: 1000,
            last_used: 200,
        },
    ];
    let ordered = common_core::cache::eviction_order(
        candidates,
        now_secs,
        |c: &Candidate| c.freed_bytes,
        |c: &Candidate| c.last_used,
    );
    assert_eq!(ordered[0].freed_bytes, 2000, "a (2000B, cold 100) outranks b (1000B, cold 200) by footprint×coldness");
    // Engine destroyed exactly one context (batch=1) and it was `a`.
    assert_eq!(*destroyed_a.lock().unwrap(), 1, "a destroyed");
    assert_eq!(*destroyed_b.lock().unwrap(), 0, "b untouched (batch reached)");
    assert_eq!(*destroyed_pinned.lock().unwrap(), 0, "pinned never evicted");
}

#[tokio::test]
async fn budget_eviction_runs_snapshot_before_destroy_through_evict() {
    // A resume-marked llama context over budget: the engine must drive the
    // eviction through `evict()` (never `destroy()` directly) and the
    // snapshot must be observed before the destroy.
    let mut work = row("base", "work", false, 100);
    work.vram_bytes = 2000;
    let w = Arc::new(StubWeights::llama("base", 5000, true, vec![work]));
    let destroyed = Arc::new(Mutex::new(0usize));
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    *w.contexts.lock().unwrap() = vec![StubContext::resume_marked(
        "work",
        Arc::clone(&destroyed),
        Arc::clone(&events),
    )];

    // used = vram 2000 > 500 → exactly one context eviction.
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        Some(500),
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[w.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");

    assert_eq!(*destroyed.lock().unwrap(), 1, "context destroyed once");
    assert_eq!(
        *events.lock().unwrap(),
        vec!["snapshot:work".to_string(), "destroy:work".to_string()],
        "snapshot observed before destroy, via evict()"
    );
}

fn admission_engine(vram_budget: Option<u64>) -> Arc<LlmResidencyEngine> {
    LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        vram_budget,
        None,
        30,
        10,
        Arc::new(|| 9_000_000_000i64),
    )
}

/// One loaded llama weights with a single unpinned context row of `vram`
/// bytes, registered for engine eviction. Returns the weights and its
/// destroy counter.
fn loaded_llama(key: &'static str, vram: u64, last_used: i64) -> (Arc<StubWeights>, Arc<Mutex<usize>>) {
    let mut ctx = row(key, "work", false, last_used);
    ctx.vram_bytes = vram;
    let w = Arc::new(StubWeights::llama(key, 5000, true, vec![ctx]));
    let destroyed = Arc::new(Mutex::new(0usize));
    *w.contexts.lock().unwrap() =
        vec![StubContext::new("work", Arc::clone(&destroyed))];
    (w, destroyed)
}

#[tokio::test]
async fn admission_no_eviction_within_budget() {
    // used 1000 + required 500 <= 2000 → the load fits, nothing evicted.
    let (a, destroyed) = loaded_llama("a", 1000, 100);
    admission_engine(Some(2000))
        .make_room_for(&[a.clone() as Arc<dyn LlmWeights>], "b", 500, MemoryPool::Vram)
        .await;
    assert_eq!(*destroyed.lock().unwrap(), 0, "no eviction within budget");
}

#[tokio::test]
async fn admission_evicts_over_budget_and_excludes_target() {
    // `b` is larger and colder but is the cold target: admission must evict
    // from `a` and never touch `b`.
    let (a, destroyed_a) = loaded_llama("a", 1000, 100);
    let (b, destroyed_b) = loaded_llama("b", 5000, 50);
    admission_engine(Some(2000))
        .make_room_for(
            &[a.clone() as Arc<dyn LlmWeights>, b.clone() as Arc<dyn LlmWeights>],
            "b",
            1500,
            MemoryPool::Vram,
        )
        .await;
    assert_eq!(*destroyed_a.lock().unwrap(), 1, "non-target evicted to make room");
    assert_eq!(*destroyed_b.lock().unwrap(), 0, "cold target excluded");
}

#[tokio::test]
async fn admission_no_budget_or_zero_required_is_noop() {
    let (a, destroyed) = loaded_llama("a", 1000, 100);
    let weights = vec![a.clone() as Arc<dyn LlmWeights>];
    admission_engine(None).make_room_for(&weights, "b", 10_000, MemoryPool::Vram).await;
    admission_engine(Some(100)).make_room_for(&weights, "b", 0, MemoryPool::Vram).await;
    assert_eq!(*destroyed.lock().unwrap(), 0, "no budget / zero need → no-op");
}

#[tokio::test]
async fn budget_eviction_lru_largest_matches_onnx_sort() {
    // Mirror `working_set_budget_evicts_lru_largest_first`: big (100B,
    // older) + small (50B) over a 60B budget → the largest is released
    // first and the small stays.
    let big = Arc::new(StubWeights::onnx("onnx/big", 100, true, None, 9_000_000_000i64 - 2_000));
    let small = Arc::new(StubWeights::onnx("onnx/small", 50, true, None, 9_000_000_000i64 - 1_000));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        Some(60),
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine
        .residency_cycle(&[big.clone() as Arc<dyn LlmWeights>, small.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(big.unload_count(), 1, "largest working set evicted first");
    assert_eq!(small.unload_count(), 0, "under budget after the big release");
}

#[tokio::test]
async fn budget_eviction_lru_largest_tie_evicts_oldest_first() {
    // Mirror `working_set_tie_evicts_oldest_first`: two 100B sessions over
    // a 150B budget → the oldest of the equal footprints goes.
    let old = Arc::new(StubWeights::onnx("onnx/old", 100, true, None, 9_000_000_000i64 - 2_000));
    let new = Arc::new(StubWeights::onnx("onnx/new", 100, true, None, 9_000_000_000i64 - 1_000));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        Some(150),
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine
        .residency_cycle(&[old.clone() as Arc<dyn LlmWeights>, new.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(old.unload_count(), 1, "oldest evicted first on the size tie");
    assert_eq!(new.unload_count(), 0, "newest of equal footprints stays");
}

#[tokio::test]
async fn budget_eviction_within_limits_releases_nothing() {
    let m = Arc::new(StubWeights::onnx("onnx/m", 100, true, None, 9_000_000_000i64 - 1_000));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        Some(1000),
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[m.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");
    assert_eq!(m.unload_count(), 0, "within budget → nothing released");
}

#[tokio::test]
async fn unload_empty_fires_for_weights_with_zero_contexts() {
    // A llama weights instance with no rows (all contexts evicted) is
    // unloaded; one with a pinned context is not. No budgets → only the
    // unload-empty pass runs (the budget-eviction pass could otherwise
    // free the empty model first, like the real llama loop).
    let empty = Arc::new(StubWeights::llama("base/empty", 5000, true, vec![]));
    let resident = Arc::new(StubWeights::llama(
        "base/resident",
        5000,
        true,
        vec![row("base/resident", "ledger", true, 0)],
    ));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine
        .residency_cycle(&[empty.clone() as Arc<dyn LlmWeights>, resident.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(empty.unload_count(), 1, "zero-context model unloaded");
    assert_eq!(resident.unload_count(), 0, "pinned context keeps the weights resident");
}

#[tokio::test]
async fn unload_empty_skips_weights_that_are_not_loaded() {
    // A down server reports empty rows too, so `unload_empty` must gate on
    // `is_loaded()` — otherwise it re-unloads (and re-logs) every unloaded
    // llama model on every residency pass, churning forever.
    let down = Arc::new(StubWeights::llama("base/down", 5000, false, vec![]));
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    for _ in 0..3 {
        engine
            .residency_cycle(&[down.clone() as Arc<dyn LlmWeights>])
            .await
            .expect("cycle");
    }
    assert_eq!(
        down.unload_count(),
        0,
        "an unloaded (down) model must not be re-unloaded per pass"
    );
}

#[tokio::test]
async fn pinned_and_refusing_weights_never_budget_evicted() {
    // A refusing (`Always`/pinned-equivalent) onnx session is never a
    // budget candidate even when far over budget.
    let refusing = Arc::new(
        StubWeights::onnx("onnx/resident", 500, true, None, 0).with_refuse_unload(),
    );
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        Some(100),
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[refusing.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");
    assert_eq!(refusing.unload_count(), 0, "Always/pinned never budget-evicted");
}

#[tokio::test]
async fn start_loops_on_injected_runtime() {
    // `start` must run on an injected `Runtime` (the real tokio backend
    // here), never ambient `tokio::spawn`, and drive the loop.
    let rt = fluent_concurrency::tokio_runtime();
    let w = Arc::new(StubWeights::onnx(
        "onnx/lazy",
        100,
        true,
        Some(1),
        9_000_000_000i64 - 10_000,
    ));
    let engine = LlmResidencyEngine::new(Duration::from_millis(5), None, None, 30, 1);
    let weights: Arc<Vec<Arc<dyn LlmWeights>>> =
        Arc::new(vec![w.clone() as Arc<dyn LlmWeights>]);
    let handle = engine.start(&rt, weights);
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    handle.abort();
    let _ = handle.await;
    assert_eq!(w.unload_count(), 1, "loop released the idle entry");
}

#[tokio::test]
async fn vram_budget_ignores_cpu_offloaded_bytes() {
    // Partial GPU offload (production 28.4G-counted-vs-14.2G-resident WARN):
    // the weights file is 14000B, the fork reports 13900B shared weights
    // resident, and one context whose 14550B all-memory footprint holds only
    // 350B in VRAM. `ps` renders 13900 + 350 = 14250 resident; the engine
    // must agree — file size + all-memory total (28550) against a 19400
    // budget spuriously evicted.
    let mut ctx = row("base", "default", false, 100);
    ctx.model_bytes = 13900;
    ctx.context_bytes = 14000;
    ctx.compute_bytes = 550;
    ctx.total_bytes = 14550;
    ctx.vram_bytes = 350;
    let w = Arc::new(StubWeights::llama("base", 14000, true, vec![ctx]));
    let destroyed = Arc::new(Mutex::new(0usize));
    *w.contexts.lock().unwrap() =
        vec![StubContext::new("default", Arc::clone(&destroyed))];
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        Some(19400),
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[w.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");
    assert_eq!(w.unload_count(), 0, "VRAM-accurate usage (14250) fits the 19400 budget");
    assert_eq!(*destroyed.lock().unwrap(), 0, "no context destroyed");
}

#[tokio::test]
async fn vram_budget_still_evicts_when_truly_over() {
    // Guard against overcorrection: the same offloaded shape against a
    // 14000 budget (14250 resident) must evict exactly one unit (batch=1).
    let mut ctx = row("base", "default", false, 100);
    ctx.model_bytes = 13900;
    ctx.total_bytes = 14550;
    ctx.vram_bytes = 350;
    let w = Arc::new(StubWeights::llama("base", 14000, true, vec![ctx]));
    let destroyed = Arc::new(Mutex::new(0usize));
    *w.contexts.lock().unwrap() =
        vec![StubContext::new("default", Arc::clone(&destroyed))];
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        Some(14000),
        None,
        30,
        1,
        Arc::new(|| 9_000_000_000i64),
    );
    engine.residency_cycle(&[w.clone() as Arc<dyn LlmWeights>]).await.expect("cycle");
    assert_eq!(
        w.unload_count() + *destroyed.lock().unwrap(),
        1,
        "genuinely over budget evicts exactly one unit"
    );
}

#[test]
fn policy_pool_mapping() {
    assert_eq!(EvictionPolicy::FootprintColdness.pool(), MemoryPool::Vram);
    assert_eq!(EvictionPolicy::LruLargest.pool(), MemoryPool::Ram);
}

#[test]
fn error_display() {
    assert_eq!(
        format!("{}", LlmRuntimeError::UnloadRefused("m".into())),
        "runtime unload refused: m"
    );
}

#[test]
fn snapshot_meta_round_trips() {
    let meta = SnapshotMeta {
        name: "scratch-resume".into(),
        size: 4096,
        mtime: 1_700_000_000,
        n_ctx_seq: 512,
    };
    let json = serde_json::to_string(&meta).unwrap();
    let back: SnapshotMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(back, meta);
}

#[test]
fn residency_row_round_trips() {
    let r = row("base", "scratch", false, 100);
    let json = serde_json::to_string(&r).unwrap();
    let back: LlmResidencyRow = serde_json::from_str(&json).unwrap();
    assert_eq!(back.context_key, "base:scratch");
    assert_eq!(back.runtime, LlmRuntime::Llama);
}

/// The on-demand race: a llama weights instance loaded moments ago (touched
/// by the dispatch that loaded it) must survive the next residency pass even
/// with zero contexts — the first request has not materialized one yet.
#[tokio::test]
async fn unload_empty_skips_recently_touched_weights() {
    // Clock 9_000_000s; touched this same second → age 0 < one pass grace.
    let now_ms = 9_000_000_000i64;
    let fresh = Arc::new(
        StubWeights::llama("base/fresh", 5000, true, vec![]).with_last_used(now_ms / 1000),
    );
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(move || now_ms),
    );
    engine
        .residency_cycle(&[fresh.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(fresh.unload_count(), 0, "just-loaded model survives its first pass");
}

/// The grace is bounded: an empty weights instance untouched for an hour is
/// still collected, so idle models do not linger.
#[tokio::test]
async fn unload_empty_unloads_stale_empty_weights() {
    let now_ms = 9_000_000_000i64;
    let stale = Arc::new(
        StubWeights::llama("base/stale", 5000, true, vec![])
            .with_last_used(now_ms / 1000 - 3600),
    );
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(move || now_ms),
    );
    engine
        .residency_cycle(&[stale.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(stale.unload_count(), 1, "long-idle empty model is collected");
}

/// An active inference holds its weights: an in-flight empty model is never
/// unloaded, however stale its last use.
#[tokio::test]
async fn unload_empty_skips_in_flight_weights() {
    let now_ms = 9_000_000_000i64;
    let busy = Arc::new(
        StubWeights::llama("base/busy", 5000, true, vec![])
            .with_last_used(now_ms / 1000 - 3600),
    );
    busy.set_in_flight(1);
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        None,
        None,
        30,
        1,
        Arc::new(move || now_ms),
    );
    engine
        .residency_cycle(&[busy.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(busy.unload_count(), 0, "in-flight model is never unloaded");
}

/// Over-budget eviction also respects the lease: the in-flight largest
/// footprint is skipped and the next-coldest candidate frees the room.
#[tokio::test]
async fn budget_eviction_skips_in_flight_weights() {
    let now_ms = 9_000_000_000i64;
    let mut big_row = row("base/big", "ctx", false, 100);
    big_row.vram_bytes = 4000;
    let big = Arc::new(StubWeights::llama("base/big", 4000, true, vec![big_row]));
    big.set_in_flight(1);
    let mut small_row = row("base/small", "ctx", false, 200);
    small_row.vram_bytes = 500;
    let small = Arc::new(StubWeights::llama("base/small", 500, true, vec![small_row]));
    let destroyed_small = Arc::new(Mutex::new(0usize));
    *small.contexts.lock().unwrap() = vec![StubContext::new("ctx", Arc::clone(&destroyed_small))];
    // used = 4000 + 4000 + 500 + 500 = 9000 > 1000: the pass must free room
    // without touching the in-flight model.
    let engine = LlmResidencyEngine::new_with_clock(
        Duration::from_secs(1),
        Some(1000),
        None,
        30,
        10,
        Arc::new(move || now_ms),
    );
    engine
        .residency_cycle(&[big.clone() as Arc<dyn LlmWeights>, small.clone() as Arc<dyn LlmWeights>])
        .await
        .expect("cycle");
    assert_eq!(big.unload_count(), 0, "in-flight weights never evicted");
    assert!(
        *destroyed_small.lock().unwrap() > 0 || small.unload_count() > 0,
        "room is freed from the idle model instead"
    );
}
