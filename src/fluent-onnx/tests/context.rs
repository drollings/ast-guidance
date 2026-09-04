use super::*;

fn profile(n_ctx: u64) -> OnnxContextProfile {
    OnnxContextProfile {
        group: "g".into(),
        n_ctx,
        max_ctx: None,
        pinned: false,
        resume: false,
    }
}

fn sample_past() -> PastState {
    let mut kv = BTreeMap::new();
    kv.insert(0, (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]));
    PastState {
        seq_len: 2,
        conv: BTreeMap::new(),
        kv,
    }
}

#[test]
fn resize_refuses_growth_past_max_ctx() {
    let ctx = OnnxContext::new(
        "a".into(),
        OnnxContextProfile {
            max_ctx: Some(4096),
            ..profile(2048)
        },
    );
    assert_eq!(ctx.n_ctx(), 2048);
    assert!(ctx.resize(8192).is_err(), "growth past the cap refused");
    ctx.resize(4096).expect("at the cap is fine");
    ctx.resize(1024).expect("shrink is fine");
    assert_eq!(ctx.n_ctx(), 1024);
}

#[test]
fn resize_without_cap_allows_growth() {
    let ctx = OnnxContext::new("a".into(), profile(2048));
    ctx.resize(65536).expect("no cap");
    assert_eq!(ctx.n_ctx(), 65536);
}

#[test]
fn default_n_ctx_applied_when_profile_zero() {
    let ctx = OnnxContext::new("a".into(), profile(0));
    assert_eq!(ctx.n_ctx(), DEFAULT_ONNX_CONTEXT_TOKENS);
}

#[test]
fn resume_flag_flips_and_starts_false() {
    let ctx = OnnxContext::new("a".into(), profile(64));
    assert!(!ctx.resume());
    ctx.set_resume(true);
    assert!(ctx.resume());
    ctx.set_resume(false);
    assert!(!ctx.resume());
}

#[test]
fn touch_advances_last_used() {
    let ctx = OnnxContext::new("a".into(), profile(64));
    assert_eq!(ctx.last_used(), 0, "never used");
    ctx.touch();
    assert!(ctx.last_used() > 0, "touch advances the clock");
    assert_eq!(ctx.vram_bytes(), 0, "RAM-resident context owns no VRAM");
}

#[tokio::test]
async fn kv_cache_round_trips_a_past_state() {
    let ctx = OnnxContext::new("a".into(), profile(64));
    ctx.store_past(sample_past());
    let kv: Arc<dyn LlmKVCache> = ctx.kv_cache() as Arc<dyn LlmKVCache>;

    kv.save("resume").await.expect("save");
    ctx.clear();
    assert!(ctx.past().is_none(), "clear frees the KV");
    assert!(kv.restore("missing").await.is_err(), "unknown snapshot is an error");

    kv.restore("resume").await.expect("restore");
    let past = ctx.past().expect("restored");
    assert_eq!(past.seq_len, 2);
    assert_eq!(past.kv[&0], (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]));

    let list = kv.list().await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "resume");
    assert_eq!(list[0].n_ctx_seq, 2);
    assert!(list[0].size > 0, "snapshot size reflects the tensors");

    kv.delete("resume").await.expect("delete");
    assert!(kv.list().await.is_empty());
}

#[test]
fn sync_save_restore_round_trips_a_past_state() {
    // The M6 sync chat-decode path (`OnnxChatBackend`) snapshots/restores
    // KV without an await — the sync helpers must round-trip exactly.
    let ctx = OnnxContext::new("a".into(), profile(64));
    ctx.store_past(sample_past());
    let kv = ctx.kv_cache();

    kv.save_sync("resume").expect("save_sync");
    ctx.clear();
    assert!(ctx.past().is_none(), "clear frees the KV");
    assert!(kv.restore_sync("missing").is_err(), "unknown snapshot is an error");

    kv.restore_sync("resume").expect("restore_sync");
    let past = ctx.past().expect("restored");
    assert_eq!(past.seq_len, 2);
    assert_eq!(past.kv[&0], (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]));
}

#[tokio::test]
async fn save_before_prefill_is_an_error() {
    let ctx = OnnxContext::new("a".into(), profile(64));
    let kv: Arc<dyn LlmKVCache> = ctx.kv_cache() as Arc<dyn LlmKVCache>;
    assert!(kv.save("resume").await.is_err(), "no KV to snapshot");
}

#[test]
fn past_truncate_keeps_most_recent_positions() {
    // heads=1, head_dim=2, seq_len=4: key = [p0h0, p0h1, p1h0, p1h1, ...].
    let key: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let value: Vec<f32> = key.iter().map(|x| x * 10.0).collect();
    let mut kv = BTreeMap::new();
    kv.insert(0, (key, value));
    let mut past = PastState {
        seq_len: 4,
        conv: BTreeMap::new(),
        kv,
    };
    let dropped = past.truncate(2, 1, 2);
    assert_eq!(dropped, 2);
    assert_eq!(past.seq_len, 2);
    // Kept positions 2,3 → [4,5,6,7].
    assert_eq!(past.kv[&0].0, vec![4.0, 5.0, 6.0, 7.0]);
    assert_eq!(past.kv[&0].1, vec![40.0, 50.0, 60.0, 70.0]);
}

#[test]
fn past_truncate_is_a_noop_within_bounds() {
    let mut past = sample_past();
    assert_eq!(past.truncate(16, 1, 2), 0);
    assert_eq!(past.seq_len, 2);
    assert_eq!(past.kv[&0].0, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn past_truncate_leaves_mismatched_layer_alone() {
    // A layer whose data length does not match heads×seq×head_dim is left
    // untouched (never a panic, never a corrupt slice).
    let mut kv = BTreeMap::new();
    kv.insert(0, (vec![1.0, 2.0], vec![3.0, 4.0]));
    let mut past = PastState {
        seq_len: 4,
        conv: BTreeMap::new(),
        kv,
    };
    past.truncate(2, 1, 2);
    assert_eq!(past.seq_len, 2, "seq_len still advances");
    assert_eq!(past.kv[&0].0, vec![1.0, 2.0], "mismatched layer untouched");
}
