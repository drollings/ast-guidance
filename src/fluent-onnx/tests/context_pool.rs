use super::*;
use crate::llm::tests::session_with_stub;

fn profile(n_ctx: u64) -> OnnxContextProfile {
    OnnxContextProfile {
        group: "g".into(),
        n_ctx,
        max_ctx: None,
        pinned: false,
        resume: false,
    }
}

/// Head-0 per-position markers of a stub KV layer (row-major
/// `[head][position][head_dim]`; head 0's positions start at `p * 64`).
fn head0_markers(past: &crate::context::PastState, layer: usize) -> Vec<f32> {
    (0..past.seq_len).map(|p| past.kv[&layer].0[p * 64]).collect()
}

#[test]
fn ensure_context_creates_on_demand_and_is_idempotent() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session, "onnx/llm");
    assert!(pool.is_empty());

    let a1 = pool.ensure_context("scratch", profile(64));
    let a2 = pool.ensure_context("scratch", profile(64));
    assert!(Arc::ptr_eq(&a1, &a2), "second resolve reuses the context");
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.context("scratch").expect("context").name(), "scratch");
    assert_eq!(pool.model_key(), "onnx/llm");
}

#[test]
fn two_contexts_interleave_on_one_loaded_session_with_independent_kv() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session.clone(), "onnx/llm");

    let ctx_a = pool.ensure_context("a", profile(64));
    let ctx_b = pool.ensure_context("b", profile(64));

    // Interleave prefill + decode on the same session — the KV must stay
    // per-context.
    session.prefill(&ctx_a, &[1, 2, 3]).unwrap();
    session.prefill(&ctx_b, &[4, 5]).unwrap();
    session.decode_step(&ctx_a, 10).unwrap();
    session.decode_step(&ctx_b, 20).unwrap();
    session.decode_step(&ctx_a, 11).unwrap();

    let a = ctx_a.past().expect("a has KV");
    let b = ctx_b.past().expect("b has KV");
    assert_eq!(a.seq_len, 5, "a: 3 prefill + 2 decode");
    assert_eq!(b.seq_len, 3, "b: 2 prefill + 1 decode");

    // The stub marks each KV position with its source token (head 0): a
    // accumulates [1,2,3,10,11], b [4,5,20] — fully independent, despite
    // interleaving on the shared session.
    let a_keys = head0_markers(&a, 2);
    let b_keys = head0_markers(&b, 2);
    assert_eq!(a_keys, vec![1.0, 2.0, 3.0, 10.0, 11.0]);
    assert_eq!(b_keys, vec![4.0, 5.0, 20.0]);
    assert_ne!(a_keys, b_keys, "contexts must not share KV");
}

#[test]
fn decode_reaches_n_ctx_truncates_oldest_kv() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session.clone(), "onnx/llm");
    let ctx = pool.ensure_context("c", profile(4));

    session.prefill(&ctx, &[1, 2, 3]).unwrap();
    session.decode_step(&ctx, 10).unwrap(); // 4 ≤ n_ctx → no truncate
    let mid = ctx.past().unwrap();
    assert_eq!(mid.seq_len, 4);
    let mid_keys = head0_markers(&mid, 2);
    assert_eq!(mid_keys, vec![1.0, 2.0, 3.0, 10.0]);

    session.decode_step(&ctx, 11).unwrap(); // would be 5 > 4 → rolling window
    let past = ctx.past().unwrap();
    assert_eq!(past.seq_len, 4, "window stays at n_ctx");
    let keys = head0_markers(&past, 2);
    assert_eq!(keys, vec![2.0, 3.0, 10.0, 11.0], "oldest KV position dropped");
}

#[test]
fn prefill_overlong_prompt_truncates_to_n_ctx() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session.clone(), "onnx/llm");
    let ctx = pool.ensure_context("c", profile(4));

    session.prefill(&ctx, &[1, 2, 3, 4, 5]).unwrap();
    let past = ctx.past().unwrap();
    assert_eq!(past.seq_len, 4, "prompt truncated to the window");
    let keys = head0_markers(&past, 2);
    assert_eq!(keys, vec![2.0, 3.0, 4.0, 5.0], "most recent positions kept");
}

#[test]
fn pool_resize_is_bounded_by_max_ctx() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session, "onnx/llm");
    pool.ensure_context(
        "capped",
        OnnxContextProfile {
            max_ctx: Some(4096),
            ..profile(2048)
        },
    );
    assert!(pool.resize("capped", 8192).is_err(), "growth past max_ctx refused");
    pool.resize("capped", 4096).expect("at the cap is fine");
    assert_eq!(pool.context("capped").unwrap().n_ctx(), 4096);

    assert!(pool.resize("nope", 4096).is_err(), "unknown context is loud");
}

#[test]
fn destroy_frees_the_context_and_its_kv() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session.clone(), "onnx/llm");
    let ctx = pool.ensure_context("a", profile(64));
    session.prefill(&ctx, &[1, 2]).unwrap();
    assert!(ctx.past().is_some());

    let removed = pool.destroy("a");
    assert!(removed.is_some());
    assert!(ctx.past().is_none(), "destroy clears the KV");
    assert!(pool.context("a").is_none());
    assert!(pool.destroy("a").is_none(), "destroy of an absent context is a no-op");
}

#[test]
fn residency_rows_report_each_context() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session.clone(), "onnx/llm");
    let ctx = pool.ensure_context("default", profile(64));
    session.prefill(&ctx, &[1, 2]).unwrap();

    let rows = pool.residency_rows();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.context_key, "onnx/llm:default");
    assert_eq!(row.runtime, LlmRuntime::Onnx);
    assert_eq!(row.state, "loaded");
    assert_eq!(row.vram_bytes, 0, "RAM-resident context");
    assert_eq!(row.n_ctx, 64);
    assert_eq!(row.parallel, 1);
    assert!(!row.pinned);
    assert!(row.last_used > 0, "prefill touched the context");

    // An empty context renders as "sleeping" with zero bytes.
    pool.ensure_context("idle", profile(64));
    let rows = pool.residency_rows();
    let idle = rows.iter().find(|r| r.context_key.ends_with(":idle")).unwrap();
    assert_eq!(idle.state, "sleeping");
    assert_eq!(idle.total_bytes, 0);
}

#[test]
fn pool_defaults_zero_n_ctx_to_the_default_window() {
    let session = session_with_stub(42);
    let pool = OnnxContextPool::new(session, "onnx/llm");
    let ctx = pool.ensure_context("a", OnnxContextProfile::default());
    assert_eq!(ctx.n_ctx(), crate::context::DEFAULT_ONNX_CONTEXT_TOKENS);
    assert_eq!(ctx.group(), "default");
}
