use super::*;
use std::sync::Arc;

use crate::node_store::new_node;
use crate::views::ParallelLedger;

/// A store whose nodes have all LOD tiers pre-filled (so assembling never
/// triggers lazy derivation), plus a `CountingBackend` summarizer to prove
/// no derivation happened.
fn pre_filled_store(contents: &[(&str, &str, &str, Option<&[f32]>)]) -> (Arc<ContentNodeStore>, usize) {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-prompt-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    let backend = Arc::new(crate::test_stubs::CountingBackend::new("derived"));
    let summarizer = crate::summarization::Summarizer::new(backend.clone(), 20);
    store.set_summarizer(summarizer);
    for (i, (role, req, text, emb)) in contents.iter().enumerate() {
        let mut node = new_node(
            NodeId::from_int(i as i64),
            "sess",
            req,
            role,
            text,
            Some(true),
        );
        // Pre-fill LOD1..LOD4 with deterministic per-tier text.
        for l in 1..=4 {
            while node.lod.len() <= l as usize {
                node.lod.push(String::new());
            }
            node.lod[l as usize] = format!("{l}-{req}");
        }
        if let Some(e) = emb {
            node.embedding = Some(e.to_vec());
        }
        store.record_content_node(&node).unwrap();
    }
    (store, backend.calls())
}

fn view_for(store: &Arc<ContentNodeStore>) -> ParallelLedger {
    ParallelLedger::for_session(Arc::clone(store), "sess")
}

const SIM: &[f32] = &[1.0, 0.0];
const OPP: &[f32] = &[0.0, 1.0];

#[test]
fn first_and_last_always_lod0_even_beyond_budget() {
    let (store, _) = pre_filled_store(&[
        ("user", "head", "head text that is long", Some(SIM)),
        ("assistant", "mid", "middle", Some(SIM)),
        ("user", "tail", "tail text that is long", Some(SIM)),
    ]);
    let view = view_for(&store);
    let worker = WorkerContext::new("analyst", "be thorough");
    // Tiny budget: intermediates get zero budget → only head + tail.
    let out = LedgerPromptAssembler.assemble(
        &view,
        &worker,
        &PromptBudget::new(0),
        None,
        &LodSpec::full(),
    );
    let first = store.session_node_ids("sess")[0];
    let last = *store.session_node_ids("sess").last().unwrap();
    assert_eq!(out.node_plan.first().unwrap(), &(first, Lod::LOD0));
    assert_eq!(out.node_plan.last().unwrap(), &(last, Lod::LOD0));
    assert_eq!(out.node_plan.len(), 2, "intermediate dropped at zero budget");
    assert!(out.system.contains("analyst"));
    assert!(out.system.contains("be thorough"));
}

#[test]
fn relevance_signal_renders_near_node_finer_than_distant_node() {
    let (store, _) = pre_filled_store(&[
        ("user", "near", "relevant near text", Some(SIM)),
        ("assistant", "mid1", "middle one", Some(OPP)),
        ("assistant", "mid2", "middle two", Some(SIM)),
        ("user", "tail", "tail text", Some(OPP)),
    ]);
    let view = view_for(&store);
    let ids = store.session_node_ids("sess");
    let near = ids[2]; // SIM (near)
    let far = ids[1]; // OPP (distant)
    let budget = PromptBudget::new(10_000); // plenty, no degradation
    let spec = LodSpec::full();

    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", ""),
        &budget,
        Some(&RelevanceSignal::query(SIM.to_vec())),
        &spec,
    );
    let near_lod = out
        .node_plan
        .iter()
        .find(|(id, _)| *id == near)
        .map(|(_, l)| *l)
        .unwrap();
    let far_lod = out
        .node_plan
        .iter()
        .find(|(id, _)| *id == far)
        .map(|(_, l)| *l)
        .unwrap();
    assert!(
        near_lod < far_lod,
        "near (relevant) node must render finer than distant node: {near_lod} vs {far_lod}"
    );
}

#[test]
fn no_signal_uses_uniform_mid_tier() {
    let (store, _) = pre_filled_store(&[
        ("user", "a", "aaaa", None),
        ("assistant", "b", "bbbb", None),
        ("assistant", "c", "cccc", None),
        ("user", "d", "dddd", None),
    ]);
    let view = view_for(&store);
    let budget = PromptBudget::new(10_000);
    let spec = LodSpec::full();
    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", ""),
        &budget,
        None,
        &spec,
    );
    let mid = spec.midpoint();
    let intermediates: Vec<&(NodeId, Lod)> =
        out.node_plan.iter().filter(|(id, _)| *id != out.node_plan[0].0 && *id != out.node_plan.last().unwrap().0).collect();
    assert_eq!(intermediates.len(), 2);
    assert!(
        intermediates.iter().all(|(_, l)| *l == mid),
        "uniform mid-tier expected, got {intermediates:?}"
    );
}

#[test]
fn budget_exhaustion_degrades_toward_coarser_never_exceeds_max() {
    let (store, _) = pre_filled_store(&[
        ("user", "a", "short head", None),
        ("assistant", "b", "short middle b", None),
        ("assistant", "c", "short middle c", None),
        ("user", "d", "short tail", None),
    ]);
    let view = view_for(&store);
    // A small budget that fits head+tail and only some intermediate text.
    let budget = PromptBudget::new(60);
    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", ""),
        &budget,
        None,
        &LodSpec::full(),
    );
    assert!(out.budget_used <= 60, "budget_used {} exceeded max_chars", out.budget_used);
    // Every intermediate plan Lod is within the band and coarser-or-equal to
    // the midpoint default (degradation only, never finer).
    let mid = LodSpec::full().midpoint();
    for (id, lod) in &out.node_plan {
        if *id == out.node_plan[0].0 || *id == out.node_plan.last().unwrap().0 {
            assert_eq!(*lod, Lod::LOD0, "anchors stay LOD0");
        } else {
            assert!(*lod >= mid, "intermediate degraded at least to mid, got {lod}");
        }
    }
}

#[test]
fn empty_view_does_not_panic() {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-prompt-empty-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    let view = ParallelLedger::for_session(store, "absent");
    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", "instr"),
        &PromptBudget::new(100),
        None,
        &LodSpec::full(),
    );
    assert!(out.body.is_empty());
    assert!(out.node_plan.is_empty());
    assert_eq!(out.budget_used, 0);
}

#[test]
fn assembling_never_triggers_lod_derivation() {
    let (store, calls) = pre_filled_store(&[
        ("user", "a", "aaaa", None),
        ("assistant", "b", "bbbb", None),
        ("user", "c", "cccc", None),
    ]);
    let view = view_for(&store);
    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", ""),
        &PromptBudget::new(10_000),
        None,
        &LodSpec::full(),
    );
    assert!(!out.body.is_empty());
    assert_eq!(
        calls, 0,
        "assembling a prompt must not trigger LOD derivation (tiers are pre-filled)"
    );
}

#[test]
fn max_chars_zero_yields_first_and_last_only() {
    let (store, _) = pre_filled_store(&[
        ("user", "a", "aaaa", None),
        ("assistant", "b", "bbbb", None),
        ("assistant", "c", "cccc", None),
        ("user", "d", "dddd", None),
    ]);
    let view = view_for(&store);
    let out = LedgerPromptAssembler.assemble(
        &view,
        &WorkerContext::new("x", ""),
        &PromptBudget::new(0),
        None,
        &LodSpec::full(),
    );
    assert_eq!(out.node_plan.len(), 2, "only head + tail at zero budget");
    assert!(out.node_plan.iter().all(|(_, l)| *l == Lod::LOD0));
}

#[test]
fn from_tokens_budget_is_conservative() {
    let b = PromptBudget::from_tokens(100, 4);
    assert_eq!(b.max_chars, 400);
    let b2 = PromptBudget::from_tokens(100, 0);
    assert_eq!(b2.max_chars, 100, "chars_per_token floored at 1");
    assert_eq!(PromptBudget::from_tokens_default(50).max_chars, 200);
}

#[test]
fn from_tokens_default_locks_8192_window() {
    // M4.1: the exact production window both callers use — integer equality.
    assert_eq!(PromptBudget::from_tokens_default(8192).max_chars, 32768);
    assert_eq!(PromptBudget::from_tokens(0, 4).max_chars, 0);
    assert_eq!(PromptBudget::from_tokens(0, 0).max_chars, 0);
}

#[test]
fn intermediate_budget_saturates_at_edges() {
    // M4.1: min(max - reserve, max - actual head/tail cost), saturating.
    let full = PromptBudget { max_chars: 100, reserve_head_tail: 30 };
    assert_eq!(full.intermediate_budget(20), 70);
    assert_eq!(full.intermediate_budget(90), 10);
    let over_reserved = PromptBudget { max_chars: 100, reserve_head_tail: 150 };
    assert_eq!(over_reserved.intermediate_budget(0), 0, "reserve > max");
    assert_eq!(full.intermediate_budget(200), 0, "head_tail_cost > max");
    let zero = PromptBudget::new(0);
    assert_eq!(zero.intermediate_budget(0), 0);
}

#[test]
fn lod_spec_interpolation_bounds() {
    let spec = LodSpec::full();
    assert_eq!(spec.interpolate(0, 5), Lod::LOD1);
    assert_eq!(spec.interpolate(4, 5), Lod::LOD5);
    assert_eq!(spec.interpolate(2, 5), Lod::LOD3);
}
