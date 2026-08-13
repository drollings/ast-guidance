//! `LedgerPromptAssembler` — the pure, deterministic prompt assembler (M1).
//!
//! This module is the **foundation** of the coordinator (M4): it turns a
//! worker's `speciality`/`instructions` plus a `LedgerView` into an optimized
//! context prompt, selecting a fidelity `Lod` per node from the ledger.
//!
//! It has **no I/O, no model calls, and no background dependency** — it is a
//! pure function object, fully unit-testable. It renders through the single
//! text-exit (`ContentNodeStore::lod_text` / `LedgerView::render` semantics) and never
//! triggers LOD derivation beyond what already exists on a node (assembling a
//! prompt must not pay a summarization cost).
//!
//! # Algorithm (M1.2)
//!
//! 1. **Always include first and last node at LOD0** (their `lod_text(0)`),
//!    even if that overruns the reserved head/tail budget — they are the
//!    highest-priority anchors.
//! 2. **Intermediates, greedy by relevance:** with a `RelevanceSignal`, sort
//!    intermediate nodes by cosine distance to the query embedding — more
//!    relevant nodes render finer (closer to `LodSpec::min`), less relevant
//!    coarser (closer to `LodSpec::max`). Without a signal, a uniform
//!    mid-tier default is used.
//! 3. **Respect the budget:** each intermediate starts at its relevance-assigned
//!    `Lod` and degrades toward coarser tiers until it fits the remaining
//!    budget; if even the coarsest tier does not fit, the node is dropped.
//!    First/last LOD0 are guaranteed regardless of budget.
//!
//! The [`AssembledPrompt::node_plan`] output is the key testable artifact: the
//! per-node fidelity decision, independent of the rendered text.

use fluent_db::vector::cosine_similarity;
use fluent_types::NodeId;

use crate::ledger::{LOD0_FULL_TEXT, LOD5_LABEL};
use crate::node_store::ContentNodeStore;
use crate::views::{Lod, LedgerView};

/// The fidelity band a worker accepts. Intermediate nodes render between
/// `max` (coarsest, e.g. `Lod::LOD5`) and `min` (finest, e.g. `Lod::LOD1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodSpec {
    /// The finest fidelity an intermediate node may render at.
    pub min: Lod,
    /// The coarsest fidelity an intermediate node may render at.
    pub max: Lod,
}

impl LodSpec {
    /// A full-fidelity band over the standard lazy tiers (LOD1 finest, LOD5
    /// coarsest).
    pub fn full() -> Self {
        Self {
            min: Lod::LOD1,
            max: Lod::LOD5,
        }
    }

    /// The discrete number of tiers in the band (inclusive of both ends).
    fn span(self) -> u8 {
        self.max.as_u8().saturating_sub(self.min.as_u8())
    }

    /// The uniform mid-tier default when no relevance signal is provided.
    fn midpoint(self) -> Lod {
        let mid = self.min.as_u8() + self.span() / 2;
        Lod::try_from(mid).unwrap_or(self.min)
    }

    /// A `Lod` in the band interpolated by a relevance rank `r` in `0..n`
    /// (0 = most relevant → finest, `n-1` = least relevant → coarsest).
    fn interpolate(self, r: usize, n: usize) -> Lod {
        let span = usize::from(self.span());
        let denom = n.saturating_sub(1).max(1);
        let offset = (span * r) / denom;
        let level = self.min.as_u8() + offset as u8;
        Lod::try_from(level.min(LOD5_LABEL)).unwrap_or(self.min)
    }
}

/// A worker's context-window budget.
///
/// `max_chars` is the total character ceiling; `reserve_head_tail` is the
/// amount guaranteed to the first/last LOD0 anchors (they are included
/// regardless — even overrunning the reserve). Intermediates compete for the
/// remaining budget.
#[derive(Debug, Clone, Copy)]
pub struct PromptBudget {
    pub max_chars: usize,
    pub reserve_head_tail: usize,
}

impl PromptBudget {
    /// A budget of `max_chars` with no reserved head/tail.
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            reserve_head_tail: 0,
        }
    }

    /// Build a budget from a token count using a conservative `chars_per_token`
    /// (no tokenizer dependency — a straight character estimate).
    pub fn from_tokens(tokens: usize, chars_per_token: usize) -> Self {
        Self::new(tokens.saturating_mul(chars_per_token.max(1)))
    }

    /// A budget from a token count at the conservative 4 chars/token default.
    pub fn from_tokens_default(tokens: usize) -> Self {
        Self::from_tokens(tokens, 4)
    }

    /// The budget available to intermediate nodes: the ceiling minus what the
    /// first/last anchors actually consume (saturating), bounded by the
    /// reserve. Intermediate text is kept within `max_chars - head/tail cost`.
    fn intermediate_budget(&self, head_tail_cost: usize) -> usize {
        let by_reserve = self.max_chars.saturating_sub(self.reserve_head_tail);
        let by_actual = self.max_chars.saturating_sub(head_tail_cost);
        by_reserve.min(by_actual)
    }
}

/// A worker's system instructions: its speciality and explicit instructions.
#[derive(Debug, Clone, Default)]
pub struct WorkerContext {
    /// The worker's speciality (e.g. `"code reviewer"`), used to focus the
    /// system prompt.
    pub speciality: String,
    /// The worker's explicit instructions, from config or a model entry.
    pub instructions: String,
}

impl WorkerContext {
    /// A new worker context with the given speciality and instructions.
    pub fn new(speciality: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            speciality: speciality.into(),
            instructions: instructions.into(),
        }
    }

    /// The composed system prompt.
    pub(crate) fn system_prompt(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.speciality.is_empty() {
            parts.push(format!("You are a {}.", self.speciality));
        }
        if !self.instructions.is_empty() {
            parts.push(self.instructions.clone());
        }
        parts.join("\n\n")
    }
}

/// The relevance signal for fidelity selection: `None` (uniform fidelity) or a
/// query/task embedding `Vec<f32>`.
#[derive(Debug, Clone)]
pub enum RelevanceSignal {
    /// Uniform fidelity — every intermediate renders at the band midpoint.
    None,
    /// A query/task embedding; intermediates are ranked by cosine similarity
    /// to it (reused from `ContentNode.embedding` / `ContentNodeStore::knn_search`).
    Query(Vec<f32>),
}

impl RelevanceSignal {
    /// A query embedding signal.
    pub fn query(embedding: Vec<f32>) -> Self {
        Self::Query(embedding)
    }

    fn embedding(&self) -> Option<&[f32]> {
        match self {
            Self::Query(v) => Some(v),
            Self::None => None,
        }
    }
}

/// The assembled prompt output.
#[derive(Debug, Clone)]
pub struct AssembledPrompt {
    /// The worker's composed system prompt.
    pub system: String,
    /// The ordered body lines (one per included node), rendered via the single
    /// text-exit.
    pub body: String,
    /// The number of characters consumed by the rendered body (node text +
    /// separators).
    pub budget_used: usize,
    /// The per-node fidelity plan in render order — the key testable artifact.
    pub node_plan: Vec<(NodeId, Lod)>,
}

/// Render a node at the requested `Lod`, degrading to the nearest **already
/// cached** coarser tier, then to LOD0 — without ever triggering lazy LOD
/// derivation (assembling a prompt must not pay a summarization cost).
///
/// Every actual text read goes through `ContentNodeStore::lod_text` (the single
/// text-exit), mirroring `LedgerView::render`'s "degrade to LOD0 on error"
/// contract.
fn render_node(store: &ContentNodeStore, id: NodeId, requested: Lod) -> String {
    let requested_level = requested.as_u8();
    // Eager tiers (LOD0/LOD5) are always present — short-circuit to the exit.
    if requested_level == LOD0_FULL_TEXT || requested_level == LOD5_LABEL {
        return store.lod_text(id, requested_level).unwrap_or_default();
    }
    // A lazy tier is only rendered if already cached on the node (never
    // triggers derivation).
    let cached_level = store
        .snapshot(id)
        .and_then(|n| {
            (requested_level..=LOD5_LABEL).find(|l| {
                n.lod
                    .get(*l as usize)
                    .is_some_and(|t| !t.is_empty())
            })
        });
    let level = cached_level.unwrap_or(LOD0_FULL_TEXT);
    store.lod_text(id, level).unwrap_or_default()
}

/// Estimate a node's rendered size at a given `Lod` (what actually fits the
/// budget) without triggering derivation.
fn estimate(store: &ContentNodeStore, id: NodeId, lod: Lod) -> usize {
    render_node(store, id, lod).chars().count()
}

/// The deterministic prompt assembler — a pure function object (M1).
#[derive(Debug, Clone, Copy, Default)]
pub struct LedgerPromptAssembler;

impl LedgerPromptAssembler {
    /// Assemble a worker's optimized context from a ledger view.
    ///
    /// # Arguments
    /// - `view` — the read surface over the shared store (any `LedgerView`).
    /// - `worker` — the worker's speciality + instructions.
    /// - `budget` — the context-window budget.
    /// - `relevance` — an optional relevance signal (`None`/`Some(None)` =
    ///   uniform fidelity; `Some(Query)` = relevance-ranked).
    /// - `lod_spec` — the fidelity band for intermediate nodes.
    ///
    /// First/last nodes always render at LOD0; intermediates are
    /// relevance/budget-selected within the band.
    pub fn assemble(
        &self,
        view: &dyn LedgerView,
        worker: &WorkerContext,
        budget: &PromptBudget,
        relevance: Option<&RelevanceSignal>,
        lod_spec: &LodSpec,
    ) -> AssembledPrompt {
        let store = view.store();
        let ids: Vec<NodeId> = view
            .node_ids()
            .into_iter()
            .filter(|id| !view.exclude(*id))
            .collect();

        if ids.is_empty() {
            return AssembledPrompt {
                system: worker.system_prompt(),
                body: String::new(),
                budget_used: 0,
                node_plan: Vec::new(),
            };
        }

        let first = ids[0];
        let last = ids[ids.len() - 1];

        // Intermediates, in render order.
        let intermediates: Vec<NodeId> = if ids.len() > 2 {
            ids[1..ids.len() - 1].to_vec()
        } else {
            Vec::new()
        };

        // Assign each intermediate a target Lod within the band.
        let targets: Vec<(NodeId, Lod)> =
            if let Some(query) = relevance.and_then(RelevanceSignal::embedding) {
                // Relevance-ranked: sort by similarity descending (most relevant
                // first → finest). Nodes without an embedding are least relevant.
                let mut ranked: Vec<(NodeId, f32)> = intermediates
                    .iter()
                    .map(|id| {
                        let sim = store
                            .snapshot(*id)
                            .and_then(|n| n.embedding.as_ref().map(|e| cosine_similarity(query, e)))
                            .unwrap_or(f32::MIN);
                        (*id, sim)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                ranked
                    .into_iter()
                    .enumerate()
                    .map(|(r, (id, _))| (id, lod_spec.interpolate(r, intermediates.len())))
                    .collect()
            } else {
                let mid = lod_spec.midpoint();
                intermediates.iter().map(|id| (*id, mid)).collect()
            };

        // Budget-fit the intermediates: degrade each toward coarser tiers until
        // it fits; drop it if even the coarsest does not fit. Priority order for
        // fitting is relevance order (most relevant kept first).
        let head_tail_cost = estimate(store, first, Lod::LOD0)
            + estimate(store, last, Lod::LOD0);
        let inter_budget = budget.intermediate_budget(head_tail_cost);
        let mut used = head_tail_cost;
        let mut fitted: Vec<(NodeId, Lod)> = Vec::new();
        for (id, target) in targets {
            let mut lod = target;
            loop {
                let cost = estimate(store, id, lod);
                if used + cost <= inter_budget {
                    used += cost;
                    fitted.push((id, lod));
                    break;
                }
                if lod.as_u8() >= lod_spec.max.as_u8() {
                    // Even the coarsest tier does not fit — drop the node.
                    break;
                }
                lod = Lod::try_from(lod.as_u8() + 1).unwrap_or(lod);
            }
        }

        // Reconstruct the render plan in original order: first, fitted
        // intermediates (kept in render order), last.
        let mut node_plan: Vec<(NodeId, Lod)> = Vec::with_capacity(ids.len());
        node_plan.push((first, Lod::LOD0));
        let fitted_by_id: std::collections::HashMap<NodeId, Lod> =
            fitted.into_iter().collect();
        for id in &intermediates {
            if let Some(lod) = fitted_by_id.get(id) {
                node_plan.push((*id, *lod));
            }
        }
        node_plan.push((last, Lod::LOD0));

        // Render the body lines in plan order.
        let mut lines: Vec<String> = Vec::with_capacity(node_plan.len());
        for (id, lod) in &node_plan {
            let node = store.snapshot(*id);
            let name = node.as_ref().map(|n| n.name.to_string()).unwrap_or_default();
            let role = node
                .as_ref()
                .and_then(|n| n.role.clone())
                .unwrap_or_default();
            let text = render_node(store, *id, *lod);
            lines.push(format!("[{role}] {name}: {text}"));
        }
        let body = lines.join("\n");
        // `budget_used` measures the content consumed: the sum of the rendered
        // node-text characters (not prefixes/separators), so it is comparable to
        // `max_chars` and the intermediate fitting budget.
        let budget_used = node_plan
            .iter()
            .map(|(id, lod)| estimate(store, *id, *lod))
            .sum();

        AssembledPrompt {
            system: worker.system_prompt(),
            body,
            budget_used,
            node_plan,
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn lod_spec_interpolation_bounds() {
        let spec = LodSpec::full();
        assert_eq!(spec.interpolate(0, 5), Lod::LOD1);
        assert_eq!(spec.interpolate(4, 5), Lod::LOD5);
        assert_eq!(spec.interpolate(2, 5), Lod::LOD3);
    }
}
