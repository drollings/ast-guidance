//! Deterministic salience prefilter + model-ranks-shortlist (ROADMAP M6).
//!
//! The local model ranks only a **deterministic shortlist** — never the full
//! session context. Four salience signals, each a pure function of ledger data
//! already on disk, order candidates deterministically (zero model calls); a
//! single grammar-constrained onnx call then scores the top-K shortlist.
//!
//! - [`SalienceSignal`] / [`SalienceScorer`] — the pure scoring core (hermetic;
//!   tests construct signals by hand).
//! - [`pagerank`] — the minimal "graph analytics — PageRank" backlog item,
//!   scoped to candidate centrality: a fixed-iteration power loop over the
//!   interlingua co-reference graph, composed from `interlingua_index`
//!   co-occurrence + `ledger.parent_id` (there is **no** separate coral edge
//!   table — the graph is built from the ledger's own rows).
//! - [`SalienceSource`] / [`LedgerSalienceProvider`] — the signal seam; the
//!   provider reads a read-only snapshot of the ledger.
//! - [`rank_candidates`] — prefilter to top-K by salience, then **one**
//!   grammar-constrained call over the shortlist only. The model never sees
//!   the full candidate set.
//! - [`SalienceRanker`] — the composable ranking step (source + backend + label
//!   source) the M5 retrieval tools' live agent-loop dispatch will call to
//!   narrow a candidate pool before presenting to a subagent.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use fluent_db::error::DbError;
use fluent_db::store::SqliteStore;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_types::NodeId;

use crate::node_store::ContentNodeStore;

/// The salience weight vector, index-aligned with [`SalienceSignal`]'s fields:
/// `[frame_frequency, interlingua_centrality, recency, reference_count]`.
/// Frequency + centrality lead (the content signals); recency and
/// reference-count tie-break (the activity signals).
pub const WEIGHTS: [f64; 4] = [0.4, 0.3, 0.2, 0.1];

/// The deterministic shortlist size the model is asked to rank.
pub const SALIENCE_SHORTLIST_K: usize = 8;

/// Fixed iteration bound for the minimal PageRank power loop.
pub const PAGERANK_ITERATIONS: usize = 20;

/// The PageRank teleport probability.
pub const PAGERANK_ALPHA: f64 = 0.85;

/// The interlingua role naming a sentence's predicate (the `frame_frequency`
/// key). This is the durable-frame fallback: while M3 frames are not yet wired
/// into the live pipeline, predicate rows in `interlingua_index` carry the same
/// signal.
const PREDICATE_ROLE: &str = "predicate";

/// Interlingua roles counted as node↔id co-references (the centrality graph).
/// The `sense`/`correction` pattern-cache rows (node_id = 0) use other roles
/// and are excluded by construction.
const INDEX_ROLES: &[&str] = &[
    "predicate",
    "subject",
    "direct_object",
    "indirect_object",
    "concept",
];

/// The normalized `[0,1]` salience components for one candidate node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SalienceSignal {
    pub frame_frequency: f64,
    pub interlingua_centrality: f64,
    pub recency: f64,
    pub reference_count: f64,
}

impl Default for SalienceSignal {
    fn default() -> Self {
        Self {
            frame_frequency: 0.0,
            interlingua_centrality: 0.0,
            recency: 0.0,
            reference_count: 0.0,
        }
    }
}

/// The weighted salience scorer — a pure function of a [`SalienceSignal`].
#[derive(Debug, Clone, Copy)]
pub struct SalienceScorer {
    weights: [f64; 4],
}

impl SalienceScorer {
    /// The default scorer with [`WEIGHTS`].
    #[must_use]
    pub fn new() -> Self {
        Self { weights: WEIGHTS }
    }

    /// A scorer with an explicit weight vector (tests / tuning).
    #[must_use]
    pub fn with_weights(weights: [f64; 4]) -> Self {
        Self { weights }
    }

    /// The weighted sum of the four signals — in `[0,1]` when each signal is
    /// `[0,1]`. Deterministic: delegates to `common_core::score::weighted_dot`
    /// (plain `f64` multiply-accumulate over the shared prefix).
    #[must_use]
    pub fn score(&self, signal: &SalienceSignal) -> f64 {
        common_core::score::weighted_dot(
            &[
                signal.frame_frequency,
                signal.interlingua_centrality,
                signal.recency,
                signal.reference_count,
            ],
            &self.weights,
        )
    }
}

impl Default for SalienceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// An adjacency map for the candidate-centrality graph (undirected).
pub type NodeGraph = BTreeMap<i64, Vec<i64>>;

/// Minimal PageRank via a fixed-iteration power loop over an undirected graph.
///
/// Deterministic in `f64` (no parallel reduction — the same graph always yields
/// the same ranks) and normalized so the top node scores `1.0`. Dangling nodes
/// (empty adjacency) redistribute their mass evenly through the teleport.
///
/// This is the minimal form of the "graph analytics — PageRank" backlog item,
/// scoped to candidate centrality. It walks the ledger's own graph
/// (`interlingua_index` co-occurrence + `parent_id`); it is not a general
/// graph-analytics subsystem and there is no `fluent-dag` PageRank primitive
/// to compose (the power loop is a small numeric routine, not a graph
/// algorithm the DAG crate provides).
#[must_use]
pub fn pagerank(graph: &NodeGraph, iterations: usize, alpha: f64) -> BTreeMap<i64, f64> {
    let n = graph.len().max(1);
    let mut ranks: BTreeMap<i64, f64> = graph.keys().map(|&k| (k, 1.0 / n as f64)).collect();
    for _ in 0..iterations {
        let teleport = (1.0 - alpha) / n as f64;
        let mut dangling = 0.0;
        let mut out_deg: BTreeMap<i64, f64> = BTreeMap::new();
        for (&u, adj) in graph {
            if adj.is_empty() {
                dangling += ranks[&u];
            }
            out_deg.insert(u, adj.len().max(1) as f64);
        }
        let mut next: BTreeMap<i64, f64> = BTreeMap::new();
        for (&u, adj) in graph {
            let mut acc = teleport + alpha * dangling / n as f64;
            for &v in adj {
                acc += alpha * ranks[&v] / out_deg[&v];
            }
            next.insert(u, acc);
        }
        ranks = next;
    }
    let max = ranks.values().copied().fold(0.0_f64, f64::max);
    if max > 0.0 {
        for rank in ranks.values_mut() {
            *rank /= max;
        }
    }
    ranks
}

/// The salience signal seam [`rank_candidates`] pre-filters over. The ledger
/// provider implements it over a read-only snapshot; tests implement fixtures.
pub trait SalienceSource: Send + Sync {
    /// One pass over the candidates → their signals, in input order.
    fn signals_for(&self, candidates: &[NodeId]) -> Vec<(NodeId, SalienceSignal)>;
}

/// The ledger-backed salience source: reads a read-only snapshot of the
/// ledger's `interlingua_index` + `ledger` tables and the shared node store.
///
/// Every signal is a pure function of ledger data — no new schema, no new
/// index, no staleness scheduler. An ephemeral store (no durable backing)
/// yields all-zero signals (fail-open: `rank_candidates` then orders by node
/// id, never an error).
pub struct LedgerSalienceProvider {
    store: Arc<ContentNodeStore>,
    /// Unix seconds used for the recency decay (injectable for tests).
    now: u64,
    session_id: Option<String>,
}

impl LedgerSalienceProvider {
    /// A provider over the shared node store, at `now` unix-seconds.
    #[must_use]
    pub fn new(store: Arc<ContentNodeStore>, now: u64) -> Self {
        Self { store, now, session_id: None }
    }

    /// Scoped to a session; snapshot loads only that session's rows (see M7).
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

impl SalienceSource for LedgerSalienceProvider {
    fn signals_for(&self, candidates: &[NodeId]) -> Vec<(NodeId, SalienceSignal)> {
        let zero = |id: NodeId| (id, SalienceSignal::default());
        let Some(store) = self.store.shared_sqlite() else {
            return candidates.iter().copied().map(zero).collect();
        };
        let snapshot = match LedgerSnapshot::load(&store, self.session_id.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    target: "router.ranking",
                    error = %e,
                    "salience snapshot load failed — all signals zero (fail-open)",
                );
                return candidates.iter().copied().map(zero).collect();
            }
        };
        // Normalize frequency + reference-count across the candidate set so
        // each signal is [0,1] relative to the pool being ranked.
        let mut freq_max = 0.0f64;
        let mut ref_max = 0.0f64;
        let mut signals: Vec<(NodeId, SalienceSignal)> = candidates
            .iter()
            .map(|&id| {
                let node = id.as_int();
                let freq = snapshot.frame_frequency(node) as f64;
                let refs = snapshot.reference_count(node) as f64;
                freq_max = freq_max.max(freq);
                ref_max = ref_max.max(refs);
                (
                    id,
                    SalienceSignal {
                        frame_frequency: freq,
                        interlingua_centrality: snapshot.centrality.get(&node).copied().unwrap_or(0.0),
                        recency: snapshot.recency(node, self.now),
                        reference_count: refs,
                    },
                )
            })
            .collect();
        for (_, s) in &mut signals {
            if freq_max > 0.0 {
                s.frame_frequency /= freq_max;
            }
            if ref_max > 0.0 {
                s.reference_count /= ref_max;
            }
        }
        signals
    }
}

/// A read-only snapshot of the ledger's salience inputs: the interlingua
/// co-reference graph + per-node predicate ids + parent in-degrees + created
/// times, loaded in a handful of queries and reduced in pure Rust.
struct LedgerSnapshot {
    /// node → its predicate interlingua ids (`role = 'predicate'`).
    node_pred_ids: BTreeMap<i64, BTreeSet<i64>>,
    /// id → nodes carrying it as a predicate (the `frame_frequency` source).
    pred_id_nodes: BTreeMap<i64, BTreeSet<i64>>,
    /// node → distinct co-referencing nodes (the reference-count source).
    co_ref: BTreeMap<i64, BTreeSet<i64>>,
    /// node → parent in-degree (children pointing at it).
    parent_indegree: BTreeMap<i64, usize>,
    /// The power-iteration centrality over the co-reference + parent graph.
    centrality: BTreeMap<i64, f64>,
    /// node → created_at unix seconds.
    created_at: BTreeMap<i64, u64>,
}

impl LedgerSnapshot {
    /// Load every salience input in one pass over the durable tables, scoped to `session_id` when Some.
    fn load(store: &SqliteStore, session_id: Option<&str>) -> Result<Self, DbError> {
        let roles: Vec<&str> = INDEX_ROLES.to_vec();
        let mut role_params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(roles.len());
        for role in &roles {
            role_params.push(role);
        }
        let placeholders = std::iter::repeat_n("?", roles.len())
            .collect::<Vec<_>>()
            .join(",");
        // Scoped interlingua read: join on ledger session when session is present.
        let rows: Vec<(i64, i64, String)> = if let Some(sid) = session_id {
            let mut params: Vec<&dyn rusqlite::ToSql> = role_params.clone();
            params.push(&sid);
            store.query_rows(
                &format!(
                    "SELECT i.node_id, i.interlingua_id, i.role FROM interlingua_index i \
                     JOIN ledger l ON l.node_id = i.node_id \
                     WHERE i.role IN ({placeholders}) AND i.review_status = 'unreviewed' AND l.session_id = ?"
                ),
                &params,
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?
        } else {
            store.query_rows(
                &format!(
                    "SELECT node_id, interlingua_id, role FROM interlingua_index \
                     WHERE role IN ({placeholders}) AND review_status = 'unreviewed'"
                ),
                &role_params,
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?
        };

        let mut id_nodes: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        let mut node_pred_ids: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        let mut pred_id_nodes: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for (node, id, role) in rows {
            id_nodes.entry(id).or_default().insert(node);
            if role == PREDICATE_ROLE {
                node_pred_ids.entry(node).or_default().insert(id);
                pred_id_nodes.entry(id).or_default().insert(node);
            }
        }

        // Undirected co-reference graph: a path linking the nodes sharing each
        // id (sorted by node id) — a bounded, deterministic connectivity
        // approximation of the full clique.
        let mut graph: NodeGraph = BTreeMap::new();
        let mut co_ref: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
        for nodes in id_nodes.values() {
            let list: Vec<i64> = nodes.iter().copied().collect();
            for w in list.windows(2) {
                add_undirected(&mut graph, w[0], w[1]);
                co_ref.entry(w[0]).or_default().insert(w[1]);
                co_ref.entry(w[1]).or_default().insert(w[0]);
            }
        }

        // Parent edges + parent in-degree (reference_count's first half).
        let parents: Vec<(i64, i64)> = if let Some(sid) = session_id {
            store.query_rows(
                "SELECT parent_id, node_id FROM ledger WHERE parent_id IS NOT NULL AND session_id = ?",
                &[&sid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
        } else {
            store.query_rows(
                "SELECT parent_id, node_id FROM ledger WHERE parent_id IS NOT NULL",
                &[],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
        };
        let mut parent_indegree: BTreeMap<i64, usize> = BTreeMap::new();
        for (parent, child) in &parents {
            *parent_indegree.entry(*parent).or_default() += 1;
            add_undirected(&mut graph, *parent, *child);
        }
        for adj in graph.values_mut() {
            adj.sort_unstable();
            adj.dedup();
        }

        let created: Vec<(i64, i64)> = if let Some(sid) = session_id {
            store.query_rows(
                "SELECT node_id, created_at FROM ledger WHERE session_id = ?",
                &[&sid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
        } else {
            store.query_rows(
                "SELECT node_id, created_at FROM ledger",
                &[],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
        };
        let created_at: BTreeMap<i64, u64> =
            created.into_iter().map(|(n, t)| (n, t as u64)).collect();

        let centrality = pagerank(&graph, PAGERANK_ITERATIONS, PAGERANK_ALPHA);
        Ok(Self {
            node_pred_ids,
            pred_id_nodes,
            co_ref,
            parent_indegree,
            centrality,
            created_at,
        })
    }

    /// The number of distinct nodes across the store whose predicate set
    /// intersects this node's predicate set (the `frame_frequency` signal).
    fn frame_frequency(&self, node: i64) -> usize {
        let Some(preds) = self.node_pred_ids.get(&node) else {
            return 0;
        };
        let mut union: BTreeSet<i64> = BTreeSet::new();
        for id in preds {
            if let Some(nodes) = self.pred_id_nodes.get(id) {
                union.extend(nodes.iter().copied());
            }
        }
        union.len()
    }

    /// The derived reference count: parent in-degree plus distinct
    /// co-referencing nodes (there is no durable ref-count column on
    /// `ContentNode` — ROADMAP M6).
    fn reference_count(&self, node: i64) -> usize {
        let parents = self.parent_indegree.get(&node).copied().unwrap_or(0);
        let co = self.co_ref.get(&node).map_or(0, BTreeSet::len);
        parents + co
    }

    /// `1 / (1 + age_days)` over `created_at` — `1.0` for a brand-new node,
    /// decaying to `0`; `0.0` when the node has no timestamp.
    fn recency(&self, node: i64, now: u64) -> f64 {
        let Some(created) = self.created_at.get(&node) else {
            return 0.0;
        };
        let age_days = now.saturating_sub(*created) as f64 / 86_400.0;
        1.0 / (1.0 + age_days)
    }
}

/// Add an undirected edge (dedup happens once, after the graph is built).
fn add_undirected(graph: &mut NodeGraph, a: i64, b: i64) {
    if a == b {
        return;
    }
    graph.entry(a).or_default().push(b);
    graph.entry(b).or_default().push(a);
}

/// One candidate after the ranking pass.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedCandidate {
    pub node_id: NodeId,
    /// The deterministic salience score (the prefilter).
    pub salience: f64,
    /// The model's shortlist score, when the ranking call succeeded.
    pub model_score: Option<f64>,
}

/// Prefilter candidates by deterministic salience to a top-K shortlist, then
/// let the local model rank **only** that shortlist in one grammar-constrained
/// call. The model never sees the full candidate set.
///
/// - `source` supplies the salience signals (the ledger provider in
///   production; a fixture in tests).
/// - `backend` is the onnx `ChatBackend`; `None` → salience-only ordering
///   (no model call).
/// - `label_of` supplies a bounded, human-readable snippet per candidate for
///   the ranking prompt (the store's LOD5 label), so the model scores a short,
///   bounded description — never full context.
///
/// A failed / unparseable / hallucinating ranking call degrades to the
/// salience order (never an error, never an empty result, never an unbounded
/// prompt).
#[must_use]
pub fn rank_candidates(
    query: &str,
    candidates: &[NodeId],
    source: &dyn SalienceSource,
    backend: Option<Arc<dyn ChatBackend>>,
    label_of: &dyn Fn(NodeId) -> Option<String>,
) -> Vec<RankedCandidate> {
    let scorer = SalienceScorer::new();
    let mut scored: Vec<RankedCandidate> = source
        .signals_for(candidates)
        .into_iter()
        .map(|(id, signal)| RankedCandidate {
            node_id: id,
            salience: scorer.score(&signal),
            model_score: None,
        })
        .collect();
    // Deterministic salience order: score desc, then node id asc.
    // M7: intentionally NOT `common_core::score::top_k_by_score` — the shared
    // primitive is single-key over `f32` scores with no tiebreak hook, while
    // this order is `f64` salience desc + id-asc tiebreak (locked by
    // `rank_candidates_tied_salience_breaks_by_id_asc`). Narrowing to `f32`
    // could reorder near-ties: a behavior change. Do not "migrate" this.
    scored.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.as_int().cmp(&b.node_id.as_int()))
    });
    let shortlist: Vec<RankedCandidate> = scored.into_iter().take(SALIENCE_SHORTLIST_K).collect();

    let Some(backend) = backend else {
        return shortlist;
    };
    match rank_with_model(query, &shortlist, &backend, label_of) {
        Some(scores) => {
            // Combined order: model score desc, then salience desc, then id asc.
            // M7: multi-key — out of `top_k_by_score`'s single-key shape (same
            // reason as the salience sort above). Missing model scores default
            // to 0.0 (fail-open: unscored members sink but survive).
            let mut ranked = shortlist.clone();
            for r in &mut ranked {
                r.model_score = scores.get(&r.node_id.as_int()).copied();
            }
            ranked.sort_by(|a, b| {
                let am = a.model_score.unwrap_or(0.0);
                let bm = b.model_score.unwrap_or(0.0);
                bm.partial_cmp(&am)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        b.salience
                            .partial_cmp(&a.salience)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.node_id.as_int().cmp(&b.node_id.as_int()))
            });
            ranked
        }
        None => shortlist,
    }
}

/// The ranking call's `response_format.schema`: a JSON array of
/// `{node_id, score}` objects. `grammar_from_json_schema` turns this into a
/// `JsonArrayGrammar` on the onnx path, so structurally-invalid ranking output
/// is impossible at generation time.
fn ranking_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "node_id": { "type": "integer" },
                "score": { "type": "number" }
            },
            "required": ["node_id", "score"]
        }
    })
}

/// One grammar-constrained ranking call over the shortlist. The prompt lists
/// only shortlist entries (never the full candidate set); the response is
/// parsed + validated against the shortlist. `None` on a failed / unparseable
/// call → the caller keeps the salience order.
fn rank_with_model(
    query: &str,
    shortlist: &[RankedCandidate],
    backend: &Arc<dyn ChatBackend>,
    label_of: &dyn Fn(NodeId) -> Option<String>,
) -> Option<BTreeMap<i64, f64>> {
    use std::fmt::Write as _;

    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "You are a relevance ranker. Given a user query and candidate items, \
         score each candidate's relevance to the query."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "User query: {query}");
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Candidate items:");
    for (i, c) in shortlist.iter().enumerate() {
        let label = label_of(c.node_id).unwrap_or_default();
        let _ = writeln!(
            prompt,
            "{}. node_id: {}, label: \"{}\"",
            i + 1,
            c.node_id.as_int(),
            label
        );
    }
    let _ = writeln!(prompt);
    let _ = writeln!(
        prompt,
        "Output ONLY this JSON array:\n\
         [{{\"node_id\": <candidate node_id>, \"score\": <0.0..1.0>}}, ...]"
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "Rules:");
    let _ = writeln!(prompt, "- Score every candidate exactly once.");
    let _ = writeln!(prompt, "- Only output the JSON array, no other text.");

    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: prompt,
        },
        ChatMessage {
            role: "user".into(),
            content: query.to_string(),
        },
    ];
    let extras = serde_json::json!({
        "response_format": { "type": "json_object", "schema": ranking_schema() }
    });
    let response = match backend.chat_complete_with_extras(&messages, &extras) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "router.ranking",
                error = %e,
                "ranking call failed — using salience order",
            );
            return None;
        }
    };
    parse_ranking(&response, shortlist)
}

/// Parse the ranking response into a `node_id → score` map, keeping only ids
/// that are actually in the shortlist (a hallucinated id is dropped, exactly
/// like the chart reranker). `None` when nothing usable survives — the caller
/// keeps the salience order.
fn parse_ranking(response: &str, shortlist: &[RankedCandidate]) -> Option<BTreeMap<i64, f64>> {
    let allowed: BTreeSet<i64> = shortlist.iter().map(|c| c.node_id.as_int()).collect();
    let value =
        fluent_llm::parse_typed::<serde_json::Value>(response, &serde_json::Value::Null, |_| {})
            .ok()?;
    let arr = value.as_array()?;
    let mut scores = BTreeMap::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(id) = obj.get("node_id").and_then(serde_json::Value::as_i64) else {
            continue;
        };
        let Some(score) = obj.get("score").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        if allowed.contains(&id) {
            scores.insert(id, score.clamp(0.0, 1.0));
        }
    }
    if scores.is_empty() {
        None
    } else {
        Some(scores)
    }
}

/// The composable ranking step: salience source + optional model backend +
/// label source. This is the seam the M5 retrieval tools' live agent-loop
/// dispatch (a follow-up) calls to narrow a candidate pool before presenting
/// to a subagent; it is deliberately a discrete step with no side effects —
/// it never changes the deterministic capability-match fast path.
pub struct SalienceRanker {
    source: Arc<dyn SalienceSource>,
    backend: Option<Arc<dyn ChatBackend>>,
    label_of: Arc<dyn Fn(NodeId) -> Option<String> + Send + Sync>,
}

impl SalienceRanker {
    /// A ranker over `source` with an optional ranking model. Labels default
    /// to `None` (the prompt then lists ids only).
    #[must_use]
    pub fn new(source: Arc<dyn SalienceSource>, backend: Option<Arc<dyn ChatBackend>>) -> Self {
        Self {
            source,
            backend,
            label_of: Arc::new(|_| None),
        }
    }

    /// Attach a label source (e.g. the store's LOD5 lookup) for the prompt.
    #[must_use]
    pub fn with_label(
        mut self,
        label_of: Arc<dyn Fn(NodeId) -> Option<String> + Send + Sync>,
    ) -> Self {
        self.label_of = label_of;
        self
    }

    /// Prefilter `candidates` to the salience shortlist and (when a backend is
    /// present) rank that shortlist with one grammar-constrained model call.
    #[must_use]
    pub fn rank(&self, query: &str, candidates: &[NodeId]) -> Vec<RankedCandidate> {
        rank_candidates(
            query,
            candidates,
            &*self.source,
            self.backend.clone(),
            &*self.label_of,
        )
    }
}

#[cfg(test)]
#[path = "../tests/ranking.rs"]
mod tests;
