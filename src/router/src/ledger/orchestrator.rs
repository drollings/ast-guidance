//! `LedgerAgentCoordinator` — the synchronization point.
//!
//! Ties together the session step graph (`DependencySession`), the shared
//! `ContentNodeStore` (content), `SnapshotStore` (per-model KV), the
//! `LedgerTierWorker` (background LOD4/LOD5), and the `LedgerPromptAssembler`
//! (optimized prompt) into one run loop: *restore-or-assemble → execute →
//! record → snapshot → enqueue → complete step*.
//!
//! It is dependency-injected throughout (no concrete wiring): it composes the
//! shared primitives — `DependencyGraph` (via `DependencySession`),
//! `SnapshotStore::save_snapshot`/`retrieve`, `ContentNodeStore`/`ContentNodeLedger`,
//! `LedgerTierWorker::enqueue`, `ContentNodeStore::lod_text` (through the assembler),
//! `Limiter`/`ArcIntern` — and introduces **no** new primitive.
//!
//! `run_agent` is async by design (the server already runs in an async
//! context); the injected `ChatBackend` transport is synchronous, so no session
//! guard is held across an LLM call.

use std::sync::Arc;
use std::time::Instant;

use fluent_concurrency::affinity::{AffinityScheduler, ScheduledTask};
use fluent_concurrency::pool::PriorityResultPool;
use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};
use fluent_types::{ContentNode, NodeId};

use crate::dag_session::{DagError, KvSnapshotPolicy, SessionRegistry, SessionStep, StepResult};
use crate::kv_cache::SnapshotStore;
use crate::ledger::prompt::{LedgerPromptAssembler, PromptBudget, WorkerContext};
use crate::ledger::tiering::LedgerTierWorker;
use crate::node_store::{new_node, ContentNodeStore};
use crate::views::{Lod, ParallelLedger};

/// Re-export the prompt `LodSpec` under its canonical name for the config.
pub use crate::ledger::prompt::LodSpec;

/// Errors produced by the coordinator.
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("ledger error: {0}")]
    Ledger(#[from] crate::ledger::LedgerError),
    #[error("dag error: {0}")]
    Dag(#[from] DagError),
    #[error("backend error: {0}")]
    Backend(#[from] LlmError),
    #[error("session lock poisoned")]
    LockPoisoned,
}

/// Coordinator configuration: the KV restore policy, the prompt budget, the
/// fidelity band, and the default agent role.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// The explicit restore-vs-re-prefill decision rule.
    pub kv_policy: KvSnapshotPolicy,
    /// The worker's context-window budget for prompt assembly.
    pub budget: PromptBudget,
    /// The fidelity band for intermediate ledger nodes.
    pub lod_spec: LodSpec,
    /// Default role recorded for agent output nodes.
    pub role: String,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            kv_policy: KvSnapshotPolicy::RestoreIfSameModel,
            budget: PromptBudget::from_tokens_default(8192),
            lod_spec: LodSpec::full(),
            role: "agent".into(),
        }
    }
}

/// The outcome of one `run_agent` synchronization step.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The agent's output content (recorded as the node's LOD0).
    pub content: String,
    /// The id of the recorded output node.
    pub node_id: NodeId,
    /// Whether the agent resumed from a same-model KV snapshot (`true`) or
    /// re-prefilled from an assembled ledger prompt (`false`).
    pub kv_restored: bool,
    /// Prompt characters consumed (the assembler's `budget_used`; `0` when KV
    /// was restored — the ledger context is already in KV).
    pub budget_used: usize,
}

/// The composable coordinator.
///
/// An optional `AffinityScheduler` is composed for KV-affinity-aware
/// scheduling — it tracks which session is currently resident so the active
/// session's agent turn gets a priority bonus (minimize context switches) while
/// starved sessions age up. The actual "prefer the last-resident model/instance"
/// decision is exposed via [`LedgerAgentCoordinator::preferred_model`], which
/// reads the durable `checkpoint` ledger nodes (the record of "which model was
/// at which KV state when").  Re-prefill is the fallback whenever the
/// requested model is not the last-resident one (a true capability gap).
pub struct LedgerAgentCoordinator {
    store: Arc<ContentNodeStore>,
    sessions: Arc<SessionRegistry>,
    kv: SnapshotStore,
    tiers: Arc<LedgerTierWorker>,
    assembler: LedgerPromptAssembler,
    backend: Arc<dyn ChatBackend>,
    config: OrchestratorConfig,
    /// Optional KV-affinity scheduler. `None` disables affinity
    /// bookkeeping entirely — existing deployments are untouched.
    affinity: Option<AffinityScheduler<String, String, String>>,
    /// Optional live retrieval service (M5/M6 dispatch seam). `None` → no
    /// NLP-grep/fuzzy retrieval is exposed; existing deployments are untouched.
    retrieval: Option<Arc<crate::retrieval::NodeRetrievalService>>,
}

impl LedgerAgentCoordinator {
    /// Construct the coordinator (DI-injected).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<ContentNodeStore>,
        sessions: Arc<SessionRegistry>,
        kv: SnapshotStore,
        tiers: Arc<LedgerTierWorker>,
        assembler: LedgerPromptAssembler,
        backend: Arc<dyn ChatBackend>,
        config: OrchestratorConfig,
    ) -> Self {
        Self {
            store,
            sessions,
            kv,
            tiers,
            assembler,
            backend,
            config,
            affinity: None,
            retrieval: None,
        }
    }

    /// Compose a KV-affinity scheduler (forward track). The scheduler's
    /// pool worker is a trivial identity echo — it exists solely to drive the
    /// shared `AffinityScheduler`'s affinity-bonus/aging priority bookkeeping;
    /// the coordinator's transport and KV-affinity decision are unchanged.
    #[must_use]
    pub fn with_affinity(
        mut self,
        scheduler: AffinityScheduler<String, String, String>,
    ) -> Self {
        self.affinity = Some(scheduler);
        self
    }

    /// Compose the live retrieval service (the M5/M6 dispatch seam): the
    /// coordinator then exposes [`Self::retrieve_nodes`] to an agent tool loop.
    /// Additive — absent this, `retrieve_nodes` returns empty (fail-open).
    #[must_use]
    pub fn with_retrieval(
        mut self,
        retrieval: Arc<crate::retrieval::NodeRetrievalService>,
    ) -> Self {
        self.retrieval = Some(retrieval);
        self
    }

    /// The live retrieval dispatch seam: lemma-grep + fuzzy + cross-check over
    /// candidate `nodes` (the M5 tools), pre-filtered by the M6 salience
    /// ranker. Empty when no retrieval service is composed (fail-open).
    #[must_use]
    pub fn retrieve_nodes(
        &self,
        query: &str,
        nodes: &[ContentNode],
    ) -> Vec<crate::retrieval::NodeRetrievalReport> {
        if let Some(svc) = &self.retrieval {
            svc.retrieve(query, nodes)
        } else {
            tracing::warn!(
                target: "router.ledger.orchestrator",
                "retrieve_nodes called without a composed retrieval service — empty (fail-open)",
            );
            Vec::new()
        }
    }

    /// Build a ready-to-compose KV-affinity scheduler over the shared runtime
    /// `cap` bounds concurrent agent turns tracked by the pool.
    pub fn build_affinity_scheduler(
        cap: usize,
    ) -> AffinityScheduler<String, String, String> {
        let pool = Arc::new(PriorityResultPool::new(
            fluent_concurrency::tokio_runtime(),
            cap,
            |identity: String| async move { Ok::<_, String>(identity) },
        ));
        AffinityScheduler::new(pool)
    }

    /// The shared `SnapshotStore` this coordinator snapshots through.
    pub fn kv_cache(&self) -> &SnapshotStore {
        &self.kv
    }

    /// The last-resident (KV-affine) model for a session, from the durable
    /// `checkpoint` ledger nodes — the record of "which model was at
    /// which KV state when." Returns the model of the most recent checkpoint
    /// node for the session, or `None` when the session has no checkpoint (no
    /// resident instance yet → a caller re-prefills).
    ///
    /// Session-scoped: walks the session's own node index (insertion order,
    /// newest last) and takes the last `kv_checkpoint` node — it never scans
    /// every session's checkpoint nodes.
    pub fn last_resident_model(&self, session_id: &str) -> Option<String> {
        self.store
            .session_node_ids(session_id)
            .into_iter()
            .rev()
            .find_map(|id| {
                let node = self.store.snapshot(id)?;
                if node.role.as_ref()?.as_str() != "checkpoint" {
                    return None;
                }
                let meta = node.metadata?;
                if meta.get("kind")?.as_str() != Some("kv_checkpoint") {
                    return None;
                }
                meta.get("model")?.as_str().map(str::to_string)
            })
    }

    /// Prefer the last-resident (KV-affine) model for a session:
    /// returns the `candidates` entry that is currently resident, so a caller
    /// avoids a context switch by re-entering that model's KV. Returns `None`
    /// when no candidate is resident — the true capability-gap fallback is to
    /// re-prefill from an assembled ledger prompt.
    pub fn preferred_model(&self, session_id: &str, candidates: &[&str]) -> Option<String> {
        let resident = self.last_resident_model(session_id)?;
        candidates
            .iter()
            .find(|c| **c == resident)
            .map(ToString::to_string)
    }

    /// The session currently marked KV-affine by the composed scheduler,
    /// or `None` when no scheduler is attached. `None` from a composed
    /// scheduler means no `run_agent` has run yet.
    pub fn affinity_session(&self) -> Option<String> {
        self.affinity.as_ref().and_then(AffinityScheduler::current_affinity)
    }

    /// Run one agent synchronization step.
    ///
    /// # Steps
    /// 1. Resolve the context source: restore a same-model KV snapshot (per
    ///    `kv_policy`) or assemble an optimized ledger prompt (re-prefill).
    /// 2. Execute the agent against the injected transport.
    /// 3. Record the output as a `ContentNode` (role = `config.role`).
    /// 4. Snapshot the model's KV on context advance (`advance_and_snapshot`).
    /// 5. Enqueue the node for background LOD4/LOD5.
    /// 6. Complete the session step.
    pub async fn run_agent(
        &self,
        session_id: &str,
        model: &str,
        worker: &WorkerContext,
        input: &str,
    ) -> Result<AgentOutcome, CoordinatorError> {
        let session = self.sessions.get_or_create(session_id);

        // 0. KV-affinity bookkeeping: when a scheduler is composed, mark
        // this session as the currently-affine one (its turns get a priority
        // bonus — minimize context switches) and submit its identity through the
        // shared scheduler so starved sessions age up. The pool worker is an
        // identity echo; this never touches the transport or the KV decision.
        if let Some(scheduler) = &self.affinity {
            scheduler.set_affinity(Some(session_id.to_string()));
            let _ = scheduler
                .submit(
                    ScheduledTask {
                        identity: session_id.to_string(),
                        task: model.to_string(),
                        enqueued_at: Instant::now(),
                    },
                    0,
                )
                .await;
        }

        // 1. Short guard: set the model, resolve the restore decision, and
        // create the step. Dropped before the LLM call (never held across it).
        //
        // The restore decision looks up THIS model's snapshot under its
        // `(model, adapter, session)` key (not the pending snapshot, which may
        // belong to a different model that ran most recently). This is what
        // lets a same-model re-entry resume its own KV while a different model
        // re-prefills.
        let (kv_restored, step_id) = {
            let mut guard = session
                .lock()
                .map_err(|_| CoordinatorError::LockPoisoned)?;
            guard.set_model(model);
            let snapshot = guard
                .kv_cache()
                .and_then(|kv| kv.retrieve(model, guard.adapter.as_deref(), session_id).ok());
            let kv_restored = snapshot.is_some_and(|s| {
                self.config
                    .kv_policy
                    .decide_restore(Some(s.model.as_str()), model)
            });
            let step_id = format!("{model}-{}", guard.step_count() + 1);
            let _ = guard.add_step(SessionStep::new(
                step_id.clone(),
                format!("agent turn for {model}"),
            ));
            (kv_restored, step_id)
        };

        // 2. Build the messages and execute the agent. The transport is
        // synchronous, so the LLM call is offloaded to a blocking task — the
        // session guard is never held across it, and the runtime thread is not
        // blocked.
        let (messages, budget_used, node_plan) =
            self.build_messages(session_id, worker, input, kv_restored);
        let backend = Arc::clone(&self.backend);
        let content = tokio::task::spawn_blocking(move || backend.chat_complete(&messages))
            .await
            .map_err(|_| CoordinatorError::LockPoisoned)?
            .map_err(CoordinatorError::Backend)?;

        // 3. Record the output node (embedding the assembled node_plan into
        // its metadata so a future workflow-extraction pass can replay the
        // same decomposition without re-prefilling — learning loop).
        let node_id = self.record_agent_node(session_id, model, &content, &step_id, &node_plan)?;

        // 4-6. Snapshot KV on context advance, record a checkpoint node, and
        // complete the step. The KV snapshot key is `(model, adapter, session)`.
        {
            let mut guard = session
                .lock()
                .map_err(|_| CoordinatorError::LockPoisoned)?;
            let instance = model.to_string();
            if let Some(snap) = guard.advance_and_snapshot(&instance)? {
                self.record_checkpoint_node(session_id, &snap)?;
            }
            let _ = guard.complete_step(
                &step_id,
                StepResult {
                    content: content.clone(),
                    accepted: true,
                    score: None,
                    latency_ms: 0,
                    error: None,
                },
            );
        }

        // 5. Enqueue the new node for background LOD4/LOD5. Credit-gated: a
        // burst of agent turns cannot grow the tier feed without bound. A
        // closed feed is not an agent-turn failure — the node's tiers are
        // filled by the next boot backfill.
        if let Err(e) = self.tiers.enqueue_with_credit(node_id).await {
            tracing::warn!(
                target: "router.ledger.orchestrator",
                node_id = node_id.as_int(),
                error = %e,
                "tier enqueue skipped after the agent turn",
            );
        }

        // Audit the run (kind = "agent").
        crate::audit::emit(
            "agent",
            serde_json::json!({
                "session_id": session_id,
                "model": model,
                "kv_restored": kv_restored,
                "budget_used": budget_used,
                "node_id": node_id.as_int(),
            }),
        );

        Ok(AgentOutcome {
            content,
            node_id,
            kv_restored,
            budget_used,
        })
    }

    /// Build the transport messages: on restore, a minimal continuation (the
    /// ledger context is already in KV); on re-prefill, the assembled optimized
    /// prompt from the ledger. Returns `(messages, budget_used, node_plan)`
    /// where `node_plan` is the per-node fidelity decision (empty on restore).
    fn build_messages(
        &self,
        session_id: &str,
        worker: &WorkerContext,
        input: &str,
        kv_restored: bool,
    ) -> (Vec<ChatMessage>, usize, Vec<(NodeId, Lod)>) {
        let system = worker.system_prompt();
        if kv_restored {
            return (
                vec![
                    ChatMessage {
                        role: "system".into(),
                        content: system,
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: input.to_string(),
                    },
                ],
                0,
                Vec::new(),
            );
        }
        let view = ParallelLedger::for_session(Arc::clone(&self.store), session_id);
        let assembled = self.assembler.assemble(
            &view,
            worker,
            &self.config.budget,
            None,
            &self.config.lod_spec,
        );
        // Emit a `prompt` audit carrying the fidelity plan so the
        // audit stream records exactly which ledger nodes were rendered at
        // which `Lod` (the workflow-learning replay signal).
        crate::audit::emit(
            "prompt",
            serde_json::json!({
                "session_id": session_id,
                "budget_used": assembled.budget_used,
                "node_plan": node_plan_json(&assembled.node_plan),
            }),
        );
        let body = assembled.body;
        let user = if body.is_empty() {
            input.to_string()
        } else {
            format!("{body}\n\nNew request:\n{input}")
        };
        (
            vec![
                ChatMessage {
                    role: "system".into(),
                    content: system,
                },
                ChatMessage {
                    role: "user".into(),
                    content: user,
                },
            ],
            assembled.budget_used,
            assembled.node_plan,
        )
    }

    /// Record the agent's output as a `ContentNode` through the canonical
    /// `record_content_node` write path (role = `config.role`).
    fn record_agent_node(
        &self,
        session_id: &str,
        model: &str,
        content: &str,
        step_id: &str,
        node_plan: &[(NodeId, Lod)],
    ) -> Result<NodeId, CoordinatorError> {
        let request_id = format!("{session_id}-{step_id}");
        let mut node = new_node(
            NodeId::from_int(0),
            session_id,
            &request_id,
            &self.config.role,
            content,
            Some(true),
        );
        node.id = None; // let the store allocate a fresh, collision-free id
        node.step_id = Some(step_id.to_string());
        node.metadata = Some(serde_json::json!({
            "model": model,
            "origin": "agent_coordinator",
            // The fidelity plan for workflow-learning replay (node→Lod).
            "node_plan": node_plan_json(node_plan),
        }));
        Ok(self.store.record_content_node(&node)?)
    }

    /// Record a checkpoint ledger node (role `"checkpoint"`, carrying the
    /// snapshot's fork-facing identity so the ledger is the durable record of
    /// "which model was at which KV state when."
    fn record_checkpoint_node(
        &self,
        session_id: &str,
        snap: &crate::kv_cache::KvSnapshot,
    ) -> Result<NodeId, CoordinatorError> {
        let request_id = format!("{session_id}-cp-{}", snap.snapshot_name);
        let mut node = new_node(
            NodeId::from_int(0),
            session_id,
            &request_id,
            "checkpoint",
            &format!("KV checkpoint: {}", snap.snapshot_name),
            Some(true),
        );
        node.id = None; // let the store allocate a fresh, collision-free id
        node.metadata = Some(serde_json::json!({
            "kind": "kv_checkpoint",
            "model": snap.model,
            "snapshot_name": snap.snapshot_name,
            "instance": snap.instance,
            "token_count": snap.token_count,
        }));
        Ok(self.store.record_content_node(&node)?)
    }
}

// Keep the `AssembledPrompt` type referenced for downstream wiring.
#[allow(unused_imports)]
use crate::ledger::prompt::AssembledPrompt;

/// Serialize a per-node fidelity plan as a JSON array of `[node_id, lod]`
/// pairs (Workflow-learning replay signal).
fn node_plan_json(plan: &[(NodeId, Lod)]) -> serde_json::Value {
    serde_json::json!(plan
        .iter()
        .map(|(id, lod)| serde_json::json!([id.as_int(), lod.as_u8()]))
        .collect::<Vec<_>>())
}
#[cfg(test)]
#[path = "../../tests/ledger_orchestrator.rs"]
mod tests;
