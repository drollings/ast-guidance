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
use fluent_types::NodeId;

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
                if node.role.as_deref()? != "checkpoint" {
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

        // 5. Enqueue the new node for background LOD4/LOD5.
        self.tiers.enqueue(node_id);

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
mod tests {
    use super::*;
    use crate::dag_session::SessionRegistry;
    use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex};
    use crate::ledger::tiering::TierConfig;
    use crate::test_stubs::StubChatBackend;

    fn test_registry() -> SessionRegistry {
        SessionRegistry::with_kv_cache(fork_kv())
    }

    fn temp_store() -> Arc<ContentNodeStore> {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-orch-{}",
            common_core::hash::uuid_v4()
        ));
        let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);
        store
    }

    /// A fork-enabled `SnapshotStore` (records snapshot metadata via a
    /// `StubServer` fork handle) so the coordinator's snapshot/restore round-trip
    /// is real.
    fn fork_kv() -> SnapshotStore {
        use crate::instances::stub::StubServer;
        use crate::instances::InstanceClient;

        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "POST" && path.ends_with("/snapshot") {
                    (200, "{}".into())
                } else {
                    (200, "[]".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let fork = Arc::new(InstanceClient::new(
            reqwest::Client::new(),
            stub.base_url(),
            None,
        ));
        let dir = tempfile::tempdir().unwrap();
        let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
        SnapshotStore::new(
            Arc::clone(&hot),
            Arc::new(ColdSnapshotIndex::new(
                dir.path(),
                1024,
                86400,
                crate::config::EvictionPolicy::Lru,
            )),
        )
        .with_fork_io(fork)
    }

    /// A coordinator whose tier worker is started and whose sessions share a
    /// fork-enabled KV cache (so snapshot metadata is recorded and restore
    /// works).
    fn coordinator(
        store: Arc<ContentNodeStore>,
        sessions: Arc<SessionRegistry>,
        backend: Arc<dyn ChatBackend>,
        kv_policy: KvSnapshotPolicy,
    ) -> (LedgerAgentCoordinator, Arc<LedgerTierWorker>) {
        let tiers = LedgerTierWorker::new(
            Arc::clone(&store),
            Arc::new(StubChatBackend::always("SUMMARY: s\nDESCRIPTION: d")),
            vec![4, 5],
            TierConfig {
                poll_interval_ms: 5,
                ..Default::default()
            },
            fluent_concurrency::tokio_runtime(),
        );
        let kv = sessions.kv_cache().clone();
        let coordinator = LedgerAgentCoordinator::new(
            store,
            sessions,
            kv,
            Arc::clone(&tiers),
            LedgerPromptAssembler,
            backend,
            OrchestratorConfig {
                kv_policy,
                ..Default::default()
            },
        );
        (coordinator, tiers)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_model_handoff_restores_same_model_only() {
        let store = temp_store();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["A1".to_string(), "B1".to_string(), "A2".to_string()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "help");

        // Model A runs (re-prefill), records a node, snapshots KV under (A).
        let a1 = coord
            .run_agent("sess", "model-a", &worker, "task 1")
            .await
            .unwrap();
        assert!(!a1.kv_restored, "first turn always re-prefills");

        // Model B runs (different model -> re-prefill, kv_restored = false),
        // records a node, snapshots KV under (B).
        let b1 = coord
            .run_agent("sess", "model-b", &worker, "task 2")
            .await
            .unwrap();
        assert!(!b1.kv_restored, "different model re-prefills");

        // Model A runs again (same model -> restores A snapshot).
        let a2 = coord
            .run_agent("sess", "model-a", &worker, "task 3")
            .await
            .unwrap();
        assert!(
            a2.kv_restored,
            "same model re-entry restores its own KV snapshot"
        );
        assert_eq!(a2.budget_used, 0, "restore sends no assembled prompt");

        // Both per-model snapshots coexist under their keys; the session's
        // pending snapshot is the most recent (model-a, after A ran last).
        let session = sessions.get_or_create("sess");
        let guard = session.lock().unwrap();
        assert!(guard.pending_snapshot().is_some());
        assert_eq!(
            guard.pending_snapshot().unwrap().model,
            "model-a",
            "pending snapshot is the most recent model's"
        );

        // The ledger records: 3 agent nodes + 3 checkpoint nodes (one per turn).
        let agent_ids = store.nodes_for_role("agent");
        let checkpoint_ids = store.nodes_for_role("checkpoint");
        assert_eq!(agent_ids.len(), 3, "one agent node per turn");
        assert_eq!(checkpoint_ids.len(), 3, "one checkpoint node per turn");

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn never_restore_always_rep_prefills() {
        let store = temp_store();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["x".into(), "y".into(), "z".into()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            backend,
            KvSnapshotPolicy::NeverRestore,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
        let again = coord.run_agent("sess", "model-a", &worker, "2").await.unwrap();
        assert!(
            !again.kv_restored,
            "NeverRestore ignores the pending snapshot and re-prefills"
        );

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn records_nodes_and_enqueues_tiers() {
        let store = temp_store();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["out-1".into(), "out-2".into()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        let o1 = coord.run_agent("sess", "model-a", &worker, "hi").await.unwrap();
        let node = store.snapshot(o1.node_id).unwrap();
        assert_eq!(node.lod[0], "out-1");
        assert_eq!(node.role.as_deref(), Some("agent"));
        assert_eq!(node.step_id.as_deref(), Some("model-a-1"));

        // LOD4/LOD5 are enqueued and filled in the background (proves the
        // coordinator enqueued every recorded node).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let n = store.snapshot(o1.node_id).unwrap();
            if !n.lod[4].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            !store.snapshot(o1.node_id).unwrap().lod[4].is_empty(),
            "recorded node enqueued + background-filled"
        );

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assembles_prompt_from_ledger_context() {
        // The coordinator's re-prefill path must assemble the ledger context
        // (not just the input) — observable via the assembled body reaching the
        // backend. Use a RecordingBackend to capture the user message.
        let store = temp_store();
        store
            .record_request("sess", "r1", "Prior ledger context node.")
            .unwrap();
        let sessions = Arc::new(test_registry());

        let captured = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let capture = Arc::clone(&captured);
        let backend: Arc<dyn ChatBackend> = Arc::new(RecordingBackend { captured: capture });
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            sessions,
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        coord
            .run_agent("sess", "model-a", &worker, "New request text")
            .await
            .unwrap();

        let msgs = captured.lock().unwrap().clone();
        let joined = msgs.join("\n");
        assert!(
            joined.contains("Prior ledger context node."),
            "assembled prompt includes prior ledger context, got: {joined}"
        );
        assert!(
            joined.contains("New request text"),
            "assembled prompt includes the new request"
        );

        handle.abort();
    }

    /// Records every user message it receives.
    struct RecordingBackend {
        captured: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl ChatBackend for RecordingBackend {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, LlmError> {
            self.captured.lock().unwrap().extend(
                messages
                    .iter()
                    .filter(|m| m.role == "user")
                    .map(|m| m.content.clone()),
            );
            Ok("recorded output".into())
        }
    }

    #[test]
    fn lod_spec_and_role_defaults() {
        let cfg = OrchestratorConfig::default();
        assert_eq!(cfg.role, "agent");
        assert_eq!(cfg.kv_policy, KvSnapshotPolicy::RestoreIfSameModel);
        assert!(cfg.budget.max_chars > 0);
        assert_eq!(cfg.lod_spec, LodSpec::full());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn agent_node_records_node_plan_for_learning_replay() {
        // On a re-prefill turn, the recorded agent node's metadata embeds
        // the assembled node_plan (node→Lod) so a future workflow-extraction
        // pass can replay the same decomposition.
        let store = temp_store();
        let prior = store
            .record_request("sess", "r0", "Prior context node text")
            .unwrap();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["out".to_string()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            sessions,
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        let outcome = coord
            .run_agent("sess", "model-a", &worker, "hi")
            .await
            .unwrap();
        assert!(!outcome.kv_restored, "re-prefill on first turn");

        let node = store.snapshot(outcome.node_id).unwrap();
        let meta = node.metadata.expect("metadata present");
        let plan = meta["node_plan"].as_array().expect("node_plan array");
        assert!(
            plan.iter().any(|pair| pair[0].as_i64() == Some(prior.as_int())),
            "node_plan must include the prior ledger node (anchored at LOD0), got: {plan:?}"
        );
        assert!(
            plan.iter().all(|pair| pair[1].as_u64() == Some(0)),
            "single prior node is the LOD0 anchor, got: {plan:?}"
        );

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn affinity_tracks_current_session_through_run_agent() {
        // Composing an `AffinityScheduler` must mark the active session
        // as the currently-affine one and submit its turn identity through the
        // shared scheduler (minimize context switches), without changing the
        // restore decision or the transport.
        let store = temp_store();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["a".into(), "b".into(), "c".into()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        let scheduler = LedgerAgentCoordinator::build_affinity_scheduler(2);
        let coord = coord.with_affinity(scheduler);
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
        let _ = coord.run_agent("sess", "model-a", &worker, "2").await.unwrap();

        // The composed scheduler marks the active session as affine (minimize
        // context switches) across interleaved sessions.
        assert_eq!(
            coord.affinity_session().as_deref(),
            Some("sess"),
            "run_agent marks the session as currently KV-affine"
        );

        // A same-model re-entry still restores KV (affinity bookkeeping must
        // not alter the restore decision or the transport).
        let outcome = coord.run_agent("sess", "model-a", &worker, "3").await.unwrap();
        assert!(outcome.kv_restored);

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preferred_model_prefers_last_resident_and_falls_back() {
        // The coordinator derives the last-resident (KV-affine) model
        // from the durable `checkpoint` ledger nodes and prefers it, falling
        // back to re-prefill (None) only on a true capability gap.
        let store = temp_store();
        let sessions = Arc::new(test_registry());
        let backend: Arc<dyn ChatBackend> =
            Arc::new(StubChatBackend::new(vec!["a".into(), "b".into()]));
        let (coord, tiers) = coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            backend,
            KvSnapshotPolicy::RestoreIfSameModel,
        );
        store.set_tier_events(tiers.sender());
        let handle = tiers.start();
        let worker = WorkerContext::new("assistant", "");

        // Empty session: no resident instance -> no preference (re-prefill).
        assert_eq!(coord.last_resident_model("sess"), None);
        assert_eq!(coord.preferred_model("sess", &["model-a"]), None);

        // Model A runs (records a checkpoint node), then Model B runs (the most
        // recent checkpoint node is now Model B's).
        let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
        let _ = coord.run_agent("sess", "model-b", &worker, "2").await.unwrap();

        assert_eq!(
            coord.last_resident_model("sess").as_deref(),
            Some("model-b"),
            "last-resident model is the most recent checkpoint"
        );
        assert_eq!(
            coord.preferred_model("sess", &["model-a", "model-b"]).as_deref(),
            Some("model-b"),
            "prefers the resident candidate (KV affinity)"
        );
        assert_eq!(
            coord.preferred_model("sess", &["model-a"]),
            None,
            "no resident candidate -> re-prefill (capability gap)"
        );

        // Model B re-enters and wins affinity again (still resident).
        assert_eq!(
            coord.preferred_model("sess", &["model-b", "model-a"]).as_deref(),
            Some("model-b")
        );

        handle.abort();
    }
}
