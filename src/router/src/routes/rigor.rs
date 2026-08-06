//! Rigor route — the fixed-pass blue/red/judge protocol (M3).
//!
//! `RigorRoute` wires the VISION's high-stakes verification loop to live
//! backends:
//!
//! 1. **Blue team** produces a candidate answer (plain text).
//! 2. A `DependencySession` checkpoint (`rigor.blue`) is recorded so a
//!    red-team-identified dead end can be **rewound for real**.
//! 3. **Red team** reads the session through M2's `FilteredLedger` at
//!    `Lod::LOD0` (exclusion set = blue's rejected dead ends) and emits a JSON
//!    array of objections.
//! 4. **Judge** emits a structured verdict (`accept` /
//!    `accept_with_caveats` / `reject`) with a confidence.
//! 5. A **material** rejection (any objection severity >= the configured
//!    threshold) rewinds to `rigor.blue` and runs a **second and final** blue
//!    pass with the objections folded in, then judges again.
//! 6. If still rejected, the route resolves to a **targeted interview**
//!    (<= 3 questions derived from the objection descriptions) and, only when
//!    judge confidence is low, marks `frontier_escalation` (an explicit
//!    config value — never "red scored a point").
//!
//! Round count is fixed at `max_passes` (default 2): never a third pass
//! (VISION: terminate, don't loop). Backends are DIP-injected
//! `Arc<dyn ChatBackend>` built exactly once in `main.rs`. There is **no**
//! `Interviewable` trait — the targeted interview is a third, distinct shape
//! from plan's binding-gap closure loop (D5).

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_llm::{parse_json_response, ChatMessage};
use fluent_types::{ContentNode, NodeId, StepStatus};
use serde::{Deserialize, Serialize};

use crate::config::RigorConfig;
use crate::dag_session::{DependencySession, SessionStep, StepResult};
use crate::dispatch::escalation::LocalBackend;
use crate::ledger::ContentNodeLedger;
use crate::views::{FilteredLedger, LedgerView, Lod, ParallelLedger};

/// Context passed to the rigor route's execute method. Carries the minimal
/// information needed for the 3-pass blue/red/judge protocol.
#[derive(Clone)]
pub struct RigorContext {
    pub user_message: String,
    pub session_id: String,
    pub model_endpoint: String,
    /// The `DependencySession` for checkpoint/rewind between passes (D5).
    /// `None` degrades to a sessionless run (no checkpoint, no rewind, no
    /// red-team ledger view).
    pub session: Option<Arc<Mutex<DependencySession>>>,
    /// The shared `ContentNodeLedger` whose store the red team reads at LOD0
    /// (D6). `None` degrades the red pass to the blue answer only.
    pub ledger: Option<Arc<ContentNodeLedger>>,
}

pub struct RigorRoute {
    /// Whether to support KV-cache checkpoint/rewind for dead-end recovery.
    pub kv_cache_enabled: bool,
    blue: Option<LocalBackend>,
    red: Option<LocalBackend>,
    judge: Option<LocalBackend>,
    /// Route behavior config: `max_passes`, material-rejection threshold,
    /// escalation trigger.
    cfg: RigorConfig,
}

#[derive(Debug, Clone)]
pub struct RigoResult {
    pub blue_answer: String,
    pub red_objections: Vec<RedObjection>,
    pub judge_verdict: JudgeVerdict,
    pub frontier_escalation: bool,
    /// The targeted-interview fallback: <= 3 questions derived from the
    /// objection descriptions when the final pass still rejects.
    pub interview_questions: Vec<String>,
    /// Whether a real `rewind_to_checkpoint("rigor.blue")` ran between passes.
    pub rewound: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedObjection {
    pub category: String,
    pub description: String,
    pub severity: f64,
    /// The claim's ledger node id the objection dereferences at LOD0
    /// (D5). `None` when the objection is not claim-anchored.
    #[serde(default)]
    pub target_claim: Option<NodeId>,
}

#[derive(Debug, Clone)]
pub enum JudgeVerdict {
    Accept,
    AcceptWithCaveats { caveats: Vec<String> },
    Reject { reasons: Vec<String> },
}

impl Default for RigorRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl RigorRoute {
    #[must_use]
    pub fn new() -> Self {
        Self {
            kv_cache_enabled: false,
            blue: None,
            red: None,
            judge: None,
            cfg: RigorConfig::default(),
        }
    }

    #[must_use]
    pub fn with_kv_cache(mut self) -> Self {
        self.kv_cache_enabled = true;
        self
    }

    /// Attach the blue-team (candidate-answer) backend. Mock-injectable.
    #[must_use]
    pub fn with_blue_backend(mut self, backend: LocalBackend) -> Self {
        self.blue = Some(backend);
        self
    }

    /// Attach the red-team (objections) backend. Mock-injectable.
    #[must_use]
    pub fn with_red_backend(mut self, backend: LocalBackend) -> Self {
        self.red = Some(backend);
        self
    }

    /// Attach the judge backend. Mock-injectable.
    #[must_use]
    pub fn with_judge_backend(mut self, backend: LocalBackend) -> Self {
        self.judge = Some(backend);
        self
    }

    /// Attach the route behavior config (passes / thresholds).
    #[must_use]
    pub fn with_config(mut self, cfg: RigorConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Resolve a role backend, degrading to an explicit error when missing
    /// (never a silent no-op).
    fn role_backend(&self, name: &str) -> Result<LocalBackend, RigorError> {
        let backend = match name {
            "blue" => self.blue.clone(),
            "red" => self.red.clone(),
            "judge" => self.judge.clone(),
            _ => return Err(RigorError::Unconfigured(name.to_string())),
        };
        backend.ok_or_else(|| RigorError::Unconfigured(name.to_string()))
    }

    /// Execute the fixed-pass rigor protocol (D5):
    /// blue -> checkpoint -> red -> judge -> (rewind + second blue + judge) ->
    /// interview/escalation.
    ///
    /// A material rejection rewinds the session for real between the first
    /// blue pass and the re-run; the final resolution is never an open loop.
    pub async fn execute(&self, ctx: &RigorContext) -> Result<RigoResult, RigorError> {
        let blue = self.role_backend("blue")?;
        let red = self.role_backend("red")?;
        let judge = self.role_backend("judge")?;

        Self::register_steps(ctx);

        // Pass 1: blue.
        let mut answer = blue_answer(&blue, &ctx.user_message).await?;
        Self::complete_blue_step(ctx, &answer);

        // Red + judge over the (no-dead-end) view.
        let objections = Self::red_pass(ctx, &red, &answer).await?;
        let (mut verdict, mut confidence) = Self::judge_pass(&judge, &answer, &objections).await?;

        let mut rewound = false;
        let mut interview_questions = Vec::new();
        let mut frontier_escalation = false;

        // A material rejection -> one rewind + a second, final blue pass.
        if is_reject(&verdict) && self.material_rejection(&objections) && self.cfg.max_passes > 1 {
            Self::record_blue_dead_end(ctx, &answer);
            rewound = self.rewind_to_blue(ctx);
            let refocused = blue_retry_prompt(&ctx.user_message, &answer, &objections);
            answer = blue_answer(&blue, &refocused).await?;
            Self::complete_blue_step(ctx, &answer);
            let objections2 = Self::red_pass(ctx, &red, &answer).await?;
            let (v2, c2) = Self::judge_pass(&judge, &answer, &objections2).await?;
            verdict = v2;
            confidence = c2;
        }

        if is_reject(&verdict) {
            interview_questions = derive_interview(&objections);
            if confidence < self.cfg.escalation_confidence {
                frontier_escalation = true;
            }
        }

        crate::audit::emit(
            "rigor",
            serde_json::json!({
                "passes": if rewound { 2 } else { 1 },
                "rewound": rewound,
                "verdict": verdict_tag(&verdict),
                "objections": objections.len(),
                "frontier_escalation": frontier_escalation,
                "interview_questions": interview_questions.len(),
            }),
        );

        Ok(RigoResult {
            blue_answer: answer,
            red_objections: objections,
            judge_verdict: verdict,
            frontier_escalation,
            interview_questions,
            rewound,
        })
    }

    /// Register the rigor steps on the session idempotently (`add_step`
    /// returns `DuplicateNode` — only add when absent), and set the model if
    /// unset (KV snapshot keying requires one, `dag_session.rs:354-363`).
    fn register_steps(ctx: &RigorContext) {
        let Some(session) = &ctx.session else {
            return;
        };
        let mut s = lock(session);
        if s.model.is_none() {
            s.set_model(ctx.model_endpoint.clone());
        }
        if s.get_step("rigor.blue").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.blue", "blue team candidate answer").with_checkpoint(),
            );
        }
        if s.get_step("rigor.red").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.red", "red team objections")
                    .with_depends(vec!["rigor.blue".into()]),
            );
        }
        if s.get_step("rigor.judge").is_none() {
            let _ = s.add_step(
                SessionStep::new("rigor.judge", "judge verdict")
                    .with_depends(vec!["rigor.red".into()]),
            );
        }
    }

    /// Complete `rigor.blue` (records the checkpoint via `complete_step`).
    fn complete_blue_step(ctx: &RigorContext, answer: &str) {
        let Some(session) = &ctx.session else {
            return;
        };
        let mut s = lock(session);
        if s.get_step("rigor.blue").is_some() {
            let _ = s.complete_step(
                "rigor.blue",
                StepResult {
                    content: answer.to_string(),
                    accepted: true,
                    score: None,
                    latency_ms: 0,
                    error: None,
                },
            );
        }
    }

    /// Rewind to `rigor.blue` (real `rewind_to_checkpoint`). Steps are reset
    /// to `Pending` with result data preserved for audit; the KV snapshot, if
    /// restored, is logged (`file_path` feeds a future dispatch slot-restore —
    /// recorded here, never dispatched). Returns whether a rewind ran.
    fn rewind_to_blue(&self, ctx: &RigorContext) -> bool {
        let Some(session) = &ctx.session else {
            return false;
        };
        let mut s = lock(session);
        match s.rewind_to_checkpoint("rigor.blue") {
            Ok(snapshot) => {
                if let Some(snap) = snapshot {
                    tracing::info!(
                        target: "router.rigor",
                        file_path = %snap.file_path.display(),
                        kv_cache_enabled = self.kv_cache_enabled,
                        "kv cache snapshot restored on rigor rewind — file_path recorded, not dispatched"
                    );
                }
                true
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.rigor",
                    error = %e,
                    "rigor rewind failed — continuing with current answer"
                );
                false
            }
        }
    }

    /// Persist the first blue answer as a rejected dead-end ledger node so the
    /// second red pass's filtered view excludes it (D6). Best-effort; a
    /// missing ledger degrades silently.
    fn record_blue_dead_end(ctx: &RigorContext, answer: &str) {
        let Some(ledger) = &ctx.ledger else {
            return;
        };
        let node = ContentNode {
            id: None,
            name: "rigor-blue-dead-end".into(),
            source: "rigor".into(),
            lod: vec![
                answer.to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ],
            embedding: None,
            capabilities: None,
            session_id: Some(ctx.session_id.clone()),
            request_id: None,
            role: Some("blue".into()),
            turn_index: None,
            accepted: Some(false),
            acceptance_score: None,
            active_lod: Some(0),
            parent_id: None,
            step_id: Some("rigor.blue".into()),
            step_status: None,
            metadata: None,
            created_at: Some(common_core::now_secs()),
        };
        let _ = ledger.record_content_node(&node);
    }

    /// Red-pass prompt: the blue answer plus the session rendered through M2's
    /// `FilteredLedger` at LOD0, excluding blue's rejected dead ends (D6).
    /// No ledger/session -> the blue answer alone (documented degradation).
    async fn red_pass(
        ctx: &RigorContext,
        red: &LocalBackend,
        answer: &str,
    ) -> Result<Vec<RedObjection>, RigorError> {
        let mut material = format!("Blue team's candidate answer:\n{answer}\n");
        let rendered = Self::red_team_view(ctx);
        if !rendered.is_empty() {
            material.push_str("\nSession ledger (LOD0, dead ends excluded):\n");
            material.push_str(&rendered);
        }
        let raw = chat(red, RED_SYSTEM_PROMPT, &material)
            .await
            .map_err(|e| RigorError::RedTeam(e.to_string()))?;
        parse_objections(&raw).map_err(RigorError::RedTeam)
    }

    /// The M2 red-team view: `ParallelLedger::for_session(...).with_default_lod
    /// (Lod::LOD0)` wrapped in `FilteredLedger` excluding dead-end node ids.
    fn red_team_view(ctx: &RigorContext) -> String {
        let Some(ledger) = &ctx.ledger else {
            return String::new();
        };
        let store = ledger.node_store().clone();
        let base = ParallelLedger::for_session(store, &ctx.session_id).with_default_lod(Lod::LOD0);
        let excluded = dead_end_node_ids(ctx);
        FilteredLedger::new(base, excluded).render()
    }

    async fn judge_pass(
        judge: &LocalBackend,
        answer: &str,
        objections: &[RedObjection],
    ) -> Result<(JudgeVerdict, f64), RigorError> {
        let prompt = judge_prompt(answer, objections);
        let raw = chat(judge, JUDGE_SYSTEM_PROMPT, &prompt)
            .await
            .map_err(|e| RigorError::Judge(e.to_string()))?;
        parse_judge(&raw).map_err(RigorError::Judge)
    }

    /// Whether any objection is material (severity >= the configured
    /// threshold) — the trigger for rewind, not "the judge rejected".
    fn material_rejection(&self, objections: &[RedObjection]) -> bool {
        objections
            .iter()
            .any(|o| o.severity >= self.cfg.severity_threshold)
    }
}

/// The blue-team system prompt: a direct candidate answer, plain text out.
const BLUE_SYSTEM_PROMPT: &str = r"You are the blue team. Produce a direct, correct candidate answer to the user's request.
Be complete and precise. Output only the answer text, no preamble.";

/// The red-team system prompt: a JSON array of objections, each anchored to a
/// claim's ledger node id where applicable.
const RED_SYSTEM_PROMPT: &str = r#"You are the red team. Given a blue team's candidate answer and the session material,
find weaknesses: factual errors, missing requirements, unsafe claims, or unsupported assertions.
Output a JSON array of objections, each:
{"category": "a short category", "description": "the objection", "severity": 0.0-1.0, "target_claim": <ledger node id as a number, or null>}
Output only valid JSON, no other text."#;

/// The judge system prompt: a structured verdict.
const JUDGE_SYSTEM_PROMPT: &str = r#"You are the judge. Given a blue team's candidate answer and the red team's objections,
decide whether the answer is acceptable.
Output JSON only:
{"verdict": "accept" | "accept_with_caveats" | "reject", "caveats": [...], "reasons": [...], "confidence": 0.0-1.0}
"confidence" is how confident you are in this verdict. Output only valid JSON, no other text."#;

/// Build the standard system+user message pair.
fn messages(system: &str, user: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".into(),
            content: system.into(),
        },
        ChatMessage {
            role: "user".into(),
            content: user.into(),
        },
    ]
}

/// Run the blue pass: candidate answer as plain text.
async fn blue_answer(blue: &LocalBackend, user_message: &str) -> Result<String, RigorError> {
    chat(blue, BLUE_SYSTEM_PROMPT, user_message)
        .await
        .map_err(|e| RigorError::BlueTeam(e.to_string()))
}

/// Run one blocking `ChatBackend::chat_complete` off the executor.
///
/// The role backends are synchronous (the codebase's DIP pattern — the same
/// `Arc<dyn ChatBackend>` used by dispatch). `RigorRoute::execute` is async by
/// locked decision D5, so the blocking calls are offloaded via
/// `spawn_blocking` and awaited, honoring the WorkUnit purity contract (never
/// block a tokio worker).
async fn chat(
    backend: &LocalBackend,
    system: &str,
    user: &str,
) -> Result<String, fluent_llm::LlmError> {
    let backend = backend.clone();
    let messages = messages(system, user);
    tokio::task::spawn_blocking(move || backend.chat_complete(&messages))
        .await
        .map_err(|e| fluent_llm::LlmError::Api(format!("chat task failed: {e}")))?
}

/// Fold the red objections into a refocused blue prompt for the second pass.
fn blue_retry_prompt(
    user_message: &str,
    prior_answer: &str,
    objections: &[RedObjection],
) -> String {
    use std::fmt::Write as _;

    let mut prompt = format!(
        "Original request:\n{user_message}\n\n\
         Your previous answer was rejected by the red team.\n\
         Previous answer:\n{prior_answer}\n\n\
         Address these objections:\n"
    );
    for o in objections {
        let _ = writeln!(
            prompt,
            "- [{:.2}] {}: {}",
            o.severity, o.category, o.description
        );
    }
    prompt
}

/// Build the judge prompt: the candidate answer plus each objection.
fn judge_prompt(answer: &str, objections: &[RedObjection]) -> String {
    use std::fmt::Write as _;

    let mut prompt = format!("Blue team's candidate answer:\n{answer}\n\nRed team objections:\n");
    if objections.is_empty() {
        prompt.push_str("(none)\n");
    }
    for (i, o) in objections.iter().enumerate() {
        let claim = o
            .target_claim
            .map_or_else(|| "null".to_string(), |id| id.as_int().to_string());
        let _ = writeln!(
            prompt,
            "{i}. [{:.2}] {}: {} (target_claim: {claim})",
            o.severity, o.category, o.description
        );
    }
    prompt
}

/// Wire shape of the red output before it is mapped into `RedObjection`.
#[derive(Debug, Deserialize)]
struct RedObjectionWire {
    #[serde(default)]
    category: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    severity: f64,
    #[serde(default)]
    target_claim: Option<NodeId>,
}

/// Tolerant parse of the red-team objections array via
/// `fluent_llm::parse_json_response` (never hand-rolled fence-stripping).
fn parse_objections(raw: &str) -> Result<Vec<RedObjection>, String> {
    let value = parse_json_response(raw).map_err(|e| e.to_string())?;
    let arr = match value {
        serde_json::Value::Array(arr) => arr,
        serde_json::Value::Object(map) => {
            // Tolerate a wrapped `{"objections": [...]}` shape.
            map.get("objections")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .ok_or_else(|| "red output is neither an array nor an objections map".to_string())?
        }
        other => return Err(format!("red output is not a JSON array: {other}")),
    };
    arr.into_iter()
        .map(|v| {
            let wire: RedObjectionWire =
                serde_json::from_value(v).map_err(|e| format!("invalid objection: {e}"))?;
            Ok(RedObjection {
                category: wire.category,
                description: wire.description,
                severity: wire.severity,
                target_claim: wire.target_claim,
            })
        })
        .collect()
}

/// Wire shape of the judge output.
#[derive(Debug, Deserialize)]
struct JudgeOutputWire {
    #[serde(default)]
    verdict: String,
    #[serde(default)]
    caveats: Vec<String>,
    #[serde(default)]
    reasons: Vec<String>,
    #[serde(default = "default_judge_confidence")]
    confidence: f64,
}

const fn default_judge_confidence() -> f64 {
    0.5
}

/// Tolerant parse of the judge verdict via `fluent_llm::parse_json_response`.
fn parse_judge(raw: &str) -> Result<(JudgeVerdict, f64), String> {
    let value = parse_json_response(raw).map_err(|e| e.to_string())?;
    let wire: JudgeOutputWire =
        serde_json::from_value(value).map_err(|e| format!("invalid judge output: {e}"))?;
    let verdict = match wire.verdict.as_str() {
        "accept" => JudgeVerdict::Accept,
        "accept_with_caveats" => JudgeVerdict::AcceptWithCaveats {
            caveats: wire.caveats,
        },
        "reject" => JudgeVerdict::Reject {
            reasons: wire.reasons,
        },
        other => return Err(format!("unknown judge verdict: {other}")),
    };
    Ok((verdict, wire.confidence))
}

/// Whether a verdict is a rejection.
fn is_reject(verdict: &JudgeVerdict) -> bool {
    matches!(verdict, JudgeVerdict::Reject { .. })
}

/// Audit-tag form of a verdict.
fn verdict_tag(verdict: &JudgeVerdict) -> &'static str {
    match verdict {
        JudgeVerdict::Accept => "accept",
        JudgeVerdict::AcceptWithCaveats { .. } => "accept_with_caveats",
        JudgeVerdict::Reject { .. } => "reject",
    }
}

/// The dead-end node ids for the red-team filtered view: ledger nodes in the
/// session whose `accepted == Some(false)` (the blue dead ends persisted by
/// `record_blue_dead_end`), plus nodes mapped from session steps whose result
/// is rejected (the roadmap's step-level mechanism, where available).
fn dead_end_node_ids(ctx: &RigorContext) -> HashSet<NodeId> {
    let Some(ledger) = &ctx.ledger else {
        return HashSet::new();
    };
    let store = ledger.node_store();
    let mut rejected_steps: HashSet<String> = HashSet::new();
    if let Some(session) = &ctx.session {
        let s = lock(session);
        for step_id in s.step_ids() {
            if let Some(step) = s.get_step(step_id) {
                let rejected = matches!(step.status, StepStatus::Failed | StepStatus::Cancelled)
                    || step
                        .result
                        .as_ref()
                        .is_some_and(|r| !r.accepted || r.error.is_some());
                if rejected {
                    rejected_steps.insert(step_id.clone());
                }
            }
        }
    }
    store
        .session_node_ids(&ctx.session_id)
        .into_iter()
        .filter(|nid| {
            store.snapshot(*nid).is_some_and(|node| {
                node.accepted == Some(false)
                    || node
                        .step_id
                        .as_deref()
                        .is_some_and(|sid| rejected_steps.contains(sid))
            })
        })
        .collect()
}

/// The targeted-interview fallback: the <= 3 highest-severity objections,
/// turned into direct clarification questions (VISION: "a targeted interview
/// with the user — not silent escalation").
fn derive_interview(objections: &[RedObjection]) -> Vec<String> {
    let mut ranked: Vec<&RedObjection> = objections.iter().collect();
    ranked.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
        .into_iter()
        .take(3)
        .map(|o| {
            format!(
                "{} — {}. Could you clarify this concern?",
                o.category, o.description
            )
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum RigorError {
    #[error("rigor role backend not configured: {0}")]
    Unconfigured(String),
    #[error("blue team error: {0}")]
    BlueTeam(String),
    #[error("red team error: {0}")]
    RedTeam(String),
    #[error("judge error: {0}")]
    Judge(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag_session::DependencySession;
    use crate::test_stubs::StubChatBackend;
    use fluent_llm::client::ChatBackend;

    fn ctx(message: &str) -> RigorContext {
        RigorContext {
            user_message: message.to_string(),
            session_id: "sess-rigor".into(),
            model_endpoint: "model-x".into(),
            session: None,
            ledger: None,
        }
    }

    fn test_cfg() -> RigorConfig {
        RigorConfig {
            max_passes: 2,
            severity_threshold: 0.7,
            escalation_confidence: 0.4,
            ..Default::default()
        }
    }

    /// A route whose judge backend pops the given responses; blue serves two
    /// candidate answers (the material-rejection path rewinds and re-runs
    /// blue), red serves two canned objection sets, judge pops `responses`.
    fn route_with_judge(responses: Vec<&str>) -> RigorRoute {
        RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::new(vec![
                "candidate answer".to_string(),
                "second candidate answer".to_string(),
            ])))
            .with_red_backend(Arc::new(StubChatBackend::new(vec![
                r#"[{"category": "factual", "description": "unsupported claim", "severity": 0.9}]"#
                    .to_string(),
                r#"[{"category": "factual", "description": "unsupported claim", "severity": 0.9}]"#
                    .to_string(),
            ])))
            .with_judge_backend(Arc::new(StubChatBackend::new(
                responses.into_iter().map(ToOwned::to_owned).collect(),
            )))
            .with_config(test_cfg())
    }

    fn accept_verdict() -> &'static str {
        r#"{"verdict": "accept", "caveats": [], "reasons": [], "confidence": 0.9}"#
    }

    fn reject_verdict() -> &'static str {
        r#"{"verdict": "reject", "caveats": [], "reasons": ["x"], "confidence": 0.8}"#
    }

    // ── M3.2: prompts + parse ────────────────────────────────────────────

    #[tokio::test]
    async fn blue_returns_plain_string() {
        let backend: LocalBackend = Arc::new(StubChatBackend::always("the answer"));
        let raw = blue_answer(&backend, "What is 2+2?").await.unwrap();
        assert_eq!(raw, "the answer");
    }

    #[test]
    fn red_objections_parse_from_array() {
        let raw = r#"[{"category": "factual", "description": "wrong", "severity": 0.8, "target_claim": 7}]"#;
        let objections = parse_objections(raw).unwrap();
        assert_eq!(objections.len(), 1);
        assert_eq!(objections[0].category, "factual");
        assert_eq!(objections[0].severity, 0.8);
        assert_eq!(objections[0].target_claim, Some(NodeId::from_int(7)));
    }

    #[test]
    fn red_objections_parse_from_wrapped_object() {
        let raw =
            r#"{"objections": [{"category": "safety", "description": "unsafe", "severity": 0.9}]}"#;
        let objections = parse_objections(raw).unwrap();
        assert_eq!(objections.len(), 1);
        assert_eq!(
            objections[0].target_claim, None,
            "target_claim defaults to null"
        );
    }

    #[test]
    fn red_objections_parse_from_fenced_json() {
        let raw =
            "```json\n[{\"category\": \"a\", \"description\": \"b\", \"severity\": 0.5}]\n```";
        let objections = parse_objections(raw).unwrap();
        assert_eq!(objections.len(), 1);
    }

    #[test]
    fn judge_accept_shape() {
        let (verdict, confidence) = parse_judge(
            r#"{"verdict": "accept", "caveats": [], "reasons": [], "confidence": 0.9}"#,
        )
        .unwrap();
        assert!(matches!(verdict, JudgeVerdict::Accept));
        assert_eq!(confidence, 0.9);
    }

    #[test]
    fn judge_accept_with_caveats_shape() {
        let (verdict, _) = parse_judge(
            r#"{"verdict": "accept_with_caveats", "caveats": ["cite sources"], "reasons": [], "confidence": 0.7}"#,
        )
        .unwrap();
        match verdict {
            JudgeVerdict::AcceptWithCaveats { caveats } => {
                assert_eq!(caveats, vec!["cite sources".to_string()]);
            }
            other => panic!("expected AcceptWithCaveats, got {other:?}"),
        }
    }

    #[test]
    fn judge_reject_shape() {
        let (verdict, _) = parse_judge(
            r#"{"verdict": "reject", "caveats": [], "reasons": ["unsupported"], "confidence": 0.3}"#,
        )
        .unwrap();
        match verdict {
            JudgeVerdict::Reject { reasons } => {
                assert_eq!(reasons, vec!["unsupported".to_string()]);
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn judge_bad_verdict_is_error_not_panic() {
        assert!(parse_judge(r#"{"verdict": "maybe", "confidence": 0.5}"#).is_err());
    }

    // ── M3.5: bounded pass loop ──────────────────────────────────────────

    #[tokio::test]
    async fn judge_accepts_first_pass_no_rewind_no_interview() {
        let route = route_with_judge(vec![accept_verdict()]);
        let result = route.execute(&ctx("question")).await.unwrap();
        assert_eq!(result.blue_answer, "candidate answer");
        assert!(!result.rewound);
        assert!(result.interview_questions.is_empty());
        assert!(!result.frontier_escalation);
        assert!(matches!(result.judge_verdict, JudgeVerdict::Accept));
    }

    #[tokio::test]
    async fn judge_rejects_then_accepts_rewinds() {
        let session = Arc::new(Mutex::new(DependencySession::new("sess-rigor")));
        let route = route_with_judge(vec![reject_verdict(), accept_verdict()]);
        let mut ctx = ctx("question");
        ctx.session = Some(Arc::clone(&session));

        let result = route.execute(&ctx).await.unwrap();
        assert!(result.rewound, "material rejection must rewind");
        assert_eq!(result.blue_answer, "second candidate answer");
        assert!(matches!(result.judge_verdict, JudgeVerdict::Accept));
        assert!(result.interview_questions.is_empty());
    }

    #[tokio::test]
    async fn judge_rejects_both_passes_interviews() {
        let session = Arc::new(Mutex::new(DependencySession::new("sess-rigor")));
        let route = route_with_judge(vec![reject_verdict(), reject_verdict()]);
        let mut ctx = ctx("question");
        ctx.session = Some(Arc::clone(&session));

        let result = route.execute(&ctx).await.unwrap();
        assert!(result.rewound);
        assert_eq!(result.blue_answer, "second candidate answer");
        assert!(!result.interview_questions.is_empty());
        assert!(result.interview_questions.len() <= 3, "bounded interview");
        assert!(
            !result.frontier_escalation,
            "high confidence -> no escalation"
        );
    }

    #[tokio::test]
    async fn immaterial_rejection_does_not_rewind() {
        let route = RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::always("candidate answer")))
            .with_red_backend(Arc::new(StubChatBackend::always(
                r#"[{"category": "style", "description": "minor nit", "severity": 0.2}]"#,
            )))
            .with_judge_backend(Arc::new(StubChatBackend::always(
                r#"{"verdict": "reject", "caveats": [], "reasons": ["nit"], "confidence": 0.5}"#,
            )))
            .with_config(test_cfg());
        let result = route.execute(&ctx("question")).await.unwrap();
        assert!(!result.rewound, "low-severity rejection must not rewind");
        assert!(
            !result.interview_questions.is_empty(),
            "still resolves to clarify"
        );
    }

    #[tokio::test]
    async fn low_confidence_final_rejection_escalates() {
        let route = RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::new(vec![
                "candidate answer".to_string(),
                "second candidate answer".to_string(),
            ])))
            .with_red_backend(Arc::new(StubChatBackend::new(vec![
                r#"[{"category": "factual", "description": "unsupported claim", "severity": 0.9}]"#
                    .to_string(),
                r#"[{"category": "factual", "description": "unsupported claim", "severity": 0.9}]"#
                    .to_string(),
            ])))
            .with_judge_backend(Arc::new(StubChatBackend::new(vec![
                r#"{"verdict": "reject", "caveats": [], "reasons": ["x"], "confidence": 0.8}"#
                    .into(),
                r#"{"verdict": "reject", "caveats": [], "reasons": ["x"], "confidence": 0.2}"#
                    .into(),
            ])))
            .with_config(test_cfg());
        let result = route.execute(&ctx("question")).await.unwrap();
        assert!(
            result.frontier_escalation,
            "low judge confidence is the explicit escalation trigger"
        );
    }

    #[tokio::test]
    async fn invalid_judge_json_returns_judge_error() {
        let route = RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::always("candidate answer")))
            .with_red_backend(Arc::new(StubChatBackend::always(
                r#"[{"category": "factual", "description": "x", "severity": 0.9}]"#,
            )))
            .with_judge_backend(Arc::new(StubChatBackend::always("not json")));
        assert!(matches!(
            route.execute(&ctx("question")).await,
            Err(RigorError::Judge(_))
        ));
    }

    #[tokio::test]
    async fn missing_role_backend_is_explicit_error() {
        let route = RigorRoute::new().with_blue_backend(Arc::new(StubChatBackend::always("x")));
        assert!(matches!(
            route.execute(&ctx("question")).await,
            Err(RigorError::Unconfigured(_))
        ));
    }

    // ── M3.3: session steps + real rewind ────────────────────────────────

    #[tokio::test]
    async fn material_rejection_resets_rigor_steps_to_pending() {
        let session = Arc::new(Mutex::new(DependencySession::new("sess-rigor")));
        let route = route_with_judge(vec![reject_verdict(), accept_verdict()]);
        let mut ctx = ctx("question");
        ctx.session = Some(Arc::clone(&session));

        let result = route.execute(&ctx).await.unwrap();
        assert!(result.rewound);

        let s = lock(&session);
        assert_eq!(s.get_step("rigor.red").unwrap().status, StepStatus::Pending);
        assert_eq!(
            s.get_step("rigor.judge").unwrap().status,
            StepStatus::Pending
        );
        assert_eq!(
            s.get_step("rigor.blue").unwrap().status,
            StepStatus::Completed
        );
    }

    #[tokio::test]
    async fn immaterial_rejection_does_not_rewind_steps() {
        let session = Arc::new(Mutex::new(DependencySession::new("sess-rigor")));
        let route = RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::always("candidate answer")))
            .with_red_backend(Arc::new(StubChatBackend::always(
                r#"[{"category": "style", "description": "nit", "severity": 0.1}]"#,
            )))
            .with_judge_backend(Arc::new(StubChatBackend::always(
                r#"{"verdict": "reject", "caveats": [], "reasons": ["nit"], "confidence": 0.5}"#,
            )))
            .with_config(test_cfg());
        let mut ctx = ctx("question");
        ctx.session = Some(Arc::clone(&session));

        let result = route.execute(&ctx).await.unwrap();
        assert!(!result.rewound);

        let s = lock(&session);
        // No rewind: rigor.red/judge stay Pending (never completed), rigor.blue
        // stays Completed (no second pass).
        assert_eq!(s.get_step("rigor.red").unwrap().status, StepStatus::Pending);
        assert_eq!(
            s.get_step("rigor.blue").unwrap().status,
            StepStatus::Completed
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rewind_restores_kv_snapshot_for_real() {
        // Mirrors `dag_session::tests::test_rewind_restores_kv_snapshot_for_real`:
        // a session carrying the KvCacheManager (D6) has its stored snapshot
        // actually restored on rewind.
        use crate::kv_cache::{ColdKvCache, HotKvCache, KvCacheManager, KvSnapshot};

        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let hot = Arc::new(HotKvCache::new(10, 1024));
        let kv = KvCacheManager::new(
            Arc::clone(&hot),
            Arc::new(ColdKvCache::new(
                dir.path(),
                1024,
                86400,
                crate::config::EvictionPolicy::Lru,
            )),
        );
        let src_file = src_dir.path().join("rewind.kv");
        tokio::fs::write(&src_file, b"kv bytes").await.unwrap();
        kv.store(KvSnapshot {
            model: "model-x".into(),
            adapter: None,
            session_id: "sess-rigor".into(),
            file_path: src_file,
            token_count: Some(42),
            created_at: common_core::now_secs(),
            last_used_at: common_core::now_secs(),
            llama_cpp_version: Some("0.1.0".into()),
            model_quant: None,
            base_model_hash: Some("abc".into()),
        })
        .await
        .unwrap();
        hot.remove("model-x", None, "sess-rigor");

        let session = Arc::new(Mutex::new(
            DependencySession::new("sess-rigor")
                .with_model("model-x")
                .with_kv_cache(kv),
        ));
        let route = route_with_judge(vec![reject_verdict(), accept_verdict()]);
        let mut ctx = ctx("question");
        ctx.session = Some(Arc::clone(&session));

        let result = route.execute(&ctx).await.unwrap();
        assert!(result.rewound, "a material rejection must rewind for real");
        // The snapshot file was restored into the hot tier on rewind.
        assert!(
            hot.get("model-x", None, "sess-rigor").is_some(),
            "rewind must promote the stored snapshot back into the hot tier"
        );
    }

    // ── M3.4: red-team filtered view at LOD0 ─────────────────────────────

    /// A recording backend that captures the user message it receives (the red
    /// prompt) so the test can assert on the rendered view material.
    struct RecordingRed {
        captured: Arc<Mutex<Vec<String>>>,
    }

    impl ChatBackend for RecordingRed {
        fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, fluent_llm::LlmError> {
            let user = messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            lock(&self.captured).push(user);
            Ok(r#"[{"category": "factual", "description": "x", "severity": 0.9}]"#.to_string())
        }
    }

    #[tokio::test]
    async fn red_prompt_contains_live_lod0_and_excludes_dead_end() {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-rigor-{}",
            common_core::hash::uuid_v4()
        ));
        let ledger = Arc::new(ContentNodeLedger::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);

        let store = ledger.node_store().clone();
        let live = store
            .record_request("sess-rigor", "r-live", "LIVE CLAIM TEXT at LOD0")
            .unwrap();
        let dead = store
            .record_request("sess-rigor", "r-dead", "DEAD END TEXT to exclude")
            .unwrap();

        // Mark the dead node rejected (accepted = false).
        store
            .record_result(dead, false, Some(0.1), "DEAD END TEXT to exclude")
            .unwrap();
        let _ = live;

        let captured = Arc::new(Mutex::new(Vec::new()));
        let red_backend: LocalBackend = Arc::new(RecordingRed {
            captured: Arc::clone(&captured),
        });
        let route = RigorRoute::new()
            .with_blue_backend(Arc::new(StubChatBackend::always("candidate answer")))
            .with_red_backend(red_backend)
            .with_judge_backend(Arc::new(StubChatBackend::always(accept_verdict())))
            .with_config(test_cfg());

        let mut ctx = ctx("question");
        ctx.ledger = Some(Arc::clone(&ledger));

        let _result = route.execute(&ctx).await.unwrap();

        let prompt = lock(&captured).last().cloned().unwrap_or_default();
        assert!(
            prompt.contains("LIVE CLAIM TEXT at LOD0"),
            "red prompt must include the live claim's LOD0 text, got: {prompt}"
        );
        assert!(
            !prompt.contains("DEAD END TEXT to exclude"),
            "red prompt must exclude the rejected dead end, got: {prompt}"
        );
    }
}
