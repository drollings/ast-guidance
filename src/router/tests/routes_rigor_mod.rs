use super::*;
use crate::config::rounds::{BoundedRounds, EscalationConfidence, SeverityThreshold};
use crate::dag_session::DependencySession;
use crate::test_stubs::StubChatBackend;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_types::StepStatus;

fn ctx(message: &str) -> RigorContext {
    RigorContext {
        user_message: message.to_string(),
        session_id: "sess-rigor".into(),
        model_endpoint: "model-x".into(),
        session: None,
        ledger: None,
        kv_instance: None,
    }
}

fn test_cfg() -> RigorConfig {
    RigorConfig {
        max_passes: BoundedRounds::try_from(2).unwrap(),
        severity_threshold: SeverityThreshold::new(0.7).unwrap(),
        escalation_confidence: EscalationConfidence::new(0.4).unwrap(),
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

// -- Prompts + parse --------------------------------------------

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

// -- Bounded pass loop ------------------------------------------

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

// -- Session steps + real rewind --------------------------------

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
    // a session carrying the SnapshotStore has its stored snapshot
    // actually restored on rewind.
    use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex, SnapshotStore, KvSnapshot};

    let dir = tempfile::tempdir().unwrap();
    let src_dir = tempfile::tempdir().unwrap();
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    );
    let src_file = src_dir.path().join("rewind.kv");
    tokio::fs::write(&src_file, b"kv bytes").await.unwrap();
    kv.store(KvSnapshot {
        model: "model-x".into(),
        adapter: None,
        session_id: "sess-rigor".into(),
        snapshot_name: "readfiles".into(),
        instance: Some("scratch".into()),
        file_path: src_file,
        token_count: Some(42),
        created_at: common_core::now_secs(),
        last_used_at: common_core::now_secs(),
        llama_cpp_version: Some("0.1.0".into()),
        model_quant: None,
        base_model_hash: Some("abc".into()),
        turn_seq: None,
    })
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blue_save_then_rewind_produces_pending_kv_fields() {
    // The blue-pass completion saves a fork snapshot on the blue
    // instance (stub fork receives the POST), then the material-rejection
    // rewind finds real metadata and the session carries pending KV fields
    // (`snapshot`/`instance`/`id_slot`) for the next dispatch.
    use crate::instances::stub::StubServer;
    use crate::instances::InstanceClient;
    use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex, SnapshotStore};

    // Stub fork: answer the snapshot save + any /instances reads.
    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "POST" && path == "/instances/scratch/snapshot" {
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
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    )
    .with_fork_io(fork);

    let session = Arc::new(Mutex::new(
        DependencySession::new("sess-rigor")
            .with_model("model-x")
            .with_kv_cache(kv),
    ));
    let route = route_with_judge(vec![reject_verdict(), accept_verdict()])
        .with_kv_cache();
    let mut ctx = ctx("question");
    ctx.session = Some(Arc::clone(&session));
    ctx.kv_instance = Some("scratch".into());

    let result = route.execute(&ctx).await.unwrap();
    assert!(result.rewound, "material rejection must rewind");

    // The fork received the snapshot-save POST.
    let recorded = stub.recorded();
    assert!(
        recorded
            .iter()
            .any(|(m, p, _)| m == "POST" && p == "/instances/scratch/snapshot"),
        "blue completion must POST the fork snapshot save, got: {recorded:?}"
    );

    // The rewind restored the just-saved snapshot: pending fields carry the
    // snapshot name + instance for the next dispatch.
    let s = lock(&session);
    let pending = s.pending_kv_fields().expect("pending fields set");
    assert_eq!(pending.0, "rigor-blue-rigor.blue");
    assert_eq!(pending.1.as_deref(), Some("scratch"));
    assert_eq!(pending.2, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blue_save_without_kv_instance_is_a_no_op() {
    // Degradation: with a kv manager + fork but no `kv_instance`,
    // the blue save is skipped (never a crash) and the stub sees no POST.
    use crate::instances::stub::StubServer;
    use crate::instances::InstanceClient;
    use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex, SnapshotStore};

    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, _path, _body| {
            if method == "POST" {
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
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    )
    .with_fork_io(fork);

    let session = Arc::new(Mutex::new(
        DependencySession::new("sess-rigor")
            .with_model("model-x")
            .with_kv_cache(kv),
    ));
    let route = route_with_judge(vec![reject_verdict(), accept_verdict()])
        .with_kv_cache();
    let mut ctx = ctx("question");
    ctx.session = Some(Arc::clone(&session));
    // kv_instance is None -> save skipped.

    let result = route.execute(&ctx).await.unwrap();
    assert!(result.rewound);
    assert!(
        stub.recorded().is_empty(),
        "no kv_instance -> no fork snapshot save",
    );
    let s = lock(&session);
    assert!(
        s.pending_kv_fields().is_none(),
        "no metadata saved -> no pending fields"
    );
}

// -- Red-team filtered view at LOD0 -----------------------------

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

// -- Judge uses the LedgerPromptAssembler -----------------------

/// A recording judge backend that captures the user message it receives.
struct RecordingJudge {
    captured: Arc<Mutex<Vec<String>>>,
}

impl ChatBackend for RecordingJudge {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, fluent_llm::LlmError> {
        let user = messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        lock(&self.captured).push(user);
        Ok(accept_verdict().to_string())
    }
}

#[tokio::test]
async fn judge_prompt_renders_ledger_via_assembler() {
    // With a ledger + assembler attached, the judge's review prompt
    // folds in the session ledger rendered through the assembler's
    // budget/relevance rules (red team keeps its LOD0 view unchanged).
    let dir = std::env::temp_dir().join(format!(
        "coral-router-rigor-judge-{}",
        common_core::hash::uuid_v4()
    ));
    let ledger = Arc::new(ContentNodeLedger::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    ledger
        .record_request("sess-rigor", "r1", "JUDGE LEDGER CONTEXT at LOD0")
        .unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let judge_backend: LocalBackend = Arc::new(RecordingJudge {
        captured: Arc::clone(&captured),
    });
    let route = RigorRoute::new()
        .with_blue_backend(Arc::new(StubChatBackend::always("candidate answer")))
        .with_red_backend(Arc::new(StubChatBackend::always("[]")))
        .with_judge_backend(judge_backend)
        .with_config(test_cfg())
        .with_prompt_assembler(
            LedgerPromptAssembler,
            PromptBudget::new(10_000),
            LodSpec::full(),
        );

    let mut ctx = ctx("question");
    ctx.ledger = Some(Arc::clone(&ledger));

    let result = route.execute(&ctx).await.unwrap();
    assert!(matches!(result.judge_verdict, JudgeVerdict::Accept));
    let prompt = lock(&captured).last().cloned().unwrap_or_default();
    assert!(
        prompt.contains("JUDGE LEDGER CONTEXT at LOD0"),
        "judge prompt must include the assembled ledger context, got: {prompt}"
    );
}
