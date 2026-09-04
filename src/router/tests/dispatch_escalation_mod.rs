use super::*;
use serde_json::Value;

use std::collections::HashMap;

use fluent_llm::{ChatMessage, LlmError};

use super::assemble::{parse_subtask_array, summarize_votes};
use crate::config::FrontierConfig;
use crate::server::dispatch::render_prompt;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse};
use crate::dag_session::{SessionStep, StepResult};
use crate::dispatch::backend::DispatchBackend;
use crate::dispatch::backend::StreamHandle;
use crate::testing::mock::TranscriptProvider;
use fluent_types::ContextHit;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

fn test_request(text: &str) -> RouterRequest {
    RouterRequest {
        model: "fast".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: None,
        agent_id: None,
        adapter: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        metadata: Default::default(),
    }
}

fn text_response(model: &str, content: &str) -> RouterResponse {
    RouterResponse {
        id: "r".into(),
        object: "chat.completion".into(),
        created: 0,
        model: model.into(),
        choices: vec![crate::types::RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text(content.into()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        }],
        usage: crate::types::Usage::default(),
    }
}

/// Async frontier stub: canned responses (popped FIFO) + a record of
/// every request it received.
struct StubFrontier {
    responses: Mutex<VecDeque<RouterResponse>>,
    received: Mutex<Vec<RouterRequest>>,
}

impl StubFrontier {
    fn new(responses: Vec<RouterResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            received: Mutex::new(Vec::new()),
        })
    }
    fn received_texts(&self) -> Vec<String> {
        self.received
            .lock()
            .unwrap()
            .iter()
            .map(render_prompt)
            .collect()
    }
}

impl DispatchBackend for StubFrontier {
    fn complete(
        &self,
        request: RouterRequest,
        _model: String,
        _params: Option<Value>,
        _idle_timeout_ms: u64,
        _total_timeout_ms: u64,
        _filter_thinking: bool,
    ) -> Pin<Box<dyn Future<Output = Result<RouterResponse, DispatchError>> + Send>> {
        self.received.lock().unwrap().push(request);
        let resp = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(DispatchError::AllBackendsFailed);
        Box::pin(async move { resp })
    }

    fn stream_complete_with_abort(
        &self,
        _request: RouterRequest,
        _model: String,
        _params: Option<Value>,
        _idle_timeout_ms: u64,
        _total_timeout_ms: u64,
        _filter_thinking: bool,
        _abort: Option<fluent_concurrency::stream::StreamAbort>,
    ) -> Pin<Box<dyn Future<Output = Result<StreamHandle, DispatchError>> + Send>> {
        Box::pin(async move { Err(DispatchError::AllBackendsFailed) })
    }
}

/// Sync local-role stub: routes each call by the user message content.
fn func_backend<F>(f: F) -> Arc<dyn fluent_llm::client::ChatBackend>
where
    F: Fn(&[ChatMessage]) -> Result<String, LlmError> + Send + Sync + 'static,
{
    struct FuncBackend<F>(F);
    impl<F> fluent_llm::client::ChatBackend for FuncBackend<F>
    where
        F: Fn(&[ChatMessage]) -> Result<String, LlmError> + Send + Sync,
    {
        fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
            (self.0)(messages)
        }
    }
    Arc::new(FuncBackend(f))
}

/// Read the assistant text out of a `HyperResponse` body.
async fn read_body_text(resp: HyperResponse) -> String {
    use http_body_util::BodyExt;
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn last_user_message(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn filter_config() -> EscalationLadderConfig {
    EscalationLadderConfig {
        modes: vec![EscalationMode::Filter],
        frontier: Some(FrontierConfig {
            endpoint: "http://frontier.test/v1/chat/completions".into(),
            api_key_env: None,
            model: "claude".into(),
        }),
        classifier_parallel: 2,
        ..Default::default()
    }
}

fn full_modes_config() -> EscalationLadderConfig {
    EscalationLadderConfig {
        modes: vec![
            EscalationMode::Filter,
            EscalationMode::Question,
            EscalationMode::Team,
            EscalationMode::Turnover,
        ],
        frontier: Some(FrontierConfig {
            endpoint: "http://frontier.test/v1/chat/completions".into(),
            api_key_env: None,
            model: "claude".into(),
        }),
        classifier_parallel: 2,
        ..Default::default()
    }
}

#[derive(Default)]
struct MapContextCache {
    map: Mutex<HashMap<String, String>>,
}

impl fluent_types::ContextCache for MapContextCache {
    fn lookup(&self, query: &str) -> Option<ContextHit> {
        self.map
            .lock()
            .unwrap()
            .get(query)
            .map(|content| ContextHit {
                source: "test-cache".into(),
                content: content.clone(),
                score: 0.99,
                metadata: None,
            })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn context_cache_short_circuits_before_any_mode() {
    let map_cache = Arc::new(MapContextCache::default());
    map_cache
        .map
        .lock()
        .unwrap()
        .insert("known question".into(), "verified answer".into());
    let cache: Arc<dyn fluent_types::ContextCache> = map_cache;

    let ladder = Ladder::new(
        full_modes_config(),
        EscalationBackends {
            // No frontier backends required: the cache hit returns first.
            frontier: StubFrontier::new(vec![]),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("known question");
    let ctx = EscalationContext {
        request: &request,
        user_text: "known question",
        model_name: "fast",
        context_cache: Some(&cache),
        session: None,
    };
    let resp = ladder.try_escalate(&ctx).await.expect("short-circuits");
    assert_eq!(read_body_text(resp).await, "verified answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_context_hit_does_not_short_circuit() {
    let cache: Arc<dyn fluent_types::ContextCache> = Arc::new(MapContextCache::default());
    let frontier = StubFrontier::new(vec![text_response("claude", "local answer")]);
    let ladder = Ladder::new(
        full_modes_config(),
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("what is 2+2?");
    let ctx = EscalationContext {
        request: &request,
        user_text: "what is 2+2?",
        model_name: "fast",
        context_cache: Some(&cache),
        session: None,
    };
    // Filter mode runs: PII transform is identity here, frontier returns
    // a clean answer → accepted.
    let resp = ladder.try_escalate(&ctx).await.expect("filter accepted");
    assert_eq!(read_body_text(resp).await, "local answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filter_mode_accepts_clean_frontier_response() {
    let frontier = StubFrontier::new(vec![text_response("claude", "The answer is 4")]);
    let ladder = Ladder::new(
        filter_config(),
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("What is 2+2?");
    let ctx = EscalationContext {
        request: &request,
        user_text: "What is 2+2?",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    let resp = ladder.try_escalate(&ctx).await.expect("accepted");
    assert_eq!(read_body_text(resp).await, "The answer is 4");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filter_mode_rejects_pii_flagged_response() {
    let frontier = StubFrontier::new(vec![text_response(
        "claude",
        "Reach me at alice@example.com",
    )]);
    let ladder = Ladder::new(
        filter_config(),
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("email me");
    let ctx = EscalationContext {
        request: &request,
        user_text: "email me",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    assert!(
        ladder.try_escalate(&ctx).await.is_none(),
        "PII-flagged frontier output must not be accepted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mode_walk_escalates_on_rejection() {
    // Filter rejects (PII in output) → turnover accepts.
    let frontier = StubFrontier::new(vec![
        text_response("claude", "call bob@example.com"),
        text_response("claude", "escalated answer"),
    ]);
    let config = EscalationLadderConfig {
        modes: vec![EscalationMode::Filter, EscalationMode::Turnover],
        frontier: filter_config().frontier,
        ..Default::default()
    };
    let ladder = Ladder::new(
        config,
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("help me");
    let ctx = EscalationContext {
        request: &request,
        user_text: "help me",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    let resp = ladder.try_escalate(&ctx).await.expect("turnover accepted");
    assert_eq!(read_body_text(resp).await, "escalated answer");
    assert_eq!(
        frontier.received_texts().len(),
        2,
        "both modes hit frontier"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn question_mode_accepts_assembled_answer() {
    let decomposer = func_backend(|_| Ok(r#"["hyp1", "hyp2"]"#.into()));
    let assembler = func_backend(|messages| {
        let msg = last_user_message(messages);
        if msg.contains("Original query") {
            Ok("assembled final answer".into())
        } else {
            // ResultScorer prompt — canonical scorer output shape.
            Ok(r#"{"score": 0.9, "accepted": true, "reason": "good", "summary": "ok"}"#.into())
        }
    });
    let frontier = StubFrontier::new(vec![
        text_response("claude", "answer one"),
        text_response("claude", "answer two"),
    ]);
    let config = EscalationLadderConfig {
        modes: vec![EscalationMode::Question],
        frontier: filter_config().frontier,
        ..Default::default()
    };
    let ladder = Ladder::new(
        config,
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: Some(decomposer),
            assembler: Some(assembler),
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("how do I tune a model?");
    let ctx = EscalationContext {
        request: &request,
        user_text: "how do I tune a model?",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    let resp = ladder.try_escalate(&ctx).await.expect("accepted");
    assert_eq!(read_body_text(resp).await, "assembled final answer");
    assert_eq!(
        frontier.received_texts().len(),
        2,
        "one frontier call per hypothetical"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn team_mode_accepts_after_judge_verdict() {
    let classifier =
        func_backend(|_| Ok(r#"{"approach": "decompose", "confidence": 0.9}"#.into()));
    let draft = func_backend(|_| Ok(r#"["subtask A", "subtask B"]"#.into()));
    let judge = func_backend(|messages| {
        let msg = last_user_message(messages);
        if msg.contains("\n\nAnswer:\n") {
            Ok(r#"{"accepted": true, "reason": "solved"}"#.into())
        } else {
            Ok("frontier prompt with only the gap".into())
        }
    });
    let frontier = StubFrontier::new(vec![text_response("claude", "frontier solution")]);
    let config = EscalationLadderConfig {
        modes: vec![EscalationMode::Team],
        frontier: filter_config().frontier,
        classifier_parallel: 2,
        ..Default::default()
    };
    let ladder = Ladder::new(
        config,
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: Some(classifier),
            draft: Some(draft),
            judge: Some(judge),
        },
    );
    let request = test_request("solve the hard problem");
    let ctx = EscalationContext {
        request: &request,
        user_text: "solve the hard problem",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    let resp = ladder.try_escalate(&ctx).await.expect("accepted");
    assert_eq!(read_body_text(resp).await, "frontier solution");
    // The frontier received the judge-crafted prompt, not the raw request.
    let received = frontier.received_texts();
    assert!(
        received[0].contains("frontier prompt"),
        "frontier must receive the judge-crafted prompt: {received:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn team_mode_rejects_when_judge_rejects() {
    let classifier = func_backend(|_| Ok(r#"{"approach": "x", "confidence": 0.5}"#.into()));
    let draft = func_backend(|_| Ok(r#"["a"]"#.into()));
    let judge = func_backend(|messages| {
        let msg = last_user_message(messages);
        if msg.contains("\n\nAnswer:\n") {
            Ok(r#"{"accepted": false, "reason": "still gap"}"#.into())
        } else {
            Ok("gap prompt".into())
        }
    });
    let frontier = StubFrontier::new(vec![text_response("claude", "partial")]);
    let config = EscalationLadderConfig {
        modes: vec![EscalationMode::Team],
        frontier: filter_config().frontier,
        classifier_parallel: 1,
        ..Default::default()
    };
    let ladder = Ladder::new(
        config,
        EscalationBackends {
            frontier,
            decomposer: None,
            assembler: None,
            classifier: Some(classifier),
            draft: Some(draft),
            judge: Some(judge),
        },
    );
    let request = test_request("hard");
    let ctx = EscalationContext {
        request: &request,
        user_text: "hard",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    assert!(
        ladder.try_escalate(&ctx).await.is_none(),
        "judge reject → no acceptance"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn turnover_marks_session_frontier_owned_and_appends_ledger() {
    let frontier = StubFrontier::new(vec![text_response("claude", "handoff answer")]);
    let config = EscalationLadderConfig {
        modes: vec![EscalationMode::Turnover],
        frontier: filter_config().frontier,
        ..Default::default()
    };
    let ladder = Ladder::new(
        config,
        EscalationBackends {
            frontier: frontier.clone(),
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );

    let session = Arc::new(Mutex::new(DependencySession::new("sess-1")));
    {
        let mut s = session.lock().unwrap();
        s.add_step(SessionStep::new("step-1", "reproduce")).unwrap();
        s.complete_step(
            "step-1",
            StepResult {
                content: "reproduced the crash".into(),
                accepted: true,
                score: Some(0.9),
                latency_ms: 0,
                error: None,
            },
        )
        .unwrap();
    }

    let request = test_request("what next?");
    let ctx = EscalationContext {
        request: &request,
        user_text: "what next?",
        model_name: "fast",
        context_cache: None,
        session: Some(&session),
    };
    let resp = ladder.try_escalate(&ctx).await.expect("handoff");
    assert_eq!(read_body_text(resp).await, "handoff answer");

    assert!(
        session.lock().unwrap().is_frontier_owned(),
        "turnover must mark the session frontier-owned"
    );
    let received = frontier.received_texts();
    assert!(
        received[0].contains("reproduced the crash"),
        "turnover must append the session ledger: {received:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exhaustion_returns_none() {
    let frontier = StubFrontier::new(vec![text_response("claude", "call bob@example.com")]);
    let ladder = Ladder::new(
        filter_config(),
        EscalationBackends {
            frontier,
            decomposer: None,
            assembler: None,
            classifier: None,
            draft: None,
            judge: None,
        },
    );
    let request = test_request("email me");
    let ctx = EscalationContext {
        request: &request,
        user_text: "email me",
        model_name: "fast",
        context_cache: None,
        session: None,
    };
    assert!(
        ladder.try_escalate(&ctx).await.is_none(),
        "exhaustion → None so the caller returns fallback_completion"
    );
}

#[test]
fn votes_distribution_counts_approaches() {
    let votes = vec![
        r#"{"approach": "decompose", "confidence": 0.9}"#.to_string(),
        r#"{"approach": "decompose", "confidence": 0.8}"#.to_string(),
        "plain text vote".to_string(),
    ];
    let dist = summarize_votes(&votes);
    assert!(dist.contains("2/3 say: decompose"), "dist: {dist}");
    assert!(dist.contains("1/3 say: plain text vote"), "dist: {dist}");
}

#[test]
fn parse_subtask_array_handles_fenced_json() {
    let arr = parse_subtask_array("```json\n[\"a\", \"b\"]\n```").unwrap();
    assert_eq!(arr, vec!["a", "b"]);
    assert!(parse_subtask_array("not json").is_none());
}

#[test]
fn ladder_builds_from_config_with_transcript_local_roles() {
    // `TranscriptProvider` is a sync `ChatBackend` — the same trait the
    // ladder's local roles use, so builder wiring type-checks.
    let backends = EscalationBackends {
        frontier: StubFrontier::new(vec![]),
        decomposer: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
        assembler: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
        classifier: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
        draft: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
        judge: Some(Arc::new(TranscriptProvider::new(HashMap::new()))),
    };
    let ladder = Ladder::new(full_modes_config(), backends);
    assert_eq!(ladder.config().modes.len(), 4);
}
