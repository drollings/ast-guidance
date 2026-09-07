use super::*;

fn base_target() -> RoutingTarget {
    crate::pipeline::RoutingTarget {
        url: "http://x/v1/chat/completions".into(),
        model: "base:swarm".into(),
        group: None,
        target_name: Some("swarm".into()),
        params: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        filter_thinking: false,
        retry_count: 0,
        retry_base_interval_s: 1,
        stream: true,
        idle_timeout_ms: 5000,
        total_timeout_ms: 30000,
        api_key: None,
        fallbacks: vec![],
        is_onnx: false,
    }
}

#[test]
fn apply_pending_snapshot_sets_request_fields() {
    let rt = apply_pending_snapshot(&base_target(), "readfiles".into(), Some("scratch".into()), 2);
    assert_eq!(rt.snapshot.as_deref(), Some("readfiles"));
    assert_eq!(rt.instance.as_deref(), Some("scratch"));
    assert_eq!(rt.id_slot, Some(2));
}

#[test]
fn apply_pending_snapshot_preserves_existing_instance() {
    let mut t = base_target();
    t.instance = Some("ledger".into());
    let rt = apply_pending_snapshot(&t, "readfiles".into(), Some("scratch".into()), 0);
    assert_eq!(rt.instance.as_deref(), Some("ledger"), "existing instance wins");
    assert_eq!(rt.snapshot.as_deref(), Some("readfiles"));
}

/// A stub that serves both the chat-completions endpoint and the
/// management `/instances` endpoint from one listener: the chat path
/// returns a 503 group-miss on the first call and a success completion on
/// the second; `/instances` allocates (201). Used to assert the
/// allocate-on-503 retry.
#[tokio::test]
async fn allocate_on_503_creates_instance_and_retries_once() {
    use crate::instances::stub::StubServer;
    use crate::instances::{management_base_url, InstanceClient, InstanceManager, InstancePool};
    use crate::config::InstanceProfile;
    use std::sync::Arc as StdArc;
    use std::sync::Mutex;

    let chat_calls = StdArc::new(Mutex::new(0usize));
    let chat_calls_c = chat_calls.clone();
    let handler: StdArc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
        StdArc::new(move |method, path, _body| {
            if method == "POST" && path.ends_with("/chat/completions") {
                let mut n = chat_calls_c.lock().unwrap();
                *n += 1;
                if *n == 1 {
                    // The fork's 503 group-miss payload.
                    return (
                        503,
                        r#"{"error":{"code":503,"message":"no free instance in group 'swarm'","type":"unavailable_error"}}"#
                            .into(),
                    );
                }
                return (
                    200,
                    r#"{"id":"x","object":"chat.completion","model":"base:swarm","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
                        .into(),
                );
            }
            (201, "{}".into())
        });
    let stub = StubServer::start(handler);

    let endpoint = format!("{}/v1/chat/completions", stub.base_url());
    let mut target = base_target();
    target.url = endpoint.clone();
    target.instance = Some("swarm".into());
    target.stream = false; // buffered dispatch for simplicity

    // A manager whose client points at the same server's management API.
    let client = InstanceClient::new(
        reqwest::Client::new(),
        management_base_url(&endpoint),
        None,
    );
    let profile = InstanceProfile {
        name: Some("swarm0".into()),
        group: Some("swarm".into()),
        count: 1,
        num_ctx: 16384,
        parallel: None,
        pinned: false,
        no_sleep: true,
        sleep_idle_seconds: None,
        default: false,
        resume: false,
        params: None,
        max_ctx: None,
        session: false,
    };
    let manager = Arc::new(InstanceManager::new(
        "base",
        client,
        vec![profile],
        crate::config::SidecarConfig::default(),
    ));
    let mut managers = std::collections::HashMap::new();
    managers.insert("base".into(), manager);
    let pool = InstancePool::from_managers(managers, None);

    let request = crate::types::RouterRequest {
        model: "base".into(),
        messages: vec![crate::types::RouterMessage {
            role: "user".into(),
            content: crate::types::RouterMessageContent::Text("hello".into()),
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
    };
    let deps = DispatchDeps {
        http_client: Arc::new(reqwest::Client::new()),
        cache: None,
        stats: Arc::new(ServerStats::default()),
        extractor: None,
        ladders: std::collections::HashMap::new(),
        context_cache: None,
        session: None,
        instance_pool: Some(pool),
        onnx_llm_backend: None,
    };
    let outcome = dispatch_real(&target, &request, "base", &deps, false, "hello")
        .await
        .expect("dispatch_real is infallible");
    assert!(outcome.response.status().is_success(), "retry succeeded");

    let recorded = stub.recorded();
    // Exactly two chat calls (first group-miss, then retry) and one
    // management `POST /instances` in between.
    let chat_hits = recorded
        .iter()
        .filter(|(m, p, _)| m == "POST" && p.ends_with("/chat/completions"))
        .count();
    let create_hits = recorded
        .iter()
        .filter(|(m, p, _)| m == "POST" && p == "/instances")
        .count();
    assert_eq!(chat_hits, 2, "group-miss then retry");
    assert_eq!(create_hits, 1, "a fresh instance was allocated between");
}

#[test]
fn dispatch_audit_emits_without_router_stage() {
    // Successful dispatch telemetry uses the `Classifier` label with the
    // `dispatched` reason preserved; no fresh audit names a `Router` stage.
    let target = base_target();
    let record = crate::audit::AuditRecord::route(
        crate::pipeline_types::PipelineStage::Classifier,
        crate::pipeline_types::StageVerdict::Passed,
        Some(&target),
        None,
        Some("dispatched"),
    );
    let detail = serde_json::to_value(&record).expect("serialize");
    assert_eq!(detail["detail"]["stage"], "Classifier");
    assert_eq!(detail["detail"]["reason"], "dispatched");
    assert_eq!(detail["detail"]["target_model"], "base:swarm");

    // Census: no production source still references the removed variant.
    // (`pipeline_types.rs` keeps only the historical `"Router"` compat string
    // in its deserializer + docs — never a variant arm.)
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut hits = Vec::new();
    for rel in [
        "src/pipeline.rs",
        "src/server/dispatch.rs",
        "src/test_stubs.rs",
    ] {
        let content =
            std::fs::read_to_string(manifest.join(rel)).expect("source readable");
        let count = content.matches("PipelineStage::Router").count();
        if count > 0 {
            hits.push((rel, count));
        }
    }
    assert!(
        hits.is_empty(),
        "fresh code must not reference PipelineStage::Router: {hits:?}"
    );
    let types = std::fs::read_to_string(manifest.join("src/pipeline_types.rs"))
        .expect("source readable");
    assert!(
        !types.contains("\n    Router,"),
        "the Router variant arm must be deleted"
    );
    assert!(
        types.contains("\"Router\""),
        "the historical Router compat string must be kept in the deserializer"
    );
}
