//! Config-synced routing integration tests.
//!
//! These replace the former `bin/router-mock-tests.sh` shell smoke suite,
//! which drifted from `env/coral-router.json` (it hardcoded model names and
//! route expectations that no longer exist). The assertions here are *derived
//! from* the config at runtime — every route (intent) declared there is probed
//! and the expected outcome is read from the config's `routes` → `model_groups`
//! mapping — so the tests cannot fall out of sync with the config. The
//! protocol-level checks (health, stats, 404, streaming, malformed input,
//! commands, PII) reproduce the retired script's breadth.
//!
//! Coverage:
//! 1. **Config sanity** (`config_route_groups_resolve_to_models`): every
//!    route's `group` names a non-empty `model_groups` ladder of declared
//!    models; `default_route` is declared.
//! 2. **Intent → model_group** (`route_intents_dispatch_to_their_model_groups`):
//!    every route is probed (multiple phrasings); the router's own route +
//!    group validation records zero mismatches and each probe is answered.
//! 3. **Direct model dispatch** (`every_declared_model_answers_directly`):
//!    every declared model answers when requested by key.
//! 4. **`always_route` semantics** (`always_route_routes_force_dispatch_over_classifier_respond`):
//!    `always_route: true` routes dispatch even when the classifier wants to
//!    answer directly.
//! 5. **Deterministic pre-filter** (`deterministic_commands_dispatch`,
//!    `pii_requests_are_blocked`): commands and PII are intercepted.
//! 6. **Protocol** (`health_and_stats_endpoints_report_ok`,
//!    `unknown_path_returns_404`, `streaming_flag_returns_sse_chunks`,
//!    `malformed_requests_are_rejected`).
//! 7. **Fixture sync** (`mock_transcript_fixture_stays_synced_with_config`,
//!    `mock_transcript_entries_serve_their_expected_answers`): the `--mock`
//!    binary's fixture stays consistent with the config and serves its
//!    declared answers end-to-end.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::config::{RouteRef, RouterConfig};
use crate::server::serve_http;
use crate::testing::mock::{
    load_transcript_file, transcript_provider_from_entries, MockDispatchContext, MockTranscriptEntry,
};
use crate::tests::common::{get, post_chat, TestServer};
use fluent_llm::client::ChatBackend;

/// `env/coral-router.json` — the single source of truth. Resolved relative to
/// the crate manifest (cargo runs tests with the package dir as CWD, so a
/// plain relative path would be fragile across invocations).
fn config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../env/coral-router.json")
}

/// `env/mock-transcripts.json` — the `--mock` binary's fixture.
fn mock_transcripts_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../env/mock-transcripts.json")
}

/// Load the live config. A failure here IS a test failure: the router boots
/// this exact file, so it must deserialize with the typed schema. The relative
/// `blacklist` path is resolved to an absolute one so the deterministic
/// pre-filter loads the real PII patterns regardless of the test process CWD
/// (cargo runs tests from the package dir, not the repo root).
fn load_config() -> RouterConfig {
    let path = config_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut config: RouterConfig = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("cannot parse {} as RouterConfig: {e}", path.display()));
    let cfg_dir = config_path();
    let repo_root = cfg_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root above env dir");
    for params in config.pipelines.values_mut() {
        if let Some(rel) = params.blacklist.as_mut() {
            *rel = repo_root.join(rel.as_str()).display().to_string();
        }
    }
    config
}

/// Curated primary probe prompts, keyed by route name. `probe_for_route`
/// falls back to the route's `description` (or its name) so a route added to
/// the config is exercised without a code change;
/// `route_probe_seeds_stay_synced_with_config` fails the build if a seed
/// names a route the config no longer declares, or a declared route is
/// neither seeded nor described (the two drift directions that would
/// otherwise degrade a probe silently).
static ROUTE_PROBE_SEEDS: &[(&str, &str)] = &[
    (
        "local",
        "What is the capital of France? Answer in one short sentence.",
    ),
    ("prose", "Write a short gothic story about a lighthouse keeper."),
    ("code", "Write a Rust function to compute Fibonacci numbers."),
    (
        "summarize",
        "Summarize this in one sentence: 'Q3 revenue reached $4.2M, up 12% YoY.'",
    ),
    (
        "explore",
        "Extract the dates and amounts from this email as JSON: 'Q3 invoice for $12,400 due October 15.'",
    ),
    (
        "explain",
        "Explain the EPR paradox and Bell's theorem.",
    ),
];

/// One demanding probe prompt per route, whose domain matches the route's
/// description so the intent is unambiguous. Falls back to the route's
/// `description` (or its name) so a route added to the config is exercised
/// without a code change.
fn probe_for_route(route: &str, rref: &RouteRef) -> String {
    let seeds = ROUTE_PROBE_SEEDS;
    seeds
        .iter()
        .find(|(name, _)| *name == route)
        .map(|(_, p)| p.to_string())
        .unwrap_or_else(|| {
            if rref.description.is_empty() {
                format!("Please help with: {route}")
            } else {
                rref.description.clone()
            }
        })
}

/// Additional phrasings per route so the dispatch path is exercised across
/// more than one surface per intent. A route without variations is still
/// covered by its primary probe.
fn varied_probes_for_route(route: &str) -> Vec<String> {
    let variations: &[(&str, &[&str])] = &[
        (
            "local",
            &[
                "What is 2+2?",
                "hi",
                "What color is the sky?",
                "Who wrote the Iliad?",
            ],
        ),
        (
            "code",
            &[
                "Write a Rust program that prints the first ten primes.",
                "Fix a deadlock in this Go program.",
                "Explain what a monad is in Haskell.",
            ],
        ),
        (
            "prose",
            &[
                "Write a haiku about autumn leaves.",
                "Draft a letter of complaint to a landlord.",
            ],
        ),
        (
            "summarize",
            &["Condense this paragraph into a single sentence: 'The company reported strong Q3 results driven by European expansion.'"],
        ),
        (
            "explore",
            &[
                "Search the web for the current asking price of a used 2018 Toyota Camry.",
                "Look up the population of Berlin.",
                "List every city mentioned in: 'We flew to Berlin, then London, then Tokyo.'",
            ],
        ),
        (
            "explain",
            &[
                "Translate 'Good morning' into French.",
                "Explain what a tort is.",
                "Why is the sky blue?",
                "Describe the mechanism of action and indications of metformin.",
            ],
        ),
    ];
    variations
        .iter()
        .find(|(name, _)| *name == route)
        .map(|(_, ps)| ps.iter().map(|p| (*p).to_string()).collect())
        .unwrap_or_default()
}

/// A classifier response that routes the probe to the given route (tree
/// verdict shape: the engine picks the child named by `route`).
fn route_classifier_response(route: &str) -> String {
    json!({
        "route": route,
        "coherence": 0.95,
        "safety": 0.9,
        "complexity": 5,
        "reason": "config-synced mock probe",
    })
    .to_string()
}

/// A mock transcript entry for a probe that must be routed to `route` and
/// dispatched through `expect_model_group`.
fn route_entry(route: &str, expect_model_group: &str, user_message: &str) -> MockTranscriptEntry {
    MockTranscriptEntry {
        user_message: user_message.to_string(),
        classifier_response: route_classifier_response(route),
        expected_route: Some(route.to_string()),
        expect_model_group: Some(expect_model_group.to_string()),
        dispatch_response: Some(format!("mock {route} answer")),
        rejected: false,
        reject_reason_contains: None,
    }
}

/// Derive mock transcript entries per route from the config: the primary probe
/// plus every variation. Each entry records the expected *route* and the
/// expected *model_group* (from the tree-derived view's `group`) for the
/// router's own validation.
fn transcripts_from_config(config: &RouterConfig) -> Vec<MockTranscriptEntry> {
    config
        .routes_view()
        .into_iter()
        .flat_map(|(route, rref)| {
            let mut probes = vec![probe_for_route(&route, &rref)];
            probes.extend(varied_probes_for_route(&route));
            probes
                .into_iter()
                .map(|probe| route_entry(&route, &rref.group, &probe))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Boot the real `serve_http` accept loop with the given config, a transcript
/// classifier that returns each probe's canned `target`, and a dispatch mock
/// that validates route + model_group resolution. Returns the server and the
/// shared mock context (whose `take_failures()` is the routing verdict).
async fn spawn_config_mock_server(
    config: RouterConfig,
    entries: Vec<MockTranscriptEntry>,
) -> (TestServer, Arc<MockDispatchContext>) {
    let backend: Arc<dyn ChatBackend> = Arc::new(transcript_provider_from_entries(&entries));
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let mock = Arc::new(MockDispatchContext::new(entries, vec![]));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let deps = crate::tests::common::test_deps(
        pipelines,
        &config,
        Some(Arc::clone(&mock)),
        None,
        None,
        HashMap::new(),
        None,
    );
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "config-sync test server failed");
        }
    });

    (TestServer { addr, handle }, mock)
}

/// The probe tables stay synced with the config in both drift directions:
/// every seed names a still-declared route, every route has a meaningful
/// probe source (a seed or a description), and the full probe set (primary +
/// variations) is unique and free of deterministic-command prefixes so the
/// transcript classifier can always distinguish a probe and no probe is
/// intercepted by the pre-filter instead of the intent being tested.
#[test]
fn route_probe_seeds_stay_synced_with_config() {
    let config = load_config();
    let view = config.routes_view();
    let seeded: Vec<&str> = ROUTE_PROBE_SEEDS.iter().map(|(n, _)| *n).collect();

    for (name, _) in ROUTE_PROBE_SEEDS {
        assert!(
            view.contains_key(*name),
            "probe seed '{name}' is not a declared route in coral-router.json (rename or drop the seed)"
        );
    }

    for (route, rref) in &view {
        assert!(
            seeded.contains(&route.as_str()) || !rref.description.is_empty(),
            "route '{route}' has neither a probe seed nor a description — its probe would be a weak name fallback"
        );
    }

    let mut seen: HashMap<String, String> = HashMap::new();
    for (route, rref) in &view {
        let mut probes = vec![probe_for_route(route, rref)];
        probes.extend(varied_probes_for_route(route));
        for probe in probes {
            assert!(!probe.is_empty(), "route '{route}' produced an empty probe");
            assert!(
                !probe.starts_with('/') && !probe.starts_with('.') && !probe.starts_with(','),
                "route '{route}' probe '{probe}' starts like a deterministic command"
            );
            let previous = seen.insert(probe.clone(), route.clone());
            assert!(
                previous.is_none(),
                "routes '{previous:?}' and '{route}' would collide on probe '{probe}' (the transcript classifier could not tell them apart)"
            );
        }
    }
}

/// Every route (intent) in the config dispatches through its configured
/// `model_group`, across every probe (primary + variations): the router's own
/// route + group validation records zero mismatches, and each probe is
/// answered with its canned dispatch response (HTTP 200).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn route_intents_dispatch_to_their_model_groups() {
    let config = load_config();
    let route_count = config.routes_view().len();
    let entries = transcripts_from_config(&config);
    let (server, mock) = spawn_config_mock_server(config, entries.clone()).await;

    let mut probed = 0;
    for entry in &entries {
        let route = entry.expected_route.as_deref().expect("derived route");
        let body = json!({
            "model": route,
            "messages": [{"role": "user", "content": entry.user_message}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("request for route '{route}' failed: {e}"));
        assert_eq!(
            response.status(),
            200,
            "route '{route}' (probe '{:?}') must answer 200",
            entry.user_message
        );
        let value: Value = response
            .json()
            .await
            .unwrap_or_else(|e| panic!("route '{route}' response must be valid JSON: {e}"));
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        let expected = format!("mock {route} answer");
        assert!(
            content.contains(&expected),
            "route '{route}' must return its dispatched answer, got: {content:?}"
        );
        probed += 1;
    }
    assert!(
        probed >= route_count,
        "expected at least one probe per declared route, probed {probed}"
    );

    // The router's own route/group validation (recorded on every mock
    // dispatch) must be clean: any mismatch means an intent did not reach the
    // model_group its route maps to in the config.
    let failures = mock.take_failures();
    assert!(
        failures.is_empty(),
        "intent -> model_group mismatches:\n  {}",
        failures.join("\n  ")
    );
}

/// Every declared model answers when requested directly by its config key
/// (bypassing the route table): `target_name` resolves to the model key and
/// each request returns its canned answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_declared_model_answers_directly() {
    let config = load_config();
    assert!(
        !config.models.is_empty(),
        "coral-router.json declares no models — nothing to test"
    );
    let entries: Vec<MockTranscriptEntry> = config
        .models
        .keys()
        .map(|key| {
            let user_message = format!("direct model probe: {key}");
            MockTranscriptEntry {
                user_message: user_message.clone(),
                // Verdict picks the key itself when it names a route (e.g.
                // `code` is both); pure model keys bypass the classifier on
                // the direct path, so the verdict is never consulted for them.
                classifier_response: route_classifier_response(key),
                expected_route: Some(key.clone()),
                expect_model_group: None,
                dispatch_response: Some(format!("mock answer from {key}")),
                rejected: false,
                reject_reason_contains: None,
            }
        })
        .collect();
    let (server, mock) = spawn_config_mock_server(config, entries.clone()).await;

    for entry in &entries {
        let key = entry.expected_route.as_deref().expect("model key");
        let body = json!({
            "model": key,
            "messages": [{"role": "user", "content": entry.user_message}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("request for model '{key}' failed: {e}"));
        assert_eq!(
            response.status(),
            200,
            "direct model '{key}' must answer 200"
        );
        let value: Value = response
            .json()
            .await
            .unwrap_or_else(|e| panic!("model '{key}' response must be valid JSON: {e}"));
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        let expected = format!("mock answer from {key}");
        assert!(
            content.contains(&expected),
            "model '{key}' must return its direct answer, got: {content:?}"
        );
    }

    let failures = mock.take_failures();
    assert!(
        failures.is_empty(),
        "direct model mismatches:\n  {}",
        failures.join("\n  ")
    );
}

/// `always_route: true` terminals dispatch through their group end-to-end:
/// every `always_route` route declared in `env/coral-router.json` answers
/// with its dispatched response (it reads the live config at runtime, so it
/// cannot drift and protects the config's flags from silent removal).
/// A verdict that picks no child rejects instead of answering directly —
/// tree mode always dispatches or rejects; there is no direct-answer path.
/// The *mechanism* and *prompt-rule* internals (respond→route override +
/// "ALWAYS dispatch" in the system prompt) are owned by the unit tier in
/// `stage_tests.rs::always_route_forces_dispatch_over_classifier_respond`
/// (see ROADMAP M2.4); flag carriage through the tree is locked by
/// `config_root.rs::shipped_config_routes_view_key_set`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn always_route_routes_force_dispatch_over_classifier_respond() {
    let config = load_config();
    let view = config.routes_view();
    let always: Vec<(String, String, String)> = view
        .iter()
        .filter(|(_, rref)| rref.always_route)
        .map(|(route, rref)| (route.clone(), rref.group.clone(), probe_for_route(route, rref)))
        .collect();
    assert!(
        !always.is_empty(),
        "config declares no always_route routes — nothing to test"
    );

    // A verdict that picks no child: tree mode rejects (no direct answer).
    let no_pick_response = json!({
        "route": null,
        "coherence": 0.95,
        "safety": 0.9,
        "complexity": 5,
        "reason": "mock probe picks no child",
    })
    .to_string();

    let mut entries: Vec<MockTranscriptEntry> = always
        .iter()
        .map(|(route, group, probe)| MockTranscriptEntry {
            user_message: probe.clone(),
            classifier_response: route_classifier_response(route),
            expected_route: Some(route.clone()),
            expect_model_group: Some(group.clone()),
            dispatch_response: Some("DISPATCHED-ANSWER".into()),
            rejected: false,
            reject_reason_contains: None,
        })
        .collect();

    // Control: a non-always_route route also dispatches when picked — and a
    // no-pick verdict on it rejects rather than answering directly.
    let control = view
        .iter()
        .find(|(_, r)| !r.always_route)
        .map(|(route, rref)| (route.clone(), rref.group.clone(), probe_for_route(route, rref)));
    let is_control = control.is_some();
    if let Some((route, group, probe)) = &control {
        entries.push(MockTranscriptEntry {
            user_message: probe.clone(),
            classifier_response: route_classifier_response(route),
            expected_route: Some(route.clone()),
            expect_model_group: Some(group.clone()),
            dispatch_response: Some("DISPATCHED-ANSWER".into()),
            rejected: false,
            reject_reason_contains: None,
        });
        entries.push(MockTranscriptEntry {
            user_message: format!("{probe} [no-pick]"),
            classifier_response: no_pick_response.clone(),
            expected_route: Some(route.clone()),
            expect_model_group: None,
            dispatch_response: None,
            rejected: true,
            reject_reason_contains: Some("no valid child".into()),
        });
    }

    let (server, mock) = spawn_config_mock_server(config, entries.clone()).await;

    for entry in &entries {
        let route = entry.expected_route.as_deref().expect("declared route");
        let body = json!({
            "model": route,
            "messages": [{"role": "user", "content": entry.user_message}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("request for '{route}' failed: {e}"));
        assert_eq!(response.status(), 200, "route '{route}' must answer");
        let text = response
            .text()
            .await
            .unwrap_or_else(|e| panic!("route '{route}' body: {e}"));
        let is_no_pick = is_control && entry.user_message.ends_with(" [no-pick]");
        if is_no_pick {
            assert!(
                text.contains("no valid child"),
                "no-pick verdict on '{route}' must reject, got: {text}"
            );
        } else {
            assert!(
                text.contains("DISPATCHED-ANSWER"),
                "route '{route}' must dispatch, got: {text}"
            );
        }
    }

    let failures = mock.take_failures();
    assert!(
        failures.is_empty(),
        "always_route dispatch mismatches:\n  {}",
        failures.join("\n  ")
    );
}

/// Deterministic pre-filter commands (`/help`, `/stats`, `/checkpoint <name>`)
/// and unknown commands are intercepted before the classifier.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deterministic_commands_dispatch() {
    let config = load_config();
    let (server, _mock) = spawn_config_mock_server(config.clone(), vec![]).await;

    let cases: &[(&str, &str)] = &[
        ("/help", "help"),
        ("/stats", "stats"),
        ("/checkpoint snap1", "checkpoint"),
        ("/nonexistent", "unknown"),
    ];
    for (command, fragment) in cases {
        let body = json!({
            "model": config.default_route,
            "messages": [{"role": "user", "content": command}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("command '{command}' request failed: {e}"));
        assert_eq!(response.status(), 200, "command '{command}' must answer 200");
        let text = response
            .text()
            .await
            .unwrap_or_else(|e| panic!("command '{command}' body: {e}"));
        assert!(
            text.contains(fragment),
            "command '{command}' must echo '{fragment}', got: {text}"
        );
    }
}

/// PII is intercepted: SSN/email score below the tree safety/coherence
/// thresholds (the engine rejects with its threshold reason) and API keys
/// hit the deterministic pre-filter's hard reject. Each body carries the
/// rejection reason, and the mock's own rejection validation records zero
/// mismatches.
///
/// Tree-mode note: the flat classifier used to pass the mock's own reason
/// text through; the tree engine owns its reject text (`rejected: safety…
/// below threshold…`), so the per-entry fragments name the engine's reason,
/// not the PII pattern. The PII specificity lives in the mock stand-in: it
/// scores these inputs unsafe because they match PII.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pii_requests_are_blocked() {
    let config = load_config();
    let entries = vec![
        MockTranscriptEntry {
            user_message: "My SSN is 123-45-6789".into(),
            classifier_response: json!({"route": null, "coherence": 1.0, "safety": 0.05, "complexity": 5, "reason": "SSN scores unsafe"}).to_string(),
            expected_route: None,
            expect_model_group: None,
            dispatch_response: None,
            rejected: true,
            reject_reason_contains: Some("safety".into()),
        },
        MockTranscriptEntry {
            user_message: "Email me@test.com please".into(),
            classifier_response: json!({"route": null, "coherence": 0.05, "safety": 1.0, "complexity": 5, "reason": "email scores incoherent"}).to_string(),
            expected_route: None,
            expect_model_group: None,
            dispatch_response: None,
            rejected: true,
            reject_reason_contains: Some("coherence".into()),
        },
        MockTranscriptEntry {
            user_message: "api_key=sk-abcdefghijklmnop123456".into(),
            classifier_response: json!({"route": null, "coherence": 1.0, "safety": 1.0, "complexity": 5, "reason": "api key detected"}).to_string(),
            expected_route: None,
            expect_model_group: None,
            dispatch_response: None,
            rejected: true,
            reject_reason_contains: Some("api_key".into()),
        },
    ];
    let (server, mock) = spawn_config_mock_server(config.clone(), entries.clone()).await;

    for entry in &entries {
        let body = json!({
            "model": config.default_route.clone(),
            "messages": [{"role": "user", "content": entry.user_message}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("PII probe '{:?}' failed: {e}", entry.user_message));
        assert_eq!(response.status(), 200, "PII probe '{:?}' must answer", entry.user_message);
        let text = response
            .text()
            .await
            .unwrap_or_else(|e| panic!("PII probe '{:?}' body: {e}", entry.user_message));
        let fragment = entry.reject_reason_contains.as_deref().expect("reason fragment");
        assert!(
            text.contains(fragment),
            "PII probe '{:?}' must carry '{fragment}', got: {text}",
            entry.user_message
        );
    }

    let failures = mock.take_failures();
    assert!(
        failures.is_empty(),
        "PII rejection validation mismatches:\n  {}",
        failures.join("\n  ")
    );
}

/// `GET /health` reports `{"status":"ok"}` and `GET /stats` reports the request
/// counters.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_and_stats_endpoints_report_ok() {
    let config = load_config();
    let (server, _mock) = spawn_config_mock_server(config, vec![]).await;

    let health = get(&server.base_url(), "/health", 10_000)
        .await
        .expect("health request must complete");
    assert_eq!(health.status(), 200, "health must be 200");
    let health_json: Value = health.json().await.expect("health body must be JSON");
    assert_eq!(health_json["status"], "ok");

    let stats = get(&server.base_url(), "/stats", 10_000)
        .await
        .expect("stats request must complete");
    assert_eq!(stats.status(), 200, "stats must be 200");
    let stats_json: Value = stats.json().await.expect("stats body must be JSON");
    assert!(stats_json.get("requests").is_some(), "stats must report requests");
    assert!(stats_json.get("errors").is_some(), "stats must report errors");
}

/// Unknown paths return 404.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_path_returns_404() {
    let config = load_config();
    let (server, _mock) = spawn_config_mock_server(config, vec![]).await;

    let response = get(&server.base_url(), "/nonexistent", 10_000)
        .await
        .expect("404 probe must complete");
    assert_eq!(response.status(), 404, "unknown path must 404");
}

/// `env/mock-transcripts.json` (the `--mock` binary's fixture) stays synced
/// with the config: every `expected_route` and every classifier verdict
/// `route` must be a declared route or model, and where both are present
/// they must agree.
#[test]
fn mock_transcript_fixture_stays_synced_with_config() {
    use crate::stages::tree::verdict::TreeClassifierVerdict;

    let config = load_config();
    let view = config.routes_view();
    let entries = load_transcript_file(mock_transcripts_path())
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", mock_transcripts_path().display()));
    assert!(
        !entries.is_empty(),
        "mock-transcripts.json is empty — the --mock binary would have no canned answers"
    );
    for entry in &entries {
        let resolved = |name: &str| view.contains_key(name) || config.models.contains_key(name);
        if let Some(expected_route) = &entry.expected_route {
            assert!(
                resolved(expected_route),
                "mock-transcripts.json: expected_route '{expected_route}' (for '{:?}') is neither a declared route nor a model in coral-router.json",
                entry.user_message
            );
        }
        let output: TreeClassifierVerdict = serde_json::from_str(&entry.classifier_response).unwrap_or_else(
            |e| panic!("mock-transcripts.json: unparseable classifier_response for '{:?}': {e}", entry.user_message),
        );
        if let Some(target) = &output.route {
            assert!(
                resolved(target),
                "mock-transcripts.json: classifier route '{target}' (for '{:?}') is neither a declared route nor a model",
                entry.user_message
            );
        }
        if let (Some(expected_route), Some(target)) = (&entry.expected_route, &output.route) {
            assert_eq!(
                expected_route, target,
                "mock-transcripts.json: expected_route and classifier route disagree for '{:?}'",
                entry.user_message
            );
        }
        if let Some(expected_group) = &entry.expect_model_group {
            let declared_group = entry
                .expected_route
                .as_ref()
                .and_then(|r| view.get(r))
                .map(|r| r.group.as_str());
            assert_eq!(
                declared_group,
                Some(expected_group.as_str()),
                "mock-transcripts.json: expect_model_group '{expected_group}' for '{:?}' must name the model_groups the expected_route maps to",
                entry.user_message
            );
        }
    }
}

/// The `--mock` fixture serves its declared answers end-to-end: every
/// non-rejected entry returns its canned `dispatch_response` and every
/// rejected entry carries its `reject_reason_contains` fragment. The router's
/// own validation (route + rejection reasons) records zero mismatches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mock_transcript_entries_serve_their_expected_answers() {
    let config = load_config();
    let default_route = config.default_route.clone();
    let entries = load_transcript_file(mock_transcripts_path())
        .expect("load mock-transcripts.json (validated by mock_transcript_fixture_stays_synced_with_config)");
    let (server, mock) = spawn_config_mock_server(config, entries.clone()).await;

    for entry in &entries {
        let body = json!({
            "model": entry.expected_route.clone().unwrap_or_else(|| default_route.clone()),
            "messages": [{"role": "user", "content": entry.user_message}],
        });
        let response = post_chat(&server.base_url(), body, 15_000)
            .await
            .unwrap_or_else(|e| panic!("fixture probe '{:?}' failed: {e}", entry.user_message));
        assert_eq!(
            response.status(),
            200,
            "fixture probe '{:?}' must answer",
            entry.user_message
        );
        let text = response
            .text()
            .await
            .unwrap_or_else(|e| panic!("fixture probe '{:?}' body: {e}", entry.user_message));

        if entry.rejected {
            let fragment = entry.reject_reason_contains.as_deref().unwrap_or("ERROR:");
            assert!(
                text.contains(fragment),
                "rejected probe '{:?}' must carry '{fragment}', got: {text}",
                entry.user_message
            );
            assert!(
                text.contains("ERROR:"),
                "rejected probe '{:?}' must surface an ERROR body, got: {text}",
                entry.user_message
            );
        } else {
            let expected = entry.dispatch_response.as_deref().expect("dispatch response");
            assert!(
                text.contains(expected),
                "routed probe '{:?}' must return its dispatch response, got: {text}",
                entry.user_message
            );
        }
    }

    let failures = mock.take_failures();
    assert!(
        failures.is_empty(),
        "fixture validation mismatches (stale expected_route / reject reason):\n  {}",
        failures.join("\n  ")
    );
}
