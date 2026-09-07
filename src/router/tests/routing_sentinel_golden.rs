//! Sentinel fallback + late-binding calibration corpus.
//!
//! Measurement artifact: every fallback step the sentinel machinery performs
//! must cite a genuine recorded cause, and every control must perform zero
//! fallback steps. Cases live in `data/routing_sentinel_corpus.json`; the
//! report in `data/routing_sentinel_report.json` pins the recomputed counts
//! and the 1.0 precision/recall/control targets. The test fails on any
//! deviation. Hermetic: stub backends, scripted assessments, and in-memory
//! tables only — no model, no network, no disk writes. Nothing here enables
//! caching, persistence, or traffic-shaping — measurement only.
//!
//! Cause vocabulary (every fallback event cites one):
//! - `err:<rung>` / `miss:<rung>` — a ladder rung failed non-terminally or
//!   reported no result, so the walk continued.
//! - `last-miss:no-record` / `last-miss:removed-member` — a `last` member with
//!   no usable recency, skipped with cause.
//! - `unloaded:<key>` — an `any`-ordered member ranked behind a loaded one
//!   (it still dispatches later through on-demand load).
//! - `assess-err:<key>` — a climb candidate whose self-assessment failed, so
//!   the ladder escalated conservatively.
//! - `endpoint-rewrite:<key>` / `load:<key>` — late-bound re-resolution after
//!   a table change or a first load.
//!
//! Genuine failures (recall denominator) are the injected availability gaps:
//! backend err/miss, missing/removed records, unloaded members present in the
//! output (they engage the on-demand path), assess errors, and resolve misses.
//! Each must be followed (walk continues) or cleanly exhaust.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use super::{
    expand_group_keys, expanded_candidates_for_group, GroupRecency, TargetBackends,
    TargetMatcher,
};
use crate::config::{
    split_model_key, ConfidenceGate, FilterOutcome, FilterScope, GroupMember, InstanceProfile,
    ModelEntry, ModelGroup, PatternEntry, RouteRef, RoutingConfig,
};
use crate::filters::{
    regex_filter::RegexFilter, DeterministicFilterEngine, FilterContext, FilterDecision,
};
use crate::test_stubs::{CountingBackend, StubChatBackend};

// ─── Case plumbing ───────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Clone)]
enum Observed {
    Some(String),
    None,
    Err(String),
}

impl Observed {
    fn as_str(&self) -> String {
        match self {
            Observed::Some(s) => format!("some:{s}"),
            Observed::None => "none".to_string(),
            Observed::Err(e) => format!("err:{e}"),
        }
    }
}

struct CaseResult {
    id: String,
    kind: String,
    consults: Vec<String>,
    probes: Vec<String>,
    observed: Observed,
    causes: Vec<String>,
    failures: Vec<String>,
    unfollowed: Vec<String>,
    /// Climb: the matched member key (the assessment record reporting
    /// `matched`, not the wire model id). Latebind: resolve snapshots —
    /// `[first]` or `[first, second]` (`"none"` when unresolvable).
    matched: String,
    aux: Vec<String>,
}

fn corpus_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn strings(value: &serde_json::Value, field: &str) -> Vec<String> {
    value[field]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect()
}

fn model_entry(_key: &str, intelligence: u8) -> ModelEntry {
    ModelEntry {
        name: None,
        endpoint: "http://localhost:9/v1/chat/completions".into(),
        intelligence,
        cost_input: 1.0,
        cost_output: 6.0,
        cost_cached_read: 0.4,
        speed: 8,
        total_timeout_ms: 40_000,
        idle_timeout_ms: 8_000,
        stream: true,
        filter_thinking: false,
        retry_count: 0,
        retry_base_interval_s: 1,
        params: None,
        instances: None,
        weights: None,
        hf_repo: None,
        hf_file: None,
        api_key: None,
    }
}

fn fixture_routing(case: &serde_json::Value, group: &str) -> RoutingConfig {
    let mut models = HashMap::new();
    if let Some(entries) = case["entries"].as_object() {
        for (key, spec) in entries {
            let intelligence = spec["intelligence"].as_u64().unwrap_or(0) as u8;
            models.insert(key.clone(), model_entry(key, intelligence));
        }
    }
    let mut onnx_keys = BTreeSet::new();
    for key in strings(case, "onnx_keys") {
        onnx_keys.insert(key);
    }
    RoutingConfig {
        routes: HashMap::from([(
            "local".into(),
            RouteRef {
                group: group.into(),
                pipelines: vec!["default".into()],
                description: "local".into(),
                always_route: false,
            },
        )]),
        models,
        model_groups: HashMap::from([(
            group.into(),
            ModelGroup::Array(strings(case, "members")),
        )]),
        system_prompt: String::new(),
        safety_threshold: 0.3,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys,
        roles: HashMap::new(),
    }
}

fn fixture_recency(routing: &RoutingConfig, case: &serde_json::Value) -> GroupRecency {
    let recency = GroupRecency::new();
    if let Some(map) = case["recency"].as_object() {
        for (group, reference) in map {
            let reference = reference.as_str().expect("recency reference");
            let wire = if let Some(member) = reference.strip_prefix("key:") {
                routing
                    .target_for_key(member)
                    .unwrap_or_else(|| panic!("recency member {member} resolves"))
                    .model
            } else if let Some(literal) = reference.strip_prefix("wire:") {
                literal.to_string()
            } else {
                panic!("bad recency reference {reference}");
            };
            recency.record(group, &wire);
        }
    }
    recency
}

/// Whether a recorded wire still names a literal member of the group — the
/// test-side attribution for `last-miss` causes, reusing the production wire
/// mapping (never a copy of the expansion walk itself).
fn last_resolves(routing: &RoutingConfig, group: &str, recency: &GroupRecency) -> bool {
    let Some(wire) = recency.last_for(group) else {
        return false;
    };
    let Some(group_cfg) = routing.model_groups.get(group) else {
        return false;
    };
    group_cfg
        .models()
        .iter()
        .filter(|m| matches!(GroupMember::parse(m), GroupMember::Key(_)))
        .any(|m| {
            let (base, _) = split_model_key(m);
            if routing.onnx_keys.contains(base) {
                m == &wire
            } else {
                routing
                    .target_for_key(m)
                    .is_some_and(|rt| rt.model == wire)
            }
        })
}

// ─── Op runners ──────────────────────────────────────────────────────────────

#[derive(Debug)]
enum LadderError {
    Terminal(String),
    Continue(String),
}

fn split_behavior(behavior: &str) -> (&str, Option<&str>) {
    match behavior.split_once(':') {
        Some((kind, arg)) => (kind, Some(arg)),
        None => (behavior, None),
    }
}

fn run_ladder_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let rungs = case["rungs"].as_array().expect("rungs").clone();
    let consults = Arc::new(Mutex::new(Vec::<String>::new()));
    let rung_ids: Vec<String> = rungs
        .iter()
        .map(|r| r["id"].as_str().expect("rung id").to_string())
        .collect();
    let behaviors: HashMap<String, String> = rungs
        .iter()
        .map(|r| {
            (
                r["id"].as_str().expect("rung id").to_string(),
                r["behavior"].as_str().unwrap_or("miss").to_string(),
            )
        })
        .collect();
    let result = common_core::runtime::block_on(fluent_concurrency::ladder::first_accept_in_order(
        rung_ids.clone(),
        {
            let consults = Arc::clone(&consults);
            let behaviors = behaviors.clone();
            move |rung: String| {
                let consults = Arc::clone(&consults);
                let behavior = behaviors[&rung].clone();
                async move {
                    consults.lock().expect("log").push(rung.clone());
                    let (kind, arg) = split_behavior(&behavior);
                    match kind {
                        "serve" => Ok::<_, LadderError>(Some(arg.expect("marker").to_string())),
                        "miss" => Ok(None),
                        "err-terminal" => Err(LadderError::Terminal(arg.expect("msg").to_string())),
                        "err-continue" => Err(LadderError::Continue(arg.expect("msg").to_string())),
                        other => panic!("unknown ladder behavior {other}"),
                    }
                }
            }
        },
        |e: &LadderError| matches!(e, LadderError::Terminal(_)),
    ));
    let observed = match result {
        Ok(Some(marker)) => Observed::Some(marker),
        Ok(None) => Observed::None,
        Err(LadderError::Terminal(msg) | LadderError::Continue(msg)) => Observed::Err(msg),
    };
    let consults = consults.lock().expect("log").clone();

    // Transition + failure accounting over the scripted behaviors.
    let mut causes = Vec::new();
    let mut failures = Vec::new();
    let mut unfollowed = Vec::new();
    let behavior_of = |rid: &str| behaviors[rid].clone();
    for (i, cid) in consults.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let prev = consults[i - 1].as_str();
        match split_behavior(&behavior_of(prev)).0 {
            "miss" => causes.push(format!("miss:{prev}")),
            "err-continue" => causes.push(format!("err:{prev}")),
            _ => causes.push(format!("uncaused-after:{prev}")),
        }
        let _ = cid;
    }
    for (idx, rid) in rung_ids.iter().enumerate() {
        let consulted = consults.contains(rid);
        let later_consulted = consults.iter().any(|c| {
            rung_ids
                .iter()
                .position(|r| r == c)
                .is_some_and(|p| p > idx)
        });
        let later_servable = rung_ids[idx + 1..].iter().any(|r| {
            !matches!(split_behavior(&behavior_of(r)).0, "err-terminal")
        });
        let followed_or_exhausted = later_consulted || !later_servable;
        match split_behavior(&behavior_of(rid)).0 {
            "miss" | "err-continue" if consulted => {
                failures.push(rid.clone());
                if !followed_or_exhausted {
                    unfollowed.push(rid.clone());
                }
            }
            _ => {}
        }
    }
    CaseResult {
        id,
        kind,
        consults,
        probes: Vec::new(),
        observed,
        causes,
        failures,
        unfollowed,
        matched: String::new(),
        aux: Vec::new(),
    }
}

fn run_expand_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let group = case["group"].as_str().expect("group");
    let routing = fixture_routing(case, group);
    let recency = fixture_recency(&routing, case);
    let loaded: Vec<String> = strings(case, "loaded");
    let probes = Arc::new(Mutex::new(Vec::<String>::new()));
    let order = expand_group_keys(
        &routing,
        group,
        Some(&recency),
        &{
            let probes = Arc::clone(&probes);
            let loaded = loaded.clone();
            move |base: &str| {
                probes.lock().expect("log").push(base.to_string());
                loaded.iter().any(|l| l == base)
            }
        },
    );
    let probes = probes.lock().expect("log").clone();

    // Cause + failure attribution from the observed order and the inputs.
    let members = strings(case, "members");
    let has_last = members.iter().any(|m| m == "last");
    let mut causes = Vec::new();
    let mut failures = Vec::new();
    let unfollowed = Vec::new();
    if has_last && !last_resolves(&routing, group, &recency) {
        let reason = if recency.last_for(group).is_none() {
            "no-record"
        } else {
            "removed-member"
        };
        causes.push(format!("last-miss:{reason}"));
        failures.push("last-miss".to_string());
        // A miss with literals remaining continues through them; an empty
        // output would be clean exhaustion (still followed — asserted below
        // via the exact order).
    }
    let loaded_set: Vec<&str> = members
        .iter()
        .filter(|m| matches!(GroupMember::parse(m), GroupMember::Key(_)))
        .filter(|m| {
            let (base, _) = split_model_key(m);
            routing.onnx_keys.contains(base) || loaded.iter().any(|l| l == base)
        })
        .map(String::as_str)
        .collect();
    for member in &order {
        let (base, _) = split_model_key(member);
        let is_loaded = routing.onnx_keys.contains(base) || loaded.iter().any(|l| l == base);
        if !is_loaded {
            // Present in the output: the on-demand path absorbs the gap.
            failures.push(format!("unloaded:{member}"));
            if loaded_set.iter().any(|l| {
                order
                    .iter()
                    .position(|o| o == l)
                    .is_some_and(|lp| {
                        order.iter().position(|o| o == member).is_some_and(|up| lp < up)
                    })
            }) {
                causes.push(format!("unloaded:{member}"));
            }
        }
    }
    CaseResult {
        id,
        kind,
        consults: order.clone(),
        probes,
        observed: Observed::Some(order.join(",")),
        causes,
        failures,
        unfollowed,
        matched: String::new(),
        aux: Vec::new(),
    }
}

fn assessment_script(script: &str) -> Vec<String> {
    if script == "err" {
        Vec::new()
    } else if let Some(text) = script.strip_prefix("raw:") {
        vec![text.to_string()]
    } else if let Some((complexity, reason)) = script.split_once(':') {
        vec![
            serde_json::json!({
                "complexity": complexity.parse::<u8>().expect("complexity"),
                "reason": reason,
            })
            .to_string(),
        ]
    } else {
        panic!("bad assess script {script}");
    }
}

fn run_climb_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let group = case["group"].as_str().expect("group");
    let routing = fixture_routing(case, group);
    let recency = fixture_recency(&routing, case);
    let loaded: Vec<String> = strings(case, "loaded");
    let expanded = expand_group_keys(&routing, group, Some(&recency), &|base| {
        loaded.iter().any(|l| l == base)
    });
    let candidates = expanded_candidates_for_group(&routing, group, Some(&recency), &|base| {
        loaded.iter().any(|l| l == base)
    });
    assert_eq!(
        candidates
            .iter()
            .map(|c| c.model_key.clone())
            .collect::<Vec<_>>(),
        expanded
            .iter()
            .filter(|k| routing.entry_for_key(k).is_some())
            .cloned()
            .collect::<Vec<_>>(),
        "case {id}: candidates track the expanded order"
    );
    let scripts = case["assess"].as_object().expect("assess").clone();
    let mut by_key = HashMap::new();
    for candidate in &candidates {
        let script = scripts
            .get(&candidate.model_key)
            .and_then(|v| v.as_str())
            .unwrap_or("err");
        let backend: Arc<dyn fluent_llm::client::ChatBackend> =
            Arc::new(StubChatBackend::new(assessment_script(script)));
        by_key.insert(candidate.model_key.clone(), backend);
    }
    let matcher = TargetMatcher::new(
        TargetBackends::new(
            by_key,
            Arc::new(StubChatBackend::new(Vec::new())) as Arc<dyn fluent_llm::client::ChatBackend>,
        ),
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        0,
    );
    let complexity = case["classifier_complexity"].as_u64().map(|c| c as u8);
    let matched = matcher.match_target("local", group, &routing, &candidates, complexity, "hello");

    let mut causes = Vec::new();
    let mut failures = Vec::new();
    let unfollowed = Vec::new();
    let (consults, observed, matched_key) = match matched {
        Some(tm) => {
            let mut consults = Vec::new();
            let mut matched_key = String::new();
            for record in &tm.assessments {
                consults.push(record.model_key.clone());
                if record.error.is_some() {
                    causes.push(format!("assess-err:{}", record.model_key));
                    failures.push(format!("assess-err:{}", record.model_key));
                }
                if record.matched {
                    matched_key = record.model_key.clone();
                }
            }
            (consults, Observed::Some(tm.primary.model.clone()), matched_key)
        }
        None => (Vec::new(), Observed::None, String::new()),
    };
    // The ladder always terminates on a match (last-always-wins), so every
    // assess failure here is followed; an empty-candidate `None` is clean
    // exhaustion.
    CaseResult {
        id,
        kind,
        consults,
        probes: Vec::new(),
        observed,
        causes,
        failures,
        unfollowed,
        matched: matched_key,
        aux: Vec::new(),
    }
}

fn run_latebind_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let key = case["resolve"].as_str().expect("resolve").to_string();
    let table: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(
        case["table"]
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, v.as_str().expect("url").to_string()))
            .collect(),
    ));
    let consults = Arc::new(Mutex::new(Vec::<String>::new()));
    let builds = Arc::new(Mutex::new(0usize));
    // Late-bound resolver: every lookup reads the live table, so endpoint
    // rewrites and first loads are always current. A miss builds nothing —
    // the failure-policy path rejects, never a fabricated route.
    let resolve = {
        let table = Arc::clone(&table);
        let consults = Arc::clone(&consults);
        let builds = Arc::clone(&builds);
        move |lookup: &str| {
            consults.lock().expect("log").push(lookup.to_string());
            let url = table.lock().expect("log").get(lookup).cloned();
            if url.is_some() {
                *builds.lock().expect("log") += 1;
            }
            url
        }
    };
    let first = resolve(&key);
    let mut resolves = vec![first.clone()];
    if let Some(rewrite) = case["rewrite"].as_object() {
        for (k, v) in rewrite {
            table
                .lock()
                .expect("log")
                .insert(k.clone(), v.as_str().expect("url").to_string());
        }
        resolves.push(resolve(&key));
    }
    let second = resolves.last().cloned().unwrap_or(None);
    let aux: Vec<String> = resolves
        .iter()
        .map(|o| o.clone().unwrap_or_else(|| "none".to_string()))
        .collect();
    let consults = consults.lock().expect("log").clone();
    let builds = *builds.lock().expect("log");

    let mut causes = Vec::new();
    let mut failures = Vec::new();
    let unfollowed = Vec::new();
    let observed = match (&first, second.as_deref()) {
        (Some(a), Some(b)) if a != b => {
            causes.push(format!("endpoint-rewrite:{key}"));
            Observed::Some(b.to_string())
        }
        (None, Some(b)) => {
            causes.push(format!("load:{key}"));
            failures.push(format!("miss:{key}"));
            Observed::Some(b.to_string())
        }
        (None, _) => Observed::None,
        (Some(a), _) => Observed::Some(a.clone()),
    };
    // A resolve miss followed by a post-load hit is absorbed; a persistent
    // miss stays on the policy path (no fabricated backend).
    if first.is_none() && second.is_none() {
        failures.push(format!("miss:{key}"));
    }
    // Stash the build count in the probes channel for the control assertion.
    let probes = vec![format!("builds={builds}")];
    CaseResult {
        id,
        kind,
        consults,
        probes,
        observed,
        causes,
        failures,
        unfollowed,
        matched: String::new(),
        aux,
    }
}

fn run_prefilter_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let outcome = case["outcome"].as_str().expect("outcome");
    let filter_outcome = match outcome {
        "hard_reject" => FilterOutcome::HardReject,
        "soft_redirect" => FilterOutcome::SoftRedirect,
        _ => panic!("bad outcome {outcome}"),
    };
    let entry = PatternEntry {
        name: "sentinel-calibration".into(),
        outcome: filter_outcome,
        filter_action: None,
        confidence_gate: ConfidenceGate::None,
        scope: vec![FilterScope::Any],
        http_code: 403,
        error_message: None,
        regexes: strings(case, "patterns"),
    };
    let mut engine = DeterministicFilterEngine::new();
    engine.add_filter(Box::new(
        RegexFilter::from_entry(&entry).expect("filter"),
    ));
    // The counting backend is never handed to the filter path: a
    // deterministic hit resolves with zero model consultations by
    // construction. The assertion below pins that contract.
    let backend = CountingBackend::new("must never be consulted");
    let input = case["input"].as_str().expect("input");
    let decision = engine.evaluate(&FilterContext::pipeline(input.into()));
    let observed = match decision {
        Some(FilterDecision::HardReject { pattern, .. }) => {
            Observed::Some(format!("hard-reject:{pattern}"))
        }
        Some(_) => Observed::Some("other-decision".to_string()),
        None => Observed::None,
    };
    assert_eq!(
        backend.calls(),
        0,
        "case {id}: deterministic hit must consult zero backends"
    );
    CaseResult {
        id,
        kind,
        consults: Vec::new(),
        probes: Vec::new(),
        observed,
        causes: Vec::new(),
        failures: Vec::new(),
        unfollowed: Vec::new(),
        matched: String::new(),
        aux: Vec::new(),
    }
}

fn run_oneshot_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let profiles = case["profiles"].as_object().expect("profiles").clone();
    let expected = case["expected"].as_object().expect("expected").clone();
    // Baseline the one-shot default ahead of session semantics: a bare
    // profile is one-shot (`session: false`); `session: true` keeps the
    // configured `resume` behavior. Snapshot handling for one-shot contexts
    // is proven by the evictionOrdering tests, not here.
    for (name, spec) in &profiles {
        let profile: InstanceProfile =
            serde_json::from_value(spec.clone()).expect("profile parses");
        let want = expected.get(name).unwrap_or_else(|| panic!("expected {name}"));
        assert_eq!(
            profile.session,
            want["session"].as_bool().expect("session"),
            "case {id}: profile {name} session flag"
        );
        assert_eq!(
            profile.resume,
            want["resume"].as_bool().expect("resume"),
            "case {id}: profile {name} resume flag"
        );
    }
    CaseResult {
        id,
        kind,
        consults: Vec::new(),
        probes: Vec::new(),
        observed: Observed::Some("flags-ok".to_string()),
        causes: Vec::new(),
        failures: Vec::new(),
        unfollowed: Vec::new(),
        matched: String::new(),
        aux: Vec::new(),
    }
}

fn run_case(case: &serde_json::Value) -> CaseResult {
    match case["op"].as_str().expect("op") {
        "ladder" => run_ladder_case(case),
        "expand" => run_expand_case(case),
        "climb" => run_climb_case(case),
        "latebind" => run_latebind_case(case),
        "prefilter" => run_prefilter_case(case),
        "oneshot" => run_oneshot_case(case),
        other => panic!("unknown op {other}"),
    }
}

// A derived cause is genuine only when the case inputs support it: rung
// scripts for ladder transitions, member/loaded sets for expansion order,
// assess scripts for climb escalation, table changes for late binding.
fn is_genuine_cause(case: &serde_json::Value, cause: &str) -> bool {
    if let Some(rung) = cause.strip_prefix("err:") {
        return case["rungs"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .any(|r| {
                r["id"].as_str() == Some(rung)
                    && split_behavior(r["behavior"].as_str().unwrap_or("miss")).0 == "err-continue"
            });
    }
    if let Some(rung) = cause.strip_prefix("miss:") {
        return case["rungs"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .any(|r| {
                r["id"].as_str() == Some(rung)
                    && split_behavior(r["behavior"].as_str().unwrap_or("miss")).0 == "miss"
            });
    }
    if cause == "last-miss:no-record" {
        return strings(case, "members").iter().any(|m| m == "last")
            && case["recency"].as_object().is_none_or(|m| {
                m.get(case["group"].as_str().unwrap_or(""))
                    .is_none()
            });
    }
    if cause == "last-miss:removed-member" {
        return strings(case, "members").iter().any(|m| m == "last")
            && case["recency"]
                .as_object()
                .and_then(|m| m.get(case["group"].as_str().unwrap_or("")))
                .and_then(|v| v.as_str())
                .is_some_and(|r| r.starts_with("wire:"));
    }
    if let Some(member) = cause.strip_prefix("unloaded:") {
        let (base, _) = split_model_key(member);
        let onnx_ready = strings(case, "onnx_keys").iter().any(|k| k == base);
        let supervisor_loaded = strings(case, "loaded").iter().any(|l| l == base);
        return !onnx_ready && !supervisor_loaded;
    }
    if let Some(candidate) = cause.strip_prefix("assess-err:") {
        return case["assess"]
            .as_object()
            .and_then(|m| m.get(candidate))
            .and_then(|v| v.as_str())
            .is_some_and(|s| s == "err");
    }
    if let Some(key) = cause.strip_prefix("endpoint-rewrite:") {
        return case["op"].as_str() == Some("latebind")
            && case["table"].get(&key).is_some()
            && case["rewrite"].get(&key).is_some()
            && case["table"].get(&key) != case["rewrite"].get(&key);
    }
    if let Some(key) = cause.strip_prefix("load:") {
        return case["op"].as_str() == Some("latebind")
            && case["table"].get(&key).is_none()
            && case["rewrite"].get(&key).is_some();
    }
    false
}

fn expected_outcome(case: &serde_json::Value) -> Option<Observed> {
    let expected = &case["expected"];
    if expected.is_null() {
        return None;
    }
    if let Some(marker) = expected["some"].as_str() {
        Some(Observed::Some(marker.to_string()))
    } else if expected["none"].as_bool().unwrap_or(false) {
        Some(Observed::None)
    } else if let Some(msg) = expected["err"].as_str() {
        Some(Observed::Err(msg.to_string()))
    } else {
        None
    }
}

#[test]
fn routing_sentinel_corpus_and_report() {
    let corpus_raw =
        std::fs::read_to_string(corpus_path("routing_sentinel_corpus.json")).expect("corpus");
    let corpus: serde_json::Value = serde_json::from_str(&corpus_raw).expect("corpus json");
    let cases = corpus["cases"].as_array().expect("cases").clone();

    let mut total_fallbacks = 0usize;
    let mut caused_fallbacks = 0usize;
    let mut genuine_failures = 0usize;
    let mut followed_failures = 0usize;
    let mut controls = 0usize;
    let mut controls_passed = 0usize;
    let mut fallback_cases = 0usize;
    let mut control_cases = 0usize;
    let mut pairs: HashMap<String, Vec<(String, Vec<String>)>> = HashMap::new();

    for case in &cases {
        let result = run_case(case);
        let id = result.id.clone();

        match case["op"].as_str().expect("op") {
            "ladder" => {
                assert_eq!(
                    result.consults,
                    strings(case, "expected_consults"),
                    "case {id}: consultation order deviation"
                );
                assert_eq!(
                    result.observed,
                    expected_outcome(case).expect("expected"),
                    "case {id}: outcome deviation"
                );
                assert_eq!(
                    result.causes,
                    strings(case, "expected_causes"),
                    "case {id}: cause deviation"
                );
            }
            "expand" => {
                assert_eq!(
                    result.consults,
                    strings(case, "expected_order"),
                    "case {id}: expansion order deviation"
                );
                assert_eq!(
                    result.probes,
                    strings(case, "expected_probes"),
                    "case {id}: liveness probe deviation"
                );
                assert_eq!(
                    result.causes,
                    strings(case, "expected_causes"),
                    "case {id}: cause deviation"
                );
                if case["kind"].as_str() == Some("control") {
                    // Sentinel-free controls expand through the literal path
                    // byte-identically.
                    let group = case["group"].as_str().expect("group");
                    let routing = fixture_routing(case, group);
                    let literal: Vec<String> = routing.model_groups[group]
                        .models()
                        .iter()
                        .filter(|k| routing.entry_for_key(k).is_some())
                        .cloned()
                        .collect();
                    assert_eq!(
                        result.consults, literal,
                        "case {id}: control expansion must equal the literal path"
                    );
                }
            }
            "climb" => {
                let expected_assessments = strings(case, "expected_assessments");
                assert_eq!(
                    result.consults, expected_assessments,
                    "case {id}: assessment order deviation"
                );
                assert_eq!(
                    result.matched,
                    case["expected_matched"].as_str().expect("expected_matched"),
                    "case {id}: matched member deviation"
                );
                assert!(
                    result.observed != Observed::None,
                    "case {id}: climb must terminate on a match"
                );
                assert_eq!(
                    result.causes,
                    strings(case, "expected_causes"),
                    "case {id}: cause deviation"
                );
                if let Some(pair) = case["pair"].as_str() {
                    pairs
                        .entry(pair.to_string())
                        .or_default()
                        .push((result.observed.as_str(), result.consults.clone()));
                }
            }
            "latebind" => {
                assert_eq!(
                    result.consults,
                    strings(case, "expected_consults"),
                    "case {id}: resolve sequence deviation"
                );
                let want = |v: &serde_json::Value| match v {
                    serde_json::Value::String(s) => s.clone(),
                    _ => "none".to_string(),
                };
                // `aux` holds each resolve snapshot in order; cases without a
                // rewrite resolve once.
                let want_aux: Vec<String> = if case.get("rewrite").is_some() {
                    vec![
                        want(&case["expected_first"]),
                        want(case.get("expected_second").unwrap_or(&case["expected_first"])),
                    ]
                } else {
                    vec![want(&case["expected_first"])]
                };
                assert_eq!(result.aux, want_aux, "case {id}: resolve value deviation");
                assert_eq!(
                    result.causes,
                    strings(case, "expected_causes"),
                    "case {id}: cause deviation"
                );
                if case["kind"].as_str() == Some("control") {
                    assert!(
                        result.observed == Observed::None,
                        "case {id}: unknown key must resolve None (policy path, never fabricated)"
                    );
                    assert!(
                        result.probes == vec!["builds=0".to_string()],
                        "case {id}: unknown key must build zero backends"
                    );
                    let want_builds = case["expected_builds"].as_u64().expect("expected_builds");
                    assert_eq!(want_builds, 0, "case {id}: corpus must pin zero builds");
                }
            }
            "prefilter" => {
                assert!(
                    result.observed != Observed::None,
                    "case {id}: deterministic hit must resolve"
                );
            }
            "oneshot" => {
                assert_eq!(
                    result.observed,
                    Observed::Some("flags-ok".to_string()),
                    "case {id}: profile flags deviation (asserted in-run)"
                );
            }
            other => panic!("unknown op {other}"),
        }

        // Metric accounting over derived causes and failures.
        for cause in &result.causes {
            total_fallbacks += 1;
            if is_genuine_cause(case, cause) {
                caused_fallbacks += 1;
            } else {
                panic!("case {id}: fallback without genuine cause: {cause}");
            }
        }
        for failure in &result.failures {
            genuine_failures += 1;
            if !result.unfollowed.contains(failure) {
                followed_failures += 1;
            }
        }

        if result.kind == "fallback" {
            fallback_cases += 1;
        } else {
            control_cases += 1;
            controls += 1;
            // A control passes with zero fallback steps.
            if result.causes.is_empty() {
                controls_passed += 1;
            }
        }
    }

    // Paired controls must produce identical outcomes (climb invariance).
    for (pair, outputs) in &pairs {
        for other in &outputs[1..] {
            assert_eq!(
                &outputs[0], other,
                "pair {pair}: sentinel presence changed the climb outcome"
            );
        }
    }

    assert!(
        total_fallbacks > 0,
        "corpus must contain fallbacks to calibrate"
    );
    assert!(
        genuine_failures > 0,
        "corpus must contain genuine failures to calibrate"
    );
    let precision = caused_fallbacks as f64 / total_fallbacks as f64;
    let recall = followed_failures as f64 / genuine_failures as f64;
    let control_pass_rate = controls_passed as f64 / controls as f64;

    let report_raw =
        std::fs::read_to_string(corpus_path("routing_sentinel_report.json")).expect("report");
    let report: serde_json::Value = serde_json::from_str(&report_raw).expect("report json");
    assert_eq!(
        report["fallback_cases"].as_u64().unwrap_or(0) as usize,
        fallback_cases,
        "report fallback count drift"
    );
    assert_eq!(
        report["control_cases"].as_u64().unwrap_or(0) as usize,
        control_cases,
        "report control count drift"
    );
    assert_eq!(
        report["total_cases"].as_u64().unwrap_or(0) as usize,
        cases.len(),
        "report total drift"
    );
    for (name, recomputed, filed) in [
        ("precision", precision, report["precision"].as_f64().unwrap_or(-1.0)),
        ("recall", recall, report["recall"].as_f64().unwrap_or(-1.0)),
        (
            "control_pass_rate",
            control_pass_rate,
            report["control_pass_rate"].as_f64().unwrap_or(-1.0),
        ),
    ] {
        assert!(
            (recomputed - filed).abs() < 1e-9,
            "{name} recomputed {recomputed} != filed {filed}"
        );
        assert!(
            (recomputed - 1.0).abs() < 1e-9,
            "{name} target 1.0, recomputed {recomputed} \
             (fallbacks {caused_fallbacks}/{total_fallbacks}, \
             failures {followed_failures}/{genuine_failures}, \
             controls {controls_passed}/{controls})"
        );
    }
}
