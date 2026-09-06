//! Routing-fallback calibration corpus.
//!
//! Measurement milestone: every fallback the registry (or the underlying
//! first-accept ladder) performs must carry a genuine recorded cause — a
//! miss, a failed-readiness skip, or a non-terminal error — and every control
//! must perform zero fallbacks. The corpus (`data/routing_fallback_corpus.json`)
//! scripts stub backends per case and pins the exact consultation sequence,
//! probe sequence, and outcome; the report (`data/routing_fallback_report.json`)
//! pins precision = recall = control-pass = 1.0. The test fails on any
//! deviation. Hermetic: counting stubs only, no model or network. Nothing
//! here enables caching, persistence, or traffic-shaping — measurement only.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─── Counting stub backend ──────────────────────────────────────────────────

/// Shared consultation log: construction calls (`consults`) and readiness
/// probes (`probes`), each in backend-id order of occurrence.
#[derive(Default)]
struct Logs {
    consults: Mutex<Vec<String>>,
    probes: Mutex<Vec<String>>,
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

struct CountingBackend {
    id: &'static str,
    keys: Vec<String>,
    chat_marker: Option<String>,
    embed_name: Option<&'static str>,
    failed: bool,
    logs: Arc<Logs>,
}

struct MarkerChat {
    text: String,
}

impl fluent_llm::client::ChatBackend for MarkerChat {
    fn chat_complete(
        &self,
        _m: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        Ok(self.text.clone())
    }
}

struct MarkerEmbedder {
    name: &'static str,
}

impl fluent_llm::EmbeddingProvider for MarkerEmbedder {
    fn name(&self) -> &'static str {
        self.name
    }
    fn dimensions(&self) -> u32 {
        1
    }
    fn embed(&self, _text: &str) -> Result<Vec<f32>, fluent_llm::EmbeddingError> {
        Ok(vec![1.0])
    }
    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> Result<fluent_llm::BatchEmbedding, fluent_llm::EmbeddingError> {
        Ok(fluent_llm::BatchEmbedding {
            flat: vec![1.0; texts.len()],
            count: texts.len(),
            dims: 1,
        })
    }
}

impl fluent_wvr::FieldAccess for CountingBackend {
    fn set_field(&mut self, name: &str, _v: &str) -> Result<(), fluent_wvr::FieldError> {
        Err(fluent_wvr::FieldError::NotFound(name.into()))
    }
    fn get_field(&self, name: &str) -> Result<String, fluent_wvr::FieldError> {
        Err(fluent_wvr::FieldError::NotFound(name.into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl fluent_wvr::Describable for CountingBackend {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

impl fluent_wvr::WorkUnit for CountingBackend {
    fn name(&self) -> &str {
        self.id
    }
    fn depends(&self) -> &[internment::ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[internment::ArcIntern<str>] {
        &[]
    }
    fn execute(
        &self,
        _ctx: &fluent_wvr::WorkContext,
    ) -> Result<fluent_wvr::WorkOutput, fluent_wvr::WorkError> {
        Ok(fluent_wvr::WorkOutput::ok("counting stub"))
    }
}

fluent_wvr::impl_component!(CountingBackend);

impl fluent_llm::backend::InferenceBackend for CountingBackend {
    fn backend_id(&self) -> &'static str {
        self.id
    }
    fn model_keys(&self) -> Vec<String> {
        self.keys.clone()
    }
    fn weights(&self, _key: &str) -> Option<Arc<dyn fluent_llm::runtime::LlmWeights>> {
        None
    }
    fn chat_backend(
        &self,
        _key: &str,
        _instance: Option<&str>,
    ) -> Option<Arc<dyn fluent_llm::client::ChatBackend>> {
        // Deliberately key-blind: the stub serves or misses purely by script
        // so the corpus can prove the REGISTRY does the key filtering (an
        // unregistered key must yield zero consultations).
        self.logs
            .consults
            .lock()
            .expect("log")
            .push(self.id.to_string());
        self.chat_marker.clone().map(|text| {
            Arc::new(MarkerChat { text }) as Arc<dyn fluent_llm::client::ChatBackend>
        })
    }
    fn embed_provider(
        &self,
        _key: &str,
    ) -> Option<Arc<dyn fluent_llm::EmbeddingProvider>> {
        self.logs
            .consults
            .lock()
            .expect("log")
            .push(self.id.to_string());
        self.embed_name.map(|name| {
            Arc::new(MarkerEmbedder { name }) as Arc<dyn fluent_llm::EmbeddingProvider>
        })
    }
    fn capabilities(&self) -> fluent_llm::backend::BackendCaps {
        fluent_llm::backend::BackendCaps::default()
    }
    fn readiness(&self, _key: &str) -> fluent_llm::backend::Readiness {
        self.logs
            .probes
            .lock()
            .expect("log")
            .push(self.id.to_string());
        if self.failed {
            fluent_llm::backend::Readiness::Failed("calibration failure".into())
        } else {
            fluent_llm::backend::Readiness::Unloaded
        }
    }
}

// ─── Corpus runner ──────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum Observed {
    Some(String),
    None,
    Err(String),
}

struct CaseResult {
    id: String,
    kind: String,
    consults: Vec<String>,
    probes: Vec<String>,
    observed: Observed,
    // Scripted per-rung outcomes in registry order, for metric causes.
    script: Vec<(String, RungScript)>,
}

#[derive(Clone)]
struct RungScript {
    behavior: String,
    failed: bool,
    key_match: bool,
}

fn split_behavior(behavior: &str) -> (&str, Option<&str>) {
    match behavior.split_once(':') {
        Some((kind, arg)) => (kind, Some(arg)),
        None => (behavior, None),
    }
}

fn corpus_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

fn run_case(case: &serde_json::Value) -> CaseResult {
    let id = case["id"].as_str().expect("id").to_string();
    let kind = case["kind"].as_str().expect("kind").to_string();
    let op = case["op"].as_str().expect("op");
    let key = case["key"].as_str().expect("key");
    let instance = case["instance"].as_str().map(str::to_string);
    let rungs = case["rungs"].as_array().expect("rungs").clone();

    if op == "ladder" {
        return run_ladder_case(&id, &kind, &rungs);
    }

    let logs = Arc::new(Logs::default());
    let mut registry = fluent_llm::backend::InferenceRegistry::new();
    // Register in reverse corpus order: the registry routes in backend-id
    // order regardless of insertion, and the corpus pins that.
    for rung in rungs.iter().rev() {
        let rung_id = rung["id"].as_str().expect("rung id");
        let (chat_kind, chat_arg) = split_behavior(rung["behavior"].as_str().unwrap_or("miss"));
        let (embed_kind, embed_arg) = split_behavior(rung["embed"].as_str().unwrap_or("miss"));
        registry.register(Arc::new(CountingBackend {
            id: leak(rung_id),
            keys: rung["keys"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect(),
            chat_marker: if chat_kind == "serve" {
                Some(chat_arg.expect("serve marker").to_string())
            } else {
                None
            },
            embed_name: if embed_kind == "serve" {
                Some(leak(embed_arg.expect("embed name")))
            } else {
                None
            },
            failed: rung["readiness"].as_str().unwrap_or("unloaded") == "failed",
            logs: Arc::clone(&logs),
        }));
    }

    let observed = match op {
        "chat" => match registry.route_chat(key, instance.as_deref()) {
            Some(b) => Observed::Some(b.chat_complete(&[]).expect("stub chat")),
            None => Observed::None,
        },
        "embed" => match registry.route_embed(key) {
            Some(p) => Observed::Some(p.name().to_string()),
            None => Observed::None,
        },
        other => panic!("unknown op {other} in case {id}"),
    };

    let base = key.split_once(':').map_or(key, |(b, _)| b);
    let script = registry
        .backend_ids()
        .into_iter()
        .map(|rid| {
            let rung = rungs
                .iter()
                .find(|r| r["id"].as_str() == Some(rid.as_str()))
                .expect("rung id");
            let keys: Vec<String> = rung["keys"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect();
            (
                rid,
                RungScript {
                    behavior: rung["behavior"].as_str().unwrap_or("miss").to_string(),
                    failed: rung["readiness"].as_str().unwrap_or("unloaded") == "failed",
                    key_match: keys.iter().any(|k| k == base),
                },
            )
        })
        .collect();

    let consults = logs.consults.lock().expect("log").clone();
    let probes = logs.probes.lock().expect("log").clone();
    CaseResult {
        id,
        kind,
        consults,
        probes,
        observed,
        script,
    }
}

#[derive(Debug)]
enum CaseError {
    Terminal(String),
    Continue(String),
}

fn run_ladder_case(id: &str, kind: &str, rungs: &[serde_json::Value]) -> CaseResult {
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
                        "serve" => Ok::<_, CaseError>(Some(arg.expect("marker").to_string())),
                        "miss" => Ok(None),
                        "err-terminal" => Err(CaseError::Terminal(arg.expect("msg").to_string())),
                        "err-continue" => Err(CaseError::Continue(arg.expect("msg").to_string())),
                        other => panic!("unknown ladder behavior {other}"),
                    }
                }
            }
        },
        |e: &CaseError| matches!(e, CaseError::Terminal(_)),
    ));
    let observed = match result {
        Ok(Some(marker)) => Observed::Some(marker),
        Ok(None) => Observed::None,
        Err(CaseError::Terminal(msg) | CaseError::Continue(msg)) => Observed::Err(msg),
    };
    let script = rung_ids
        .into_iter()
        .map(|rid| {
            (
                rid.clone(),
                RungScript {
                    behavior: behaviors[&rid].clone(),
                    failed: false,
                    key_match: true,
                },
            )
        })
        .collect();
    let consults = consults.lock().expect("log").clone();
    CaseResult {
        id: id.to_string(),
        kind: kind.to_string(),
        consults,
        probes: Vec::new(),
        observed,
        script,
    }
}

fn expected_outcome(case: &serde_json::Value) -> Observed {
    let expected = &case["expected"];
    if let Some(marker) = expected["some"].as_str() {
        Observed::Some(marker.to_string())
    } else if expected["none"].as_bool().unwrap_or(false) {
        Observed::None
    } else if let Some(msg) = expected["err"].as_str() {
        Observed::Err(msg.to_string())
    } else {
        panic!("bad expected in case {}", case["id"])
    }
}

fn expected_strings(case: &serde_json::Value, field: &str) -> Vec<String> {
    case[field]
        .as_array()
        .expect("expected array")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect()
}

// A rung's scripted outcome is a genuine fallback cause when it is a miss, a
// failed-readiness skip, or a non-terminal error — never a success and never
// a terminal error (both must stop the walk).
fn is_genuine_cause(script: &RungScript) -> bool {
    if !script.key_match {
        return false;
    }
    if script.failed {
        return true;
    }
    matches!(
        split_behavior(&script.behavior).0,
        "miss" | "err-continue"
    )
}

fn is_serving(script: &RungScript) -> bool {
    script.key_match && !script.failed && split_behavior(&script.behavior).0 == "serve"
}

#[test]
fn routing_fallback_corpus_and_report() {
    let corpus_raw =
        std::fs::read_to_string(corpus_path("routing_fallback_corpus.json")).expect("corpus");
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

    for case in &cases {
        let result = run_case(case);
        let expected_consults = expected_strings(case, "expected_consults");
        let expected_probes = expected_strings(case, "expected_probes");
        let expected = expected_outcome(case);
        assert_eq!(
            result.consults, expected_consults,
            "case {}: consultation order deviation",
            result.id
        );
        assert_eq!(
            result.probes, expected_probes,
            "case {}: readiness probe deviation",
            result.id
        );
        assert_eq!(
            result.observed, expected,
            "case {}: outcome deviation",
            result.id
        );

        let script_by_id: HashMap<&str, &RungScript> = result
            .script
            .iter()
            .map(|(id, s)| (id.as_str(), s))
            .collect();
        let pos_of = |rid: &str| result.script.iter().position(|(id, _)| id == rid);
        if result.kind == "fallback" {
            fallback_cases += 1;
            // Consult-after-consult fallbacks: caused iff the previous
            // consult failed non-terminally (a success or terminal error
            // must have stopped the walk instead).
            for (i, cid) in result.consults.iter().enumerate() {
                if i == 0 {
                    continue;
                }
                total_fallbacks += 1;
                let prev = result.consults[i - 1].as_str();
                if is_genuine_cause(script_by_id[prev]) {
                    caused_fallbacks += 1;
                }
                let _ = cid;
            }
            // Per-rung failure accounting in registry order.
            for (idx, (rid, script)) in result.script.iter().enumerate() {
                if !script.key_match {
                    continue;
                }
                let consulted = result.consults.contains(rid);
                let later_consulted = result
                    .consults
                    .iter()
                    .any(|c| pos_of(c).is_some_and(|p| p > idx));
                let later_candidate = result.script[idx + 1..].iter().any(|(_, s)| {
                    s.key_match && !s.failed && split_behavior(&s.behavior).0 != "err-terminal"
                });
                // A failure with no later candidate is legitimate exhaustion
                // (the walk cannot continue); otherwise it must be followed.
                let followed_or_exhausted = later_consulted || !later_candidate;
                if script.failed {
                    // Failed-readiness skip: probed, never constructed.
                    if result.probes.contains(rid) && !consulted {
                        genuine_failures += 1;
                        if followed_or_exhausted {
                            followed_failures += 1;
                        }
                        if later_consulted {
                            total_fallbacks += 1;
                            caused_fallbacks += 1;
                        }
                    }
                } else if consulted {
                    match split_behavior(&script.behavior).0 {
                        "miss" | "err-continue" => {
                            genuine_failures += 1;
                            if followed_or_exhausted {
                                followed_failures += 1;
                            }
                        }
                        // "serve" wins and "err-terminal" aborts by design:
                        // neither is a fallback cause nor a failure to follow.
                        _ => {}
                    }
                }
            }
        } else {
            control_cases += 1;
            controls += 1;
            // A control passes with the exact expected sequences, the expected
            // outcome (already asserted), and no consult after a serve.
            let mut ok = true;
            let mut served = false;
            for (rid, script) in &result.script {
                if result.consults.contains(rid) {
                    if served {
                        ok = false;
                    }
                    if is_serving(script) {
                        served = true;
                    }
                }
            }
            if ok {
                controls_passed += 1;
            }
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
        std::fs::read_to_string(corpus_path("routing_fallback_report.json")).expect("report");
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
