use super::*;
use crate::config::{OnnxConfig, OnnxTask, Quant};
use crate::error::OrtError;
use crate::session::{OrtSessionRegistry, SessionHandle, SessionLoader};
use std::sync::Arc;

/// A stub loader that returns a canned handle (no ort, no model).
#[derive(Default)]
struct StubLoader;

impl SessionLoader for StubLoader {
    fn load(&self, _config: &OnnxConfig, _model_key: &str) -> Result<SessionHandle, OrtError> {
        Ok(SessionHandle::new("stub-session"))
    }
}

fn config_for(task: OnnxTask, resident: bool) -> OnnxConfig {
    OnnxConfig::new()
        .model_path("/models/test.onnx")
        .tokenizer_path("/models/tokenizer.json")
        .task(task)
        .quantization(Quant::Q8)
        .resident(resident)
        .build()
}

fn unloadable() -> OnnxConfig {
    config_for(OnnxTask::FillMask, false)
}

fn unloadable_with_resident(bytes: u64) -> OnnxConfig {
    OnnxConfig::new()
        .model_path("/models/test.onnx")
        .tokenizer_path("/models/tokenizer.json")
        .task(OnnxTask::FillMask)
        .quantization(Quant::Q8)
        .resident(false)
        .maybe_resident_bytes(Some(bytes))
        .build()
}

fn registry() -> Arc<OrtSessionRegistry> {
    Arc::new(OrtSessionRegistry::new(Arc::new(StubLoader)))
}

/// A clock frozen at `now_unix_ms() + offset` so entries loaded against the
/// real clock are deterministically idle/fresh relative to the loop.
fn clock_offset(offset_ms: i64) -> Arc<dyn Fn() -> i64 + Send + Sync> {
    Arc::new(move || now_unix_ms() + offset_ms)
}

#[test]
fn idle_release_frees_loaded_unloadable_entry() {
    let reg = registry();
    reg.register_with_lifecycle(
        "lazy",
        unloadable(),
        unloadable_policy(),
        false,
        Some(1),
    )
    .expect("register");
    // An `Always` entry is the control: never released.
    reg.register("always", config_for(OnnxTask::FillMask, true))
        .expect("register");

    reg.ensure_loaded("lazy").expect("load");
    assert_eq!(reg.unloadable_keys(), vec!["lazy".to_string()]);

    // Fresh clock: the entry was just used → not released.
    let fresh = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        None,
        clock_offset(0),
    );
    fresh.residency_cycle();
    assert_eq!(reg.unloadable_keys(), vec!["lazy".to_string()]);

    // Clock 10s ahead: the 1s-idle entry is released; `Always` stays.
    let stale = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        None,
        clock_offset(10_000),
    );
    stale.residency_cycle();
    assert_eq!(reg.unloadable_keys(), Vec::<String>::new());
    assert!(reg.last_used_of("always").unwrap() > 0, "Always stays loaded");
    assert!(
        reg.residency_report()
            .iter()
            .find(|r| r.key == "always")
            .unwrap()
            .loaded,
        "Always entry never released"
    );
}

#[test]
fn idle_release_uses_entry_sleep_idle_and_skips_pinned() {
    let reg = registry();
    // A pinned `Unloadable` entry must never be released — it is loaded but
    // is not an idle-release candidate.
    reg.register_with_lifecycle(
        "pinned",
        unloadable(),
        unloadable_policy(),
        true,
        Some(1),
    )
    .expect("register");
    reg.ensure_loaded("pinned").expect("load");

    let loop_ = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        None,
        clock_offset(1_000_000),
    );
    loop_.residency_cycle();
    let report = reg.residency_report();
    let pinned = report.iter().find(|r| r.key == "pinned").unwrap();
    assert!(pinned.loaded, "pinned entry never released");
    assert_eq!(reg.unloadable_keys(), Vec::<String>::new(), "pinned is not a candidate");
}

#[test]
fn working_set_budget_evicts_lru_largest_first() {
    let reg = registry();
    // big: 100 bytes, loaded first (older last_used). small: 50 bytes.
    let big = unloadable_with_resident(100);
    let small = unloadable_with_resident(50);
    reg.register_with_lifecycle("big", big, unloadable_policy(), false, None)
        .expect("big");
    std::thread::sleep(std::time::Duration::from_millis(20));
    reg.register_with_lifecycle("small", small, unloadable_policy(), false, None)
        .expect("small");
    reg.ensure_loaded("big").expect("load big");
    reg.ensure_loaded("small").expect("load small");

    // Budget 60: total 150 > 60 → evict the largest (big) first, which
    // brings usage to 50 ≤ 60, so small stays.
    let loop_ = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        Some(60),
        clock_offset(0),
    );
    loop_.residency_cycle();
    assert_eq!(reg.unloadable_keys(), vec!["small".to_string()]);
}

#[test]
fn working_set_tie_evicts_oldest_first() {
    let reg = registry();
    // Two equal-size entries; `old` is loaded first, so it has the older
    // last_used. Budget 150: total 200 > 150 → one eviction, and the
    // oldest of the equal footprints goes.
    let a = unloadable_with_resident(100);
    let b = unloadable_with_resident(100);
    reg.register_with_lifecycle("old", a, unloadable_policy(), false, None)
        .expect("old");
    reg.register_with_lifecycle("new", b, unloadable_policy(), false, None)
        .expect("new");
    // Loads happen back-to-back by design here; `last_used` is stamped at
    // load time, so give the older entry a visibly older clock.
    reg.ensure_loaded("old").expect("load old");
    std::thread::sleep(std::time::Duration::from_millis(20));
    reg.ensure_loaded("new").expect("load new");

    let loop_ = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        Some(150),
        clock_offset(0),
    );
    loop_.residency_cycle();
    assert_eq!(reg.unloadable_keys(), vec!["new".to_string()], "oldest evicted first");
}

#[test]
fn working_set_budget_within_limits_releases_nothing() {
    let reg = registry();
    let cfg = unloadable_with_resident(100);
    reg.register_with_lifecycle("m", cfg, unloadable_policy(), false, None)
        .expect("register");
    reg.ensure_loaded("m").expect("load");

    let loop_ = OrtResidencyLoop::new_with_clock(
        Arc::clone(&reg),
        Duration::from_secs(1),
        DEFAULT_SLEEP_IDLE_SECONDS,
        Some(1000),
        clock_offset(0),
    );
    loop_.residency_cycle();
    assert_eq!(reg.unloadable_keys(), vec!["m".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_loops_on_injected_runtime() {
// `start` must run on an injected `Runtime` (the real tokio backend here),
// never ambient `tokio::spawn`, and drive the loop.
let rt = fluent_concurrency::tokio_runtime();
let reg = registry();
reg.register_with_lifecycle(
    "lazy",
    unloadable(),
    unloadable_policy(),
    false,
    Some(1),
)
.expect("register");
reg.ensure_loaded("lazy").expect("load");
let loop_ = OrtResidencyLoop::new(
    Arc::clone(&reg),
    Duration::from_millis(5),
    DEFAULT_SLEEP_IDLE_SECONDS,
    None,
);
let handle = loop_.start(&rt);
// NoopRuntime's sleep returns immediately, so the loop spins; a couple
// of cycles are enough to observe the idle release via the real clock
// (the entry's sleep_idle is 1s, so give it a moment).
tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
handle.abort();
let _ = handle.await;
assert_eq!(reg.unloadable_keys(), Vec::<String>::new(), "loop released idle entry");
}

/// The default `Unloadable` policy (weights+context both releasable).
fn unloadable_policy() -> crate::config::ResidencyPolicy {
    crate::config::ResidencyPolicy::Unloadable {
        weights: true,
        context: true,
    }
}
