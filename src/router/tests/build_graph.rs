//! Build-graph guards for optional `fluent-onnx`.
//!
//! The default build must not compile `fluent-onnx` (or `ort`) at all; the
//! `onnx` feature opts back in. These tests pin the Cargo declarations that
//! enforce that (hermetic: they read the in-tree manifests as text, never the
//! network) plus the schema facts the optionality relies on: the onnx fleet
//! declaration schema and role keys live in `fluent-llm` and parse identically
//! in both feature modes.

fn manifest(name: &str) -> String {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// The `[dependencies]`-section lines naming `dep` (skips `[dev-dependencies]`
/// and `[features]` so an opt-in dev-dep or feature wiring cannot fake the
/// assertion).
fn dependency_lines(manifest_text: &str, dep: &str) -> Vec<String> {
    let mut section = String::new();
    let mut out = Vec::new();
    for line in manifest_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed.to_string();
            continue;
        }
        if section == "[dependencies]" && !trimmed.starts_with('#') && line.contains(dep) {
            out.push(line.to_string());
        }
    }
    out
}

fn feature_line(manifest_text: &str, feature: &str) -> String {
    let mut section = String::new();
    for line in manifest_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed.to_string();
            continue;
        }
        if section == "[features]" && trimmed.starts_with(&format!("{feature} =")) {
            return line.to_string();
        }
    }
    panic!("no [{feature}] line in [features]");
}

#[test]
fn router_dep_on_onnx_is_optional() {
    let text = manifest("Cargo.toml");
    let deps = dependency_lines(&text, "fluent-onnx");
    assert_eq!(deps.len(), 1, "exactly one fluent-onnx dependency line");
    assert!(
        deps[0].contains("optional = true"),
        "fluent-onnx must be optional (default builds exclude it): {}",
        deps[0]
    );
    let onnx = feature_line(&text, "onnx");
    assert!(
        onnx.contains("dep:fluent-onnx"),
        "the onnx feature must pull the optional dep: {onnx}"
    );
    assert!(
        onnx.contains("fluent-onnx/onnx"),
        "the onnx feature must enable fluent-onnx/onnx: {onnx}"
    );
}

#[test]
fn bin_dep_on_onnx_is_optional_and_forwards_router() {
    let text = manifest("../bin/coral-router/Cargo.toml");
    let deps = dependency_lines(&text, "fluent-onnx");
    assert_eq!(deps.len(), 1, "exactly one fluent-onnx dependency line");
    assert!(
        deps[0].contains("optional = true"),
        "fluent-onnx must be optional (default builds exclude it): {}",
        deps[0]
    );
    let onnx = feature_line(&text, "onnx");
    assert!(
        onnx.contains("fluent-router/onnx"),
        "the bin onnx feature must forward fluent-router/onnx: {onnx}"
    );
    assert!(
        onnx.contains("fluent-onnx/onnx"),
        "the bin onnx feature must enable fluent-onnx/onnx: {onnx}"
    );
}

#[test]
fn llm_crate_has_no_onnx_dependency() {
    // Layering: `fluent-onnx` consumes `fluent-llm`, never the reverse, so
    // the declaration schema can move down without a dependency cycle.
    let text = manifest("../llm/Cargo.toml");
    assert!(
        !text.lines().any(|l| l.contains("fluent-onnx")),
        "fluent-llm must not depend on fluent-onnx in any section"
    );
}

/// The onnx fleet declaration schema parses identically with no onnx crate
/// in the build (config compatibility without the dependency).
#[test]
fn onnx_fleet_schema_parses_without_onnx_crate() {
    let fleet: fluent_llm::onnx_config::OnnxFleetConfig =
        serde_json::from_value(serde_json::json!({
            "encoder": {
                "pinned": false,
                "no_sleep": false,
                "sleep_idle_seconds": null,
                "total_timeout_ms": 0,
                "idle_timeout_ms": 0,
                "params": null,
                "instances": null,
                "model_path": "/models/encoder.onnx"
            }
        }))
        .expect("fleet schema parses");
    assert!(!fleet.is_empty());
    assert!(fleet.encoder.is_some());
    assert!(fleet.llm.is_none());
}

/// Routing keys are stable framework facts, not onnx-crate facts.
#[test]
fn onnx_role_registry_keys_are_stable() {
    use fluent_llm::onnx_config::OnnxRole;
    assert_eq!(OnnxRole::Encoder.registry_key(), "onnx/encoder");
    assert_eq!(OnnxRole::Pii.registry_key(), "onnx/pii");
    assert_eq!(OnnxRole::Router.registry_key(), "onnx/router");
    assert_eq!(OnnxRole::Colbert.registry_key(), "onnx/colbert");
    assert_eq!(OnnxRole::Llm.registry_key(), "onnx/llm");
}
