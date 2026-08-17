use fluent_types::MemberType;
use guidance_core::sync::json_store;

mod common;
use common::make_engine;

const FIXTURE_ZIG: &str = r#"/// Sample Zig file for AST parsing tests
const std = @import("std");

pub fn greet(name: []const u8) []const u8 {
    return "Hello, " ++ name;
}

pub const Config = struct {
    port: u16 = 8079,
    host: []const u8 = "localhost",
};
"#;

const FIXTURE_PYTHON: &str = r#""""Sample Python file for AST parsing tests."""


def greet(name: str) -> str:
    return f"Hello, {name}"


class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
"#;

#[test]
fn e2e_zig_gen_roundtrip() {
    let fx = make_engine();
    let zig_file = fx.source_dir.join("main.zig");
    std::fs::write(&zig_file, FIXTURE_ZIG).expect("write fixture");

    let mut engine = fx.engine;
    let doc = engine.gen(&zig_file).expect("gen");

    // Verify structure matches Zig output (2 members: fn greet + struct Config)
    assert_eq!(doc.meta.language.as_str(), "zig");
    assert_eq!(doc.meta.module.as_str(), "main");
    assert_eq!(doc.meta.source.as_str(), "src/main.zig");

    let greet = doc.members.iter().find(|m| m.name == "greet");
    assert!(greet.is_some(), "should find greet function");
    assert_eq!(greet.unwrap().type_name, MemberType::FnDecl);
    assert!(greet.unwrap().is_pub);

    let config = doc.members.iter().find(|m| m.name == "Config");
    assert!(config.is_some(), "should find Config struct");
    assert_eq!(config.unwrap().type_name, MemberType::Struct);

    // Verify JSON serialization round-trip
    let json_path = fx.guidance_dir.join("src").join("main.zig.json");
    assert!(json_path.exists(), "JSON file should exist");

    let loaded = json_store::load_guidance(&json_path)
        .expect("load")
        .expect("should have doc");
    assert_eq!(loaded.members.len(), doc.members.len());
    assert_eq!(loaded.meta.module, doc.meta.module);

    // Verify staleness
    assert!(!guidance_core::sync::staleness::should_generate(
        &json_path, &zig_file
    ));
}

#[test]
fn e2e_python_gen_roundtrip() {
    let fx = make_engine();
    let py_file = fx.source_dir.join("main.py");
    std::fs::write(&py_file, FIXTURE_PYTHON).expect("write fixture");

    let mut engine = fx.engine;
    let doc = engine.gen(&py_file).expect("gen python");

    assert_eq!(doc.meta.language.as_str(), "python");

    let greet = doc.members.iter().find(|m| m.name == "greet");
    assert!(greet.is_some(), "should find greet function");

    let calc = doc.members.iter().find(|m| m.name == "Calculator");
    assert!(calc.is_some(), "should find Calculator class");
}

#[test]
fn e2e_incremental_sync() {
    let fx = make_engine();
    let zig_file = fx.source_dir.join("lib.zig");
    std::fs::write(&zig_file, "pub fn alpha() void {}\n").expect("write");

    let mut engine = fx.engine;

    // First gen
    assert!(engine.gen_if_stale(&zig_file).expect("gen if stale first"));

    // Second gen should not be stale (JSON is newer)
    std::thread::sleep(std::time::Duration::from_millis(1100));
    assert!(!engine.gen_if_stale(&zig_file).expect("gen if stale second"));

    // Modify source (wait >1s so mtime difference is detectable)
    std::thread::sleep(std::time::Duration::from_secs(2));
    std::fs::write(&zig_file, "pub fn alpha() void {}\npub fn beta() void {}\n").expect("write");

    // Third gen should be stale again (source mtime updated)
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(engine.gen_if_stale(&zig_file).expect("gen if stale third"));

    // Verify both functions are present
    let doc = engine
        .load_doc(&zig_file)
        .expect("load doc")
        .expect("should have doc");
    assert_eq!(doc.members.len(), 2);
    assert!(doc.members.iter().any(|m| m.name == "beta"));
}

#[test]
fn e2e_sync_status_clean() {
    let fx = make_engine();
    let zig_file = fx.source_dir.join("foo.zig");
    std::fs::write(&zig_file, "pub fn foo() void {}\n").expect("write");

    let mut engine = fx.engine;
    assert!(engine.gen_if_stale(&zig_file).expect("gen if stale"));

    let status = engine.status().expect("status");
    assert_eq!(status.total_files, 1);
    assert_eq!(status.stale_files, 0);
    assert_eq!(status.up_to_date, 1);
    assert!(status.is_clean());
}
