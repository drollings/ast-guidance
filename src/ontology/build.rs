//! ROADMAP M11.4 — build-time validation of the embedded YaGO class registry.
//!
//! `ontology/data/yago_classes.json` is embedded into the crate via
//! `include_str!` (`yago_loader.rs`). This build script fails the build if the
//! registry ever becomes malformed, so a bad hand-edit or a broken
//! `gen_yago_classes.py` run is caught at compile time, not in production.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=data/yago_classes.json");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let path = Path::new(&manifest_dir).join("data/yago_classes.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let entries: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("malformed yago_classes.json: {e}"));
    let arr = entries
        .as_array()
        .unwrap_or_else(|| panic!("yago_classes.json must be a JSON array"));
    for (i, entry) in arr.iter().enumerate() {
        assert!(
            entry.get("iri").and_then(|v| v.as_str()).is_some(),
            "yago_classes.json entry {i} is missing a string 'iri'"
        );
        assert!(
            entry.get("label").and_then(|v| v.as_str()).is_some(),
            "yago_classes.json entry {i} is missing a string 'label'"
        );
    }
    println!("cargo:info=validated yago_classes.json ({} classes)", arr.len());
}