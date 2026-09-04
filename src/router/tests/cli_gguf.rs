use super::*;
use std::io::Write;

fn write_gguf(path: &Path, arch: &str, file_type: u32) {
    let mut f = File::create(path).unwrap();
    let write_bytes = |f: &mut File, bytes: &[u8]| {
        f.write_all(bytes).unwrap();
    };
    write_bytes(&mut f, b"GGUF");
    write_bytes(&mut f, &1u32.to_le_bytes());
    write_bytes(&mut f, &0u64.to_le_bytes()); // tensor_count
    write_bytes(&mut f, &2u64.to_le_bytes()); // kv_count
    let write_kv_str = |f: &mut File, key: &str, value: &str| {
        write_bytes(f, &(key.len() as u64).to_le_bytes());
        write_bytes(f, key.as_bytes());
        write_bytes(f, &8u32.to_le_bytes()); // string type
        write_bytes(f, &(value.len() as u64).to_le_bytes());
        write_bytes(f, value.as_bytes());
    };
    write_kv_str(&mut f, "general.architecture", arch);
    write_bytes(&mut f, &("general.file_type".len() as u64).to_le_bytes());
    write_bytes(&mut f, b"general.file_type");
    write_bytes(&mut f, &4u32.to_le_bytes()); // uint32
    write_bytes(&mut f, &file_type.to_le_bytes());
}

#[test]
fn read_gguf_metadata_parses_string_and_u32() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("model.gguf");
    write_gguf(&gguf, "llama", 15);
    let meta = read_gguf_metadata(&gguf);
    assert_eq!(
        meta.get("general.architecture").and_then(Value::as_str),
        Some("llama")
    );
    assert_eq!(
        meta.get("general.file_type").and_then(Value::as_u64),
        Some(15)
    );
}

#[test]
fn read_gguf_metadata_is_empty_for_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("bad.gguf");
    std::fs::write(&gguf, b"not a gguf file at all").unwrap();
    assert!(read_gguf_metadata(&gguf).is_empty());
}

#[test]
fn quant_name_maps_file_type() {
    assert_eq!(quant_name(15), "q4_k_m");
    assert_eq!(quant_name(18), "q6_k");
    assert_eq!(quant_name(999), "ftype_999");
}

#[test]
fn strip_gguf_suffix_strips_each_component() {
    assert_eq!(
        strip_gguf_suffix("unsloth/Hy-MT2-1.8B-GGUF"),
        "unsloth/Hy-MT2-1.8B"
    );
}

#[test]
fn ollama_display_elides_library() {
    assert_eq!(ollama_display("library/code", "latest"), "code:latest");
    assert_eq!(ollama_display("library/code", ""), "code");
    assert_eq!(ollama_display("abiray/x", "q4_k_m"), "abiray/x:q4_k_m");
}

#[test]
fn format_size_is_si() {
    assert_eq!(format_size(512), "512B");
    assert_eq!(format_size(6_500_000_000), "6.5 GB");
    assert_eq!(format_size(843_000_000), "843 MB");
    assert_eq!(format_size(843_400_000), "843.4 MB");
}

#[test]
fn format_keep_alive_is_ollama_style() {
    assert_eq!(format_keep_alive(9000), "2h30m");
    assert_eq!(format_keep_alive(-1), "Forever");
    assert_eq!(format_keep_alive(45), "45s");
}

#[test]
fn resolve_model_matches_name_tag_and_library_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("library/code");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_gguf(&model_dir.join("latest.gguf"), "llama", 15);
    write_gguf(&model_dir.join("q4_k_m.gguf"), "llama", 15);

    let entries = list_gguf_entries(dir.path());
    assert!(!entries.is_empty());
    let resolved = resolve_model(dir.path(), "code").expect("resolves code");
    assert_eq!(resolved.name, "library/code");
    let tagged = resolve_model(dir.path(), "code:q4_k_m").expect("resolves tag");
    assert_eq!(tagged.tag, "q4_k_m");
    assert!(resolve_model(dir.path(), "CODE:latest").is_some());
    assert!(resolve_model(dir.path(), "does-not-exist").is_none());
}

#[test]
fn sync_cache_is_stable_and_schema_compatible() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("abiray/test");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_gguf(&model_dir.join("latest.gguf"), "llama", 15);

    sync_cache(dir.path());
    let cache = load_gguf_cache(dir.path()).expect("cache written");
    assert_eq!(cache.version, 2);
    assert_eq!(cache.model_count, cache.models.len());
    let entry = cache
        .models
        .get("abiray/test")
        .expect("default entry keyed by name");
    assert!(entry.path.ends_with("latest.gguf"));
    assert!(!needs_rescan(dir.path()), "fresh cache requires no rescan");
}

#[test]
fn sync_cache_remains_fresh_across_rewrite_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("abiray/test");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_gguf(&model_dir.join("latest.gguf"), "llama", 15);

    for _ in 0..5 {
        sync_cache(dir.path());
        assert!(!needs_rescan(dir.path()), "fresh cache requires no rescan");
    }
}

#[test]
fn mtime_snapshot_round_trips_through_json() {
    // A fractional-second mtime at ~1.7e9 loses up to one ULP (~238 ns) in
    // serde_json's f64 parser (1786778176.386162519 round-trips as
    // ...3861623), so the raw value itself cannot be persisted losslessly.
    // The integer-valued microsecond snapshot must be exactly representable
    // and survive the JSON round-trip bit-for-bit, which is what keeps the
    // `parent_mtime` token stable across a cache rewrite.
    let raw = 1786778176.386162519f64;
    let snap = mtime_snapshot(raw);
    assert_eq!(snap.fract(), 0.0, "snapshot must be an integer-valued f64");
    assert_eq!(snap, 1786778176386163.0);
    let text = serde_json::to_string(&snap).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.as_f64(), Some(snap));
}
