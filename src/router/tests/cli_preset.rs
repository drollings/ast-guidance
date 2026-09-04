use super::*;

fn write_gguf(path: &std::path::Path, arch: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"GGUF").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&2u64.to_le_bytes()).unwrap();
    let write_kv_str = |f: &mut std::fs::File, key: &str, value: &str| {
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key.as_bytes()).unwrap();
        f.write_all(&8u32.to_le_bytes()).unwrap(); // string
        f.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        f.write_all(value.as_bytes()).unwrap();
    };
    write_kv_str(&mut f, "general.architecture", arch);
    let key = format!("{arch}.rope.freq_base");
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key.as_bytes()).unwrap();
    f.write_all(&6u32.to_le_bytes()).unwrap();
    f.write_all(&1_000_000.0f32.to_le_bytes()).unwrap();
}

#[test]
fn preset_has_default_section_and_model_section() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("library/code");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_gguf(&model_dir.join("latest.gguf"), "llama");
    std::fs::write(
        model_dir.join("params.json"),
        r#"{"temperature": 0.7, "num_ctx": 8192}"#,
    )
    .unwrap();

    let content = render_llama_preset(dir.path(), None);
    assert!(content.contains("[*]"));
    assert!(content.contains("[code]"));
    assert!(content.contains("model = "));
    assert!(content.contains("temp = 0.7"));
    assert!(content.contains("ctx-size = 8192"));
    assert!(content.contains("rope-freq-base = 1000000.0"));
}

#[test]
fn litellm_and_aichat_render_all_models() {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("abiray/x");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_gguf(&model_dir.join("latest.gguf"), "llama");
    let models = scan_gguf_models(dir.path());
    let litellm = render_litellm_yaml(&models);
    assert!(litellm.contains("model_name: abiray/x"));
    assert!(litellm.contains("model: llama.cpp/abiray/x"));
    let aichat = render_aichat_config(&models);
    assert!(aichat.contains("name: abiray/x"));
}
