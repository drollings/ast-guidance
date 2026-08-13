//! Filesystem admin commands (`list`, `scan`, `rm`, `show`, `pull`) operating
//! on the GGUF model layout.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use reqwest::Client;
use serde_json::{json, Value};

use super::{cli_err, sync_preset};
use super::ShowFlags;
use crate::cli::gguf::{
    cached_entries, format_relative_time, format_size, quant_name, read_gguf_metadata,
    resolve_model, scan_gguf_models, GgufEntry,
};
use crate::cli::preset::{render_aichat_config, render_litellm_yaml, write_models_preset};
use crate::cli::{CliContext, CliResult};

pub fn list(ctx: &CliContext) -> CliResult {
    let mut entries = cached_entries(&ctx.gguf_dir);
    if entries.is_empty() {
        return Ok(());
    }
    entries.sort_by(|a, b| a.display.cmp(&b.display));
    let na_width = entries
        .iter()
        .map(|e| e.display.len())
        .max()
        .unwrap_or(4)
        .clamp(4, 90);
    println!(
        "{:<width$}  {:>8}  {:12}  {:10}  {:14}",
        "NAME",
        "SIZE",
        "QUANT",
        "ARCH",
        "MODIFIED",
        width = na_width
    );
    for e in entries {
        let quant = if e.tag.is_empty() || e.tag == "latest" {
            e.file_type.map_or_else(|| "-".to_string(), quant_name)
        } else {
            e.tag.clone()
        };
        let arch = if e.arch.is_empty() {
            "?".to_string()
        } else {
            e.arch.clone()
        };
        println!(
            "{:<width$}  {:>8}  {:12}  {:10}  {:14}",
            e.display,
            format_size(e.size),
            quant,
            arch,
            format_relative_time(e.mtime),
            width = na_width
        );
    }
    Ok(())
}

// ── scan ────────────────────────────────────────────────────────────────────

/// `coral-router scan` — scan the GGUF directory and regenerate serving
/// configs (llama.cpp preset, optionally LiteLLM/aichat/JSON).
pub fn scan(
    ctx: &CliContext,
    write_litellm: Option<&PathBuf>,
    write_aichat: Option<&PathBuf>,
    path_prefix: Option<&str>,
    json: bool,
) -> CliResult {
    let gguf_dir = &ctx.gguf_dir;
    if !gguf_dir.is_dir() {
        return Err(cli_err(format!(
            "GGUF directory not found: {}",
            gguf_dir.display()
        )));
    }
    let models = scan_gguf_models(gguf_dir);

    if let Some(path) = write_litellm {
        std::fs::write(path, render_litellm_yaml(&models))
            .map_err(|e| cli_err(format!("cannot write {}: {e}", path.display())))?;
        println!("Wrote LiteLLM config: {}", path.display());
    }
    if let Some(path) = write_aichat {
        std::fs::write(path, render_aichat_config(&models))
            .map_err(|e| cli_err(format!("cannot write {}: {e}", path.display())))?;
        println!("Wrote aichat config: {}", path.display());
    }

    if ctx.dry_run {
        ctx.log_debug("[DRY-RUN] would write models-preset.ini");
    } else {
        let preset = write_models_preset(gguf_dir, path_prefix)
            .map_err(|e| cli_err(format!("preset write failed: {e}")))?;
        println!("Wrote llama.cpp preset to {}", preset.display());
    }

    if json {
        let value: Vec<Value> = models
            .iter()
            .map(|m| {
                let dir = m.model_dir();
                json!({
                    "name": m.display,
                    "model_file": m.path.to_string_lossy(),
                    "params_file": opt_path(&dir.join("params.json")),
                    "template_file": opt_path(&dir.join("template.txt")),
                    "config_file": opt_path(&dir.join("config.json")),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&value)?);
    }

    if write_litellm.is_none() && write_aichat.is_none() && !json {
        let mut names: Vec<&str> = models.iter().map(|m| m.display.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        for name in names {
            let model = models
                .iter()
                .find(|m| m.display == name)
                .expect("name from list");
            let file = model.path.file_name().map_or_else(
                || "(no gguf)".to_string(),
                |s| s.to_string_lossy().into_owned(),
            );
            println!("  {name:<40} {file}");
        }
    }
    Ok(())
}

fn opt_path(path: &Path) -> Value {
    if path.exists() {
        Value::String(path.to_string_lossy().into_owned())
    } else {
        Value::Null
    }
}

// ── rm ──────────────────────────────────────────────────────────────────────

/// `coral-router rm <model>` — remove a model directory and resync the preset.
pub fn rm(ctx: &CliContext, model: &str) -> CliResult {
    let resolved = resolve_model(&ctx.gguf_dir, model).ok_or_else(|| {
        cli_err(format!(
            "model '{model}' not found in {}",
            ctx.gguf_dir.display()
        ))
    })?;
    let model_dir = resolved.model_dir().to_path_buf();
    if ctx.dry_run {
        ctx.log_debug(&format!("[DRY-RUN] rm -rf {}", model_dir.display()));
        return Ok(());
    }
    std::fs::remove_dir_all(&model_dir)
        .map_err(|e| cli_err(format!("cannot remove {}: {e}", model_dir.display())))?;
    println!("Removed {}", model_dir.display());
    sync_preset(&ctx.gguf_dir);
    Ok(())
}

// ── show ────────────────────────────────────────────────────────────────────

/// `coral-router show <model>` — print a model summary, Modelfile, or one of
/// the specific sections (license/parameters/system/template).
pub fn show(ctx: &CliContext, model: &str, flags: &ShowFlags) -> CliResult {
    let resolved = resolve_model(&ctx.gguf_dir, model).ok_or_else(|| {
        cli_err(format!(
            "model '{model}' not found in {}",
            ctx.gguf_dir.display()
        ))
    })?;
    sync_preset(&ctx.gguf_dir);

    let parent = resolved.model_dir();
    let config = read_json(&parent.join("config.json"));
    let params = read_json(&parent.join("params.json"));
    let license_text = read_text(&parent.join("LICENSE.txt"));
    let template_text = read_text(&parent.join("template.txt"));
    let system_text = read_text(&parent.join("system.txt"));
    let gguf_meta = read_gguf_metadata(&resolved.path);

    if flags.modelfile {
        print_modelfile(
            &resolved,
            &ctx.gguf_dir,
            &params,
            template_text.as_deref(),
            license_text.as_deref(),
            system_text.as_deref(),
        );
        return Ok(());
    }
    if flags.license {
        println!("{}", license_text.as_deref().unwrap_or("(no license)"));
        return Ok(());
    }
    if flags.parameters {
        if let Some(map) = params.as_object().filter(|o| !o.is_empty()) {
            for (k, v) in map {
                println!("  {k:<24} {v}");
            }
        } else {
            println!("(no parameters)");
        }
        return Ok(());
    }
    if flags.system {
        println!(
            "{}",
            system_text.as_deref().unwrap_or("(no system message)")
        );
        return Ok(());
    }
    if flags.template {
        println!("{}", template_text.as_deref().unwrap_or("(no template)"));
        return Ok(());
    }

    print_model_summary(
        &resolved,
        &gguf_meta,
        &config,
        &params,
        license_text.as_deref(),
    );
    Ok(())
}

fn read_json(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| json!({}))
}

fn read_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn print_model_summary(
    entry: &GgufEntry,
    gguf_meta: &HashMap<String, Value>,
    config: &Value,
    params: &Value,
    license_text: Option<&str>,
) {
    let arch = gguf_meta
        .get("general.architecture")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            config
                .get("model_family")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "?".into());
    let param_count = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("?");
    let ctx_len = params
        .get("num_ctx")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            gguf_meta
                .get(&format!("{arch}.context_length"))
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "?".into());
    let embed_len = gguf_meta
        .get(&format!("{arch}.embedding_length"))
        .map_or_else(|| "?".to_string(), ToString::to_string);
    let file_type = config
        .get("file_type")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| gguf_meta.get("general.file_type").map(ToString::to_string))
        .unwrap_or_else(|| "?".into());

    println!("  Model");
    println!("    {:24} {}", "name", entry.display);
    println!("    {:24} {arch}", "architecture");
    println!("    {:24} {param_count}", "parameters");
    println!("    {:24} {ctx_len}", "context length");
    println!("    {:24} {embed_len}", "embedding length");
    println!("    {:24} {file_type}", "quantization");

    let mut caps = vec!["completion".to_string()];
    if !arch.to_lowercase().contains("embed") {
        caps.push("tools".into());
    }
    if entry.model_dir().join("projector.gguf").exists() {
        caps.push("vision".into());
    }
    println!("\n  Capabilities");
    for c in caps {
        println!("    {c}");
    }

    if let Some(map) = params.as_object().filter(|o| !o.is_empty()) {
        println!("\n  Parameters");
        for (k, v) in map {
            println!("    {k:<24} {v}");
        }
    }

    if let Some(license) = license_text {
        let lines: Vec<&str> = license.trim().lines().take(5).collect();
        if !lines.is_empty() {
            println!("\n  License");
            for line in &lines {
                println!("    {line}");
            }
            if license.trim().lines().count() > 5 {
                println!("    ...");
            }
        }
    }
}

fn print_modelfile(
    entry: &GgufEntry,
    gguf_dir: &Path,
    params: &Value,
    template_text: Option<&str>,
    license_text: Option<&str>,
    system_text: Option<&str>,
) {
    println!("# Modelfile generated by \"coral-router show\"");
    println!("# To build a new Modelfile based on this, replace FROM with:");
    println!("# FROM {}", entry.display);
    println!();
    let rel = entry.path.strip_prefix(gguf_dir).unwrap_or(&entry.path);
    println!("FROM {}/{}", gguf_dir.display(), rel.display());
    if let Some(t) = template_text {
        println!("TEMPLATE \"\"\"{t}\"\"\"");
    }
    if let Value::Object(map) = params {
        for (k, v) in map {
            println!("PARAMETER {k} {v}");
        }
    }
    if let Some(s) = system_text {
        println!("SYSTEM \"\"\"{s}\"\"\"");
    }
    if let Some(l) = license_text {
        println!("LICENSE \"\"\"{l}\"\"\"");
    }
}

// ── pull ────────────────────────────────────────────────────────────────────

/// `coral-router pull <namespace/model:tag>` — download a GGUF model from
/// HuggingFace (or copy a local file) and fetch its companion metadata.
pub async fn pull(ctx: &CliContext, model: &str, input: Option<PathBuf>, force: bool) -> CliResult {
    if !model.contains('/') || !model.contains(':') {
        return Err(cli_err(
            "Usage: coral-router pull <namespace/model:tag>\n  e.g. coral-router pull unsloth/Hy-MT2-7B-GGUF:ud-q5_k_xl",
        ));
    }
    let (base, tag) = model.rsplit_once(':').unwrap_or((model, "latest"));
    let base = base.to_lowercase();
    let tag = tag.to_lowercase();
    let clean_base = hf_repo_from_base(&base);
    let target_dir = ctx.gguf_dir.join(clean_base);
    let gguf_target = target_dir.join(format!("{tag}.gguf"));

    let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

    if let Some(input_path) = input {
        let src = input_path
            .canonicalize()
            .map_err(|e| cli_err(format!("input file not found: {e}")))?;
        if !src.is_file() {
            return Err(cli_err(format!("input file not found: {}", src.display())));
        }
        if gguf_target.exists() && !force {
            return Err(cli_err(format!(
                "Destination exists: {} (use --force to overwrite)",
                gguf_target.display()
            )));
        }
        if ctx.dry_run {
            ctx.log_debug(&format!(
                "[DRY-RUN] cp {} -> {}",
                src.display(),
                gguf_target.display()
            ));
        } else {
            std::fs::create_dir_all(&target_dir)
                .map_err(|e| cli_err(format!("cannot create {}: {e}", target_dir.display())))?;
            let size = std::fs::metadata(&src).map_or(0, |m| m.len());
            std::fs::copy(&src, &gguf_target).map_err(|e| cli_err(format!("copy failed: {e}")))?;
            println!(
                "Copied {} from {} to {}",
                format_size(size),
                src.display(),
                gguf_target.display()
            );
        }
    } else {
        if gguf_target.exists() {
            println!("Already exists: {}", gguf_target.display());
            return Ok(());
        }
        if ctx.dry_run {
            ctx.log_debug("[DRY-RUN] download skipped");
            return Ok(());
        }
        let hf_url = resolve_gguf_url(&client, clean_base, &tag).await;
        println!("Downloading {model} ...");
        println!("  from: {hf_url}");
        std::fs::create_dir_all(&target_dir)
            .map_err(|e| cli_err(format!("cannot create {}: {e}", target_dir.display())))?;
        download_to(&client, &hf_url, &gguf_target).await?;
    }

    if !ctx.dry_run {
        setup_model_from_hf(&client, clean_base, &target_dir).await;
        sync_preset(&ctx.gguf_dir);
    }
    Ok(())
}

/// Strip an `hf.co/` / `huggingface.co/` prefix from a repo id.
fn hf_repo_from_base(base: &str) -> &str {
    for prefix in ["hf.co/", "huggingface.co/"] {
        if let Some(rest) = base.strip_prefix(prefix) {
            return rest;
        }
    }
    base
}

async fn resolve_gguf_url(client: &Client, clean_base: &str, tag: &str) -> String {
    let exact = format!("https://huggingface.co/{clean_base}/resolve/main/{tag}.gguf");
    if let Ok(resp) = client.head(&exact).send().await {
        if resp.status().is_success() {
            return exact;
        }
    }
    let Some(files) = fetch_hf_json(
        client,
        &format!("https://huggingface.co/api/models/{clean_base}/tree/main"),
    )
    .await
    else {
        return exact;
    };
    let Some(files) = files.as_array() else {
        return exact;
    };
    let norm_tag = tag.to_lowercase().replace(['-', '.'], "_");
    let normalise = |s: &str| s.to_lowercase().replace(['-', '.'], "_");

    let mut best: Option<(usize, String)> = None;
    for entry in files {
        let Some(path) = entry.get("path").and_then(Value::as_str) else {
            continue;
        };
        if !path.ends_with(".gguf") {
            continue;
        }
        let stem = path.trim_end_matches(".gguf");
        if normalise(stem).ends_with(&norm_tag)
            && best.as_ref().is_none_or(|(len, _)| stem.len() < *len)
        {
            best = Some((stem.len(), path.to_string()));
        }
    }
    if let Some((_, path)) = best {
        return format!("https://huggingface.co/{clean_base}/resolve/main/{path}");
    }
    if tag == "latest" {
        if let Some(largest) = files
            .iter()
            .filter(|e| {
                e.get("path")
                    .and_then(Value::as_str)
                    .is_some_and(|p| p.ends_with(".gguf"))
            })
            .max_by_key(|e| e.get("size").and_then(Value::as_u64).unwrap_or(0))
        {
            if let Some(path) = largest.get("path").and_then(Value::as_str) {
                return format!("https://huggingface.co/{clean_base}/resolve/main/{path}");
            }
        }
    }
    exact
}

async fn fetch_hf_json(client: &Client, url: &str) -> Option<Value> {
    let resp = client.get(url).send().await.ok()?;
    resp.error_for_status().ok()?.json().await.ok()
}

async fn fetch_hf_raw(client: &Client, path: &str) -> Option<String> {
    let url = format!("https://huggingface.co/{path}");
    let resp = client.get(&url).send().await.ok()?;
    if resp.status().is_success() {
        resp.text().await.ok()
    } else {
        None
    }
}

/// Resolve a single base-model repo string from the HF API response.
fn resolve_base_model(data: &Value) -> Option<String> {
    let card = data
        .get("cardData")
        .and_then(Value::as_object)
        .and_then(|o| o.get("base_model"));
    let raw = card.or_else(|| data.get("base_model"))?;
    match raw {
        Value::String(s) if s.contains('/') => Some(s.clone()),
        Value::Array(arr) => arr
            .iter()
            .find_map(|v| v.as_str().filter(|s| s.contains('/')).map(str::to_string)),
        _ => None,
    }
}

async fn download_to(client: &Client, url: &str, dest: &Path) -> CliResult {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| cli_err(format!("download failed: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(cli_err(format!("download failed: HTTP {status}")));
    }
    let total = resp.content_length().unwrap_or(0);
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut file = std::fs::File::create(dest)
        .map_err(|e| cli_err(format!("cannot create {}: {e}", dest.display())))?;
    let mut last_log = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| cli_err(format!("download interrupted: {e}")))?;
        file.write_all(&chunk)
            .map_err(|e| cli_err(format!("write failed: {e}")))?;
        downloaded += chunk.len() as u64;
        if total > 0 && last_log.elapsed() >= Duration::from_secs(1) && downloaded < total {
            let pct = downloaded as f64 / total as f64 * 100.0;
            eprintln!(
                "  {pct:.0}% ({} / {})",
                format_size(downloaded),
                format_size(total)
            );
            last_log = Instant::now();
        }
    }
    println!(
        "Downloaded {} to {}",
        format_size(downloaded),
        dest.display()
    );
    Ok(())
}

/// Scrape the HF GGUF repo (and its safetensors base, when declared) for
/// companion files: chat template, config.json, and LICENSE.
async fn setup_model_from_hf(client: &Client, clean_base: &str, target_dir: &Path) {
    println!("Setting up model metadata from {clean_base} ...");
    let mut sources: Vec<String> = Vec::new();
    if let Some(data) = fetch_hf_json(
        client,
        &format!("https://huggingface.co/api/models/{clean_base}"),
    )
    .await
    {
        if let Some(base_model) = resolve_base_model(&data) {
            println!("Found base model: {base_model}");
            sources.push(base_model);
        } else {
            println!("No base model found for {clean_base}; falling back to GGUF repo");
        }
    }
    sources.push(clean_base.to_string());

    let params_file = target_dir.join("params.json");
    if !params_file.exists() {
        let params = json!({ "num_ctx": 8192, "temperature": 0.7, "top_k": 40, "top_p": 0.9 });
        if let Ok(text) = serde_json::to_string_pretty(&params) {
            let _ = std::fs::write(&params_file, text);
            println!("Wrote default params to {}", params_file.display());
        }
    }

    for source in &sources {
        let source = source.as_str();
        if let Some(tok_text) =
            fetch_hf_raw(client, &format!("{source}/raw/main/tokenizer_config.json")).await
        {
            if let Ok(tok_config) = serde_json::from_str::<Value>(&tok_text) {
                if let Some(template) = tok_config.get("chat_template").and_then(Value::as_str) {
                    let tmpl_file = target_dir.join("template.txt");
                    if !tmpl_file.exists() {
                        let _ = std::fs::write(&tmpl_file, template);
                        println!("Wrote template to {}", tmpl_file.display());
                    }
                }
            }
        }
        if let Some(cfg_text) =
            fetch_hf_raw(client, &format!("{source}/raw/main/config.json")).await
        {
            if let Ok(cfg) = serde_json::from_str::<Value>(&cfg_text) {
                let cfg_file = target_dir.join("config.json");
                if !cfg_file.exists() {
                    if let Ok(text) = serde_json::to_string_pretty(&cfg) {
                        let _ = std::fs::write(&cfg_file, text);
                        println!("Wrote config to {}", cfg_file.display());
                    }
                }
            }
        }
        let lic_file = target_dir.join("LICENSE.txt");
        if !lic_file.exists() {
            for lic_name in ["LICENSE", "LICENSE.txt"] {
                if let Some(text) =
                    fetch_hf_raw(client, &format!("{source}/raw/main/{lic_name}")).await
                {
                    let _ = std::fs::write(&lic_file, text);
                    println!("Wrote LICENSE.txt from {source}");
                    break;
                }
            }
        }
        if ["template.txt", "config.json", "LICENSE.txt"]
            .iter()
            .any(|n| target_dir.join(n).exists())
        {
            break;
        }
    }
}
