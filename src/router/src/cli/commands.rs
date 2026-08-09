//! Implementation of the `coral-router` admin subcommands, ported from
//! `gguf_tool.py`.
//!
//! Filesystem commands (`list`, `scan`, `rm`, `show`, `pull`) operate on the
//! GGUF layout. Server commands (`ps`, `stop`, `speedtest`) drive a running
//! Coral Router through its HTTP API.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use reqwest::Client;
use serde_json::{json, Value};

use super::gguf::{
    cached_entries, compute_short_id, format_keep_alive, format_relative_time, format_size,
    ollama_display, quant_name, read_gguf_metadata, resolve_model, scan_gguf_models, sync_cache,
    GgufEntry,
};
use super::preset::{render_aichat_config, render_litellm_yaml, write_models_preset};
use super::{CliContext, CliError, CliResult};
use crate::config::RouterConfig;

/// Flags controlling `show` output modes (mirror `gguf_tool.py`).
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct ShowFlags {
    pub modelfile: bool,
    pub license: bool,
    pub parameters: bool,
    pub system: bool,
    pub template: bool,
}

/// Sampling overrides for `speedtest` (mirror `gguf_tool.py`).
#[derive(Debug, Clone)]
pub struct SpeedtestArgs {
    pub model: String,
    pub tokens: u32,
    pub prompt: Option<String>,
    pub temperature: f64,
}

/// Resolve the router base URL from `--api-url`, else the config's
/// `server.bind_addr`, else a sane default.
pub fn router_base_url(api_url: Option<&str>, config_path: Option<&Path>) -> String {
    if let Some(url) = api_url {
        return url.trim_end_matches('/').to_string();
    }
    if let Some(path) = config_path {
        let mut config = common_core::config::load_json_or_default::<RouterConfig>(path);
        config.apply_defaults();
        let addr = &config.server.bind_addr;
        if !addr.is_empty() {
            return format!("http://{addr}");
        }
    }
    "http://127.0.0.1:8079".to_string()
}

/// Load the router config for model-key/weights resolution (best-effort).
fn load_config(config_path: Option<&Path>) -> Option<RouterConfig> {
    let path = config_path?;
    let mut config = common_core::config::load_json_or_default::<RouterConfig>(path);
    config.apply_defaults();
    Some(config)
}

fn cli_err(message: impl Into<String>) -> CliError {
    CliError::new(message)
}

fn sync_preset(gguf_dir: &Path) {
    sync_cache(gguf_dir);
    let _ = write_models_preset(gguf_dir, Some(super::DEFAULT_GGUF_DIR));
}

// ── list ────────────────────────────────────────────────────────────────────

/// `coral-router list` — table of scanned GGUF models.
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
    println!("FROM /app/ai/models/gguf/{}", rel.display());
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

// ── ps ──────────────────────────────────────────────────────────────────────

/// `coral-router ps` — list running models via the router's `/v1/models` and
/// `/instances` API.
pub async fn ps(ctx: &CliContext, api_url: Option<&str>, config_path: Option<&Path>) -> CliResult {
    let base = router_base_url(api_url, config_path);
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let config = load_config(config_path);

    let models = get_json(&client, &format!("{base}/v1/models"))
        .await
        .map_err(|e| {
            cli_err(format!(
                "Error: llama-server not reachable. Is it running? {e}"
            ))
        })?;
    let loaded: Vec<Value> = models
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter(|m| {
                    m.get("state")
                        .and_then(Value::as_str)
                        .is_some_and(|s| s == "loaded" || s == "sleeping")
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if loaded.is_empty() {
        println!("(no loaded models)");
        return Ok(());
    }

    let instances = get_json(&client, &format!("{base}/instances"))
        .await
        .unwrap_or_else(|_| json!({ "instances": [] }));
    let instance_list = instances
        .get("instances")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // Group instance entries by owning model key (instance id = `<key>:<name>`).
    let mut by_model: std::collections::BTreeMap<String, Vec<Value>> =
        std::collections::BTreeMap::new();
    for inst in &instance_list {
        let id = inst.get("id").and_then(Value::as_str).unwrap_or_default();
        let key = id.split(':').next().unwrap_or(id).to_string();
        by_model.entry(key).or_default().push(inst.clone());
    }

    let mut printed = 0usize;
    for (key, insts) in &by_model {
        print_weight_block(config.as_ref(), &ctx.gguf_dir, key, insts);
        printed += 1;
    }
    for m in &loaded {
        let id = m.get("id").and_then(Value::as_str).unwrap_or_default();
        let key = id.split(':').next().unwrap_or(id).to_string();
        if !by_model.contains_key(&key) {
            print_weight_block(config.as_ref(), &ctx.gguf_dir, &key, &[]);
            printed += 1;
        }
    }
    if printed == 0 {
        println!("(no loaded models)");
    }
    Ok(())
}

fn print_weight_block(
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    model_key: &str,
    insts: &[Value],
) {
    let (weights_bytes, gguf_file) = model_weights(config, gguf_dir, model_key);
    let ctx_mem_sum: u64 = insts
        .iter()
        .map(|i| i.get("vram_bytes").and_then(Value::as_u64).unwrap_or(0))
        .sum();
    let loaded_ctx: u64 = insts
        .iter()
        .filter(|i| {
            i.get("state")
                .and_then(Value::as_str)
                .is_some_and(|s| s != "sleeping")
        })
        .map(|i| i.get("vram_bytes").and_then(Value::as_u64).unwrap_or(0))
        .sum();
    let weights_vram = weights_bytes.saturating_sub(loaded_ctx.min(weights_bytes));
    let weights_cpu = weights_bytes.saturating_sub(weights_vram);
    let total_sys = weights_bytes
        .saturating_add(ctx_mem_sum)
        .saturating_sub(weights_vram);

    let (short_id, arch) = match &gguf_file {
        Some(path) => (
            compute_short_id(path),
            read_gguf_metadata(path)
                .get("general.architecture")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        ),
        None => ("?".repeat(12), "?".to_string()),
    };
    let tag = gguf_file
        .as_ref()
        .and_then(|p| p.file_stem())
        .map_or_else(|| "-".to_string(), |s| s.to_string_lossy().to_lowercase());
    let quant = if tag.is_empty() || tag == "latest" {
        "-".to_string()
    } else {
        tag.clone()
    };
    let display = ollama_display(model_key, &tag);

    println!("{display}  (id={short_id}, arch={arch}, quant={quant})");
    println!(
        "  weights   {:>12}   VRAM {:>10} / SYS {:>10}",
        format_size(weights_bytes),
        format_size(weights_vram),
        format_size(weights_cpu)
    );
    if !insts.is_empty() {
        println!("  instances");
        for inst in insts {
            let raw_id = inst.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = raw_id
                .strip_prefix(&format!("{model_key}:"))
                .unwrap_or(raw_id)
                .to_string();
            let n_ctx = inst
                .get("n_ctx")
                .and_then(Value::as_u64)
                .map_or_else(|| "?".to_string(), |v| v.to_string());
            let par = inst
                .get("parallel")
                .and_then(Value::as_u64)
                .map_or_else(|| "?".to_string(), |v| v.to_string());
            let state = inst.get("state").and_then(Value::as_str).unwrap_or("?");
            let sleep = inst_sleep_label(inst);
            let mem = format_size(inst.get("vram_bytes").and_then(Value::as_u64).unwrap_or(0));
            println!(
                "    {name:<16}  ctx {n_ctx:>7}  par {par:>3}  sleep {sleep:<9}  state {state:<9}  ctx-mem {mem:>10}"
            );
        }
    }
    println!(
        "  total     VRAM {:>10} / SYS RAM {:>10}",
        format_size(weights_vram.saturating_add(ctx_mem_sum)),
        format_size(total_sys)
    );
    println!();
}

/// Resolve a model key's weights bytes and GGUF path from the config or the
/// GGUF layout.
fn model_weights(
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    model_key: &str,
) -> (u64, Option<PathBuf>) {
    if let Some(cfg) = config {
        if let Some(entry) = cfg.models.get(model_key) {
            if let Some(weights) = &entry.weights {
                let path = PathBuf::from(weights);
                let size = std::fs::metadata(&path).map_or(0, |m| m.len());
                return (size, Some(path));
            }
        }
    }
    // Fall back to the GGUF layout: `gguf_dir/<key>/latest.gguf` or the
    // smallest GGUF under the key's directory.
    let dir = gguf_dir.join(model_key);
    let Ok(read) = std::fs::read_dir(&dir) else {
        return (0, None);
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "gguf") {
            files.push(path);
        }
    }
    let chosen = files
        .iter()
        .find(|p| p.file_stem().is_some_and(|s| s == "latest"))
        .or_else(|| {
            files
                .iter()
                .min_by_key(|p| std::fs::metadata(p).map_or(u64::MAX, |m| m.len()))
        });
    match chosen {
        Some(path) => {
            let size = std::fs::metadata(path).map_or(0, |m| m.len());
            (size, Some(path.clone()))
        }
        None => (0, None),
    }
}

fn inst_sleep_label(inst: &Value) -> String {
    let pinned = inst.get("pinned").and_then(Value::as_bool).unwrap_or(false);
    let no_sleep = inst
        .get("no_sleep")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if pinned || no_sleep {
        return "Forever".into();
    }
    match inst.get("sleep_idle_seconds").and_then(Value::as_i64) {
        Some(v) if v < 0 => "inherit".into(),
        Some(0) => "Forever".into(),
        Some(v) => format_keep_alive(v),
        None => "inherit".into(),
    }
}

// ── stop ────────────────────────────────────────────────────────────────────

/// `coral-router stop <model>` — unload a running model via `POST
/// /models/unload` and wait for it to disappear from `/v1/models`.
pub async fn stop(
    ctx: &CliContext,
    api_url: Option<&str>,
    config_path: Option<&Path>,
    model: &str,
    force: bool,
) -> CliResult {
    let base = router_base_url(api_url, config_path);
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
    let config = load_config(config_path);
    let model_key = resolve_router_model_key(config.as_ref(), &ctx.gguf_dir, model)
        .ok_or_else(|| cli_err(format!("model '{model}' not found")))?;
    if force {
        ctx.log_debug("force=true — the router owns the server; unload is graceful");
    }
    println!("Unloading model '{model_key}' ...");

    let resp = client
        .post(format!("{base}/models/unload"))
        .json(&json!({ "model": model_key }))
        .send()
        .await
        .map_err(|e| cli_err(format!("Cannot connect to {base}: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status == 404 {
            println!("Model '{model_key}' has no managed server (nothing to unload)");
            return Ok(());
        }
        return Err(cli_err(format!(
            "/models/unload failed: HTTP {status}: {text}"
        )));
    }

    for attempt in 0..6 {
        let loaded = loaded_model_ids(&client, &base).await;
        if !loaded.contains(&model_key) {
            println!("Model '{model_key}' unloaded");
            return Ok(());
        }
        if attempt < 5 {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    Err(cli_err(format!(
        "Model '{model_key}' still listed as loaded after unload"
    )))
}

async fn loaded_model_ids(client: &Client, base: &str) -> Vec<String> {
    let Ok(models) = get_json(client, &format!("{base}/v1/models")).await else {
        return Vec::new();
    };
    models
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Map a CLI model argument to a router config model key: the exact key, a
/// `key:qualifier` base, or a GGUF-layout name resolved against configured
/// weights paths.
fn resolve_router_model_key(
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    arg: &str,
) -> Option<String> {
    let arg = arg.to_lowercase();
    let config = config?;
    if config.models.contains_key(&arg) {
        return Some(arg);
    }
    let (base, _) = arg.rsplit_once(':').unwrap_or((arg.as_str(), ""));
    if config.models.contains_key(base) {
        return Some(base.to_string());
    }
    let resolved = resolve_model(gguf_dir, &arg)?;
    config.models.iter().find_map(|(key, entry)| {
        entry
            .weights
            .as_deref()
            .is_some_and(|w| Path::new(w) == resolved.path)
            .then(|| key.clone())
    })
}

// ── speedtest ───────────────────────────────────────────────────────────────

/// `coral-router speedtest` — measure generation throughput: runs a chat
/// completion through the router, reports per-run tokens/s plus the server's
/// lifetime-average gauges from `/metrics`.
pub async fn speedtest(
    ctx: &CliContext,
    api_url: Option<&str>,
    config_path: Option<&Path>,
    args: &SpeedtestArgs,
) -> CliResult {
    let base = router_base_url(api_url, config_path);
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let config = load_config(config_path);

    let model_name = resolve_speedtest_model(config.as_ref(), &args.model);
    let prompt = args
        .prompt
        .as_deref()
        .unwrap_or("Write a brief overview of machine learning, at least 4000 words.");
    let run_test = args.tokens > 0;

    let baseline = if run_test {
        Some(
            fetch_model_metrics(&client, &base, &model_name)
                .await
                .map_err(|e| cli_err(format!("Cannot read metrics from {base}: {e}")))?,
        )
    } else {
        None
    };

    let wall = if run_test {
        let payload = json!({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens,
            "temperature": args.temperature,
            "stream": false,
        });
        let wall_start = Instant::now();
        let resp = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| cli_err(format!("Generation request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(cli_err(format!(
                "Generation request failed: HTTP {status}: {text}"
            )));
        }
        let data: Value = resp.json().await?;
        ctx.log_debug(&format!("completion: {data}"));
        let elapsed = wall_start.elapsed();
        let usage = data.get("usage").cloned().unwrap_or_else(|| json!({}));
        speedtest_report(&usage, elapsed);
        elapsed.as_secs_f64()
    } else {
        0.0
    };

    let final_metrics = fetch_model_metrics(&client, &base, &model_name)
        .await
        .map_err(|e| cli_err(format!("Cannot read metrics after generation: {e}")))?;

    print_speedtest_summary(
        &model_name,
        run_test,
        baseline.as_ref(),
        &final_metrics,
        wall,
    );
    Ok(())
}

/// Resolve the speedtest model: the arg if given, else the default route's
/// first model, else the first configured model key, else `code`.
fn resolve_speedtest_model(config: Option<&RouterConfig>, arg: &str) -> String {
    if !arg.is_empty() {
        return arg.to_string();
    }
    if let Some(cfg) = config {
        if let Some(route) = cfg.routes.get(&cfg.default_route) {
            if let Some(group) = cfg.model_groups.get(&route.group) {
                if let Some(first) = group.models().first() {
                    return first.clone();
                }
            }
        }
        let mut keys: Vec<&String> = cfg.models.keys().collect();
        keys.sort_unstable();
        if let Some(first) = keys.first() {
            return (*first).clone();
        }
    }
    "code".to_string()
}

fn speedtest_report(usage: &Value, elapsed: Duration) {
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let secs = elapsed.as_secs_f64();
    eprintln!(
        "  completion: {completion_tokens} tokens in {secs:.2}s ({:.1} tok/s), prompt {prompt_tokens} tokens",
        completion_tokens as f64 / secs.max(1e-9)
    );
}

fn print_speedtest_summary(
    model_name: &str,
    run_test: bool,
    baseline: Option<&HashMap<String, f64>>,
    final_metrics: &HashMap<String, f64>,
    wall: f64,
) {
    let prompt_avg_tps = final_metrics
        .get("llamacpp:prompt_tokens_seconds")
        .copied()
        .unwrap_or(0.0);
    let gen_avg_tps = final_metrics
        .get("llamacpp:predicted_tokens_seconds")
        .copied()
        .unwrap_or(0.0);

    println!("Speedtest · {model_name}");
    println!(
        "{:<12} {:>10} {:>10} {:>14}",
        "Phase", "Tokens", "Time (s)", "Speed (tok/s)"
    );
    if run_test {
        if let Some(b) = baseline {
            let delta = |name: &str| {
                final_metrics.get(name).copied().unwrap_or(0.0)
                    - b.get(name).copied().unwrap_or(0.0)
            };
            let tok_prompt = delta("llamacpp:prompt_tokens_total");
            let time_prompt = delta("llamacpp:prompt_seconds_total");
            let tok_gen = delta("llamacpp:tokens_predicted_total");
            let time_gen = delta("llamacpp:tokens_predicted_seconds_total");
            let tok_total = tok_prompt + tok_gen;
            println!(
                "{:<12} {:>10.0} {:>10.2} {:>14.1}",
                "Prompt",
                tok_prompt,
                time_prompt,
                fmt_speed(tok_prompt, time_prompt)
            );
            println!(
                "{:<12} {:>10.0} {:>10.2} {:>14.1}",
                "Generation",
                tok_gen,
                time_gen,
                fmt_speed(tok_gen, time_gen)
            );
            println!(
                "{:<12} {:>10.0} {:>10.2} {:>14.1}",
                "Total",
                tok_total,
                wall,
                fmt_speed(tok_total, wall)
            );
        }
    }
    println!(
        "{:<12} {:>10} {:>10} {:>14.1}",
        "Prompt (avg)",
        "—",
        "—",
        fmt_speed(prompt_avg_tps, 1.0)
    );
    println!(
        "{:<12} {:>10} {:>10} {:>14.1}",
        "Generation (avg)",
        "—",
        "—",
        fmt_speed(gen_avg_tps, 1.0)
    );
}

fn fmt_speed(tokens: f64, seconds: f64) -> f64 {
    if seconds > 0.0 && tokens > 0.0 {
        tokens / seconds
    } else {
        0.0
    }
}

/// Fetch `/metrics?model=<model>` from the router and parse the Prometheus
/// text exposition into a metric-name → value map.
async fn fetch_model_metrics(
    client: &Client,
    base: &str,
    model: &str,
) -> Result<HashMap<String, f64>, CliError> {
    let url = format!("{base}/metrics?model={}", urlencoding(model));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| cli_err(format!("metrics request failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        return Err(cli_err(format!("metrics request failed: HTTP {status}")));
    }
    let text = resp.text().await.map_err(|e| cli_err(e.to_string()))?;
    Ok(parse_prometheus_metrics(&text))
}

fn urlencoding(s: &str) -> String {
    s.replace(':', "%3A").replace('/', "%2F")
}

/// Parse a Prometheus text exposition into `{metric_name: value}`.
fn parse_prometheus_metrics(text: &str) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name_part, value_str)) = line.rsplit_once(' ') else {
            continue;
        };
        let name = name_part.split('{').next().unwrap_or(name_part).trim();
        if let Ok(value) = value_str.parse::<f64>() {
            metrics.insert(name.to_string(), value);
        }
    }
    metrics
}

// ── shared HTTP helpers ─────────────────────────────────────────────────────

async fn get_json(client: &Client, url: &str) -> Result<Value, CliError> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| cli_err(e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status();
        return Err(cli_err(format!("GET {url} -> HTTP {status}")));
    }
    resp.json().await.map_err(|e| cli_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_parse_handles_labels_and_help() {
        let text = "# HELP x total\n# TYPE x counter\nllamacpp:prompt_tokens_total 42\nllamacpp:predicted_tokens_seconds{model=\"x\"} 3.5\nnot_a_number abc\n";
        let metrics = parse_prometheus_metrics(text);
        assert_eq!(metrics.get("llamacpp:prompt_tokens_total"), Some(&42.0));
        assert_eq!(metrics.get("llamacpp:predicted_tokens_seconds"), Some(&3.5));
        assert_eq!(metrics.len(), 2);
    }

    #[tokio::test]
    async fn pull_rejects_name_without_namespace_and_tag() {
        let ctx = CliContext::new(None, true, false, false);
        let err = pull(&ctx, "nomodel", None, false).await.unwrap_err();
        assert!(err.to_string().contains("namespace/model:tag"));
    }
}
