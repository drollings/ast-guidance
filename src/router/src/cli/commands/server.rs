//! Server admin commands (`ps`, `stop`, `speedtest`) driving a running Coral
//! Router through its HTTP API.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};

use super::{cli_err, load_config, router_base_url};
use super::SpeedtestArgs;
use crate::config::RouterConfig;
use crate::cli::gguf::{
    compute_short_id, format_keep_alive, format_size, ollama_display, read_gguf_metadata,
    resolve_model,
};
use crate::cli::{CliContext, CliError, CliResult};

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

pub(crate) fn print_weight_block(
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    model_key: &str,
    insts: &[Value],
) {
    print!("{}", render_weight_block(config, gguf_dir, model_key, insts));
}

/// Render one model's `ps` block. The llama block is byte-identical to the
/// pre-M4 layout (weights / instances / total resident, `ctx-mem` = VRAM); an
/// onnx key (rows carry `runtime: onnx`) renders the same layout with a
/// `(onnx)` marker and the RAM memory column (`ram-mem` from `total_bytes`).
pub(crate) fn render_weight_block(
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    model_key: &str,
    insts: &[Value],
) -> String {
    let mut out = String::new();
    // Weights footprint: the router's reported shared-weights bytes
    // (`model_bytes`, carried on every instance) is authoritative for a
    // managed model; fall back to the configured weights file size, then the
    // GGUF layout, for standalone/remote models with no instance detail.
    let weights_bytes = model_weights_bytes(insts, config, gguf_dir, model_key);
    // An onnx model's rows carry `runtime: onnx` (the unified `/instances`
    // envelope): its memory column is RAM (`total_bytes`) not VRAM.
    let is_onnx = insts
        .iter()
        .any(|i| i.get("runtime").and_then(Value::as_str) == Some("onnx"));
    let ctx_mem_sum: u64 = insts
        .iter()
        .map(|i| {
            let mem = if is_onnx { "total_bytes" } else { "vram_bytes" };
            i.get(mem).and_then(Value::as_u64).unwrap_or(0)
        })
        .sum();

    if is_onnx {
        // Onnx block: same layout/labels as llama except the runtime marker
        // and the RAM memory column. Added only for onnx keys.
        let _ = writeln!(out, "{}  (onnx)", ollama_display(model_key, ""));
        let _ = writeln!(out, "  weights   {:>12}", format_size(weights_bytes));
        if !insts.is_empty() {
            out.push_str("  instances\n");
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
                let mem = format_size(inst.get("total_bytes").and_then(Value::as_u64).unwrap_or(0));
                let resume = if inst.get("resume").and_then(Value::as_bool).unwrap_or(false) {
                    "resume"
                } else {
                    "-"
                };
                let _ = writeln!(
                    out,
                    "    {name:<16}  ctx {n_ctx:>7}  par {par:>3}  resume {resume:<6}  sleep {sleep:<9}  state {state:<9}  ram-mem {mem:>10}"
                );
            }
        }
        let _ = writeln!(
            out,
            "  total     {:>12} resident (weights + contexts)",
            format_size(weights_bytes.saturating_add(ctx_mem_sum))
        );
        out.push('\n');
        return out;
    }

    let gguf_file = config_weights_path(config, model_key).or_else(|| gguf_weights_path(gguf_dir, model_key));

    // Weights identity. The router overlays `short_id`/`arch`/`quant` on the
    // `/instances` rows from the file IT loaded — authoritative even when the
    // weights file isn't on the CLI's host. Fall back to local resolution
    // (config path or GGUF layout) only when the router didn't report it.
    let identity_from_insts = insts
        .iter()
        .find(|i| {
            i.get("short_id")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        });
    let (short_id, arch, quant) = if let Some(inst) = identity_from_insts {
        (
            inst.get("short_id").and_then(Value::as_str).unwrap_or("").to_string(),
            inst.get("arch").and_then(Value::as_str).unwrap_or("").to_string(),
            inst.get("quant").and_then(Value::as_str).unwrap_or("").to_string(),
        )
    } else {
        let (sid, arc) = match &gguf_file {
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
            .map_or_else(String::new, |s| s.to_string_lossy().to_lowercase());
        let quant = if tag.is_empty() || tag == "latest" {
            "-".to_string()
        } else {
            tag.clone()
        };
        (sid, arc, quant)
    };
    let display = ollama_display(model_key, &quant);

    let _ = writeln!(out, "{display}  (id={short_id}, arch={arch}, quant={quant})");
    let _ = writeln!(out, "  weights   {:>12}", format_size(weights_bytes));
    if !insts.is_empty() {
        out.push_str("  instances\n");
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
            let resume = if inst.get("resume").and_then(Value::as_bool).unwrap_or(false) {
                "resume"
            } else {
                "-"
            };
            let _ = writeln!(
                out,
                "    {name:<16}  ctx {n_ctx:>7}  par {par:>3}  resume {resume:<6}  sleep {sleep:<9}  state {state:<9}  ctx-mem {mem:>10}"
            );
        }
    }
    let _ = writeln!(
        out,
        "  total     {:>12} resident (weights + contexts)",
        format_size(weights_bytes.saturating_add(ctx_mem_sum))
    );
    out.push('\n');
    out
}

/// The configured `weights` path for a model key, if the entry declares one.
fn config_weights_path(config: Option<&RouterConfig>, model_key: &str) -> Option<PathBuf> {
    config
        .and_then(|cfg| cfg.models.get(model_key))
        .and_then(|entry| entry.weights.as_ref())
        .map(PathBuf::from)
}

/// The configured weights file size, if the file exists on this host.
fn config_weights_size(config: Option<&RouterConfig>, model_key: &str) -> Option<u64> {
    config_weights_path(config, model_key)
        .and_then(|p| std::fs::metadata(&p).ok())
        .map(|m| m.len())
        .filter(|b| *b > 0)
}

/// Resolve a model key's GGUF file from the GGUF layout:
/// `<gguf_dir>/<key>/latest.gguf` or the smallest GGUF under the directory.
fn gguf_weights_path(gguf_dir: &Path, model_key: &str) -> Option<PathBuf> {
    let dir = gguf_dir.join(model_key);
    let read = std::fs::read_dir(&dir).ok()?;
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "gguf") {
            files.push(path);
        }
    }
    files
        .iter()
        .find(|p| p.file_stem().is_some_and(|s| s == "latest"))
        .or_else(|| {
            files
                .iter()
                .min_by_key(|p| std::fs::metadata(p).map_or(u64::MAX, |m| m.len()))
        })
        .cloned()
}

/// The GGUF-layout weights file size, if a file resolves for the key.
fn gguf_weights_size(gguf_dir: &Path, model_key: &str) -> Option<u64> {
    gguf_weights_path(gguf_dir, model_key)
        .and_then(|p| std::fs::metadata(&p).ok())
        .map(|m| m.len())
        .filter(|b| *b > 0)
}

/// Resolve a model key's resident weights bytes: the router's reported
/// `model_bytes` when instance detail exists (a plain model's synthesized
/// entry reports `0` while the fork has its weights slept out of VRAM, and
/// the shared weights size once loaded), else the configured weights file,
/// else the GGUF layout, else `0`.
pub(super) fn model_weights_bytes(
    insts: &[Value],
    config: Option<&RouterConfig>,
    gguf_dir: &Path,
    model_key: &str,
) -> u64 {
    if !insts.is_empty() {
        return insts
            .iter()
            .filter_map(|i| i.get("model_bytes").and_then(Value::as_u64))
            .max()
            .unwrap_or(0);
    }
    config_weights_size(config, model_key)
        .or_else(|| gguf_weights_size(gguf_dir, model_key))
        .unwrap_or(0)
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
    if model_name.is_empty() {
        return Err(cli_err(
            "no model to benchmark: pass --model, or configure a default_route with a model group",
        ));
    }
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
/// first model, else the first configured model key. Role members fan out to
/// their head candidate so a role-first group still yields a concrete model.
fn resolve_speedtest_model(config: Option<&RouterConfig>, arg: &str) -> String {
    if !arg.is_empty() {
        return arg.to_string();
    }
    if let Some(cfg) = config {
        if let Some(route) = cfg.routes_view().get(&cfg.default_route) {
            if let Some(group) = cfg.model_groups.get(&route.group) {
                if let Some(first) = group.models().first() {
                    if let Some(role) = cfg.roles.get(first) {
                        if let Some(head) = role.models.first() {
                            return head.clone();
                        }
                    }
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
    String::new()
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
pub(super) fn parse_prometheus_metrics(text: &str) -> HashMap<String, f64> {
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