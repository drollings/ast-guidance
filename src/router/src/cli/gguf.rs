//! GGUF directory scanning, caching, model resolution, and metadata parsing.
//!
//! Ported from `gguf_tool.py` (`scan_gguf_models`, `_list_gguf_entries`,
//! `_incremental_scan`, `needs_rescan`, `_resolve_model`, `_read_gguf_metadata`,
//! and the ollama-style display helpers). The `models.json` cache schema is
//! kept byte-compatible with `gguf_tool.py` (version 2) so the two tools can
//! share a layout without thrashing each other's cache.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use common_core::hash::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Model metadata associated with one GGUF weights file on disk.
#[derive(Debug, Clone)]
pub struct GgufEntry {
    /// Model path relative to `gguf_dir` (lowercased, `-gguf` suffix stripped),
    /// e.g. `abiray/lfm2.5-2.6b-heretic-abliterated`.
    pub name: String,
    /// Quantization tag (`""` for the default entry, else the file stem, e.g.
    /// `iq4_xs`).
    pub tag: String,
    /// Ollama-style display name (`name:tag`).
    pub display: String,
    pub path: PathBuf,
    pub size: u64,
    /// Modification time as a Unix timestamp (seconds).
    pub mtime: f64,
    pub arch: String,
    pub file_type: Option<u32>,
    pub short_id: String,
}

/// The on-disk `models.json` cache (schema-compatible with `gguf_tool.py` v2).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GgufCache {
    pub version: u32,
    #[serde(default)]
    pub newest_mtime: f64,
    #[serde(default)]
    pub parent_mtime: f64,
    #[serde(default)]
    pub model_count: usize,
    #[serde(default)]
    pub models: HashMap<String, CacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub name: String,
    #[serde(default)]
    pub tag: String,
    pub display: String,
    pub path: String,
    pub size: u64,
    pub mtime: f64,
    #[serde(default)]
    pub dir_mtime: f64,
    #[serde(default)]
    pub companion_mtime: f64,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub file_type: Option<u32>,
    pub short_id: String,
}

impl From<&CacheEntry> for GgufEntry {
    fn from(e: &CacheEntry) -> Self {
        Self {
            name: e.name.clone(),
            tag: e.tag.clone(),
            display: e.display.clone(),
            path: PathBuf::from(&e.path),
            size: e.size,
            mtime: e.mtime,
            arch: e.arch.clone(),
            file_type: e.file_type,
            short_id: e.short_id.clone(),
        }
    }
}

impl GgufEntry {
    /// The cache key (`name` for the default entry, else `name:tag`).
    pub fn cache_key(&self) -> String {
        if self.tag.is_empty() {
            self.name.clone()
        } else {
            format!("{}:{}", self.name, self.tag)
        }
    }

    /// The model's directory on disk (parent of the weights file).
    pub fn model_dir(&self) -> &Path {
        self.path.parent().unwrap_or_else(|| Path::new("."))
    }
}

/// Mapping from GGUF `general.file_type` to quantization name (llama_ftype).
pub const FTYPE_TO_QUANT: &[(u32, &str)] = &[
    (0, "f32"),
    (1, "f16"),
    (2, "q4_0"),
    (3, "q4_1"),
    (7, "q8_0"),
    (8, "q5_0"),
    (9, "q5_1"),
    (10, "q2_k"),
    (11, "q3_k_s"),
    (12, "q3_k_m"),
    (13, "q3_k_l"),
    (14, "q4_k_s"),
    (15, "q4_k_m"),
    (16, "q5_k_s"),
    (17, "q5_k_m"),
    (18, "q6_k"),
    (19, "iq2_xxs"),
    (20, "iq2_xs"),
    (21, "q2_k_s"),
    (22, "iq3_xs"),
    (23, "iq3_xxs"),
    (24, "iq1_s"),
    (25, "iq4_nl"),
    (26, "iq3_s"),
    (27, "iq3_m"),
    (28, "iq2_s"),
    (29, "iq2_m"),
    (30, "iq4_xs"),
    (31, "iq1_m"),
    (32, "bf16"),
    (36, "tq1_0"),
    (37, "tq2_0"),
];

/// Quantization name for a `general.file_type` value.
pub fn quant_name(file_type: u32) -> String {
    FTYPE_TO_QUANT
        .iter()
        .find(|(ft, _)| *ft == file_type)
        .map_or_else(|| format!("ftype_{file_type}"), |(_, name)| (*name).into())
}

/// Strip a case-insensitive `-gguf` suffix from each path component.
pub fn strip_gguf_suffix(name: &str) -> String {
    let mut parts = Vec::new();
    for part in name.split('/') {
        if part.to_ascii_lowercase().ends_with("-gguf") {
            parts.push(&part[..part.len() - "-gguf".len()]);
        } else {
            parts.push(part);
        }
    }
    parts.join("/")
}

/// Derive `(name, tag, tagged_name)` from a GGUF file's path components
/// (relative to `gguf_dir`).
pub fn model_name_and_tag(path_parts: &[String], gguf_file: &Path) -> (String, String, String) {
    let name = if path_parts.is_empty() {
        gguf_file
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().to_lowercase())
    } else {
        strip_gguf_suffix(&path_parts.join("/")).to_lowercase()
    };
    let tag = gguf_file
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().to_lowercase());
    let tagged_name = if name.is_empty() {
        tag.clone()
    } else {
        format!("{name}:{tag}")
    };
    (name, tag, tagged_name)
}

/// Convert an internal model path to an ollama-style display name. A leading
/// `library/` namespace is elided (it is the assumed default).
pub fn ollama_display(name: &str, tag: &str) -> String {
    let rest = name
        .strip_prefix("library/")
        .map_or_else(|| name.to_string(), str::to_string);
    match (rest.is_empty(), tag.is_empty()) {
        (_, true) => rest,
        (true, false) => tag.to_string(),
        (false, false) => format!("{rest}:{tag}"),
    }
}

/// Compute a short 12-hex-char model id from file size, mtime, and the first
/// 4 KiB of the file (matching `gguf_tool.py`).
pub fn compute_short_id(gguf_path: &Path) -> String {
    let Ok(stat) = std::fs::metadata(gguf_path) else {
        return "?".repeat(12);
    };
    let mut prefix = [0u8; 4096];
    let n = File::open(gguf_path)
        .and_then(|mut f| f.read(&mut prefix))
        .unwrap_or(0);
    let size = stat.len();
    let mtime = stat
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0u64, |d| d.as_secs());
    let mut data = Vec::with_capacity(n + 32);
    data.extend_from_slice(&prefix[..n]);
    data.extend_from_slice(size.to_string().as_bytes());
    data.extend_from_slice(mtime.to_string().as_bytes());
    sha256_hex(&data)[..12].to_string()
}

/// Human-readable byte size using base-1000 SI units (`gguf_tool.py` format).
pub fn format_size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1000.0 && unit < units.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}B")
    } else if (value - value.round()).abs() < f64::EPSILON {
        format!("{} {}", value.round() as u64, units[unit])
    } else {
        format!("{value:.1} {}", units[unit])
    }
}

/// Relative modification time label (`2 days ago`).
pub fn format_relative_time(mtime: f64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let delta = now - mtime;
    if delta < 0.0 {
        return "just now".into();
    }
    if delta < 60.0 {
        return format!("{} seconds", delta as u64);
    }
    if delta < 3600.0 {
        let m = (delta / 60.0) as u64;
        return if m == 1 {
            "1 minute".into()
        } else {
            format!("{m} minutes")
        };
    }
    if delta < 86_400.0 {
        let h = (delta / 3600.0) as u64;
        return if h == 1 {
            "1 hour".into()
        } else {
            format!("{h} hours")
        };
    }
    if delta < 604_800.0 {
        let d = (delta / 86_400.0) as u64;
        return if d == 1 {
            "1 day".into()
        } else {
            format!("{d} days")
        };
    }
    if delta < 2_592_000.0 {
        let w = (delta / 604_800.0) as u64;
        return if w == 1 {
            "1 week".into()
        } else {
            format!("{w} weeks")
        };
    }
    if delta < 31_536_000.0 {
        let m = (delta / 2_592_000.0) as u64;
        return if m == 1 {
            "1 month".into()
        } else {
            format!("{m} months")
        };
    }
    let y = (delta / 31_536_000.0) as u64;
    if y == 1 {
        "1 year".into()
    } else {
        format!("{y} years")
    }
}

/// Human keep-alive timeout label (`2h30m`, `Forever`, ...).
pub fn format_keep_alive(seconds: i64) -> String {
    if seconds < 0 {
        return "Forever".into();
    }
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;
    if h > 0 {
        if m > 0 {
            format!("{h}h{m:02}m")
        } else {
            format!("{h}h")
        }
    } else if m > 0 {
        format!("{m}m")
    } else {
        format!("{s}s")
    }
}

/// Read the metadata KV pairs from a GGUF header (v1-v3 compatible). Returns
/// an empty map on any parse failure.
pub fn read_gguf_metadata(path: &Path) -> HashMap<String, Value> {
    let Ok(file) = File::open(path) else {
        return HashMap::new();
    };
    let mut meta = HashMap::new();
    if parse_gguf_header(&mut BufReader::new(file), &mut meta) {
        meta
    } else {
        HashMap::new()
    }
}

/// Binary GGUF value types (llama.cpp `GGUFValueType`).
fn parse_gguf_header(r: &mut impl Read, meta: &mut HashMap<String, Value>) -> bool {
    let mut magic = [0u8; 4];
    if r.read_exact(&mut magic).is_err() || &magic != b"GGUF" {
        return false;
    }
    let Some(version) = read_u32(r) else {
        return false;
    };
    let Some(_tensor_count) = read_u64(r) else {
        return false;
    };
    let Some(kv_count) = read_u64(r) else {
        return false;
    };
    for _ in 0..kv_count {
        let Some(key) = read_string(r) else {
            return false;
        };
        let Some(vtype) = read_u32(r) else {
            return false;
        };
        let Some(value) = read_value(r, vtype) else {
            return false;
        };
        meta.insert(key, value);
    }
    meta.insert("_version".into(), Value::from(version));
    true
}

fn read_exact_n(r: &mut impl Read, n: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn read_u8(r: &mut impl Read) -> Option<u8> {
    read_exact_n(r, 1).map(|b| b[0])
}

fn read_i8(r: &mut impl Read) -> Option<i8> {
    read_exact_n(r, 1).map(|b| b[0] as i8)
}

fn read_u16(r: &mut impl Read) -> Option<u16> {
    read_exact_n(r, 2).map(|b| u16::from_le_bytes(b.try_into().unwrap()))
}

fn read_i16(r: &mut impl Read) -> Option<i16> {
    read_exact_n(r, 2).map(|b| i16::from_le_bytes(b.try_into().unwrap()))
}

fn read_u32(r: &mut impl Read) -> Option<u32> {
    read_exact_n(r, 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn read_i32(r: &mut impl Read) -> Option<i32> {
    read_exact_n(r, 4).map(|b| i32::from_le_bytes(b.try_into().unwrap()))
}

fn read_u64(r: &mut impl Read) -> Option<u64> {
    read_exact_n(r, 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
}

fn read_i64(r: &mut impl Read) -> Option<i64> {
    read_exact_n(r, 8).map(|b| i64::from_le_bytes(b.try_into().unwrap()))
}

fn read_f32(r: &mut impl Read) -> Option<f32> {
    read_exact_n(r, 4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))
}

fn read_f64(r: &mut impl Read) -> Option<f64> {
    read_exact_n(r, 8).map(|b| f64::from_le_bytes(b.try_into().unwrap()))
}

fn read_string(r: &mut impl Read) -> Option<String> {
    let len = read_u64(r)?;
    let buf = read_exact_n(r, len as usize)?;
    String::from_utf8(buf).ok()
}

/// Decode one GGUF value of `vtype` (GGUFValueType 0-12).
fn read_value(r: &mut impl Read, vtype: u32) -> Option<Value> {
    match vtype {
        0 => read_u8(r).map(Value::from),
        1 => read_i8(r).map(Value::from),
        2 => read_u16(r).map(Value::from),
        3 => read_i16(r).map(Value::from),
        4 => read_u32(r).map(Value::from),
        5 => read_i32(r).map(Value::from),
        6 => read_f32(r).map(Value::from),
        7 => read_u8(r).map(|v| Value::from(v != 0)),
        8 => read_string(r).map(Value::String),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            let mut arr = Vec::with_capacity(count.min(1 << 20) as usize);
            for _ in 0..count {
                arr.push(read_value(r, elem_type)?);
            }
            Some(Value::Array(arr))
        }
        10 => read_u64(r).map(Value::from),
        11 => read_i64(r).map(Value::from),
        12 => read_f64(r).map(Value::from),
        _ => None,
    }
}

/// The path components of a directory relative to `gguf_dir`.
fn rel_parts(dir: &Path, gguf_dir: &Path) -> Vec<String> {
    dir.strip_prefix(gguf_dir)
        .map(|rel| {
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Enumerate every GGUF weights file grouped by parent directory (excluding
/// `projector.gguf`), sorted by directory.
fn dir_gguf_files(gguf_dir: &Path) -> Vec<(PathBuf, Vec<PathBuf>)> {
    let mut dirs: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    let mut stack = vec![gguf_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "gguf")
                && path.file_stem().is_some_and(|s| s != "projector")
            {
                dirs.entry(dir.clone()).or_default().push(path);
            }
        }
    }
    let mut out: Vec<(PathBuf, Vec<PathBuf>)> = Vec::with_capacity(dirs.len());
    for (dir, files) in dirs {
        out.push((dir, files));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Pick the default GGUF: `latest.gguf` if present, else the smallest file.
fn default_gguf(gguf_files: &[PathBuf]) -> &PathBuf {
    gguf_files
        .iter()
        .find(|f| {
            f.file_stem()
                .is_some_and(|s| s.to_string_lossy().to_ascii_lowercase() == "latest")
        })
        .unwrap_or_else(|| {
            gguf_files
                .iter()
                .min_by_key(|f| std::fs::metadata(f).map_or(u64::MAX, |m| m.len()))
                .unwrap_or(&gguf_files[0])
        })
}

fn file_mtime(path: &Path) -> f64 {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs_f64())
}

fn make_entry(name: &str, tag: &str, gguf_path: &Path) -> GgufEntry {
    let stat = std::fs::metadata(gguf_path).ok();
    let meta = read_gguf_metadata(gguf_path);
    let file_type = meta
        .get("general.file_type")
        .and_then(Value::as_u64)
        .map(|v| v as u32);
    GgufEntry {
        name: name.to_string(),
        tag: tag.to_string(),
        display: ollama_display(name, tag),
        path: gguf_path.to_path_buf(),
        size: stat.as_ref().map_or(0, std::fs::Metadata::len),
        mtime: file_mtime(gguf_path),
        arch: meta
            .get("general.architecture")
            .and_then(Value::as_str)
            .map_or_else(String::new, str::to_lowercase),
        file_type,
        short_id: compute_short_id(gguf_path),
    }
}

/// Full scan of `gguf_dir`: one default (untagged) entry per model directory,
/// plus a `:tag` entry for every other GGUF file.
pub fn list_gguf_entries(gguf_dir: &Path) -> Vec<GgufEntry> {
    let mut entries = Vec::new();
    for (parent, gguf_files) in dir_gguf_files(gguf_dir) {
        let parts = rel_parts(&parent, gguf_dir);
        let (name, _, _) = model_name_and_tag(&parts, &gguf_files[0]);
        let default = default_gguf(&gguf_files);
        entries.push(make_entry(&name, "", default));
        for gguf_path in &gguf_files {
            if gguf_path == default {
                continue;
            }
            let (_, tag, _) = model_name_and_tag(&parts, gguf_path);
            entries.push(make_entry(&name, &tag, gguf_path));
        }
    }
    entries
}

/// Snapshot resolution for the cache's directory-mtime tokens (`parent_mtime`,
/// `newest_mtime`), in units of the epoch (microseconds).
///
/// The mtime is stored as an integer-valued `f64` count of microseconds.
/// Integer-valued f64s ≤ 2^53 (≈ 285 years in microseconds) round-trip through
/// `serde_json` losslessly, whereas fractional-second f64s at ~1.7e9 lose up to
/// one ULP (~238 ns) in serde_json's parser — which made `needs_rescan` see a
/// spurious change on a freshly written cache. Writer and reader must derive
/// the token through this same function so they agree on the snapshot value.
pub const MTIME_SNAPSHOT_RES: f64 = 1e6;

/// Quantize a Unix-second mtime (fractional) to the integer-valued snapshot
/// token both `write_gguf_cache` and `needs_rescan` compare against.
fn mtime_snapshot(secs: f64) -> f64 {
    (secs * MTIME_SNAPSHOT_RES).round()
}

/// Build the version-2 cache data from a full entry list.
fn cache_data_from_entries(entries: Vec<GgufEntry>) -> GgufCache {
    let newest_mtime = entries
        .iter()
        .fold(0.0f64, |acc, e| acc.max(mtime_snapshot(e.mtime)));
    let models: HashMap<String, CacheEntry> = entries
        .into_iter()
        .map(|e| {
            let key = e.cache_key();
            let entry = CacheEntry {
                name: e.name,
                tag: e.tag,
                display: e.display,
                path: e.path.to_string_lossy().into_owned(),
                size: e.size,
                mtime: e.mtime,
                dir_mtime: 0.0,
                companion_mtime: 0.0,
                arch: e.arch,
                file_type: e.file_type,
                short_id: e.short_id,
            };
            (key, entry)
        })
        .collect();
    GgufCache {
        version: 2,
        newest_mtime,
        parent_mtime: 0.0,
        model_count: models.len(),
        models,
    }
}

fn cache_path(gguf_dir: &Path) -> PathBuf {
    gguf_dir.join("models.json")
}

/// Fast O(1) staleness gate: compare `gguf_dir`'s mtime against the
/// `parent_mtime` recorded when the cache was written.
pub fn needs_rescan(gguf_dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(cache_path(gguf_dir)) else {
        return true;
    };
    let Ok(data) = serde_json::from_str::<Value>(&text) else {
        return true;
    };
    if data.get("version").and_then(Value::as_u64) != Some(2) {
        return true;
    }
    let current = mtime_snapshot(file_mtime(gguf_dir));
    if let Some(parent) = data.get("parent_mtime").and_then(Value::as_f64) {
        if parent > 0.0 {
            return current > parent;
        }
    }
    if let Some(newest) = data.get("newest_mtime").and_then(Value::as_f64) {
        return newest > 0.0 && current > newest;
    }
    true
}

/// Load the full cache, or `None` when missing/stale/wrong schema.
pub fn load_gguf_cache(gguf_dir: &Path) -> Option<GgufCache> {
    let text = std::fs::read_to_string(cache_path(gguf_dir)).ok()?;
    let cache: GgufCache = serde_json::from_str(&text).ok()?;
    if cache.version != 2 || cache.models.values().any(|e| e.short_id.is_empty()) {
        return None;
    }
    Some(cache)
}

/// Write the cache, capturing `gguf_dir`'s post-write mtime as `parent_mtime`
/// so `needs_rescan` sees a stable snapshot until the directory changes.
///
/// Writing the cache file can bump `gguf_dir`'s own mtime (asynchronously on
/// some filesystems), so the value captured immediately after the first write
/// may be stale and trigger a spurious rescan. Rewrite + recapture until the
/// parent mtime is stable across consecutive reads: once the cache file exists,
/// rewriting it no longer changes the parent directory's mtime, so the value
/// settles after at most a few passes.
pub fn write_gguf_cache(gguf_dir: &Path, cache: &mut GgufCache) {
    let mut last = 0.0f64;
    for _ in 0..8 {
        if let Ok(text) = serde_json::to_string_pretty(cache) {
            if std::fs::write(cache_path(gguf_dir), text).is_err() {
                return;
            }
        }
        let now = mtime_snapshot(file_mtime(gguf_dir));
        cache.parent_mtime = now;
        if now <= last {
            return;
        }
        last = now;
    }
}

/// Ensure the cache is fresh, rescanning when the directory changed.
pub fn sync_cache(gguf_dir: &Path) {
    if needs_rescan(gguf_dir) {
        let entries = list_gguf_entries(gguf_dir);
        let mut cache = cache_data_from_entries(entries);
        write_gguf_cache(gguf_dir, &mut cache);
    }
}

/// The model entries to operate on, refreshing the cache if stale.
pub fn cached_entries(gguf_dir: &Path) -> Vec<GgufEntry> {
    sync_cache(gguf_dir);
    match load_gguf_cache(gguf_dir) {
        Some(cache) => cache.models.values().map(GgufEntry::from).collect(),
        None => list_gguf_entries(gguf_dir),
    }
}

/// Resolve an ollama-style model name (`model`, `model:tag`, `ns/model:tag`)
/// to a GGUF entry, case-insensitively. `library/` is assumed when no matching
/// top-level namespace exists; a requested-but-missing tag falls back to the
/// default (untagged) entry.
pub fn resolve_model(gguf_dir: &Path, model_name: &str) -> Option<GgufEntry> {
    let lowered = model_name.to_lowercase();
    let (base, tag) = lowered
        .rsplit_once(':')
        .map_or((lowered.as_str(), ""), |(b, t)| (b, t));
    let entries = cached_entries(gguf_dir);
    let effective = |t: &str| -> bool { !t.is_empty() && t != "latest" };
    let library_base = format!("library/{base}");

    let direct = entries.iter().find(|e| {
        effective(tag) == effective(&e.tag) && (e.name == base || e.name == library_base)
    });
    if let Some(e) = direct {
        return Some(e.clone());
    }
    // Fallback: return the default (untagged) entry for the base name.
    let fallback = entries
        .iter()
        .find(|e| e.tag.is_empty() && (e.name == base || e.name == library_base));
    if let Some(e) = fallback {
        return Some(e.clone());
    }
    // Single-component name with no matching directory: try `library/<name>`.
    if !base.contains('/') && !gguf_dir.join(base).exists() {
        if let Some(e) = entries
            .iter()
            .find(|e| e.name == library_base && (tag.is_empty() || e.tag == tag))
        {
            return Some(e.clone());
        }
    }
    None
}

/// Scan the GGUF directory into model descriptors (used by `scan`, `show`,
/// and `rm`).
pub fn scan_gguf_models(gguf_dir: &Path) -> Vec<GgufEntry> {
    list_gguf_entries(gguf_dir)
}

/// Model names matching any of these substrings are treated as
/// embedding-only models.
pub const EMBEDDING_KEYWORDS: &[&str] = &["embed", "bert", "bge", "gte", "e5"];

/// Whether the model name or its `params.json` marks it as embedding-only.
pub fn is_embedding_model(name: &str, model_dir: &Path) -> bool {
    let lower = name.to_lowercase();
    if EMBEDDING_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        return true;
    }
    let Ok(text) = std::fs::read_to_string(model_dir.join("params.json")) else {
        return false;
    };
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v.get("embedding").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Effective keep-alive (sleep-idle-seconds) for a model from `params.json`
/// `keep_alive`, else a 1-hour default.
pub fn model_keep_alive(model_dir: &Path) -> i64 {
    const DEFAULT_KEEP_ALIVE: i64 = 3600;
    let Ok(text) = std::fs::read_to_string(model_dir.join("params.json")) else {
        return DEFAULT_KEEP_ALIVE;
    };
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| v.get("keep_alive").and_then(Value::as_i64))
        .unwrap_or(DEFAULT_KEEP_ALIVE)
}

#[cfg(test)]
mod tests {
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
}
