//! Implementation of the `coral-router` admin subcommands, ported from
//! `gguf_tool.py`.
//!
//! Filesystem commands (`list`, `scan`, `rm`, `show`, `pull`) operate on the
//! GGUF layout. Server commands (`ps`, `stop`, `speedtest`) drive a running
//! Coral Router through its HTTP API.
//!
//! Shared helpers (`router_base_url` / `load_config` / `cli_err` /
//! `sync_preset`, and the `ShowFlags` / `SpeedtestArgs` types) live here; the
//! command bodies live in [`filesystem`] and [`server`].

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

pub mod filesystem;
pub mod server;

pub use filesystem::{list, pull, rm, scan, show};
pub use server::{ps, speedtest, stop};

use std::path::Path;

use crate::config::RouterConfig;
use super::CliError;
use crate::cli::gguf::sync_cache;
use crate::cli::preset::write_models_preset;

/// Whether to show the model's modelfile and license/params/system/template.
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
pub(super) fn router_base_url(api_url: Option<&str>, config_path: Option<&Path>) -> String {
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
pub(super) fn load_config(config_path: Option<&Path>) -> Option<RouterConfig> {
    let path = config_path?;
    let mut config = common_core::config::load_json_or_default::<RouterConfig>(path);
    config.apply_defaults();
    Some(config)
}

pub(super) fn cli_err(message: impl Into<String>) -> CliError {
    CliError::new(message)
}

/// Refresh the on-disk GGUF cache from the configured models directory.
pub(super) fn sync_preset(gguf_dir: &Path) {
    sync_cache(gguf_dir);
    let prefix = gguf_dir.to_string_lossy().into_owned();
    let _ = write_models_preset(gguf_dir, Some(&prefix));
}
#[cfg(test)]
#[path = "../../../tests/cli_commands_mod.rs"]
mod tests;
