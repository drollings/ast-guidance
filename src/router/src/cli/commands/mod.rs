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
mod tests {
    use super::*;
    use super::server::{model_weights_bytes, parse_prometheus_metrics};
    use serde_json::json;
    use crate::cli::CliContext;

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

    #[test]
    fn ps_weights_prefer_router_model_bytes_over_config_and_gguf() {
        let dir = tempfile::tempdir().unwrap();
        // A config weights file that exists on disk (would be the old answer).
        let model_dir = dir.path().join("swarm");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("latest.gguf"), vec![0u8; 4096]).unwrap();
        let config: RouterConfig = serde_json::from_value(json!({
            "models": {
                "swarm": {
                    "endpoint": "http://127.0.0.1:1/v1/chat/completions",
                    "name": "abiray/test",
                    "weights": model_dir.join("latest.gguf").to_string_lossy(),
                    "intelligence": 2,
                    "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7,
                    "speed": 8
                }
            }
        }))
        .unwrap();

        // No instance detail: falls back to the config weights file.
        assert_eq!(
            model_weights_bytes(&[], Some(&config), dir.path(), "swarm"),
            4096
        );
        // Router-reported model_bytes wins over the config file size.
        let insts = vec![json!({ "model_bytes": 1_000_000_000u64, "vram_bytes": 100 })];
        assert_eq!(
            model_weights_bytes(&insts, Some(&config), dir.path(), "swarm"),
            1_000_000_000
        );
        // A sleeping plain model reports model_bytes = 0: its weights are NOT
        // resident, so 0 is returned - never the on-disk weights file size.
        let sleeping = vec![json!({ "model_bytes": 0u64, "vram_bytes": 0 })];
        assert_eq!(
            model_weights_bytes(&sleeping, Some(&config), dir.path(), "swarm"),
            0,
            "sleeping weights are not resident"
        );
        // No config, no instance → GGUF layout still resolves the file.
        assert_eq!(
            model_weights_bytes(&[], None, dir.path(), "swarm"),
            4096
        );
        // Nothing at all → 0 (never crashes).
        assert_eq!(model_weights_bytes(&[], None, dir.path(), "absent"), 0);
    }
}
