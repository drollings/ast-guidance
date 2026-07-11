use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::CopilotError;

pub const CONFIG_FILE_DEFAULT: &str = "job-copilot.toml";

/// Daemon configuration. Loadable from a JSON file, overridable via CLI flags.
///
/// All fields are `pub` so the binary crate can read them directly.
/// The `bon::Builder` pattern is used for CLI-flag construction.
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[builder(start_fn = new)]
pub struct DaemonConfig {
    /// TCP listen address — must be loopback (`127.0.0.1` or `localhost`).
    #[serde(default = "default_rest_bind_addr")]
    #[builder(default = "127.0.0.1".to_string())]
    pub rest_bind_addr: String,

    /// TCP listen port for the HTTP loopback endpoint.
    #[serde(default = "default_rest_port")]
    #[builder(default = 7182)]
    pub rest_port: u16,

    /// Enable the HTTP loopback JSON-RPC endpoint.
    #[serde(default = "default_enable_rest")]
    #[builder(default = true)]
    pub enable_rest: bool,

    /// Optional bearer token for HTTP endpoint auth.
    #[serde(default)]
    pub auth_token: Option<String>,

    /// Local OpenAI-compatible LLM base URL (must be loopback).
    #[serde(default = "default_llm_url")]
    #[builder(default = "http://127.0.0.1:11434/v1".to_string())]
    pub llm_url: String,

    /// Model name to use for LLM completions.
    #[serde(default = "default_llm_model")]
    #[builder(default = "llama3".to_string())]
    pub llm_model: String,

    /// Maximum concurrent LLM requests.
    #[serde(default = "default_llm_concurrency")]
    #[builder(default = 2)]
    pub llm_concurrency: usize,

    /// LLM request timeout in milliseconds.
    #[serde(default = "default_llm_timeout_ms")]
    #[builder(default = 60_000)]
    pub llm_timeout_ms: u64,

    /// Path to the user's profile TOML file.
    #[serde(default)]
    pub profile_path: PathBuf,

    /// Optional path for the append-only JSONL audit log.
    #[serde(default)]
    pub audit_log_path: Option<PathBuf>,

    /// Maximum characters for page context sent to the LLM.
    #[serde(default = "default_max_context_field_len")]
    #[builder(default = 4096)]
    pub max_context_field_len: usize,

    /// Maximum Native Messaging frame size in bytes (default 1 MiB).
    #[serde(default = "default_max_nm_payload")]
    #[builder(default = 1_048_576)]
    pub max_nm_payload: usize,
}

fn default_rest_bind_addr() -> String {
    "127.0.0.1".to_string()
}
fn default_rest_port() -> u16 {
    7182
}
fn default_enable_rest() -> bool {
    true
}
fn default_llm_url() -> String {
    "http://127.0.0.1:11434/v1".to_string()
}
fn default_llm_model() -> String {
    "llama3".to_string()
}
fn default_llm_concurrency() -> usize {
    2
}
fn default_llm_timeout_ms() -> u64 {
    60_000
}
fn default_max_context_field_len() -> usize {
    4096
}
fn default_max_nm_payload() -> usize {
    1_048_576
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            rest_bind_addr: default_rest_bind_addr(),
            rest_port: default_rest_port(),
            enable_rest: default_enable_rest(),
            auth_token: None,
            llm_url: default_llm_url(),
            llm_model: default_llm_model(),
            llm_concurrency: default_llm_concurrency(),
            llm_timeout_ms: default_llm_timeout_ms(),
            profile_path: PathBuf::new(),
            audit_log_path: None,
            max_context_field_len: default_max_context_field_len(),
            max_nm_payload: default_max_nm_payload(),
        }
    }
}

impl DaemonConfig {
    /// Load config from a JSON file, falling back to defaults for missing fields.
    /// If the file does not exist, returns the default config.
    pub fn load(path: &std::path::Path) -> Result<Self, CopilotError> {
        Ok(common_core::config::load_json_or_default(path))
    }

    /// Validate the configuration. Returns `CopilotError::Config` on failure.
    pub fn validate(&self) -> Result<(), CopilotError> {
        if !self.rest_bind_addr.starts_with("127.0.0.1")
            && !self.rest_bind_addr.starts_with("localhost")
        {
            return Err(CopilotError::Config(format!(
                "rest_bind_addr must be loopback, got {}",
                self.rest_bind_addr
            )));
        }

        let llm_host = self
            .llm_url
            .split("://")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .and_then(|host| host.split(':').next())
            .unwrap_or("");
        if !llm_host.starts_with("127.0.0.1") && !llm_host.starts_with("localhost") {
            return Err(CopilotError::Config(format!(
                "llm_url host must be loopback, got {llm_host}"
            )));
        }

        if !self.profile_path.exists() {
            return Err(CopilotError::Context(Box::new(
                common_core::error_context::ErrorContext::new(
                    "validate_config",
                    Some("profile_path"),
                    Some(&self.profile_path.display().to_string()),
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "profile_path does not exist",
                    ),
                ),
            )));
        }

        if self.max_context_field_len == 0 || self.max_context_field_len > 65_536 {
            return Err(CopilotError::Config(format!(
                "max_context_field_len must be 1..=65536, got {}",
                self.max_context_field_len
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn validate_rejects_non_loopback_bind_addr() {
        let dir = TempDir::new().unwrap();
        let profile = dir.path().join("profile.toml");
        std::fs::write(&profile, "").unwrap();
        let config = DaemonConfig::new()
            .rest_bind_addr("0.0.0.0".to_string())
            .profile_path(profile)
            .build();
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("rest_bind_addr must be loopback"));
    }

    #[test]
    fn validate_rejects_non_loopback_llm_url() {
        let dir = TempDir::new().unwrap();
        let profile = dir.path().join("profile.toml");
        std::fs::write(&profile, "").unwrap();
        let config = DaemonConfig::new()
            .llm_url("http://10.0.0.1:11434/v1".to_string())
            .profile_path(profile)
            .build();
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("llm_url host must be loopback"));
    }

    #[test]
    fn validate_accepts_loopback_default_config() {
        let dir = TempDir::new().unwrap();
        let profile = dir.path().join("profile.toml");
        std::fs::write(&profile, "").unwrap();
        let config = DaemonConfig::new().profile_path(profile).build();
        config.validate().unwrap();
    }

    #[test]
    fn load_missing_file_returns_default() {
        let config = DaemonConfig::load(std::path::Path::new("/nonexistent/config.json")).unwrap();
        assert_eq!(config.rest_bind_addr, "127.0.0.1");
        assert_eq!(config.rest_port, 7182);
        assert!(config.enable_rest);
        assert_eq!(config.llm_concurrency, 2);
    }
}
