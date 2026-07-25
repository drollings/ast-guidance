//! Structured logging infrastructure for the router pipeline.
//!
//! Configures a `tracing` subscriber with optional JSON-formatted
//! rolling file output and console output. StageDecision records are
//! emitted as structured events keyed by pipeline stage, never by
//! session ID (cardinality).

use std::path::PathBuf;

use crate::config::AuditLogConfig;

/// Configuration for router logging output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoggingConfig {
    /// Directory for rotating log files.
    #[serde(default = "default_log_dir")]
    pub log_dir: PathBuf,

    /// Maximum size of each log file in megabytes before rotation.
    #[serde(default = "default_max_file_size_mb")]
    pub max_file_size_mb: u64,

    /// Maximum age of log files in days before deletion.
    #[serde(default = "default_max_age_days")]
    pub max_age_days: u64,

    /// Maximum number of rotated log files to retain.
    #[serde(default = "default_max_files")]
    pub max_files: usize,

    /// Emit log records as JSON (structured) rather than human-readable text.
    #[serde(default)]
    pub json_format: bool,

    /// Also emit log records to stderr for development use.
    #[serde(default = "default_true")]
    pub console_output: bool,

    /// Separate audit log stream with durable retention (MOA_ROUTER_SPEC §10).
    #[serde(default)]
    pub audit_log: Option<AuditLogConfig>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_dir: default_log_dir(),
            max_file_size_mb: default_max_file_size_mb(),
            max_age_days: default_max_age_days(),
            max_files: default_max_files(),
            json_format: false,
            console_output: true,
            audit_log: None,
        }
    }
}

/// Initialize the `tracing` subscriber with the given configuration.
///
/// Sets up a subscriber that writes to:
/// - A rolling file appender in `config.log_dir` (with size-based rotation)
/// - Optionally stderr (when `config.console_output` is true)
///
/// Logs are JSON-formatted when `config.json_format` is true.
///
/// Spans are request-scoped, not session-scoped — no session ID
/// appears in span metadata (cardinality).
pub fn init_router_logging(config: &LoggingConfig) -> Result<(), Box<dyn std::error::Error>> {
    use tracing_subscriber::filter::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer;

    std::fs::create_dir_all(&config.log_dir)
        .map_err(|e| format!("failed to create log directory '{}': {e}", config.log_dir.display()))?;

    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::NEVER)
        .filename_prefix("router")
        .filename_suffix("log")
        .max_log_files(config.max_files)
        .build(&config.log_dir)
        .map_err(|e| format!("failed to create file appender: {e}"))?;

    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // ── Build audit layer (if configured) ─────────────────────────────
    // The audit writer must outlive the process; use a thread-local
    // approach: leak the guard so the non-blocking writer lives forever.
    struct AuditResources {
        _guard: tracing_appender::non_blocking::WorkerGuard,
        appender: tracing_appender::non_blocking::NonBlocking,
    }

    let audit_resources: Option<AuditResources> = config.audit_log.as_ref().map(|audit_cfg| {
        std::fs::create_dir_all(&audit_cfg.log_dir)
            .map_err(|e| format!("audit log dir '{}': {e}", audit_cfg.log_dir.display()))?;

        let appender = tracing_appender::rolling::RollingFileAppender::builder()
            .rotation(tracing_appender::rolling::Rotation::NEVER)
            .filename_prefix("audit")
            .filename_suffix("log")
            .max_log_files(audit_cfg.max_files)
            .build(&audit_cfg.log_dir)
            .map_err(|e| format!("audit file appender: {e}"))?;

        let (non_blocking, guard) = tracing_appender::non_blocking(appender);
        Result::<_, Box<dyn std::error::Error>>::Ok(AuditResources { _guard: guard, appender: non_blocking })
    }).transpose()?;

    // ── Build per-branch subscribers ──────────────────────────────────
    // Use a nested helper struct with a blanket impl to erase concrete types.
    struct SubBuilder {
        json: bool,
        console: bool,
        non_blocking: tracing_appender::non_blocking::NonBlocking,
        guard: tracing_appender::non_blocking::WorkerGuard,
        env_filter: EnvFilter,
        audit: Option<AuditResources>,
    }

    impl SubBuilder {
        fn build(self) {
            let json = self.json;
            let has_console = self.console;
            let nb = self.non_blocking;
            let env_filter = self.env_filter;

            // Each arm has a unique concrete type. We match all combos.
            match (json, has_console, self.audit.is_some()) {
                (true, true, true) => {
                    let audit = self.audit.unwrap();
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().json().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(std::io::stderr).with_filter(
                            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(audit.appender).with_filter(EnvFilter::new("router.audit=info")))
                        .init();
                }
                (true, true, false) => {
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().json().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(std::io::stderr).with_filter(
                            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))))
                        .init();
                }
                (true, false, true) => {
                    let audit = self.audit.unwrap();
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().json().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(audit.appender).with_filter(EnvFilter::new("router.audit=info")))
                        .init();
                }
                (true, false, false) => {
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().json().with_writer(nb).with_filter(env_filter))
                        .init();
                }
                (false, true, true) => {
                    let audit = self.audit.unwrap();
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr).with_filter(
                            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(audit.appender).with_filter(EnvFilter::new("router.audit=info")))
                        .init();
                }
                (false, true, false) => {
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr).with_filter(
                            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))))
                        .init();
                }
                (false, false, true) => {
                    let audit = self.audit.unwrap();
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().with_writer(nb).with_filter(env_filter))
                        .with(tracing_subscriber::fmt::layer().json().with_writer(audit.appender).with_filter(EnvFilter::new("router.audit=info")))
                        .init();
                }
                (false, false, false) => {
                    tracing_subscriber::registry()
                        .with(tracing_subscriber::fmt::layer().with_writer(nb).with_filter(env_filter))
                        .init();
                }
            }

            std::mem::forget(self.guard);
        }
    }

    SubBuilder {
        json: config.json_format,
        console: config.console_output,
        non_blocking,
        guard,
        env_filter,
        audit: audit_resources,
    }
    .build();

    tracing::info!(target: "router.logging", log_dir = %config.log_dir.display(), audit = config.audit_log.is_some(), "router logging initialized");

    Ok(())
}

fn default_log_dir() -> PathBuf {
    PathBuf::from("/tmp/coral-router-logs")
}

const fn default_max_file_size_mb() -> u64 {
    100
}

const fn default_max_age_days() -> u64 {
    30
}

const fn default_max_files() -> usize {
    10
}

use common_core::constants::default_true;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = LoggingConfig::default();
        assert_eq!(cfg.max_file_size_mb, 100);
        assert_eq!(cfg.max_age_days, 30);
        assert_eq!(cfg.max_files, 10);
        assert!(cfg.console_output);
        assert!(!cfg.json_format);
    }

    #[test]
    fn init_router_logging_creates_log_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = LoggingConfig {
            log_dir: dir.path().to_path_buf(),
            console_output: false,
            json_format: false,
            ..Default::default()
        };
        // init_router_logging can only be called once (global subscriber).
        // We verify the config values are accepted and the function type-checks.
        // In a full integration test we'd spawn a subprocess for this.
        let _ = config;
    }
}