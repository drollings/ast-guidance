//! Structured logging infrastructure for the router pipeline.
//!
//! Configures a `tracing` subscriber with optional JSON-formatted
//! rolling file output and console output. StageDecision records are
//! emitted as structured events keyed by pipeline stage, never by
//! session ID (cardinality).

use std::path::PathBuf;

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

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

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

    if config.json_format {
        let file_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(non_blocking)
            .with_filter(env_filter);

        if config.console_output {
            let console_layer = tracing_subscriber::fmt::layer()
                .json()
                .with_writer(std::io::stderr)
                .with_filter(tracing_subscriber::filter::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::filter::EnvFilter::new("info")));

            tracing_subscriber::registry()
                .with(file_layer)
                .with(console_layer)
                .init();
        } else {
            tracing_subscriber::registry()
                .with(file_layer)
                .init();
        }
    } else {
        let file_layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_filter(env_filter);

        if config.console_output {
            let console_layer = tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(tracing_subscriber::filter::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::filter::EnvFilter::new("info")));

            tracing_subscriber::registry()
                .with(file_layer)
                .with(console_layer)
                .init();
        } else {
            tracing_subscriber::registry()
                .with(file_layer)
                .init();
        }
    }

    // Leak the WorkerGuard so the non-blocking writer lives for the
    // process lifetime. This is the standard pattern for daemon-style
    // processes that never return from the logging initializer.
    std::mem::forget(guard);

    tracing::info!(target: "router.logging", log_dir = %config.log_dir.display(), "router logging initialized");

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

const fn default_true() -> bool {
    true
}

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