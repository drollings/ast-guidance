//! Admin CLI support for Coral Router.
//!
//! This module backs the `coral-router` binary's administration subcommands
//! (`list`, `ps`, `pull`, `scan`, `rm`, `show`, `stop`, `speedtest`), ported
//! from `gguf_tool.py`. The argument parsing lives in the binary crate; the
//! implementations live here so the GGUF-directory scanning, model
//! resolution, preset rendering, and router-API clients are unit-testable
//! inside the domain crate.
//!
//! The commands split into two families:
//!
//! - **Filesystem commands** (`list`, `scan`, `rm`, `show`, `pull`) operate on
//!   the GGUF layout: scan `gguf_dir` for `*.gguf` weights, cache the scan in
//!   `models.json` (schema-compatible with `gguf_tool.py`), and render the
//!   llama.cpp `models-preset.ini`.
//! - **Server commands** (`ps`, `stop`, `speedtest`) drive a *running* Coral
//!   Router through its HTTP API (`/v1/models`, `/instances`, `/models/unload`,
//!   `/metrics`, `/v1/chat/completions`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

pub mod commands;
pub mod gguf;
pub mod preset;

use std::path::PathBuf;

use thiserror::Error;

/// The GGUF root that mirrors `gguf_tool.py`'s default layout and the
/// `weights` paths in `env/coral-router.json`.
pub const DEFAULT_GGUF_DIR: &str = "/app/ai/models/gguf";

/// Error type for CLI command execution. A single display message is
/// sufficient — the CLI prints it and exits non-zero.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct CliError(String);

impl CliError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<reqwest::Error> for CliError {
    fn from(e: reqwest::Error) -> Self {
        Self(e.to_string())
    }
}

pub type CliResult = Result<(), CliError>;

/// Shared per-invocation state threaded through every subcommand.
#[derive(Debug, Clone)]
pub struct CliContext {
    /// The GGUF directory scanned by the filesystem commands.
    pub gguf_dir: PathBuf,
    /// Show what would be done without making changes.
    pub dry_run: bool,
    /// Extra diagnostics on stderr.
    pub verbose: bool,
    /// Backtrace-style detail on failure.
    pub debug: bool,
}

impl CliContext {
    /// Build a context with the default GGUF directory.
    pub fn new(gguf_dir: Option<PathBuf>, dry_run: bool, verbose: bool, debug: bool) -> Self {
        Self {
            gguf_dir: gguf_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF_DIR)),
            dry_run,
            verbose,
            debug,
        }
    }

    /// Log a verbose diagnostic line to stderr.
    pub fn log_debug(&self, message: &str) {
        if self.verbose || self.debug {
            eprintln!("[debug] {message}");
        }
    }
}
