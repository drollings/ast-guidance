//! Git operations — thin wrappers around `git` CLI for diff, commit, and rev-parse.
//!
//! All functions take a workspace `&Path` and run `git` in that directory.
//! Returns `Result` types instead of exiting on failure.

use std::path::Path;
use std::process::Command;

#[derive(Debug)]
pub enum GitError {
    /// The `git` binary was not found or failed to execute.
    Io(std::io::Error),
    /// `git` exited with a non-zero status.
    NonZeroExit { stderr: String },
}

impl std::fmt::Display for GitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "git io error: {e}"),
            Self::NonZeroExit { stderr } => write!(f, "git error: {stderr}"),
        }
    }
}

impl std::error::Error for GitError {}

impl From<std::io::Error> for GitError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Returns the staged diff (`git diff --staged`).
pub fn diff_staged(workspace: &Path) -> Result<String, GitError> {
    let output = Command::new("git")
        .args(["diff", "--staged"])
        .current_dir(workspace)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(GitError::NonZeroExit { stderr });
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Creates a git commit with the given message.
///
/// Returns `Ok(true)` on success, `Ok(false)` if git exited non-zero
/// (e.g. nothing to commit), or `Err` on I/O failure.
pub fn commit(workspace: &Path, message: &str) -> Result<bool, GitError> {
    let status = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(workspace)
        .status()?;

    Ok(status.success())
}

/// Returns the current HEAD commit hash (short or full, depending on git config).
pub fn rev_parse_head(workspace: &Path) -> Result<String, GitError> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(GitError::NonZeroExit { stderr });
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

