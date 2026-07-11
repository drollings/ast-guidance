//! Subprocess helpers: `run_capture`, `run_shell_capture`, `run_command`, `CommandOutput`.

use std::process::{Command, Output};

/// Returns the platform-specific shell program and argument.
///
/// On Unix: `("sh", "-c")`. On Windows: `("cmd", "/C")`.
pub fn shell_cmd() -> (&'static str, &'static str) {
    if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    }
}

/// Captured output from a subprocess, including exit status, stdout, and stderr.
#[derive(Debug)]
pub struct CommandOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

impl CommandOutput {
    fn from_output(output: &Output) -> Self {
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        Self {
            success: output.status.success(),
            stdout,
            stderr,
        }
    }
}

/// Runs a command with the given argv and captures stdout/stderr.
///
/// Returns `Err` only if the process fails to spawn (e.g. binary not found).
/// Check `CommandOutput::success` for exit status.
pub fn run_capture(argv: &[&str]) -> std::io::Result<CommandOutput> {
    if argv.is_empty() {
        return Ok(CommandOutput {
            success: false,
            stdout: String::new(),
            stderr: String::new(),
        });
    }
    let output = Command::new(argv[0]).args(&argv[1..]).output()?;
    Ok(CommandOutput::from_output(&output))
}

/// Runs a shell command string using the platform shell (`sh -c` / `cmd /C`).
///
/// Convenience wrapper over `run_capture` that prepends the shell prefix.
pub fn run_shell_capture(command: &str) -> std::io::Result<CommandOutput> {
    let (prog, arg) = shell_cmd();
    let output = Command::new(prog).arg(arg).arg(command).output()?;
    Ok(CommandOutput::from_output(&output))
}

pub fn run_command(argv: &[&str]) -> bool {
    if argv.is_empty() {
        return false;
    }
    Command::new(argv[0])
        .args(&argv[1..])
        .status()
        .is_ok_and(|s| s.success())
}

pub fn add_unique_path(list: &mut Vec<String>, path: &str, project_root: Option<&str>) -> bool {
    if list.iter().any(|p| p == path) {
        return false;
    }
    if let Some(root) = project_root {
        if !root.is_empty() {
            let full_path = if root.ends_with('/') {
                format!("{root}{path}")
            } else {
                format!("{root}/{path}")
            };
            if !std::path::Path::new(&full_path).exists() {
                return false;
            }
        }
    }
    list.push(path.to_string());
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_command_true() {
        assert!(run_command(&["true"]));
    }

    #[test]
    fn run_command_false() {
        assert!(!run_command(&["false"]));
    }

    #[test]
    fn add_unique_path_deduplicates() {
        let mut list = Vec::new();
        assert!(add_unique_path(&mut list, "path1", None));
        assert!(!add_unique_path(&mut list, "path1", None));
        assert!(add_unique_path(&mut list, "path2", None));
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn run_command_empty_argv_returns_false() {
        assert!(!run_command(&[]));
    }

    #[test]
    fn add_unique_path_with_project_root_existing() {
        let mut list = Vec::new();
        let root = std::env::current_dir().unwrap();
        let root_str = root.to_str().unwrap();
        assert!(add_unique_path(&mut list, "src", Some(root_str)));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn add_unique_path_with_project_root_missing() {
        let mut list = Vec::new();
        assert!(!add_unique_path(
            &mut list,
            "nonexistent_path_xyz",
            Some("/tmp")
        ));
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn add_unique_path_with_project_root_trailing_slash() {
        let mut list = Vec::new();
        let root = std::env::current_dir().unwrap();
        let root_str = format!("{}/", root.to_str().unwrap());
        assert!(add_unique_path(&mut list, "src", Some(&root_str)));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn add_unique_path_with_empty_project_root() {
        let mut list = Vec::new();
        assert!(add_unique_path(&mut list, "some_path", Some("")));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn shell_cmd_returns_valid_pair() {
        let (prog, arg) = shell_cmd();
        assert!(!prog.is_empty());
        assert!(!arg.is_empty());
        if cfg!(target_os = "windows") {
            assert_eq!(prog, "cmd");
            assert_eq!(arg, "/C");
        } else {
            assert_eq!(prog, "sh");
            assert_eq!(arg, "-c");
        }
    }

    #[test]
    fn run_capture_true() {
        let result = run_capture(&["true"]).unwrap();
        assert!(result.success);
        assert!(result.stderr.is_empty());
    }

    #[test]
    fn run_capture_false() {
        let result = run_capture(&["false"]).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn run_capture_empty_argv() {
        let result = run_capture(&[]).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn run_capture_stdout() {
        let result = run_capture(&["echo", "hello"]).unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello");
    }

    #[test]
    fn run_shell_capture_echo() {
        let result = run_shell_capture("echo world").unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "world");
    }

    #[test]
    fn run_shell_capture_false() {
        let result = run_shell_capture("false").unwrap();
        assert!(!result.success);
    }
}
