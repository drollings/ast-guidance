//! Editor interaction utilities for human-in-the-loop commit message editing.
//!
//! Writes content to a temp file, opens the user's `$EDITOR`, detects whether
//! the file was modified (mtime comparison), and reads the cleaned result.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Writes `content` to a temporary file with the given prefix.
///
/// The file is created in the system temp directory (e.g. `/tmp`).
/// Returns the path to the created file.
pub fn write_temp_file(content: &str, prefix: &str) -> std::io::Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    path.push(format!("{prefix}{timestamp}.txt"));

    let mut file = std::fs::File::create(&path)?;
    file.write_all(content.as_bytes())?;
    file.write_all(b"\n\n# Lines starting with '#' will be ignored.\n")?;
    file.write_all(b"# Edit the commit message above. Save and close to commit.\n")?;
    file.flush()?;

    Ok(path)
}

/// Opens the user's editor on the given file path.
///
/// Resolves `$EDITOR`, then `$VISUAL`, then falls back to `vi`.
/// Blocks until the editor exits.
///
/// Note: This intentionally uses `std::process::Command` directly (not
/// `common_core::shell::run_capture`) because the editor requires stdin
/// inheritance for interactive use — `run_capture` would capture stdin and
/// block the editor from receiving user input.
pub fn open_editor(path: &Path) -> std::io::Result<()> {
    let editor = resolve_editor();
    let status = std::process::Command::new(&editor).arg(path).status()?;
    if !status.success() {
        eprintln!("Editor {editor} exited with status {status}");
    }
    Ok(())
}

/// Returns the modification time of a file, or `None` if the file doesn't exist.
pub fn file_mtime(path: &Path) -> Option<SystemTime> {
    common_core::io::mtime(path)
}

/// Reads a file, strips comment lines (starting with `#`), joins, and trims.
///
/// Returns the cleaned string. Returns an empty string if the result is only
/// whitespace.
pub fn read_cleaned(path: &Path) -> std::io::Result<String> {
    // kept: see editor.rs:36-39 rationale for stdin/utf8 handling
    let raw = common_core::io::read_to_string_err(path).map_err(|e| e.0)?;
    let cleaned: String = raw
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(cleaned.trim().to_string())
}

/// Removes a temporary file, ignoring errors.
pub fn cleanup_temp(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn resolve_editor() -> String {
    if let Ok(editor) = std::env::var("EDITOR") {
        if !editor.is_empty() {
            return editor;
        }
    }
    if let Ok(visual) = std::env::var("VISUAL") {
        if !visual.is_empty() {
            return visual;
        }
    }
    "vi".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr_testutil::tempdir;

    #[test]
    fn test_read_cleaned_strips_comments() {
        let dir = tempdir();
        let path = dir.path().join("test.txt");
        std::fs::write(
            &path,
            "* first change\n# this is a comment\n* second change\n\n",
        )
        .expect("write");

        let result = read_cleaned(&path).expect("read");
        assert_eq!(result, "* first change\n* second change");
    }

    #[test]
    fn test_read_cleaned_empty_after_strip() {
        let dir = tempdir();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "# only comments\n# another\n").expect("write");

        let result = read_cleaned(&path).expect("read");
        assert!(result.is_empty());
    }

    #[test]
    fn test_file_mtime_exists() {
        let dir = tempdir();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").expect("write");

        assert!(file_mtime(&path).is_some());
    }

    #[test]
    fn test_file_mtime_missing() {
        let dir = tempdir();
        let path = dir.path().join("nonexistent.txt");
        assert!(file_mtime(&path).is_none());
    }

    #[test]
    fn test_write_temp_file() {
        let path = write_temp_file("test content", "test_prefix_").expect("write");
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.starts_with("test content"));
        cleanup_temp(&path);
        assert!(!path.exists());
    }
}
