use std::fs;
use std::path::Path;
use std::time::SystemTime;

use crate::constants::MAX_FILE_SIZE;
use crate::error::IoError;

pub fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

/// Atomically write data to a file by writing to a temp file and renaming.
///
/// This prevents partial writes if the process is interrupted mid-write.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use common_core::io::write_atomic;
///
/// write_atomic(Path::new("config.json"), b"{\"key\": \"value\"}")?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a file to a string, returning `IoError` for I/O failures and files
/// exceeding the 100 MiB size cap.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use common_core::io::read_to_string_err;
///
/// let content = read_to_string_err(Path::new("Cargo.toml"))?;
/// assert!(content.contains("[package]"));
/// # Ok::<(), common_core::error::IoError>(())
/// ```
pub fn read_to_string_err(path: &Path) -> Result<String, IoError> {
    let meta = fs::metadata(path).map_err(IoError)?;
    let size = meta.len() as usize;
    if size > MAX_FILE_SIZE {
        return Err(IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("file too large: {size} bytes exceeds maximum {MAX_FILE_SIZE} bytes"),
        )));
    }
    fs::read_to_string(path).map_err(IoError)
}

pub fn make_path_absolute(path: &str) -> std::io::Result<String> {
    let pb = Path::new(path);
    let abs = if pb.is_absolute() {
        pb.to_path_buf()
    } else {
        std::env::current_dir()?.join(pb)
    };
    fs::create_dir_all(&abs)?;
    Ok(abs.to_string_lossy().to_string())
}

pub fn read_file_alloc(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

pub fn read_file_alloc_err(path: &str) -> Result<String, std::io::Error> {
    fs::read_to_string(path)
}

pub fn resolve_path(base: &str, relative: &str) -> String {
    if relative == "." {
        let base_path = Path::new(base);
        return base_path
            .parent()
            .unwrap_or(base_path)
            .to_string_lossy()
            .to_string();
    }
    let base_path = Path::new(base);
    if base_path.is_absolute() {
        let joined = base_path.parent().unwrap_or(base_path).join(relative);
        return joined.to_string_lossy().to_string();
    }
    let rel_path = Path::new(relative);
    if rel_path.is_absolute() {
        return relative.to_string();
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let joined = cwd.join(base).parent().unwrap_or(&cwd).join(relative);
    joined.to_string_lossy().to_string()
}

pub fn strip_path_prefix<'a>(path: &'a str, prefix: &str) -> &'a str {
    if let Some(stripped) = path.strip_prefix(prefix) {
        stripped.trim_start_matches('/')
    } else {
        path
    }
}

/// Idempotent directory creation with canonical error wrapping.
/// Replaces the 10+ ad-hoc `std::fs::create_dir_all(...)` calls across the workspace.
pub fn ensure_dir(path: impl AsRef<Path>) -> Result<(), IoError> {
    std::fs::create_dir_all(path.as_ref()).map_err(IoError)
}

/// Same as `ensure_dir` but `.expect`s on failure (for test setup and config init).
pub fn ensure_dir_or_panic(path: impl AsRef<Path>) {
    std::fs::create_dir_all(path.as_ref()).expect("ensure_dir");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn read_file_alloc_returns_none_for_nonexistent() {
        assert!(read_file_alloc("/nonexistent/path/file.txt").is_none());
    }

    #[test]
    fn read_file_alloc_reads_content() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello").unwrap();
        assert_eq!(
            read_file_alloc(&path.to_string_lossy()),
            Some("hello".into())
        );
    }

    #[test]
    fn make_path_absolute_creates_nested_dirs() {
        let dir = TempDir::new().unwrap();
        let rel = "a/b/c";
        let abs = make_path_absolute(&format!("{}/{}", dir.path().to_string_lossy(), rel)).unwrap();
        assert!(Path::new(&abs).exists());
    }

    #[test]
    fn make_path_absolute_idempotent() {
        let dir = TempDir::new().unwrap();
        let abs = dir.path().join("test").to_string_lossy().to_string();
        let result = make_path_absolute(&abs).unwrap();
        assert_eq!(result, abs);
    }

    #[test]
    fn resolve_path_absolute_unchanged() {
        let result = resolve_path("/base/dir", "/other/path");
        assert_eq!(result, "/other/path");
    }

    #[test]
    fn resolve_path_dot_returns_base() {
        let result = resolve_path("/base/dir/file.txt", ".");
        assert_eq!(result, "/base/dir");
    }

    #[test]
    fn resolve_path_joins_relative() {
        let result = resolve_path("/base/dir/file.txt", "sub/file.rs");
        assert_eq!(result, "/base/dir/sub/file.rs");
    }

    #[test]
    fn resolve_path_relative_base() {
        let result = resolve_path("relative/base/file.txt", "sub/file.rs");
        assert!(result.ends_with("sub/file.rs"));
    }

    #[test]
    fn strip_path_prefix_basic() {
        assert_eq!(strip_path_prefix("/a/b/c", "/a/b"), "c");
        assert_eq!(strip_path_prefix("/a/b/c", "/x"), "/a/b/c");
        assert_eq!(strip_path_prefix("/a/b/c", "/a/b/c"), "");
    }

    #[test]
    fn read_file_alloc_err_missing() {
        let result = read_file_alloc_err("/nonexistent/path/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn mtime_returns_some_for_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello").unwrap();
        assert!(mtime(&path).is_some());
    }

    #[test]
    fn mtime_returns_none_for_nonexistent() {
        assert!(mtime(Path::new("/nonexistent/file.txt")).is_none());
    }

    #[test]
    fn write_atomic_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        write_atomic(&path, b"hello").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new");
    }

    #[test]
    fn read_to_string_err_reads_content() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello").unwrap();
        assert_eq!(read_to_string_err(&path).unwrap(), "hello");
    }

    #[test]
    fn read_to_string_err_missing_file() {
        assert!(read_to_string_err(Path::new("/nonexistent/file.txt")).is_err());
    }

    #[test]
    fn ensure_dir_creates_nested() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a/b/c");
        ensure_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_dir_idempotent() {
        let dir = TempDir::new().unwrap();
        ensure_dir(dir.path()).unwrap();
        ensure_dir(dir.path()).unwrap();
    }
}
