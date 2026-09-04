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

