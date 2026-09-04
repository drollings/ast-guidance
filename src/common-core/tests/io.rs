use common_core::io::*;
use std::fs;
use std::path::Path;


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
