//! Directory walker: `walk_files` (callback), `should_skip_dir`, `SOURCE_EXTENSIONS`.

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

/// Source file extensions recognized by the guidance pipeline.
pub const SOURCE_EXTENSIONS: &[&str] = &["zig", "zon", "py", "rs", "md"];

/// Default directories to skip during recursive walks.
const DEFAULT_SKIP: &[&str] = &["target", "fixtures"];

/// Recursively walk `root`, calling `callback` for each file whose extension
/// matches one of `extensions`.  Hidden directories (starting with `.`),
/// `target/`, and `fixtures/` are always skipped.  Additional directories can
/// be skipped via [`should_skip_dir`].
pub fn walk_files<F>(root: &Path, extensions: &[&str], mut callback: F)
where
    F: FnMut(&Path),
{
    let ext_set: HashSet<&str> = extensions.iter().copied().collect();
    let mut skip: HashSet<&str> = DEFAULT_SKIP.iter().copied().collect();
    walk_recursive(root, &ext_set, &mut skip, &mut callback);
}

fn walk_recursive<F>(
    dir: &Path,
    extensions: &HashSet<&str>,
    skip: &mut HashSet<&str>,
    callback: &mut F,
) where
    F: FnMut(&Path),
{
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if should_skip_dir(&path, skip) {
                continue;
            }
            walk_recursive(&path, extensions, skip, callback);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if extensions.contains(ext) {
                callback(&path);
            }
        }
    }
}

/// Returns `true` if `path` is a hidden directory (starts with `.`) or its
/// file name is in the `extra` skip set.
pub fn should_skip_dir<S: BuildHasher>(path: &Path, extra: &HashSet<&str, S>) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return true;
    };
    name.starts_with('.') || extra.contains(name)
}

/// Collect every file extension found under `dirs`, returned as `[".ext", …]`.
/// Recurses into non-hidden, non-build directories.
pub fn collect_extensions(dirs: &[PathBuf]) -> HashSet<String> {
    let mut exts = HashSet::new();
    for dir in dirs {
        collect_ext_recursive(dir, &mut exts);
    }
    exts
}

fn collect_ext_recursive(dir: &Path, exts: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if should_skip_dir(&path, &HashSet::new()) {
                continue;
            }
            collect_ext_recursive(&path, exts);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            exts.insert(format!(".{ext}"));
        }
    }
}

/// Test helper: create a directory tree with the given files and directories.
///
/// Public so the cross-crate test-support crate (`fluent-wvr-testutil`) can
/// delegate to it — the single implementation of tree scaffolding.
pub fn make_tree(root: &Path, files: &[&str], dirs: &[&str]) {
    for d in dirs {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    for f in files {
        let p = root.join(f);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, "").unwrap();
    }
}

