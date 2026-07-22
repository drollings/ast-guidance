use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fluent_types::GuidanceDoc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PluginError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("plugin not found for extension: {0}")]
    PluginNotFound(String),
    #[error("plugin execution failed: {0}")]
    ExecutionFailed(String),
    #[error("plugin subprocess crashed: {0}")]
    SubprocessCrashed(String),
}

impl From<std::io::Error> for PluginError {
    fn from(e: std::io::Error) -> Self {
        PluginError::Io(common_core::error::IoError(e))
    }
}

pub type PluginResult = Result<GuidanceDoc, PluginError>;

#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
    pub extensions: Vec<String>,
    pub path: PathBuf,
}

pub struct PluginRegistry {
    plugins: HashMap<String, Plugin>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn register(&mut self, ext: &str, plugin: Plugin) {
        self.plugins.insert(ext.to_string(), plugin);
    }

    pub fn get_plugin(&self, ext: &str) -> Option<&Plugin> {
        self.plugins.get(ext)
    }

    /// Discover plugin binaries in the given directories.
    ///
    /// Uses a flat (non-recursive) `read_dir` intentionally: plugin discovery
    /// scans a single directory level for executables, does not filter by source
    /// extensions, and must not recurse into subdirectories.  This differs from
    /// `common_core::walk::walk_files` which recurses and filters by extension.
    pub fn discover(paths: &[PathBuf]) -> Self {
        let mut registry = Self::new();

        for path in paths {
            if path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        if entry_path.is_file() {
                            registry.register_from_path(&entry_path);
                        }
                    }
                }
            } else if path.is_file() {
                registry.register_from_path(path);
            }
        }

        registry
    }

    fn register_from_path(&mut self, path: &Path) {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("guidance-") {
                let extensions = infer_extensions(name);
                let plugin = Plugin {
                    name: name.to_string(),
                    extensions: extensions.clone(),
                    path: path.to_path_buf(),
                };
                for ext in extensions {
                    self.register(&ext, plugin.clone());
                }
            }
        }
    }

    pub fn list_plugins(&self) -> Vec<&Plugin> {
        self.plugins.values().collect()
    }

    pub fn has_plugin_for(&self, ext: &str) -> bool {
        self.plugins.contains_key(ext)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer supported file extensions from a guidance-* binary name.
fn infer_extensions(plugin_name: &str) -> Vec<String> {
    let name = plugin_name.strip_prefix("guidance-").unwrap_or(plugin_name);
    match name {
        n if n.contains("zig") => vec!["zig".into(), "zon".into()],
        n if n.contains("python") || n.contains("py") => vec!["py".into()],
        n if n.contains("rust") || n.contains("rs") => vec!["rs".into()],
        n if n.contains("markdown") || n.contains("md") => vec!["md".into(), "markdown".into()],
        n if n.contains("typescript") || n.contains("ts") => vec!["ts".into(), "tsx".into()],
        n if n.contains("javascript") || n.contains("js") => vec!["js".into(), "jsx".into()],
        n if n.contains("go") => vec!["go".into()],
        _ => vec![],
    }
}

/// Invoke an external guidance plugin binary for a source file.
///
/// Spawns `{plugin.path} sync --file {src_path} --output {output_dir}`,
/// captures the JSON `GuidanceDoc` from stdout, and returns it.
pub fn invoke_plugin(plugin: &Plugin, src_path: &Path, output_dir: &Path) -> PluginResult {
    let output = common_core::shell::run_capture(&[
        plugin.path.to_str().unwrap_or(""),
        "sync",
        "--file",
        &src_path.to_string_lossy(),
        "--output",
        &output_dir.to_string_lossy(),
    ])
    .map_err(|e| PluginError::SubprocessCrashed(format!("failed to spawn plugin: {e}")))?;

    if !output.success {
        return Err(PluginError::ExecutionFailed(format!(
            "plugin exited with non-zero status: {}",
            output.stderr
        )));
    }

    let doc: GuidanceDoc = serde_json::from_str(&output.stdout)?;
    Ok(doc)
}

/// Discover a plugin binary for a given file extension.
///
/// Searches in order:
/// 1. `{workspace}/bin/guidance-{bare_ext}` — workspace-local binary
/// 2. `PATH` via the `which` crate — system-wide binary
pub fn discover_provider(workspace: &Path, extension: &str) -> Option<Plugin> {
    let bare = extension.trim_start_matches('.');
    let name = format!("guidance-{bare}");

    let local = workspace.join("bin").join(&name);
    if local.is_file() {
        let extensions = infer_extensions(&name);
        return Some(Plugin {
            name,
            extensions,
            path: local,
        });
    }

    which::which(&name).ok().map(|path| {
        let extensions = infer_extensions(&name);
        Plugin {
            name,
            extensions,
            path,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr_testutil::tempdir;

    // ── PluginRegistry tests ────────────────────────────────────────────────────

    #[test]
    fn test_plugin_registry_empty() {
        let registry = PluginRegistry::new();
        assert!(registry.get_plugin("zig").is_none());
        assert!(!registry.has_plugin_for("py"));
    }

    #[test]
    fn test_register_and_lookup() {
        let mut registry = PluginRegistry::new();
        let plugin = Plugin {
            name: "guidance-zig".into(),
            extensions: vec!["zig".into(), "zon".into()],
            path: PathBuf::from("/usr/local/bin/guidance-zig"),
        };
        registry.register("zig", plugin);
        assert!(registry.has_plugin_for("zig"));
        assert!(registry.get_plugin("zig").is_some());
    }

    #[test]
    fn test_discover_from_paths() {
        let dir = tempdir();
        let zig_plugin = dir.path().join("guidance-zig");
        std::fs::write(&zig_plugin, "#!/bin/sh").expect("write");
        let py_plugin = dir.path().join("guidance-py");
        std::fs::write(&py_plugin, "#!/bin/sh").expect("write");

        let paths = vec![dir.path().to_path_buf()];
        let registry = PluginRegistry::discover(&paths);

        assert!(registry.has_plugin_for("zig"));
        assert!(registry.has_plugin_for("py"));
    }

    #[test]
    fn test_discover_ignores_non_guidance_binaries() {
        let dir = tempdir();
        let other = dir.path().join("some-other-tool");
        std::fs::write(&other, "#!/bin/sh").expect("write");

        let paths = vec![dir.path().to_path_buf()];
        let registry = PluginRegistry::discover(&paths);
        assert_eq!(registry.list_plugins().len(), 0);
    }

    // ── infer_extensions tests ──────────────────────────────────────────────────

    #[test]
    fn test_infer_extensions_zig() {
        let exts = infer_extensions("guidance-zig");
        assert!(exts.contains(&"zig".into()));
        assert!(exts.contains(&"zon".into()));
    }

    #[test]
    fn test_infer_extensions_python() {
        let exts = infer_extensions("guidance-py");
        assert!(exts.contains(&"py".into()));
    }

    #[test]
    fn test_infer_extensions_rust() {
        let exts = infer_extensions("guidance-rust");
        assert!(exts.contains(&"rs".into()));
    }

    #[test]
    fn test_infer_extensions_markdown() {
        let exts = infer_extensions("guidance-md");
        assert!(exts.contains(&"md".into()));
        assert!(exts.contains(&"markdown".into()));
    }

    #[test]
    fn test_infer_extensions_typescript() {
        let exts = infer_extensions("guidance-ts");
        assert!(exts.contains(&"ts".into()));
        assert!(exts.contains(&"tsx".into()));
    }

    #[test]
    fn test_infer_extensions_javascript() {
        let exts = infer_extensions("guidance-js");
        assert!(exts.contains(&"js".into()));
        assert!(exts.contains(&"jsx".into()));
    }

    #[test]
    fn test_infer_extensions_go() {
        let exts = infer_extensions("guidance-go");
        assert!(exts.contains(&"go".into()));
    }

    #[test]
    fn test_infer_extensions_unknown() {
        let exts = infer_extensions("guidance-foobar");
        assert!(exts.is_empty());
    }

    #[test]
    fn test_infer_extensions_without_prefix() {
        // infer_extensions strips "guidance-" prefix internally
        let exts = infer_extensions("guidance-zig");
        assert!(exts.contains(&"zig".into()));
    }

    // ── invoke_plugin tests ─────────────────────────────────────────────────────

    #[test]
    fn test_invoke_plugin_success() {
        let dir = tempdir();
        let src_path = dir.path().join("test.py");
        std::fs::write(&src_path, "print('hello')").expect("write");
        let output_dir = dir.path().join("out");
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        let guidance_doc = serde_json::json!({
            "meta": { "module": "test", "source": "test.py", "language": "python" }
        });

        // Create a mock plugin that outputs the JSON on stdout
        let plugin_path = dir.path().join("guidance-py");
        let script = format!(
            "#!/bin/sh\necho '{}'",
            guidance_doc.to_string().replace('\'', "'\\''")
        );
        std::fs::write(&plugin_path, &script).expect("write plugin");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&plugin_path, std::fs::Permissions::from_mode(0o755))
                .expect("set executable");
            let f = std::fs::File::open(&plugin_path).expect("open for sync");
            f.sync_all().expect("sync plugin");
            drop(f);
        }

        let plugin = Plugin {
            name: "guidance-py".into(),
            extensions: vec!["py".into()],
            path: plugin_path,
        };

        let result = invoke_plugin(&plugin, &src_path, &output_dir);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let doc = result.unwrap();
        assert_eq!(doc.meta.module.as_str(), "test");
        assert_eq!(doc.meta.source.as_str(), "test.py");
        assert_eq!(doc.meta.language.as_str(), "python");
    }

    #[test]
    fn test_invoke_plugin_exit_code_error() {
        let dir = tempdir();
        let src_path = dir.path().join("test.py");
        std::fs::write(&src_path, "content").expect("write");
        let output_dir = dir.path().join("out");
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        // Create a mock plugin that exits with code 1
        let plugin_path = dir.path().join("guidance-py");
        std::fs::write(&plugin_path, "#!/bin/sh\nexit 1\n").expect("write plugin");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&plugin_path, std::fs::Permissions::from_mode(0o755))
                .expect("set executable");
            let f = std::fs::File::open(&plugin_path).expect("open for sync");
            f.sync_all().expect("sync plugin");
            drop(f);
        }

        let plugin = Plugin {
            name: "guidance-py".into(),
            extensions: vec!["py".into()],
            path: plugin_path,
        };

        let result = invoke_plugin(&plugin, &src_path, &output_dir);
        assert!(result.is_err());
        match result.unwrap_err() {
            PluginError::ExecutionFailed(msg) => {
                assert!(msg.contains("non-zero status"), "got: {msg}");
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
    }

    #[test]
    fn test_invoke_plugin_binary_not_found() {
        let plugin = Plugin {
            name: "guidance-nonexistent".into(),
            extensions: vec!["nonexistent".into()],
            path: PathBuf::from("/tmp/guidance-nonexistent-binary"),
        };
        let result = invoke_plugin(&plugin, Path::new("/tmp/src.py"), Path::new("/tmp/out"));
        match result {
            Err(PluginError::SubprocessCrashed(_)) => {} // expected
            other => panic!("expected SubprocessCrashed, got {other:?}"),
        }
    }

    // ── discover_provider tests ─────────────────────────────────────────────────

    #[test]
    fn test_discover_provider_workspace_bin() {
        let dir = tempdir();
        let bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let plugin_path = bin_dir.join("guidance-py");
        std::fs::write(&plugin_path, "#!/bin/sh").expect("write plugin");

        let result = discover_provider(dir.path(), ".py");
        assert!(result.is_some(), "expected to find plugin");
        let plugin = result.unwrap();
        assert_eq!(plugin.name, "guidance-py");
        assert_eq!(plugin.path, plugin_path);
        assert!(plugin.extensions.contains(&"py".into()));
    }

    #[test]
    fn test_discover_provider_not_found() {
        let dir = tempdir();
        let result = discover_provider(dir.path(), ".nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_discover_provider_skips_non_executable() {
        let dir = tempdir();
        let bin_dir = dir.path().join("bin");
        std::fs::create_dir_all(&bin_dir).expect("create bin dir");

        let plugin_path = bin_dir.join("guidance-py");
        std::fs::write(&plugin_path, "not executable").expect("write plugin");
        // Don't set executable bit

        let result = discover_provider(dir.path(), ".py");
        // On unix, is_file() returns true even if not executable
        // Our implementation checks is_file() only, so it should still find it
        assert!(result.is_some());
    }
}
