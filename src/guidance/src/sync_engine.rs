use std::path::{Path, PathBuf};

use guidance_types::GuidanceDoc;
use thiserror::Error;

use crate::ast_parser::AstParser;
use crate::enhancer::{enhance_doc, Enhancer};
use crate::sync::comments;
use crate::sync::json_store;
use crate::sync::staleness;
use crate::walk;
use guidance_search_vector::GuidanceDb;

#[derive(Error, Debug)]
pub enum SyncEngineError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("JSON error: {0}")]
    Json(#[from] json_store::JsonError),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("source file not found: {0}")]
    SourceNotFound(PathBuf),
    #[error("database error: {0}")]
    Db(String),
}

impl From<std::io::Error> for SyncEngineError {
    fn from(e: std::io::Error) -> Self {
        SyncEngineError::Io(common_core::error::IoError(e))
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenConfig {
    pub db_sync: bool,
    pub db_path: Option<PathBuf>,
    pub json_base: Option<PathBuf>,
}

pub struct SyncEngine {
    pub ast_parser: AstParser,
    pub guidance_dir: PathBuf,
    pub workspace_root: PathBuf,
    pub source_dir: PathBuf,
    pub enhancer: Option<Enhancer>,
}

impl SyncEngine {
    pub fn new(guidance_dir: PathBuf, source_dir: PathBuf) -> Self {
        let workspace_root = guidance_dir
            .parent()
            .map_or_else(|| source_dir.clone(), Path::to_path_buf);
        Self {
            ast_parser: AstParser::new(),
            guidance_dir,
            workspace_root,
            source_dir,
            enhancer: None,
        }
    }

    pub fn with_parser(guidance_dir: PathBuf, source_dir: PathBuf, ast_parser: AstParser) -> Self {
        let workspace_root = guidance_dir
            .parent()
            .map_or_else(|| source_dir.clone(), Path::to_path_buf);
        Self {
            ast_parser,
            guidance_dir,
            workspace_root,
            source_dir,
            enhancer: None,
        }
    }

    #[must_use]
    pub fn with_enhancer(mut self, enhancer: Enhancer) -> Self {
        self.enhancer = Some(enhancer);
        self
    }

    pub fn gen(&mut self, source_path: &Path) -> Result<GuidanceDoc, SyncEngineError> {
        self.gen_with_config(source_path, &GenConfig::default())
    }

    pub fn gen_with_config(
        &mut self,
        source_path: &Path,
        config: &GenConfig,
    ) -> Result<GuidanceDoc, SyncEngineError> {
        let source = common_core::io::read_to_string_err(source_path)?;

        // Module name is relative to source_dir (for logical module path).
        let module_rel = source_path
            .strip_prefix(&self.source_dir)
            .unwrap_or(source_path);
        let module_name = module_rel
            .to_string_lossy()
            .strip_suffix(&format!(
                ".{}",
                module_rel
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
            ))
            .unwrap_or(&module_rel.to_string_lossy())
            .replace(['/', '\\'], ".");

        // Source path is relative to workspace root (canonical display path).
        let source_path_str = source_path
            .strip_prefix(&self.workspace_root)
            .unwrap_or(source_path)
            .to_string_lossy();

        let mut doc = self
            .ast_parser
            .parse_file(source_path, &source)
            .map_err(|e| SyncEngineError::Parse(e.to_string()))?;

        doc.meta.module = module_name.as_str().into();
        doc.meta.source = source_path_str.as_ref().into();

        // G5: LLM enhancement — generate missing comments before JSON save
        if let Some(ref enhancer) = self.enhancer {
            if let Err(e) = enhance_doc(enhancer, &mut doc, &source) {
                tracing::warn!("LLM enhancement failed for {:?}: {e}", source_path);
            }
        }

        let json_path = self.guidance_json_path(source_path);
        json_store::save_guidance(&json_path, &doc)?;

        // Sync comments back to source file after generation
        if let Err(e) = comments::sync_comments(source_path, &doc, &mut self.ast_parser) {
            tracing::warn!("comment sync failed for {:?}: {e}", source_path);
        }

        // Database sync after JSON write
        if config.db_sync {
            let db_path = config
                .db_path
                .clone()
                .unwrap_or_else(|| self.guidance_dir.join("..").join(".guidance.db"));

            let json_base = config
                .json_base
                .clone()
                .unwrap_or_else(|| self.guidance_dir.join("src"));

            if let Ok(db) = GuidanceDb::open(&db_path) {
                let _ = db.sync_from_dir(&json_base);
            }
        }

        Ok(doc)
    }

    pub fn gen_if_stale(&mut self, source_path: &Path) -> Result<bool, SyncEngineError> {
        let json_path = self.guidance_json_path(source_path);

        if !staleness::should_generate(&json_path, source_path) {
            return Ok(false);
        }

        self.gen(source_path)?;
        Ok(true)
    }

    pub fn load_doc(&self, source_path: &Path) -> Result<Option<GuidanceDoc>, SyncEngineError> {
        let json_path = self.guidance_json_path(source_path);
        let doc = json_store::load_guidance(&json_path)?;
        Ok(doc)
    }

    pub fn status(&self) -> Result<SyncStatus, SyncEngineError> {
        let mut total_files = 0;
        let mut stale_files = 0;
        let mut up_to_date = 0;

        self.walk_source_files(|source_path| {
            total_files += 1;
            let json_path = self.guidance_json_path(source_path);
            if staleness::should_generate(&json_path, source_path) {
                stale_files += 1;
            } else {
                up_to_date += 1;
            }
        });

        Ok(SyncStatus {
            total_files,
            stale_files,
            up_to_date,
        })
    }

    fn guidance_json_path(&self, source_path: &Path) -> PathBuf {
        let relative = source_path
            .strip_prefix(&self.source_dir)
            .unwrap_or(source_path);
        let json_name = format!("{}.json", relative.display());
        self.guidance_dir.join("src").join(&json_name)
    }

    fn walk_source_files<F>(&self, mut callback: F)
    where
        F: FnMut(&Path),
    {
        walk::walk_files(&self.source_dir, walk::SOURCE_EXTENSIONS, &mut callback);
    }
}

#[derive(Debug, Clone)]
pub struct SyncStatus {
    pub total_files: usize,
    pub stale_files: usize,
    pub up_to_date: usize,
}

impl SyncStatus {
    pub fn is_clean(&self) -> bool {
        self.stale_files == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr_testutil::tempdir;

    #[test]
    fn test_gen_and_load_round_trip() {
        let dir = tempdir();
        let source_dir = dir.path().join("src");
        std::fs::create_dir(&source_dir).expect("create src");

        let zig_file = source_dir.join("test.zig");
        std::fs::write(&zig_file, "/// A test module\npub fn hello() void {}\n").expect("write");

        let guidance_dir = dir.path().join(".guidance");
        let mut engine = SyncEngine::new(guidance_dir.clone(), source_dir);

        let doc = engine.gen(&zig_file).expect("gen");
        assert_eq!(doc.meta.module.as_str(), "test");
        assert_eq!(doc.members.len(), 1);
        assert_eq!(doc.members[0].name.as_str(), "hello");
    }

    #[test]
    fn test_gen_if_stale() {
        let dir = tempdir();
        let source_dir = dir.path().join("src");
        std::fs::create_dir(&source_dir).expect("create src");

        let zig_file = source_dir.join("test.zig");
        std::fs::write(&zig_file, "pub fn foo() void {}").expect("write");

        let guidance_dir = dir.path().join(".guidance");
        let mut engine = SyncEngine::new(guidance_dir, source_dir);

        assert!(engine.gen_if_stale(&zig_file).expect("gen if stale"));
    }

    #[test]
    fn test_status() {
        let dir = tempdir();
        let source_dir = dir.path().join("src");
        std::fs::create_dir(&source_dir).expect("create src");

        let zig_file = source_dir.join("test.zig");
        std::fs::write(&zig_file, "pub fn bar() void {}").expect("write");

        let guidance_dir = dir.path().join(".guidance");
        let mut engine = SyncEngine::new(guidance_dir, source_dir);
        engine.gen(&zig_file).expect("gen");

        let status = engine.status().expect("status");
        assert_eq!(status.total_files, 1);
    }

    #[test]
    fn test_gen_syncs_comments() {
        let dir = tempdir();
        let source_dir = dir.path().join("src");
        std::fs::create_dir(&source_dir).expect("create src");

        // Source with a function but no comment
        let zig_file = source_dir.join("test.zig");
        std::fs::write(&zig_file, "pub fn hello() void {}\n").expect("write");

        let guidance_dir = dir.path().join(".guidance");
        let mut engine = SyncEngine::new(guidance_dir, source_dir);

        // gen() should call sync_comments which adds /// comments
        let doc = engine.gen(&zig_file).expect("gen");
        assert_eq!(doc.members.len(), 1);

        // Reread source to verify comments were synced
        let source_after = std::fs::read_to_string(&zig_file).expect("read");
        // The member has no comment, so no /// should be added — just verifying no crash
        assert!(source_after.contains("pub fn hello() void {}"));
    }
}
