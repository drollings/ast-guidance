use std::path::{Path, PathBuf};

use fluent_types::GuidanceDoc;
use fluent_wvr::wrapper::Pipeline;
use thiserror::Error;

use crate::ast_parser::AstParser;
use crate::enhancer::{enhance_doc, Enhancer};
use crate::sync::comments;
use crate::sync::json_store;
use crate::sync::staleness;
use crate::walk;
use search_vector::GuidanceDb;

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

struct SyncContext {
    doc: GuidanceDoc,
    source_path: PathBuf,
    source: String,
    config: GenConfig,
    source_dir: PathBuf,
    guidance_dir: PathBuf,
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

        let source_path_str = source_path
            .strip_prefix(&self.workspace_root)
            .unwrap_or(source_path)
            .to_string_lossy()
            .to_string();

        let mut doc = self
            .ast_parser
            .parse_file(source_path, &source)
            .map_err(|e| SyncEngineError::Parse(e.to_string()))?;

        doc.meta.module = module_name.as_str().into();
        doc.meta.source = source_path_str.as_str().into();

        let mut ctx = SyncContext {
            doc,
            source_path: source_path.to_path_buf(),
            source,
            config: config.clone(),
            source_dir: self.source_dir.clone(),
            guidance_dir: self.guidance_dir.clone(),
        };

        let mut pipeline =
            Self::build_pipeline(self.enhancer.as_ref(), &mut self.ast_parser, config.db_sync);
        pipeline.run(&mut ctx)?;

        Ok(ctx.doc)
    }

    fn build_pipeline<'a>(
        enhancer: Option<&'a Enhancer>,
        ast_parser: &'a mut AstParser,
        db_sync: bool,
    ) -> Pipeline<'a, SyncContext, SyncEngineError> {
        let mut p = Pipeline::new()
            .step(move |ctx: &mut SyncContext| {
                if let Some(enhancer) = enhancer {
                    if let Err(e) = enhance_doc(enhancer, &mut ctx.doc, &ctx.source) {
                        tracing::warn!("LLM enhancement failed for {:?}: {e}", ctx.source_path);
                    }
                }
                Ok(())
            })
            .step(|ctx: &mut SyncContext| {
                let json_path = guidance_json_path(ctx);
                json_store::save_guidance(&json_path, &ctx.doc)?;
                Ok(())
            })
            .step(move |ctx: &mut SyncContext| {
                if let Err(e) = comments::sync_comments(&ctx.source_path, &ctx.doc, ast_parser) {
                    tracing::warn!("comment sync failed for {:?}: {e}", ctx.source_path);
                }
                Ok(())
            });

        if db_sync {
            p = p.maybe(
                |_ctx: &SyncContext| true,
                move |ctx: &mut SyncContext| {
                    let db_path = ctx
                        .config
                        .db_path
                        .clone()
                        .unwrap_or_else(|| ctx.guidance_dir.join("..").join(".guidance.db"));
                    let json_base = ctx
                        .config
                        .json_base
                        .clone()
                        .unwrap_or_else(|| ctx.guidance_dir.join("src"));
                    if let Ok(db) = GuidanceDb::open(&db_path) {
                        let _ = db.sync_from_dir(&json_base);
                    }
                    Ok(())
                },
            );
        }

        p
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

fn guidance_json_path(ctx: &SyncContext) -> PathBuf {
    let relative = ctx
        .source_path
        .strip_prefix(&ctx.source_dir)
        .unwrap_or(&ctx.source_path);
    let json_name = format!("{}.json", relative.display());
    ctx.guidance_dir.join("src").join(&json_name)
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

        let zig_file = source_dir.join("test.zig");
        std::fs::write(&zig_file, "pub fn hello() void {}\n").expect("write");

        let guidance_dir = dir.path().join(".guidance");
        let mut engine = SyncEngine::new(guidance_dir, source_dir);

        let doc = engine.gen(&zig_file).expect("gen");
        assert_eq!(doc.members.len(), 1);

        let source_after = std::fs::read_to_string(&zig_file).expect("read");
        assert!(source_after.contains("pub fn hello() void {}"));
    }
}
