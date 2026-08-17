//! Tier-2 (crate-root `tests/`) shared e2e fixtures for guidance-core.
//!
//! Public-API only — this module is a separate crate linked against
//! `guidance_core`, so it cannot reach crate-internal items. The Tier-1
//! fixtures used by inline suites live in `src/tests/common.rs` instead;
//! only the e2e `SyncEngine`+tempdir setup that `e2e_gen_roundtrip.rs`
//! (and future e2e files) needs belongs here.

use fluent_wvr_testutil::tempdir;
use guidance_core::sync_engine::SyncEngine;

/// A tempdir with `src/` + `.guidance/` subdirectories and a `SyncEngine`
/// rooted at them. `dir` is never read directly but must stay alive: dropping
/// the `TempDir` would delete the filesystem the engine is working on.
pub struct EngineFixture {
    #[allow(dead_code)]
    pub dir: tempfile::TempDir,
    pub source_dir: std::path::PathBuf,
    pub guidance_dir: std::path::PathBuf,
    pub engine: SyncEngine,
}

pub fn make_engine() -> EngineFixture {
    let dir = tempdir();
    let source_dir = dir.path().join("src");
    let guidance_dir = dir.path().join(".guidance");
    std::fs::create_dir_all(&source_dir).expect("create src");
    std::fs::create_dir_all(&guidance_dir).expect("create guidance");
    let engine = SyncEngine::new(guidance_dir.clone(), source_dir.clone());
    EngineFixture {
        dir,
        source_dir,
        guidance_dir,
        engine,
    }
}