//! Test utilities for Fluent WVR crates.
//!
//! Provides `impl_component_for_test!`, `PassthroughUnit`, `tempdir()`,
//! and `make_tree()` to reduce boilerplate in test modules.

pub use fluent_wvr::{
    impl_component, Component, Describable, FieldAccess, FieldError, WorkContext, WorkError,
    WorkOutput, WorkUnit,
};
pub use internment::ArcIntern;

/// Generates trivial `FieldAccess` + `Describable` impls so test types
/// satisfy the `Component` supertrait bound.
///
/// # Example
/// ```ignore
/// use fluent_wvr_testutil::impl_component_for_test;
///
/// struct MyTestType;
/// impl_component_for_test!(MyTestType);
/// ```
#[macro_export]
macro_rules! impl_component_for_test {
    ($type:ty) => {
        impl $crate::FieldAccess for $type {
            fn set_field(&mut self, _: &str, _: &str) -> Result<(), $crate::FieldError> {
                Ok(())
            }
            fn get_field(&self, _: &str) -> Result<String, $crate::FieldError> {
                Err($crate::FieldError::NotFound("test type: no fields".into()))
            }
            fn field_names(&self) -> &'static [&'static str] {
                &[]
            }
        }
        impl $crate::Describable for $type {
            fn describe(&self) -> serde_json::Value {
                serde_json::json!({})
            }
        }
        ::fluent_wvr::impl_component!($type);
    };
}

/// A trivial `WorkUnit` + `Component` for tests that need a passthrough target.
pub struct PassthroughUnit {
    pub name: String,
}

impl PassthroughUnit {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl WorkUnit for PassthroughUnit {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("passthrough"))
    }
}

impl_component_for_test!(PassthroughUnit);

/// Create a temporary directory. Panics if the OS fails to create it.
pub fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

/// Create a directory tree with the given files and directories.
///
/// Files are created as empty. Parent directories are created automatically.
pub fn make_tree(root: &std::path::Path, files: &[&str], dirs: &[&str]) {
    for d in dirs {
        common_core::ensure_dir(root.join(d)).unwrap();
    }
    for f in files {
        let p = root.join(f);
        if let Some(parent) = p.parent() {
            common_core::ensure_dir(parent).unwrap();
        }
        std::fs::write(&p, "").unwrap();
    }
}
