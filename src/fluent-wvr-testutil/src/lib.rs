//! Test utilities for Fluent WVR crates.
//!
//! Provides `impl_component_for_test!`, `PassthroughUnit`, `tempdir()`,
//! and `make_tree()` to reduce boilerplate in test modules.

use std::sync::Arc;

pub use fluent_wvr::prelude::*;

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

/// A configurable stub `Component` for tests that replaces ad‑hoc test structs.
///
/// # Examples
///
/// ```ignore
/// use fluent_wvr_testutil::StubComponent;
///
/// let ok_unit = StubComponent::ok("worker");
/// let fail_unit = StubComponent::fail("flaky");
/// let panic_unit = StubComponent::panic("crash");
/// ```
#[allow(clippy::type_complexity)]
pub struct StubComponent {
    pub name: String,
    pub deps: Vec<ArcIntern<str>>,
    pub provides: Vec<ArcIntern<str>>,
    pub fail: bool,
    pub panic: bool,
    pub max_retries: u32,
    pub execute_fn: Option<Arc<dyn Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync>>,
}

impl StubComponent {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            deps: Vec::new(),
            provides: Vec::new(),
            fail: false,
            panic: false,
            max_retries: 0,
            execute_fn: None,
        }
    }

    /// Shorthand for `StubComponent::new(name)` — a unit that succeeds.
    pub fn ok(name: &str) -> Self {
        Self::new(name)
    }

    /// A unit that returns `Err(WorkError::Execution(...))`.
    pub fn fail(name: &str) -> Self {
        let mut s = Self::new(name);
        s.fail = true;
        s
    }

    /// A unit that panics when executed.
    pub fn panic(name: &str) -> Self {
        let mut s = Self::new(name);
        s.panic = true;
        s
    }

    /// Add a dependency asset.
    #[must_use]
    pub fn with_dep(mut self, asset: &str) -> Self {
        self.deps.push(ArcIntern::from(asset));
        self
    }

    /// Add a provides asset.
    #[must_use]
    pub fn with_provides(mut self, asset: &str) -> Self {
        self.provides.push(ArcIntern::from(asset));
        self
    }

    /// Replace `execute` with a custom handler.
    #[must_use]
    pub fn with_handler<F>(mut self, f: F) -> Self
    where
        F: Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync + 'static,
    {
        self.execute_fn = Some(Arc::new(f));
        self
    }
}

impl WorkUnit for StubComponent {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.deps
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        if let Some(ref f) = self.execute_fn {
            return f(ctx);
        }
        if self.panic {
            panic!("StubComponent::panic: {}", self.name);
        }
        if self.fail {
            return Err(WorkError::Execution(format!("stub fail: {}", self.name)));
        }
        Ok(WorkOutput::ok("stub ok"))
    }
}

impl_component_for_test!(StubComponent);

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
