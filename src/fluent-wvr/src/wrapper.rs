use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use common_core::metrics::LatencyHistogram;
use internment::ArcIntern;
use tracing::info;

use crate::{
    impl_component, impl_fieldless, Component, Describable, FieldAccess, FieldError, WorkContext,
    WorkError, WorkOutput, WorkUnit,
};

pub struct RetryResult<T> {
    pub result: T,
    pub attempts: usize,
}

/// Retry a synchronous closure with jittered-exponential backoff using
/// `std::thread::sleep`.
///
/// The delay schedule is delegated to `common_core::retry::backoff_ms`
/// (base × 2^(attempt-1) plus 100% jitter). This is the synchronous,
/// explicitly-blocking counterpart to `common_core::retry::retry_async`;
/// it is documented as such and must NOT be called from a `WorkUnit::execute`
/// body (the WorkUnit purity contract — see `WorkUnit`'s doc comment).
pub fn retry_call<F, T, E>(max_attempts: usize, base_ms: u64, f: F) -> Result<RetryResult<T>, E>
where
    F: Fn() -> Result<T, E>,
{
    assert!(max_attempts >= 1);
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f() {
            Ok(v) => {
                return Ok(RetryResult {
                    result: v,
                    attempts,
                })
            }
            Err(e) => {
                if attempts >= max_attempts {
                    return Err(e);
                }
                std::thread::sleep(Duration::from_millis(common_core::retry::backoff_ms(
                    base_ms,
                    attempts as u32,
                    100,
                )));
            }
        }
    }
}

/// A `WorkUnit` wrapper that logs execution timing and optionally records
/// durations into a shared `LatencyHistogram`.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_wvr::wrapper::Instrumented;
/// use common_core::metrics::LatencyHistogram;
/// # use fluent_wvr::{WorkUnit, WorkContext, WorkOutput, WorkError};
/// # use internment::ArcIntern;
/// # struct MyUnit;
/// # impl WorkUnit for MyUnit {
/// #     fn name(&self) -> &str { "my_unit" }
/// #     fn depends(&self) -> &[ArcIntern<str>] { &[] }
/// #     fn provides(&self) -> &[ArcIntern<str>] { &[] }
/// #     fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> { Ok(WorkOutput::ok("ok")) }
/// # }
///
/// let hist = Arc::new(LatencyHistogram::new());
/// let unit = Instrumented::with_metrics(MyUnit, "my.unit", hist.clone());
/// // unit.execute(ctx) will log timing and record to hist
/// ```
#[derive(Clone)]
pub struct Instrumented<U> {
    inner: U,
    label: String,
    histogram: Option<Arc<LatencyHistogram>>,
}

impl<U: WorkUnit> Instrumented<U> {
    pub fn new(inner: U, label: impl Into<String>) -> Self {
        Self {
            inner,
            label: label.into(),
            histogram: None,
        }
    }

    /// Build an `Instrumented` wrapper that, in addition to `tracing::info!`
    /// on every `execute`, records the observed execution duration into a
    /// shared `LatencyHistogram`.
    ///
    /// # In-tree consumer
    ///
    /// Coral's `QueueReactor` wraps each tier (`L3GraphUnit`,
    /// `L4SemanticUnit`, `L5FrontierUnit`) in `Instrumented::with_metrics`
    /// before type erasure into `Arc<dyn Component>`. The per-tier
    /// histograms are exposed via the `coral_stats` MCP method.
    pub fn with_metrics(
        inner: U,
        label: impl Into<String>,
        histogram: Arc<LatencyHistogram>,
    ) -> Self {
        Self {
            inner,
            label: label.into(),
            histogram: Some(histogram),
        }
    }
}

impl<U: WorkUnit> WorkUnit for Instrumented<U> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        self.inner.depends()
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        self.inner.provides()
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let start = Instant::now();
        let result = self.inner.execute(ctx);
        let elapsed = start.elapsed();
        if let Some(ref hist) = self.histogram {
            hist.observe_duration(start);
        }
        info!(target: "instrumented", label = %self.label, elapsed = ?elapsed, name = %self.inner.name(), "executed");
        result
    }

    fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }
}

impl<U: crate::Component> FieldAccess for Instrumented<U> {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        <U as FieldAccess>::set_field(&mut self.inner, name, value)
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        <U as FieldAccess>::get_field(&self.inner, name)
    }
    fn field_names(&self) -> &'static [&'static str] {
        <U as FieldAccess>::field_names(&self.inner)
    }
}

impl<U: crate::Component> crate::Describable for Instrumented<U> {
    fn describe(&self) -> serde_json::Value {
        <U as crate::Describable>::describe(&self.inner)
    }
}

impl_component!(generic (U: crate::Component + 'static) for Instrumented<U>);

/// A wrapper that adapts any `Arc<dyn Component>` at runtime.
///
/// `ComponentAdapter` lets callers override one or more of the four
/// `Component` facets — `name`, `execute`, and any field — without
/// subclassing or owning the inner component. Overrides stack in
/// reverse order of insertion (last override wins), so a caller that
/// wraps a component multiple times can layer configuration.
///
/// # When to use
///
/// - **Configuration injection** — wrap a component to set a field
///   that the inner type does not expose.
/// - **Behavior override** — swap `execute` for a test double or a
///   memoized fast-path without touching the inner implementation.
/// - **Renaming** — keep the inner name as the canonical identifier
///   while presenting a stable, user-facing name in error messages
///   or schemas.
///
/// # When NOT to use
///
/// - If you own the inner type, configure it directly — `Arc::get_mut`
///   + `set_field` is the canonical path.
/// - If the override is permanent, add it to the inner type's impl.
pub type ExecuteFn = Arc<dyn Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync>;

pub struct ComponentAdapter {
    inner: Arc<dyn Component>,
    name_override: Option<String>,
    execute_override: Option<ExecuteFn>,
    field_overrides: HashMap<String, String>,
}

impl Clone for ComponentAdapter {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            name_override: self.name_override.clone(),
            execute_override: self.execute_override.clone(),
            field_overrides: self.field_overrides.clone(),
        }
    }
}

impl ComponentAdapter {
    /// Wrap an `Arc<dyn Component>` with no overrides — delegates every
    /// call to the inner component.
    pub fn new(inner: Arc<dyn Component>) -> Self {
        Self {
            inner,
            name_override: None,
            execute_override: None,
            field_overrides: HashMap::new(),
        }
    }

    /// Override the name returned by `WorkUnit::name`. Pass-through to
    /// the inner component if no override is set.
    #[must_use]
    pub fn with_name_override(mut self, name: impl Into<String>) -> Self {
        self.name_override = Some(name.into());
        self
    }

    /// Replace the `execute` implementation. Useful for test doubles
    /// and policy enforcement layers (e.g. an audit wrapper around
    /// an existing component).
    #[must_use]
    pub fn with_execute_override(mut self, f: ExecuteFn) -> Self {
        self.execute_override = Some(f);
        self
    }

    /// Add a field override. If a value for this name already exists,
    /// it is replaced — the most recently set value wins.
    #[must_use]
    pub fn with_field_override(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.field_overrides.insert(name.into(), value.into());
        self
    }

    /// Borrow the wrapped component.
    pub fn inner(&self) -> &Arc<dyn Component> {
        &self.inner
    }

    /// Bulk-set field overrides from a pre-built `HashMap`. Existing
    /// overrides for the same keys are replaced.
    #[must_use]
    pub fn with_field_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        self.field_overrides.extend(overrides);
        self
    }

    /// Remove all field overrides. After calling this, the inner
    /// component is the sole authority for all field values.
    pub fn clear_field_overrides(&mut self) {
        self.field_overrides.clear();
    }
}

impl WorkUnit for ComponentAdapter {
    fn name(&self) -> &str {
        self.name_override
            .as_deref()
            .unwrap_or_else(|| self.inner.name())
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        self.inner.depends()
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        self.inner.provides()
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        match &self.execute_override {
            Some(f) => f(ctx),
            None => self.inner.execute(ctx),
        }
    }

    fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }
}

impl FieldAccess for ComponentAdapter {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.set_field(name, value)?;
        }
        self.field_overrides.insert(name.into(), value.into());
        Ok(())
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        if let Some(v) = self.field_overrides.get(name) {
            return Ok(v.clone());
        }
        self.inner.get_field(name)
    }
    fn field_names(&self) -> &'static [&'static str] {
        self.inner.field_names()
    }
}

impl Describable for ComponentAdapter {
    fn describe(&self) -> serde_json::Value {
        let mut schema = self.inner.describe();
        if let Some(name) = &self.name_override {
            schema["name"] = serde_json::Value::String(name.clone());
        }
        if !self.field_overrides.is_empty() {
            let mut pairs: Vec<(&String, &String)> = self.field_overrides.iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            let arr: Vec<Vec<String>> = pairs
                .into_iter()
                .map(|(k, v)| vec![k.clone(), v.clone()])
                .collect();
            schema["field_overrides"] = serde_json::json!(arr);
        }
        schema["adapted"] = serde_json::Value::Bool(true);
        schema
    }
}

impl_component!(ComponentAdapter);

/// A first-Ok-wins cascade of `Arc<dyn Component>` units.
///
/// Runs units in registration order and returns the first `Ok` `WorkOutput`
/// (coral's `TierRegistry` semantics, generalized). An `Err` from a unit
/// advances to the next one; when every unit fails, the last `Err` is
/// returned (falling back to a descriptive error when the cascade is empty).
///
/// The cascade itself is a `Component`, so it can be registered, wrapped, or
/// nested like any other unit. `depends`/`provides` are empty — the cascade
/// is a dispatch container, not a producer of assets.
pub struct ComponentCascade {
    units: Vec<Arc<dyn Component>>,
}

impl ComponentCascade {
    pub fn new() -> Self {
        Self { units: Vec::new() }
    }

    /// Build a cascade from a pre-registered unit list.
    pub fn with_units(units: Vec<Arc<dyn Component>>) -> Self {
        Self { units }
    }

    /// Append a unit to the end of the cascade.
    pub fn push(&mut self, unit: Arc<dyn Component>) {
        self.units.push(unit);
    }

    /// Append a unit to the end of the cascade (alias for `push`).
    pub fn register(&mut self, unit: Arc<dyn Component>) {
        self.units.push(unit);
    }

    /// Number of units in the cascade.
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// `true` when the cascade holds no units.
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Run units in order; return the first `Ok` `WorkOutput`, or the last
    /// `Err` when all units fail.
    pub fn execute_first_ok(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let mut last_err = None;
        for unit in &self.units {
            match unit.execute(ctx) {
                Ok(output) => return Ok(output),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| WorkError::Execution("component cascade has no units".into())))
    }
}

impl Default for ComponentCascade {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkUnit for ComponentCascade {
    fn name(&self) -> &str {
        "component_cascade"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        self.execute_first_ok(ctx)
    }
}

impl_fieldless!(ComponentCascade);

impl Describable for ComponentCascade {
    fn describe(&self) -> serde_json::Value {
        let units: Vec<String> = self.units.iter().map(|u| u.name().to_string()).collect();
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
            "cascade": {
                "first_ok": true,
                "units": units,
            },
        })
    }
}

impl_component!(ComponentCascade);

/// Post-erasure Component wrapper. Unlike the newtype wrappers
/// (`Instrumented`, `ComponentAdapter`), middleware wraps an
/// already-erased `Arc<dyn Component>`.
pub trait Middleware: Send + Sync {
    fn wrap(&self, inner: Arc<dyn Component>) -> Arc<dyn Component>;
}

/// Ordered chain of `Middleware` layers.  `apply` folds the component
/// through each layer: `result = inner; for mw in &middlewares { result = mw.wrap(result); }`.
pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }
    #[must_use]
    pub fn push(mut self, m: Box<dyn Middleware>) -> Self {
        self.middlewares.push(m);
        self
    }
    pub fn apply(&self, unit: Arc<dyn Component>) -> Arc<dyn Component> {
        let mut result = unit;
        for mw in &self.middlewares {
            result = mw.wrap(result);
        }
        result
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// A sequential, fallible step chain.  Steps are applied in registration
/// order.  Optional steps (gated by `.maybe(condition, step)`) are skipped
/// when their condition returns false.
pub struct Pipeline<'a, T: 'a, E> {
    steps: Vec<Step<'a, T, E>>,
}

type StepFn<'a, T, E> = Box<dyn FnMut(&mut T) -> Result<(), E> + Send + Sync + 'a>;
type StepCond<'a, T> = Box<dyn Fn(&T) -> bool + Send + Sync + 'a>;

enum Step<'a, T: 'a, E> {
    Required(StepFn<'a, T, E>),
    Optional {
        condition: StepCond<'a, T>,
        step: StepFn<'a, T, E>,
    },
}

impl<'a, T: 'a, E> Pipeline<'a, T, E> {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    #[must_use]
    pub fn step(mut self, f: impl FnMut(&mut T) -> Result<(), E> + Send + Sync + 'a) -> Self {
        self.steps.push(Step::Required(Box::new(f)));
        self
    }

    #[must_use]
    pub fn maybe(
        mut self,
        condition: impl Fn(&T) -> bool + Send + Sync + 'a,
        step: impl FnMut(&mut T) -> Result<(), E> + Send + Sync + 'a,
    ) -> Self {
        self.steps.push(Step::Optional {
            condition: Box::new(condition),
            step: Box::new(step),
        });
        self
    }

    pub fn run(&mut self, initial: &mut T) -> Result<(), E> {
        for step in &mut self.steps {
            match step {
                Step::Required(f) => f(initial)?,
                Step::Optional { condition, step } => {
                    if condition(initial) {
                        step(initial)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<'a, T: 'a, E> Default for Pipeline<'a, T, E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Wraps a `Component` with a name suffix (useful for scatter-gather SupervisedBatch
/// dispatch where the same component type is registered multiple times).
///
/// `execute` merges the batch-supplied runtime (`ctx.rt`) with per-task
/// configuration from `self.ctx`, so the task gets the live runtime but
/// retains its own capabilities and metadata.
pub struct SuffixedComponent {
    inner: Arc<dyn Component>,
    name: String,
    ctx: WorkContext,
}

impl SuffixedComponent {
    pub fn new(inner: Arc<dyn Component>, suffix: impl AsRef<str>, ctx: WorkContext) -> Self {
        let name = format!("{}:{}", inner.name(), suffix.as_ref());
        Self { inner, name, ctx }
    }
}

impl WorkUnit for SuffixedComponent {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        self.inner.depends()
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        self.inner.provides()
    }
    fn execute(&self, batch_ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let merged = WorkContext {
            rt: Arc::clone(&batch_ctx.rt),
            ..self.ctx.clone()
        };
        self.inner.execute(&merged)
    }
    fn default_timeout_ms(&self) -> u64 {
        self.inner.default_timeout_ms()
    }
    fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }
}

impl FieldAccess for SuffixedComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        self.inner.set_field(name, value)
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        self.inner.get_field(name)
    }
    fn field_names(&self) -> &'static [&'static str] {
        self.inner.field_names()
    }
}

impl Describable for SuffixedComponent {
    fn describe(&self) -> serde_json::Value {
        self.inner.describe()
    }
}

impl_component!(SuffixedComponent);
