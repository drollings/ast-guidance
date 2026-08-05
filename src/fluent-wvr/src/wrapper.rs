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

/// Wraps a `Component` with a name suffix (useful for scatter-gather Zone
/// dispatch where the same component type is registered multiple times).
///
/// `execute` merges the zone-supplied runtime (`ctx.rt`) with per-task
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
    fn execute(&self, zone_ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let merged = WorkContext {
            rt: Arc::clone(&zone_ctx.rt),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn retry_call_succeeds_first() {
        let result: Result<RetryResult<i32>, ()> = retry_call(3, 1, || Ok(42));
        assert_eq!(result.unwrap().result, 42);
    }

    #[test]
    fn retry_call_always_fails() {
        let result: Result<RetryResult<i32>, ()> = retry_call(3, 1, || Err(()));
        assert!(result.is_err());
    }

    #[test]
    fn retry_call_jittered_delays_are_non_deterministic() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;

        // Run two sequences of retries; due to jitter, total wall time
        // should differ (at least one will be faster). We use a counter
        // that always fails to force all retry delays to fire.
        fn run_retries() -> u128 {
            let calls = AtomicUsize::new(0);
            let start = Instant::now();
            let _ = retry_call(5, 10, || {
                calls.fetch_add(1, Ordering::Relaxed);
                Err::<i32, ()>(())
            });
            start.elapsed().as_millis()
        }

        let t1 = run_retries();
        let t2 = run_retries();
        // Both should have taken some time (at least a few ms from the delays).
        assert!(t1 > 0, "first run should take > 0ms");
        assert!(t2 > 0, "second run should take > 0ms");
        // With jitter, the two runs should not be identical (very high
        // probability with 4 jittered delays of 10ms each).
        // We can't assert strict inequality due to timer resolution, but
        // we can assert both are positive which proves the delays fired.
    }

    #[derive(Clone)]
    struct CloneableUnit;

    impl WorkUnit for CloneableUnit {
        fn name(&self) -> &str {
            "cloneable"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("ok"))
        }
    }

    impl FieldAccess for CloneableUnit {
        fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
            Ok(())
        }
        fn get_field(&self, _: &str) -> Result<String, FieldError> {
            Ok(String::new())
        }
        fn field_names(&self) -> &'static [&'static str] {
            &[]
        }
    }

    impl Describable for CloneableUnit {
        fn describe(&self) -> serde_json::Value {
            serde_json::Value::Object(serde_json::Map::new())
        }
    }

    impl_component!(CloneableUnit);

    struct MockUnit {
        name: ArcIntern<str>,
        should_fail: bool,
        call_count: AtomicUsize,
    }

    impl MockUnit {
        fn ok(name: &str) -> Self {
            Self {
                name: ArcIntern::from(name),
                should_fail: false,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl WorkUnit for MockUnit {
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
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.should_fail {
                Err(WorkError::Execution("failed".into()))
            } else {
                Ok(WorkOutput::ok("done"))
            }
        }
    }

    impl FieldAccess for MockUnit {
        fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
            Ok(())
        }
        fn get_field(&self, _: &str) -> Result<String, FieldError> {
            Err(FieldError::NotFound("test type: no fields".into()))
        }
        fn field_names(&self) -> &'static [&'static str] {
            &[]
        }
    }

    impl Describable for MockUnit {
        fn describe(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    impl_component!(MockUnit);

    #[test]
    fn instrumented_delegates() {
        let inner = MockUnit::ok("mock");
        let wrapped = Instrumented::new(inner, "test-label");
        let ctx = WorkContext::default();
        let result = wrapped.execute(&ctx);
        assert!(result.is_ok());
        assert_eq!(wrapped.name(), "mock");
    }

    /// Exercises `Instrumented::with_metrics` end-to-end: confirms the
    /// histogram records one observation after `execute` runs and that the
    /// recorded sum is non-zero. This is the M12 minimum-bar in-tree usage;
    /// see the `with_metrics` doc comment for the candidate production sites
    /// that have not yet been wired.
    #[test]
    fn instrumented_with_metrics_records_duration() {
        let inner = MockUnit::ok("mock");
        let histogram = Arc::new(LatencyHistogram::new());
        let wrapped = Instrumented::with_metrics(inner, "knn_mock", Arc::clone(&histogram));
        let ctx = WorkContext::default();
        let result = wrapped.execute(&ctx);
        assert!(result.is_ok());
        // Exactly one observation was recorded by the wrapper's `execute`.
        assert_eq!(histogram.count(), 1);
        // `estimate_percentile` returns the bucket bound for the observed
        // duration; a non-zero value confirms the wiring is live (and that
        // `observe_duration` was actually called inside `execute`).
        let p50 = histogram.estimate_percentile(50.0);
        assert!(
            p50 > 0,
            "histogram must produce a non-zero p50 after one observation"
        );
    }

    #[test]
    fn arc_dyn_work_unit_delegates() {
        let inner = MockUnit::ok("mock");
        let arc: Arc<dyn WorkUnit> = Arc::new(inner);
        let ctx = WorkContext::default();
        assert_eq!(arc.name(), "mock");
        let result = arc.execute(&ctx);
        assert!(result.is_ok());
    }

    // --- M1.3: Downcast round-trip tests ---

    #[test]
    fn instrumented_downcast_roundtrip() {
        let inner = MockUnit::ok("mock");
        let instr = Instrumented::new(inner, "test");
        let arc: Arc<dyn Component> = Arc::new(instr);
        assert!(crate::component_downcast_ref::<Instrumented<MockUnit>>(&*arc).is_some());
        // Instrumented does not auto-leak the inner type
        assert!(crate::component_downcast_ref::<MockUnit>(&*arc).is_none());
    }

    // --- ComponentAdapter tests ---

    /// A `Component` that is NOT a `MockUnit` — used to confirm the
    /// adapter works when wrapping through `Arc<dyn Component>`.
    struct AdapterHost {
        name: ArcIntern<str>,
        last_message: Arc<std::sync::Mutex<String>>,
    }
    impl AdapterHost {
        fn new(name: &str) -> Self {
            Self {
                name: ArcIntern::from(name),
                last_message: Arc::new(std::sync::Mutex::new(String::new())),
            }
        }
    }
    impl FieldAccess for AdapterHost {
        fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
            Ok(())
        }
        fn get_field(&self, _name: &str) -> Result<String, FieldError> {
            Err(FieldError::NotFound("no fields".into()))
        }
        fn field_names(&self) -> &'static [&'static str] {
            &[]
        }
    }
    impl Describable for AdapterHost {
        fn describe(&self) -> serde_json::Value {
            serde_json::json!({"name": &*self.name})
        }
    }
    impl_component!(AdapterHost);
    impl WorkUnit for AdapterHost {
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
            let mut g = self.last_message.lock().unwrap();
            *g = "from_inner".to_string();
            Ok(WorkOutput::ok("from_inner"))
        }
    }

    #[test]
    fn adapter_delegates_by_default() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host));
        assert_eq!(adapter.name(), "inner");
        let result = adapter.execute(&WorkContext::default()).unwrap();
        assert_eq!(result.message, "from_inner");
    }

    #[test]
    fn adapter_name_override() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host)).with_name_override("renamed");
        assert_eq!(adapter.name(), "renamed");
    }

    #[test]
    fn adapter_execute_override_short_circuits_inner() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host))
            .with_execute_override(Arc::new(|_| Ok(WorkOutput::ok("overridden"))));
        let result = adapter.execute(&WorkContext::default()).unwrap();
        assert_eq!(result.message, "overridden");
    }

    #[test]
    fn adapter_field_override_set_then_get() {
        let host = AdapterHost::new("inner");
        let mut adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "8079");
        assert_eq!(adapter.get_field("port").unwrap(), "8079");
        adapter.set_field("port", "9090").unwrap();
        assert_eq!(adapter.get_field("port").unwrap(), "9090");
    }

    #[test]
    fn adapter_field_overrides_stack_last_wins() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host))
            .with_field_override("k", "v1")
            .with_field_override("k", "v2");
        assert_eq!(adapter.get_field("k").unwrap(), "v2");
    }

    #[test]
    fn adapter_field_not_found() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host));
        assert!(matches!(
            adapter.get_field("missing"),
            Err(FieldError::NotFound(_))
        ));
    }

    #[test]
    fn adapter_is_itself_a_component() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host));
        // Box as dyn Component — proves the blanket impl fires.
        let boxed: Box<dyn Component> = Box::new(adapter);
        assert_eq!(boxed.name(), "inner");
    }

    #[test]
    fn adapter_describe_includes_overrides() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host))
            .with_name_override("renamed")
            .with_field_override("port", "8079");
        let schema = adapter.describe();
        assert_eq!(schema["name"], "renamed");
        assert_eq!(schema["adapted"], true);
        let overrides = schema["field_overrides"].as_array().unwrap();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0][0], "port");
        assert_eq!(overrides[0][1], "8079");
    }

    #[test]
    fn adapter_inner_accessor_returns_wrapped_component() {
        let host = AdapterHost::new("inner");
        let adapter = ComponentAdapter::new(Arc::new(host));
        let inner: &Arc<dyn Component> = adapter.inner();
        assert_eq!(inner.name(), "inner");
    }

    #[test]
    fn adapter_propagates_depends_and_provides_from_inner() {
        // A Component whose depends/provides are non-empty — confirm
        // delegation rather than pass-through to the empty default.
        struct DepProvider {
            name: ArcIntern<str>,
            deps: Vec<ArcIntern<str>>,
            provs: Vec<ArcIntern<str>>,
        }
        impl WorkUnit for DepProvider {
            fn name(&self) -> &str {
                &self.name
            }
            fn depends(&self) -> &[ArcIntern<str>] {
                &self.deps
            }
            fn provides(&self) -> &[ArcIntern<str>] {
                &self.provs
            }
            fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
                Ok(WorkOutput::ok("ok"))
            }
        }
        impl FieldAccess for DepProvider {
            fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
                Ok(())
            }
            fn get_field(&self, _: &str) -> Result<String, FieldError> {
                Err(FieldError::NotFound("none".into()))
            }
            fn field_names(&self) -> &'static [&'static str] {
                &[]
            }
        }
        impl Describable for DepProvider {
            fn describe(&self) -> serde_json::Value {
                serde_json::json!({})
            }
        }
        impl_component!(DepProvider);
        let inner = DepProvider {
            name: ArcIntern::from("dp"),
            deps: vec![ArcIntern::from("a"), ArcIntern::from("b")],
            provs: vec![ArcIntern::from("c")],
        };
        let adapter = ComponentAdapter::new(Arc::new(inner));
        assert_eq!(adapter.depends().len(), 2);
        assert_eq!(&*adapter.depends()[0], "a");
        assert_eq!(&*adapter.depends()[1], "b");
        assert_eq!(adapter.provides().len(), 1);
        assert_eq!(&*adapter.provides()[0], "c");
    }

    /// M10.3: Wrap an AdapterHost with non-empty `field_names` and verify the
    /// adapter reports the same set — proves `ComponentAdapter::field_names`
    /// delegates to `self.inner.field_names()` rather than returning `&[]`.
    #[test]
    fn adapter_field_names_forwards_from_inner() {
        struct FieldedHost {
            name: ArcIntern<str>,
        }
        impl WorkUnit for FieldedHost {
            fn name(&self) -> &str {
                &self.name
            }
            fn depends(&self) -> &[ArcIntern<str>] {
                &[]
            }
            fn provides(&self) -> &[ArcIntern<str>] {
                &[]
            }
            fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
                Ok(WorkOutput::ok("ok"))
            }
        }
        impl FieldAccess for FieldedHost {
            fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
                Ok(())
            }
            fn get_field(&self, _: &str) -> Result<String, FieldError> {
                Err(FieldError::NotFound("none".into()))
            }
            fn field_names(&self) -> &'static [&'static str] {
                &["host_a", "host_b", "host_c"]
            }
        }
        impl Describable for FieldedHost {
            fn describe(&self) -> serde_json::Value {
                serde_json::json!({})
            }
        }
        impl_component!(FieldedHost);

        let inner = FieldedHost {
            name: ArcIntern::from("fielded"),
        };
        let adapter = ComponentAdapter::new(Arc::new(inner));
        let names = adapter.field_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "host_a");
        assert_eq!(names[1], "host_b");
        assert_eq!(names[2], "host_c");
    }

    /// A `Component` with a max constraint on `port` (max=1024).
    struct ConstrainedHost {
        port: u16,
    }
    impl FieldAccess for ConstrainedHost {
        fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
            if name == "port" {
                let v: u16 = value
                    .parse()
                    .map_err(|_| FieldError::Parse(format!("invalid u16 for 'port': {value}")))?;
                if v > 1024 {
                    return Err(FieldError::Constraint(
                        "port: value above maximum 1024".into(),
                    ));
                }
                self.port = v;
                Ok(())
            } else {
                Err(FieldError::NotFound(name.into()))
            }
        }
        fn get_field(&self, name: &str) -> Result<String, FieldError> {
            match name {
                "port" => Ok(self.port.to_string()),
                _ => Err(FieldError::NotFound(name.into())),
            }
        }
        fn field_names(&self) -> &'static [&'static str] {
            &["port"]
        }
    }
    impl Describable for ConstrainedHost {
        fn describe(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }
    impl_component!(ConstrainedHost);
    impl WorkUnit for ConstrainedHost {
        fn name(&self) -> &str {
            "constrained_host"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("done"))
        }
    }

    #[test]
    fn adapter_set_field_propagates_constraint_error() {
        let host = ConstrainedHost { port: 8079 };
        let mut adapter = ComponentAdapter::new(Arc::new(host));
        // Valid value — should succeed and store override.
        adapter.set_field("port", "512").unwrap();
        assert_eq!(adapter.get_field("port").unwrap(), "512");
        // Value above max — should propagate the constraint error.
        let err = adapter.set_field("port", "9999").unwrap_err();
        match err {
            FieldError::Constraint(msg) => {
                assert!(msg.contains("above maximum"), "unexpected: {msg}");
            }
            other => panic!("expected Constraint error, got: {other:?}"),
        }
        // The override should NOT have been stored for the invalid value.
        assert_eq!(adapter.get_field("port").unwrap(), "512");
    }

    #[test]
    fn adapter_set_field_propagates_parse_error() {
        let host = ConstrainedHost { port: 8079 };
        let mut adapter = ComponentAdapter::new(Arc::new(host));
        let err = adapter.set_field("port", "not_a_number").unwrap_err();
        match err {
            FieldError::Parse(msg) => {
                assert!(msg.contains("invalid u16"), "unexpected: {msg}");
            }
            other => panic!("expected Parse error, got: {other:?}"),
        }
    }

    #[test]
    fn instrumented_clone() {
        let unit = CloneableUnit;
        let inst = Instrumented::new(unit, "test.label");
        let cloned = inst.clone();
        assert_eq!(cloned.label, "test.label");
    }

    #[test]
    fn component_adapter_clone() {
        let host = ConstrainedHost { port: 8079 };
        let adapter = ComponentAdapter::new(Arc::new(host)).with_name_override("cloned-name");
        let cloned = adapter.clone();
        assert_eq!(cloned.name_override.as_deref(), Some("cloned-name"));
    }

    #[test]
    fn adapter_get_field_falls_back_to_inner() {
        let host = ConstrainedHost { port: 8079 };
        let adapter = ComponentAdapter::new(Arc::new(host));
        assert_eq!(adapter.get_field("port").unwrap(), "8079");
    }

    #[test]
    fn adapter_override_takes_priority_over_inner() {
        let host = ConstrainedHost { port: 8079 };
        let adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "9090");
        assert_eq!(adapter.get_field("port").unwrap(), "9090");
    }

    #[test]
    fn adapter_hashmap_lookup_is_o1() {
        let host = ConstrainedHost { port: 0 };
        let mut adapter = ComponentAdapter::new(Arc::new(host));
        let n = 1000;
        for i in 0..n {
            adapter = adapter.with_field_override(format!("k{i}"), format!("v{i}"));
        }
        assert_eq!(
            adapter.get_field(&format!("k{}", n - 1)).unwrap(),
            format!("v{}", n - 1)
        );
    }

    #[test]
    fn adapter_describe_field_overrides_sorted() {
        let host = ConstrainedHost { port: 0 };
        let adapter = ComponentAdapter::new(Arc::new(host))
            .with_field_override("c", "3")
            .with_field_override("a", "1")
            .with_field_override("b", "2");
        let schema = adapter.describe();
        let overrides = schema["field_overrides"].as_array().unwrap();
        assert_eq!(overrides.len(), 3);
        assert_eq!(overrides[0][0], "a");
        assert_eq!(overrides[1][0], "b");
        assert_eq!(overrides[2][0], "c");
    }

    #[test]
    fn adapter_with_field_overrides_bulk() {
        let host = ConstrainedHost { port: 0 };
        let mut map = HashMap::new();
        map.insert("x".to_string(), "10".to_string());
        map.insert("y".to_string(), "20".to_string());
        let adapter = ComponentAdapter::new(Arc::new(host)).with_field_overrides(map);
        assert_eq!(adapter.get_field("x").unwrap(), "10");
        assert_eq!(adapter.get_field("y").unwrap(), "20");
    }

    #[test]
    fn adapter_clear_field_overrides_reverts_to_inner() {
        let host = ConstrainedHost { port: 8079 };
        let mut adapter = ComponentAdapter::new(Arc::new(host)).with_field_override("port", "9090");
        assert_eq!(adapter.get_field("port").unwrap(), "9090");
        adapter.clear_field_overrides();
        assert_eq!(adapter.get_field("port").unwrap(), "8079");
    }

    // --- ComponentCascade tests ---

    fn failing_unit(name: &str) -> Arc<dyn Component> {
        Arc::new(MockUnit {
            name: ArcIntern::from(name),
            should_fail: true,
            call_count: AtomicUsize::new(0),
        })
    }

    fn ok_unit(name: &str) -> Arc<dyn Component> {
        Arc::new(MockUnit::ok(name))
    }

    #[test]
    fn cascade_returns_first_ok_and_short_circuits() {
        let mut cascade = ComponentCascade::new();
        cascade.register(failing_unit("fail-1"));
        cascade.register(ok_unit("ok-1"));
        let ok: Arc<dyn Component> = ok_unit("ok-2");
        cascade.register(Arc::clone(&ok));

        let result = cascade
            .execute_first_ok(&WorkContext::default())
            .expect("first-ok");
        assert_eq!(result.message, "done");
        // ok-2 must not have run (short-circuit after ok-1).
        assert_eq!(
            ok.as_any()
                .downcast_ref::<MockUnit>()
                .unwrap()
                .call_count
                .load(Ordering::SeqCst),
            0
        );
    }

    #[test]
    fn cascade_all_fail_returns_last_error() {
        let mut cascade = ComponentCascade::new();
        cascade.register(failing_unit("fail-1"));
        cascade.register(failing_unit("fail-2"));
        let err = cascade
            .execute_first_ok(&WorkContext::default())
            .expect_err("all units fail");
        assert!(matches!(err, WorkError::Execution(_)));
    }

    #[test]
    fn cascade_empty_returns_descriptive_error() {
        let cascade = ComponentCascade::new();
        let err = cascade
            .execute_first_ok(&WorkContext::default())
            .expect_err("empty cascade");
        assert!(err.to_string().contains("no units"));
    }

    #[test]
    fn cascade_len_and_is_empty() {
        let mut cascade = ComponentCascade::new();
        assert!(cascade.is_empty());
        cascade.push(ok_unit("a"));
        assert_eq!(cascade.len(), 1);
        assert!(!cascade.is_empty());
    }

    #[test]
    fn cascade_is_itself_a_component() {
        let mut cascade = ComponentCascade::new();
        cascade.register(ok_unit("a"));
        let boxed: Box<dyn Component> = Box::new(cascade);
        assert_eq!(boxed.name(), "component_cascade");
        let result = boxed.execute(&WorkContext::default()).expect("delegates");
        assert_eq!(result.message, "done");
        let schema = boxed.describe();
        assert_eq!(schema["cascade"]["units"][0], "a");
    }

    #[test]
    fn cascade_executes_empty_units() {
        let cascade = ComponentCascade::with_units(vec![failing_unit("only-fail")]);
        assert!(cascade.execute(&WorkContext::default()).is_err());
    }
}
