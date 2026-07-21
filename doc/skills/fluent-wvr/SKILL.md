# Fluent WVR in Rust — The Synthesis Pipeline

**A design pattern guide for human coders and AI agents working in the Rust codebase.**

---

## Who This Document Is For

**For humans coming from Python or C++**: Rust's trait system replaces inheritance, its ownership model replaces garbage collection, and its derive macros replace runtime reflection. This document shows how twelve composable patterns together give you Python's runtime ergonomics and C++'s polymorphic flexibility, with absolute memory safety and zero hidden cost.

**For AI agents**: This document is the authoritative reference for how code in `rust-src/` is structured. When writing new code, check the pattern table in §1, find the applicable pattern, and follow the rules and anti-pattern sections. All patterns are in production use in the codebase.

---

## The Core Thesis

Every unit of work in this system — a DAG target, a WASM plugin, a query strategy, an embedding provider — presents the **same interface** to the orchestrator regardless of whether it was assembled at compile time or at runtime. The orchestrator never branches on implementation type. It iterates over uniform handles and calls trait methods. The compiler enforces the interface at every implementation site.

The `Component` supertrait is the concrete expression of this principle:

```rust
pub trait Component: FieldAccess + Describable + WorkUnit + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// `impl_component!` writes the `as_any`/`as_any_mut` body — never hand-roll it.
impl_component!(MyConfig);
```

The `as_any`/`as_any_mut` methods are required for safe downcasting after
`Arc<dyn Component>` erasure. The `impl_component!` macro eliminates the
7-line boilerplate (it has a concrete-type arm and a `generic (bounds)`
arm for wrapper types). See §2 for the full definition and §5 for
wrapper usage.

This interface is valid for **both compile-time and runtime assembly**:

| Construction path | Resulting type | Interface |
|---|---|---|
| `bon::Builder` on a Rust struct | `Arc<dyn Component>` | Same |
| WASM plugin loaded at runtime | `Arc<dyn Component>` (via `WasmComponent` bridge) | Same |
| Database-driven config | `Arc<dyn Component>` (via `DynamicComponent`) | Same |
| Newtype wrapper at registration | `Arc<dyn Component>` | Same |
| `ComponentAdapter` at runtime | `Arc<dyn Component>` | Same |

The key design constraint: **construction reads like configuration, validation surfaces at exactly one point, dispatch is uniform, serialization is automatic, runtime configuration is type-safe, and cross-cutting concerns compose without modifying business logic.**

### The Full Pipeline

```
Developer writes:   Target::builder().name("build").depends(bits).build()
                           ↓
bon::Builder:      Generates chained setter methods at compile time (zero cost)
                           ↓
Validation:        build() returns Result; ? propagation at call site (type safety)
                           ↓
FieldAccess:       component.set_field("port", "9000")?  (runtime configuration)
                           ↓
Trait Object:      Arc<dyn Component> stored in registry (uniform interface)
                           ↓
Runtime:           Orchestrator calls unit.execute() — no branching needed
```

---

## 1. Pattern Table

Twelve patterns compose into a single coherent architecture.

| Pattern | Problem solved | Cost | Primary source |
|---------|---------------|------|---------------|
| **Fluent Builder** | Multi-parameter init that callers can't read | Zero — `#[derive(bon::Builder)]` | `dag/src/target.rs` |
| **Trait-Based Reflection** | Schema defined in multiple places; runtime config by name | Zero — `serde` + `FieldAccess` derive | `fluent-wvr-macros/src/lib.rs` |
| **Trait Composition** | Cross-cutting logic duplicated across handlers | Zero — newtype wrappers | `fluent-wvr/src/wrapper.rs` |
| **Trait Objects** | Runtime polymorphism with branching in hot loops | One vtable pointer per `dyn Trait` | `llm/src/embeddings.rs` |
| **Binary IPC** | Executing untrusted code across a WASM boundary | Memcpy + `#[repr(C, packed)]` | `wasm_ipc/src/lib.rs` |
| **Scoped Ownership** | Repeated malloc/free in batch processing | Zero — RAII scope drop | Throughout |
| **Newtype Handles** | ID type confusion across module boundaries | Zero — same representation | `guidance-types/src/lib.rs` |
| **Unit of Work** | Uniform orchestration of heterogeneous tasks | One trait impl per task | `dag/src/work_unit.rs` |
| **Middleware Chain** | Composable cross-cutting on trait objects | One allocation per layer | `dag/src/middleware.rs` |
| **Component Adapter** | Runtime type adaptation without losing uniform interface | One `Arc` per adapter | `dag/src/adapter.rs` |
| **Structured Logging Context** | Request-scoped observability without manual context passing | Thread-local storage | `fluent-wvr/src/wrapper.rs` (Instrumented) |
| **Runtime Composition** | Full lifecycle: build → configure → wrap → execute → inspect | Zero — uses above patterns | Throughout |

---

## 2. The Component Interface in Detail

Understanding this section is a prerequisite for every pattern below. All patterns either produce a `Component`, consume one, or compose with one.

### Trait definitions

```rust
/// Runtime field access by name with validation.
/// Generated by #[derive(FieldAccess)] for compile-time-known structs.
/// Implemented manually for WASM plugins and DB-driven schemas.
pub trait FieldAccess {
    /// Set a field by name from a string value. Performs type parsing and validation.
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError>;
    /// Get a field by name as a string.
    fn get_field(&self, name: &str) -> Result<String, FieldError>;
    /// All field names for this type. Used by schema generators and TUI editors.
    fn field_names(&self) -> &'static [&'static str];
}

/// Schema description for MCP tool parameter validation and TUI editors.
/// Must be implemented alongside every FieldAccess implementation.
pub trait Describable {
    fn describe(&self) -> serde_json::Value;
}

/// Errors produced by `FieldAccess` implementations.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum FieldError {
    #[error("field not found: {0}")]
    NotFound(String),
    #[error("field parse error: {0}")]
    Parse(String),
    #[error("constraint violation: {0}")]
    Constraint(String),
    /// Returned by the `Arc<dyn Component>` blanket `set_field` impl when
    /// the Arc has multiple owners. Callers who need shared mutation
    /// should either (a) use `ComponentArcExt::try_as_any_mut` and
    /// handle `None`, or (b) use a wrapper with interior mutability.
    #[error("field {0:?} is read-only on a shared Arc: {1}")]
    ReadOnly(String, String),
}

/// Optional JSON-Schema-style description of a single field. Returned
/// by `SchemaProvider::schema`. The derive macro on `FieldAccess` does
/// **not** auto-generate `SchemaProvider`; implement it explicitly when
/// MCP tool validation or TUI editors need typed schema metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub type_name: String,
    pub description: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub required: bool,
    pub format: Option<String>,
    pub max_len: Option<usize>,
    pub sanitize: Option<String>,
    pub pattern: Option<String>,
}
pub trait SchemaProvider {
    fn schema(&self) -> Vec<FieldSchema>;
}

/// Uniform orchestration interface. Every task the orchestrator executes implements this.
pub trait WorkUnit: Send + Sync {
    fn name(&self) -> &str;
    fn depends(&self) -> &[ArcIntern<str>];
    fn provides(&self) -> &[ArcIntern<str>];
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError>;

    /// Default per-unit timeout. Override to specialise.
    /// The `Zone` supervisor reads this via `WorkContext::for_unit` and
    /// enforces it through `execute_with_timeout_and_retry`.
    fn default_timeout_ms(&self) -> u64 { 30_000 }

    /// Concrete type name. Default uses `std::any::type_name::<Self>()`
    /// — useful for logging and metrics aggregation.
    fn type_name(&self) -> &'static str { std::any::type_name::<Self>() }
}

/// The unified process boundary. Requires as_any/as_any_mut for runtime
/// type identification after dyn Component erasure.
pub trait Component: FieldAccess + Describable + WorkUnit + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Use the `impl_component!` macro instead of hand-writing `as_any`/`as_any_mut`:
///
/// ```ignore
/// impl_component!(MyConfig);                       // concrete type
/// impl_component!(generic (U: Component + 'static) for Instrumented<U>);  // generic
/// ```
///
/// Two arms: the concrete-type form writes `impl Component for $type`
/// directly; the `generic (bounds) for Type<…>` form writes
/// `impl <$($generics)*> Component for $type` so wrapper types like
/// `Instrumented<U>` and `WithRetry<U>` can satisfy the supertrait
/// without copy-pasting the same body.

/// Free functions for downcasting (no blanket impl — Rust dyn-compatibility).
pub fn component_downcast_ref<T: 'static>(comp: &dyn Component) -> Option<&T>;
pub fn component_downcast_mut<T: 'static>(comp: &mut dyn Component) -> Option<&mut T>;

/// Extension trait for safe mutable downcasting through an `Arc<dyn Component>`.
pub trait ComponentArcExt {
    fn try_as_any_mut(&mut self) -> Option<&mut dyn Any>;
}
impl ComponentArcExt for Arc<dyn Component> {
    fn try_as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Arc::get_mut(self).map(Component::as_any_mut)
    }
}

/// `Arc<dyn Component>` and `Arc<dyn WorkUnit>` themselves satisfy the
/// full supertrait surface. The blanket impls in `lib.rs:50-129` make
/// every `Arc<dyn Component>` callable as a `Component`/`WorkUnit`/
/// `FieldAccess`/`Describable`. The `set_field` impl returns
/// `FieldError::ReadOnly(name, reason)` when the `Arc` is shared
/// (i.e. `Arc::get_mut` returns `None`).
```

### Error and output types

`WorkError` and `WorkOutput` are the data flow types for `WorkUnit::execute`.

```rust
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WorkError {
    /// Synchronous failure inside `execute`. The `Zone` supervisor
    /// routes this to `summary.failed` — NOT to `summary.panicked`.
    #[error("execution failed: {0}")]
    Execution(String),

    /// Dependency prerequisite could not be satisfied. Used by the
    /// orchestrator when a `depends()` asset is not yet `provides()`'d.
    #[error("dependency not satisfied: {0}")]
    Dependency(String),

    /// Returned by `Zone::execute_with_timeout_and_retry` when the
    /// wall-clock budget is exhausted. `Zone` routes this to
    /// `summary.cancelled`, NOT to `summary.failed`.
    #[error("timeout after {duration_ms}ms ({unit})")]
    Timeout { duration_ms: u64, unit: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkOutput {
    pub success: bool,
    pub message: String,
    pub data: serde_json::Value,
}

impl WorkOutput {
    pub fn ok(message: impl Into<String>) -> Self;
    pub fn ok_with_data(message: impl Into<String>, data: serde_json::Value) -> Self;
    pub fn fail(message: impl Into<String>) -> Self;

    /// Serialise a typed value into `data` and return `Ok(Self)`.
    /// Returns `Err(WorkError::Execution)` on serialisation failure —
    /// use this for untrusted input.
    pub fn typed<T: Serialize>(message: impl Into<String>, data: &T)
        -> Result<Self, WorkError>;

    /// Same as `typed` but panics on serialisation failure. For
    /// trusted input only.
    pub fn typed_infallible<T: Serialize>(message: impl Into<String>, data: &T) -> Self;

    /// Deserialise `data` to a typed value (borrows `self`).
    pub fn data_as<T: for<'de> Deserialize<'de>>(&self) -> Result<T, WorkError>;

    /// Deserialise `data`, consuming `self` — avoids the clone that
    /// `data_as` requires.
    pub fn data_take<T: for<'de> Deserialize<'de>>(self) -> Result<T, WorkError>;
}

impl std::fmt::Display for WorkOutput {
    // "OK: msg" or "FAIL: msg"
}
```

### Execution context

`WorkContext` carries the per-call configuration plus a `Runtime` and
a `CapabilitySet`. The default uses `NoopRuntime` and an empty
capability set; `Zone` overrides both at registration time.

```rust
#[derive(Clone)]
pub struct WorkContext {
    pub dry_run: bool,                   // short-circuit mutating handlers
    pub max_retries: u32,                // 0 = no retry; >0 = Zone retries this many times
    pub timeout_ms: u64,                 // per-attempt wall-clock budget
    pub metadata: HashMap<String, MetadataValue>,
    pub rt: Arc<dyn Runtime>,            // pluggable: TokioRuntime, TestRuntime, NoopRuntime
    pub caps: CapabilitySet,             // type-erased capability tokens
}

impl Default for WorkContext { /* dry_run=false, max_retries=0, timeout_ms=30_000, NoopRuntime */ }

impl WorkContext {
    /// Build a context for a single unit, using its `default_timeout_ms()`.
    pub fn for_unit(unit: &dyn WorkUnit, caps: CapabilitySet) -> Self;

    /// Build a context that inherits the zone's `rt` and `caps`, with
    /// optional per-unit overrides via a closure. The intended
    /// registration-site helper — `Zone::register` uses it internally.
    pub fn for_unit_in_zone(
        zone_rt: &Arc<dyn Runtime>,
        zone_caps: &CapabilitySet,
        mutate: impl FnOnce(&mut WorkContext),
    ) -> Self;
}
```

### The canonical prelude

```rust
use fluent_wvr::prelude::*;
// Brings in: Component, WorkUnit, FieldAccess, Describable, FieldError,
// WorkContext, WorkError, WorkOutput, Capability, CapabilitySet,
// impl_component, retry_call, ComponentAdapter, ExecuteFn,
// Instrumented, WithRetry, MetadataValue, ArcIntern.
```

> `ComponentArcExt`, `FieldSchema`, `SchemaProvider`, and
> `PersistableComponent` are **not** in the prelude — they are niche
> types that should be imported from `fluent_wvr::traits` or
> `fluent_wvr::capability` when actually needed. The M2 milestone
> removed them from the prelude because no consumer in the in-tree
> call graph imported them through the prelude.

> **Critical design note on `field_names` and `describe`:** Both are defined as `&self` instance methods, not static associated functions. This is deliberate: it makes them callable through `dyn Component`. A `fn field_names() -> &'static [&'static str]` associated function is not object-safe and cannot be dispatched through a trait object. Always use the instance method form.

### Why `describe` is not `json_schema() -> serde_json::Value` (static)

The original design used a static associated function `fn json_schema() -> serde_json::Value`. That form is **not object-safe** — you cannot call it through `Arc<dyn Component>` or `Arc<dyn Describable>`. The turbofish syntax `<ToolConfig as Describable>::json_schema()` only works when the concrete type is known at the call site.

The corrected `describe(&self)` method is callable through any trait object:

```rust
// This works with the instance-method form:
let schema = component.describe();  // works on Arc<dyn Component>

// This does NOT work if describe is a static associated function:
// let schema = component.describe();  // compile error: not object-safe
```

### FieldAccess mutability and interior mutability

`set_field` takes `&mut self`, which requires mutable access to the concrete type. For trait objects stored in shared registries (`Arc<dyn Component>`), this means callers need `Arc::get_mut` (exclusive ownership) or the implementation must use interior mutability:

```rust
// Pattern A: exclusive ownership (configure before sharing)
let mut component = ToolConfig::builder().port(8080).build()?;
component.set_field("port", "9000")?;
let shared: Arc<dyn Component> = Arc::new(component);

// Pattern B: interior mutability inside the implementation
pub struct WasmComponent {
    config: Mutex<HashMap<String, String>>,  // interior mutability
    plugin: Mutex<extism::Plugin>,
}

impl FieldAccess for WasmComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        // self is &mut, but we could also offer a set_field_shared(&self, ...) variant
        self.config.lock().unwrap().insert(name.to_string(), value.to_string());
        Ok(())
    }
}
```

The rule: **configure Rust-struct components before wrapping in `Arc`; configure WASM/dynamic components through their internal `Mutex`.**

---

## 3. Pattern 1 — Fluent Builder

### The problem

```cpp
// C++: Can you tell what the 4th argument means?
Target t("build", TargetType::File, {"compile", "link"}, {"artifact"}, true, "zig build");
```

```python
# Python: kwargs help, but errors surface at runtime, mid-construction
target = Target(name="build", depends=["compile"], provides=["artifact"], essential=True)
```

### The Rust solution

`#[derive(bon::Builder)]` generates zero-boilerplate fluent builders. Validation moves to `build()` or a separate `register()` step. The `?` operator replaces error accumulation.

### Canonical implementation: `Target` in `dag/src/target.rs`

```rust
use bon::Builder;
use bitvec::vec::BitVec;
use internment::ArcIntern;

#[derive(Debug, Clone, Builder)]
#[builder(start_fn = new)]
pub struct Target {
    pub id: i64,
    pub name: ArcIntern<str>,
    pub target_type: TargetType,
    pub executor: ExecutorKind,
    pub depends: BitVec,
    pub provides: BitVec,
    #[builder(default)]
    pub command: String,
    #[builder(default = false)]
    pub essential: bool,
}
```

**Call site:**

```rust
let target = Target::new()
    .id(1)
    .name("build".into())
    .target_type(TargetType::File)
    .executor(ExecutorKind::Native)
    .depends(registry.capabilities().to_bitvec(&["compile", "link"]))
    .provides(registry.capabilities().to_bitvec(&["artifact"]))
    .command("cargo build --release".into())
    .essential(true)
    .build();

registry.register(target)?;
```

### Fallible two-step construction

When construction involves fallible validation (string interning into bitset indices), separate infallible argument collection from fallible registration:

```rust
// Step 1: infallible — bon builder collects arguments
let args = TargetCreateArgs::builder()
    .name("build".into())
    .target_type(TargetType::File)
    .build();

// Step 2: fallible — registry validates and allocates
let target = registry.validate_and_allocate(args)?;
```

### Rules

- **Always derive `bon::Builder`** for structs with 4+ fields. Never write manual builder structs.
- **Use `#[builder(default)]`** for optional fields. Use `#[builder(default = value)]` for non-trivial defaults.
- **Use `#[builder(start_fn = new)]`** to generate `Type::new()` as the entry point.
- **Validation belongs in `build()` or `register()`, not in setters.**
- **Decouple construction from registration.** `Target::builder().build()` produces an owned `Target`; `registry.register(target)` commits it.
- **Do NOT apply to structs with 2–3 parameters.** Three params are already readable.

### When to avoid

- **2–3 parameters with no validation:** A struct literal or `fn new()` is already readable.
- **Performance-critical construction in hot loops:** Construction of millions of instances per second — use direct construction.
- **Single-use internal structs:** If the struct is only constructed in one place and never exposed.

---

## 4. Pattern 2 — Trait-Based Reflection

### The Boundary Rule

**Data that arrives from outside the process is always a string at the boundary. Data moving inside the process is not a string and should never be treated as one.**

This governs which of the four reflection tiers to use.

### The problem

```python
# Python: getattr/setattr at runtime — convenient, but:
# - No compile-time checking
# - No permission model
# - No schema description
# - String hashing at every access
setattr(config, "port", 9000)
```

### Four tiers of reflection

| Tier | Mechanism | Relative cost | When to use |
|------|-----------|---------------|-------------|
| 1 | Direct field access | 1x (baseline) | Hot inner loops, trusted internal code |
| 2 | `serde` boundary serialization | ~10x | Boundary crossing, type known at compile time |
| 3 | `FieldAccess` trait | ~20x | Field names from runtime data; config editors, WASM configs |
| 4 | `HashMap` fallback | ~12–20x† | Schema genuinely unknown until runtime |

†`HashMap` cost is dominated by hashing and heap allocation, not a string match. For types with many fields, a generated `match` (Tier 3) is often faster than `HashMap::get` (Tier 4). Do not assume Tier 4 is always cheaper than Tier 3.

#### Tier 1: Direct field access

```rust
config.port = 9001;
let port = config.port;
```

#### Tier 2: `serde` boundary serialization

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: ArcIntern<str>,
}

fn default_port() -> u16 { 8080 }
fn default_host() -> ArcIntern<str> { "localhost".into() }

let config: Config = serde_json::from_str(json)?;
```

#### Tier 3: `FieldAccess` derive macro

```rust
#[derive(FieldAccess, Describable, bon::Builder)]
pub struct ToolConfig {
    #[field(desc = "TCP listen port", min = 1, max = 65535)]
    pub port: u16,
    #[field(desc = "Host address", max_len = 255)]
    pub host: String,
    #[field(desc = "Enable verbose logging")]
    pub verbose: bool,
    #[field(desc = "API endpoint URL", pattern = "https://")]
    pub endpoint: String,
    #[field(desc = "Display name", sanitize = "trim,lowercase")]
    pub name: String,
}
```

> **Note:** Patterns are substring matches, not regex. The value must contain the pattern string.

The derive macro generates this implementation:

```rust
impl FieldAccess for ToolConfig {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "port" => {
                let wide: f64 = value.parse().map_err(|_| FieldError::Parse(
                    format!("invalid u16 for 'port': {}", value)
                ))?;
                if wide < 1.0 {
                    return Err(FieldError::Constraint("port: value below minimum 1".into()));
                }
                if wide > 65535.0 {
                    return Err(FieldError::Constraint("port: value above maximum 65535".into()));
                }
                self.port = wide as u16;
                Ok(())
            }
            "host" => { self.host = value.to_string(); Ok(()) }
            "verbose" => {
                self.verbose = value.parse().map_err(|_| FieldError::Parse(
                    format!("invalid bool for 'verbose': {}", value)
                ))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into()))
        }
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "port"    => Ok(self.port.to_string()),
            "host"    => Ok(self.host.clone()),
            "verbose" => Ok(self.verbose.to_string()),
            _ => Err(FieldError::NotFound(name.into()))
        }
    }

    fn field_names(&self) -> &'static [&'static str] {
        &["port", "host", "verbose"]
    }
}

impl Describable for ToolConfig {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "port":    { "type": "integer", "minimum": "1", "maximum": "65535",
                             "description": "TCP listen port" },
                "host":    { "type": "string", "description": "Host address" },
                "verbose": { "type": "boolean", "description": "Enable verbose logging" }
            },
            "required": ["port", "host", "verbose"]
        })
    }
}
```

#### Tier 4: `HashMap` fallback

When the schema is genuinely unknown at compile time (WASM plugin exports its schema, database-driven schemas):

```rust
use std::collections::HashMap;
use internment::ArcIntern;

pub type DynamicConfig = HashMap<ArcIntern<str>, serde_json::Value>;

config.insert("port".into(), serde_json::json!(9000));

let port: u16 = config.get("port")
    .and_then(|v| v.as_u64())
    .ok_or(ConfigError::MissingField("port"))? as u16;
```

### Decision tree

```
Is the access in a hot loop?
  YES → Tier 1: direct field access
  NO  → Is data arriving from outside the process?
    YES → Is the type known at compile time?
      YES → Tier 2: serde (Deserialize)
      NO  → Is the schema known at runtime but values are typed?
        YES → Implement FieldAccess manually (Tier 3)
        NO  → Tier 4: HashMap<ArcIntern<str>, serde_json::Value>
    NO  → Tier 1: direct field access

Is a field name known only at runtime (config editor, WASM config)?
  YES → Tier 3: FieldAccess
  NO  → Tier 1 or Tier 2 per above
```

### Rules

- **Every `FieldAccess` implementation must also implement `Describable`.** The schema is the single source of truth — field access without a schema description is incomplete.
- **Use `serde` for all boundary serialization.** JSON, database rows, RPC.
- **Never call `set_field` in a hot loop.** Use Tier 1.
- **Never add `FieldAccess` to a type that never crosses a boundary.** It adds code for no benefit.

---

## 5. Pattern 3 — Trait Composition (Cross-Cutting Concerns)

### The problem

```python
@timing
@retry(max=3)
def ingest_yago(path: str) -> None:
    # business logic
```

Python decorators execute at import time and wrap functions transparently. They're runtime closures and can be hard to type correctly.

### The Rust solution

Newtype wrappers around trait implementations. Each wrapper implements the same trait, delegating to the inner type while adding its cross-cutting concern. The compiler monomorphizes or dispatches through `dyn Trait`.

### Canonical shape: `Instrumented<U>`

```rust
pub struct Instrumented<U> {
    inner: U,
    label: String,                                     // owned, not &'static str
    histogram: Option<Arc<LatencyHistogram>>,          // optional metrics sink
}

impl<U: WorkUnit> Instrumented<U> {
    pub fn new(inner: U, label: impl Into<String>) -> Self { /* … */ }
    pub fn with_metrics(
        inner: U, label: impl Into<String>, histogram: Arc<LatencyHistogram>,
    ) -> Self { /* … */ }
}

impl<U: WorkUnit> WorkUnit for Instrumented<U> {
    fn name(&self) -> &str { self.inner.name() }   // delegates, does NOT shadow
    fn depends(&self) -> &[ArcIntern<str>] { self.inner.depends() }
    fn provides(&self) -> &[ArcIntern<str>] { self.inner.provides() }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let start = Instant::now();
        let result = self.inner.execute(ctx);
        if let Some(ref hist) = self.histogram {
            hist.observe_duration(start);
        }
        info!(target: "instrumented", label = %self.label,
              elapsed = ?start.elapsed(), name = %self.inner.name(), "executed");
        result
    }
}
impl_component!(generic (U: Component + 'static) for Instrumented<U>);
```

The wrapper also implements `FieldAccess` (delegates `get_field`/`field_names` to the inner; `set_field` returns `FieldError::NotFound` so callers configure the inner before wrapping) and `Describable` (delegates to inner). The `name()` call passes through to the inner type — `Instrumented` does not rename the unit; it only adds observability.

### Retry wrapper

```rust
pub struct WithRetry<U> {
    inner: U,
    max_attempts: u32,
    base_ms: u64,
    jitter_pct: u32,
}

impl<U: WorkUnit> WithRetry<U> {
    pub fn new(inner: U, max_attempts: u32, backoff_ms: u64) -> Self { /* … */ }
    pub fn new_jittered(
        inner: U, max_attempts: u32, base_ms: u64, jitter_pct: u32,
    ) -> Self { /* … */ }
}

impl<U: WorkUnit> WorkUnit for WithRetry<U> {
    fn name(&self) -> &str { self.inner.name() }
    fn depends(&self) -> &[ArcIntern<str>] { self.inner.depends() }
    fn provides(&self) -> &[ArcIntern<str>] { self.inner.provides() }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        // Linear backoff with configurable jitter; uses std::thread::sleep,
        // NOT tokio::time::sleep — sync work-unit contract.
        let mut attempts = 0u32;
        loop {
            attempts += 1;
            match self.inner.execute(ctx) {
                Ok(output) => return Ok(output),
                Err(e) if attempts < self.max_attempts => {
                    let jitter = if self.jitter_pct > 0 {
                        fastrand::u64(0..self.base_ms * self.jitter_pct as u64 / 100)
                    } else { 0 };
                    std::thread::sleep(Duration::from_millis(
                        self.base_ms * attempts as u64 + jitter,
                    ));
                }
                Err(e) => return Err(e),
            }
        }
    }
}
impl_component!(generic (U: Component + 'static) for WithRetry<U>);
```

`WithRetry` is a **synchronous** retry — it sleeps with `std::thread::sleep`
inside the executor, which is allowed because `WorkUnit::execute` is
documented as a sync boundary. For async retry semantics, use the
`Zone` supervisor (M6) instead, which routes the underlying error to
`ZoneSummary::failed` and runs `execute` on a fresh task budget per
attempt.

### Application at the registration site

Apply wrappers **before** type erasure — this is the only point where the compiler can inline through the wrapper. Once stored as `Arc<dyn WorkUnit>`, you must use Middleware (Pattern 9) instead.

```rust
// Compose before type erasure: wraps are inlined
let unit = Instrumented::new(
    WithRetry::new(MyWorkUnit::new(), /* max_attempts: */ 3, /* backoff_ms: */ 50),
    "ingest_yago",
);
registry.push(Arc::new(unit));   // one vtable boundary total
```

### Composition order (outer to inner)

| Layer | Purpose |
|-------|---------|
| 1 | Rate limiting — reject early if overloaded |
| 2 | Auth — reject early if unauthorized |
| 3 | Tracing — start span |
| 4 | Timing — measure full duration |
| 5 | Retry — retry on transient failure |
| 6 | Validation — validate input |
| 7 | Core handler |

### Rules

- **Wrappers must implement the same trait as the inner type.**
- **Apply wrappers at the registration site, before type erasure.**
- **Do NOT wrap when there is only one implementation.**
- **Use `impl Trait` or generics for hot paths** to allow inlining.

### When to avoid

- After type erasure (use Middleware, Pattern 9).
- When the wrapper doesn't add any cross-cutting concern.
- When the wrapper changes the interface (use Adapter, Pattern 10).

---

## 6. Pattern 4 — Trait Objects (Runtime Polymorphism)

### The problem

```cpp
// C++: vtable per class hierarchy, hidden in the ABI
class Engine { virtual std::vector<Row> query(std::string sql) = 0; };
```

```python
# Python: duck typing — convenient but zero static safety
def run_query(engine, sql):
    return engine.query(sql)  # crashes at runtime if wrong method signature
```

### The Rust solution

`dyn Trait` + `Arc<dyn Trait>` for shared ownership. The compiler generates the vtable. `Send + Sync` bounds enforce thread safety at every implementation site.

### Canonical implementation: `EmbeddingProvider`

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn dimensions(&self) -> u32;
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError>;
}

pub struct ProviderRegistry {
    providers: Vec<Arc<dyn EmbeddingProvider>>,
}

impl ProviderRegistry {
    pub fn register(&mut self, provider: Arc<dyn EmbeddingProvider>) {
        self.providers.push(provider);
    }
    pub fn get(&self, name: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        self.providers.iter().find(|p| p.name() == name).cloned()
    }
}
```

### Trait objects vs. generics

| Scenario | Use | Why |
|----------|-----|-----|
| 2+ implementations, stored in registry | `dyn Trait` + `Arc` | Uniform handles, shared ownership |
| Single implementation, hot loop | `<T: Trait>` or `impl Trait` | Zero-cost monomorphization |
| Single implementation, not hot | Concrete type | No indirection needed |
| Plugin system (WASM, dynamic load) | `dyn Trait` + `Arc` | Runtime-discovered implementations |

### Serializing trait objects

Rust cannot serialize `dyn Trait` directly. Use a tagged enum wrapper:

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    Ollama { model: String, base_url: String },
    OpenAi { model: String, api_key: String },
}
```

### Component serialization (deferred)

Adding `Serialize` to the `Component` trait would require every implementor to be serializable — a breaking change for types holding `Arc<dyn Provider>`, `Mutex<Plugin>`, etc. A `PersistableComponent` trait stub exists in `fluent-wvr/src/lib.rs` as a design placeholder. Do NOT implement it until a second consumer materializes that needs to persist component state.

### Rules

1. **Always add `Send + Sync`** to traits stored in registries or shared across threads.
2. **Use `Arc<dyn Trait>` for shared ownership.** Use `Box<dyn Trait>` only for exclusive ownership.
3. **Never use `dyn Trait` with only one implementation.** Use the concrete type.
4. **Never create a trait speculatively.** Start with a concrete type; add the trait when the second implementation arrives.

---

## 7. Pattern 5 — Binary IPC (`#[repr(C, packed)]`)

### The problem

When executing untrusted or dynamically loaded code (WASM tools), you need a safe, portable, zero-copy message format that works across the host/guest boundary.

### The Rust solution

`#[repr(C, packed)]` removes padding. Encode/decode explicitly with `to_le_bytes()` / `from_le_bytes()`. Validate magic and version before reading any other field. Never use `transmute`.

### Struct definitions in `wasm_ipc/src/lib.rs`

```rust
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub payload_type: u32,
    pub payload_size: u32,
    pub checksum: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryExecutionRequest {
    pub header: BinaryHeader,
    pub target_id: i64,
    pub input_offset: u32,
    pub input_len: u32,
    pub flags: u32,
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BinaryExecutionResult {
    pub header: BinaryHeader,
    pub success: u32,
    pub error_code: u32,
    pub output_offset: u32,
    pub output_len: u32,
    pub provides_words_offset: u32,
    pub provides_words_count: u32,
}
```

### Encoding and decoding

```rust
pub fn encode_request(req: &BinaryExecutionRequest, input: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(
        std::mem::size_of::<BinaryExecutionRequest>() + input.len()
    );
    buf.extend_from_slice(&req.header.magic);
    buf.extend_from_slice(&req.header.version.to_le_bytes());
    buf.extend_from_slice(&req.header.payload_type.to_le_bytes());
    buf.extend_from_slice(&req.header.payload_size.to_le_bytes());
    buf.extend_from_slice(&req.header.checksum.to_le_bytes());
    buf.extend_from_slice(&req.target_id.to_le_bytes());
    buf.extend_from_slice(&req.input_offset.to_le_bytes());
    buf.extend_from_slice(&req.input_len.to_le_bytes());
    buf.extend_from_slice(&req.flags.to_le_bytes());
    buf.extend_from_slice(input);
    buf
}

pub fn get_provides_bitset(
    result: &BinaryExecutionResult,
    payload: &[u8],
) -> Result<BitVec, IpcError> {
    // SAFETY: packed struct fields must be read with read_unaligned to avoid
    // undefined behavior from misaligned access.
    let count = unsafe {
        std::ptr::addr_of!(result.provides_words_count).read_unaligned()
    } as usize;
    let offset = unsafe {
        std::ptr::addr_of!(result.provides_words_offset).read_unaligned()
    } as usize;

    let mut bits = BitVec::with_capacity(count * 64);
    for i in 0..count {
        let start = offset + i * 8;
        let word = u64::from_le_bytes(
            payload[start..start + 8].try_into().map_err(|_| IpcError::BufferTooSmall)?
        );
        for bit in 0..64 { bits.push((word >> bit) & 1 == 1); }
    }
    Ok(bits)
}
```

### Rules

1. **Always use `#[repr(C, packed)]`** for structs that cross the Extism boundary.
2. **Always encode/decode explicitly** with `to_le_bytes()` / `from_le_bytes()`. Never `transmute`.
3. **Validate magic and version** before reading any other field.
4. **Offsets are absolute from buffer start**, never relative to struct end.
5. **Variable-length data** is appended after the fixed header. The header contains `offset` and `len` fields.
6. **Use `read_unaligned`** for packed struct field access — this is the only required `unsafe`.

### When to avoid

- **Internal structs that never cross a boundary:** `#[repr(C, packed)]` prevents optimizations.
- **Human-readable protocols:** Use JSON or text.
- **When the schema changes frequently:** Use Protocol Buffers or FlatBuffers.

---

## 8. Pattern 6 — Scoped Ownership (Replacing Arenas)

### The problem

Rust has no built-in arena allocator (and `bumpalo` is forbidden by project policy). Intermediate allocations are scoped to logical units of work via RAII ownership.

### The Rust solution

**Trust the stack and `Drop`.** Ownership scopes naturally replicate the arena pattern. All intermediate `Vec`s and `String`s owned by local variables are dropped at scope exit.

### Per-request scoping

```rust
async fn handle_request(&self, raw_json: &str) -> Result<String, McpError> {
    let req: JsonRpcRequest = serde_json::from_str(raw_json)?;
    let result = self.reactor.route(&req.query).await?;
    // `req` and all intermediates are dropped at end of scope
    Ok(serde_json::to_string(&result)?)
}
```

### Batch processing

```rust
{
    let mut mapper = TripleMapper::new(&library, &config);
    mapper.process_triples(triples)?;
    mapper.flush()?;  // escaped data cloned into library
}  // all locals dropped automatically
```

### Rules

- **Do not pre-allocate reusable buffers** unless profiling proves allocation is a bottleneck.
- **If you want an arena, use a scoped function** with local `Vec`s.
- **Builder ownership transfer:** `build()` consumes the builder and returns an owned struct.

---

## 9. Pattern 7 — Newtype Handles (Typed Opaque IDs)

### The problem

```python
NodeId = int
SessionId = int
process_node(session_id)  # Accepted silently — crashes later
```

### The Rust solution

Newtype wrappers create distinct types that share the integer representation. The compiler rejects mixing `NodeId` with `SessionId` at every call site.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetId(pub i64);

impl NodeId {
    pub fn from_int(i: i64) -> Self { Self(i) }
    pub fn as_int(self) -> i64 { self.0 }
}
```

```rust
fn process_node(id: NodeId) { ... }

let node = NodeId::from_int(42);
let sess = SessionId::from_int(42);
process_node(node);  // OK
process_node(sess);  // Compile error: expected NodeId, found SessionId
```

### Rules

- **Use for any integer ID that crosses module boundaries** and must not be mixed with other IDs.
- **Do NOT use when you need arithmetic on the ID.** Use a raw integer for counters.
- **Derive `Serialize, Deserialize`** so the newtype is transparent in JSON/DB.

---

## 10. Pattern 8 — Unit of Work (Orchestration Interface)

### The problem

An orchestrator (DAG executor, MCP server, WASM plugin host) must execute heterogeneous tasks uniformly. Without a common interface, the orchestrator branches on implementation type.

### Definition

`WorkUnit`, `WorkContext`, `WorkOutput`, and `WorkError` live in
`fluent-wvr/src/work.rs` and `fluent-wvr/src/traits.rs`. The full
shapes are listed in §2. The unit-of-work pattern itself is short:

```rust
pub trait WorkUnit: Send + Sync {
    fn name(&self) -> &str;
    fn depends(&self) -> &[ArcIntern<str>];
    fn provides(&self) -> &[ArcIntern<str>];
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError>;
    fn default_timeout_ms(&self) -> u64 { 30_000 }
    fn type_name(&self) -> &'static str { std::any::type_name::<Self>() }
}
```

### Implementation: native command

`CommandUnit` in `dag/src/work_unit.rs`. Note that `WorkContext` does
**not** carry a `library` or `input` field — those belong in
component-specific configuration or in the `data` payload of
`WorkOutput`. `WorkError` does not have `ExecutionFailed` or
`WasmFailed` variants; the only variants are `Execution(String)`,
`Dependency(String)`, and `Timeout { duration_ms, unit }`.

```rust
pub struct CommandUnit {
    name: ArcIntern<str>,
    command: String,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl WorkUnit for CommandUnit {
    fn name(&self) -> &str { &self.name }
    fn depends(&self) -> &[ArcIntern<str>] { &self.depends }
    fn provides(&self) -> &[ArcIntern<str>] { &self.provides }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        // Run in a worker thread — never block the executor directly.
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(&self.command)
            .output()
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(if output.status.success() {
            WorkOutput::ok(stdout)
        } else {
            WorkOutput::fail(stdout)
        })
    }
}
impl FieldAccess for CommandUnit { /* …see derive in §4… */ }
impl Describable for CommandUnit { /* … */ }
impl_component!(CommandUnit);
```

### Implementation: WASM plugin bridge

The real `WasmComponent` (in `coral/src/wasm_runtime.rs`) uses field
construction, not a `new(plugin, schema)` constructor. `WasmComponent`
also implements `Component` via `impl_component!` (not via a blanket
impl — there is no blanket impl).

```rust
pub struct WasmComponent {
    name: ArcIntern<str>,
    plugin: Mutex<extism::Plugin>,
    config: Mutex<HashMap<String, String>>,
    schema: serde_json::Value,            // loaded from plugin at init time
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl WorkUnit for WasmComponent {
    fn name(&self) -> &str { &self.name }
    fn depends(&self) -> &[ArcIntern<str>] { &self.depends }
    fn provides(&self) -> &[ArcIntern<str>] { &self.provides }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        // Inputs flow through the `data` payload of `WorkOutput` or
        // through a side-channel like `ctx.metadata`, not `ctx.input`.
        let mut plugin = self.plugin.lock().unwrap();
        let payload = ctx.metadata.get("input")
            .and_then(|v| v.as_str().map(String::as_bytes))
            .unwrap_or(&[]);
        let result = plugin.call("execute", payload)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        Ok(decode_output(&result)?)
    }
}

impl FieldAccess for WasmComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        let mut plugin = self.plugin.lock().unwrap();
        plugin.call("set_config", format!("{}={}", name, value).as_bytes())
            .map_err(|e| FieldError::Parse(
                format!("wasm set_config failed for '{}': {}", name, e)
            ))?;
        self.config.lock().unwrap().insert(name.to_string(), value.to_string());
        Ok(())
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        self.config.lock().unwrap().get(name).cloned()
            .ok_or_else(|| FieldError::NotFound(name.into()))
    }
    fn field_names(&self) -> &'static [&'static str] { &[] }   // schema is dynamic
}

impl Describable for WasmComponent {
    fn describe(&self) -> serde_json::Value { self.schema.clone() }
}
impl_component!(WasmComponent);
```

### Orchestration — uniform loop

The orchestrator sees a single `Arc<dyn Component>` handle regardless
of origin. The two key traits at the registration site are `impl_component!`
(to satisfy the supertrait) and the prelude import.

```rust
use fluent_wvr::prelude::*;

let mut registry: Vec<Arc<dyn Component>> = Vec::new();
registry.push(Arc::new(Instrumented::new(
    CommandUnit { /* … */ },
    "build",
)));
registry.push(Arc::new(WasmComponent { /* … */ }));

// The orchestrator sees one interface regardless of origin:
for unit in &registry {
    let output = unit.execute(&ctx)?;
    if !output.success { /* … */ }
}
```

### Rules

- **Every orchestratable task implements `WorkUnit`.** No exceptions.
- **Store as `Arc<dyn WorkUnit>` or `Arc<dyn Component>`** in registries.
- **Do NOT add methods to `WorkUnit` speculatively.** Start with `name`, `depends`, `provides`, `execute`. Add `default_timeout_ms`/`type_name` overrides only when the second implementation requires it (both have sensible defaults).
- **For full runtime configurability**, implement all three sub-traits and call `impl_component!` to satisfy the `Component` supertrait.
- **`execute` MUST be synchronous and non-blocking.** See the purity contract in the `WorkUnit` doc comment — violations defeat the `Zone` timeout/retry invariants.

---

## 11. Pattern 9 — Middleware Chain (Post-Erasure Cross-Cutting)

### The problem

When you already have `Arc<dyn WorkUnit>` and need to add logging, retry, or rate limiting, you cannot use newtype wrappers (Pattern 3) because the type is erased. You need post-erasure composition.

### Definition in `dag/src/middleware.rs`

```rust
pub trait Middleware: Send + Sync {
    fn wrap(&self, inner: Arc<dyn Component>) -> Arc<dyn Component>;
}

pub struct TimingMiddleware {
    histogram: Option<Arc<LatencyHistogram>>,
}

impl Middleware for TimingMiddleware {
    fn wrap(&self, inner: Arc<dyn Component>) -> Arc<dyn Component> {
        match &self.histogram {
            // The actual implementation reuses `Instrumented::with_metrics`
            // from fluent-wvr — no parallel `InstrumentedWorkUnit` struct.
            Some(hist) => Arc::new(Instrumented::with_metrics(
                inner, "middleware", Arc::clone(hist),
            )),
            None => Arc::new(Instrumented::new(inner, "middleware")),
        }
    }
}
```

`RetryMiddleware` (in the same file) wraps with `WithRetry::new`
from `fluent-wvr`. The middleware crate does **not** redefine its own
wrapper types — it composes the canonical ones. The actual
`TimingMiddleware::with_histogram(histogram)` constructor takes an
`Arc<LatencyHistogram>`; without one, it produces an `Instrumented`
that logs via `tracing::info!` only.

### Stacking middleware

```rust
let unit: Arc<dyn Component> = Arc::new(CommandUnit { ... });

let chain: MiddlewareChain = MiddlewareChain::new()
    .push(Box::new(TimingMiddleware::new()))
    .push(Box::new(RetryMiddleware::new(3, 50)));

let wrapped = chain.apply(unit);
```

`MiddlewareChain::apply` folds the unit through each layer:
`result = unit; for mw in &middlewares { result = mw.wrap(result); }`.

### Rules

- **Prefer newtype wrappers (Pattern 3) when the type is not yet erased.** Middleware adds a vtable call layer; wrappers can be inlined.
- **Use `Arc<dyn Component>`**, not `Box`, because middleware may be shared across threads.
- **Apply middleware at registration time**, not at execution time.
- **Each middleware layer adds one vtable dispatch.** Minimize layers on hot paths.
- **Reuse `Instrumented` and `WithRetry` from fluent-wvr.** Do not define parallel wrapper types inside `dag` — the whole point of fluent-wvr is the single source of truth.

---

## 12. Pattern 10 — Component Adapter (Runtime Type Adaptation)

### The problem

When adaptation decisions are made at runtime — renaming a component, overriding its execution behavior, or bridging between interfaces — you cannot use compile-time generics.

### Definition in `fluent-wvr/src/wrapper.rs`

`ComponentAdapter` wraps any `Arc<dyn Component>` and lets callers
override `name`, `execute`, and field values at runtime. The
overrides stack — calling `with_field_override` twice with the same
key keeps the most recently set value.

```rust
/// Type alias used by `with_execute_override`. The closure must be
/// `Send + Sync + 'static` because the adapter is shared across threads.
pub type ExecuteFn =
    Arc<dyn Fn(&WorkContext) -> Result<WorkOutput, WorkError> + Send + Sync>;

pub struct ComponentAdapter {
    inner: Arc<dyn Component>,
    name_override: Option<String>,
    execute_override: Option<ExecuteFn>,
    field_overrides: HashMap<String, String>,
}

impl ComponentAdapter {
    pub fn new(inner: Arc<dyn Component>) -> Self { /* all None */ }

    /// Rename the component. `WorkUnit::name()` returns this when set.
    #[must_use]
    pub fn with_name_override(mut self, name: impl Into<String>) -> Self;

    /// Replace `execute` with a custom handler — useful for test
    /// doubles, memoised fast paths, or policy enforcement layers.
    /// Pass an `Arc::new(|ctx| ...)` (not a bare closure).
    #[must_use]
    pub fn with_execute_override(mut self, f: ExecuteFn) -> Self;

    /// Add one field override. Re-setting the same key keeps the new value.
    #[must_use]
    pub fn with_field_override(self, name: impl Into<String>, value: impl Into<String>) -> Self;

    /// Bulk-set from a `HashMap`. Existing keys are replaced.
    #[must_use]
    pub fn with_field_overrides(mut self, overrides: HashMap<String, String>) -> Self;

    /// Drop all field overrides. After this call, `get_field` falls
    /// through to `self.inner`.
    pub fn clear_field_overrides(&mut self);

    /// Borrow the wrapped component.
    pub fn inner(&self) -> &Arc<dyn Component>;
}
```

`ComponentAdapter` must implement all three sub-traits to remain a
`Component`. Note the `set_field` semantics: it propagates the inner
error when the inner has the field but rejects it, and it stores the
override only after the inner accepts the value.

```rust
impl FieldAccess for ComponentAdapter {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.set_field(name, value)?;   // inner validates; on Err, the
                                              // override below is NOT stored
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
        self.inner.field_names()      // never returns the override set
    }
}

impl Describable for ComponentAdapter {
    fn describe(&self) -> serde_json::Value {
        let mut schema = self.inner.describe();
        if let Some(name) = &self.name_override {
            schema["name"] = serde_json::Value::String(name.clone());
        }
        if !self.field_overrides.is_empty() {
            let mut pairs: Vec<(&String, &String)> =
                self.field_overrides.iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            schema["field_overrides"] = serde_json::json!(
                pairs.into_iter()
                     .map(|(k, v)| vec![k.clone(), v.clone()])
                     .collect::<Vec<_>>()
            );
        }
        schema["adapted"] = serde_json::Value::Bool(true);
        schema
    }
}

impl WorkUnit for ComponentAdapter {
    fn name(&self) -> &str {
        self.name_override.as_deref()
            .unwrap_or_else(|| self.inner.name())
    }
    fn depends(&self) -> &[ArcIntern<str>] { self.inner.depends() }
    fn provides(&self) -> &[ArcIntern<str>] { self.inner.provides() }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        match &self.execute_override {
            Some(f) => f(ctx),
            None    => self.inner.execute(ctx),
        }
    }
}
impl_component!(ComponentAdapter);   // <-- required, not a blanket impl
```

### Usage: adapting a WASM plugin

```rust
use std::sync::Arc;
use fluent_wvr::prelude::*;

let wasm_unit: Arc<dyn Component> = Arc::new(WasmComponent { /* … */ });

let adapted = Arc::new(
    ComponentAdapter::new(wasm_unit)
        .with_name_override("custom_name")
        .with_execute_override(Arc::new(|_ctx| {
            Ok(WorkOutput::ok("overridden by adapter"))
        }))
        .with_field_override("port", "8080"),
);

registry.push(adapted);
```

### Rules

- **Use when adaptation decisions are made at runtime.** For compile-time adaptation, use newtype wrappers.
- **Adapters implement `Component`** via `impl_component!(ComponentAdapter)`, so they store in the same registry as any other component.
- **Delegate to the inner component** for all methods you don't override — `depends`, `provides`, `field_names`, and `get_field` (when no override is set) all fall through.
- **`set_field` validates against the inner first, then stores.** If `Arc::get_mut(&mut self.inner)` returns `None`, the override is stored silently — the inner sees the value on the next call only if the inner has interior mutability. If the inner rejects the value, the override is *not* stored.
- **`get_field` falls back to the inner component** when no override exists for the requested field.

---

## 13. Runtime Composition: The Full Lifecycle

This section demonstrates how all twelve patterns compose into a single workflow. Each step uses the same `Arc<dyn Component>` handle.

### Compile-time assembly (Rust struct)

```rust
// 1. Build: fluent builder collects arguments
let mut component = ToolConfig::builder()
    .port(8080)
    .host("localhost".into())
    .verbose(false)
    .build()?;

// 2. Configure: set fields by name before sharing
component.set_field("port", "9000")?;
component.set_field("verbose", "true")?;

// 3. Wrap: add cross-cutting concerns before type erasure (inlineable)
let wrapped = Instrumented::new(
    WithRetry::new(component, /* max_attempts: */ 3, /* backoff_ms: */ 50),
    "my_tool",
);

// 4. Erase to Arc<dyn Component>: uniform handle from this point on
let handle: Arc<dyn Component> = Arc::new(wrapped);
registry.push(handle.clone());

// 5. Execute uniformly
let output = handle.execute(&ctx)?;

// 6. Inspect: read fields through the trait object
let port = handle.get_field("port")?;

// 7. Describe: generate JSON Schema through the trait object
let schema = handle.describe();
```

### Runtime assembly (WASM plugin)

```rust
// 1. Load plugin and extract schema from the plugin itself
let plugin = extism::Plugin::new(wasm_bytes)?;
let schema_bytes = plugin.call("get_schema", &[])?;
let schema: serde_json::Value = serde_json::from_slice(&schema_bytes)?;

// 2. Bridge into Component interface (field construction — no `new` constructor)
let mut wasm_comp = WasmComponent {
    name: ArcIntern::from("my_tool"),
    plugin: Mutex::new(plugin),
    config: Mutex::new(HashMap::new()),
    schema,
    depends: vec![],
    provides: vec![],
};

// 3. Configure by name (delegates into the WASM plugin)
// Note: WasmComponent uses interior mutability, so this works on &mut self
wasm_comp.set_field("timeout_ms", "5000")?;

// 4. Erase to Arc<dyn Component>: same uniform handle as the Rust struct case
let handle: Arc<dyn Component> = Arc::new(wasm_comp);

// 5-7: execute, inspect, describe — identical call sites
let output = handle.execute(&ctx)?;
let schema = handle.describe();
```

### Runtime assembly (database-driven config)

```rust
// 1. Load config from database
let rows = db.query("SELECT key, value FROM tool_config WHERE tool_id = ?", &[tool_id])?;
let config: HashMap<ArcIntern<str>, String> = rows.into_iter()
    .map(|r| (r.get::<_, String>("key").into(), r.get::<_, String>("value")))
    .collect();

// 2. Create dynamic component
let mut dyn_comp = DynamicComponent::new(config);
dyn_comp.set_field("retries", "3")?;

// 3. Erase to Arc<dyn Component>: same uniform handle
let handle: Arc<dyn Component> = Arc::new(dyn_comp);
let output = handle.execute(&ctx)?;
```

### Key guarantee

In all three cases — Rust struct, WASM plugin, database config — the orchestrator sees the same six operations through the same `Arc<dyn Component>` handle: `execute`, `get_field`, `set_field`, `field_names`, `describe`, and `as_any` (for runtime downcasting). No branching on origin.

---

## 14. Pattern 11 — Structured Logging Context

### The problem

In a multi-threaded server, log messages from different requests interleave. Without request context, correlating log lines with a specific request is difficult.

### Canonical implementation

The codebase does **not** implement a standalone `LogContext` / `Scope` / `call_logged`
module. Instead, timing and observability are handled by two composable primitives:

1. **`fluent_wvr::wrapper::Instrumented<U>`** — wraps any `WorkUnit` and logs
   execution duration via `tracing::info!` on every `execute` call. Optionally
   records into a `LatencyHistogram` when constructed with `with_metrics`.

2. **`common_core::metrics::LatencyHistogram`** — a lock-free, atomic
   histogram with bucket-based approximation. Supports `observe(duration_ms)`,
   `observe_duration(start)`, and `estimate_percentile(pct)`.

### Usage

```rust
use std::sync::Arc;
use common_core::LatencyHistogram;
use fluent_wvr::wrapper::Instrumented;

// Basic instrumentation (tracing only):
let unit = Instrumented::new(my_component, "ingest_yago");

// With metrics (tracing + histogram):
let histogram = Arc::new(LatencyHistogram::new());
let unit = Instrumented::with_metrics(my_component, "ingest_yago", histogram.clone());

// After execution, query the histogram:
let p50 = histogram.estimate_percentile(50.0);
let p99 = histogram.estimate_percentile(99.0);
println!("p50={p50}ms p99={p99}ms total={}ms", histogram.sum_ms());
```

**Production consumer:** The `with_metrics` wrapper is wired into
three sites in the workspace:

1. `coral/src/cache_reactor.rs:100,113,125,142` — Coral's `QueueReactor`
   wraps each cache tier (`L2WasmUnit`, `L3GraphUnit`, `L4SemanticUnit`,
   `L5FrontierUnit`) in `Instrumented::with_metrics(adapted, label, hist)`
   before erasing to `Arc<dyn Component>`. The histograms are exposed
   via the `coral_stats` MCP method, returning aggregated p50/p99/count/sum
   per tier.

2. `dag/src/middleware.rs:37,42` — `TimingMiddleware` wraps an
   `Arc<dyn Component>` in `Instrumented::with_metrics` when a histogram
   is provided, falling back to `Instrumented::new` otherwise.

3. `job-copilot/src/server/handler.rs:882` — per-component instrumentation
   for form analysis.

A consumer test that exercises the histogram path end-to-end lives in
`wrapper.rs:619-637` (`instrumented_with_metrics_records_duration`).

### Key properties

1. **Thread-safe:** `LatencyHistogram` uses `AtomicU64` counters — no locks.
2. **Zero overhead when unused:** `histogram: None` skips the observe call.
3. **Composable:** `Instrumented` wraps before type erasure (Pattern 3), preserving inlining.

### Rules

- **Use `Instrumented` for per-unit timing.** It logs via `tracing` and optionally records to a histogram.
- **Use `LatencyHistogram` for aggregate latency stats.** Expose via `--debug` or a telemetry command.
- **Do NOT hand-roll `Instant::now()` + `elapsed()` + `info!`** when `Instrumented` can do it.

---

## 15. Pattern Synergies

The patterns are not independently beneficial — their value multiplies when composed. Each pattern produces something another consumes.

| Pattern | Produces | Consumed by |
|---------|----------|-------------|
| Fluent Builder | Fully-configured owned object | Trait Composition (wrap before erasure) |
| Trait-Based Reflection | `FieldAccess` + `Describable` | `Component` supertrait (via `impl_component!`) |
| Trait Composition | Instrumented concrete type | Trait Object (erase after wrapping) |
| Trait Objects | `Arc<dyn Component>` uniform handle | Middleware Chain, Component Adapter, registry |
| Scoped Ownership | RAII-managed intermediates | Binary IPC payload lifetime |
| Newtype Handles | Distinct integer types | Trait Object registries (prevent ID confusion) |
| Unit of Work | `WorkUnit` impl | `Component` supertrait (via `impl_component!`), registry |
| Middleware Chain | Wrapped `Arc<dyn WorkUnit>` | Registry, orchestrator |
| Component Adapter | Runtime-adapted `Arc<dyn Component>` | Registry, orchestrator |
| Structured Logging Context | Request-scoped observability | All handler entry points |
| Cache Tier WorkUnit | Per-tier `Arc<dyn Component>` (`L4SemanticUnit`, etc.) | `TierRegistry`, `QueueReactor` |

### Key synergies

**Builder + Trait Object** eliminates per-call type branching. Builder accumulation is lock-free; only registry insertion requires synchronization. The orchestrator iterates `Arc<dyn WorkUnit>` and calls `.execute()` without any `match`.

**Trait-Based Reflection + Describable** makes the struct definition the single source of truth. `#[derive(FieldAccess, Describable)]` generates field access, validation, and JSON Schema from one definition. A new field automatically appears in the accessor, the validator, and the schema — one edit, one file.

**Trait Composition + Trait Objects** gives zero-modification observability. Wrap before erasure; the wrapper is inlined by the compiler. The orchestrator gets instrumented execution without any business logic change.

**Scoped Ownership + Binary IPC** eliminates payload lifetime management. `encode_request` returns an owned `Vec<u8>`; the call takes a byte slice; the `Vec` is dropped after the call. No individual allocation to track.

**Component Adapter + Middleware** enables runtime orchestration policies. An adapter can change execution behavior; middleware can add retry or rate limiting. The orchestrator composes both dynamically without recompilation.

---

## 16. Anti-Patterns

### ❌ Manual builder structs

```rust
// Wrong: 50+ lines of boilerplate
pub struct TargetBuilder { name: Option<String>, depends: Option<BitVec>, ... }
impl TargetBuilder {
    pub fn name(mut self, name: String) -> Self { self.name = Some(name); self }
    // ...
}

// Right
#[derive(bon::Builder)]
pub struct Target { ... }
```

### ❌ Raw-pointer vtables for compile-time-known types

```rust
// Wrong: bypasses borrow checker, requires manual Box::from_raw cleanup
pub struct WvrHandle { pub ptr: *mut (), pub vtable: &'static WvrVTable }

// Right: compiler generates the vtable, Arc manages cleanup
let unit: Arc<dyn Component> = Arc::new(MyComponent::new());
```

### ✅ Manual vtable bridges for runtime-assembled interfaces (legitimate use case)

When some methods come from a compiled struct and others from a WASM plugin, bridge through a struct that implements `Component` — not through raw pointers:

```rust
// Right: bridge struct implements the full Component interface
pub struct WasmComponent { plugin: Mutex<extism::Plugin>, schema: serde_json::Value, ... }
impl WorkUnit for WasmComponent { ... }
impl FieldAccess for WasmComponent { ... }
impl Describable for WasmComponent { ... }
impl_component!(WasmComponent);   // <-- required; there is no blanket impl
```

### ❌ Hand-rolling `as_any`/`as_any_mut`

The `Component` supertrait requires the two `Any` methods. The
7-line body is identical for every type:

```rust
// Wrong: boilerplate at every implementation site
impl Component for MyType {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

// Right: one line restores the "blanket-impl feel"
impl_component!(MyType);

// And for wrapper types:
impl_component!(generic (U: Component + 'static) for Instrumented<U>);
```

There is **no blanket `impl<T: FieldAccess + Describable + WorkUnit +
Send + Sync> Component for T`**. The blanket impl is unsound because it
collides with `impl Component for Arc<dyn Component>` (every trait
object would then be a `Component` twice). The `impl_component!` macro
is the canonical mechanism; if a second consumer materializes that
needs a different style, the macro is the extension point.

### ❌ Parallel trait hierarchies

Do not invent a second `as_any`/`as_any_mut` pair (or any second
`Component`-shaped trait) inside a sub-crate. This is what
`guidance-content-node` does today — its `ContentNode` trait
(`node.rs:28-39`) duplicates the `Component` supertrait surface. New
code should either (a) depend on `fluent-wvr` and implement
`Component` directly, or (b) extend `ContentNode` only when the
content-specific methods cannot be expressed on `Component`.

### ❌ `dyn Trait` with only one implementation

```rust
// Wrong: vtable dispatch for a single type
let provider: Arc<dyn EmbeddingProvider> = Arc::new(NoopEmbedding::new(768));

// Right
let provider = NoopEmbedding::new(768);
```

### ❌ Speculative traits

```rust
// Wrong: trait for one implementation with no second planned
pub trait DataStore: Send + Sync { ... }
pub struct SqliteStore;
impl DataStore for SqliteStore { ... }

// Right: use the concrete type; add the trait when the second implementation arrives
pub struct SqliteStore;
impl SqliteStore { ... }
```

### ❌ Cosmopolitan Polymorphism (identical execute bodies)

```rust
// Wrong: three impls routing to the same function
impl WorkUnit for IdentifierQuery {
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        run_staged_query(ctx, &self.config)  // same body
    }
}
impl WorkUnit for CapabilityQuery {
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        run_staged_query(ctx, &self.config)  // same body
    }
}

// Right: use a function pointer for the varying predicate; one execute body
pub struct QueryMatch {
    pub matches: fn(&str, &GuidanceDb) -> bool,
    pub intent: QueryIntent,
}
fn execute_query(ctx: &WorkContext, config: &StagedConfig) -> Result<WorkOutput, WorkError> {
    run_staged_query(ctx, config)  // ONE body
}
```

**Detection:** If three implementations have word-for-word identical `execute` bodies, the trait is routing for the routing mechanism, not for polymorphism. Use a function-pointer array instead.

### ❌ `transmute` for binary IPC

```rust
// Wrong: undefined behavior if layout changes
let result: BinaryExecutionResult = unsafe { std::mem::transmute::<&[u8], _>(buf) };

// Right: explicit byte-order decoding
let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
```

### ❌ `FieldAccess` or `serde` in hot loops

```rust
// Wrong: string parsing on every iteration
for item in &items { item.set_field("count", &item.count.to_string()).unwrap(); }

// Right: direct field access
for item in &items { item.count = compute_count(); }
```

### ❌ Wrapping after type erasure

```rust
// Wrong: cannot inline through dyn Trait; two vtable calls
let unit: Arc<dyn WorkUnit> = Arc::new(MyUnit);
let wrapped = Instrumented { inner: unit };

// Right: wrap before storing in the registry; one vtable call total
let unit = Instrumented { inner: MyUnit };
registry.register(Arc::new(unit));
```

### ❌ Stack-allocating a struct that contains vtables

The most common memory bug in this pattern family. Heap-allocate any struct that stores vtable pointers internally. Use `Arc::new(...)`, never `let x = StructWithVtable { ... }` on the stack when the struct will be used as a trait object.

### ❌ Orchestrator branch-on-tier (cache cascade)

```rust
// Wrong: the orchestrator knows which tier comes next — tight coupling
fn route_with_depth(query: &str, depth: u8) -> Result<Arc<RoutingResult>> {
    if let Ok(result) = router.route(query) { return Ok(result); }       // L3
    if let Ok(result) = router.route_with_embedding(query) { ... }       // L4
    if let Some(subtasks) = decomposer.decompose(query) { ... }          // L4.5
    if let Ok(resp) = client.chat_complete(query) { ... }                // L5
    Err(CacheError::Miss)
}

// Right: each tier is an Arc<dyn Component>; the registry dispatches uniformly
let plan = self.tier_registry.execute(query, depth)?;
// Each tier's execute() receives the prior miss as its trigger via WorkContext
```

**Why it's wrong:** The orchestrator encodes domain knowledge about tier ordering, retry policy, and error propagation — violating the "orchestrator never branches on implementation type" principle (§10). The fix (M3 of `ROADMAP_REFINE.md`) wraps each tier as a `WorkUnit` in a `TierRegistry`.

---

## 17. Pattern Selection Guide

```
Does construction have 4+ parameters?
  YES → #[derive(bon::Builder)]
  NO  → fn new() or struct literal

Does data arrive as a string from outside the process?
  YES → Type known at compile time?
    YES → serde (Tier 2)
    NO  → Schema known at runtime but values are typed?
      YES → Implement FieldAccess manually (Tier 3)
      NO  → HashMap<ArcIntern<str>, serde_json::Value> (Tier 4)
  NO  → Is the access in a hot loop?
    YES → Direct field access (Tier 1)
    NO  → Field name from runtime data?
      YES → FieldAccess trait (Tier 3)
      NO  → Direct field access (Tier 1)

Do you have 2+ concrete implementations today?
  YES → dyn Trait + Arc<dyn Trait>
  NO  → Concrete type (add the trait when the second implementation arrives)

Does a struct cross the WASM host/guest boundary?
  YES → #[repr(C, packed)] + explicit encode/decode
  NO  → Normal struct

Does a batch operation allocate many short-lived intermediates?
  YES → Scoped function with local Vecs (RAII)
  NO  → Normal ownership

Is an integer ID passed across module boundaries?
  YES → Newtype wrapper: struct NodeId(i64)
  NO  → Plain integer

Does the task need uniform orchestration (DAG, MCP, WASM)?
  YES → impl WorkUnit; if runtime-configurable, also FieldAccess + Describable → Component
  NO  → Concrete type with methods

Do multiple implementations share the same execute body, differing only in a predicate?
  YES → Function-pointer array (Vec<QueryMatch>) — not a trait
  NO  → Normal trait with multiple implementations

Do you need cross-cutting concerns?
  Type not yet erased (compile-time known) → Newtype wrapper (Pattern 3)
  Type already erased → Middleware (Pattern 9)

Do you need to adapt component behavior at runtime?
  YES → ComponentAdapter (Pattern 10)
  NO  → Use the component directly
```

---

## 18. Thread Safety

### Thread safety by pattern

| Pattern | Thread safety mechanism |
|---------|------------------------|
| `dyn Trait` + `Arc` | `Send + Sync` bounds enforced at compile time |
| `serde` | Stateless serialization — zero contention |
| `FieldAccess` on Rust structs | Requires `&mut self` — configure before sharing |
| `FieldAccess` on WASM/dynamic | Interior mutability (`Mutex`) inside the impl |
| `FieldAccess` on shared `Arc<dyn Component>` | `Arc::get_mut` succeeds only with exclusive ownership; the blanket impl returns `FieldError::ReadOnly` otherwise. Use `ComponentArcExt::try_as_any_mut` for safe fallback. |
| Newtype wrappers | No state — no contention |
| `bon::Builder` | Per-request, lock-free until `build()` |
| Scoped ownership | Stack-local — no sharing |
| `WorkUnit` | `Send + Sync` required; implementors must be thread-safe |
| `Middleware` | `Send + Sync` required; stateless middleware is zero-contention |
| `ComponentAdapter` | `Send + Sync` required; closures must be `Send + Sync` |
| `Scope` (`fluent-concurrency`) | Holds a `JoinSet` internally; not `Sync`. Use `Scope::defer()` to guarantee cleanup on drop. Dropping without `close().await` panics (a structured-concurrency violation). |
| `Zone` (`fluent-concurrency`) | Holds a `JoinSet` and a per-zone `Runtime` reference. `Zone::register` returns `Result<&mut Self, ZoneError>` — duplicate names are rejected with `Err(ZoneError::DuplicateName(_))`. |
| `Reserve` (`fluent-concurrency`) | RAII permit from a shared `AtomicUsize` counter. `Send + Sync` because the counter is `Arc`-shared. Drop without `commit()` returns the permit; `commit()` consumes it permanently. |
| `LogContext` | Thread-local — no synchronization needed |

### Detailed rules

1. **Create trait objects on a single thread during initialization.** Do not create `Arc<dyn Trait>` handles concurrently.
2. **Shared registries require synchronization.** Use `Mutex`, `RwLock`, or `DashMap` for registries accessed from multiple threads.

   > **`DashMap` vs `RwLock<HashMap>`:** `DashMap` is a shard-lock map optimized for write concurrency. For read-dominated workloads (registries are typically read-heavy after initialization), `RwLock<HashMap>` is often faster because it allows fully parallel reads with no per-shard overhead. Profile before choosing `DashMap` for a hot registry.

3. **Destroy after all concurrent calls complete.** Before dropping an `Arc<dyn Trait>`, ensure all threads that hold a reference have finished.

### Shared mutable objects requiring explicit protection

| Object | Mechanism | Reason |
|--------|-----------|--------|
| `Library` | `Mutex<rusqlite::Connection>` | SQLite writes must serialize |
| `CapabilityRegistry` | `RwLock<HashMap<...>>` | Read-heavy concurrent intern calls |
| `L1Cache` | `DashMap<String, RoutingResult>` | Write-concurrent cache access |
| `WasmComponent.plugin` | `Mutex<extism::Plugin>` | Plugin state is not thread-safe |
| `LogContext` | Thread-local | Each thread has its own context |

### Typical usage pattern

```rust
// Init thread: create and share
let provider: Arc<dyn EmbeddingProvider> = Arc::new(OllamaProvider::new("model"));

// Worker threads: read-only through Arc
let p = provider.clone();
let handle = std::thread::spawn(move || { p.embed("hello").unwrap(); });

// After all workers join, drop is safe
handle.join().unwrap();
drop(provider);
```

---

## 19. Schema Evolution

### Versioning with `serde`

```rust
#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_field: Option<String>,  // added in v1.1
}
```

### Field-level version annotations

```rust
#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[field(version_added = "1.1")]
    pub new_field: Option<String>,

    #[serde(skip)]
    #[field(version_removed = "2.0")]
    pub old_field: Option<String>,

    #[serde(alias = "old_name")]
    #[field(version_added = "1.2")]
    pub renamed_field: String,
}
```

### Migration functions

```rust
fn migrate_timeout<'de, D>(deserializer: D) -> Result<u64, D::Error>
where D: Deserializer<'de> {
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(s) => parse_duration(&s).map_err(serde::de::Error::custom),
        serde_json::Value::Number(n) => n.as_u64().ok_or_else(|| serde::de::Error::custom("invalid")),
        _ => Err(serde::de::Error::custom("invalid timeout format")),
    }
}
```

### Binary IPC versioning

```rust
pub const BINARY_SCHEMA_VERSION: u32 = 1;

let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
if version != BINARY_SCHEMA_VERSION {
    return Err(IpcError::UnsupportedVersion);
}
```

### Upgrade matrix

| Change | Semver | Action |
|--------|--------|--------|
| Add field with default | Minor | `#[serde(default)]` — backward compatible |
| Remove field | Major | `#[serde(skip)]` — forward compatible |
| Rename field | Major | `#[serde(alias = "old_name")]` on new field |
| Type change | Major | Old field + `deserialize_with` migration function |
| Binary IPC breaking change | Major | Bump `BINARY_SCHEMA_VERSION` |

---

## 20. Quick Reference for AI Agents

When writing new code in `rust-src/`:

1. **Check the source first.** Run `guidance explain "<topic>"` before writing. Check `dag/src/target.rs` for builders, `coral/src/tier_units.rs` for trait objects, `wasm_ipc/src/lib.rs` for binary IPC, `fluent-wvr/src/wrapper.rs` for adapters and wrappers.

2. **New multi-parameter construction** → `#[derive(bon::Builder)]`. Use `#[builder(default)]` for optional fields. Never write manual builders.

3. **New boundary serialization** → `#[derive(Serialize, Deserialize)]` with `#[serde(default)]` and `#[serde(skip_serializing_if = "Option::is_none")]`.

4. **New runtime-configurable component** → Implement `FieldAccess` (instance method `field_names(&self)`), `Describable` (instance method `describe(&self)`), and `WorkUnit`. Then add `impl_component!(MyType);` after the three impls. There is **no blanket impl** — `impl_component!` is the canonical way to satisfy the supertrait. Both `field_names` and `describe` **must be instance methods** for trait-object dispatch to work.

5. **New cross-cutting logic** → Newtype wrapper **before type erasure** (Pattern 3). If the type is already erased, use Middleware (Pattern 9). Never wrap after type erasure. The canonical wrappers are `Instrumented::new(inner, label)` (timing/observability) and `WithRetry::new(inner, max_attempts, backoff_ms)` (sync retry with jittered backoff).

6. **New subsystem with multiple implementations** → Define a trait with `Send + Sync`. Store as `Arc<dyn Trait>`. Never use `dyn Trait` with only one implementation.

7. **New WASM/binary IPC type** → `#[repr(C, packed)]` + explicit `to_le_bytes()` / `from_le_bytes()`. Validate magic + version first. Never `transmute`.

8. **New batch-processing loop** → Scoped function with local `Vec`s. RAII drops everything at scope exit. For long-lived async groups, use `Scope::defer()` instead of manually calling `close().await`; dropping a `Scope` without closing panics.

9. **New orchestratable task** → Implement `WorkUnit`. For runtime configurability, add `FieldAccess` + `Describable` and call `impl_component!` to become a `Component`. Store as `Arc<dyn Component>`. Use `Zone::register` (returns `Result<&mut Self, ZoneError>`) to participate in a `Zone` — the supervisor handles timeout, retry, and panic/fail distinct tracking.

10. **New request-scoped observability** → Wrap the unit in `Instrumented::with_metrics(inner, label, histogram)`. The histogram is `Arc<common_core::metrics::LatencyHistogram>`; p50/p99/count/sum are queryable after `execute`.

11. **New runtime adaptation** → Use `ComponentAdapter::new(arc).with_name_override("...").with_field_override("k", "v")` to layer runtime decisions on top of an existing `Component` without modifying it.

12. **New test component** → Use `fluent_wvr_testutil::StubComponent` for `ok`/`fail`/`panic`/`with_dep`/`with_provides`/`with_handler` builders. Avoid hand-rolled `WorkUnit + FieldAccess + Describable + impl_component!` quads in test code.

### Never do these

11. **Never use raw pointers for vtables.** Use `dyn Trait` + `Arc`.

12. **Never use `transmute` for binary IPC.** Use explicit byte-order encoding.

13. **Never use `dyn Trait` with only one implementation.**

14. **Never use `serde` or `FieldAccess` in hot loops.** Use direct field access.

15. **Never stack-allocate a struct that stores vtables.** Heap-allocate with `Arc::new(...)`.

16. **Never wrap after type erasure.** Wrap before storing in the registry.

17. **Never use the string reflection path in hot loops.** It allocates.

18. **Never apply Fluent Builder to a 3-parameter init.** Three params are readable.

19. **Never add `FieldAccess` to a type that never crosses a boundary.**

20. **Never create a trait speculatively.** Start with a concrete type.

21. **Never define `field_names` or `describe` as static associated functions.** They must be `&self` instance methods for trait-object dispatch.

22. **Never hand-write `as_any`/`as_any_mut`.** Use `impl_component!`.

23. **Never drop a `Scope` without closing.** Use `Scope::defer()` or `scope.close().await` — drop without close panics.

24. **Never treat `WorkError::Execution` as a panic.** Handler errors land in `ZoneSummary::failed`; only `JoinError::is_panic()` lands in `panicked`.

### The throwaway rule

25. **Never add a new public API surface (function, type, error variant, or builder helper) without at least one in-tree production consumer.** The codebase has a documented rule against throwaway additions — three APIs shipped in a prior roadmap cycle (`Reserve`, `FieldError::ReadOnly`, `WorkContext::for_unit_in_zone`) had no production consumer on landing. The cost shows up in unused code paths, surprise behaviour, and dead documentation. If a feature must ship ahead of its consumer, file a tracking issue and link it from the API's doc comment.

### Verification

28. **Run `cargo clippy --workspace -- -D warnings` before finishing.** The project enforces `#![deny(warnings)]`. (Pre-existing `uninlined_format_args` warnings at `fluent-wvr/src/work.rs:278,284` are grandfathered — do not introduce more.)

29. **Run `cargo test --workspace` before finishing.** All 1213 tests must pass.

30. **Check for `unsafe` blocks.** The workspace has 3 `unsafe` blocks, all in `wasm_ipc/src/lib.rs` for `read_unaligned` of packed struct fields. Every new `unsafe` block must be justified and documented; the `forbid(unsafe_code)` lint is set at the crate level for both `fluent-wvr` and `fluent-concurrency`.

31. **Run `cargo doc --workspace --no-deps` before finishing.** Doc warnings are part of the quality bar.

---

## 21. Companion crate: `fluent-concurrency`

The trait framework in `fluent-wvr` is paired with a structured-concurrency
runtime in `fluent-concurrency`. The full surface is documented in
`doc/skills/fluent-concurrency/SKILL.md`. The high-level primitives are:

| Primitive | File | Purpose |
|-----------|------|---------|
| `Runtime` (trait) | `fluent-concurrency/src/runtime/` | Pluggable backend: `TokioRuntime` (production), `TestRuntime` (deterministic, paused-time), `NoopRuntime` (in `fluent-wvr/src/runtime.rs`, init-only) |
| `Capability` + `CapabilitySet` | `fluent-wvr/src/capability.rs` | Type-map of capability tokens; `with`/`get`/`remove`/`remove_as`/`contains`/`iter`/`len`/`is_empty` |
| `Scope` | `fluent-concurrency/src/scope.rs` | Structured concurrency; `spawn` + `close().await` or `defer()`; drop-without-close panics |
| `Zone` | `fluent-concurrency/src/zone.rs` | `Scope` + dependency graph + panic/fail/cancel distinct tracking; `ZoneSummary { completed, panicked, failed, cancelled }`; `Zone::register` returns `Result<_, ZoneError>`; `ZoneError::DuplicateName` |
| `WorkerPool<T>` | `fluent-concurrency/src/pool.rs` | Bounded FIFO worker pool; capacity + backpressure |
| `ResultPool<T, R, E>` | `fluent-concurrency/src/pool.rs` | Worker pool that returns a typed `Result<R, ResultPoolError<E>>`; `ResultPoolError` is `Inner(E) \| Canceled \| Pool(PoolError)` |
| `PriorityResultPool<T, R, E>` | `fluent-concurrency/src/pool.rs` | Priority-ordered variant; workers drain fully before blocking on `Notify`; `submit` uses `notify_waiters` to avoid wakeup collapse |
| `Limiter` | `fluent-concurrency/src/pool.rs` | Concurrency cap (N-at-a-time), no queue |
| `Queue<T>` | `fluent-concurrency/src/queue.rs` | Bounded async FIFO; `PoolError::Full` on overflow |
| `PriorityQueue<T>` | `fluent-concurrency/src/queue.rs` | O(log P) for distinct priorities, O(1) for all-zero fast path |
| `PartitionedRouter<K, J>` | `fluent-concurrency/src/router.rs` | Hash-based sharding; preserves causal ordering per key |
| `CreditFlow` | `fluent-concurrency/src/flow.rs` | Sender/receiver backpressure with `CreditSpec { initial, more_after }` |
| `Reserve` | `fluent-concurrency/src/reserve.rs` | RAII permit from a shared `AtomicUsize`; `try_acquire`/`commit`; **currently has no in-tree production consumer** (throwaway rule candidate) |

A `Zone` is a `Scope` plus the `WorkUnit` integration: tasks implement
`Component` and are registered with `zone.register(arc).unwrap()`.
The `Zone` runs each unit through `execute_with_timeout_and_retry`
using the per-unit `default_timeout_ms()` and the zone-wide
`max_retries`/`timeout_ms` set on the `WorkContext`. The four-way
summary discriminates: `Ok(_)` → `completed`, `Err(WorkError::*)` →
`failed`, `JoinError::is_panic()` → `panicked`, `JoinError::is_cancelled()`
or `Timeout` → `cancelled` (with `CancelReason::{Timeout, DependencyFailed, Aborted}`).

---

## 22. Future Work — MCP Handler as WorkUnit

An MCP method handler (e.g. `coral_query`, `guidance_explain`) is a natural `WorkUnit`: the method name maps to `name()`, and the dispatch body maps to `execute()`. Converting MCP handlers to `WorkUnit` implementations would allow them to participate in the DAG orchestration pipeline, middleware chains, and `Instrumented`/`WithRetry` wrappers.

**Do not implement this now.** There is currently only one `JsonRpcHandler` per MCP server, and the `WorkUnit` interface would require adapting the JSON-RPC request/response model into `WorkContext`/`WorkOutput`. This is a follow-up task to be undertaken only when a second consumer of MCP method dispatch appears or when MCP methods need DAG orchestration.

---

*This document is the authoritative reference for Fluent WVR patterns in the Rust codebase. The deprecated `FLUENT_WVR_RUST.md` (raw-pointer variant) must not be followed.*
