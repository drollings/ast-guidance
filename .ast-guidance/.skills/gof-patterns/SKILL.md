---
name: gof-patterns
description: Documentation of 12 Gang of Four (GoF) design patterns with AST detection heuristics and cross-language mapping for transpilation.
---

# Gang of Four (GoF) Design Patterns

**Purpose**: Reference for AST-based pattern detection and cross-language transpilation mapping.

For each pattern: detection heuristic → idiomatic target mapping. Language-specific callouts appear only where idiomatic divergence is sharp enough to cause errors.

---

## 1. Adapter

Wraps an incompatible object to expose a different interface.

**Detect**: A class holding an instance of another type; wrapper methods delegate to that instance, transforming arguments or return values.

**Idioms**:
- Pine Script → wrapper function
- Python → class with delegating methods
- Go → struct field of interface/struct type; methods delegate
- Rust → struct wrapping a type; `impl TargetTrait for Wrapper`

> **Go/Rust**: map to composition, not inheritance. The adapter implements the *target* interface/trait; the adaptee is a field.

---

## 2. Decorator

Adds behavior to an object at runtime without changing its interface.

**Detect**: A type that implements the same interface as what it wraps *and* holds a reference to an instance of that same interface.

**Idioms**:
- Pine Script → function wrapping another function call
- Python → `@decorator` syntax or wrapper class with identical signature
- Go → struct implementing an interface and holding the same interface
- Rust → struct implementing a trait and holding `Box<dyn Trait>`

> **Python**: `@decorator` syntax is a language shorthand — it compiles to a wrapper function/class, not a class hierarchy. When transpiling *to* Python, prefer `@decorator` for function-level wrapping; use wrapper classes when state is needed.

---

## 3. Factory Method

A function or method that returns an object whose concrete type is determined at runtime.

**Detect**: A static/class method returning a base type or interface; conditional logic or a registry selects the concrete type.

**Idioms**:
- Pine Script → function returning a typed map or value based on input
- Python → `@classmethod` or module-level function returning a subclass instance
- Go → `NewXxx(...)` function returning an interface type
- Rust → `impl X { fn new() -> Self }` or `fn new() -> Box<dyn Trait>`

---

## 4. Strategy

A family of interchangeable algorithms selected at runtime.

**Detect**: A context holding a reference to a strategy interface/trait; execution is delegated to it.

**Idioms**:
- Pine Script → flag parameter or function passed into a function
- Python → callable or object with a specific method (first-class functions suffice)
- Go → struct field of interface type; method calls the interface
- Rust → generic `T: Trait` or `Box<dyn Trait>` field

> **Python → Go/Rust**: Python callables (lambdas, plain functions) must be mapped to an interface or trait — there is no direct equivalent of a bare callable in static-typed targets.

---

## 5. Observer

Subject notifies a list of observers on state change.

**Detect**: Subject maintains a collection of observers; has `attach`/`detach` and `notify` iterating over them.

**Idioms**:
- Pine Script → manual global state checks; no native mechanism
- Python → list of callables or objects; `for obs in self._observers: obs.update(...)`
- Go → slice of interfaces, or channels for async notification
- Rust → `Vec<Box<dyn ObserverTrait>>`; loop calling trait method

> **Memory**: Observers must be explicitly detached when no longer needed. In Rust, prefer `Weak<RefCell<T>>` references in the observer list to avoid reference cycles.

---

## 6. Composite

Part-whole tree where leaves and composites share one interface.

**Detect**: A base interface used by both leaf nodes and a composite that holds a collection of that same base type.

**Idioms**:
- Pine Script → recursive function calls or nested structures
- Python → class with `self.children: list[ComponentBase]`
- Go → interface with a slice of the same interface
- Rust → recursive enum or `Vec<Box<dyn Component>>`

> **Rust**: recursive types require `Box` (or `Rc`/`Arc`) — a bare `Vec<Self>` is a compile error.

---

## 7. Singleton

One instance, globally accessible.

**Detect**: Private/absent constructor; static variable holding the instance; static accessor.

**Idioms**:
- Pine Script → top-level global variable
- Python → module-level instance (the module import system guarantees single initialization)
- Go → package-level variable initialized with `sync.Once`
- Rust → `OnceLock` / `LazyLock` (std, stable 1.80+) or `lazy_static`

> **Python**: a module-level instance is the idiomatic singleton — avoid `__new__` overrides. **Go/Rust**: use `sync.Once` / `OnceLock` for thread-safe initialization; a bare global without synchronization is a data race.

---

## 8. Proxy

Surrogate controlling access to a real subject; same interface.

**Detect**: Class implementing the same interface as the real object; holds a reference to it; adds logic (lazy init, access control, logging) around delegation.

**Idioms**:
- Pine Script → wrapper function with precondition checks
- Python → class with `__getattr__` delegation or explicit method forwarding
- Go → struct implementing the same interface as the real object
- Rust → struct implementing the same trait as the real object

---

## 9. Command

Encapsulates a request as a first-class object.

**Detect**: Interface/trait with a single `execute()` method; concrete implementations hold receiver and parameters.

**Idioms**:
- Pine Script → function reference stored and called later
- Python → callable object (`__call__`) or object with `execute()`
- Go → `struct` with `Execute()`, or a bare `func()` type
- Rust → closure (`impl Fn()`) or struct implementing a `Command` trait

---

## 10. Template Method

Fixed algorithm skeleton in a base; variable steps deferred to subclasses/overrides.

**Detect**: Base class with a "template" method calling `self.step1()`, `self.step2()` etc.; subclasses override steps, not the template.

**Idioms**:
- Pine Script → manual function composition
- Python → abstract base class; template calls `self.hook()` methods
- Go → struct embedding + interface; base method calls interface methods
- Rust → trait with default method implementations calling other (required) trait methods

> **Go/Rust**: there is no classical inheritance. Map inheritance-based templates to **composition + interface/trait** — the "base" logic lives in a function that accepts the interface, not in a parent class.

---

## 11. Builder

Separates complex object construction from representation; fluent interface.

**Detect**: Class with setter methods returning `self`; terminal `build()` returning the final object.

**Idioms**:
- Pine Script → multi-argument function or configuration map
- Python → class with method chaining; `build()` returns result
- Go → **functional options** (`NewXxx(WithA(), WithB())`) — this is the idiomatic Go builder, not fluent method chaining
- Rust → consuming builder (`fn set_x(mut self, x: T) -> Self`); terminal `build() -> Product`

> **Go**: fluent method chaining (returning `*Builder`) exists but the idiomatic Go style is functional options. When transpiling *to* Go, prefer `func(c *Config)` option functions over a chained builder.

---

## 12. Visitor

Separates an operation from the object structure it operates on; double dispatch.

**Detect**: Element classes have `accept(visitor)` methods; visitor has `visit(element)` overloads.

**Idioms**:
- Pine Script → manual type checking and dispatch
- Python → `functools.singledispatch` or `isinstance` dispatch
- Go → `Accept(v Visitor)` method on each element; visitor interface per element type
- Rust → `match` over an enum of variants (preferred over double-dispatch OOP)

> **Rust**: the idiomatic Visitor is an `enum` + `match`, not OOP double dispatch. When transpiling to Rust, collapse the element hierarchy into a single enum and match on variants in the visitor function.

---

## Transpilation Checklist

1. **Detect** — identify the pattern via AST heuristics above
2. **Idioms** — select the idiomatic target-language form from the table
3. **Constraint-adjust** — no inheritance in Go/Rust (use composition + interfaces/traits); ownership in Rust requires `Box`/`Rc`/`Arc` for recursive/shared structures
4. **Contract-verify** — transpiled code must preserve the original logical interface
