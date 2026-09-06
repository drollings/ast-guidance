# MockRouter — Audit: Redundancies, DRY, and SOLID Violations

## MockRouter Overview

`MockRouter` (`src/router/src/testing/mock.rs:111`) is a test-only harness that builds a
`PipelineOrchestrator` with a `StubComponent` (from `fluent_wvr_testutil`) substituted for the
real `ClassifierStage`. It returns fixture data from a `HashMap<String, String>` keyed on the
user message. No LLM, no network.

It is **not** the live router. The live routing is split across:
- `ClassifierStage` (decides *where to send* via an LLM call)
- `dispatch_to_frontier()` in `server.rs` (actually sends the HTTP request)

---

## Redundancies

### 1. `RouterOnlyMock` is dead code — identical to `MockRouter`

**Location**: `src/router/src/testing/mock.rs:151`

`RouterOnlyMock` is a structural clone of `MockRouter`:

| Aspect | MockRouter | RouterOnlyMock |
|--------|-----------|---------------|
| Fields | `pipeline: PipelineOrchestrator`, `fixtures: Arc<MockFixtures>` | Same |
| `new()` | Same body | Same body |
| `from_fixture_file()` | Same body | Same body |
| `route()` | Same body | Same body |
| Underlying pipeline | `build_mock_pipeline()` | Same function |

`RouterOnlyMock` is re-exported at `testing/mod.rs:6` but **never referenced** anywhere in the
codebase — not in tests, not in other modules, not in downstream crates.

### 2. `MockRouter` and `RouterServer` share no code despite identical pipeline-execution logic

`MockRouter::route()` (mock.rs:128) serializes a request, stuffs it into `WorkContext.metadata`,
calls `pipeline.execute()`, and deserializes the result. `RouterServer::handle_connection()`
(server.rs:347–413) does the same thing with the same pattern inlined. There is no shared
`execute_pipeline(request, pipeline) -> PipelineResult` function that both could call.

### 3. `build_mock_pipeline()` hardcodes stage wiring that duplicates `Config::build_named_pipeline()`

`mock.rs:83-109` manually assembles a 3-stage pipeline with DeterministicPreFilter → StubComponent
→ RouterStage. `config.rs:288-348` (`build_named_pipeline()`) does the same wiring for the live
path, differing only in the classifier type (StubComponent vs ClassifierStage). The boilerplate
of pushing stages into a `Vec<Arc<dyn Component>>` is identical.

---

## DRY Violations

### 4. Duplicate `get_metadata_string` — two identical definitions

| File | Visibility |
|------|-----------|
| `src/router/src/stages/common.rs:29` | `pub fn` |
| `src/router/src/pipeline.rs:260` | `fn` (private) |

Both have the same body:

```rust
ctx.metadata.get(key).and_then(|v| match v {
    MetadataValue::String(s) => Some(s.clone()),
    _ => None,
})
```

The duplicate in `pipeline.rs` should be removed in favour of `stages::common::get_metadata_string`.

### 5. `server.rs` response formatting duplicated 5× over 150 lines

Lines 456–606 contain five nearly identical blocks that each:

1. Build a `RouterResponse`
2. Optionally create a `StreamingHandler`
3. Call `handler.format_choice_chunk(choice)` then `handler.format_done()`
4. Serialize with `normalize::normalize_response()`
5. Format the HTTP response line with content-type, content-length, CORS headers, connection: close

The only differences are the source of the response content (rejection error, classifier response,
routing target, classifier_url fallback, no-target fallback). This should be a single helper
function parameterized on the content source.

### 6. `dispatch_to_frontier()` in `server.rs` duplicates `dispatch/frontier.rs`

`dispatch/frontier.rs` defines a proper `FrontierBackend` trait with `OpenAiBackend`,
`AnthropicBackend`, `OpenAiCompatibleBackend`, and a `FrontierDispatcher` struct with
concurrency `Limiter`. The server completely ignores this module and defines its own standalone
`dispatch_to_frontier()` function (server.rs:647) that creates a `reqwest::Client` inline,
builds the request body manually, and retries with exponential backoff.

Two parallel implementations of the same concept, each with its own retry logic, URL
construction, and response parsing.

### 7. Duplicate `FieldAccess`/`Describable` boilerplate across every stage

Every `Component` implementation in the router has this identical boilerplate:

```rust
impl FieldAccess for XxxStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound("XxxStage has no configurable fields".into()))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("XxxStage has no configurable fields".into()))
    }
    fn field_names(&self) -> &'static [&'static str] { &[] }
}

impl Describable for XxxStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }
}
```

Appears in these files (5 locations, ~28 lines each = ~140 lines of copy-paste):

| File | Type |
|------|------|
| `stages/deterministic.rs:326` | `DeterministicPreFilter` |
| `stages/classifier.rs:214` | `ClassifierStage` |
| `stages/router.rs:131` | `RouterStage` |
| `pipeline.rs:267` | `PipelineOrchestrator` |
| `server.rs:104` | `RouterServer` |

A macro (`impl_noop_fields!(TypeName)`) or a blanket impl for types that opt out could eliminate
this entirely.

### 8. `PipelineOrchestratorBuilder` is a manual builder

`pipeline.rs:86-102` defines a hand-rolled builder struct when `#[derive(bon::Builder)]` would
generate the same code for free, as mandated by the Fluent WVR skill:

> "Always derive `bon::Builder` for structs with 4+ fields."

The builder wraps only a `Vec<Arc<dyn Component>>` — a single field — and its `.push()` method
is just `self.stages.push(stage)`.

---

## SOLID Violations

### 9. ❌ SRP — `RouterServer::handle_connection()` does everything

600 lines (server.rs:222–609) handling:
- Raw TCP byte reading and HTTP/1.1 header parsing
- CORS preflight
- Route matching
- Body length checking and read
- JSON deserialization
- Pipeline execution
- 5-way response branching (rejection / classifier / routing_target / fallback / noop)
- Streaming chunk formatting (repeated 5×)
- Frontier HTTP dispatch (with its own URL construction)
- Stats counters

### 10. ❌ OCP — Response branching is hardcoded

Adding a new response type (e.g., local-agent dispatch) means adding a new `else if` branch to
the 5-way chain in `handle_connection()`. The branching should be a
`match pipeline_result.action { ... }` on a sealed enum.

### 11. ❌ ISP — Every type forced to implement `FieldAccess` + `Describable`

All five stage types are routing infrastructure that never needs runtime field access. They
implement `FieldAccess` and `Describable` only because `Component` requires them. The supertrait
hierarchy forces every `WorkUnit` to also be `FieldAccess + Describable`, even when those
capabilities are never exercised.

### 12. ❌ DIP — `ClassifierStage` hard-depends on `LlmClient`

```rust
pub struct ClassifierStage {
    client: LlmClient,   // concrete type, not a trait
    ...
}
```

`LlmClient` is instantiated in the constructor rather than injected as `Arc<dyn ChatProvider>`.
This is why `MockRouter` exists — there is no other way to test the pipeline without a live LLM.
The `StubComponent` in the mock pipeline is a parallel implementation of the same interface.

---

## Summary

| # | Issue | Category | Severity |
|---|-------|----------|----------|
| 1 | `RouterOnlyMock` — dead code, identical to `MockRouter` | Redundancy | Low |
| 2 | Pipeline execution logic duplicated between `MockRouter` and `RouterServer` | DRY | Medium |
| 3 | Stage wiring duplicated in `build_mock_pipeline` and `Config::build_named_pipeline` | DRY | Medium |
| 4 | `get_metadata_string` defined twice | DRY | Low |
| 5 | Response formatting duplicated 5× in 150 lines | DRY | High |
| 6 | Inline `dispatch_to_frontier` bypasses `dispatch/frontier.rs` | DRY | High |
| 7 | `FieldAccess`/`Describable` boilerplate duplicated 5× | DRY | Medium |
| 8 | Manual `PipelineOrchestratorBuilder` instead of `bon::Builder` | DRY | Low |
| 9 | `handle_connection()` — 600-line monolith | SRP | High |
| 10 | Hardcoded response branching | OCP | Medium |
| 11 | Forced `FieldAccess`/`Describable` on non-configurable types | ISP | Medium |
| 12 | `ClassifierStage` depends on concrete `LlmClient` | DIP | High |
