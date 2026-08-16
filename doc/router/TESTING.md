# Coral Router — Testing Guide

## Layers

| Layer | Location | What it tests |
|-------|----------|---------------|
| Unit tests | `src/router/src/stage_tests.rs` | Individual pipeline stages in isolation |
| Golden tests | `src/router/src/tests/golden.rs` | Labeled corpus — intent, PII, adversarial cases |
| E2E tests | `src/router/src/tests/e2e_tests.rs` | Full pipeline with `TranscriptProvider` (no LLM) |
| Server tests | `src/router/src/server_tests.rs` | `RouterServer` construction and config |
| Mock mode | `coral-router start --mock` | Real HTTP server, real pipeline, transcript-driven |

## Quick commands

```sh
# All 176 router tests (unit + golden + e2e + server)
make router-test

# Workspace-wide (1,400+ tests)
cargo test --workspace
```

## Unit tests

Pipeline stages are tested independently by constructing `WorkContext` with
a serialized `RouterRequest` in metadata:

```rs
fn make_ctx(user_text: &str) -> WorkContext {
    let mut ctx = WorkContext::default();
    ctx.metadata.insert("request".into(), MetadataValue::String(
        serde_json::json!({"model":"test","messages":[{"role":"user","content":user_text}]}).to_string()
    ));
    ctx
}

#[test]
fn test_deterministic_command_help() {
    let filter = DeterministicPreFilter::new();
    let output = filter.execute(&make_ctx("/help")).expect("execute");
    let decision: StageDecision = output.data_as().expect("data_as");
    assert_eq!(decision.verdict, StageVerdict::Rejected);
}
```

Stage 1 (DeterministicPreFilter) and Stage 3 (RouterStage) need no LLM.
Stage 2 (ClassifierStage) is tested via the E2E and golden tests below.

## Golden tests

A checked-in labeled corpus in `tests/golden.rs` covers:

| Category | Cases |
|----------|-------|
| Intent | question, command, creative, code, chitchat |
| PII | SSN, email, card number, phone, multiple |
| Adversarial | prose-resembling-command, empty, special chars |

Every case asserts the expected reject stage (or pass) and expected PII
classes. The tests build a real `PipelineOrchestrator` with a
`TranscriptProvider` that returns a default `ClassifierOutput`
(`action=route, target=fast`).

## E2E tests

Full 3-stage pipeline tests — deterministic pre-filter → classifier → router.
The `ClassifierStage` gets a `TranscriptProvider` implementing `ChatBackend`,
which returns canned `ClassifierOutput` JSON keyed by user message:

```rs
fn test_e2e_garbage_rejected_by_classifier() {
    let mut entries = HashMap::new();
    entries.insert("asdfghjkl qwerty zxcvbnm".into(),
        classify_output("reject", 0.15, 0.9, "incoherent input"));
    let pipeline = make_pipeline(TranscriptProvider::new(entries));
    let result = route(&pipeline, &make_request("asdfghjkl qwerty zxcvbnm"));
    assert!(result.unwrap().rejected);
}
```

## Mock mode (end-to-end HTTP)

The `--mock` flag starts the real HTTP server with a transcript file.
All pipeline stages run normally; `ClassifierStage` uses a `TranscriptProvider`
instead of a real LLM; dispatch calls are intercepted with routing validation.

### Transcript format

A JSON array of entries. Each entry maps a user message to the expected
classifier output, the expected route, and a canned dispatch response:

```json
[
    {
        "user_message": "What is 2+2?",
        "classifier_response": "{\"action\":\"route\",\"target\":\"fast\",\"coherence_score\":0.95,\"safety_score\":0.9,\"intent\":\"question\",\"reason\":\"simple factual query\"}",
        "expected_route": "fast",
        "dispatch_response": "2 + 2 = 4",
        "rejected": false
    },
    {
        "user_message": "My SSN is 123-45-6789",
        "classifier_response": "{\"action\":\"reject\",\"coherence_score\":1.0,\"safety_score\":1.0,\"intent\":null,\"reason\":\"PII detected\"}",
        "expected_route": null,
        "dispatch_response": null,
        "rejected": true,
        "reject_reason_contains": "blocked"
    }
]
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `user_message` | yes | Exact user message text — lookup key |
| `classifier_response` | yes | JSON string matching `ClassifierOutput` schema |
| `expected_route` | no | Route name the request must be dispatched to (null if rejected) |
| `expect_model_group` | no | `model_groups` name the target must have dispatched through (intent→model_group check) |
| `dispatch_response` | no | Canned response from the target model |
| `rejected` | no | `true` if the pipeline should reject this request |
| `reject_reason_contains` | no | Substring expected in the rejection reason |

### Running mock mode

```sh
# Build + start with transcript
cargo run -p coral-router -- start --config env/coral-router.json --mock env/mock-transcripts.json

# Or via config:
# Add "mock": { "transcript_path": "env/mock-transcripts.json" } to env/coral-router.json
cargo run -p coral-router -- start --config env/coral-router.json
```

### Sending test requests

```sh
curl -s -X POST http://127.0.0.1:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 2+2?"}]}'
# → {"choices":[{"message":{"content":"2 + 2 = 4",...}}],...}
```

### Validation

Mock mode validates after each request:
- **Route match**: the pipeline's `routing_target` must match `expected_route`
- **Model-group match**: the resolved `routing_target.group` must equal `expect_model_group`
- **Rejection match**: if `rejected: true`, the pipeline must reject with a reason containing `reject_reason_contains`
- **Dispatch interception**: instead of calling an actual model, the server returns `dispatch_response` directly

Failures are collected in memory during server operation.

## Config-synced integration tests (replaces the former curl smoke suite)

`make router-mock` runs the in-crate suite in `config_route_tests.rs` instead of
the old `bin/router-mock-tests.sh` shell smoke checks (which drifted from the
config — it hardcoded model names like `fast`/`tiny` that no longer exist).
The suite is derived from `env/coral-router.json` at runtime, so it cannot fall
out of sync:

- `config_route_groups_resolve_to_models` — every route's `group` names a
  non-empty `model_groups` ladder of declared models.
- `route_intents_dispatch_to_their_model_groups` — boots an in-process mock
  server, probes every route (intent), and asserts the router's own route +
  model-group validation records zero mismatches (HTTP 200 + canned answer).
- `mock_transcript_fixture_stays_synced_with_config` — `env/mock-transcripts.json`
  (the `--mock` binary's fixture) must not reference a route/model the config
  no longer declares.

```sh
make router-mock
```

## Adding a new test case

1. **Unit test**: add a `#[test]` to `stage_tests.rs` for stage 1 or 3
2. **Golden test**: add a `GoldenCase` entry to the appropriate constant array in `golden.rs`
3. **E2E test**: add a test function to `e2e_tests.rs` using `TranscriptProvider::new(entries)` with custom classifier output fixtures
4. **Mock mode test**: add an entry to `env/mock-transcripts.json`, start the server with `--mock`, and send a curl request
