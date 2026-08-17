# fluent-router — Live-AI tests

Tests in this directory perform **real model calls** against a live inference
endpoint. They are **never** part of the default test suite: each is
`#[ignore]`d **and** gated behind the `live-ai` feature (so the test file is
only compiled when the feature is enabled) and they are excluded from
`make test` / `make router-test` / `make router-mock` / CI. The only way to
run them is:

```sh
make router-test-live   # or the workspace-wide:
make test-live
```

## Env contract

| Variable | Required | Description |
|---|---|---|
| `LLM_BASE_URL` | yes | OpenAI-compatible chat-completions base URL (e.g. `http://127.0.0.1:11434/v1`) |
| `LLM_MODEL` | yes | Model name to request |

The router's live test drives the real `PipelineOrchestrator` with an injected
`LlmClient` backend pointed at `LLM_BASE_URL`. It never boots the managed
`llama-server` supervisor, and its config model endpoint is a refused loopback
(`127.0.0.1:1`) so the test process itself cannot dial a real host.

## Skip-not-fail policy

Every live test guards on the env contract with `std::env::var` + early
`return` and **never panics** when `LLM_BASE_URL`/`LLM_MODEL` are absent — it
prints a notice and passes (skips). This keeps `make test-live` green on
machines without a model endpoint.

## What is asserted

Only **structural invariants** (stage order, valid pipeline outcome / routing
target present; well-formed bounded response text). Never model output quality
or routing-decision quality — model outputs are non-deterministic and are not
asserted.
