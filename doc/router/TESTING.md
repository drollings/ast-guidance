# Coral Router — Testing

The workspace-wide testing convention, including the Coral Router's test
pyramid (unit / golden / e2e / config-synced / mock-mode) and the AI-inference
isolation rules, now lives in the single source of truth:

**[`doc/TESTING.md`](../TESTING.md)**

This file is retained as a pointer so existing links keep working. See
`doc/TESTING.md` for:

- the test pyramid and canonical homes (Tier 0–3),
- the shared test infrastructure (DRY/SOLID) and router-specific fixtures,
- the AI-inference isolation rules (live tests are `#[ignore]` +
  `live-ai`-gated and run only via `make test-live`),
- the Makefile/CI contract (`make test`, `make router-test`,
  `make router-mock`, `make test-live`, `make lint-live-ai`).
