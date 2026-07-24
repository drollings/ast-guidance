# Coral Router - Vision

## Mission

Coral Router is a local-first control plane for LLM traffic: a single OpenAI-compatible endpoint that decides, for every request, the cheapest and safest way to answer it — deterministic logic where possible, a small local model where sufficient, larger local models where warranted, and frontier providers only when genuinely necessary. The goal is to make a local workstation's mixture of models behave like one coherent, cost-aware assistant rather than a pile of separately-addressed endpoints.

## Source code location

The source code may be referenced at ./src/router/src/

Do not read ./src/router/src/tests unless unit tests and e2e tests are specifically being worked on.

## Current status

The pipeline is live at a basic level:

1. **Deterministic pre-filter** — regex-based rejection patterns and PII detection (SSN, card number, email, phone) run before any model is invoked.
2. **Fast classifier/router** — a small local model (`fast`) evaluates coherence and safety, decides whether to answer trivially, route the request onward, or reject it outright, and emits its verdict as structured JSON rather than free text.
3. **Routing to a model group** — verdicts route to one of three configured destinations: `fast` (the classifier's own model, for simple queries), `code`, or `agent`, each with its own context size, timeout, and generation settings.
4. **Guardrail** — currently scoped to frontier-bound traffic only; local agent calls are not yet guardrail-checked.
5. **Session and cache infrastructure** — session context compacts by recency past a node-count threshold; KV cache state is tracked across a hot (in-memory, size-bounded) and cold (disk-backed, TTL- and LRU-evicted) tier.

This is a working, opinionated first cut, not the end state. Two things are visibly not yet wired up: there is no distinct orchestrator model (the `orchestrator` route currently aliases to `code`), and the adapter registry is empty — role specialization today comes from model choice alone, not from adapter switching on a shared base model.

## Design principles

- **Deterministic before probabilistic.** Anything that can be decided with a regex or a fixed rule should never reach a model call. This is a cost and latency floor, not an optimization — it also gives the system a layer that's fully unit-testable without any model in the loop.
- **Cheap before expensive.** Every model in the config carries its own cost and speed profile. Routing decisions are economic decisions as much as capability decisions: escalate only as far as a request actually requires.
- **Condensed context, not accumulated context.** Sessions compact rather than grow without bound. The orchestrator's working context should stay small and high-signal; raw exploration and dead ends belong in durable storage, not in the live session.
- **Local-first, frontier as an escape hatch.** Frontier providers are for genuine difficulty, privacy-sensitive decomposition, or capability gaps — not a default. Local models should absorb as much traffic as they credibly can.
- **Auditable by construction.** Every stage — filter, classify, route, guardrail — should produce a legible reason, not just a verdict, so a rejected or rerouted request can be explained after the fact.

## Near-term direction

- **A real orchestrator role**, distinct from `code`, holding a long-lived session with its own KV cache and acting as the durable record-keeper for a task rather than a stateless classifier target.
- **Adapter-based agent specialization.** The adapter registry exists in configuration but is unpopulated; the intended model is a small number of resident base models serving many narrow specialist roles via per-request LoRA adapter selection, rather than one model per role.
- **Guardrail coverage for local agents**, not just frontier egress, once the cost of doing so locally is negligible.
- **Scheduling with cache affinity.** As agent+adapter+session combinations multiply, request ordering should actively minimize KV-cache context switches rather than serving requests in arrival order.

## Longer-term direction

- **Structured frontier fallback**, beyond a single escalation path: pure fallback for hard problems, PII-anonymized fallback for sensitive content, and decomposition into small, anonymized, rubric-validated hypothetical questions for problems that only need a narrow piece of frontier reasoning.
- **A session model that looks like a build graph** — steps with tracked dependencies, the ability to rewind to a checkpoint and rebuild forward, rather than a flat linear transcript.
- **Levels-of-detail context compaction** that goes beyond recency — older or resolved work compresses to a short "completed" summary rather than aging out uniformly, with the full record retained in storage even after it leaves the live session.

## What this project deliberately is not

- Not a general-purpose LLM gateway or multi-tenant API product — it's built for one local workstation's traffic.
- Not a wrapper around a third-party gateway crate's routing or caching logic — routing, scheduling, and caching are purpose-built around KV-cache affinity, which generic LLM gateways have no concept of.
- Not reliant on frontier models for anything a local model can be made to handle credibly — frontier usage is a deliberate, bounded exception, not the default path.
