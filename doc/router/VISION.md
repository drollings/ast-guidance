# Coral Router — Vision

*This is a design vision, not a status report.  It describes what Coral Router is for and what a fully realized version of it should be.  For current implementation state, see the project roadmap; for detailed mechanism, see the MOA router specification.*

## Mission

Coral Router is a local-first control plane for LLM traffic: a single OpenAI-compatible endpoint that decides, for every request, the cheapest and safest way to answer it — deterministic logic where possible, a small local model where sufficient, larger local models where warranted, and frontier providers only when genuinely necessary.  To anything calling it, it behaves like one coherent, capable model.  Underneath, it's a disciplined mixture of deterministic filters, small classifiers, local reasoning models, and occasional frontier calls, none of which are consulted unless a cheaper stage has already failed to resolve the request.

## Design principles

- **Deterministic before probabilistic.** Anything decidable by a regex or a fixed rule should never reach a model call.  This is a cost and latency floor, not an optimization — it also gives the system a layer that's fully unit-testable with no model in the loop.

- **Cheap before expensive.** Every model carries its own cost and speed profile.  Routing is an economic decision as much as a capability one: the ladder runs deterministic filter → fast classifier/score-matrix → local orchestrator or agent → frontier, and a request only reaches a given rung after the previous one has genuinely failed to resolve it — never by default.

- **Condensed context, not accumulated context.** Sessions compact rather than grow without bound.  A ledger stands between raw session history and the orchestrator's live KV cache, so the orchestrator never has to reason over noise, dead ends, or superseded exploration — that material stays in durable storage, retrievable if needed, but off the model's working context.

- **Local-first, frontier as a bounded, audited exception.** Frontier calls are for genuine difficulty, privacy-sensitive decomposition, or a real capability gap — never a default path.  Every frontier interaction writes back to either a durable audit log or a reusable local artifact (a stored workflow, a validated rubric/answer pair).  The measure of whether this design is working is frontier-call frequency trending *down* over the life of an installation as those local libraries fill in — not staying flat.

- **Terminate, don't loop.** Anywhere the system reaches for more than one model pass on a single request, the round count is fixed in advance, never open-ended.  This is a deliberate rejection of the failure mode common to multi-agent ensembles and debate systems, which run every available model on every query with no adaptive gating and burn tokens accordingly.  Escalation past a fixed structure happens only on a specific, named trigger — never as a default resolution to disagreement or difficulty.

- **Auditable by construction.** Every filter, classification, route, and frontier decision produces a legible reason alongside its verdict.  A rejected, redirected, or escalated request should be explainable after the fact without guesswork.

- **Reuse infrastructure, extend it, don't parallel-build it.** A component that reimplements something shared infrastructure already provides — graph algorithms, hashing, config loading, error handling — is a defect to fix, not a style choice to debate.  Specialized needs are met by extending general-purpose primitives, not by forking them.

## The fully realized system

A request arrives and passes through a strict escalation ladder, spending as little as possible at each rung before the next is even considered.

**Deterministic filters** run first, with no model in the loop: whitelists, blacklists, and pattern rules that resolve to one of three outcomes — a hard rejection that ends the request outright, a soft redirect that sends it down a different path, or an output filter that redacts, anonymizes, or omits specific content before anything continues.  These filters are scoped (some apply only to frontier-bound traffic) and can be gated behind a secondary check, so a rule never fires on a bare pattern match alone when a cheap confirmation is available.

**A fast classifier** — small, fast, running in parallel across many requests — evaluates intent, coherence, safety, and complexity, and resolves the result through a weighted score matrix rather than nested thresholds.  Most requests are fully decided by this point: answered trivially, routed to a specific local model, or rejected, all without touching the system's larger models.

**A ledger** records the session as it unfolds, at full detail as it happens and compressing toward short summaries as it ages or as work resolves.  This ledger — not raw history — is what fills the orchestrator's context, so the orchestrator's own reasoning stays high-signal regardless of how long or exploratory a session gets.  Abandoned approaches and dead ends collapse to a single line rather than vanishing, so nothing is silently forgotten, but nothing bloats the live context either.

**Two purpose-built routes** handle requests that don't fit the standard path.  A vague or underspecified request goes through **planning**: matched against a library of prior workflows where possible, or built fresh by identifying exactly what's missing and asking the user a short, targeted set of questions to fill the gap — never an open-ended back-and-forth.  A complete but high-stakes request can go through **rigor**: a fixed blue-team/red-team/judge sequence, checkpointing the reasoning model's KV cache first so a red-team-identified dead end can be rewound rather than argued out of in place.  When red team raises something material, the default resolution is a targeted interview with the user — not silent escalation.

**Local reasoning models handle the bulk of real work** — an orchestrator holding a long-lived, condensed session, and specialist agents reached via adapter switching on shared base models rather than one model per role, scheduled with awareness of KV-cache affinity so context switches are minimized rather than incidental.

**Frontier models are the last, narrowest rung**, used in one of a small set of deliberate modes: a pure fallback for problems genuinely beyond local capability; a PII-anonymized fallback for sensitive content; a decomposed, anonymized hypothetical question with a validation rubric, for when only a narrow piece of frontier reasoning is needed; or a copilot/judge role reviewing the local model's in-progress reasoning at checkpoints.  Every mode is logged to a durable, separate audit trail, and every frontier answer that proves out feeds back into a stored workflow or a validated rubric — so the same class of question never has to pay frontier cost twice.  Retrieval against these stored artifacts, and against the blacklist's harder-to-express categories, runs through similarity search — kept as separate indices for each purpose, since a false positive means something different, and costs something different, in each case.

The system as a whole should feel, from the outside, like a single capable assistant.  From the inside, it should be legible at every step: which rung handled a given request, why, and what it cost.

## What this project deliberately is not

- Not a general-purpose LLM gateway or multi-tenant API product — it's built for one local workstation's traffic.

- Not a wrapper around a third-party gateway crate's, or reference project's, routing, auth, or caching logic.  Such projects are useful for mining patterns, never for importing as dependencies.  Routing, scheduling, and caching are purpose-built around KV-cache affinity, which generic LLM gateways have no concept of.

- Not reliant on frontier models for anything a well-scaffolded local model can be made to handle credibly — frontier usage is a deliberate, bounded, audited exception, not the default path.

- Not an ensemble-by-default system.  Unlike designs that improve output quality by running every available model on every query with no cost constraint, Coral Router treats every additional model call — local or frontier — as something a prior, cheaper stage must have failed to resolve first.  Quality comes from routing and verification discipline, not from brute-force ensembling.

