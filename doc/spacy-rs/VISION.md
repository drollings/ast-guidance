# spacy-rs — Vision

*This document is the aspirational brief: the goals of spacy-rs and its ideal
finished design. It deliberately does not track what is landed today — that
lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md), which describes the current
implementation and which pieces are load-bearing.*

> **Section status legend.** Each section below carries a `Status:` line so
> the vision stays honest without being rewritten as it lands:
>
> - **Implemented** — the described shape exists today (details in ARCHITECTURE.md).
> - **Partial** — a working core exists; the section describes extensions or
>   refinements not yet built.
> - **Design-only** — aspirational; nothing (or only scaffolding) exists yet.
>
> Marking a section does not make it current — it tells the reader how much of
> the vision to expect. When a section's status changes, update the line here.

## Mission

spacy-rs is a native, idiomatic Rust reimplementation of the core of spaCy
(`/opt/src/nlp/spaCy`, v3.8.15), composed from the fluent-monorepo's own
primitives. Its mission is not to clone spaCy for its own sake: it is to be the
**deterministic spine** of the monorepo's LLM infrastructure — a fast,
**confidence-scored structural index primitive** that gives Coral Router and
the orchestration stack a unit-testable, model-free understanding of text
before any model is consulted. Enrichment — frame disambiguation, review
corrections, entity linking — is supplied by a **general-purpose local model**
served behind the `fluent_llm::client::ChatBackend` seam, never a task-specific
fine-tune: spacy-rs derives the structure, the model resolves the residue.

To anything calling it, spacy-rs is a pipeline: tokenize → annotate →
validate → attach → **frame** → resolve → sentence boundaries, producing a `Doc`
whose tokens carry surface forms, lemmas, POS, dependency heads, and — the
defining contribution — **content-addressed interlingua ids** that are pure
functions of the text and the YaGO 4.5 taxonomy. The `frame` stage derives a
typed argument structure per predicate (`predicate_lemma_id`, role slots,
polarity, modality) with a typed ambiguity list, minting a **permanent**
interlingua key only for an ambiguity-free frame (a provisional key otherwise,
resolved via the model and promoted into the `PreferredSenseIndex` —
golden-corpus rule genesis applied to senses). Those ids are what let
deterministic routing, durable ledgers, and knowledge graphs agree on what a
request is *about*, without any of them asking a model.

The animating belief behind the project: **an LLM should be consulted about
ambiguity, not about things that are already decided.** If "show me the report",
"display the report", and "get the sales report" all collapse to the same
predicate + object ids, then the router already knows what is being asked — the
LLM's job is the residue, not the whole.

## Design principles

- **Deterministic before probabilistic.** The tokenizer, the lemmatizer, the
  sentencizer, the validator, and the interlingua resolver are pure and
  model-free. Every id is a pure function of content. Nothing in this crate
  should require a network call or a model to produce its primary outputs.

- **Honest about its limits.** spacy-rs is *not* a claim to spaCy parity. Its
  POS is heuristic (lexeme flags + a closed English function-word map), its
  dependency parser is a deterministic transition parser whose oracle is a
  hand-coded heuristic rather than a trained model, and it says so loudly.
  Confidence is a first-class output: the parser reports margin-aware
  confidence, and the routing layer is told which rung produced a parse and how
  much the oracle doubted itself. When the heuristic is wrong, the system says
  "needs disambiguation" rather than pretending otherwise.

- **The LLM is the fallback, not the default.** Producing a base parse is a
  two-rung, unconditional, model-free walk: `deterministic parser → rule star`.
  The deterministic parser *always* returns a parse (confidence gates routing,
  never rung fallthrough); the rule rung is reached only on a genuinely empty
  doc. A request is never left unparsed because a model was unavailable,
  unreachable, or simply not wired — the deterministic floor holds regardless.

  The LLM is not a competing rung in this walk — it cannot be, since the
  deterministic parser's always-accept contract means nothing after it in an
  accept/defer order could ever be reached. Instead, the LLM is a **selective
  enrichment step applied after** the base parse, gated on confidence: when the
  deterministic layer's `ParseConfidence`/role-coverage signal says the parse is
  genuinely uncertain, the LLM is consulted to amend it; when the base parse is
  confident, the LLM is never called at all. This is the literal meaning of
  "fallback, not default" — the model is consulted about the residue the
  deterministic layer flagged as doubtful, never run as a matter of course on
  every request.

  Enrichment output is passed through the same 7-check gate as any other
  annotation source before it's allowed to amend the base result, and it
  re-stamps provenance (`ProvenanceTier::LocalModel`/`Frontier`) rather than
  silently overwriting the deterministic-tier base. Whether enrichment runs at
  all is the caller's choice: no `LlmFetch` wired means the deterministic base
  result is returned unamended — a fully supported mode, not a degraded one.

  There is no separate "LLM-first" ordering in which the model is consulted
  before the deterministic parse exists to be judged: the deterministic parser
  always runs and always produces a result first, by construction. A caller
  wanting model input on every parse regardless of confidence can configure
  `should_enrich` to always enrich — but that is a call-site policy on top of
  the same two-stage shape (parse, then optionally enrich), not a different
  rung ordering, and it does not change what the deterministic layer itself
  guarantees.

- **Two token surfaces: the detail baseline and the model's index.** Lexical
  tokenization — boundaries and surface attributes (orth, idx, spacy, lower,
  shape, flags) — is *decided* fact, not ambiguity, and the deterministic
  tokenizer owns it. It is the **detail baseline**: exhaustive, offset-exact
  against the raw request bytes, reproducible, and free (the hot path). Above
  it, the LLM maps to a **sparse, high-value surface**: the entities, concepts,
  and summary tokens (and the vectors behind them) the model recognizes. The
  LLM holds the *index*, not the content — it emits few, coarse semantic tokens
  that reference the baseline by span/token-id, never the exhaustive lexical
  stream. This is exactly why validator check 1 requires every annotation
  record's `text` to equal the tokenizer's orth: whatever the model asserts must
  align to a lexical token or a span of them. The model can add a layer *above*
  the baseline; it can never contradict it. Four properties make the baseline
  load-bearing: (1) **alignment** — routing transcripts, `token_ids`, and ledger
  rows index the exact request bytes; (2) **reproducibility** — the ledger, the
  content-addressed graph, and the router must agree on what a request *is*,
  run after run; (3) **the hot path** — lexical tokenization runs on 100% of
  requests and costs microseconds per token versus milliseconds-to-seconds for a
  model call; (4) **fail-open** — the router must understand text with no model
  reachable. Lexical accuracy therefore improves by **rule genesis, not runtime
  models**: a discovered boundary disagreement is compiled back into the
  version-pinned special-case table and committed as a new golden case (LLM
  proposes → validator + golden corpus accept → the tokenizer absorbs the rule
  as deterministic data). The sparse index is the *complementary* surface, and
  it is the LLM's proper home — entity recognition and summary tokens live there
  (see "When to use which layer"). Whether the LLM is consulted at all — and for
  which requests — is a policy of the caller that injects the `LlmFetch` seam,
  not a crate decision: deterministic-first with LLM-on-demand is as supported
  as the LLM-first ladder.

- **Pure and shared by design.** The resolver is stateless in the common path:
  ids are pure functions, so every `ResultPool` worker and the whole pipeline
  share one `Arc<dyn ConceptStore>` with no locks. Registration is boot-only;
  nothing in the data plane writes the concept store. Corrections happen in an
  asynchronous review worker that never blocks the hot path.

- **Content addressing is the point.** "cat" and "https://schema.org/Person"
  both become stable 64-bit ids under the same discipline: truncate a content
  hash to 48 bits, place it under a 16-bit namespace. Different systems —
  annotated text, RDF knowledge stores, routing tables — speak the same id
  language, so they can be stored, queried, and reconciled against each other
  without translation.

## Status

- **Core spaCy reimplementation (tokenizer, vocab, doc, labels, lemmatizer,
  validator):** *Implemented.* The two-level lexicon, hashed string store,
  dependency-tree rebuild, and the 7-check annotation gate all exist and are
  unit-tested.

- **Deterministic transition parser (heuristic ArcEager):** *Implemented.*
  `SHIFT/REDUCE/LEFT/RIGHT/BREAK` with an oracle that derives
  `nsubj/dobj/prep/pobj/compound/det` structure, margin-aware confidence, and a
  property-tested guarantee that every output passes the 7-check validator.

- **Interlingua bridge:** *Implemented.* `InterlinguaId` (16-bit namespace +
  48-bit truncated hash), the pure resolver, collision surfacing, the
  taxonomy-grounded concept store, and doc stamping (`resolve_doc`). The
  resolver is wired into the live request path at boot: a pipeline with
  `nlp: true` is built with the shared `SqliteConceptStore`-backed resolver
  (built before the pipeline build), so real `interlingua_lemma_id` /
  `predicate_id` / `direct_object_id` + confidence are stamped and
  `match_interlingua` filters and the classifier's interlingua context are
  live, and the parse node + `interlingua_index` rows land on the ledger in
  one transaction.

- **YaGO 4.5 taxonomy:** *Implemented.* Decoupled `SLM2` lemma (compile-time) + `YSM1` YaGO (runtime `fst`+CSR lazy, `Loading→Ready` fail-open provisional, `sha256` pinned) with hermetic `n2` fixture; `YagoView` safe `read`, `YagoResolveStage` `semantic_plausibility` separate, `FrameKey` permanence gated on `Ready`.

- **Async review mechanism:** *Implemented.* The `CorrectionIndex` trait, the
  SQLite implementation, the review worker (a bounded `WorkerPool` + credit
  gate), the taxonomy-grounded review prompt, the `parse_review` ledger
  handoff, the HTTP endpoints, and the server-ownership wiring are all live
  (`POST`/`GET /v1/sessions/{id}/review-parse`).

- **Entity resolution (PROPN → YaGO entity, `interlingua_entity_id`):**
  *Partial.* The field, the attribute id, the signal plumbing, and the async
  entity-link overlay (a credit-gated worker that scores unresolved PROPN
  spans against boot-baked ColBERT concept-label embeddings and writes
  candidates to the `overlay_candidates` ledger table) exist. The overlay
  writes **candidates only** — it never stamps a doc-id at parse time, so
  parse-time entity resolution (populating `interlingua_entity_id` /
  `concept_ids` directly) remains future refinement.

- **Sparse model index (LLM-owned summary tokens / concepts / vectors):**
  *Partial.* The deterministic baseline is the detail; the model's sparse,
  high-value tokens are the index it holds over that detail. The alignment
  plumbing exists (`token_ids`, `concept_ids`, `interlingua_entity_id`), and
  the entity-link overlay above is its first live instance (candidate-plane
  only). The surface that produces the sparse tokens for routing use is
  future work.

- **ArcReady annotation document (`ArcReadyAnnotation`):** *Implemented.* A
  fully-materialized, immutable annotation document
  (`spacy-rs::arcready::ArcReadyAnnotation`, built by `from_doc` / `arc_ready`)
  exposed as a `fluent_types::NodeOverlay` so a shared node can carry it in its
  `annotation` slot. The router's background overlay worker
  (`ledger/overlay_worker.rs`, opt-in via `overlay.arc_ready`) derives it —
  alongside the LLM enrichment and embedding overlays — lazily, at-most-once,
  and in parallel from LOD0.

- **Frame extraction stage (`frame.rs`):** *Implemented.* A `FrameStage` between
  `attach` and `resolve` derives per-predicate `Frame`s (`predicate_lemma_id`,
  role slots with token spans + candidate concept ids, polarity, modality) from
  the attached dependency tree, emits a typed `AmbiguityEntry` list
  (`AttachmentNearTie` from oracle margins, `PredicatePolysemy`, `NegationModalScope`;
  anaphora and coordination/ellipsis documented future work), and mints a
  **permanent** `FrameKey` only for ambiguity-free frames — a frame with any
  open ambiguity gets a **provisional** key and is never persisted as resolved
  structure. Resolved `(predicate, ambiguity_kind)` patterns promote into the
  `PreferredSenseIndex` (the router's `FrameResolutionWorker`, wave-batched, one
  grammar-constrained call per tick) so the same pattern never re-triggers an
  LLM call.

- **Provenance tiers (`ProvenanceTier`):** *Implemented* (in `fluent-types`, the
  router's `AnnotationStore`). Every ledger annotation carries a producing tier
  (`Deterministic < LocalModel < Frontier < HumanReview`); `AnnotationSource::tier()`
  maps the producing rung exhaustively, a higher-tier claim supersedes (never
  deletes) a lower-tier one on the same node version, and annotations are keyed
  to the node's `content_hash` so a content mutation invalidates them by keying,
  never a scheduler.

- **Subagent tool surface (`retrieval.rs`):** *Implemented.* Lemma-grep hits
  carry their `ParseConfidence` + `interlingua_lemma_id` on every span; a fourth
  embedding-based fuzzy retrieval tool covers the paraphrase gap (different
  lemmas, same intent); a `cross_check` combiner surfaces both lemma and fuzzy
  hits when they materially disagree on the same region. The router's
  `NodeRetrievalService` (`router::retrieval`) is the **live dispatch seam** —
  it parses a candidate node's LOD0, runs these tools, and pre-filters the pool
  through the `SalienceRanker`, exposed on the agent coordinator via
  `with_retrieval`/`retrieve_nodes`. The tool-calling agent loop that invokes it
  is the remaining consumer (none exists yet).

## What it accomplishes

spacy-rs gives the monorepo a shared, model-free understanding of language that
downstream systems build on:

- **Deterministic routing inputs.** The `NlpStage` publishes per-sentence
  routing signals (`predicate`, `subject`, `direct_object`, arguments, and the
  interlingua frames). The classification tree's `match_interlingua` filters
  dispatch on the ids directly — the same phrasing collapses to the same route
  with zero tokens. The classifier is *given* the parsed grammar as
  deterministic context rather than re-deriving it, and low-confidence parses
  are flagged "needs disambiguation" for a more capable model.

- **A durable, queryable understanding.** Lemmas, ids, confidence, and review
  status land in the ledger's `interlingua_index` and `interlingua_concepts`
  tables, and in coral's content-addressed graph — the same taxonomy feeding
  both, reconciled at boot. "Show me the report" from yesterday and today are
  the same ids, in the same tables, joinable to the same knowledge graph. Every
  annotation is tiered (`Deterministic < LocalModel < Frontier < HumanReview`)
  and keyed to the node's content hash, so a higher-authority claim supersedes
  (never silently coexists with) a lower one and a content mutation invalidates
  cached claims by keying.

- **A deterministic frame layer.** Per-predicate `Frame`s — a typed role
  structure plus a typed ambiguity list — give the router a structural index
  that is cheap to query and honest about uncertainty: ambiguity-free frames
  mint permanent interlingua keys that are persisted, frames with open
  ambiguities mint only provisional keys resolved through the model and
  promoted into the `PreferredSenseIndex` when the pattern repeats. The model is
  consulted about the residue — never asked to re-derive structure that is
  already decided.

- **An honesty layer.** The deterministic parser tells you when it is guessing
  (oracle ties, role coverage, source). That honesty is what lets the router
  spend a more capable model only where the deterministic layer was genuinely
  unsure — the escalation philosophy of VISION, made legible in data.

## When to use which layer

The division of labor is a consequence of the principles above:

| Job | Right tool | Why |
|---|---|---|
| Token boundaries, idx, surface attributes | Deterministic tokenizer — **always** | decided facts; must align to the raw request; the hot path; reproducible across systems |
| POS / dep / head / lemma on common phrasing | Deterministic parser (free, fail-open) | gated-valid, confidence-scored; flags "needs disambiguation" instead of guessing silently |
| Disambiguation, frame resolution, full UD depth, domain semantics | General-purpose local model (grammar-constrained `ChatBackend`, e.g. the onnx LLM) | ambiguity is the residue worth a model call; output is gate-validated with a deterministic fallback beneath, and structured decoding prevents invalid shapes at generation time |
| Entity recognition, concepts, summary index | LLM — sparse tokens + vectors, aligned to the baseline | sparse, high-value tokens index the content; the LLM holds the index, not the detail |
| Fixing tokenizer boundary mistakes | Rule genesis: LLM proposes → golden corpus accepts → data absorbs | turns stochastic findings into permanent, model-free rules |
| Repeating frame ambiguities | Sense promotion: LLM resolves once → `PreferredSenseIndex` absorbs | the same `(predicate, ambiguity_kind)` pattern replays deterministically, zero LLM cost — golden-corpus rule genesis applied to senses |
| Amending annotations | Async review worker (`apply_corrections`, `HumanReview` provenance) | persisted, audited, and re-validated through the same gate |

## Where it is going

The long arc is a single, content-addressed understanding shared by the whole
stack — text, knowledge, and workflows speaking the same id language, with the
LLM consulted only about the residue. spacy-rs is the text half of that bridge;
YaGO is the knowledge half; the router, the ledger, and coral are the halves
that consume both.
