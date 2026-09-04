# spacy-rs — Architecture

*The working overview of the current implementation, written for code
assistants and maintainers. For the aspirational brief see
[`VISION.md`](./VISION.md).*

spacy-rs (`src/spacy-rs/`) is a native Rust reimplementation of the core of
spaCy, composed from fluent-monorepo primitives. It is the **deterministic NLP
spine** of the stack: a tokenizer + annotation ladder + validated doc model
plus a content-addressed **interlingua** bridge that stamps every token with a
stable, model-free id. `#![forbid(unsafe_code)]`.

The crate deliberately has **no** dependency on `guidance`, `coral`, or
`wasm_ipc`; it depends on `fluent-types`, `fluent-wvr`, `fluent-concurrency`,
`fluent-dag`, and (for the routing signal model) nothing router-owned. The
router implements `ConceptStore`/`CorrectionIndex` over spacy-rs' traits.

## Module map

| Module | Role |
|---|---|
| `hash.rs` | MurmurHash64A (seed 1) — the content-addressing hash |
| `strings.rs` | `StringStore`: interned `(u64 hash → ArcIntern<str>)`, **first-wins**, durable (`to_bytes`/`from_bytes`/`save`/`load_or_empty`) |
| `lexeme.rs` | Two-level lexicon: `Lexeme` (word-type, shared by orth) + `LexiconConfig`/`OOV_RANK` |
| `vocab.rs` | `Vocab` = `StringStore` + `Lexicon` + `Morphology`; `save`/`load_or_empty` for durability |
| `doc.rs` | `Doc` + `TokenRecord`: orth/lemma/dep heads (relative offsets), tree rebuild, `to_array`/`from_array`, attribute dispatch |
| `attrs.rs` | `Attribute` ids incl. 89–91 (`InterlinguaLemmaId`/`InterlinguaEntityId`/`AnnotationConfidence`) |
| `labels.rs` | Closed `Upos`/`DepRel`/`NerType`/`EntIoB` + `DepLabelSet` |
| `lang/` | English defaults (`lang::en::tokenizer`, `lexicon_config`) |
| `tokenizer.rs` | Deterministic tokenizer over the lexicon |
| `sentencizer.rs` | Deterministic sentence-boundary prediction (`predict`/`process`) |
| `lemmatizer.rs` + `lemma_blob.rs` | Rule lemmatizer over a versioned blob built by `build.rs` from `data/en_lemmatizer.json` |
| `morph.rs`, `tag_map.rs` | Morphology + tag→UPOS map |
| `validate.rs` | The 7-check annotation gate; checks 5+6 compose `DependencyGraph` (never hand-rolled DFS) |
| `llm.rs` | `AnnotationRecord`/`AnnotationSet` (wire JSON), `AnnotationResult` + `AnnotationSource` (provenance + confidence), `LlmRefinePrompt` (span-scoped corrections contract), `attach`/`apply_with` |
| `arc_eager.rs` | The deterministic transition parser (heuristic ArcEager) + `ParseConfidence` |
| `arcready.rs` | `ArcReadyAnnotation` — a fully-materialized, immutable annotation document (`from_doc` over a `Doc` + `AnnotationResult` + routing signals), exposed as a `fluent_types::NodeOverlay` so a node can carry it in its `annotation` slot |
| `concept_store.rs` | `ConceptStore` trait + `TaxonomyHierarchy` (DAG-backed ancestors/is-subclass), `ConceptStoreState::Loading/Ready` gate |
| `concept_store_mem.rs` | `InMemoryConceptStore` — the **hermetic test double**, never production |
| `interlingua.rs` | `InterlinguaResolver` (pure), `CollisionNote`, `resolve_doc` stamping |
| `routing.rs` | `extract_routing_signals` → `RoutingSignal` (+ `InterlinguaSignal`) |
| `frame.rs` | `RoleType`/`Frame`/`AmbiguityKind`/`FrameAnalysis`, `FrameKey::permanent/provisional`, `extract_frames`, `PreferredSenseIndex` trait — the deterministic structural index (M3) |
| `taxonomy_blob.rs` | `LemmaView::from_bytes` (`SLM2` compile-time `include_bytes!`), `LazyLock<Arc<LemmaView>>` safe embed |
| `yago_view.rs` | `YagoView::load` via `std::fs::read` safe (no `mmap`), CSR parents + a `RwLock<HashMap>` ancestors memo; class lookup is a `HashMap` for now (an `fst`/`OnceLock` codegen lands later); YSM1 binary decode is not yet wired — load parses TTL directly |
| `triple.rs` | Deterministic `(subject, predicate, object)` extraction + YaGO taxonomy plausibility (`extract_triples`, `semantic_plausibility`; hermetic `Dog→Animal` tests; never touches `oracle_margins`) |
| `yago_resolve.rs` | `YagoResolveStage` between `attach` and `frame`, `semantic_plausibility` separate field (M3 guarded spike: `enabled` default off, triple-vs-taxonomy when on) |
| `cache.rs` | Span-level detail cache (`SpanCache` trait, `InMemorySpanCache` hermetic, content-addressed `span_key`; M6.1 amortized cache, read-through view over the ledger sqlite in the router — key stored as fixed-width hex `span_key TEXT` (`{:016x}`), not `i64` cast, F7) |
| `genesis.rs` | Rule genesis for POS/NER (`GenesisIndex` trait, `InMemoryGenesisIndex` hermetic, threshold-promoted version-pinned data; M6.2 pos/ner genesis — `CorrectionField::Ner` promotes `ner` on same `count >= threshold` path, `RuleAnnotator::annotate` consults `genesis.get_ner` for `ent_type`, F4) |
| `review.rs` | `ParseReview`/`Correction` (`CorrectionField::Ner` for entity type)/`ReviewStatus`, `CorrectionIndex` trait, `review_prompt`, shared `apply_edits` amendment helper (guards `old_value`: empty = don't-care, non-empty mismatch = `warn!` + skip, F6) — `ParseReview::parse_json` preserves `serde_json::Error` source (line/column) via `AnnotationError::Json { source }` (F6) |
| `retrieval.rs` | The subagent tool surface: `RetrievalSource`, `LemmaGrepHit` (confidence mandatory), `FuzzyHit`, `EmbeddingProvider`/`FuzzyRetrieval` substitution points + hermetic `InMemoryFuzzyIndex`, `cross_check` combiner (M5) |
| `pipeline.rs` + `pipeline/` | `NlpPipeline`, `StagePipeline` (DAG), `AnnotationRung` ladder + gated `AnnotationRefiner` refine phase (`LlmRefineRung`, `EncoderResidualRung`, `RefineSeams` + `span_cache`, `RuleAnnotator` + `genesis`), `RefinePolicy`/`should_refine`/`refine_focus`/`frame_coverage`, `PipelineState` |

## Core data model

- **Hashes are the identity.** `hash_utf8(text)` (Murmur64A, seed 1) is a
  token's `orth`/`lemma`; the `StringStore` maps hash → canonical string
  **first-wins**. `vocab.strings()` is the reverse lookup every consumer uses.
- **Two-level lexicon.** A `Lexeme` is shared by every token of the same orth;
  `TokenRecord` holds per-context state. This mirrors spaCy and keeps memory
  flat.
- **Relative heads (F8).** `TokenRecord.head` is a signed offset from self
  (`root` = 0 via the dep label, never a head value). `Doc::head_index`,
  `Doc::ancestors`, `Doc::children` rebuild the tree.
- **Closed vs open vocabularies.** UPOS/dep-relation/entity labels are closed
  enums; orth/lemma/tag/open dep labels are `u64` hashes resolved through the
  `StringStore`.
- **Provenance tiers.** Every ledger annotation carries a producing tier —
  `fluent-types::ProvenanceTier { Deterministic < LocalModel < Frontier <
  HumanReview }` with `AnnotationClaim`/`ClaimStatus { Provisional, Confirmed,
  Superseded }`. This mirrors the `AnnotationSource`/`ReviewStatus` pattern
  rather than a parallel scheme: `AnnotationSource::tier()` maps the producing
  rung exhaustively, higher tiers override lower ones for the same claim on the
  same node version, and claims are keyed to the node's `content_hash` (M4).

## Frame extraction (`frame.rs`)

Per predicate, a `Frame` is a typed argument structure derived deterministically
from the attached dependency tree — reusing `Doc::children`/heads, the same
discipline as validator checks 5/6. A `FrameAnalysis` pairs the frames with a
typed `AmbiguityEntry` list (attachment near-tie from oracle margins, predicate
polysemy from >1 `ConceptStore` candidate, negation/modal scope conflict;
anaphora and coordination/ellipsis are documented future work). Key minting is
the contract: an **ambiguity-free frame mints a permanent `FrameKey`** (the only
kind persisted to the ledger/graph); a frame with any open ambiguity mints a
**provisional** key and stays unresolved until a resolution is applied. Resolved
`(predicate_lemma_id, ambiguity_kind)` patterns promote into the
`PreferredSenseIndex` trait (the router's `SqlitePreferredSenseIndex` over the
`interlingua_index` pattern-cache rows, `role='sense'`), so a repeating pattern
replays deterministically — the tokenizer golden-corpus rule-genesis flow
applied to senses, zero LLM cost on the next occurrence.

## The annotation ladder (`pipeline.rs`)

The `AnnotationRung` trait returns `Result<Option<AnnotationResult>>` where
`AnnotationResult` carries records + `AnnotationSource` + per-token/parse
confidence. The ladder is a **two-phase deterministic-first** design
(ROADMAP_20260831_ARCEAGER §2.3):

### Phase 1: Base (deterministic, unconditional)

`first_accept_in_order` over `[ArcEagerRung, RuleRung]`:

1. **`ArcEagerRung`** — the deterministic parser. Always produces a parse on a
   non-empty doc (a parse that passes the 7-check gate); `Ok(None)` only on a
   genuinely empty doc.
2. **`RuleRung`** — the infallible rule rung (terminal guarantee).

This always produces a validated `AnnotationResult` — a request is never left
unparsed.

### Phase 2: Refine (model, conditional)

Only when [`should_refine`] returns `true` (controlled by [`RefinePolicy`]):
`first_accept_in_order` over `[EncoderRung, LlmRung]`:

1. **`EncoderRung`** *(optional)* — the trained-encoder seam.
2. **`LlmRung`** *(optional — only when a fetch is wired)* — the injected
   `LlmFetch`/`LlmFetchSync` seam. **Never** dials a network directly;
   hermetic tests inject stubs.

The frame-completeness view `should_refine` and the adoption gate read is the
`parse_view` helper: the parse is attached onto a **scratch clone** of the
doc, sentencized, and interlingua-resolved (when a resolver is wired) before
`extract_routing_signals` runs — so the decision sees the roles the base
*actually* resolved, not an unattached canvas. The real doc is never mutated
by the decision path.

Each refiner must pass BOTH adoption gates: the 7-check validator (well-
formedness) **and** no regression of `frame_coverage` versus the base
(routing value — both pure functions, no model involved in evaluating
either). `Ok(None)` keeps the base (fallback — never worse).

### Focused (span-scoped) refinement (M2)

`OnUncertain` refinement defaults to the **focused** path when the
corresponding [`RefineSeams`] are wired: instead of re-annotating the whole
doc, the refiner is asked to reconsider only the tokens the base flagged —
`refine_focus` derives the indices from low per-token scores, near-tie
oracle margins, and unresolved (none-sentinel) interlingua ids.

- **`LlmRefineRung`** (implements `AnnotationRefiner` over the sync
  `LlmRefineFetchSync` seam): builds the [`LlmRefinePrompt`] (base parse +
  FOCUS marks + corrections-object contract), calls the fetch, parses the
  reply into `ParseReview`-shaped corrections, **drops any edit outside the
  focus**, amends the base records via the shared `apply_edits` helper (the
  same amendment vocabulary `apply_corrections` uses — DRY), gates the
  amended set through the validator, and re-stamps provenance to `Llm`
  while the base's token/parse confidence ride along untouched (the refiner
  improves only what it touched). Empty or gate-rejected replies yield
  `Ok(None)` — base kept.
- **`EncoderResidualRung`** (M2.3, `EncoderResidualFetch` seam): the same
  focus-scoped amendment for a task-specific encoder head proposing
  per-token residuals. Opt-in only — not configured by default.

The seam decision (M2.2): `LlmRefineFetchSync` is a **new** type rather than
a composite payload on `LlmFetchSync` — the two seams carry different wire
contracts (full §10.1 array vs corrections object), and grammar-constrained
backends attach a schema per seam.

[`RefinePolicy`] controls when the refine phase runs:

- `RefineMode::Off` — base only (default for callers who don't need model
  enrichment; `NlpStage::new` defaults to `Off`, the builder selects
  `Always`/`OnUncertain` from `NlpOrdering`/`RouterRefinePolicy`).
- `RefineMode::OnUncertain` — refine only when the base is uncertain
   (low confidence, near-tie margins) or routing-incomplete (unresolved
   critical roles, unresolved PROPN tokens, collision notes). Uses the
   focused refiners when the seams are wired, falling back to full
   re-annotation adapters otherwise. The `UnresolvedPropn` trigger is a
   **task-value** fractional threshold `unresolved_token_threshold` (default
   `0.3`, **provisional** pending M5.4 production metrics): `unresolved_fraction
   > threshold` fires aggregated across the whole multi-sentence document
   (total unresolved / total tokens), not per-sentence, so the second paragraph
   is not ignored. The 0.3 value is justified only as a mechanism wiring — the
   calibration corpus `tests/refine_calibration.rs` exercises the threshold at
   0.4 vs 0.1 and control cases (precision/recall 1.0 on the hand-constructed
   set) but is not evidence for the absolute value; tune from `RefineMetrics`
   histogram without a code change.
- `RefineMode::Always` — always consult models after the base (preserves
  the old LLM-first behavior via the router's `NlpOrdering::LlmFirst`);
  Always **always** runs the full re-annotation adapters.

The sync hot path `run_ladder_sync` returns `(AnnotationResult, RefineReason)`; `process_sync_with_refine_and_reason` consumes the tuple — no second `ArcEagerAnnotator::en_default` + `parse_view` reconstruction (F1). Both ladders use per-refiner `frame_coverage` `continue` (F3). The router deserializes `RouterRefinePolicy` DTO (`fluent-router` `config/refine_policy.rs`, field-for-field mirror with `From` both ways) and converts at the builder boundary — `spacy-rs` never imports the router (F5).

The `refiner_order` helper (async) and the sync ladder both consume the one
`refine_slots` decision — the `[Encoder, Llm]` ordering **and** the
full-vs-focused slot selection are decided in exactly one place (DRY).

Either way the walk always terminates with an `AnnotationResult`, with or
without any model configured: a request is never left unparsed because a model
was unavailable, unreachable, or simply not wired.

Each worker in `process_sync`/`process_async`/`annotate_batch_async` shares the
resolver `Arc`, no locks; calls that hit the injected model are batched across
a `SupervisedBatch` wave where multiple documents in the same tick are pending,
rather than issued one call per document. The router's `NlpStage` invokes the
sync ladder (`process_sync_with_confidence(&text, self.fetch.as_ref(),
self.encoder.as_ref(), policy)`) under its own `Limiter`.

## Tokenization parity boundary & amendment

Tokenization is the crate's **one byte-for-byte parity surface**: the committed
golden corpus (`tools/gen_golden_corpus.py`, generated from pinned spaCy 3.8.15)
is replayed by `tests/en_tokenization.rs`, which asserts per token the orth,
`idx`, `spacy` flag, norm/lower/shape/prefix/suffix, and the 17 lexeme flags.
POS and dependency structure are explicitly *not* parity claims (heuristic;
VISION, "honest about its limits") — the asymmetry matters because it dictates
how each layer improves.

Consequences:

- **The LLM never owns *lexical* tokenization — it owns the sparse index above
  it.** Validator check 1 requires every annotation record's `text` to equal
  the tokenizer's orth, so annotation (pos/dep/head/lemma) *and* any sparse
  semantic tokens (entities, concepts, summary tokens) must align to a lexical
  token or a span of them; `idx`/offsets always match the raw request text
  (routing transcript, `token_ids`, ledger rows). The model's own surface is a
  separate, coarser granularity — the recognized entities and concepts behind
  `interlingua_entity_id` and `concept_ids` (VISION: entity resolution and the
  sparse index are Design-only) — which references the baseline by token-id and
  never re-derives it.
- **Boundary fixes are rule genesis at data time.** Special cases are
  version-pinned data compiled into the tokenizer (`lang/en/exceptions.rs`,
  longest-first via `filter_special_spans`). A discovered boundary disagreement
  becomes a new special-case rule plus a new golden corpus case — verified
  hermetically, then permanent and model-free. There is no runtime token
  mutation API, and the LLM never emits the lexical stream (its sparse index
  tokens are a separate, coarser surface — see above).
- **Annotation fixes are corrections.** The review boundary (`review.rs`):
  `review_prompt` (taxonomy-grounded), `apply_corrections` (dep/head/lemma/pos
  over an `AnnotationResult`, re-stamped `AnnotationSource::HumanReview`), and
  the `CorrectionIndex` the router implements over `interlingua_index` and
  persists through the `ReviewWorker`. Corrected annotations re-enter as a new
  annotation run and must pass the same 7-check gate.

## Division of labor

| Owner | Owns |
|---|---|
| Tokenizer | Boundaries + surface attributes — the detail baseline, always, model-free |
| LLM (when wired) | Enrichment — frame disambiguation, review corrections, entity linking, and (via the LLM rung) token annotation over *given* tokens; general-purpose local model behind the `ChatBackend` seam, never a task-specific fine-tune |
| LLM (index layer) | Sparse high-value tokens: recognized entities, concepts, summary tokens + vectors — aligned to the baseline, never re-deriving it |
| Validator | What is trusted — the 7-check gate arbitrates every rung and every correction |
| Confidence | When the LLM is worth consulting (escalation for deterministic-first / LLM-on-demand) |
| Review worker | Persistence of corrections (`HumanReview` provenance, ledger) |
| Caller (router) | Whether/when the `LlmFetch` seam is wired (LLM-first vs deterministic-first) |

## The stage DAG (`pipeline.rs`)

`StagePipeline` composes `DependencyGraph` of WVR components. With a resolver
wired via `StagePipeline::new_with_resolver`, the DAG is
`annotate → validate → attach → yago_resolve → frame → resolve → sentencize`:
the **`yago_resolve` stage** (`YagoResolveStage`) runs right after attach
(scoring `semantic_plausibility`); the **`frame` stage** (a `FrameStage`
WorkUnit) depends on `yago_resolved` and derives per-predicate `Frame`s + a
typed `AmbiguityEntry` list, minting provisional-vs-permanent `FrameKey`s; the
**read-only `resolve` stage** then stamps `interlingua_lemma_id` +
`confidence` on every token before sentencize. Without a resolver, the graph
collapses to `annotate → validate → attach → sentencize`. `PipelineState` (`Arc<Mutex<…>>`,
WVR handoff) carries the doc, the annotation set, the ladder's
`AnnotationResult` (for confidence), the validated flag, surfaced `CollisionNote`s,
and the stage's `frames`/`ambiguities`/`frame_keys`.

`process_sync` also resolves directly after attach (§11.8); `NlpPipeline` offers
`en_default` / `en_default_with_strings` / `new_with_resolver` /
`persist_strings`. The router builds pipelines *through the builder* with a
resolver threaded in (`NlpDeps { concept_store, strings_path }`): when
`nlp: true` and a concept store is present, the pipeline is built with
`InterlinguaResolver::new(concept_store, vocab.strings())` and a live
`NlpStage`; absent one, it `warn!`s and uses `en_default()` (fail-open).
`en_default_with_strings` / `persist_strings` are used when a strings path is
configured (durable `StringStore`).

## The interlingua bridge

- `fluent-types` owns `InterlinguaId` (16-bit namespace + 48-bit truncated
  hash), `InterlinguaNamespace` (`YagoClass`, `YagoEntity`, `SpacyLemma`,
  `RdfProperty`, `UserDefined`), `local_id_of`, `ConceptMetadata` (with a
  stored `node_id` — never derived from the truncated local, F5).
- **`InterlinguaResolver`** is stateless in the common path: `lemma_id(canonical)`
  is pure; `resolve_hash(hash, canonical)` flags a `CollisionNote` when a
  second canonical claims a taken id (first-wins, spaCy-`StringStore`-faithful);
  `resolve_doc` stamps ids + confidence on an attached doc **without writing
  the store** (registration is boot-only, C2).
- **`ConceptStore`** trait (`get`/`resolve_name`/`resolve_yago_iri`/`insert`/
  `ancestors_of`/`is_subclass_of`/…) has three implementations: the hermetic
  `InMemoryConceptStore`, the router's `SqliteConceptStore` (`interlingua_concepts`),
  and coral's durable content-addressed graph — **one loader, two homes,
  reconciled at boot**.
- **`TaxonomyHierarchy`** backs ancestor/subclass queries by composing
  `DependencyGraph<InterlinguaId>` (parents depend on children + self-provide,
  so `dependents_of(child)` = the superclass chain).

## Deterministic RDF triple enrichment (M3 — investigation + guarded spike)

**Goal:** investigate whether the deterministic ArcEager parse can be enriched
with metadata from the YaGO 4.5 taxonomy for nouns and direct objects.

**Deterministic surface today (M3.1 map):**

| Layer | Where | What it provides |
|---|---|---|
| `arc_eager.rs` | `annotate_with_confidence` → `ParseConfidence{role_coverage}` | `{nsubj,dobj}` slots filled (shallow verb-centric structure) |
| `routing.rs` | `extract_routing_signals` | per-sentence `predicate/subject/direct_object` lemmas + `InterlinguaSignal` role ids |
| `interlingua.rs` | `InterlinguaResolver::resolve_doc` | lemma → `InterlinguaId` (pure, read-only) |
| `concept_store.rs` | `ConceptStore` / `TaxonomyHierarchy` (DAG-backed) | `ancestors_of` / `is_subclass_of` over `rdfs:subClassOf` edges |
| `yago_view.rs` | `YagoView` (CSR + memo) | runtime YaGO class hierarchy (`Loading→Ready` fail-open, `ancestors_of`) |
| `frame.rs` | `FrameExtractor` | per-predicate `Frame` + `AmbiguityKind` + provisional vs permanent `FrameKey` |
| ontology | `YaGoLoader` + router `SqliteConceptStore` | the two durable homes, reconciled at boot (id-membership) |

**Design proposal (M3.2):**

1. Derive `(subject, predicate, object)` triples from ArcEager roles — the same
   `nsubj/dobj` discipline the router already uses for `role_coverage` (see
   `triple.rs::extract_triples`). One triple per sentence: the predicate is
   always the root, arguments are the first matching child per role (matching
   `routing.rs` single-slot extraction).
2. Resolve the subject/object noun lemmas to YaGO classes via `InterlinguaResolver`
   + `ConceptStore` (lemma_id → `store.contains` / `store.resolve_name("yago:Capitalized")`).
3. Score semantic plausibility by whether the predicate's type signature accepts
   the subject/object class — transitive `is_subclass_of` via `TaxonomyHierarchy`
   (`store.is_subclass_of(subjectClass, expectedDomain)`). When no explicit
   domain/range is declared, the score degrades to "argument lemma known in the
   taxonomy" (a hit counts as plausible), so `dog → Animal` via `subClassOf`
   is a hit. This mirrors the frame extractor's reuse of `Doc::children`/heads
   and the validator's DAG discipline — never a hand-rolled graph walk.
4. The score fills the existing `ParseConfidence.semantic_plausibility: Option<f64>`
   field (`arc_eager.rs:711`) — a **separate** field, never blended into
   `oracle_margins` (roadmap E7). `Loading` state yields `None` (provisional,
   mirroring `FrameKey` gating).

**Guarded spike (M3.3):** `yago_resolve::YagoResolveStage` now carries an
`enabled: bool` (default **off**). When disabled it leaves
`semantic_plausibility` as `None` (pre-M3 behavior, ladder unchanged). When
enabled it runs `triple::extract_triples` → `triple::semantic_plausibility`
(the triple-vs-taxonomy mean in `[0,1]`). Hermetic tests use
`InMemoryConceptStore` with `Dog → Animal` edges. The ladder behavior is
unchanged either way — this stage only stamps confidence, never a rung decision.

**Decision (M3.4):** keep default-off and document; the enrich signal is real
(`dog`/`cat` known vs unknown) but is not wired as a ladder gate today.

## Amortized detail cache + rule genesis (M6)

**M6.1 Span-level cache.** The refine phase's focused corrections are
content-addressed (`span_key(doc, focus)` → `hash_utf8` of lowercased focused
orths, 0x1F-separated) and cached as `Vec<Correction>` behind the `SpanCache`
trait (`Send + Sync`, object-safe). `InMemorySpanCache` (`Mutex<HashMap>`) is
the hermetic test double; the router's `SqliteSpanCache` (`ledger/span_cache.rs`)
is the production **read-through view over the shared ledger sqlite** — same
`interlingua_index` table, `role='span_cache'`, `interlingua_id = span_key`
— so no parallel store exists. The ladder checks the cache before the model
(`LlmRefineRung`/`EncoderResidualRung` with `with_cache`, both async and sync
paths) and write-throughs on a gated adoption. Invalidation is explicit
`SpanCache::invalidate(key)` when a `CorrectionIndex::record_correction` for the
same span lands (the "invalidated through the correction index" contract); a
content mutation is a different key by construction.

**M6.2 Rule genesis for POS/NER.** The tokenizer's "LLM proposes → golden
corpus accepts → data absorbs" pattern is extended from lexical boundaries to
POS/NER. `GenesisIndex` (`Send + Sync`) counts corrections per normalized
orth (`to_ascii_lowercase`) with **separate thresholds and counters**: POS
promotes at `threshold` (default 3), NER at `ner_threshold` (default 5) — entity
type is context-variant (Washington/Jordan/Paris) so the NER bar is
substantially higher and POS evidence does not accelerate it (separate
`count`/`ner_count` and `promoted`/`ner_promoted`; first value wins for each).
A promoted entry is consulted by `RuleAnnotator::annotate` before the heuristic
(`pos_of` / `ent_type`) — the correction becomes permanent, version-pinned
deterministic data. `InMemoryGenesisIndex` (`Mutex<HashMap>` +
`threshold`/`ner_threshold`) is the hermetic double; file persistence
(`load_or_empty`/`save` JSON blob, analogous to `Vocab::load_or_empty`) is the
version-pinned store the router loads at boot (old `{count,promoted}` files
migrate: a pre-split NER-promoted entry is read as `ner_promoted`+`ner_count`).
The pipeline holds an optional `Arc<dyn GenesisIndex>` (`with_genesis` rebuilds
the rule annotator) and exposes `record_genesis(corrections, doc)` so a
refiner's recurring correction amortizes to zero model cost.

## The deterministic parser (`arc_eager.rs`)

Heuristic ArcEager: `SHIFT/REDUCE/LEFT/RIGHT/BREAK` with absolute internal
heads (`-1` = unset, F8), a `DeterministicOracle` scoring label-specific POS
heuristics, `best_with_margin` reporting oracle ties, and a repair pass that
guarantees one ROOT per sentence + connectivity + acyclicity. `ParseConfidence`
= `max(0, mean(token_scores) − 0.05·ties)` with `role_coverage`. POS is
lexeme-flag + closed function-word/verb maps; PROPN fires on `is_upper()` only
(never `is_title()`). Property test: 100+ random POS sequences all pass the
7-check gate. Golden corpus in `tests/arceager_golden.rs`.

## Routing signals & review

- `extract_routing_signals(doc)` → per-sentence `RoutingSignal` (predicate/
  subject/object/arguments/modifiers + transcript) with `interlingua:
  InterlinguaSignal` (role ids, `concept_ids`, aligned `token_ids`, and a
  per-sentence `confidence` — the sentence-mean of the tokens' parse
  confidence, `None` when the producing rung carried none). `concept_ids`
  carries live entity-link candidates only via the overlay's candidate plane
  (`overlay_candidates`) — parse-time `concept_ids` stays empty; the routing
  filters (`match_interlingua`) dispatch on the role/token ids and gate on
  `confidence_min` from the signal confidence. The live request path writes
  the parse node + `interlingua_index` rows (one per sentence × role id,
  `review_status='unreviewed'`) in a single ledger transaction
  (`record_parse_node_with_confidence`), so "requests whose
  `direct_object_id` is X" is a real SQL query. The router's
  `NlpConfidenceSummary::needs_disambiguation` gates via
  `AnnotationSource::is_confidence_bearing()` (an exhaustive, no-wildcard
  classification of the producing rung — a future rung is a compile error,
  not a silent fail-open).
- `review.rs` is the async-review boundary: the `CorrectionIndex` **trait**
  (owned here), `review_prompt` (taxonomy-grounded, pure), `apply_corrections`
  → `AnnotationResult` with `HumanReview` provenance. The router implements the
  index over `interlingua_index` and owns the `ReviewWorker` and the HTTP
  endpoints (`POST`/`GET /v1/sessions/{id}/review-parse`).

## Contracts that must not break

- **7-check validator:** count/text, closed vocab, head bounds + root, one ROOT,
  connectivity, acyclicity, BILUO. Projectivity is an **optional 8th check**
  gated on `require_projectivity` (default off). Every parser output passes the
  seven mandatory checks.
- **Tokenizer owns lexical tokenization:** no annotation record may disagree
  with the tokenizer's orth (check 1) or alter a token boundary; lexical
  accuracy changes only via version-pinned rule data + golden corpus cases. The
  sparse model index (entities / concepts / summary tokens) is a separate
  surface that references the baseline by token-id and never re-derives it.
- **First-wins** everywhere ids/canonicals are stored.
- **Boot-only registration:** no pipeline stage or resolver writes the store.
  The entity-link overlay writes **candidates only** (`overlay_candidates`),
  never a doc-level `interlingua_entity_id` / `concept_ids`.
- **Provisional keys are never persisted as resolved structure.** A frame with
  an open ambiguity mints only a provisional `FrameKey`; only ambiguity-free
  (permanent) keys reach the ledger/graph. Resolved patterns promote into the
  `PreferredSenseIndex`, never back into the deterministic extractor's output.
- **`Arc<dyn Runtime>`** everywhere async (WVR); no ambient `tokio::spawn`.
- **No `guidance`/`coral`/`wasm_ipc` imports** in this crate.

## Known gaps

- `triple.rs token_known` per-token alloc — deferred, profile before fixing.

## Verification

`cargo test -p spacy-rs` (unit + golden + integration), `cargo clippy -p spacy-rs`.
The router suites (`cargo test -p fluent-router`) cover the integration:
`NlpStage` accessors, `match_interlingua` filters, ledger interlingua tables,
and the review worker.