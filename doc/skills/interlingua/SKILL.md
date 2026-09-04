# Interlingua — unified, disambiguated IDs from text & YaGO 4.5

**Context**: the monorepo's cross-system identity layer. An **interlingua ID**
is a single 64-bit value that names the *same thing* whether it came from
annotated text (a spaCy lemma), an RDF knowledge store (a YaGO 4.5 class or
property), or a deterministic routing action. This skill is the reference for
how lemmas and YaGO schemas become those IDs, how collisions are handled, and
how the ID unifies storage across LLM context, databases, and knowledge
graphs.

**Sources of truth**: `src/types/src/interlingua.rs` (the ID type),
`src/spacy-rs/src/interlingua.rs` (the resolver), `src/spacy-rs/src/pipeline.rs`
+ `arc_eager.rs` (the annotation ladder — LLM/encoder rungs or the
deterministic parser — that produces the lemma annotations),
`src/fluent-onnx/src/encoder.rs` + `colbert.rs` (the trained-encoder rung and the
entity-link index), `src/ontology/src/yago_loader.rs` + `yago.rs` (the YaGO 4.5
schema), `src/router/src/concept_store_sqlite.rs` (the durable store),
`src/coral/src/db/nodes.rs` (the content-addressed graph).

## Why a unified ID layer

Three systems had to agree on what a request was about:

- **Annotated text** knows the lemma `"report"` but not that it is the same
  thing across phrasing ("show the report" / "display the report").
- **YaGO 4.5** knows `http://yago-knowledge.org/resource/Person` but not that
  a token `"NASA"` refers to it.
- **Routing/workflow storage** wants to look up "requests whose object is the
  report" without running a model.

The interlingua is the agreement: everything is content-addressed, so the
same content — lemma, IRI, or action — yields the same id everywhere, with no
central registry consulted on the hot path.

## The ID shape

`InterlinguaId(u64)` = **16-bit namespace** (bits 48–63) + **48-bit truncated
content hash** (bits 0–47).

| Namespace | Value | Content hash | Example content |
|---|---|---|---|
| `Reserved` | `0x0000` | — | (no ids) |
| `YagoClass` | `0x0100` | `hash_iri` (BLAKE3) | `http://schema.org/Person` |
| `YagoEntity` | `0x0200` | `hash_iri` | `http://yago-knowledge.org/resource/Paris` |
| `SpacyLemma` | `0x0300` | `hash_utf8` (Murmur64A seed 1) | `"report"`, `"show"` |
| `UserDefined` | `0x0400` | caller-defined | routes, actions |
| `RdfProperty` | `0x0500` | `hash_iri` | `rdfs:subClassOf` |

**Derivation rules** (never a raw constructor — there is deliberately no
`from_raw`, F1):

- `local_id_of(full_hash) = full_hash & 0x0000_FFFF_FFFF_FFFF` (truncate to 48).
- Lemma id: `SpacyLemma::new(local_id_of(hash_utf8("report")))`.
- Class id: `YagoClass::new(local_id_of(hash_iri("http://schema.org/Person")))`.
- Property id: `RdfProperty::new(local_id_of(hash_iri(prop.iri)))`.
- `node_id` (the coral graph key) is the **full 64-bit** `hash_iri` — stored on
  `ConceptMetadata`, **never derived** from the truncated 48-bit local (the 16
  truncated-away bits are unrecoverable, F5).

## Lemmas → ids (the text side)

1. **Annotate via the fallback ladder.** The pipeline runs
   `first_accept_in_order` over `LlmRung → EncoderRung → ArcEagerRung →
   RuleRung` (`src/spacy-rs/src/pipeline.rs`). The LLM rung (a `ChatBackend`
   bridged by the router's `stages/nlp.rs::annotation_fetch`) and the
   trained-encoder rung (`fluent-onnx`, built in `src/router/src/ort.rs` via
   `nlp_encoder_fetch`) produce full Universal-Dependency-quality parses; the
   deterministic `arc_eager.rs` heuristic parser and the infallible rule rung
   are the fail-open falls. Every rung's output passes the 7-check gate, so
   the lemmas the resolver sees are always validated. Lemmas come from the
   rule lemmatizer.
2. **Resolve.** `InterlinguaResolver::resolve_doc` stamps
   `interlingua_lemma_id` (+ `confidence`) on every token — **pure and
   read-only**; the concept store is never written by the data plane.
3. **Surfaces in routing signals.** `extract_routing_signals` emits
   `InterlinguaSignal { predicate_id, subject_id, direct_object_id,
   indirect_object_id, concept_ids, token_ids, confidence }` per sentence —
   `confidence` is the sentence-mean of the tokens' parse confidence (`None`
   when the producing rung carried none). The classifier's `match_interlingua`
   filter nodes dispatch on these ids directly — same phrasing → same ids →
   same route, zero tokens — and enforce the node's `confidence_min` gate from
   that signal confidence (a sentence whose tokens carry no confidence is
   treated as below the floor, fail-closed). `concept_ids` carries live
   entity-link **candidates** only, via the async overlay (`server/entity_link.rs`,
   `ledger/overlay.rs`): a credit-gated `WorkerPool` scores unresolved PROPN
   spans against the boot-baked ColBERT `EntitySimilarityIndex`
   (`fluent-onnx::colbert`, built in `src/router/src/ort.rs::colbert_entity_scorer`)
   and writes to `overlay_candidates`. **Parse-time** `concept_ids` / `interlingua_entity_id`
   stamping remains future (boot-only registration is an invariant); `token_ids`
   never carries the `InterlinguaId(0)` none-sentinel (the correction cache
   must not key on fake ids).

## YaGO 4.5 → ids (the knowledge side)

`guidance-ontology` provides the **one loader** (`YaGoLoader`) that feeds the
**two durable homes**, and a boot reconciliation (C3/§11.7/§13.10) that fails
loudly on drift:

1. **coral's content-addressed graph** — `context_nodes` keyed by the full
   `hash_iri` (`node_id`).
2. **the router's `SqliteConceptStore`** — `interlingua_concepts` keyed by
   `(namespace, canonical_name)` with `id` as a plain indexed column
   (collision-tolerant: two canonicals that truncate to the same 48-bit local
   id are both stored, each unique by canonical name), carrying the same
   `node_id` cross-reference.

Boot assertion: **id-membership** reconciliation (every loader concept's `id`
resolves in the router store and its `node_id` in coral, distinct-id counts
agree — never raw count equality, which a rare 48-bit collision would misread
as drift) + every reference class resolves in both. The **hermetic
`InMemoryConceptStore`** is a test double only — never production.

- Canonical names: `schema:Person`, `yago:Person`, else the IRI.
- The 7 reference classes: Entity, Person, Organization, Place, Event,
  Artifact, Concept (plus Agent in the sample taxonomy).
- `subClassOf` edges build the `TaxonomyHierarchy` (`DependencyGraph`-backed,
  C5): `ancestors_of(dog)` → `[mammal, animal, entity]`.
- `is_whitelisted_id` / `is_whitelisted_hash` are **truncation-aware**
  (namespace + 48-bit local), never full-hash comparisons.
- The 11 `OntologyProperty` constants (label, comment, description, prefLabel,
  type, subClassOf, hasGender, hasNationality, bornIn, diedIn,
  hasWikipediaArticle) each map to an `RdfProperty` id.

## Collisions: first-wins with canonical disambiguation

Collision probability for *n* entries under one namespace ≈ `n² / (2·2⁴⁸)`
(130k YaGO classes → ≈ 3e-5; 1M lemmas → ≈ 1.8e-3). When a second canonical
truncates to a taken id:

- The id stays (a stable **bucket**); the registry keeps **both** canonicals.
- `CollisionNote::Collision { id, prior_canonical }` is surfaced (log + audit +
  the "needs disambiguation" routing signal). Consumers that need injectivity
  key on `(InterlinguaId, canonical_name)`.
- A **probe family** was explicitly rejected (F2): ids must be
  order-independent and universe-independent; the bucket + note approach mirrors
  spaCy's own `StringStore` semantics.

## Unifying storage

Because every layer speaks the same id language, one request traces through
everything:

```
"show me the sales report"
   │  tokenize + parse (arc_eager)
   ▼
lemma ids (SpacyLemma)  +  role ids (predicate/object)
   │  resolve + signals
   ▼
routing signal (interlingua frame) ──► match_interlingua filter ──► route
   │  record_parse_node_with_confidence
   ▼
ledger: interlingua_index (node_id, id, role, confidence, review_status)
   │  same loader
   ▼
knowledge: YaGO classes/entities (interlingua_concepts) ⇄ coral graph (node_id)
```

- **LLM context**: the classifier prompt is *given* the parsed ids as
  deterministic context; low-confidence parses escalate ("needs
  disambiguation"). The LLM is consulted about the residue, not the already
  decided.
- **Databases**: the ledger's `interlingua_index` and `interlingua_concepts`
  are joinable by id across sessions — "requests whose direct_object_id is X"
  is a SQL query, not a model call. The live request path populates
  `interlingua_index` (`record_parse_node_with_confidence` in `ledger/nlp.rs`
  writes the parse node **and** the index rows in one transaction, one row per
  sentence × role id, `review_status='unreviewed'`); `record_parse_node` is a
  thin wrapper over it. The correction cache lives in
  `interlingua_index` keyed on `(node_id, interlingua_id, role, entity_id)`
  (pattern-cache rows use the sentinel `node_id = 0`, `role = 'correction'`,
  `entity_id` = the entity-scoping id or 0, `review_status = 'cached'` —
  status never holds an id).
- **Knowledge graphs**: coral's `context_nodes`/`edges` are keyed by the same
  `node_id` values; taxonomy traversals (`is_a`, `ancestors_of`) cross-reference
  without translation.

## Invariants to preserve

- No raw `InterlinguaId` constructor; ids are always content-derived
  (`from_u64`/`from_i64` exist only to reconstruct a stored id at the
  serde/DB round-trip — never to manufacture one).
- `node_id` (64-bit) stored, never derived from `local_id` (48-bit).
- First-wins on every id/canonical store.
- The resolver and pipeline never write the concept store (boot-only
  registration); corrections happen in the async review worker.
- One loader, two homes; boot reconciliation is **id-membership**
  (collision-tolerant), not raw count equality.
- Collisions are surfaced, never silently resolved to a single canonical;
  `interlingua_concepts` keeps both canonicals (PK `(namespace, canonical_name)`).

## Key files

- `src/types/src/interlingua.rs` — `InterlinguaId`, `InterlinguaNamespace`,
  `local_id_of`, `ConceptMetadata`.
- `src/spacy-rs/src/interlingua.rs` — `InterlinguaResolver`, `CollisionNote`,
  `resolve_doc`.
- `src/concept/src/concept_store.rs` + `concept_store_mem.rs` —
  `ConceptStore`, `TaxonomyHierarchy`, `InMemoryConceptStore` (neutral
  `fluent-concept` home, M3 — spacy-rs keeps a read-only view import only).
- `src/spacy-rs/src/pipeline.rs` — the annotation ladder
  (`LlmRung`/`EncoderRung`/`ArcEagerRung`/`RuleRung`) that produces the lemmas
  `resolve_doc` stamps.
- `src/fluent-onnx/src/colbert.rs`, `src/fluent-onnx/src/encoder.rs` — the `fluent-onnx`
  entity-link index and trained-encoder rung that feed the same ids.
- `src/router/src/ort.rs` — the router's onnx composition root
  (`nlp_encoder_fetch`, `colbert_entity_scorer`, `disambiguation_overlays`).
- `src/router/src/stages/nlp.rs` — `NlpStage` publishes `RoutingSignal`s;
  `stages/tree/engine.rs` `match_interlingua` dispatches on them.
- `src/router/src/concept_store_sqlite.rs`, `ledger/nlp.rs` — the durable
  store + parse-node/index population; `ledger/correction_index.rs` the
  `CorrectionIndex`; `ledger/overlay.rs` the `overlay_candidates` plane.
- `src/router/src/server/entity_link.rs`, `server/review.rs` — the async
  entity-link overlay and parse-review workers.
- `src/ontology/src/yago_loader.rs`, `yago.rs` — the YaGO 4.5 loader + schema
  ids + whitelist.
- `src/ontology/src/yago_view.rs`, `plausibility.rs` — the runtime YaGO
  class-hierarchy view (CSR/memo, moved from spacy-rs in M5) + the
  triple-vs-taxonomy `score_plausibility` kernel over the shared
  `fluent-concept::PlausibilityTriple` input (spacy-rs keeps only
  `extract_triples` + the `PlausibilityFetch` seam).
- `src/coral/src/db/nodes.rs`, `ingest.rs` — the content-addressed graph.