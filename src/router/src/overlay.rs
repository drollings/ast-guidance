//! Pure overlay-plane data (ROADMAP_20260827_ORT §6.3).
//!
//! The overlay plane's deterministic, model-free helpers: the versioned
//! canonical-form table used for predicate canonicalization. Unlike the async
//! entity-link overlay, canonicalization is **pure data** — an offline-clustered
//! `(surface lemma → canonical lemma)` table loaded like the lemmatizer blob and
//! consulted with a plain lookup on the hot path (never an inference call). This
//! keeps the "ids are pure functions of content" contract intact.
//!
//! ## Deferred: the Diffusion spike (§6.4)
//!
//! The masked-diffusion paraphrase overlay (a 32-step full-canvas re-run per
//! pass, template reimplemented from `config.json`'s `diffusion` block, q8
//! paraphrases) is a **research spike, not a resident worker**. It is explicitly
//! not budgeted as a resident worker and carries no acceptance gate. Cost
//! estimate: a 32-step loop over a canvas model with a full-canvas re-run per
//! pass is roughly 32× the per-pass latency of the equivalent encoder forward —
//! an order of magnitude too expensive for the hot path — so it is deferred
//! until a cheaper formulation (e.g. single-pass paraphrase) is justified. No
//! code lands here for it.

pub mod canonical;

pub use canonical::CanonicalFormTable;