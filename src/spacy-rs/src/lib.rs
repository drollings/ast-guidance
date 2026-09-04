//! # spacy-rs
//!
//! A native, idiomatic Rust reimplementation of core spaCy, composed from
//! fluent-monorepo idioms. This crate owns the **data plane** of the NLP
//! stack: the hashed string store, the two-level lexicon, the doc/token
//! model with its dependency-tree rebuild, and the closed label
//! vocabularies. The deterministic tokenizer, the annotation validator, and
//! the LLM-JSON bridge build on top of these types.
//!
//! Design contracts preserved from spaCy (`/opt/src/nlp/spaCy`, v3.8.15):
//! - **Hashes**: strings are content-addressed with MurmurHash64A seed 1
//!   ([`hash::hash_utf8`]), stored as `ArcIntern<str>` for reverse lookup.
//! - **Two-level model**: [`lexeme::Lexeme`] (word-type) is shared by every
//!   token of the same orth; [`doc::TokenRecord`] (word-token) holds context.
//! - **Relative heads**: `head` is a signed offset from self; children are
//!   found in `[l_edge, i) ∪ (i, r_edge]`.
//! - **Closed vocabularies**: [`labels::Upos`], [`labels::DepRel`],
//!   [`labels::NerType`], [`labels::EntIoB`] mirror `symbols.pxd` ids; open
//!   vocabularies (orth, lemma, tag, model-specific dep labels) are `u64`
//!   hashes resolved through the shared [`strings::StringStore`].

#![forbid(unsafe_code)]
#![allow(clippy::pedantic, clippy::all)]

pub mod arc_eager;
pub mod arcready;
pub mod attrs;
pub mod concept_store;
pub mod concept_store_mem;
pub mod doc;
pub mod error;
pub mod frame;
pub mod hash;
pub mod interlingua;
pub mod labels;
pub mod lang;
pub mod lemmatizer;
mod lemma_blob;
pub mod lex_attrs;
pub mod taxonomy_blob;
pub mod triple;
pub mod yago_resolve;
pub mod yago_view;
pub mod lexeme;
pub mod llm;
pub mod morph;
pub mod pipeline;
pub mod cache;
pub mod genesis;
pub mod routing;
pub mod review;
pub mod retrieval;
pub mod sentencizer;
pub mod strings;
pub mod tag_map;
pub mod tokenizer;
pub mod validate;
pub mod vocab;

pub use arc_eager::{
    infer_pos, ArcEagerAction, ArcEagerAnnotator, ArcEagerMove, ArcEagerRung, ArcEagerState,
    DepLabels, DeterministicOracle, ParseConfidence,
};
pub use arcready::ArcReadyAnnotation;
pub use attrs::Attribute;
pub use concept_store::{ConceptStore, ConceptStoreError, TaxonomyHierarchy};
pub use concept_store_mem::InMemoryConceptStore;
pub use doc::{
    get_token_attr, set_children_from_heads, set_token_attr, Doc, SentStart, TokenRecord,
};
pub use error::SpacyError;
pub use frame::{
    extract_frames, mint_frame_key, AmbiguityEntry, AmbiguityKind, Frame, FrameAnalysis,
    FrameExtractor, FrameKey, Modality, Polarity, PreferredSenseIndex, Resolution, RoleType,
    TIE_MARGIN_EPSILON,
};
pub use hash::{hash_utf8, murmur64a, HASH_SEED};
pub use interlingua::{CollisionNote, InterlinguaResolver};
pub use labels::{DepLabelSet, DepRel, EntIoB, NerType, Upos};
pub use lemmatizer::{Lemmatizer, LemmatizerMode};
pub use lex_attrs::word_shape;
pub use lexeme::{Lexeme, LexemeFlags, Lexicon, LexiconConfig, OOV_RANK};
pub use llm::{AnnotationRecord, AnnotationResult, AnnotationSet, AnnotationSource, LlmRefinePrompt};
pub use morph::Morphology;
pub use pipeline::{
    AnnotateError, AnnotateStage, AnnotationRefiner, AnnotationRung, AttachStage,
    ConfidenceReason, EncoderFetchSync, EncoderResidualFetch, EncoderResidualRung,
    EncoderRung, FrameStage, LlmFetch, LlmFetchSync, LlmRefineFetchSync, LlmRefineRequest,
    LlmRefineRung, LlmRung, NlpPipeline, NlpPipelineConfig, PipelineError,
    PipelineStageFailure, PipelineState, RefineMetrics, RefineMetricsSnapshot, RefineMode,
    RefinePolicy, RefineReason, RefineSeams, ResolveStage, RuleAnnotator, RuleRung,
    SentencizeStage, StagePipeline, TaskValueReason, frame_coverage, frame_coverage_signal,
    refine_focus, refine_reason, should_refine,
};
pub use cache::{span_key, InMemorySpanCache, SpanCache};
pub use genesis::{GenesisEntry, GenesisIndex, InMemoryGenesisIndex};
pub use routing::{extract_routing_signals, InterlinguaSignal, RoutingSignal};
pub use review::{
    apply_corrections, apply_edits, review_prompt, Correction, CorrectionField, CorrectionIndex,
    LinkedEntity, ParseReview, ReviewStatus,
};
pub use retrieval::{
    cosine, cross_check, lemma_grep, CrossCheckReport, EmbeddingProvider, FuzzyHit,
    FuzzyRetrieval, InMemoryFuzzyIndex, LemmaGrepHit, RegionVerdict, RetrievalHit,
    RetrievalSource, Span,
};
pub use sentencizer::Sentencizer;
pub use strings::StringStore;
pub use tag_map::TagMap;
pub use tokenizer::{SpecialRule, SpecialToken, Tokenizer, TokenizerConfig};
pub use validate::{AnnotationError, AnnotationValidator};
pub use vocab::Vocab;
