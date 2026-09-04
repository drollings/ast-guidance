//! Ledger annotation provenance — the producing tier of every annotation
//! (ROADMAP M4).
//!
//! Every annotation a ledger carries is stamped with **who produced it**: a
//! `ProvenanceTier` ordered by authority. A higher tier strictly overrides a
//! lower tier for the same claim on the same node, and a lower-tier annotation
//! is marked `Provisional` until confirmed or superseded — never silently
//! coexisting with a conflicting higher-tier one.
//!
//! This reuses the `AnnotationSource`/`ReviewStatus` *pattern* (a typed,
//! exhaustive enum with a lifecycle) rather than inventing a parallel scheme:
//! `AnnotationSource::tier()` maps the spacy-rs producing rung onto a
//! `ProvenanceTier`, and `ReviewStatus` maps the review lifecycle onto one
//! too.

use serde::{Deserialize, Serialize};

/// The producing authority of a ledger annotation. Ordered by strength:
/// `Deterministic < LocalModel < Frontier < HumanReview`.
///
/// - **Deterministic** — pure, model-free output (tokenizer, ArcEager/rule
///   parser, deterministic frame extraction).
/// - **LocalModel** — a local enrichment model (the ONNX LLM, an encoder rung).
/// - **Frontier** — a frontier/remote provider (a future substitution point).
/// - **HumanReview** — a human reviewer overrode or confirmed the claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceTier {
    Deterministic,
    LocalModel,
    Frontier,
    HumanReview,
}

/// The lifecycle status of one annotation claim (parallels spacy-rs
/// `ReviewStatus`): a claim is `Provisional` until confirmed, and `Superseded`
/// (never deleted) when a higher-tier claim replaces it for the same node
/// version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    /// Not yet authoritative; waiting on a higher-tier confirmation.
    Provisional,
    /// The authoritative claim for this node version + claim key.
    Confirmed,
    /// Replaced by a higher-tier claim; retained for audit, never deleted.
    Superseded,
}

/// One tiered annotation claim on one node **version** (the `content_hash` of
/// the node it describes). The version (`claim_id`) is managed by the
/// `AnnotationStore`; this is the logical claim a writer produces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationClaim {
    /// The claim's identity within a node version (e.g. a frame key or an
    /// annotation record key). Two claims with the same key on the same hash
    /// are versions of one claim; different keys are independent claims.
    pub claim_key: String,
    /// The producing authority.
    pub tier: ProvenanceTier,
    /// The lifecycle status.
    pub status: ClaimStatus,
    /// The claim's value (arbitrary JSON).
    #[serde(default)]
    pub payload: serde_json::Value,
    /// A legible producer label (e.g. "arceager", "onnx/llm", "human-review").
    pub produced_by: String,
    /// Unix seconds when the claim was produced.
    pub produced_at: u64,
}

impl AnnotationClaim {
    /// A writer minting an authoritative claim.
    #[must_use]
    pub fn confirmed(
        claim_key: impl Into<String>,
        tier: ProvenanceTier,
        payload: serde_json::Value,
        produced_by: impl Into<String>,
        produced_at: u64,
    ) -> Self {
        Self {
            claim_key: claim_key.into(),
            tier,
            status: ClaimStatus::Confirmed,
            payload,
            produced_by: produced_by.into(),
            produced_at,
        }
    }

    /// A writer minting a provisional (not-yet-authoritative) claim.
    #[must_use]
    pub fn provisional(
        claim_key: impl Into<String>,
        tier: ProvenanceTier,
        payload: serde_json::Value,
        produced_by: impl Into<String>,
        produced_at: u64,
    ) -> Self {
        Self {
            claim_key: claim_key.into(),
            tier,
            status: ClaimStatus::Provisional,
            payload,
            produced_by: produced_by.into(),
            produced_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering_is_authority_ascending() {
        assert!(ProvenanceTier::Deterministic < ProvenanceTier::LocalModel);
        assert!(ProvenanceTier::LocalModel < ProvenanceTier::Frontier);
        assert!(ProvenanceTier::Frontier < ProvenanceTier::HumanReview);
        assert!(ProvenanceTier::Deterministic < ProvenanceTier::HumanReview);
    }

    #[test]
    fn tier_and_status_wire_names_are_snake_case() {
        assert_eq!(serde_json::to_string(&ProvenanceTier::HumanReview).unwrap(), "\"human_review\"");
        assert_eq!(serde_json::to_string(&ProvenanceTier::LocalModel).unwrap(), "\"local_model\"");
        assert_eq!(serde_json::to_string(&ClaimStatus::Superseded).unwrap(), "\"superseded\"");
    }

    #[test]
    fn claim_serde_roundtrip_and_constructors() {
        let confirmed = AnnotationClaim::confirmed(
            "frame:predicate:obj",
            ProvenanceTier::LocalModel,
            serde_json::json!({ "direct_object_id": 42 }),
            "onnx/llm",
            1700000000,
        );
        assert_eq!(confirmed.status, ClaimStatus::Confirmed);
        assert_eq!(confirmed.tier, ProvenanceTier::LocalModel);

        let provisional = AnnotationClaim::provisional(
            "frame:predicate:obj",
            ProvenanceTier::Deterministic,
            serde_json::json!({}),
            "arceager",
            1700000000,
        );
        assert_eq!(provisional.status, ClaimStatus::Provisional);

        for claim in [&confirmed, &provisional] {
            let json = serde_json::to_string(claim).expect("serialize");
            let back: AnnotationClaim = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&back, claim, "round trip for {json}");
        }
    }
}