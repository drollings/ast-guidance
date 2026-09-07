//! Router configuration types - deserialized from JSON via `common_core::config`.
//!
//!  This file is a thin facade (M3) - the authoritative types live in
//!  `config/root.rs`. It re-exports submodules and root types so
//!  `crate::config::RouterConfig` remains stable.

pub mod addr;
pub mod builder;
pub mod classification;
pub mod escalation;
pub mod filters;
pub mod ledger_group;
pub mod refine_policy;
pub mod root;
pub mod rounds;
pub mod routing;
pub mod runtime;
pub mod serving;

pub use self::addr::{hosts_equivalent, parse_bind_addr, validate_no_self_routing};
pub use self::builder::{NlpConfig, NlpOrdering, PipelineParams, TargetMatchMode};
pub use self::refine_policy::{RouterRefineMode, RouterRefinePolicy};
pub use self::classification::{ClassificationChild, ClassificationNode, ClassificationTree};
pub use self::escalation::{EscalationLadderConfig, FrontierConfig, GroupMember, ModelGroup};
pub use self::rounds::{BoundedRounds, EscalationConfidence, SeverityThreshold};
pub use self::filters::{
    CommandConfig, ConfidenceGate, FilterAction, FilterOutcome, FilterScope, MockConfig,
    PatternEntry, RejectPatterns,
};
pub use self::routing::{RoleEntry, RouteRef, RoutingConfig};
pub use common_core::constants::default_true;
pub use self::root::{
    AuditLogConfig, ArcReadyConfig, ChartsConfig, ClassifierFailurePolicy, ClassifierOutput,
    DefaultModelParams, EvictionPolicy, InstanceProfile, LedgerConfig, ModelEntry,
    OrchestratorSection, OverlayConfig, PostProcessConfig, ReviewConfig, RigorConfig,
    RouterConfig, ServerConfig, SessionConfig, SidecarConfig, WorkflowExtractionMode,
    DECLARATION_PARAM_KEYS, DEFAULT_LEDGER_MAX_SUMMARY_TOKENS, detect_device_vram_total,
    resolve_inference_point, role_head_key, strip_declaration_params,
};
#[allow(unused_imports)]
pub(crate) use self::root::{merge_sampling_params, split_model_key};
