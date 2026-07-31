//! Detection of config-selected features that the binary does not implement.
//!
//! `RouterConfig`'s typed deserialization silently drops unknown keys, so a
//! config that selects an unimplemented surface would pass unnoticed. These
//! checks run on the raw JSON (before typed deserialization) so such configs
//! are loud at startup instead of silently degrading behavior.

/// A configured-but-unimplemented feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnimplementedFeature {
    /// A model group declares an `escalation` ladder, but frontier
    /// involvement modes are not wired to live dispatch
    /// (`crate::frontier::modes::execute_frontier_mode` returns a
    /// not-implemented error).
    EscalationLadder { group: String },
}

/// Scan raw config JSON for keys that select unimplemented paths.
///
/// Must be called on the raw `serde_json::Value` (before typed
/// deserialization) because `RouterConfig` drops unknown fields. Only
/// configuration-triggered keys are reported; absent keys and array-valued
/// model groups (the shipped shape) produce no warnings.
pub fn detect_unimplemented_features(raw: &serde_json::Value) -> Vec<UnimplementedFeature> {
    let mut out = Vec::new();
    let Some(groups) = raw.get("model_groups").and_then(|v| v.as_object()) else {
        return out;
    };
    for (name, group) in groups {
        let has_ladder = group
            .as_object()
            .and_then(|o| o.get("escalation"))
            .is_some_and(|v| !v.is_null());
        if has_ladder {
            out.push(UnimplementedFeature::EscalationLadder {
                group: name.clone(),
            });
        }
    }
    out
}

/// Emit one aggregate `tracing::warn!` per configured-but-unimplemented
/// surface.
pub fn log_unimplemented_features(features: &[UnimplementedFeature]) {
    for feature in features {
        match feature {
            UnimplementedFeature::EscalationLadder { group } => {
                tracing::warn!(
                    target: "router.config",
                    group = %group,
                    "escalation ladder configured but not implemented; frontier calls will fail",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_has_no_features() {
        assert!(detect_unimplemented_features(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn shipped_array_group_shape_has_no_features() {
        let raw = serde_json::json!({
            "model_groups": {
                "fast": ["fast", "small"],
                "code": ["code-model"],
            }
        });
        assert!(detect_unimplemented_features(&raw).is_empty());
    }

    #[test]
    fn escalation_ladder_in_group_is_detected() {
        let raw = serde_json::json!({
            "model_groups": {
                "fast": ["fast"],
                "code": {
                    "models": ["code-model"],
                    "escalation": ["question", "turnover"],
                },
            }
        });
        let features = detect_unimplemented_features(&raw);
        assert_eq!(
            features,
            vec![UnimplementedFeature::EscalationLadder {
                group: "code".into(),
            }]
        );
    }

    #[test]
    fn null_escalation_key_is_ignored() {
        let raw = serde_json::json!({
            "model_groups": {
                "fast": {
                    "escalation": null,
                },
            }
        });
        assert!(detect_unimplemented_features(&raw).is_empty());
    }
}
