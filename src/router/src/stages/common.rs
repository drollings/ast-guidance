//! Shared helpers for pipeline stages.

use serde_json::{Map, Value};

use crate::pipeline::RoutingTarget;
use crate::pipeline_types::StageDecision;

/// Typed handoff key for the `RoutingTarget` channel.
pub const ROUTING_TARGET_TYPED_KEY: &str = "routing_target.typed";

/// Publish a `RoutingTarget` to the typed `WorkContext` store (zero-copy).
/// This is the single call site that hands a `RoutingTarget` to the
/// orchestrator; producers return the target by value and the orchestrator
/// publishes it here.
#[allow(clippy::needless_pass_by_value)]
pub fn publish_routing_target(
    ctx: &mut fluent_wvr::WorkContext,
    _decision: &mut StageDecision,
    rt: RoutingTarget,
) {
    // Typed channel (the only channel).
    ctx.set(ROUTING_TARGET_TYPED_KEY, rt);
}

use fluent_wvr::prelude::*;

use crate::types::{RouterMessageContent, RouterRequest};

/// Ensure a floating-point field exists on a classifier/tree JSON object,
/// coercing a string-valued number back to numeric. The shared "surviving
/// normalization" used by both the flat `ClassifierStage` sanitizer and the
/// classification-tree parser.
pub(crate) fn coerce_float(obj: &mut Map<String, Value>, key: &str, default: f64) {
    match obj.get(key) {
        None => {
            if let Some(n) = serde_json::Number::from_f64(default) {
                obj.insert(key.into(), Value::Number(n));
            }
        }
        Some(Value::String(s)) => {
            if let Ok(n) = s.parse::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    obj[key] = Value::Number(num);
                }
            }
        }
        _ => {}
    }
}

/// Ensure an unsigned-integer field exists on a classifier/tree JSON object,
/// coercing from float or string. The shared "surviving normalization" for the
/// `complexity` axis.
pub(crate) fn coerce_u8(obj: &mut Map<String, Value>, key: &str, default: u8) {
    match obj.get(key) {
        None => {
            obj.insert(key.into(), Value::Number(serde_json::Number::from(default)));
        }
        Some(Value::Number(n)) => {
            let as_u8 = n
                .as_u64()
                .map(|i| i.min(u64::from(u8::MAX)) as u8)
                .or_else(|| n.as_f64().map(|f| f.round().min(f64::from(u8::MAX)) as u8));
            if let Some(v) = as_u8 {
                obj[key] = Value::Number(serde_json::Number::from(v));
            }
        }
        Some(Value::String(s)) => {
            let as_u8 = s
                .parse::<u8>()
                .ok()
                .or_else(|| s.parse::<f64>().ok().map(|f| f.round() as u8));
            if let Some(v) = as_u8 {
                obj[key] = Value::Number(serde_json::Number::from(v));
            }
        }
        _ => {}
    }
}

/// Ensure a string field exists on a classifier/tree JSON object with a
/// default when absent.
pub(crate) fn coerce_string(obj: &mut Map<String, Value>, key: &str, default: &str) {
    if !obj.contains_key(key) {
        obj.insert(key.into(), Value::String(default.into()));
    }
}

/// Extract the last user message from the request carried in
/// `ctx.structured["request"]`.
///
/// The request is the structured canonical `RouterRequest` (a typed
/// `serde_json::Value` in the structured channel, not a JSON string), so
/// content may be either a plain string (`RouterMessageContent::Text`) or an
/// array of content parts (`RouterMessageContent::Parts`, the OpenAI
/// multi-part form used by clients like Brave Leo). Text rendering for both
/// forms lives in the single canonical helper
/// `RouterMessageContent::to_string_lossy` — this function only picks the
/// message, it never re-implements content rendering.
///
/// Selection semantics: the last `role == "user"` message whose content
/// renders to text. A `Text` message is returned verbatim (an empty string is
/// a valid — if degenerate — user message). A `Parts` message with no
/// extractable text (e.g. image-only) is skipped so an earlier text message
/// still resolves, matching the historical skip of non-string content.
pub fn extract_user_message(ctx: &WorkContext) -> Result<String, WorkError> {
    let request: RouterRequest = ctx
        .structured("request")
        .map_err(|e| WorkError::Execution(format!("missing request: {e}")))?;

    request
        .messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.role != "user" {
                return None;
            }
            match &m.content {
                RouterMessageContent::Text(s) => Some(s.clone()),
                RouterMessageContent::Parts(_) => {
                    let text = m.content.to_string_lossy();
                    (!text.trim().is_empty()).then_some(text)
                }
            }
        })
        .ok_or_else(|| WorkError::Execution("no user message found".into()))
}

pub fn get_metadata_string(ctx: &WorkContext, key: &str) -> Option<String> {
    ctx.metadata.get(key).and_then(|v| match v {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
}
#[cfg(test)]
#[path = "../../tests/stages_common.rs"]
mod tests;
