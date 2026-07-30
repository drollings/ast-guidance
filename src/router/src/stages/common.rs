//! Shared helpers for pipeline stages.

use fluent_wvr::prelude::*;

pub fn extract_user_message(ctx: &WorkContext) -> Result<String, WorkError> {
    let request_str = ctx
        .metadata
        .get("request")
        .and_then(|v| match v {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .ok_or_else(|| WorkError::Execution("missing request".into()))?;

    let parsed: serde_json::Value =
        serde_json::from_str(request_str).map_err(|e| WorkError::Execution(e.to_string()))?;

    parsed["messages"]
        .as_array()
        .ok_or_else(|| WorkError::Execution("no messages array in request".into()))?
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .next_back()
        .map(std::string::ToString::to_string)
        .ok_or_else(|| WorkError::Execution("no user message found".into()))
}

pub fn get_metadata_string(ctx: &WorkContext, key: &str) -> Option<String> {
    ctx.metadata.get(key).and_then(|v| match v {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
}
