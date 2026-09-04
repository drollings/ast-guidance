pub mod codeword_anonymize;
pub mod decompose_hypothetical;
pub mod decompose_subtasks;
pub mod none;
pub mod pii_anonymize;
pub mod sanitize;
pub mod secret_mask;

#[cfg(test)]
#[path = "../../tests/transforms_tests.rs"]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{RouterMessageContent, RouterRequest};

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum TransformError {
    #[error("transform failed: {0}")]
    Failed(String),
    #[error("transform not applicable: {0}")]
    NotApplicable(String),
}

pub trait TransformStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError>;
}

pub type TransformStrategyRef = Arc<dyn TransformStrategy>;

/// Shared clone-messages/iterate/match boilerplate for transforms that rewrite
/// each `Text` message in place (the six-copy clone/iterate/match
/// skeleton is consolidated here). `Parts` messages are left untouched.
///
/// Each transform becomes a thin closure: only the per-message rewrite
/// (and any side-channel it wants, e.g. a captured anonymize map) lives in the
/// caller.
pub fn rewrite_text_messages(
    request: &RouterRequest,
    mut rewrite: impl FnMut(&str) -> Result<String, TransformError>,
) -> Result<RouterRequest, TransformError> {
    let mut transformed = request.clone();
    for message in &mut transformed.messages {
        let text = match &message.content {
            RouterMessageContent::Text(s) => s.clone(),
            RouterMessageContent::Parts(_) => continue,
        };
        message.content = RouterMessageContent::Text(rewrite(&text)?);
    }
    Ok(transformed)
}

/// Insert a `HashMap<String,String>` into `request.metadata[key]` as a JSON
/// object of string values, **only when non-empty** (no spurious empty object).
///
/// DRY helper (C2): the pii and codeword anonymize transforms share the
/// identical 6-line conversion/guarded-insertion skeleton. Drains `map`.
pub fn insert_string_map<S>(
    request: &mut RouterRequest,
    key: &str,
    map: HashMap<String, String, S>,
) where
    S: std::hash::BuildHasher,
{
    if map.is_empty() {
        return;
    }
    let obj: serde_json::Map<String, serde_json::Value> = map
        .into_iter()
        .map(|(k, v)| (k, serde_json::Value::String(v)))
        .collect();
    request.metadata.insert(key.into(), serde_json::Value::Object(obj));
}

#[cfg(test)]
mod helper_tests {
    use super::*;
    use crate::types::{RouterMessage, RouterMessageContent};

    fn request() -> RouterRequest {
        RouterRequest {
            model: "test-model".into(),
            messages: vec![RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text("hello".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            instance: None,
            snapshot: None,
            id_slot: None,
            metadata: Default::default(),
        }
    }

    fn map(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn empty_map_inserts_nothing() {
        let mut req = request();
        insert_string_map(&mut req, "k", HashMap::new());
        assert!(req.metadata.get("k").is_none());
    }

    #[test]
    fn singleton_inserts_object() {
        let mut req = request();
        insert_string_map(&mut req, "map", map(&[("a", "1")]));
        let obj = req.metadata.get("map").unwrap().as_object().unwrap();
        assert_eq!(obj.len(), 1);
        assert_eq!(obj["a"], serde_json::Value::String("1".into()));
    }

    #[test]
    fn multi_entry_and_unicode() {
        let mut req = request();
        insert_string_map(
            &mut req,
            "map",
            map(&[("café", "北京"), ("email", "user@example.com")]),
        );
        let obj = req.metadata.get("map").unwrap().as_object().unwrap();
        assert_eq!(obj.len(), 2);
        assert_eq!(obj["café"], serde_json::Value::String("北京".into()));
        assert_eq!(obj["email"], serde_json::Value::String("user@example.com".into()));
    }

    #[test]
    fn preserves_existing_metadata_keys() {
        let mut req = request();
        req.metadata.insert("pre".into(), serde_json::json!({"x": 1}));
        insert_string_map(&mut req, "map", map(&[("a", "1")]));
        assert!(req.metadata.contains_key("pre"));
        assert!(req.metadata.contains_key("map"));
    }
}
