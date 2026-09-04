use serde::{Deserialize, Serialize};

use crate::pipeline::QualifiedModelId;

/// Single routing-fields value type, collapsing the four `Option` threadings
/// through `normalize.rs`, `server/handler.rs`, `server/dispatch.rs`,
/// and `dispatch/backend.rs`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingContext {
    pub instance: Option<String>,
    pub snapshot: Option<String>,
    pub id_slot: Option<i32>,
    pub num_ctx: Option<u64>,
}

impl RoutingContext {
    #[must_use]
    pub fn from_query(q: &[(String, String)]) -> Self {
        let mut ctx = Self::default();
        for (k, v) in q {
            match k.as_str() {
                "instance" => ctx.instance = Some(v.clone()),
                "snapshot" => ctx.snapshot = Some(v.clone()),
                "id_slot" => ctx.id_slot = v.parse::<i32>().ok(),
                "num_ctx" => ctx.num_ctx = v.parse::<u64>().ok(),
                _ => {}
            }
        }
        ctx
    }

    #[must_use]
    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn from_body(value: &serde_json::Value) -> Self {
        Self {
            instance: value
                .get("instance")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            snapshot: value
                .get("snapshot")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            id_slot: value.get("id_slot").and_then(|v| v.as_i64()).map(|i| i as i32),
            num_ctx: value.get("num_ctx").and_then(|v| v.as_u64()),
        }
    }

    /// Merge `self` (query) with `over` (body); body wins.
    #[must_use]
    pub fn merge(self, over: Self) -> Self {
        Self {
            instance: over.instance.or(self.instance),
            snapshot: over.snapshot.or(self.snapshot),
            id_slot: over.id_slot.or(self.id_slot),
            num_ctx: over.num_ctx.or(self.num_ctx),
        }
    }

    /// Render as params object, stripping `None`.
    #[must_use]
    pub fn into_params(self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        if let Some(v) = self.instance {
            obj.insert("instance".into(), serde_json::Value::String(v));
        }
        if let Some(v) = self.snapshot {
            obj.insert("snapshot".into(), serde_json::Value::String(v));
        }
        if let Some(v) = self.id_slot {
            obj.insert("id_slot".into(), serde_json::json!(v));
        }
        if let Some(v) = self.num_ctx {
            obj.insert("num_ctx".into(), serde_json::json!(v));
        }
        serde_json::Value::Object(obj)
    }

    #[must_use]
    pub fn qualified_model_id(&self, base: &str) -> QualifiedModelId {
        match &self.instance {
            Some(q) if !q.is_empty() => QualifiedModelId::qualified(base, q.as_str()),
            _ => QualifiedModelId::bare(base),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.instance.is_none() && self.snapshot.is_none() && self.id_slot.is_none() && self.num_ctx.is_none()
    }
}

#[cfg(test)]
#[path = "../tests/routing_context.rs"]
mod tests;
