//! The three-axis verdict a classifier node's LLM call must return, and its
//! tolerant parse (`parse_tree_verdict`) through the shared
//! `fluent_llm::parse_typed` codec + `stages::common` field coercion.

use fluent_wvr::prelude::*;

use crate::stages::common::{coerce_float, coerce_string, coerce_u8};

/// The three-axis verdict a classifier node's LLM call must return.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TreeClassifierVerdict {
    /// The child key to route to. `None`/empty → the classifier's fallback
    /// child (or a rejection when there is none).
    #[serde(default)]
    pub route: Option<String>,
    #[serde(default = "default_one")]
    pub coherence: f64,
    #[serde(default = "default_one")]
    pub safety: f64,
    #[serde(default = "default_five")]
    pub complexity: u8,
    #[serde(default)]
    pub reason: String,
}

fn default_one() -> f64 {
    1.0
}

fn default_five() -> u8 {
    5
}

/// Tolerant parse of a classifier node's LLM response into the three-axis
/// verdict: the shared `fluent_llm::parse_typed` codec runs the direct-
/// deserialize fast path, then the shared fence-strip → parse → extract
/// pipeline, then the shared field coercion (`stages::common`).
pub fn parse_tree_verdict(response: &str) -> Result<TreeClassifierVerdict, WorkError> {
    fluent_llm::parse_typed::<TreeClassifierVerdict>(
        response,
        &serde_json::Value::Null,
        |v| {
            if let Some(obj) = v.as_object_mut() {
                coerce_float(obj, "coherence", 1.0);
                coerce_float(obj, "safety", 1.0);
                coerce_u8(obj, "complexity", 5);
                coerce_string(obj, "reason", "");
            }
        },
    )
    .map_err(|e| WorkError::Execution(format!("tree classifier verdict parse error: {e}")))
}
