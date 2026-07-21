use serde::{Deserialize, Serialize};

/// Typed metadata value for `WorkContext`.
///
/// Replaces the old `Vec<(String, String)>` with a type-safe, structured
/// representation. Supports string, integer, float, boolean, and null values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Number(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<i64> for MetadataValue {
    fn from(n: i64) -> Self {
        MetadataValue::Number(n)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Bool(b)
    }
}

/// Legacy type alias for backward compatibility. Prefer `MetadataValue`.
pub type MetadataEntry = (String, String);
