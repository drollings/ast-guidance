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
#[deprecated(note = "Use `MetadataValue` instead. This alias has no known consumers.")]
pub type MetadataEntry = (String, String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_value_from_strings_and_numbers() {
        assert_eq!(MetadataValue::from("x"), MetadataValue::String("x".into()));
        assert_eq!(MetadataValue::from("x".to_string()), MetadataValue::String("x".into()));
        assert_eq!(MetadataValue::from(7i64), MetadataValue::Number(7));
        assert_eq!(MetadataValue::from(0.5f64), MetadataValue::Float(0.5));
        assert_eq!(MetadataValue::from(true), MetadataValue::Bool(true));
    }

    #[test]
    fn metadata_value_serde_round_trip() {
        for value in [
            MetadataValue::String("s".into()),
            MetadataValue::Number(42),
            MetadataValue::Float(1.5),
            MetadataValue::Bool(false),
            MetadataValue::Null,
        ] {
            let json = serde_json::to_string(&value).expect("serialize");
            let back: MetadataValue = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, value, "round trip for {json}");
        }
    }

    #[test]
    fn metadata_value_untagged_shapes() {
        // Untagged enum: strings serialize as JSON strings, numbers as numbers.
        assert_eq!(
            serde_json::to_string(&MetadataValue::String("hi".into())).unwrap(),
            "\"hi\""
        );
        assert_eq!(
            serde_json::to_string(&MetadataValue::Number(3)).unwrap(),
            "3"
        );
        assert_eq!(
            serde_json::from_str::<MetadataValue>("\"text\"").unwrap(),
            MetadataValue::String("text".into())
        );
        assert_eq!(
            serde_json::from_str::<MetadataValue>("null").unwrap(),
            MetadataValue::Null
        );
    }

    #[test]
    fn metadata_value_works_in_context_metadata_map() {
        use std::collections::HashMap;
        let mut map: HashMap<String, MetadataValue> = HashMap::new();
        map.insert("classifier_system_prompt".into(), "system".into());
        map.insert("max_retries".into(), 3.into());
        assert_eq!(map["classifier_system_prompt"], MetadataValue::String("system".into()));
        assert_eq!(map["max_retries"], MetadataValue::Number(3));
    }
}
