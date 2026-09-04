use internment::ArcIntern;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Whether a string is a valid instance/snapshot name (`[A-Za-z0-9._-]+`).
pub fn is_valid_instance_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstanceIdError {
    MissingSeparator,
    ExtraSeparator,
    EmptyModel,
    InvalidName(String),
    EmptyName,
}

impl fmt::Display for InstanceIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSeparator => write!(f, "missing ':' separator"),
            Self::ExtraSeparator => write!(f, "extra ':' separator in instance name"),
            Self::EmptyModel => write!(f, "model id is empty"),
            Self::InvalidName(n) => write!(f, "invalid instance name '{n}' (allowed: [A-Za-z0-9._-])"),
            Self::EmptyName => write!(f, "instance name is empty"),
        }
    }
}

impl std::error::Error for InstanceIdError {}

/// Typed opaque pair `(model_key, instance_name)` with grammar validation.
/// Backed by `ArcIntern<str>` per VISION §Shared Content Nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstanceId {
    model: ArcIntern<str>,
    name: ArcIntern<str>,
}

impl InstanceId {
    pub fn new(
        model: impl Into<ArcIntern<str>>,
        name: impl Into<ArcIntern<str>>,
    ) -> Result<Self, InstanceIdError> {
        let model: ArcIntern<str> = model.into();
        let name: ArcIntern<str> = name.into();
        if model.is_empty() {
            return Err(InstanceIdError::EmptyModel);
        }
        if name.is_empty() {
            return Err(InstanceIdError::EmptyName);
        }
        if !is_valid_instance_name(&name) {
            return Err(InstanceIdError::InvalidName(name.to_string()));
        }
        if name.contains(':') {
            return Err(InstanceIdError::ExtraSeparator);
        }
        if model.contains(':') {
            return Err(InstanceIdError::ExtraSeparator);
        }
        Ok(Self { model, name })
    }

    pub fn parse(s: &str) -> Result<Self, InstanceIdError> {
        let (model, name) = s.split_once(':').ok_or(InstanceIdError::MissingSeparator)?;
        if model.is_empty() {
            return Err(InstanceIdError::EmptyModel);
        }
        if name.is_empty() {
            return Err(InstanceIdError::EmptyName);
        }
        if name.contains(':') {
            return Err(InstanceIdError::ExtraSeparator);
        }
        if !is_valid_instance_name(name) {
            return Err(InstanceIdError::InvalidName(name.to_string()));
        }
        Ok(Self {
            model: ArcIntern::from(model),
            name: ArcIntern::from(name),
        })
    }

    pub fn model(&self) -> &ArcIntern<str> {
        &self.model
    }

    pub fn name(&self) -> &ArcIntern<str> {
        &self.name
    }

    pub fn as_pair(&self) -> (&str, &str) {
        (&self.model, &self.name)
    }

    /// Alias set for one public instance id: the bare model id and `latest`
    /// form on the default instance, the group form, and the exact id.
    pub fn aliases(&self, group: &str, is_default: bool) -> Vec<String> {
        let model_key: &str = self.model.as_ref();
        let instance_id = self.to_string();
        let mut aliases = Vec::new();
        if is_default {
            aliases.push(model_key.to_string());
            aliases.push(format!("{model_key}:latest"));
        }
        aliases.push(format!("{model_key}:{group}"));
        aliases.push(instance_id);
        aliases.sort();
        aliases.dedup();
        aliases
    }
}

impl fmt::Display for InstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.model, self.name)
    }
}

impl FromStr for InstanceId {
    type Err = InstanceIdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for InstanceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for InstanceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_parse_and_display() {
        let id = InstanceId::parse("model:instance").unwrap();
        assert_eq!(id.model(), &ArcIntern::from("model"));
        assert_eq!(id.name(), &ArcIntern::from("instance"));
        assert_eq!(id.to_string(), "model:instance");
        assert_eq!(id.as_pair(), ("model", "instance"));
    }

    #[test]
    fn parse_rejects_missing_separator() {
        assert_eq!(
            InstanceId::parse("baremodel").unwrap_err(),
            InstanceIdError::MissingSeparator
        );
    }

    #[test]
    fn parse_rejects_extra_separator() {
        assert_eq!(
            InstanceId::parse("m:a:b").unwrap_err(),
            InstanceIdError::ExtraSeparator
        );
    }

    #[test]
    fn parse_rejects_empty_name() {
        assert_eq!(
            InstanceId::parse("model:").unwrap_err(),
            InstanceIdError::EmptyName
        );
    }

    #[test]
    fn parse_rejects_empty_model() {
        assert_eq!(
            InstanceId::parse(":name").unwrap_err(),
            InstanceIdError::EmptyModel
        );
    }

    #[test]
    fn parse_rejects_invalid_chars() {
        for bad in ["m:bad/name", "m:bad name", "m:bad:"] {
            assert!(
                matches!(
                    InstanceId::parse(bad).unwrap_err(),
                    InstanceIdError::ExtraSeparator | InstanceIdError::InvalidName(_)
                ),
                "bad input {bad}"
            );
        }
        assert_eq!(
            InstanceId::parse("m:bad/name").unwrap_err(),
            InstanceIdError::InvalidName("bad/name".into())
        );
        assert_eq!(
            InstanceId::parse("m:bad name").unwrap_err(),
            InstanceIdError::InvalidName("bad name".into())
        );
    }

    #[test]
    fn accepts_valid_chars() {
        for name in ["m:n", "model:my.instance_name-1", "swarm:ledger", "my-model:my_instance.name-1"] {
            assert!(InstanceId::parse(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn new_validates() {
        assert!(InstanceId::new("model", "valid_name").is_ok());
        assert_eq!(
            InstanceId::new("", "name").unwrap_err(),
            InstanceIdError::EmptyModel
        );
        assert_eq!(
            InstanceId::new("model", "").unwrap_err(),
            InstanceIdError::EmptyName
        );
        assert!(matches!(
            InstanceId::new("model", "bad/name").unwrap_err(),
            InstanceIdError::InvalidName(_)
        ));
    }

    #[test]
    fn display_round_trip() {
        let original = "my-model:my_instance.name-1";
        let id = InstanceId::parse(original).unwrap();
        assert_eq!(id.to_string(), original);
        assert_eq!(InstanceId::parse(&id.to_string()).unwrap(), id);
    }

    #[test]
    fn from_str_round_trip() {
        let id: InstanceId = "model:instance".parse().unwrap();
        assert_eq!(id.to_string(), "model:instance");
    }

    #[test]
    fn serde_round_trip() {
        let id = InstanceId::parse("model:instance").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"model:instance\"");
        let back: InstanceId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn aliases_default_includes_bare_and_latest() {
        let id = InstanceId::parse("model:inst").unwrap();
        let aliases = id.aliases("group", true);
        assert!(aliases.contains(&"model".to_string()));
        assert!(aliases.contains(&"model:latest".to_string()));
        assert!(aliases.contains(&"model:group".to_string()));
        assert!(aliases.contains(&"model:inst".to_string()));
    }

    #[test]
    fn aliases_non_default_excludes_bare() {
        let id = InstanceId::parse("model:inst").unwrap();
        let aliases = id.aliases("group", false);
        assert!(!aliases.contains(&"model".to_string()));
        assert!(!aliases.contains(&"model:latest".to_string()));
        assert!(aliases.contains(&"model:group".to_string()));
        assert!(aliases.contains(&"model:inst".to_string()));
    }

    #[test]
    fn is_valid_instance_name_cases() {
        assert!(is_valid_instance_name("abc-123._"));
        assert!(!is_valid_instance_name(""));
        assert!(!is_valid_instance_name("bad/name"));
        assert!(!is_valid_instance_name("bad name"));
        assert!(!is_valid_instance_name("bad:name"));
    }
}
