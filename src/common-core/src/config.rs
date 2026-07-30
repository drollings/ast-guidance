//! JSON config loaders: `load_json_or_default` (fallback to `T::default()`) and `load_json` (strict).

use std::path::Path;

use serde::de::DeserializeOwned;

use crate::error::IoError;

/// Load a JSON config file, falling back to `T::default()` if the file is
/// missing or cannot be read.
///
/// This is the "load-or-default" pattern: read, parse, and on any failure
/// return the type's default.  When the file **exists but deserialization
/// fails**, a warning is emitted to stderr so that silent config skew
/// (e.g. forward-looking JSON fields that don't match current Rust types)
/// does not go unnoticed.
pub fn load_json_or_default<T: DeserializeOwned + Default>(path: &Path) -> T {
    match load_json(path) {
        Ok(t) => t,
        Err(e) => {
            if path.exists() {
                eprintln!(
                    "WARNING: config file '{}' exists but failed to parse: {}. Falling back to default.",
                    path.display(),
                    e
                );
            }
            T::default()
        }
    }
}

/// Load a JSON config file strictly — errors on missing file, invalid JSON,
/// or I/O failure.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T, IoError> {
    let content = crate::io::read_to_string_err(path)?;
    serde_json::from_str(&content).map_err(|e| {
        IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("JSON parse error: {e}"),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn load_json_or_default_returns_default_on_missing() {
        let result = load_json_or_default::<TestConfig>(Path::new("/nonexistent/config.json"));
        assert_eq!(result.name, "default");
        assert_eq!(result.count, 0);
    }

    #[test]
    fn load_json_or_default_loads_valid_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, r#"{"name":"loaded","count":42}"#).unwrap();

        let result = load_json_or_default::<TestConfig>(&path);
        assert_eq!(result.name, "loaded");
        assert_eq!(result.count, 42);
    }

    #[test]
    fn load_json_or_default_returns_default_on_invalid_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, "not json").unwrap();

        let result = load_json_or_default::<TestConfig>(&path);
        assert_eq!(result.name, "default");
    }

    #[test]
    fn load_json_or_default_warns_on_malformed_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, r#"{"name": 42}"#).unwrap(); // wrong type for "name"

        let result = load_json_or_default::<TestConfig>(&path);
        assert_eq!(result.name, "default"); // falls back
        // Warning should have been printed to stderr; structural test only.
    }

    #[test]
    fn load_json_strict_errors_on_missing() {
        let result = load_json::<TestConfig>(Path::new("/nonexistent/config.json"));
        assert!(result.is_err());
    }

    #[test]
    fn load_json_strict_loads_valid_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, r#"{"name":"strict","count":7}"#).unwrap();

        let result = load_json::<TestConfig>(&path).unwrap();
        assert_eq!(result.name, "strict");
        assert_eq!(result.count, 7);
    }

    #[test]
    fn load_json_strict_errors_on_invalid_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, "not json").unwrap();

        let result = load_json::<TestConfig>(&path);
        assert!(result.is_err());
    }

    #[derive(Debug, serde::Deserialize, serde::Serialize, PartialEq)]
    struct TestConfig {
        #[serde(default = "default_name")]
        name: String,
        #[serde(default)]
        count: u32,
    }

    impl Default for TestConfig {
        fn default() -> Self {
            Self {
                name: default_name(),
                count: 0,
            }
        }
    }

    fn default_name() -> String {
        "default".to_string()
    }
}
