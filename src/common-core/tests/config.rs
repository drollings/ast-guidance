use common_core::config::*;
use std::path::Path;


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
