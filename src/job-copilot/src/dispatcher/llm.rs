use std::sync::{Arc, RwLock};

use guidance_llm::client::{ChatMessage, LlmClient};

use crate::profile::Profile;
use crate::prompt;
use crate::sanitize;
use crate::schema::{FieldDescription, PreFilledValue, ValueSource};
use crate::similarity::FieldSimilarityStore;

/// Deserialized response from the LLM matching the strict JSON schema.
#[derive(serde::Deserialize)]
struct LlmResponse {
    value: String,
    confidence: f32,
    reasoning: String,
}

/// LLM-based dispatcher (Tier 1). Generates values for fields the local
/// dispatcher cannot match by prompting a local OpenAI-compatible LLM.
///
/// Includes a similarity store for caching previously-resolved field values.
/// Before calling the LLM, the store is checked for exact/alias matches.
/// After a successful LLM fill, the result is recorded for future forms.
pub struct LlmDispatcher {
    client: Arc<LlmClient>,
    profile: Arc<RwLock<Profile>>,
    store: Arc<RwLock<FieldSimilarityStore>>,
}

impl LlmDispatcher {
    pub fn new(client: Arc<LlmClient>, profile: Arc<RwLock<Profile>>) -> Self {
        Self {
            client,
            profile,
            store: Arc::new(RwLock::new(FieldSimilarityStore::new())),
        }
    }

    /// Create an LlmDispatcher with a pre-loaded similarity store.
    pub fn with_store(
        client: Arc<LlmClient>,
        profile: Arc<RwLock<Profile>>,
        store: Arc<RwLock<FieldSimilarityStore>>,
    ) -> Self {
        Self {
            client,
            profile,
            store,
        }
    }

    /// Attempt to generate a value for a form field.
    ///
    /// First checks the similarity store for exact/alias matches. If found
    /// with high confidence, short-circuits without calling the LLM.
    /// Otherwise prompts the LLM and records the result for future forms.
    pub fn route(
        &self,
        field: &FieldDescription,
        page_context: &str,
        company_hint: Option<&str>,
    ) -> Option<PreFilledValue> {
        if sanitize::is_sensitive_field(field) {
            return None;
        }

        // Check similarity store first.
        if let Ok(store) = self.store.read() {
            let matches = store.find_by_label(&field.label, 3);
            if let Some(entry) = matches.first() {
                // Exact or alias match found — return with high confidence.
                let confidence = if entry.label.to_lowercase() == field.label.to_lowercase() {
                    1.0
                } else {
                    0.95
                };
                return Some(PreFilledValue {
                    field_id: field.field_id.clone(),
                    value: entry.value.clone(),
                    confidence,
                    source: ValueSource::LocalInference,
                    reasoning: format!("matched from stored field '{}'", entry.label),
                });
            }
        }

        // No store match — call LLM.
        let profile = self.profile.read().ok()?;
        let prompt = prompt::build_field_prompt(field, &profile, company_hint, page_context);

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: prompt.system,
            },
            ChatMessage {
                role: "user".into(),
                content: prompt.user,
            },
        ];

        let raw = self.client.chat_complete(&messages).ok()?;

        let resp: LlmResponse = serde_json::from_str(&raw).ok()?;

        let confidence = resp.confidence.clamp(0.0, 1.0);

        // Record in store for future forms.
        if let Ok(mut store) = self.store.write() {
            store.record(field.label.clone(), resp.value.clone(), None, "llm".into());
        }

        Some(PreFilledValue {
            field_id: field.field_id.clone(),
            value: resp.value,
            confidence,
            source: ValueSource::LlmGenerated,
            reasoning: resp.reasoning,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Profile;
    use crate::schema::FieldDescription;
    use std::sync::{Arc, RwLock};

    fn test_profile() -> Profile {
        let mut p = Profile::default();
        p.personal.first_name = "Ada".into();
        p.personal.last_name = "Lovelace".into();
        p.personal.email = "ada@example.com".into();
        p
    }

    fn test_field() -> FieldDescription {
        FieldDescription {
            field_id: "why_us".into(),
            label: "Why do you want to work here?".into(),
            input_type: "textarea".into(),
            selector: "#why".into(),
            context_text: "Tell us why you're interested".into(),
            required: true,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        }
    }

    fn sensitive_field() -> FieldDescription {
        FieldDescription {
            field_id: "password".into(),
            label: "Password".into(),
            input_type: "password".into(),
            selector: "#pass".into(),
            context_text: String::new(),
            required: true,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        }
    }

    fn make_dispatcher_with_mock(server_url: &str) -> LlmDispatcher {
        let client = Arc::new(LlmClient::new(server_url, "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));
        LlmDispatcher::new(client, profile)
    }

    #[test]
    fn sensitive_field_returns_none_without_calling_llm() {
        let dispatcher = make_dispatcher_with_mock("http://127.0.0.1:9999");
        let result = dispatcher.route(&sensitive_field(), "", None);
        assert!(result.is_none());
    }

    #[test]
    fn valid_json_response_returns_prefilled_value() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200)
                .json_body(serde_json::json!({
                    "choices": [{
                        "message": {
                            "content": r#"{"value":"I love building things","confidence":0.9,"reasoning":"Matches experience"}"#
                        }
                    }]
                }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher
            .route(&test_field(), "We love innovation", Some("Acme"))
            .unwrap();
        assert_eq!(result.value, "I love building things");
        assert_eq!(result.source, ValueSource::LlmGenerated);
        assert!((result.confidence - 0.9).abs() < 0.01);
        assert_eq!(result.field_id, "why_us");
        mock.assert();
    }

    #[test]
    fn malformed_json_returns_none() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "not json at all"
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None);
        assert!(result.is_none());
        mock.assert();
    }

    #[test]
    fn json_missing_required_field_returns_none() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": r#"{"value":"hello"}"#
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None);
        assert!(result.is_none());
        mock.assert();
    }

    #[test]
    fn refusal_text_returns_none() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "I cannot help with that"
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None);
        assert!(result.is_none());
        mock.assert();
    }

    #[test]
    fn confidence_is_clamped() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": r#"{"value":"ok","confidence":2.5,"reasoning":"test"}"#
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None).unwrap();
        assert_eq!(result.confidence, 1.0);
        mock.assert();
    }

    #[test]
    fn negative_confidence_is_clamped() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": r#"{"value":"ok","confidence":-0.5,"reasoning":"low"}"#
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None).unwrap();
        assert_eq!(result.confidence, 0.0);
        mock.assert();
    }

    #[test]
    fn llm_server_error_returns_none() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(500).body("Internal Server Error");
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None);
        assert!(result.is_none());
        mock.assert();
    }

    #[test]
    fn empty_llm_response_returns_none() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": ""
                    }
                }]
            }));
        });
        let dispatcher = make_dispatcher_with_mock(&server.base_url());
        let result = dispatcher.route(&test_field(), "", None);
        assert!(result.is_none());
        mock.assert();
    }

    #[test]
    fn store_exact_match_short_circuits_llm() {
        let store = Arc::new(RwLock::new({
            let mut s = crate::similarity::FieldSimilarityStore::new();
            s.record(
                "Why do you want to work here?".into(),
                "Because I love it".into(),
                None,
                "profile".into(),
            );
            s
        }));
        let client = Arc::new(LlmClient::new("http://127.0.0.1:9999", "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));
        let dispatcher = LlmDispatcher::with_store(client, profile, store);

        // Should return from store without calling LLM (no mock set up).
        let result = dispatcher.route(&test_field(), "", None).unwrap();
        assert_eq!(result.value, "Because I love it");
        assert_eq!(result.source, ValueSource::LocalInference);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn store_alias_match_short_circuits_llm() {
        let store = Arc::new(RwLock::new({
            let mut s = crate::similarity::FieldSimilarityStore::new();
            s.record("Phone".into(), "555-1234".into(), None, "profile".into());
            s
        }));
        let client = Arc::new(LlmClient::new("http://127.0.0.1:9999", "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));
        let dispatcher = LlmDispatcher::with_store(client, profile, store);

        let field = FieldDescription {
            field_id: "phone".into(),
            label: "Telephone".into(),
            input_type: "tel".into(),
            selector: "#phone".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: Some("tel".into()),
            options: vec![],
        };
        let result = dispatcher.route(&field, "", None).unwrap();
        assert_eq!(result.value, "555-1234");
        assert_eq!(result.source, ValueSource::LocalInference);
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn llm_result_recorded_in_store() {
        let store = Arc::new(RwLock::new(crate::similarity::FieldSimilarityStore::new()));
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": r#"{"value":"Because tech","confidence":0.85,"reasoning":"match"}"#
                    }
                }]
            }));
        });
        let client = Arc::new(LlmClient::new(&server.base_url(), "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));
        let dispatcher = LlmDispatcher::with_store(client, profile, store.clone());

        let result = dispatcher.route(&test_field(), "", None).unwrap();
        assert_eq!(result.value, "Because tech");
        mock.assert();

        // Verify store was updated.
        let store = store.read().unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.entries()[0].label, "Why do you want to work here?");
        assert_eq!(store.entries()[0].value, "Because tech");
    }
}
