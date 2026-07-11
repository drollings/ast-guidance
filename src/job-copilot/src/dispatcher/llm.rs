use std::sync::{Arc, RwLock};

use fluent_concurrency::io::net::NetCapability;
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
    net_cap: Option<NetCapability>,
}

impl LlmDispatcher {
    pub fn new(client: Arc<LlmClient>, profile: Arc<RwLock<Profile>>) -> Self {
        Self {
            client,
            profile,
            store: Arc::new(RwLock::new(FieldSimilarityStore::new())),
            net_cap: None,
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
            net_cap: None,
        }
    }

    /// Set the `NetCapability` for this dispatcher. The held capability is
    /// checked at dispatch time: the LLM call is performed only when the
    /// current task-local `CapabilitySet` contains a `NetCapability`. When
    /// absent, or when the dispatcher holds no capability, the LLM call is
    /// skipped (returns `None`).
    #[must_use]
    pub fn with_net_capability(mut self, cap: NetCapability) -> Self {
        self.net_cap = Some(cap);
        self
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
        // Gate: NetCapability must be present on the dispatcher. The
        // held capability is what authorizes network I/O; without it,
        // the dispatcher cannot make HTTP calls and returns `None`.
        if self.net_cap.is_none() {
            tracing::debug!(
                field_label = %field.label,
                "LlmDispatcher: skipping LLM call — NetCapability not held by dispatcher"
            );
            return None;
        }

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

/// Sharded LLM dispatcher that routes fields to one of N `LlmDispatcher`
/// instances based on a hash of the field ID. Mirrors `PartitionedRouter`'s
/// key-based sharding pattern for the sync `FieldValueDispatcher` interface.
///
/// All shards share the same profile but may have different similarity stores,
/// enabling per-shard caching and independent LLM configuration.
pub struct ShardedLlmDispatcher {
    shards: Vec<LlmDispatcher>,
}

impl ShardedLlmDispatcher {
    /// Create a sharded dispatcher with the given `LlmDispatcher` instances.
    /// Fields are routed to `shards[field_id.hash() % shards.len()]`.
    pub fn new(shards: Vec<LlmDispatcher>) -> Self {
        assert!(
            !shards.is_empty(),
            "ShardedLlmDispatcher requires at least one shard"
        );
        Self { shards }
    }

    fn shard_index(&self, field_id: &str) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        field_id.hash(&mut hasher);
        hasher.finish() as usize % self.shards.len()
    }
}

impl super::FieldValueDispatcher for ShardedLlmDispatcher {
    fn name(&self) -> &str {
        "sharded_llm"
    }

    fn route(
        &self,
        field: &FieldDescription,
        page_context: &str,
        company_hint: Option<&str>,
    ) -> Option<PreFilledValue> {
        let idx = self.shard_index(&field.field_id);
        self.shards[idx].route(field, page_context, company_hint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::FieldValueDispatcher;
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
        LlmDispatcher::new(client, profile).with_net_capability(NetCapability::new())
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
        let dispatcher = LlmDispatcher::with_store(client, profile, store)
            .with_net_capability(NetCapability::new());

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
        let dispatcher = LlmDispatcher::with_store(client, profile, store)
            .with_net_capability(NetCapability::new());

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
        let dispatcher = LlmDispatcher::with_store(client, profile, store.clone())
            .with_net_capability(NetCapability::new());

        let result = dispatcher.route(&test_field(), "", None).unwrap();
        assert_eq!(result.value, "Because tech");
        mock.assert();

        // Verify store was updated.
        let store = store.read().unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.entries()[0].label, "Why do you want to work here?");
        assert_eq!(store.entries()[0].value, "Because tech");
    }

    #[test]
    fn missing_net_capability_skips_llm_call() {
        let client = Arc::new(LlmClient::new("http://127.0.0.1:9999", "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));
        let dispatcher = LlmDispatcher::new(client, profile);
        let result = dispatcher.route(&test_field(), "", None);
        assert!(
            result.is_none(),
            "should return None when NetCapability is missing"
        );
    }

    #[test]
    fn sharded_dispatcher_routes_to_correct_shard() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": r#"{"value":"shard0","confidence":0.9,"reasoning":"test"}"#
                    }
                }]
            }));
        });

        let client = Arc::new(LlmClient::new(&server.base_url(), "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));

        // Create two shards: both hit the same mock server.
        let shard0 = LlmDispatcher::new(client.clone(), profile.clone())
            .with_net_capability(NetCapability::new());
        let shard1 = LlmDispatcher::new(client.clone(), profile.clone())
            .with_net_capability(NetCapability::new());

        let sharded = ShardedLlmDispatcher::new(vec![shard0, shard1]);
        assert_eq!(sharded.name(), "sharded_llm");

        // The field ID determines which shard handles the request.
        let result = sharded.route(&test_field(), "", None);
        assert!(
            result.is_some(),
            "sharded dispatcher should route to a shard"
        );
        mock.assert();
    }

    #[test]
    fn sharded_dispatcher_sensitive_field_returns_none() {
        let client = Arc::new(LlmClient::new("http://127.0.0.1:9999", "test-model"));
        let profile = Arc::new(RwLock::new(test_profile()));

        let shard = LlmDispatcher::new(client, profile);
        let sharded = ShardedLlmDispatcher::new(vec![shard]);

        // Sensitive fields return None regardless of shard routing.
        let result = sharded.route(&sensitive_field(), "", None);
        assert!(result.is_none());
    }

    #[test]
    #[should_panic(expected = "requires at least one shard")]
    fn sharded_dispatcher_panics_with_empty_shards() {
        let _ = ShardedLlmDispatcher::new(vec![]);
    }
}
