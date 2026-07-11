pub mod llm;
pub mod local;

use std::sync::Arc;

use crate::schema::{FieldDescription, PreFilledValue};

/// Trait for dispatchers that suggest field values from a profile or LLM.
pub trait FieldValueDispatcher: Send + Sync {
    /// Human-readable tier name (e.g. "local", "llm", "tiered").
    fn name(&self) -> &str;

    /// Attempt to generate a value for a form field.
    ///
    /// Returns `None` if the field cannot be filled (sensitive, no match,
    /// LLM error, etc.).
    fn route(
        &self,
        field: &FieldDescription,
        page_context: &str,
        company_hint: Option<&str>,
    ) -> Option<PreFilledValue>;
}

/// Tiered dispatcher that chains multiple dispatchers in order.
///
/// The first tier to return `Some(PreFilledValue)` wins.
pub struct TieredDispatcher {
    tiers: Vec<Arc<dyn FieldValueDispatcher>>,
}

impl TieredDispatcher {
    pub fn new() -> Self {
        Self { tiers: Vec::new() }
    }

    #[must_use]
    pub fn with(mut self, tier: Arc<dyn FieldValueDispatcher>) -> Self {
        self.tiers.push(tier);
        self
    }
}

impl Default for TieredDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldValueDispatcher for TieredDispatcher {
    fn name(&self) -> &str {
        "tiered"
    }

    fn route(
        &self,
        field: &FieldDescription,
        page_context: &str,
        company_hint: Option<&str>,
    ) -> Option<PreFilledValue> {
        for tier in &self.tiers {
            if let Some(v) = tier.route(field, page_context, company_hint) {
                return Some(v);
            }
        }
        None
    }
}

impl FieldValueDispatcher for local::LocalDispatcher {
    fn name(&self) -> &str {
        "local"
    }

    fn route(
        &self,
        field: &FieldDescription,
        _page_context: &str,
        _company_hint: Option<&str>,
    ) -> Option<PreFilledValue> {
        self.route(field)
    }
}

impl FieldValueDispatcher for llm::LlmDispatcher {
    fn name(&self) -> &str {
        "llm"
    }

    fn route(
        &self,
        field: &FieldDescription,
        page_context: &str,
        company_hint: Option<&str>,
    ) -> Option<PreFilledValue> {
        self.route(field, page_context, company_hint)
    }
}

pub use llm::LlmDispatcher;
pub use llm::ShardedLlmDispatcher;
pub use local::LocalDispatcher;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiered_returns_first_match() {
        use crate::profile::Profile;
        use std::sync::{Arc, RwLock};

        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared));

        let tiered = TieredDispatcher::new().with(local);

        let field = FieldDescription {
            field_id: "firstName".into(),
            label: "First Name".into(),
            input_type: "text".into(),
            selector: "#fn".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };

        let result = tiered.route(&field, "", None).unwrap();
        assert_eq!(result.value, "Ada");
    }

    #[test]
    fn tiered_returns_none_when_no_tier_matches() {
        let tiered = TieredDispatcher::new();
        let field = FieldDescription {
            field_id: "x".into(),
            label: "X".into(),
            input_type: "text".into(),
            selector: "#x".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        assert!(tiered.route(&field, "", None).is_none());
    }

    #[test]
    fn tiered_name_is_tiered() {
        let tiered = TieredDispatcher::new();
        assert_eq!(tiered.name(), "tiered");
    }

    #[test]
    fn tiered_skips_sensitive_through_all_tiers() {
        use crate::profile::Profile;
        use std::sync::{Arc, RwLock};

        let profile = Profile::default();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let client = Arc::new(guidance_llm::client::LlmClient::new(
            "http://127.0.0.1:9999",
            "test",
        ));
        let llm = Arc::new(LlmDispatcher::new(client, shared));

        let tiered = TieredDispatcher::new().with(local).with(llm);

        let field = FieldDescription {
            field_id: "password".into(),
            label: "Password".into(),
            input_type: "password".into(),
            selector: "#pass".into(),
            context_text: String::new(),
            required: true,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };

        assert!(tiered.route(&field, "", None).is_none());
    }
}
