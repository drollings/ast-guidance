use std::sync::Arc;

use bon::Builder;
use common_core::hash::blake3_hex;
use fluent_wvr::{
    ArcIntern, Capability, Describable, FieldAccess, WorkContext, WorkError, WorkOutput, WorkUnit,
};

use crate::dispatcher::FieldValueDispatcher;
use crate::memory::FormFillStore;
use crate::profile::Profile;
use crate::sanitize;
use crate::schema::{
    AnalyzeFormResponse, PageAnalyzeFormParams, SkippedField, SkippedReason, ValueSource,
};
use crate::server::audit::AuditLog;
use std::sync::RwLock;

/// Typed capability for passing `PageAnalyzeFormParams` through `WorkContext.caps`.
pub struct AnalyzeFormParamsCap(pub PageAnalyzeFormParams);
impl Capability for AnalyzeFormParamsCap {
    fn name(&self) -> &'static str {
        "AnalyzeFormParams"
    }
}

/// WorkUnit component that performs form analysis.
///
/// Wraps the dispatcher + profile + optional audit log into a composable
/// `Component` that can be wrapped with middleware (retry, timing).
#[derive(Builder, FieldAccess, Describable)]
pub struct AnalyzeFormComponent {
    /// Tiered dispatcher (Local → LLM).
    #[field(skip)]
    pub dispatcher: Arc<dyn FieldValueDispatcher>,
    /// Shared user profile.
    #[field(skip)]
    pub profile: Arc<RwLock<Profile>>,
    /// Optional audit log for append-only event recording.
    #[field(skip)]
    pub audit: Option<Arc<AuditLog>>,
    /// Optional form fill memory store for querying past fills.
    #[field(skip)]
    pub memory: Option<Arc<FormFillStore>>,
    /// Maximum context length for redaction (default 4096).
    #[field(desc = "Maximum context length for redaction")]
    #[builder(default = 4096)]
    pub max_context_len: usize,
}

impl WorkUnit for AnalyzeFormComponent {
    fn name(&self) -> &str {
        "analyze_form"
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let params = ctx.caps.get::<AnalyzeFormParamsCap>().ok_or_else(|| {
            WorkError::Dependency("missing AnalyzeFormParamsCap in context caps".into())
        })?;

        let start = std::time::Instant::now();
        let sanitized = sanitize::redact_context(
            params.0.page_context.as_deref().unwrap_or(""),
            self.max_context_len,
        );

        let mut prefilled = Vec::new();
        let mut skipped = Vec::new();

        for field in &params.0.fields {
            if sanitize::is_sensitive_field(field) {
                skipped.push(SkippedField {
                    field_id: field.field_id.clone(),
                    reason: SkippedReason::SensitiveType,
                    suggested_action: "skip".into(),
                });
                continue;
            }

            // Check memory store first for high-trust past fills.
            if let Some(memory) = &self.memory {
                if let Ok(entries) = memory.search(&field.label, 1) {
                    if let Some(entry) = entries.first() {
                        if entry.trust_score > 0.8 {
                            prefilled.push(crate::schema::PreFilledValue {
                                field_id: field.field_id.clone(),
                                value: entry.value.clone(),
                                confidence: entry.confidence,
                                source: ValueSource::Resume,
                                reasoning: format!(
                                    "Previously used (trust {:.0}%)",
                                    entry.trust_score * 100.0
                                ),
                            });
                            continue;
                        }
                    }
                }
            }

            // Fall through to tiered dispatcher.
            if let Some(v) =
                self.dispatcher
                    .route(field, &sanitized, params.0.company_hint.as_deref())
            {
                prefilled.push(v);
            } else {
                skipped.push(SkippedField {
                    field_id: field.field_id.clone(),
                    reason: SkippedReason::NoMatch,
                    suggested_action: "manual".into(),
                });
            }
        }

        let duration_us = start.elapsed().as_micros() as u64;

        // Record in audit log (all values are pre-hashed or counts).
        if let Some(audit) = &self.audit {
            let _ = audit.record_analyze(
                &blake3_hex(params.0.request_id.as_bytes()),
                &blake3_hex(params.0.url.as_bytes()),
                prefilled.len() as u64,
                skipped.len() as u64,
                duration_us,
            );
        }

        let response = AnalyzeFormResponse {
            request_id: params.0.request_id.clone(),
            prefilled,
            skipped,
        };

        let data = serde_json::to_value(&response)
            .map_err(|e| WorkError::Execution(format!("serialize response: {e}")))?;

        Ok(WorkOutput::ok_with_data("form analyzed", data))
    }
}

impl AnalyzeFormComponent {
    /// Execute the form analysis and return the deserialized response.
    pub fn execute_analyze(
        &self,
        params: &PageAnalyzeFormParams,
    ) -> Result<AnalyzeFormResponse, String> {
        let ctx = WorkContext::for_unit(
            self,
            fluent_wvr::CapabilitySet::new().with(AnalyzeFormParamsCap(params.clone())),
        );
        let output = self.execute(&ctx).map_err(|e| e.to_string())?;
        serde_json::from_value(output.data).map_err(|e| format!("deserialize response: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::{LocalDispatcher, TieredDispatcher};
    use crate::schema::FieldDescription;

    fn test_component() -> AnalyzeFormComponent {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.last_name = "Lovelace".into();
        profile.personal.email = "ada@example.com".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let dispatcher: Arc<dyn FieldValueDispatcher> =
            Arc::new(TieredDispatcher::new().with(local));
        AnalyzeFormComponent::builder()
            .dispatcher(dispatcher)
            .profile(shared)
            .build()
    }

    #[test]
    fn component_name_is_analyze_form() {
        let c = test_component();
        assert_eq!(c.name(), "analyze_form");
    }

    #[test]
    fn component_depends_and_provides_are_empty() {
        let c = test_component();
        assert!(c.depends().is_empty());
        assert!(c.provides().is_empty());
    }

    #[test]
    fn derive_field_names_excludes_skipped() {
        let c = test_component();
        let names = c.field_names();
        assert!(names.contains(&"max_context_len"));
        assert!(!names.contains(&"dispatcher"));
        assert!(!names.contains(&"profile"));
        assert!(!names.contains(&"audit"));
    }

    #[test]
    fn set_and_get_max_context_len() {
        let mut c = test_component();
        c.set_field("max_context_len", "8192").unwrap();
        assert_eq!(c.get_field("max_context_len").unwrap(), "8192");
    }

    #[test]
    fn set_invalid_max_context_len_fails() {
        let mut c = test_component();
        assert!(c.set_field("max_context_len", "not_a_number").is_err());
    }

    #[test]
    fn get_unknown_field_fails() {
        let c = test_component();
        assert!(c.get_field("nonexistent").is_err());
    }

    #[test]
    fn describe_returns_object_schema() {
        let c = test_component();
        let schema = c.describe();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["max_context_len"].is_object());
    }

    #[test]
    fn execute_with_matching_fields_returns_prefilled() {
        let c = test_component();
        let params = PageAnalyzeFormParams {
            url: "https://example.com".into(),
            company_hint: None,
            page_context: None,
            fields: vec![FieldDescription {
                field_id: "firstName".into(),
                label: "First Name".into(),
                input_type: "text".into(),
                selector: "#fn".into(),
                context_text: String::new(),
                required: false,
                current_value_hash: None,
                autocomplete: None,
                options: vec![],
            }],
            request_id: "req-1".into(),
        };
        let response = c.execute_analyze(&params).unwrap();
        assert_eq!(response.prefilled.len(), 1);
        assert_eq!(response.prefilled[0].value, "Ada");
        assert!(response.skipped.is_empty());
    }

    #[test]
    fn execute_skips_sensitive_fields() {
        let c = test_component();
        let params = PageAnalyzeFormParams {
            url: "https://example.com".into(),
            company_hint: None,
            page_context: None,
            fields: vec![FieldDescription {
                field_id: "password".into(),
                label: "Password".into(),
                input_type: "password".into(),
                selector: "#pass".into(),
                context_text: String::new(),
                required: true,
                current_value_hash: None,
                autocomplete: None,
                options: vec![],
            }],
            request_id: "req-2".into(),
        };
        let response = c.execute_analyze(&params).unwrap();
        assert!(response.prefilled.is_empty());
        assert_eq!(response.skipped.len(), 1);
        assert_eq!(response.skipped[0].reason, SkippedReason::SensitiveType);
    }

    #[test]
    fn execute_via_workunit_trait() {
        let c = test_component();
        let params = PageAnalyzeFormParams {
            url: "https://example.com".into(),
            company_hint: None,
            page_context: None,
            fields: vec![FieldDescription {
                field_id: "email".into(),
                label: "Email".into(),
                input_type: "email".into(),
                selector: "#email".into(),
                context_text: String::new(),
                required: true,
                current_value_hash: None,
                autocomplete: Some("email".into()),
                options: vec![],
            }],
            request_id: "req-3".into(),
        };
        let ctx = WorkContext::for_unit(
            &c,
            fluent_wvr::CapabilitySet::new().with(AnalyzeFormParamsCap(params)),
        );
        let output = WorkUnit::execute(&c, &ctx).unwrap();
        assert!(output.success);
        assert_eq!(output.message, "form analyzed");
        let response: AnalyzeFormResponse = serde_json::from_value(output.data).unwrap();
        assert_eq!(response.prefilled.len(), 1);
        assert_eq!(response.prefilled[0].value, "ada@example.com");
    }

    #[test]
    fn execute_missing_params_cap_fails() {
        let c = test_component();
        let ctx = WorkContext::default();
        let result = WorkUnit::execute(&c, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn max_context_len_default_is_4096() {
        let c = test_component();
        assert_eq!(c.max_context_len, 4096);
    }

    #[test]
    fn builder_sets_custom_max_context_len() {
        let mut profile = Profile::default();
        profile.personal.first_name = "Test".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let dispatcher: Arc<dyn FieldValueDispatcher> =
            Arc::new(TieredDispatcher::new().with(local));
        let c = AnalyzeFormComponent::builder()
            .dispatcher(dispatcher)
            .profile(shared)
            .max_context_len(8192)
            .build();
        assert_eq!(c.max_context_len, 8192);
    }
}
