use std::sync::Arc;
use std::time::Instant;

use crate::components::AnalyzeFormParamsCap;
use crate::memory::FormFillStore;
use crate::profile::Profile;
use crate::schema::{DaemonHealth, FeedbackParams, HistogramSummary, PageAnalyzeFormParams};
use crate::server::audit::AuditLog;
use common_core::hash::blake3_hex;
use common_core::jsonrpc::{JsonRpcError, JsonRpcHandler, JsonRpcRequest, JsonRpcResponse};
use common_core::metrics::LatencyHistogram;
use fluent_wvr::{CapabilitySet, Component, WorkContext, WorkUnit};
use std::sync::RwLock;

/// Central handler for all JSON-RPC methods.
///
/// Holds the shared profile, an optional append-only audit log, and a
/// WorkUnit component that performs form analysis (wrapped with middleware
/// for retry + timing).
pub struct DaemonHandler {
    pub(crate) profile: Arc<RwLock<Profile>>,
    pub(crate) started_at: Instant,
    audit: Option<Arc<AuditLog>>,
    histogram: Arc<LatencyHistogram>,
    unit: Arc<dyn Component>,
    memory: Option<Arc<FormFillStore>>,
}

impl DaemonHandler {
    pub fn new(profile: Arc<RwLock<Profile>>, unit: Arc<dyn Component>) -> Self {
        Self {
            profile,
            started_at: Instant::now(),
            audit: None,
            histogram: Arc::new(LatencyHistogram::new()),
            unit,
            memory: None,
        }
    }

    /// Set the audit log for this handler.
    #[must_use]
    pub fn with_audit(mut self, audit: Arc<AuditLog>) -> Self {
        self.audit = Some(audit);
        self
    }

    /// Set the shared latency histogram.
    #[must_use]
    pub fn with_histogram(mut self, histogram: Arc<LatencyHistogram>) -> Self {
        self.histogram = histogram;
        self
    }

    /// Set the form fill memory store for feedback recording.
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<FormFillStore>) -> Self {
        self.memory = Some(memory);
        self
    }

    fn dispatch(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "page.analyzeForm" => self.handle_analyze_form(request),
            "session.feedback" => self.handle_feedback(request),
            "daemon.health" => self.handle_health(request),
            method => {
                if let Some(audit) = &self.audit {
                    let rid = request
                        .id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_default();
                    let _ =
                        audit.record_error(&blake3_hex(rid.as_bytes()), "unknown_method", method);
                }
                common_core::jsonrpc::method_not_found(request.id.clone(), method)
            }
        }
    }

    fn handle_analyze_form(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params: PageAnalyzeFormParams = if let Some(p) = request.params.as_ref() {
            match serde_json::from_value(p.clone()) {
                Ok(p) => p,
                Err(e) => {
                    self.record_error_from_request(request, "invalid_params", &e.to_string());
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id.clone(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: format!("invalid params: {e}"),
                        }),
                    };
                }
            }
        } else {
            self.record_error_from_request(request, "invalid_params", "missing params");
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "missing params".into(),
                }),
            };
        };

        let ctx = WorkContext {
            caps: CapabilitySet::new().with(AnalyzeFormParamsCap(params)),
            ..WorkContext::default()
        };

        match WorkUnit::execute(self.unit.as_ref(), &ctx) {
            Ok(output) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: Some(output.data),
                error: None,
            },
            Err(e) => {
                self.record_error_from_request(request, "execute_error", &e.to_string());
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id.clone(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32603,
                        message: format!("internal error: {e}"),
                    }),
                }
            }
        }
    }

    fn handle_feedback(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params: FeedbackParams = if let Some(p) = request.params.as_ref() {
            match serde_json::from_value(p.clone()) {
                Ok(p) => p,
                Err(e) => {
                    self.record_error_from_request(request, "invalid_params", &e.to_string());
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id.clone(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: format!("invalid params: {e}"),
                        }),
                    };
                }
            }
        } else {
            self.record_error_from_request(request, "invalid_params", "missing params");
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "missing params".into(),
                }),
            };
        };

        if let Some(audit) = &self.audit {
            let action_str =
                serde_json::to_string(&params.action).unwrap_or_else(|_| "unknown".into());
            let action_str = action_str.trim_matches('"');
            let _ = audit.record_feedback(
                &blake3_hex(params.request_id.as_bytes()),
                &blake3_hex(params.field_id.as_bytes()),
                action_str,
                &params.final_value_hash,
            );
        }

        // Record accepted/overridden feedback in the memory store.
        if let Some(memory) = &self.memory {
            let helpful = matches!(
                params.action,
                crate::schema::FeedbackAction::Accepted | crate::schema::FeedbackAction::Overridden
            );
            let _ = memory.record_feedback(&params.field_id, &params.final_value_hash, helpful);
        }

        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: request.id.clone(),
            result: Some(serde_json::json!({"ok": true})),
            error: None,
        }
    }

    fn handle_health(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let h = &self.histogram;
        let health = DaemonHealth {
            version: env!("CARGO_PKG_VERSION").to_string(),
            profile_loaded: self.profile.read().is_ok(),
            llm_reachable: false,
            uptime_s: self.started_at.elapsed().as_secs(),
            analyze_count: h.count(),
            histogram: HistogramSummary {
                count: h.count(),
                p50_ms: h.estimate_percentile(50.0) as f64,
                p99_ms: h.estimate_percentile(99.0) as f64,
            },
        };

        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: request.id.clone(),
            result: Some(serde_json::to_value(health).unwrap()),
            error: None,
        }
    }

    fn record_error_from_request(&self, request: &JsonRpcRequest, kind: &str, message: &str) {
        if let Some(audit) = &self.audit {
            let rid = request
                .id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default();
            let _ = audit.record_error(&blake3_hex(rid.as_bytes()), kind, message);
        }
    }
}

impl JsonRpcHandler for DaemonHandler {
    fn handle_request(&self, raw: &str) -> Result<String, JsonRpcError> {
        let req: JsonRpcRequest = serde_json::from_str(raw)?;
        let resp = self.dispatch(&req);
        Ok(serde_json::to_string(&resp)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AnalyzeFormComponent;
    use crate::dispatcher::{FieldValueDispatcher, LocalDispatcher, TieredDispatcher};
    use crate::profile::Profile;
    use crate::schema::{AnalyzeFormResponse, FieldDescription, PageAnalyzeFormParams};
    use dag::middleware::{MiddlewareChain, RetryMiddleware, TimingMiddleware};
    use proptest::prelude::*;
    use tempfile::TempDir;

    fn test_unit() -> Arc<dyn Component> {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.last_name = "Lovelace".into();
        profile.personal.email = "ada@example.com".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let dispatcher: Arc<dyn FieldValueDispatcher> =
            Arc::new(TieredDispatcher::new().with(local));

        let base: Arc<dyn Component> = Arc::new(
            AnalyzeFormComponent::builder()
                .dispatcher(dispatcher)
                .profile(shared)
                .build(),
        );
        let chain = MiddlewareChain::new()
            .push(Box::new(TimingMiddleware))
            .push(Box::new(RetryMiddleware::new(2, 50)));
        chain.apply(base)
    }

    fn test_handler() -> DaemonHandler {
        DaemonHandler::new(
            Arc::new(RwLock::new({
                let mut p = Profile::default();
                p.personal.first_name = "Ada".into();
                p.personal.last_name = "Lovelace".into();
                p.personal.email = "ada@example.com".into();
                p
            })),
            test_unit(),
        )
    }

    fn test_handler_with_audit() -> (DaemonHandler, tempfile::TempDir) {
        let dir = TempDir::new().unwrap();
        let log_path = dir.path().join("audit.jsonl");
        let audit = Arc::new(AuditLog::open(&log_path).unwrap());

        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.last_name = "Lovelace".into();
        profile.personal.email = "ada@example.com".into();
        let shared = Arc::new(RwLock::new(profile));
        let local = Arc::new(LocalDispatcher::new(shared.clone()));
        let dispatcher: Arc<dyn FieldValueDispatcher> =
            Arc::new(TieredDispatcher::new().with(local));

        let base: Arc<dyn Component> = Arc::new(
            AnalyzeFormComponent::builder()
                .dispatcher(dispatcher)
                .profile(shared.clone())
                .audit(audit.clone())
                .build(),
        );
        let chain = MiddlewareChain::new()
            .push(Box::new(TimingMiddleware))
            .push(Box::new(RetryMiddleware::new(2, 50)));
        let unit = chain.apply(base);

        let handler = DaemonHandler::new(shared, unit).with_audit(audit);
        (handler, dir)
    }

    fn make_request(method: &str, params: serde_json::Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: method.into(),
            id: Some(serde_json::json!(1)),
            params: Some(params),
        }
    }

    #[test]
    fn dispatch_analyze_form_prefills_matching_fields() {
        let handler = test_handler();
        let params = serde_json::to_value(PageAnalyzeFormParams {
            url: "https://example.com".into(),
            company_hint: None,
            page_context: None,
            fields: vec![
                FieldDescription {
                    field_id: "firstName".into(),
                    label: "First Name".into(),
                    input_type: "text".into(),
                    selector: "#fn".into(),
                    context_text: String::new(),
                    required: false,
                    current_value_hash: None,
                    autocomplete: None,
                    options: vec![],
                },
                FieldDescription {
                    field_id: "favorite_color".into(),
                    label: "Favorite Color".into(),
                    input_type: "text".into(),
                    selector: "#color".into(),
                    context_text: String::new(),
                    required: false,
                    current_value_hash: None,
                    autocomplete: None,
                    options: vec![],
                },
            ],
            request_id: "req-1".into(),
        })
        .unwrap();

        let request = make_request("page.analyzeForm", params);
        let resp = handler.dispatch(&request);

        assert!(resp.error.is_none());
        let analyze: AnalyzeFormResponse = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(analyze.prefilled.len(), 1);
        assert_eq!(analyze.prefilled[0].value, "Ada");
        assert_eq!(analyze.skipped.len(), 1);
        assert_eq!(analyze.skipped[0].field_id, "favorite_color");
    }

    #[test]
    fn dispatch_analyze_form_skips_sensitive_fields() {
        let handler = test_handler();
        let params = serde_json::to_value(PageAnalyzeFormParams {
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
        })
        .unwrap();

        let request = make_request("page.analyzeForm", params);
        let resp = handler.dispatch(&request);

        let analyze: AnalyzeFormResponse = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert!(analyze.prefilled.is_empty());
        assert_eq!(analyze.skipped.len(), 1);
    }

    #[test]
    fn dispatch_method_not_found() {
        let handler = test_handler();
        let request = make_request("unknown.method", serde_json::json!(null));
        let resp = handler.dispatch(&request);
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, common_core::jsonrpc::METHOD_NOT_FOUND);
    }

    #[test]
    fn dispatch_health_returns_daemon_health() {
        let handler = test_handler();
        let request = make_request("daemon.health", serde_json::json!(null));
        let resp = handler.dispatch(&request);
        assert!(resp.error.is_none());
        let health: DaemonHealth = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
        assert!(health.profile_loaded);
    }

    #[test]
    fn handle_request_returns_valid_json_string() {
        let handler = test_handler();
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"daemon.health"}"#;
        let resp_str = handler.handle_request(raw).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn analyze_with_audit_records_entry() {
        let (handler, dir) = test_handler_with_audit();
        let params = serde_json::to_value(PageAnalyzeFormParams {
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
            request_id: "req-audit-1".into(),
        })
        .unwrap();

        let request = make_request("page.analyzeForm", params);
        let _resp = handler.dispatch(&request);

        let log_path = dir.path().join("audit.jsonl");
        let content = std::fs::read_to_string(&log_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["kind"], "analyze");
        assert_eq!(v["prefilled_count"], 1);
        assert_eq!(v["skipped_count"], 0);
        assert!(v["duration_us"].as_u64().is_some());
        assert!(!content.contains("https://example.com"));
    }

    #[test]
    fn feedback_with_audit_records_entry() {
        let (handler, dir) = test_handler_with_audit();
        let params = serde_json::to_value(crate::schema::FeedbackParams {
            request_id: "req-fb-1".into(),
            field_id: "firstName".into(),
            action: crate::schema::FeedbackAction::Accepted,
            final_value_hash: "abc123".into(),
        })
        .unwrap();

        let request = make_request("session.feedback", params);
        let _resp = handler.dispatch(&request);

        let log_path = dir.path().join("audit.jsonl");
        let content = std::fs::read_to_string(&log_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["kind"], "feedback");
        assert_eq!(v["action"], "accepted");
        assert_eq!(v["final_value_hash"], "abc123");
        assert!(!content.contains("firstName"));
    }

    #[test]
    fn unknown_method_with_audit_records_error() {
        let (handler, dir) = test_handler_with_audit();
        let request = make_request("unknown.method", serde_json::json!(null));
        let _resp = handler.dispatch(&request);

        let log_path = dir.path().join("audit.jsonl");
        let content = std::fs::read_to_string(&log_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["kind"], "error");
        assert_eq!(v["error_kind"], "unknown_method");
        assert_eq!(v["message"], "unknown.method");
    }

    #[test]
    fn health_returns_real_histogram_data() {
        let histogram = Arc::new(LatencyHistogram::new());
        histogram.observe(10);
        histogram.observe(50);

        let handler = DaemonHandler::new(
            Arc::new(RwLock::new({
                let mut p = Profile::default();
                p.personal.first_name = "Test".into();
                p
            })),
            test_unit(),
        )
        .with_histogram(histogram);

        let request = make_request("daemon.health", serde_json::json!(null));
        let resp = handler.dispatch(&request);
        assert!(resp.error.is_none());
        let health: DaemonHealth = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(health.analyze_count, 2);
        assert!(health.histogram.count > 0);
    }

    proptest::proptest! {
        #[test]
        fn audit_log_contains_no_pii(
            first_name in "[a-z]{4,12}",
            last_name in "[a-z]{4,12}",
            email_user in "[a-z]{4,10}",
            email_domain in "[a-z]{4,8}\\.(com|org|net)",
            phone in "1?[0-9]{10,12}",
            url_path in "[a-z]{5,12}/[a-z]{5,12}",
        ) {
            let mut profile = Profile::default();
            profile.personal.first_name = first_name.clone();
            profile.personal.last_name = last_name.clone();
            profile.personal.email = format!("{email_user}@{email_domain}");
            profile.personal.phone = phone.clone();

            let shared = Arc::new(RwLock::new(profile));
            let local = Arc::new(LocalDispatcher::new(shared.clone()));
            let dispatcher: Arc<dyn FieldValueDispatcher> =
                Arc::new(TieredDispatcher::new().with(local));

            let base: Arc<dyn Component> = Arc::new(
                AnalyzeFormComponent::builder()
                    .dispatcher(dispatcher)
                    .profile(shared)
                    .build(),
            );
            let chain = MiddlewareChain::new()
                .push(Box::new(TimingMiddleware))
                .push(Box::new(RetryMiddleware::new(2, 50)));
            let unit = chain.apply(base);

            let dir = TempDir::new().unwrap();
            let log_path = dir.path().join("audit.jsonl");
            let audit = Arc::new(AuditLog::open(&log_path).unwrap());
            let handler = DaemonHandler::new(
                Arc::new(RwLock::new(Profile::default())),
                unit,
            )
            .with_audit(audit);

            let params = serde_json::to_value(PageAnalyzeFormParams {
                url: format!("https://{url_path}"),
                company_hint: Some("Acme Corp".into()),
                page_context: Some(format!(
                    "Contact {first_name} at the office. Phone: {phone}"
                )),
                fields: vec![
                    FieldDescription {
                        field_id: "firstName".into(),
                        label: "First Name".into(),
                        input_type: "text".into(),
                        selector: "#fn".into(),
                        context_text: String::new(),
                        required: false,
                        current_value_hash: None,
                        autocomplete: None,
                        options: vec![],
                    },
                    FieldDescription {
                        field_id: "email".into(),
                        label: "Email".into(),
                        input_type: "email".into(),
                        selector: "#email".into(),
                        context_text: String::new(),
                        required: true,
                        current_value_hash: None,
                        autocomplete: Some("email".into()),
                        options: vec![],
                    },
                ],
                request_id: format!("req-{url_path}"),
            }).unwrap();

            let request = make_request("page.analyzeForm", params);
            let _resp = handler.dispatch(&request);

            let fb_params = serde_json::to_value(crate::schema::FeedbackParams {
                request_id: format!("req-{url_path}"),
                field_id: "firstName".into(),
                action: crate::schema::FeedbackAction::Accepted,
                final_value_hash: "abc123def".into(),
            }).unwrap();
            let fb_request = make_request("session.feedback", fb_params);
            let _resp = handler.dispatch(&fb_request);

            let content = std::fs::read_to_string(&log_path).unwrap();

            let pii_values = [
                &first_name,
                &last_name,
                &format!("{email_user}@{email_domain}"),
                &phone,
                &format!("https://{url_path}"),
                "Acme Corp",
            ];

            for pii in &pii_values {
                prop_assert!(
                    !content.contains(pii),
                    "audit log contains PII substring '{pii}':\n{content}"
                );
            }

            for line in content.lines() {
                let _: serde_json::Value = serde_json::from_str(line)
                    .expect("audit line is not valid JSON");
            }
        }
    }
}
