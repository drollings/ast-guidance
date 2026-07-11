use serde::{Deserialize, Serialize};

/// Description of a form field extracted from a web page.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldDescription {
    pub field_id: String,
    pub label: String,
    /// "text" | "email" | "tel" | "number" | "textarea" | "select" | …
    pub input_type: String,
    pub selector: String,
    /// Surrounding paragraph / help text.
    pub context_text: String,
    #[serde(default)]
    pub required: bool,
    /// blake3 hex of the user's typed value; never the value itself.
    #[serde(default)]
    pub current_value_hash: Option<String>,
    #[serde(default)]
    pub autocomplete: Option<String>,
    /// For `<select>` elements.
    #[serde(default)]
    pub options: Vec<SelectOption>,
}

impl FieldDescription {
    /// Lowercase version of `field_id`, used for regex matching in dispatchers.
    #[must_use]
    pub fn field_id_lower(&self) -> String {
        self.field_id.to_lowercase()
    }
}

/// An option within a `<select>` element.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
    #[serde(default)]
    pub selected: bool,
}

/// Parameters for the `page.analyzeForm` RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PageAnalyzeFormParams {
    pub url: String,
    #[serde(default)]
    pub company_hint: Option<String>,
    #[serde(default)]
    pub page_context: Option<String>,
    pub fields: Vec<FieldDescription>,
    /// UUID generated client-side.
    #[serde(default)]
    pub request_id: String,
}

/// Source of a prefilled value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ValueSource {
    Resume,
    LocalInference,
    LlmGenerated,
    UserOverride,
}

/// A suggested value for a form field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreFilledValue {
    pub field_id: String,
    /// The actual value to paste.
    pub value: String,
    /// Confidence score, clamped to 0.0..=1.0.
    pub confidence: f32,
    pub source: ValueSource,
    /// Brief reasoning, <= 140 chars.
    pub reasoning: String,
}

/// Why a field was skipped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SkippedReason {
    SensitiveType,
    NoMatch,
    LlmError,
    LlmRefused,
    ContextTooUntrusted,
}

/// A field that was skipped during analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkippedField {
    pub field_id: String,
    pub reason: SkippedReason,
    /// "skip" | "manual" | "ask_user"
    pub suggested_action: String,
}

/// Response from the `page.analyzeForm` RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyzeFormResponse {
    pub request_id: String,
    pub prefilled: Vec<PreFilledValue>,
    pub skipped: Vec<SkippedField>,
}

/// User feedback action on a field value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackAction {
    Accepted,
    Overridden,
    Rejected,
    Skipped,
}

/// Parameters for the `session.feedback` RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeedbackParams {
    pub request_id: String,
    pub field_id: String,
    pub action: FeedbackAction,
    /// blake3 hex of the final user value.
    pub final_value_hash: String,
}

/// Latency histogram summary for `daemon.health`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HistogramSummary {
    pub count: u64,
    pub p50_ms: f64,
    pub p99_ms: f64,
}

/// Response from the `daemon.health` RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonHealth {
    pub version: String,
    pub profile_loaded: bool,
    pub llm_reachable: bool,
    pub uptime_s: u64,
    pub analyze_count: u64,
    pub histogram: HistogramSummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_description_roundtrip_minimal() {
        let fd = FieldDescription {
            field_id: "f1".into(),
            label: "Name".into(),
            input_type: "text".into(),
            selector: "#name".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        let json = serde_json::to_string(&fd).unwrap();
        let parsed: FieldDescription = serde_json::from_str(&json).unwrap();
        assert_eq!(fd, parsed);
    }

    #[test]
    fn field_description_roundtrip_full() {
        let fd = FieldDescription {
            field_id: "country".into(),
            label: "Country".into(),
            input_type: "select".into(),
            selector: "#country".into(),
            context_text: "Select your country".into(),
            required: true,
            current_value_hash: Some("abc123".into()),
            autocomplete: Some("country".into()),
            options: vec![
                SelectOption {
                    value: "US".into(),
                    label: "United States".into(),
                    selected: false,
                },
                SelectOption {
                    value: "CA".into(),
                    label: "Canada".into(),
                    selected: true,
                },
            ],
        };
        let json = serde_json::to_string(&fd).unwrap();
        let parsed: FieldDescription = serde_json::from_str(&json).unwrap();
        assert_eq!(fd, parsed);
    }

    #[test]
    fn field_description_unknown_optional_field() {
        let json = r##"{"field_id":"x","label":"X","input_type":"text","selector":"#x","context_text":"","unknown_field":"ignored"}"##;
        let parsed: FieldDescription = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.field_id, "x");
    }

    #[test]
    fn page_analyze_form_params_roundtrip() {
        let params = PageAnalyzeFormParams {
            url: "https://example.com".into(),
            company_hint: Some("Acme".into()),
            page_context: None,
            fields: vec![],
            request_id: "uuid-123".into(),
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: PageAnalyzeFormParams = serde_json::from_str(&json).unwrap();
        assert_eq!(params, parsed);
    }

    #[test]
    fn value_source_serde() {
        let v = ValueSource::LlmGenerated;
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, r#""llm_generated""#);
        let parsed: ValueSource = serde_json::from_str(&json).unwrap();
        assert_eq!(v, parsed);
    }

    #[test]
    fn prefilled_value_clamp() {
        let pf = PreFilledValue {
            field_id: "f1".into(),
            value: "hello".into(),
            confidence: 1.5,
            source: ValueSource::Resume,
            reasoning: "test".into(),
        };
        let json = serde_json::to_string(&pf).unwrap();
        let parsed: PreFilledValue = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.confidence, 1.5);
    }

    #[test]
    fn skipped_field_roundtrip() {
        let sf = SkippedField {
            field_id: "password".into(),
            reason: SkippedReason::SensitiveType,
            suggested_action: "skip".into(),
        };
        let json = serde_json::to_string(&sf).unwrap();
        let parsed: SkippedField = serde_json::from_str(&json).unwrap();
        assert_eq!(sf, parsed);
    }

    #[test]
    fn analyze_form_response_roundtrip() {
        let resp = AnalyzeFormResponse {
            request_id: "req-1".into(),
            prefilled: vec![PreFilledValue {
                field_id: "f1".into(),
                value: "Ada".into(),
                confidence: 1.0,
                source: ValueSource::Resume,
                reasoning: "from profile".into(),
            }],
            skipped: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: AnalyzeFormResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    #[test]
    fn feedback_params_roundtrip() {
        let fp = FeedbackParams {
            request_id: "r1".into(),
            field_id: "f1".into(),
            action: FeedbackAction::Overridden,
            final_value_hash: "abc".into(),
        };
        let json = serde_json::to_string(&fp).unwrap();
        let parsed: FeedbackParams = serde_json::from_str(&json).unwrap();
        assert_eq!(fp, parsed);
    }

    #[test]
    fn daemon_health_roundtrip() {
        let h = DaemonHealth {
            version: "0.1.0".into(),
            profile_loaded: true,
            llm_reachable: false,
            uptime_s: 42,
            analyze_count: 10,
            histogram: HistogramSummary {
                count: 10,
                p50_ms: 1.5,
                p99_ms: 5.0,
            },
        };
        let json = serde_json::to_string(&h).unwrap();
        let parsed: DaemonHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(h, parsed);
    }

    #[test]
    fn histogram_summary_default() {
        let h = HistogramSummary::default();
        assert_eq!(h.count, 0);
        assert_eq!(h.p50_ms, 0.0);
        assert_eq!(h.p99_ms, 0.0);
    }

    #[test]
    fn field_id_lower() {
        let fd = FieldDescription {
            field_id: "FirstName".into(),
            label: "First Name".into(),
            input_type: "text".into(),
            selector: "#fn".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        assert_eq!(fd.field_id_lower(), "firstname");
    }
}
