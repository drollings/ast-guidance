use guidance_llm::anonymize;

use crate::transforms::{TransformError, TransformStrategy};
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

pub struct DecomposeToAnonymizedHypothetical;

impl TransformStrategy for DecomposeToAnonymizedHypothetical {
    fn name(&self) -> &str {
        "decompose_to_anonymized_hypothetical"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        let mut system_msg: Option<RouterMessage> = None;
        let mut hypothetical_msg: Option<RouterMessage> = None;

        for message in &request.messages {
            let content_str = match &message.content {
                RouterMessageContent::Text(s) => s.clone(),
                RouterMessageContent::Parts(_) => continue,
            };

            let anonymized = anonymize(&content_str);
            let hypothetical = build_hypothetical(&anonymized);

            system_msg = Some(RouterMessage {
                role: "system".into(),
                content: RouterMessageContent::Text(
                    "You are analyzing an anonymized hypothetical scenario. \
                     The original user query may have contained PII which has been \
                     replaced with placeholders like [EMAIL], [SSN], [PHONE], etc. \
                     Respond to the anonymized query below. Keep in mind that \
                     placeholders represent real sensitive data. Your response \
                     should not contain any actual PII.\n\n\
                     Rubric for response:\n\
                     1. Address the core question/request in the anonymized query\n\
                     2. If placeholders are present, refer to them generically \
                     (e.g., 'the user's email' rather than repeating the placeholder)\n\
                     3. Do NOT attempt to guess or reconstruct the original PII values\n\
                     4. Provide the same level of detail you would for the original query"
                        .into(),
                ),
                tool_calls: None,
                tool_call_id: None,
            });

            hypothetical_msg = Some(RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text(hypothetical),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        let mut transformed = request.clone();
        if let (Some(sys), Some(hyp)) = (system_msg, hypothetical_msg) {
            transformed.messages = vec![sys, hyp];
        }

        transformed.metadata.insert(
            "transform".into(),
            serde_json::json!("decompose_to_anonymized_hypothetical"),
        );

        Ok(transformed)
    }
}

fn build_hypothetical(anonymized: &str) -> String {
    [
        "# Anonymized Query",
        "",
        "The following user input has been anonymized for privacy. \
         Placeholders replace any PII that was present in the original message.",
        "",
        "---",
        "",
        anonymized,
        "",
        "---",
        "",
        "Please respond as if this were the original query, without reconstructing or repeating PII.",
    ]
    .join("\n")
}
