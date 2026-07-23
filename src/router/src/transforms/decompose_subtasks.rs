use guidance_llm::Decomposer;

use crate::transforms::{TransformError, TransformStrategy};
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

pub struct DecomposeToSubtasks {
    decomposer: Box<dyn Decomposer>,
}

impl DecomposeToSubtasks {
    pub fn new(decomposer: Box<dyn Decomposer>) -> Self {
        Self { decomposer }
    }
}

impl TransformStrategy for DecomposeToSubtasks {
    fn name(&self) -> &str {
        "decompose_to_subtasks"
    }

    fn transform(
        &self,
        request: &RouterRequest,
        _pii_classes: &[String],
    ) -> Result<RouterRequest, TransformError> {
        let mut transformed = request.clone();

        let user_text: String = request
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| match &m.content {
                RouterMessageContent::Text(s) => Some(s.as_str()),
                RouterMessageContent::Parts(_) => None,
            })
            .next_back()
            .unwrap_or("")
            .to_string();

        if user_text.is_empty() {
            return Err(TransformError::NotApplicable(
                "no user message to decompose".into(),
            ));
        }

        let subtasks = self.decomposer.decompose(&user_text);

        let subtasks_json = serde_json::json!({
            "original_query": user_text,
            "subtasks": subtasks,
            "count": subtasks.len(),
        });

        let subtask_intro = format!(
            "The request has been decomposed into {} subtask(s). Please execute each subtask in order:\n\n{}",
            subtasks.len(),
            subtasks
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let system_msg = RouterMessage {
            role: "system".into(),
            content: RouterMessageContent::Text(
                "You are executing decomposed subtasks. Complete each subtask \
                 in order and provide the combined result."
                    .into(),
            ),
            tool_calls: None,
            tool_call_id: None,
        };

        let subtask_msg = RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(subtask_intro),
            tool_calls: None,
            tool_call_id: None,
        };

        transformed.messages = vec![system_msg, subtask_msg];
        transformed.metadata.insert(
            "subtasks".into(),
            subtasks_json,
        );
        transformed.metadata.insert(
            "transform".into(),
            serde_json::json!("decompose_to_subtasks"),
        );

        Ok(transformed)
    }
}
