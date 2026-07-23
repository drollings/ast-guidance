#[cfg(test)]
mod tests {
    use crate::transforms::decompose_hypothetical::DecomposeToAnonymizedHypothetical;
    use crate::transforms::decompose_subtasks::DecomposeToSubtasks;
    use crate::transforms::none::NoTransform;
    use crate::transforms::pii_anonymize::PiiAnonymize;
    use crate::transforms::TransformStrategy;
    use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

    fn make_request(text: &str) -> RouterRequest {
        RouterRequest {
            model: "test-model".into(),
            messages: vec![RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text(text.into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            metadata: Default::default(),
        }
    }

    // ── M3.9: NoTransform ────────────────────────────────────────────────

    #[test]
    fn test_no_transform_passes_through_unchanged() {
        let transform = NoTransform;
        let request = make_request("Hello, world!");
        let result = transform.transform(&request, &[]).unwrap();
        assert_eq!(result.model, "test-model");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(
            result.messages[0].content.to_string_lossy(),
            "Hello, world!"
        );
    }

    #[test]
    fn test_no_transform_preserves_all_fields() {
        let transform = NoTransform;
        let mut request = make_request("test");
        request.temperature = Some(0.7);
        request.max_tokens = Some(2048);
        request.session_id = Some("sess-1".into());
        let result = transform.transform(&request, &[]).unwrap();
        assert_eq!(result.temperature, Some(0.7));
        assert_eq!(result.max_tokens, Some(2048));
        assert_eq!(result.session_id, Some("sess-1".into()));
    }

    #[test]
    fn test_no_transform_name() {
        let transform = NoTransform;
        assert_eq!(transform.name(), "none");
    }

    // ── M3.9: PiiAnonymize ──────────────────────────────────────────────

    #[test]
    fn test_pii_anonymize_redacts_ssn() {
        let transform = PiiAnonymize;
        let request = make_request("My SSN is 123-45-6789. Please help.");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert!(!output.contains("123-45-6789"), "SSN should be redacted");
        assert!(
            output.contains("[SSN]"),
            "output should contain [SSN] placeholder, got: {output}"
        );
    }

    #[test]
    fn test_pii_anonymize_redacts_email() {
        let transform = PiiAnonymize;
        let request = make_request("Contact me at user@example.com for details.");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert!(
            !output.contains("user@example.com"),
            "email should be redacted"
        );
        assert!(
            output.contains("[EMAIL]"),
            "output should contain [EMAIL] placeholder, got: {output}"
        );
    }

    #[test]
    fn test_pii_anonymize_redacts_credit_card() {
        let transform = PiiAnonymize;
        let request = make_request("Card: 4111-1111-1111-1111 expires 12/25.");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert!(
            !output.contains("4111-1111-1111-1111"),
            "card number should be redacted"
        );
    }

    #[test]
    fn test_pii_anonymize_redacts_phone() {
        let transform = PiiAnonymize;
        let request = make_request("Call me at 555-123-4567.");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert!(
            !output.contains("555-123-4567"),
            "phone should be redacted"
        );
    }

    #[test]
    fn test_pii_anonymize_stores_anonymize_map() {
        let transform = PiiAnonymize;
        let request = make_request("My email is user@example.com and SSN is 123-45-6789.");
        let result = transform.transform(&request, &[]).unwrap();
        let map = result.metadata.get("anonymize_map");
        assert!(map.is_some(), "anonymize_map should be present in metadata");
    }

    #[test]
    fn test_pii_anonymize_no_pii_passes_unchanged() {
        let transform = PiiAnonymize;
        let request = make_request("What is the capital of France?");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert_eq!(output, "What is the capital of France?");
    }

    #[test]
    fn test_pii_anonymize_name() {
        let transform = PiiAnonymize;
        assert_eq!(transform.name(), "pii_anonymize");
    }

    // ── M3.9: DecomposeToAnonymizedHypothetical ─────────────────────────

    #[test]
    fn test_decompose_hypothetical_no_pii_in_output() {
        let transform = DecomposeToAnonymizedHypothetical;
        let request = make_request("My SSN is 123-45-6789 and email is user@example.com");
        let result = transform.transform(&request, &[]).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert!(
            !output.contains("123-45-6789"),
            "SSN should not appear in output"
        );
        assert!(
            !output.contains("user@example.com"),
            "email should not appear in output"
        );
    }

    #[test]
    fn test_decompose_hypothetical_creates_system_and_user_messages() {
        let transform = DecomposeToAnonymizedHypothetical;
        let request = make_request("What is Rust?");
        let result = transform.transform(&request, &[]).unwrap();
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[1].role, "user");
    }

    #[test]
    fn test_decompose_hypothetical_contains_rubric() {
        let transform = DecomposeToAnonymizedHypothetical;
        let request = make_request("Help me with my code.");
        let result = transform.transform(&request, &[]).unwrap();
        let system_content = result.messages[0].content.to_string_lossy();
        assert!(
            system_content.contains("Rubric"),
            "system message should contain rubric"
        );
        assert!(
            system_content.contains("PII"),
            "system message should mention PII"
        );
    }

    #[test]
    fn test_decompose_hypothetical_metadata() {
        let transform = DecomposeToAnonymizedHypothetical;
        let request = make_request("Test query");
        let result = transform.transform(&request, &[]).unwrap();
        let transform_val = result.metadata.get("transform");
        assert!(transform_val.is_some());
        assert_eq!(
            transform_val.unwrap().as_str(),
            Some("decompose_to_anonymized_hypothetical")
        );
    }

    #[test]
    fn test_decompose_hypothetical_name() {
        let transform = DecomposeToAnonymizedHypothetical;
        assert_eq!(transform.name(), "decompose_to_anonymized_hypothetical");
    }

    // ── M3.9: DecomposeToSubtasks ───────────────────────────────────────

    #[test]
    fn test_decompose_subtasks_with_stub_decomposer() {
        use guidance_llm::Decomposer;

        let subtasks = vec![
            "Research the topic".to_string(),
            "Write the code".to_string(),
            "Test the solution".to_string(),
        ];

        struct StubDecomposer {
            output: Vec<String>,
        }

        impl Decomposer for StubDecomposer {
            fn decompose(&self, _task: &str) -> Vec<String> {
                self.output.clone()
            }
        }

        let decomposer = StubDecomposer {
            output: subtasks.clone(),
        };

        let transform = DecomposeToSubtasks::new(Box::new(decomposer));
        let request = make_request("Build a web app");
        let result = transform.transform(&request, &[]).unwrap();

        // Should have system + user messages
        assert_eq!(result.messages.len(), 2);
        let user_content = result.messages[1].content.to_string_lossy();
        assert!(
            user_content.contains("1."),
            "should contain numbered subtasks"
        );
        assert!(
            user_content.contains("Research"),
            "should contain first subtask"
        );
        assert!(
            user_content.contains("Test"),
            "should contain last subtask"
        );

        // Should have subtasks in metadata
        let subtasks_val = result.metadata.get("subtasks").unwrap();
        assert_eq!(
            subtasks_val["count"].as_u64().unwrap(),
            3
        );
    }

    #[test]
    fn test_decompose_subtasks_empty_input_returns_not_applicable() {
        struct EmptyDecomposer;
        impl guidance_llm::Decomposer for EmptyDecomposer {
            fn decompose(&self, _task: &str) -> Vec<String> {
                vec![]
            }
        }

        let transform = DecomposeToSubtasks::new(Box::new(EmptyDecomposer));
        let request = make_request(""); // empty user message triggers NotApplicable
        let result = transform.transform(&request, &[]);
        assert!(result.is_err(), "empty user text should error");
    }

    #[test]
    fn test_decompose_subtasks_name() {
        struct StubDecomposer;
        impl guidance_llm::Decomposer for StubDecomposer {
            fn decompose(&self, _task: &str) -> Vec<String> {
                vec!["subtask".into()]
            }
        }

        let transform = DecomposeToSubtasks::new(Box::new(StubDecomposer));
        assert_eq!(transform.name(), "decompose_to_subtasks");
    }

    // ── M3.9: No LLM calls in any test ──────────────────────────────────
    // All tests above use only fixture data — no live models or network.
}
