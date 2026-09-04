use std::collections::HashMap;

use fluent_llm::{anonymize, build_anonymize_map};

use crate::transforms::{
    insert_string_map, rewrite_text_messages, TransformError, TransformStrategy,
};
use crate::types::RouterRequest;

pub struct PiiAnonymize;

impl TransformStrategy for PiiAnonymize {
    fn name(&self) -> &str {
        "pii_anonymize"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        let mut anonymize_map: HashMap<String, String> = HashMap::new();

        let mut transformed = rewrite_text_messages(request, |content| {
            let anonymized = anonymize(content);
            build_anonymize_map(content, &anonymized, &mut anonymize_map);
            Ok(anonymized)
        })?;

        insert_string_map(&mut transformed, "anonymize_map", anonymize_map);

        Ok(transformed)
    }
}

// M6: the byte-diff reverse-map builder lives in `fluent_llm::anonymize`
// (`build_anonymize_map`, verbatim semantics) — this module keeps only the
// transform wiring. Do not reintroduce a local copy.
