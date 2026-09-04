use common_core::string::{filter_unsafe_chars, AnsiStripper};

use crate::transforms::{rewrite_text_messages, TransformError, TransformStrategy};
use crate::types::RouterRequest;

pub struct Sanitize;

impl TransformStrategy for Sanitize {
    fn name(&self) -> &str {
        "sanitize"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        rewrite_text_messages(request, |content| {
            let cleaned: String = AnsiStripper::new(content).collect();
            Ok(filter_unsafe_chars(&cleaned))
        })
    }
}
#[cfg(test)]
#[path = "../../tests/transforms_sanitize.rs"]
mod tests;
