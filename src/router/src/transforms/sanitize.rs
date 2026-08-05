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
mod tests {
    use super::*;
    use crate::testing::{test_request, text_of};
    use crate::types::RouterRequest;

    fn make_request(text: &str) -> RouterRequest {
        let mut req = test_request(text);
        req.messages[0].role = "assistant".into();
        req.model = "test".into();
        req
    }

    // ── Transform integration tests ──────────────────────────────────

    #[test]
    fn bidi_override_removed() {
        let m = Sanitize;
        let req = make_request("hello\u{202E}world"); // RLO
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "helloworld");
    }

    #[test]
    fn plane14_tags_removed() {
        let m = Sanitize;
        let req = make_request("text\u{E0001}more"); // Plane-14 tag
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "textmore");
    }

    #[test]
    fn null_byte_removed() {
        let m = Sanitize;
        let req = make_request("before\x00after");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "beforeafter");
    }

    #[test]
    fn c1_control_removed() {
        let m = Sanitize;
        let req = make_request("a\u{0081}b"); // C1 control
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "ab");
    }

    #[test]
    fn ansi_color_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[31mRED\u{1B}[0m normal");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "RED normal");
    }

    #[test]
    fn ansi_256_color_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[38;5;196mbright red\u{1B}[0m");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "bright red");
    }

    #[test]
    fn ansi_rgb_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[38;2;255;0;0mRGB red\u{1B}[0m");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "RGB red");
    }

    #[test]
    fn normal_text_preserved() {
        let m = Sanitize;
        let req = make_request("Hello, 世界! 😀 and more text.");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "Hello, 世界! 😀 and more text.");
    }

    #[test]
    fn plain_text_passes_unchanged() {
        let m = Sanitize;
        let req = make_request("just some plain text");
        let result = m.transform(&req).unwrap();
        assert_eq!(text_of(&result), "just some plain text");
    }
}
