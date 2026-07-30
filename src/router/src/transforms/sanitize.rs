use std::str::Chars;

use crate::transforms::{TransformError, TransformStrategy};
use crate::types::{RouterMessageContent, RouterRequest};

pub struct Sanitize;

impl TransformStrategy for Sanitize {
    fn name(&self) -> &str {
        "sanitize"
    }

    fn transform(
        &self,
        request: &RouterRequest,
        _pii_classes: &[String],
    ) -> Result<RouterRequest, TransformError> {
        let mut transformed = request.clone();

        for message in &mut transformed.messages {
            let text = match &message.content {
                RouterMessageContent::Text(s) => s.clone(),
                RouterMessageContent::Parts(_) => continue,
            };

            let cleaned: String = AnsiStripper::new(&text).collect();
            let cleaned = filter_unsafe_chars(&cleaned);

            if cleaned != text {
                message.content = RouterMessageContent::Text(cleaned);
            }
        }

        Ok(transformed)
    }
}

fn filter_unsafe_chars(text: &str) -> String {
    text.chars()
        .filter(|&c| is_safe_char(c))
        .collect()
}

fn is_safe_char(c: char) -> bool {
    !matches!(
        c,
        '\u{0000}'
            | '\u{007F}'..='\u{009F}'
            | '\u{0001}'..='\u{0008}'
            | '\u{000B}'..='\u{000C}'
            | '\u{000E}'..='\u{001F}'
            | '\u{200E}'
            | '\u{200F}'
            | '\u{202A}'
            | '\u{202B}'
            | '\u{202C}'
            | '\u{202D}'
            | '\u{202E}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{E0000}'..='\u{E007F}'
    )
}

struct AnsiStripper<'a> {
    chars: Chars<'a>,
}

impl<'a> AnsiStripper<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            chars: text.chars(),
        }
    }
}

impl Iterator for AnsiStripper<'_> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        if c == '\u{1B}' {
            // Check for '[' following ESC — that starts a CSI sequence
            if let Some('[') = self.chars.clone().next() {
                self.chars.next(); // consume '['
                skip_csi_params(&mut self.chars);
                skip_csi_final(&mut self.chars);
                // Recurse to get the next visible character
                self.next()
            } else {
                // Lone ESC, not part of a CSI sequence
                Some('\u{1B}')
            }
        } else {
            Some(c)
        }
    }
}

fn skip_csi_params(chars: &mut Chars<'_>) {
    // Skip parameter bytes (0x30-0x3F) and intermediate bytes (0x20-0x2F)
    loop {
        let mut peek = chars.clone();
        match peek.next() {
            Some(p) if ('\u{0030}'..='\u{003F}').contains(&p)
                || ('\u{0020}'..='\u{002F}').contains(&p) =>
            {
                chars.next();
            }
            _ => break,
        }
    }
}

fn skip_csi_final(chars: &mut Chars<'_>) {
    // Skip the final byte (0x40-0x7E) if present
    let mut peek = chars.clone();
    if let Some(f) = peek.next() {
        if ('\u{0040}'..='\u{007E}').contains(&f) {
            chars.next();
        }
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

    // ── AnsiStripper unit tests ──────────────────────────────────────

    #[test]
    fn ansi_stripper_passthrough_plain_text() {
        let result: String = AnsiStripper::new("hello world").collect();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn ansi_stripper_removes_sgr_color() {
        let result: String = AnsiStripper::new("\u{1B}[31mRED\u{1B}[0m").collect();
        assert_eq!(result, "RED");
    }

    #[test]
    fn ansi_stripper_removes_256_color() {
        let result: String = AnsiStripper::new("\u{1B}[38;5;196mbright").collect();
        assert_eq!(result, "bright");
    }

    #[test]
    fn ansi_stripper_removes_rgb_color() {
        let result: String = AnsiStripper::new("\u{1B}[38;2;255;0;0mRGB red").collect();
        assert_eq!(result, "RGB red");
    }

    #[test]
    fn ansi_stripper_lone_esc_preserved() {
        let result: String = AnsiStripper::new("a\u{1B}x").collect();
        assert_eq!(result, "a\u{1B}x");
    }

    #[test]
    fn ansi_stripper_cjk_preserved() {
        let result: String = AnsiStripper::new("こんにちは").collect();
        assert_eq!(result, "こんにちは");
    }

    #[test]
    fn ansi_stripper_empty_input() {
        let result: String = AnsiStripper::new("").collect();
        assert_eq!(result, "");
    }

    // ── Transform integration tests ──────────────────────────────────

    #[test]
    fn bidi_override_removed() {
        let m = Sanitize;
        let req = make_request("hello\u{202E}world"); // RLO
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "helloworld");
    }

    #[test]
    fn plane14_tags_removed() {
        let m = Sanitize;
        let req = make_request("text\u{E0001}more"); // Plane-14 tag
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "textmore");
    }

    #[test]
    fn null_byte_removed() {
        let m = Sanitize;
        let req = make_request("before\x00after");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "beforeafter");
    }

    #[test]
    fn c1_control_removed() {
        let m = Sanitize;
        let req = make_request("a\u{0081}b"); // C1 control
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "ab");
    }

    #[test]
    fn ansi_color_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[31mRED\u{1B}[0m normal");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "RED normal");
    }

    #[test]
    fn ansi_256_color_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[38;5;196mbright red\u{1B}[0m");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "bright red");
    }

    #[test]
    fn ansi_rgb_stripped() {
        let m = Sanitize;
        let req = make_request("\u{1B}[38;2;255;0;0mRGB red\u{1B}[0m");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "RGB red");
    }

    #[test]
    fn normal_text_preserved() {
        let m = Sanitize;
        let req = make_request("Hello, 世界! 😀 and more text.");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "Hello, 世界! 😀 and more text.");
    }

    #[test]
    fn plain_text_passes_unchanged() {
        let m = Sanitize;
        let req = make_request("just some plain text");
        let result = m.transform(&req, &[]).unwrap();
        assert_eq!(text_of(&result), "just some plain text");
    }
}
