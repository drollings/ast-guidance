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

            let cleaned = strip_ansi(&text);
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

fn strip_ansi(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    #[allow(clippy::while_let_loop)]
    loop {
        match chars.next() {
            None => break,
            Some('\u{1B}') => {
                if let Some(&'[') = chars.peek() {
                    chars.next(); // consume '['
                    loop {
                        match chars.peek() {
                            Some(&p) if ('\u{0030}'..='\u{003F}').contains(&p)
                                || ('\u{0020}'..='\u{002F}').contains(&p) =>
                            {
                                chars.next();
                            }
                            _ => break,
                        }
                    }
                    if let Some(&f) = chars.peek() {
                        if ('\u{0040}'..='\u{007E}').contains(&f) {
                            chars.next();
                        }
                    }
                } else {
                    result.push('\u{1B}');
                }
            }
            Some(c) => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RouterMessage, RouterRequest};

    fn make_request(text: &str) -> RouterRequest {
        RouterRequest {
            model: "test".into(),
            messages: vec![RouterMessage {
                role: "assistant".into(),
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

    fn text_of(result: &RouterRequest) -> String {
        result.messages[0].content.to_string_lossy()
    }

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
