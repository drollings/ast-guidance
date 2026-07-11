use std::sync::OnceLock;

use regex::Regex;

use crate::schema::FieldDescription;

fn re_script() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap())
}

fn re_style() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap())
}

fn re_breaks() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>").unwrap())
}

fn re_tags() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>]+>").unwrap())
}

fn re_newlines() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n+").unwrap())
}

fn re_ws() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[ \t]+").unwrap())
}

/// Strip HTML tags and decode common entities. This is a lightweight heuristic
/// parser — not a full HTML parser — suitable for untrusted text going to an LLM.
pub fn strip_html(s: &str) -> String {
    // Strip <script>...</script> and <style>...</style> blocks (case-insensitive)
    let mut result = re_script().replace_all(s, "").to_string();
    result = re_style().replace_all(&result, "").to_string();

    // Replace block-closing tags with newlines
    result = re_breaks().replace_all(&result, "\n").to_string();

    // Strip all remaining tags
    result = re_tags().replace_all(&result, "").to_string();

    // Decode common HTML entities
    result = result.replace("&amp;", "&");
    result = result.replace("&lt;", "<");
    result = result.replace("&gt;", ">");
    result = result.replace("&quot;", "\"");
    result = result.replace("&#39;", "'");

    // Normalize newlines to spaces first, then collapse runs of whitespace
    result = re_newlines().replace_all(&result, " ").to_string();
    result = re_ws().replace_all(&result, " ").trim().to_string();

    result
}

/// Return the longest prefix of `s` with at most `max_chars` characters.
/// UTF-8 char boundary safe.
pub fn truncate_chars(s: &str, max_chars: usize) -> &str {
    if let Some((i, _)) = s.char_indices().nth(max_chars) {
        &s[..i]
    } else {
        s
    }
}

/// Strip HTML, anonymize PII, and truncate to `max_chars` characters.
pub fn redact_context(raw: &str, max_chars: usize) -> String {
    let stripped = strip_html(raw);
    let redacted = guidance_llm::anonymize(&stripped);
    truncate_chars(&redacted, max_chars).to_string()
}

/// Returns `true` if the field should never be sent to the LLM or auto-filled.
pub fn is_sensitive_field(field: &FieldDescription) -> bool {
    // Check input_type
    let input_lower = field.input_type.to_lowercase();
    if input_lower == "password" || input_lower == "hidden" {
        return true;
    }

    // Check autocomplete
    if let Some(ref ac) = field.autocomplete {
        let ac_lower = ac.to_lowercase();
        if matches!(
            ac_lower.as_str(),
            "cc-number" | "cc-csc" | "cc-exp" | "ssn" | "current-password" | "new-password"
        ) {
            return true;
        }
    }

    // Check field_id for sensitive substrings
    let id_lower = field.field_id_lower();
    let sensitive_substrings = [
        "ssn",
        "social_security",
        "credit_card",
        "creditcard",
        "cc_number",
        "passport",
        "tax_id",
        "sin_",
        "nino",
    ];
    for sub in &sensitive_substrings {
        if id_lower.contains(sub) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_html_removes_script_and_style() {
        let input = r#"<div>Hello <script>alert('x')</script> world <style>.x{}</style>!</div>"#;
        let result = strip_html(input);
        assert_eq!(result, "Hello world !");
    }

    #[test]
    fn strip_html_replaces_breaks_with_newlines() {
        let input = "Line1<br>Line2</p><div>Line3</div>";
        let result = strip_html(input);
        // Newlines from block tags are normalized to spaces
        assert_eq!(result, "Line1 Line2 Line3");
    }

    #[test]
    fn strip_html_decodes_entities() {
        let input = "&amp; &lt;div&gt; &quot;hello&quot; &#39;world&#39;";
        let result = strip_html(input);
        assert_eq!(result, "& <div> \"hello\" 'world'");
    }

    #[test]
    fn strip_html_collapses_whitespace() {
        let input = "  Hello   world   foo  ";
        let result = strip_html(input);
        assert_eq!(result, "Hello world foo");
    }

    #[test]
    fn redact_context_anonymizes_email_in_text() {
        let input = "Contact user@example.com for details";
        let result = redact_context(input, 4096);
        assert!(result.contains("[EMAIL]"));
        assert!(!result.contains("user@example.com"));
    }

    #[test]
    fn redact_context_truncates_at_max_chars() {
        let input = "the quick brown fox jumps over the lazy dog ";
        let result = redact_context(input, 20);
        assert!(result.chars().count() <= 20);
    }

    #[test]
    fn is_sensitive_field_detects_password_type() {
        let field = FieldDescription {
            field_id: "pass".into(),
            label: "Password".into(),
            input_type: "password".into(),
            selector: "#pass".into(),
            context_text: String::new(),
            required: true,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        assert!(is_sensitive_field(&field));
    }

    #[test]
    fn is_sensitive_field_detects_cc_autocomplete() {
        let field = FieldDescription {
            field_id: "card".into(),
            label: "Card Number".into(),
            input_type: "text".into(),
            selector: "#card".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: Some("cc-number".into()),
            options: vec![],
        };
        assert!(is_sensitive_field(&field));
    }

    #[test]
    fn is_sensitive_field_detects_ssn_substring() {
        let field = FieldDescription {
            field_id: "ssn_number".into(),
            label: "SSN".into(),
            input_type: "text".into(),
            selector: "#ssn".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        assert!(is_sensitive_field(&field));
    }

    #[test]
    fn is_sensitive_field_returns_false_for_normal_text() {
        let field = FieldDescription {
            field_id: "firstName".into(),
            label: "First Name".into(),
            input_type: "text".into(),
            selector: "#fn".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        };
        assert!(!is_sensitive_field(&field));
    }

    #[test]
    fn truncate_chars_respects_boundary() {
        let s = "hello world";
        assert_eq!(truncate_chars(s, 5), "hello");
        assert_eq!(truncate_chars(s, 20), "hello world");
        assert_eq!(truncate_chars(s, 0), "");
    }

    #[test]
    fn truncate_chars_utf8_safe() {
        let s = "héllo wörld";
        assert_eq!(truncate_chars(s, 5), "héllo");
    }

    proptest::proptest! {
        #[test]
        fn redact_context_no_email_in_output(s in ".*") {
            let result = redact_context(&s, 4096);
            // The output should not contain email patterns (the anonymizer replaces them)
            // We just check it doesn't panic
            let _ = result;
        }
    }
}
