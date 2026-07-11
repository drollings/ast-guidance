use std::sync::{Arc, OnceLock, RwLock};

use regex::Regex;

use crate::profile::Profile;
use crate::sanitize;
use crate::schema::{FieldDescription, PreFilledValue, ValueSource};

/// Profile extractor function type.
type ProfileExtractor = fn(&Profile) -> &str;

/// Static table of (compiled-regex, extractor-fn) pairs for local field matching.
///
/// Each entry maps a regex pattern to a function that extracts a value from
/// the user's profile. The signal passed to each regex is the lowercased
/// value of a *single* field attribute (field_id, label, or autocomplete),
/// joined by a separator when the regex is anchored with `^`/`$`. Each
/// pattern is expected to be word-bounded so it does not match substrings
/// (e.g. `r"^tel$"` instead of `r"tel\b"`) — the dispatcher runs the regex
/// per-attribute rather than over a concatenated signal.
static FIELD_TABLE: &[(&str, ProfileExtractor)] = &[
    (r"\b(first.?name|given.?name|fname)\b", |p| {
        &p.personal.first_name
    }),
    (r"\b(last.?name|family.?name|surname|lname)\b", |p| {
        &p.personal.last_name
    }),
    (r"\b(e?mail|email)\b", |p| &p.personal.email),
    (r"\b(phone|mobile|cell|tel|telephone)\b", |p| &p.personal.phone),
    (r"\b(address.?1|street|addr1)\b", |p| &p.personal.address_line1),
    (
        r"\b(address.?2|apt|suite|addr2)\b",
        |p| &p.personal.address_line2,
    ),
    (r"\b(city|locality)\b", |p| &p.personal.city),
    (r"\b(state|region|province)\b", |p| &p.personal.region),
    (r"\b(zip|postal)\b", |p| &p.personal.postal_code),
    (r"\b(country)\b", |p| &p.personal.country),
    (r"\b(linkedin)\b", |p| &p.personal.linkedin_url),
    (r"\b(github)\b", |p| &p.personal.github_url),
    (
        r"\b(portfolio|website|homepage)\b",
        |p| &p.personal.portfolio_url,
    ),
];

fn compiled_field_table() -> &'static [(Regex, ProfileExtractor)] {
    static TABLE: OnceLock<Vec<(Regex, ProfileExtractor)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        FIELD_TABLE
            .iter()
            .map(|(pattern, extractor)| (Regex::new(pattern).unwrap(), *extractor))
            .collect()
    })
}

fn re_full_name() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(full.?name)\b").unwrap())
}

/// Local profile-based dispatcher (Tier 0). Matches form fields against the
/// user's profile using regex heuristics — no LLM required.
pub struct LocalDispatcher {
    profile: Arc<RwLock<Profile>>,
}

impl LocalDispatcher {
    pub fn new(profile: Arc<RwLock<Profile>>) -> Self {
        Self { profile }
    }

    /// Attempt to match a form field to a value from the user's profile.
    ///
    /// Returns `None` if the field is sensitive, unmatched, or the extracted
    /// value is empty.
    pub fn route(&self, field: &FieldDescription) -> Option<PreFilledValue> {
        if sanitize::is_sensitive_field(field) {
            return None;
        }

        let parts = self.build_signal_parts(field);
        let profile = self.profile.read().ok()?;

        // Special case: "full name" combines first + last.
        if parts.iter().any(|p| re_full_name().is_match(p)) {
            let first = &profile.personal.first_name;
            let last = &profile.personal.last_name;
            let value = match (!first.is_empty(), !last.is_empty()) {
                (true, true) => format!("{first} {last}"),
                (true, false) => first.clone(),
                (false, true) => last.clone(),
                (false, false) => return None,
            };
            if value.is_empty() {
                return None;
            }
            return Some(PreFilledValue {
                value,
                confidence: 1.0,
                source: ValueSource::Resume,
                reasoning: "matched local profile field: full name".into(),
                field_id: field.field_id.clone(),
            });
        }

        for (re, extractor) in compiled_field_table() {
            if parts.iter().any(|p| re.is_match(p)) {
                let raw = extractor(&profile);
                if raw.is_empty() {
                    return None;
                }
                return Some(PreFilledValue {
                    value: raw.to_string(),
                    confidence: 1.0,
                    source: ValueSource::Resume,
                    reasoning: "matched local profile field".into(),
                    field_id: field.field_id.clone(),
                });
            }
        }

        None
    }

    #[allow(clippy::unused_self)]
    fn build_signal_parts(&self, field: &FieldDescription) -> Vec<String> {
        let mut parts = Vec::with_capacity(3);
        parts.push(field.field_id.to_lowercase());
        if !field.label.is_empty() {
            parts.push(field.label.to_lowercase());
        }
        if let Some(ref ac) = field.autocomplete {
            if !ac.is_empty() {
                parts.push(ac.to_lowercase());
            }
        }
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    fn test_profile() -> Profile {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.last_name = "Lovelace".into();
        profile.personal.email = "ada@example.com".into();
        profile.personal.phone = "+1-555-0101".into();
        profile.personal.address_line1 = "123 Main St".into();
        profile.personal.address_line2 = "Apt 4".into();
        profile.personal.city = "London".into();
        profile.personal.region = "England".into();
        profile.personal.postal_code = "SW1A 1AA".into();
        profile.personal.country = "UK".into();
        profile.personal.linkedin_url = "https://linkedin.com/in/adalovelace".into();
        profile.personal.github_url = "https://github.com/adalovelace".into();
        profile.personal.portfolio_url = "https://adalovelace.dev".into();
        profile
    }

    fn make_dispatcher() -> LocalDispatcher {
        LocalDispatcher::new(Arc::new(RwLock::new(test_profile())))
    }

    fn make_field(field_id: &str, label: &str) -> FieldDescription {
        FieldDescription {
            field_id: field_id.into(),
            label: label.into(),
            input_type: "text".into(),
            selector: "#x".into(),
            context_text: String::new(),
            required: false,
            current_value_hash: None,
            autocomplete: None,
            options: vec![],
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn matches_first_name() {
        let d = make_dispatcher();
        let field = make_field("firstName", "First Name");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "Ada");
        assert_eq!(result.source, ValueSource::Resume);
        assert_eq!(result.confidence, 1.0);
    }
    #[test]
    fn matches_last_name() {
        let d = make_dispatcher();
        let field = make_field("lastName", "Last Name");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "Lovelace");
    }

    #[test]
    fn matches_full_name() {
        let d = make_dispatcher();
        let field = make_field("fullName", "Full Name");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "Ada Lovelace");
    }

    #[test]
    fn matches_email() {
        let d = make_dispatcher();
        let field = make_field("email", "Email Address");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "ada@example.com");
    }

    #[test]
    fn matches_phone() {
        let d = make_dispatcher();
        let field = make_field("phone", "Phone Number");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "+1-555-0101");
    }

    #[test]
    fn matches_address_line1() {
        let d = make_dispatcher();
        let field = make_field("address1", "Street Address");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "123 Main St");
    }

    #[test]
    fn matches_address_line2() {
        let d = make_dispatcher();
        let field = make_field("address2", "Apt/Suite");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "Apt 4");
    }

    #[test]
    fn matches_city() {
        let d = make_dispatcher();
        let field = make_field("city", "City");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "London");
    }

    #[test]
    fn matches_state() {
        let d = make_dispatcher();
        let field = make_field("state", "State/Region");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "England");
    }

    #[test]
    fn matches_zip() {
        let d = make_dispatcher();
        let field = make_field("zip", "Postal Code");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "SW1A 1AA");
    }

    #[test]
    fn matches_country() {
        let d = make_dispatcher();
        let field = make_field("country", "Country");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "UK");
    }

    #[test]
    fn matches_linkedin() {
        let d = make_dispatcher();
        let field = make_field("linkedin", "LinkedIn Profile");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "https://linkedin.com/in/adalovelace");
    }

    #[test]
    fn matches_github() {
        let d = make_dispatcher();
        let field = make_field("github", "GitHub Profile");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "https://github.com/adalovelace");
    }

    #[test]
    fn matches_portfolio() {
        let d = make_dispatcher();
        let field = make_field("portfolio", "Portfolio Website");
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "https://adalovelace.dev");
    }

    #[test]
    fn unmatched_field_returns_none() {
        let d = make_dispatcher();
        let field = make_field("favorite_color", "Favorite Color");
        assert!(d.route(&field).is_none());
    }

    #[test]
    fn sensitive_field_returns_none() {
        let d = make_dispatcher();
        let mut field = make_field("password", "Password");
        field.input_type = "password".into();
        assert!(d.route(&field).is_none());
    }

    #[test]
    fn empty_profile_value_returns_none() {
        let profile = Profile::default();
        let d = LocalDispatcher::new(Arc::new(RwLock::new(profile)));
        let field = make_field("firstName", "First Name");
        assert!(d.route(&field).is_none());
    }

    #[test]
    fn matches_autocomplete_signal() {
        let d = make_dispatcher();
        // field_id "email_work" starts with "email" → matches ^e?mail
        let mut field = make_field("email_work", "Work Email");
        field.autocomplete = Some("work-email".into());
        let result = d.route(&field).unwrap();
        assert_eq!(result.value, "ada@example.com");
    }

    #[test]
    fn rejects_substring_match_in_label() {
        let d = make_dispatcher();
        // "Headphone Review" contains the substring "phone" but is not a
        // phone field — the regex must not match a substring of the label.
        let field = make_field("review_topic", "Headphone Review");
        assert!(d.route(&field).is_none());
    }

    #[test]
    fn rejects_substring_match_in_field_id() {
        let d = make_dispatcher();
        // "Motel reservation" contains "tel" — the regex must not match.
        let field = make_field("reservation_type", "Motel reservation");
        assert!(d.route(&field).is_none());
    }

    #[test]
    fn rejects_state_substring_in_field_id() {
        let d = make_dispatcher();
        // "real_estate_status" contains the substring "state" — must not match.
        let field = make_field("real_estate_status", "Property status");
        assert!(d.route(&field).is_none());
    }
}
