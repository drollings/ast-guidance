use std::sync::{Arc, RwLock};

use regex::Regex;

use crate::profile::Profile;
use crate::sanitize;
use crate::schema::{FieldDescription, PreFilledValue, ValueSource};

/// Profile extractor function type.
type ProfileExtractor = fn(&Profile) -> &str;

/// Static table of (regex, extractor-fn) pairs for local field matching.
///
/// Each entry maps a regex pattern to a function that extracts a value from
/// the user's profile. The signal passed to the regex is the lowercased
/// concatenation of `field_id`, `label`, and `autocomplete`.
static FIELD_TABLE: &[(&str, ProfileExtractor)] = &[
    (r"first.?name|given.?name", |p| &p.personal.first_name),
    (r"last.?name|family.?name|surname", |p| {
        &p.personal.last_name
    }),
    (r"^e?mail", |p| &p.personal.email),
    (r"phone|mobile|cell|tel\b", |p| &p.personal.phone),
    (r"address.?1|street|addr1", |p| &p.personal.address_line1),
    (r"address.?2|apt|suite|addr2", |p| &p.personal.address_line2),
    (r"city|locality", |p| &p.personal.city),
    (r"state|region|province", |p| &p.personal.region),
    (r"zip|postal", |p| &p.personal.postal_code),
    (r"country", |p| &p.personal.country),
    (r"linkedin", |p| &p.personal.linkedin_url),
    (r"github", |p| &p.personal.github_url),
    (r"portfolio|website|homepage", |p| &p.personal.portfolio_url),
];

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

        let signal = self.build_signal(field);
        let profile = self.profile.read().ok()?;

        // Special case: "full name" combines first + last.
        if Regex::new(r"full.?name")
            .ok()
            .is_some_and(|re| re.is_match(&signal))
        {
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

        for &(pattern, extractor) in FIELD_TABLE {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(&signal) {
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
        }

        None
    }

    #[allow(clippy::unused_self)]
    fn build_signal(&self, field: &FieldDescription) -> String {
        let mut signal = field.field_id.to_lowercase();
        signal.push_str(&field.label.to_lowercase());
        if let Some(ref ac) = field.autocomplete {
            signal.push_str(&ac.to_lowercase());
        }
        signal
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
}
