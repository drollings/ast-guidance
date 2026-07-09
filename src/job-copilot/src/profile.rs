use std::path::Path;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::error::CopilotError;

/// User profile loaded from a TOML file. Contains personal info, work history,
/// education, skills, and cover letter templates.
///
/// # TOML Schema
///
/// ```toml
/// [personal]
/// first_name = "Ada"
/// last_name = "Lovelace"
/// email = "ada@example.com"
/// phone = "+1-555-0101"
/// address_line1 = "123 Main St"
/// city = "London"
/// region = "England"
/// postal_code = "SW1A 1AA"
/// country = "UK"
/// linkedin_url = "https://linkedin.com/in/adalovelace"
/// github_url = "https://github.com/adalovelace"
///
/// [[work]]
/// company = "Analytical Engine Ltd"
/// title = "Mathematician"
/// start = "1842"
/// end = "1852"
/// bullets = ["First computer program"]
///
/// [[education]]
/// institution = "University of London"
/// degree = "BA"
/// field = "Mathematics"
/// graduation_year = 1833
///
/// skills = ["Rust", "Python", "Mathematics"]
///
/// [[cover_letter_templates]]
/// name = "default"
/// role_pattern = ".*"
/// body = "Dear Hiring Manager..."
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub personal: Personal,
    #[serde(default)]
    pub work: Vec<WorkEntry>,
    #[serde(default)]
    pub education: Vec<Education>,
    #[serde(default)]
    pub skills: Vec<String>,
    #[serde(default)]
    pub cover_letter_templates: Vec<CoverLetterTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Personal {
    #[serde(default)]
    pub first_name: String,
    #[serde(default)]
    pub last_name: String,
    #[serde(default)]
    pub email: String,
    #[serde(default)]
    pub phone: String,
    #[serde(default)]
    pub address_line1: String,
    #[serde(default)]
    pub address_line2: String,
    #[serde(default)]
    pub city: String,
    #[serde(default)]
    pub region: String,
    #[serde(default)]
    pub postal_code: String,
    #[serde(default)]
    pub country: String,
    #[serde(default)]
    pub linkedin_url: String,
    #[serde(default)]
    pub github_url: String,
    #[serde(default)]
    pub portfolio_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkEntry {
    pub company: String,
    pub title: String,
    pub start: String,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub bullets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Education {
    pub institution: String,
    pub degree: String,
    pub field: String,
    #[serde(default)]
    pub graduation_year: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverLetterTemplate {
    pub name: String,
    pub role_pattern: String,
    pub body: String,
}

impl Profile {
    /// Load a profile from a TOML file path.
    pub fn load_from_path(p: &Path) -> Result<Self, CopilotError> {
        let raw = common_core::io::read_to_string_err(p)?;
        let profile: Self = toml::from_str(&raw)?;
        Ok(profile)
    }

    /// Validate the profile. Returns `CopilotError::ProfileNotLoaded` on failure.
    pub fn validate(&self) -> Result<(), CopilotError> {
        if self.personal.first_name.is_empty() && self.personal.last_name.is_empty() {
            return Err(CopilotError::ProfileNotLoaded(
                "personal.first_name and personal.last_name are both empty".into(),
            ));
        }
        if !self.personal.email.is_empty() {
            let email_re = regex::Regex::new(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").unwrap();
            if !email_re.is_match(&self.personal.email) {
                return Err(CopilotError::ProfileNotLoaded(format!(
                    "personal.email is not a valid email: {}",
                    self.personal.email
                )));
            }
        }
        Ok(())
    }
}

/// Wrap a `Profile` in `Arc<RwLock<...>>` for shared access across dispatchers.
pub fn shared(profile: Profile) -> Arc<RwLock<Profile>> {
    Arc::new(RwLock::new(profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const FIXTURE: &str = r#"
skills = ["Rust", "Python", "Mathematics", "Algorithms"]

[personal]
first_name = "Ada"
last_name = "Lovelace"
email = "ada@example.com"
phone = "+1-555-0101"
address_line1 = "123 Main St"
city = "London"
region = "England"
postal_code = "SW1A 1AA"
country = "UK"
linkedin_url = "https://linkedin.com/in/adalovelace"
github_url = "https://github.com/adalovelace"
portfolio_url = "https://adalovelace.dev"

[[work]]
company = "Analytical Engine Ltd"
title = "Mathematician"
start = "1842"
end = "1852"
location = "London"
bullets = ["First computer program", "Algorithms"]

[[work]]
company = "University of London"
title = "Researcher"
start = "1835"
bullets = ["Mathematical analysis"]

[[education]]
institution = "University of London"
degree = "BA"
field = "Mathematics"
graduation_year = 1833

[[cover_letter_templates]]
name = "default"
role_pattern = ".*"
body = "Dear Hiring Manager,\n\nI am writing to express my interest..."
"#;

    #[test]
    fn roundtrip_fixture() {
        let profile: Profile = toml::from_str(FIXTURE).unwrap();
        let serialized = toml::to_string(&profile).unwrap();
        let reparsed: Profile = toml::from_str(&serialized).unwrap();
        assert_eq!(profile.personal.first_name, reparsed.personal.first_name);
        assert_eq!(profile.work.len(), reparsed.work.len());
        assert_eq!(profile.skills, reparsed.skills);
        assert_eq!(
            profile.cover_letter_templates.len(),
            reparsed.cover_letter_templates.len()
        );
    }

    #[test]
    fn load_from_path_with_fixture() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("profile.toml");
        std::fs::write(&path, FIXTURE).unwrap();
        let profile = Profile::load_from_path(&path).unwrap();
        assert_eq!(profile.personal.first_name, "Ada");
        assert_eq!(profile.personal.last_name, "Lovelace");
        assert_eq!(profile.work.len(), 2);
        assert_eq!(profile.education.len(), 1);
        assert_eq!(profile.skills.len(), 4);
    }

    #[test]
    fn validate_rejects_empty_name() {
        let profile = Profile::default();
        let err = profile.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("first_name") && msg.contains("last_name") && msg.contains("empty"),
            "got: {msg}"
        );
    }

    #[test]
    fn validate_rejects_invalid_email() {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.email = "not-an-email".into();
        let err = profile.validate().unwrap_err();
        assert!(format!("{err}").contains("not a valid email"));
    }

    #[test]
    fn validate_accepts_valid_profile() {
        let mut profile = Profile::default();
        profile.personal.first_name = "Ada".into();
        profile.personal.email = "ada@example.com".into();
        profile.validate().unwrap();
    }

    #[test]
    fn shared_returns_arc_rwlock() {
        let profile = Profile::default();
        let shared = shared(profile);
        let guard = shared.read().unwrap();
        assert!(guard.personal.first_name.is_empty());
    }
}
