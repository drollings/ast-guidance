use super::*;

#[test]
fn router_refine_policy_roundtrip() {
    let dto = RouterRefinePolicy {
        mode: RouterRefineMode::OnUncertain,
        min_overall: 0.5,
        min_role_coverage: 0.4,
        refine_on_ties: false,
        min_token_score: 0.6,
        refine_on_unresolved_critical_role: false,
        refine_on_unresolved_propn: true,
        refine_on_collision_note: false,
        unresolved_token_threshold: 0.25,
    };
    let json = serde_json::to_string(&dto).expect("ser");
    let back: RouterRefinePolicy = serde_json::from_str(&json).expect("de");
    assert_eq!(back.mode, dto.mode);
    assert!((back.min_overall - dto.min_overall).abs() < 1e-9);
    assert!((back.unresolved_token_threshold - dto.unresolved_token_threshold).abs() < 1e-9);
    // spacy round-trip
    let spacy: spacy_rs::RefinePolicy = dto.into();
    let dto2: RouterRefinePolicy = spacy.into();
    let json2 = serde_json::to_string(&dto2).expect("ser2");
    let back2: RouterRefinePolicy = serde_json::from_str(&json2).expect("de2");
    assert_eq!(back2.mode, dto.mode);
    assert!((back2.unresolved_token_threshold - dto.unresolved_token_threshold).abs() < 1e-9);
}

#[test]
fn default_matches_spacy_default() {
    let dto = RouterRefinePolicy::default();
    let spacy = spacy_rs::RefinePolicy::default();
    assert_eq!(dto.mode as u8, spacy.mode as u8);
    assert!((dto.min_overall - spacy.min_overall).abs() < 1e-9);
    assert!((dto.unresolved_token_threshold - spacy.unresolved_token_threshold).abs() < 1e-9);
    assert_eq!(dto.refine_on_ties, spacy.refine_on_ties);
}
