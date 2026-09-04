use super::*;

#[test]
fn embed_error_display() {
    let err = EmbedError::NoApiKey;
    assert_eq!(format!("{err}"), "no API key provided");
}

#[test]
fn embed_error_unknown_provider() {
    assert_eq!(
        format!("{}", EmbedError::UnknownProvider("ollama".into())),
        "unknown provider: ollama"
    );
}

#[test]
fn embed_error_request_failed() {
    assert_eq!(
        format!("{}", EmbedError::RequestFailed("timeout".into())),
        "embedding request failed: timeout"
    );
}

#[test]
fn embed_error_unit_variants() {
    assert_eq!(format!("{}", EmbedError::InvalidApiUrl), "invalid API URL");
    assert_eq!(format!("{}", EmbedError::InsecureApiUrl), "insecure API URL");
    assert_eq!(format!("{}", EmbedError::SsrfBlockedUrl), "SSRF blocked URL");
}

#[test]
fn embed_error_parse() {
    assert_eq!(
        format!("{}", EmbedError::ParseError("bad json".into())),
        "parse error: bad json"
    );
}
