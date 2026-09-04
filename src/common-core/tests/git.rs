use common_core::git::*;


#[test]
fn test_diff_staged_no_repo() {
        let dir = tempfile::tempdir().expect("temp dir");
        let result = diff_staged(dir.path());
        assert!(result.is_err());
}

#[test]
fn test_commit_no_repo() {
        let dir = tempfile::tempdir().expect("temp dir");
        let result = commit(dir.path(), "test");
        assert!(result.is_ok());
        assert!(!result.unwrap());
}

#[test]
fn test_rev_parse_head_no_repo() {
        let dir = tempfile::tempdir().expect("temp dir");
        let result = rev_parse_head(dir.path());
        assert!(result.is_err());
}
