use common_core::walk::*;


#[test]
fn collects_source_files() {
        let tmp = tempfile::tempdir().unwrap();
        make_tree(
            tmp.path(),
            &["a.zig", "b.py", "c.rs", "d.txt", "sub/e.zon"],
            &[],
        );

        let mut found = Vec::new();
        walk_files(tmp.path(), SOURCE_EXTENSIONS, |p| {
            found.push(p.file_name().unwrap().to_string_lossy().to_string());
        });
        found.sort();
        assert_eq!(found, vec!["a.zig", "b.py", "c.rs", "e.zon"]);
}

#[test]
fn skips_hidden_and_target() {
        let tmp = tempfile::tempdir().unwrap();
        make_tree(
            tmp.path(),
            &["a.zig", ".hidden/b.zig", "target/c.zig", "fixtures/d.zig"],
            &[],
        );

        let mut found = Vec::new();
        walk_files(tmp.path(), SOURCE_EXTENSIONS, |p| {
            found.push(p.file_name().unwrap().to_string_lossy().to_string());
        });
        assert_eq!(found, vec!["a.zig"]);
}

#[test]
fn collects_extensions() {
        let tmp = tempfile::tempdir().unwrap();
        make_tree(
            tmp.path(),
            &["a.zig", "b.py", "c.rs", "d.txt", "sub/e.zon"],
            &[],
        );

        let exts = collect_extensions(&[tmp.path().to_path_buf()]);
        assert!(exts.contains(".zig"));
        assert!(exts.contains(".py"));
        assert!(exts.contains(".rs"));
        assert!(exts.contains(".zon"));
        assert!(exts.contains(".txt"));
}
