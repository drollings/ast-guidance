use common_core::format::*;


#[test]
fn parse_size_various_units() {
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("1 KB"), Some(1024));
        assert_eq!(parse_size("2 MB"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size("1 GB"), Some(1_073_741_824));
        assert_eq!(parse_size("1 TB"), Some(1_099_511_627_776));
}

#[test]
fn format_json_empty_object() {
        let v = serde_json::json!({});
        let s = format_json(&v, 2);
        assert_eq!(s, "{}");
}

#[test]
fn format_json_nested_and_indent() {
        let v = serde_json::json!({"a": {"b": 1}});
        let s = format_json(&v, 2);
        assert!(s.contains("\"a\""));
        assert!(s.contains("\"b\""));
        let s4 = format_json(&v, 4);
        let indent_line = s4
                .lines()
                .find(|l| l.trim_start().starts_with('"'))
                .unwrap();
        assert!(indent_line.starts_with("  "));
}

#[test]
fn format_csv_basic() {
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(
            r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#,
        )
        .unwrap();
        let csv = format_csv(&rows, Some(&["name", "age"]));
        assert!(csv.contains("Alice"));
        assert!(csv.contains("Bob"));
        assert!(csv.contains("name,age"));
}

#[test]
fn format_csv_empty_rows() {
        let csv = format_csv(&[], Some(&["name"]));
        assert_eq!(csv, "");
}

#[test]
fn format_csv_special_chars() {
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(
            r#"[{"name": "Alice, Inc.", "note": "line1\nline2"}]"#,
        )
        .unwrap();
        let csv = format_csv(&rows, Some(&["name"]));
        assert!(csv.contains("Alice, Inc."));
}



#[test]
fn format_csv_auto_fieldnames() {
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(r#"[{"x": 1, "y": 2}]"#).unwrap();
        let csv = format_csv(&rows, None);
        assert!(csv.starts_with("x,y"));
        assert!(csv.contains("1,2"));
}

#[test]
fn format_csv_null_value() {
        let rows =
            serde_json::from_str::<Vec<serde_json::Value>>(r#"[{"name": "Alice", "extra": null}]"#)
                .unwrap();
        let csv = format_csv(&rows, Some(&["name", "extra"]));
        assert!(csv.contains("Alice,"));
}

#[test]
fn parse_size_invalid_returns_none() {
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("bogus"), None);
}

#[test]
fn column_effective_width() {
        let col = Column {
            header: "Name".into(),
            key: "name".into(),
            width: 0,
            align_left: true,
        };
        assert_eq!(col.effective_width(), 6);
        let col = Column {
            header: "Name".into(),
            key: "name".into(),
            width: 20,
            align_left: false,
        };
        assert_eq!(col.effective_width(), 20);
}

#[test]
fn table_render_empty() {
        let table = Table::new(vec![], "");
        assert_eq!(table.render(), "");
}

#[test]
fn table_render_with_title() {
        let columns = vec![Column {
            header: "Name".into(),
            key: "name".into(),
            width: 0,
            align_left: true,
        }];
        let mut table = Table::new(columns, "People");
        let rows = serde_json::from_str::<Vec<serde_json::Value>>(
            r#"[{"name": "Alice"}, {"name": "Bob"}]"#,
        )
        .unwrap();
        table.with_rows(rows);
        let rendered = table.render();
        assert!(rendered.starts_with("People"));
        assert!(rendered.contains("Name"));
        assert!(rendered.contains("Alice"));
        assert!(rendered.contains("Bob"));
}

#[test]
fn format_size_roundtrip() {
        let sizes: &[(u64, &str)] = &[
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.0 KB"),
            (1_048_576, "1.0 MB"),
            (1_073_741_824, "1.0 GB"),
            (1_099_511_627_776, "1.0 TB"),
        ];
        for &(bytes, expected) in sizes {
            assert_eq!(format_size(bytes), expected, "format_size({bytes})");
        }
}
