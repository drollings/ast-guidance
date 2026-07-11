//! Human-readable output: `format_json`, `format_csv`, `format_size`/`parse_size`, `Column`/`Table`.

use std::fmt::Write as _;

/// Formats a JSON value as a pretty-printed string with 2-space indent.
/// The `_indent` parameter is accepted for API compatibility but only 2-space
/// indent is supported (matching `serde_json::to_string_pretty`).
pub fn format_json(value: &serde_json::Value, _indent: usize) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

pub fn format_csv(rows: &[serde_json::Value], fieldnames: Option<&[&str]>) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let keys: Vec<&str> = match fieldnames {
        Some(names) => names.to_vec(),
        None => rows[0]
            .as_object()
            .map(|obj| obj.keys().map(String::as_str).collect())
            .unwrap_or_default(),
    };
    let mut out = String::new();
    out.push_str(
        &keys
            .iter()
            .map(|k| csv_escape(k))
            .collect::<Vec<_>>()
            .join(","),
    );
    out.push('\n');
    for row in rows {
        let vals: Vec<String> = keys
            .iter()
            .map(|k| {
                row.get(k)
                    .map(|v| match v {
                        serde_json::Value::String(s) => csv_escape(s),
                        serde_json::Value::Null => String::new(),
                        other => csv_escape(&other.to_string()),
                    })
                    .unwrap_or_default()
            })
            .collect();
        out.push_str(&vals.join(","));
        out.push('\n');
    }
    out
}

/// Format a byte count as a human-readable string (e.g. "1.5 MB").
///
/// # Examples
///
/// ```
/// use common_core::format::format_size;
///
/// assert_eq!(format_size(0), "0 B");
/// assert_eq!(format_size(1023), "1023 B");
/// assert_eq!(format_size(1024), "1.0 KB");
/// assert_eq!(format_size(1_536), "1.5 KB");
/// assert_eq!(format_size(1_048_576), "1.0 MB");
/// assert_eq!(format_size(1_073_741_824), "1.0 GB");
/// ```
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

pub fn parse_size(size_str: &str) -> Option<u64> {
    let s = size_str.trim();
    if s.is_empty() {
        return None;
    }
    let (num_part, suffix) = s
        .chars()
        .partition::<String, _>(|c| c.is_ascii_digit() || *c == '.');
    let num: f64 = num_part.parse().ok()?;
    let multiplier = match suffix.trim().to_uppercase().as_str() {
        "B" | "" => 1,
        "KB" => 1024,
        "MB" => 1024 * 1024,
        "GB" => 1024 * 1024 * 1024,
        "TB" => 1024u64 * 1024 * 1024 * 1024,
        _ => return None,
    };
    Some((num * multiplier as f64) as u64)
}

#[derive(Debug, Clone)]
pub struct Column {
    pub header: String,
    pub key: String,
    pub width: usize,
    pub align_left: bool,
}

impl Column {
    pub fn effective_width(&self) -> usize {
        if self.width > 0 {
            self.width
        } else {
            self.header.len() + 2
        }
    }
}

pub struct Table {
    pub columns: Vec<Column>,
    pub rows: Vec<serde_json::Value>,
    pub title: String,
}

impl Table {
    pub fn new(columns: Vec<Column>, title: &str) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            title: title.to_string(),
        }
    }

    pub fn with_rows(&mut self, rows: Vec<serde_json::Value>) {
        self.rows = rows;
    }

    /// Render the table as an ASCII-formatted string.
    ///
    /// # Examples
    ///
    /// ```
    /// use common_core::format::{Column, Table};
    ///
    /// let cols = vec![
    ///     Column { header: "Name".into(), key: "name".into(), width: 10, align_left: true },
    ///     Column { header: "Count".into(), key: "count".into(), width: 8, align_left: false },
    /// ];
    /// let mut table = Table::new(cols, "Items");
    /// table.with_rows(vec![
    ///     serde_json::json!({"name": "alpha", "count": 3}),
    ///     serde_json::json!({"name": "beta", "count": 7}),
    /// ]);
    /// let rendered = table.render();
    /// assert!(rendered.contains("Items"));
    /// assert!(rendered.contains("alpha"));
    /// ```
    pub fn render(&self) -> String {
        if self.columns.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        if !self.title.is_empty() {
            out.push_str(&self.title);
            out.push('\n');
            out.push_str(&"-".repeat(self.title.len()));
            out.push('\n');
        }
        for col in &self.columns {
            let w = col.effective_width();
            if col.align_left {
                let _ = write!(out, " {:<w$}", col.header, w = w);
            } else {
                let _ = write!(out, " {:>w$}", col.header, w = w);
            }
        }
        out.push('\n');
        for row in &self.rows {
            for col in &self.columns {
                let val = row
                    .get(&col.key)
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .unwrap_or_default();
                let w = col.effective_width();
                if col.align_left {
                    let _ = write!(out, " {val:<w$}");
                } else {
                    let _ = write!(out, " {val:>w$}");
                }
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(500), "500 B");
    }

    #[test]
    fn format_size_kb() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
    }

    #[test]
    fn parse_size_plain_number() {
        assert_eq!(parse_size("1024"), Some(1024));
    }

    #[test]
    fn parse_size_kb() {
        assert_eq!(parse_size("1 KB"), Some(1024));
    }

    #[test]
    fn parse_size_mb() {
        assert_eq!(parse_size("2 MB"), Some(2 * 1024 * 1024));
    }

    #[test]
    fn format_json_empty_object() {
        let v = serde_json::json!({});
        let s = format_json(&v, 2);
        assert_eq!(s, "{}");
    }

    #[test]
    fn format_json_nested() {
        let v = serde_json::json!({"a": {"b": 1}});
        let s = format_json(&v, 2);
        assert!(s.contains("\"a\""));
        assert!(s.contains("\"b\""));
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
    fn format_json_uses_two_space_indent() {
        let v = serde_json::json!({"a": {"b": 1}});
        let s = format_json(&v, 4);
        assert!(s.contains("\"a\""));
        let lines: Vec<&str> = s.lines().collect();
        let indent_line = lines
            .iter()
            .find(|l| l.trim_start().starts_with('"'))
            .unwrap();
        assert!(indent_line.starts_with("  "));
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
    fn format_size_large() {
        assert_eq!(format_size(1_073_741_824), "1.0 GB");
        assert_eq!(format_size(1_099_511_627_776), "1.0 TB");
    }

    #[test]
    fn parse_size_large_and_invalid() {
        assert_eq!(parse_size("1 GB"), Some(1_073_741_824));
        assert_eq!(parse_size("1 TB"), Some(1_099_511_627_776));
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
}
