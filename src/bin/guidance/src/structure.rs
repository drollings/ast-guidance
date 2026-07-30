use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use content_node::doc_node::DocumentContentNode;
use content_node::file_node::FileContentNode;
use content_node::node::{ContentNode, LodLevel};
use content_node::source_node::SourceCodeContentNode;
use fluent_types::{FileType, GuidanceDoc};
use guidance_core::sync::json_store::load_guidance;
use ptree::{PrintConfig, TreeBuilder};

const HEADER: &str = r#"# AST-Guidance Project Structure

A fast, lightweight code navigation and orchestration framework friendly to
human and human-in-the-loop LLM agentic software engineering.  It is based
on enriched AST, and uses optional AI for documentation which is cached,
idempotent, and upcycled for lightweight searches and local agentic
intelligence.

## Quick Navigation (Coding Assistants)

| Purpose | File | Use When |
|---------|------|----------|
| **Find related code** | `make query QUERY="search terms"` | Searching for code |
| **Check Implementation** | `make explore QUERY="search terms"` | Before implementing anything |
| **Understand patterns** | `doc/capabilities/*.md` | Implementation examples + patterns |
| **Find existing code** | `mcp_grep` or `mcp_lsp_find_references` | Searching for implementations |

## **Attention**: Skills needed to understand files

Skills are referenced per-file in comments below.  The lookup path for the skills is: 
`{guidance_dir}/skills/{skill}/SKILL.md`

So if you find a file you're looking for named file.rs:
`file.rs      # [zig-current, gof-patterns] Summary of files' contents` , 
Then you you must read

```
{guidance_dir}/skills/zig-current/SKILL.md
{guidance_dir}/skills/gof-patterns/SKILL.md
```

---

## Directory Tree (Git-Tracked Files Only)

```
"#;

enum Node {
    Dir(BTreeMap<String, Node>),
    File { rel_path: String },
}

fn insert_path(root: &mut BTreeMap<String, Node>, components: &[&str], rel_path: &str) {
    if components.is_empty() {
        return;
    }
    let first = components[0].to_string();
    if components.len() == 1 {
        root.entry(first).or_insert(Node::File {
            rel_path: rel_path.to_string(),
        });
    } else {
        let child = root
            .entry(first)
            .or_insert_with(|| Node::Dir(BTreeMap::new()));
        if let Node::Dir(children) = child {
            insert_path(children, &components[1..], rel_path);
        }
    }
}

fn build_ptree(node: &Node, builder: &mut TreeBuilder, json_dir: &Path, skills: &[String]) {
    match node {
        Node::Dir(children) => {
            for (name, child) in children {
                match child {
                    Node::Dir(_) => {
                        builder.begin_child(format!("{name}/"));
                        build_ptree(child, builder, json_dir, skills);
                        builder.end_child();
                    }
                    Node::File { rel_path } => {
                        let annotation = annotate_file(rel_path, name, json_dir, skills);
                        builder.add_empty_child(annotation);
                    }
                }
            }
        }
        Node::File { rel_path } => {
            let annotation = annotate_file(rel_path, "", json_dir, skills);
            builder.add_empty_child(annotation);
        }
    }
}

fn is_annotatable(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let dotted = format!(".{ext}");
    matches!(
        FileType::from_extension(&dotted),
        FileType::Source | FileType::Markdown
    )
}

fn annotate_file(rel_path: &str, name: &str, json_dir: &Path, skills: &[String]) -> String {
    let display_name = if name.is_empty() {
        Path::new(rel_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| rel_path.to_string())
    } else {
        name.to_string()
    };

    let path = PathBuf::from(rel_path);

    let mut desc = String::new();
    let mut file_skills: Vec<String> = Vec::new();

    let json_path = json_dir.join("src").join(rel_path).with_extension("json");
    if let Some(doc) = load_guidance(&json_path).ok().flatten() {
        let (inode, hash) = file_metadata(&path);
        let file_node = FileContentNode::new(path, inode, hash);
        let source_node = SourceCodeContentNode::new(file_node).with_ast(doc.clone());
        if let Some(summary) = source_node.lod(LodLevel::Summary) {
            if !summary.is_empty() {
                desc = first_line(summary);
            }
        }
        file_skills = extract_skills(&doc, skills);
    } else if is_annotatable(&path) && path.exists() {
        if let Ok(content) = common_core::io::read_to_string_err(&path) {
            let (inode, hash) = file_metadata(&path);
            let file_node = FileContentNode::new(path, inode, hash);
            let doc_node = DocumentContentNode::new(file_node, &content);
            if let Some(summary) = doc_node.lod(LodLevel::Tiny) {
                if !summary.is_empty() {
                    desc = first_line(summary);
                }
            }
        }
    }

    desc = truncate(&desc, 40);

    if !file_skills.is_empty() || !desc.is_empty() {
        let tag_str = if file_skills.is_empty() {
            String::new()
        } else {
            format!("[{}] ", file_skills.join(", "))
        };
        format!("{display_name}  # {tag_str}{desc}")
    } else {
        display_name
    }
}

fn extract_skills(doc: &GuidanceDoc, available: &[String]) -> Vec<String> {
    let mut matched: Vec<String> = Vec::new();
    for skill in &doc.skills {
        let skill_name = Path::new(skill.ref_path.as_str())
            .parent()
            .and_then(|p| p.file_name())
            .or_else(|| Path::new(skill.ref_path.as_str()).file_stem())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| skill.ref_path.to_string());
        if available.iter().any(|a| a == &skill_name) && !matched.contains(&skill_name) {
            matched.push(skill_name);
        }
    }
    matched
}

fn file_metadata(path: &Path) -> (u64, [u8; 32]) {
    let meta = std::fs::metadata(path).ok();
    let inode = meta
        .as_ref()
        .map(|m| {
            #[cfg(unix)]
            {
                std::os::unix::fs::MetadataExt::ino(m)
            }
            #[cfg(not(unix))]
            {
                0
            }
        })
        .unwrap_or(0);
    let content = std::fs::read(path).unwrap_or_default();
    let hash = common_core::hash::blake3_hash(&content);
    (inode, hash)
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or(s).to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else if max > 3 {
        let idx = s.floor_char_boundary(max - 3);
        format!("{}...", &s[..idx])
    } else {
        s[..s.floor_char_boundary(max)].to_string()
    }
}

pub fn generate(guidance_dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let json_dir = guidance_dir;

    let files = get_git_tracked_files()?;
    let skills = get_available_skills(guidance_dir);

    let mut root: BTreeMap<String, Node> = BTreeMap::new();
    for file_path in &files {
        // Skip hidden files/dirs (names starting with '.')
        if file_path.split('/').any(|c| c.starts_with('.')) {
            continue;
        }
        let components: Vec<&str> = file_path.split('/').collect();
        insert_path(&mut root, &components, file_path);
    }

    let mut builder = TreeBuilder::new(".".to_string());
    for (name, node) in &root {
        match node {
            Node::Dir(_) => {
                builder.begin_child(format!("{name}/"));
                build_ptree(node, &mut builder, json_dir, &skills);
                builder.end_child();
            }
            Node::File { rel_path } => {
                let annotation = annotate_file(rel_path, name, json_dir, &skills);
                builder.add_empty_child(annotation);
            }
        }
    }
    let tree = builder.build();

    let mut buf: Vec<u8> = Vec::new();
    let config = PrintConfig {
        indent: 4,
        ..PrintConfig::default()
    };
    ptree::write_tree_with(&tree, &mut buf, &config)?;
    let tree_str = String::from_utf8(buf)?;

    let output = format!("{HEADER}{tree_str}```\n");
    Ok(output)
}

fn get_git_tracked_files() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let output = common_core::shell::run_capture(&["git", "ls-files"])?;
    if !output.success {
        return Err("git ls-files failed".into());
    }
    let mut files: Vec<String> = output
        .stdout
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    files.sort();
    Ok(files)
}

/// List available skill directories. Uses a flat (non-recursive) `read_dir`
/// intentionally: this enumerates subdirectory names, not files, which is
/// outside the scope of `common_core::walk::walk_files`.
fn get_available_skills(guidance_dir: &Path) -> Vec<String> {
    let skills_dir = guidance_dir.join("skills");
    let mut skills = Vec::new();
    if skills_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&skills_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        skills.push(name.to_string());
                    }
                }
            }
        }
    }
    skills.sort();
    skills
}
