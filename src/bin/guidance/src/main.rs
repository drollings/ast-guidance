use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use common_core::ensure_dir_or_panic;
use common_core::shell::run_command;
use fluent_concurrency::pool::ResultPoolError;
use guidance_core::config;
use guidance_core::memory::MemoryBridge;
use guidance_core::runtime;
use guidance_core::sync::json_store::walk_guidance_docs;
use guidance_core::sync_engine::SyncEngine;
use guidance_core::walk;
use search_vector::GuidanceDb;
use time::OffsetDateTime;

use notify::{Config as NotifyConfig, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc;

mod benchmark;
mod commit;
mod editor;
mod mcp;
mod structure;

#[derive(Parser)]
#[command(
    name = "guidance",
    about = "AST-guided vector search & edge AI orchestrator"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(global = true, long)]
    debug: bool,

    #[arg(global = true, long)]
    show_prompts: bool,
}

#[derive(Subcommand)]
enum Commands {
    Explain {
        query: String,

        #[arg(long, default_value = ".guidance")]
        guidance: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,

        #[arg(short = 'w', long, default_value = ".")]
        workspace: String,

        #[arg(short = 'l', long, default_value_t = 10)]
        limit: usize,

        #[arg(long)]
        no_llm: bool,

        #[arg(long, default_value = "auto")]
        filter: String,
    },
    Test,
    Telemetry {
        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,

        #[arg(long)]
        reset: bool,
    },
    CacheStats {
        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,
    },
    Init {
        #[arg(default_value = ".")]
        dir: String,

        #[arg(short = 'g', long, default_value = ".guidance")]
        guidance_dir: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,
    },
    Sync {
        #[arg(short, long)]
        file: Option<String>,

        #[arg(long)]
        scan: Option<String>,

        #[arg(short = 'w', long, default_value = ".")]
        workspace: String,

        #[arg(long, default_value = ".guidance")]
        json_dir: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,

        #[arg(long)]
        force: bool,

        #[arg(long)]
        no_db: bool,

        #[arg(long)]
        dry_run: bool,

        #[arg(long)]
        verbose: bool,

        #[arg(long)]
        watch: bool,

        #[arg(long, default_value_t = 500)]
        watch_debounce_ms: u64,
    },
    Status {
        #[arg(short = 'g', long, default_value = ".guidance")]
        guidance_dir: String,
    },
    Clean {
        #[arg(long, default_value = ".guidance")]
        json_dir: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,
    },
    Commit {
        #[arg(long)]
        dry_run: bool,

        #[arg(long)]
        debug: bool,

        #[arg(long)]
        force: bool,
    },
    Check {
        #[arg(short = 'w', long, default_value = ".")]
        workspace: String,
    },
    Todo,
    Diary {
        #[arg(default_value = "")]
        text_or_path: String,
    },
    Benchmark {
        query: Option<String>,

        #[arg(short = 'g', long, default_value = ".guidance")]
        guidance: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,

        #[arg(short = 'w', long, default_value = ".")]
        workspace: String,

        #[arg(short = 'n', long)]
        num: Option<usize>,

        #[arg(long)]
        no_llm: bool,

        #[arg(short = 'v', long)]
        verbose: bool,

        #[arg(long)]
        api_url: Option<String>,

        #[arg(short = 'm', long)]
        model: Option<String>,

        #[arg(long, default_value_t = 300)]
        timeout: u64,

        #[arg(long, default_value_t = 2)]
        concurrency: usize,
    },
    Structure {
        #[arg(long, default_value = ".guidance")]
        json_dir: String,
    },
    Health {
        #[arg(short = 'w', long, default_value = ".")]
        workspace: String,

        #[arg(long, default_value_t = 30)]
        min_age: u32,

        #[arg(long, default_value = "ai")]
        format: String,

        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,
    },
    Mcp {
        #[arg(short = 'o', long, default_value = ".guidance.db")]
        db: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.debug {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    }

    // Observes total command time. Per-unit observation is wired via
    // `Instrumented::with_metrics` in the browser-copilot handler. The
    // histogram here captures the outer command dispatch duration.
    let cmd_histogram = std::sync::Arc::new(common_core::LatencyHistogram::new());
    let cmd_start = std::time::Instant::now();

    match &cli.command {
        Commands::Explain {
            query,
            guidance,
            db,
            workspace,
            limit,
            no_llm,
            filter,
        } => {
            let memory = guidance_core::memory::init_memory_bridge();
            cmd_explain(
                query,
                guidance,
                db,
                workspace,
                *limit,
                *no_llm,
                filter,
                memory.as_ref(),
            )
            .await;
        }
        Commands::Test => cmd_test(),
        Commands::Telemetry { db, .. } => cmd_db_stats(db, "Telemetry stats"),
        Commands::CacheStats { db } => cmd_db_stats(db, "Cache statistics"),
        Commands::Init {
            dir,
            guidance_dir: _,
            db: _,
        } => cmd_init(dir),
        Commands::Sync {
            file,
            scan,
            workspace,
            json_dir,
            db,
            force,
            no_db,
            dry_run,
            verbose,
            watch,
            watch_debounce_ms,
        } => {
            cmd_sync(
                file.as_deref(),
                scan.as_deref(),
                workspace,
                json_dir,
                db,
                *force,
                *no_db,
                *dry_run,
                *verbose,
                *watch,
                *watch_debounce_ms,
            )
            .await;
        }
        Commands::Status { guidance_dir } => cmd_status(guidance_dir),
        Commands::Clean { json_dir, db } => cmd_clean(json_dir, db),
        Commands::Commit {
            dry_run,
            debug,
            force,
        } => cmd_commit(*dry_run, *debug, *force).await,
        Commands::Check { workspace } => cmd_check(workspace),
        Commands::Todo => cmd_todo(),
        Commands::Diary { text_or_path } => cmd_diary(text_or_path),
        Commands::Benchmark {
            query,
            guidance,
            db,
            workspace,
            num,
            no_llm,
            verbose,
            api_url,
            model,
            timeout,
            concurrency,
        } => {
            let config = benchmark::BenchmarkConfig::from_cli(
                query.clone(),
                guidance,
                db,
                workspace,
                *num,
                *no_llm,
                *verbose,
                cli.debug,
                cli.show_prompts,
                api_url.clone(),
                model.clone(),
                *timeout,
                *concurrency,
            );
            if let Err(e) = benchmark::run_benchmark(config).await {
                eprintln!("benchmark error: {e}");
                std::process::exit(1);
            }
        }
        Commands::Structure { json_dir } => cmd_structure(json_dir),
        Commands::Health {
            workspace,
            min_age,
            format,
            db,
        } => {
            cmd_health(workspace, *min_age, format, db);
        }
        Commands::Mcp { db } => cmd_mcp(db),
    }

    cmd_histogram.observe_duration(cmd_start);
    if cli.debug && cmd_histogram.count() > 0 {
        eprintln!(
            "[telemetry] command latency: count={} sum_ms={} p50={}ms p99={}ms",
            cmd_histogram.count(),
            cmd_histogram.sum_ms(),
            cmd_histogram.estimate_percentile(50.0),
            cmd_histogram.estimate_percentile(99.0),
        );
    }
}

fn load_project_config(workspace: &Path) -> config::ProjectConfig {
    config::load_config(workspace).unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
async fn cmd_explain(
    query: &str,
    guidance_dir: &str,
    db_path: &str,
    _workspace: &str,
    limit: usize,
    _no_llm: bool,
    _filter: &str,
    memory: Option<&MemoryBridge>,
) {
    let gdir = PathBuf::from(guidance_dir);
    let db = PathBuf::from(db_path);

    // Pre-fetch memory context for injection into the system prompt
    let memory_context = if let Some(bridge) = memory {
        bridge.prefetch_context(query).await
    } else {
        String::new()
    };

    let mut results: Vec<search_vector::db::SearchResult> = Vec::new();

    if db.exists() {
        if let Ok(gdb) = GuidanceDb::open(&db) {
            if let Ok(hybrid) = gdb.hybrid_search(query, None, limit) {
                results = hybrid;
            }
        }
    }

    if results.is_empty() {
        let src_dir = gdir.join("src");
        if src_dir.is_dir() {
            let lower_query = query.to_lowercase();
            let tokens: Vec<&str> = query.split_whitespace().collect();
            collect_json_results(&src_dir, &lower_query, &tokens, &mut results);
        }
    }

    results.truncate(limit);

    println!("## Explain: {query}");
    if !memory_context.is_empty() {
        println!();
        println!("### Memory Context");
        println!("{memory_context}");
    }
    println!();
    if results.is_empty() {
        println!("No results found.");
        return;
    }
    println!("| Name | Source | Score |");
    println!("|------|--------|-------|");
    for r in &results {
        println!("| {} | {} | {:.2} |", r.name, r.source, r.similarity);
    }

    // Sync turn with memory plugin after synthesis
    if let Some(bridge) = memory {
        let assistant_output = format!(
            "Explain: {query}\n\nResults:\n{}",
            results
                .iter()
                .map(|r| format!("- {} ({})", r.name, r.source))
                .collect::<Vec<_>>()
                .join("\n")
        );
        bridge.sync_turn(query, &assistant_output).await;
    }
}

fn collect_json_results(
    dir: &Path,
    lower_query: &str,
    tokens: &[&str],
    results: &mut Vec<search_vector::db::SearchResult>,
) {
    for (_path, doc) in walk_guidance_docs(dir) {
        for member in &doc.members {
            let name_lower = member.name.as_str().to_lowercase();
            let sig_lower = member
                .signature
                .as_ref()
                .map(|s| s.as_str().to_lowercase())
                .unwrap_or_default();
            let comment_lower = member
                .comment
                .as_ref()
                .map(|c| c.as_str().to_lowercase())
                .unwrap_or_default();

            let exact = name_lower == *lower_query;
            let name_match = name_lower.contains(lower_query);
            let token_match = tokens.iter().any(|t| {
                let tl = t.to_lowercase();
                name_lower.contains(&tl) || sig_lower.contains(&tl) || comment_lower.contains(&tl)
            });

            if exact {
                results.push(search_vector::db::SearchResult {
                    id: 0,
                    name: member.name.as_str().to_string(),
                    source: doc.meta.source.as_str().to_string(),
                    signature: member.signature.as_ref().map(|s| s.as_str().to_string()),
                    similarity: 1.0,
                });
            } else if name_match {
                results.push(search_vector::db::SearchResult {
                    id: 0,
                    name: member.name.as_str().to_string(),
                    source: doc.meta.source.as_str().to_string(),
                    signature: member.signature.as_ref().map(|s| s.as_str().to_string()),
                    similarity: 0.8,
                });
            } else if token_match {
                results.push(search_vector::db::SearchResult {
                    id: 0,
                    name: member.name.as_str().to_string(),
                    source: doc.meta.source.as_str().to_string(),
                    signature: member.signature.as_ref().map(|s| s.as_str().to_string()),
                    similarity: 0.5,
                });
            }
        }
    }
}

fn cmd_test() {
    println!("Running tests...");
    let args: Vec<&str> = vec!["cargo", "test", "--workspace"];
    let ok = run_command(&args);
    if !ok {
        std::process::exit(1);
    }
}

fn cmd_db_stats(db_path: &str, label: &str) {
    println!("{label}:");
    let db = PathBuf::from(db_path);
    if db.exists() {
        match GuidanceDb::open(&db) {
            Ok(gdb) => {
                match gdb.get_node_count() {
                    Ok(count) => println!("  Nodes: {count}"),
                    Err(_) => println!("  Could not query node count"),
                }
                match gdb.get_embedding_count() {
                    Ok(count) => println!("  Embeddings: {count}"),
                    Err(_) => println!("  Could not query embedding count"),
                }
            }
            Err(e) => println!("  Could not open database: {e}"),
        }
    } else {
        println!("  No database found at {db_path}");
    }
}

fn cmd_init(dir: &str) {
    let d = Path::new(dir).join(".guidance");
    ensure_dir_or_panic(&d);
    let config_path = d.join("guidance-config.json");
    if !config_path.exists() {
        let default_config = serde_json::json!({
            "version": "1",
            "guidance_dir": ".guidance",
            "db_path": ".guidance.db",
            "skills_dir": "doc/skills",
            "capabilities_dir": "doc/capabilities",
            "src_dirs": ["src"],
            "providers": {
                "local": { "base_url": "http://localhost:11434", "chat_endpoint": "/v1/chat/completions" },
                "ollama": { "base_url": "http://localhost:11434", "chat_endpoint": "/api/chat" }
            },
            "models": {
                "default": "ollama:code:latest",
                "fast": "ollama:code:latest",
                "thinking": "ollama:code:latest",
                "batch": "ollama:code:latest",
                "embed": "ollama:embed:latest"
            },
            "embed": { "dims": 768, "cache_limit": 400 }
        });
        common_core::io::write_atomic(
            &config_path,
            serde_json::to_string_pretty(&default_config)
                .unwrap()
                .as_bytes(),
        )
        .expect("write config");
    }
    println!("Initialized guidance in {}", d.display());
}

#[allow(clippy::too_many_arguments)]
async fn cmd_sync(
    file: Option<&str>,
    scan: Option<&str>,
    workspace: &str,
    json_dir: &str,
    db_path: &str,
    force: bool,
    no_db: bool,
    dry_run: bool,
    verbose: bool,
    watch: bool,
    watch_debounce_ms: u64,
) {
    let workspace_path = PathBuf::from(workspace);
    let guidance_dir = PathBuf::from(json_dir);
    let db = PathBuf::from(db_path);
    let cfg = load_project_config(&workspace_path);

    if dry_run {
        println!("Dry run — no files will be written.");
    }

    if let Some(path) = file {
        let source_path = Path::new(path);
        if !source_path.exists() {
            eprintln!("error: file not found: {path}");
            std::process::exit(1);
        }
        let source_dir = if source_path.is_dir() {
            source_path.to_path_buf()
        } else {
            source_path.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        ensure_dir_or_panic(&guidance_dir);

        if source_path.is_dir() {
            let engine = SyncEngine::new(guidance_dir.clone(), source_dir.clone());
            let status = engine.status().expect("status");
            println!(
                "Generated guidance for directory ({} stale)",
                status.stale_files
            );
        } else {
            let json_path = guidance_dir.join("src").join(format!(
                "{}.json",
                source_path
                    .strip_prefix(&source_dir)
                    .unwrap_or(source_path)
                    .display()
            ));
            if !force && !guidance_core::sync::staleness::should_generate(&json_path, source_path) {
                if verbose {
                    println!("  skip (up to date): {path}");
                }
                return;
            }
            match runtime::AST_POOL
                .submit(runtime::AstGenPayload {
                    source_path: source_path.to_path_buf(),
                    source_dir,
                    guidance_dir: guidance_dir.clone(),
                    config: guidance_core::sync_engine::GenConfig::default(),
                })
                .await
            {
                Ok(doc) => {
                    if verbose {
                        println!("  gen: {path} ({} members)", doc.members.len());
                    }
                    println!("Generated guidance for {path}");
                    println!(
                        "  {} members, language: {}",
                        doc.members.len(),
                        doc.meta.language
                    );
                }
                Err(ResultPoolError::Inner(e)) => eprintln!("error generating {path}: {e}"),
                Err(ResultPoolError::Canceled) => {
                    eprintln!("error: pool response canceled for {path}")
                }
                Err(ResultPoolError::Pool(e)) => eprintln!("error: pool queue {e} for {path}"),
            }
        }
    } else if let Some(scan_dir) = scan {
        let scan_path = PathBuf::from(scan_dir);
        ensure_dir_or_panic(&guidance_dir);
        let generated =
            walk_and_gen_async(guidance_dir.clone(), scan_path, force, verbose, &[]).await;
        println!("Scanned {scan_dir}: generated {generated} files");
    } else {
        let src_dirs = src_dirs_from_config_with(&workspace_path, &cfg);

        let mut total_files = 0usize;
        let mut stale_files = 0usize;
        let mut generated = 0usize;

        for src_dir in &src_dirs {
            if !src_dir.is_dir() {
                continue;
            }
            ensure_dir_or_panic(&guidance_dir);
            let engine = SyncEngine::new(guidance_dir.clone(), src_dir.clone());
            let status = engine.status().expect("status");
            total_files += status.total_files;
            stale_files += status.stale_files;

            if stale_files > 0 || force {
                generated +=
                    walk_and_gen_async(guidance_dir.clone(), src_dir.clone(), force, verbose, &[])
                        .await;
            }
        }

        println!("Syncing {total_files} total files ({stale_files} stale)...");
        if generated > 0 {
            println!("Generated {generated} files.");
        }

        if !no_db && !dry_run {
            let json_src = guidance_dir.join("src");
            if json_src.is_dir() {
                match runtime::DB_POOL
                    .submit(runtime::DbSyncPayload {
                        json_dir: json_src,
                        db_path: db,
                    })
                    .await
                {
                    Ok(count) => println!("Synced {count} nodes to {db_path}"),
                    Err(ResultPoolError::Inner(e)) => {
                        eprintln!("Warning: db sync failed: {e}")
                    }
                    Err(ResultPoolError::Canceled) => eprintln!("Warning: db sync canceled"),
                    Err(ResultPoolError::Pool(e)) => {
                        eprintln!("Warning: db sync queue: {e}")
                    }
                }
            }
        }
        println!("Sync complete.");
    }

    if watch {
        let src_dirs = src_dirs_from_config(&workspace_path);
        start_watcher(
            guidance_dir,
            &src_dirs,
            workspace_path,
            force,
            verbose,
            watch_debounce_ms,
        )
        .await;
    }
}

fn src_dirs_from_config_with(workspace_path: &Path, cfg: &config::ProjectConfig) -> Vec<PathBuf> {
    if cfg.src_dirs.is_empty() {
        vec![workspace_path.to_path_buf()]
    } else {
        cfg.src_dirs
            .iter()
            .map(|d| workspace_path.join(d))
            .collect()
    }
}

fn src_dirs_from_config(workspace_path: &Path) -> Vec<PathBuf> {
    let cfg = load_project_config(workspace_path);
    src_dirs_from_config_with(workspace_path, &cfg)
}

async fn walk_and_gen_async(
    guidance_dir: PathBuf,
    source_dir: PathBuf,
    force: bool,
    verbose: bool,
    filter_exts: &[&str],
) -> usize {
    let exts: Vec<&str> = if filter_exts.is_empty() {
        walk::SOURCE_EXTENSIONS.to_vec()
    } else {
        filter_exts.to_vec()
    };
    let mut files = Vec::new();
    walk::walk_files(&source_dir, &exts, |p| files.push(p.to_path_buf()));
    if files.is_empty() {
        return 0;
    }

    let mut handles = Vec::with_capacity(files.len());
    let mut generated = 0usize;

    for path in &files {
        let rel = path.strip_prefix(&source_dir).unwrap_or(path);
        let json_path = guidance_dir
            .join("src")
            .join(format!("{}.json", rel.display()));
        let should_gen = force || guidance_core::sync::staleness::should_generate(&json_path, path);
        if !should_gen {
            if verbose {
                println!("  skip: {}", rel.display());
            }
            continue;
        }

        let pool = Arc::clone(&*runtime::AST_POOL);
        let source_path = path.clone();
        let src_dir = source_dir.clone();
        let gd = guidance_dir.clone();
        let rel_str = rel.display().to_string();
        handles.push(tokio::spawn(async move {
            let result = pool
                .submit(runtime::AstGenPayload {
                    source_path,
                    source_dir: src_dir,
                    guidance_dir: gd,
                    config: guidance_core::sync_engine::GenConfig::default(),
                })
                .await;
            (rel_str, result)
        }));
    }

    let mut gen_failed = 0usize;
    for handle in handles {
        let (rel_path, result) = handle.await.unwrap();
        match result {
            Ok(doc) => {
                generated += 1;
                if verbose {
                    println!("  gen: {rel_path} ({} members)", doc.members.len());
                }
            }
            Err(ResultPoolError::Inner(e)) => {
                gen_failed += 1;
                if verbose {
                    eprintln!("  warn: {rel_path} — {e}");
                }
            }
            Err(ResultPoolError::Canceled) => {
                gen_failed += 1;
                if verbose {
                    eprintln!("  warn: {rel_path} — pool response canceled");
                }
            }
            Err(ResultPoolError::Pool(e)) => {
                gen_failed += 1;
                if verbose {
                    eprintln!("  warn: {rel_path} — pool queue {e}");
                }
            }
        }
    }

    if gen_failed > 0 {
        eprintln!("Warning: {gen_failed} files failed to generate");
    }

    generated
}

#[allow(clippy::needless_pass_by_value)]
async fn start_watcher(
    guidance_dir: PathBuf,
    src_dirs: &[PathBuf],
    workspace_path: PathBuf,
    _force: bool,
    verbose: bool,
    debounce_ms: u64,
) {
    if src_dirs.is_empty() {
        return;
    }

    let (tx, rx) = mpsc::channel();
    let mut watcher = RecommendedWatcher::new(tx, NotifyConfig::default())
        .expect("failed to create file watcher");
    for dir in src_dirs {
        if dir.is_dir() {
            watcher
                .watch(dir, RecursiveMode::Recursive)
                .unwrap_or_else(|e| eprintln!("Warning: could not watch {:?}: {e}", dir));
        }
    }

    println!("Watching for changes (debounce: {debounce_ms}ms)...");

    loop {
        match rx.recv() {
            Ok(Ok(Event { paths, .. })) => {
                for path in &paths {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if !walk::SOURCE_EXTENSIONS.contains(&ext) {
                        continue;
                    }
                    let source_dir = find_source_dir(path, src_dirs);

                    if verbose {
                        println!("  change detected: {}", path.display());
                    }

                    match runtime::AST_POOL
                        .submit(runtime::AstGenPayload {
                            source_path: path.clone(),
                            source_dir: source_dir.unwrap_or_else(|| workspace_path.clone()),
                            guidance_dir: guidance_dir.clone(),
                            config: guidance_core::sync_engine::GenConfig::default(),
                        })
                        .await
                    {
                        Ok(doc) => {
                            if verbose {
                                println!(
                                    "  regenerated: {} ({} members)",
                                    path.display(),
                                    doc.members.len()
                                );
                            }
                        }
                        Err(ResultPoolError::Inner(e)) => {
                            eprintln!("  regeneration failed: {e}");
                        }
                        Err(ResultPoolError::Canceled) => {
                            eprintln!("  regeneration canceled");
                        }
                        Err(ResultPoolError::Pool(e)) => {
                            eprintln!("  regeneration queue error: {e}");
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Watch error: {e}");
            }
            Err(_) => break,
        }
    }
}

fn find_source_dir(path: &Path, src_dirs: &[PathBuf]) -> Option<PathBuf> {
    for dir in src_dirs {
        if path.starts_with(dir) {
            return Some(dir.clone());
        }
    }
    None
}

fn cmd_status(guidance_dir: &str) {
    let gdir = PathBuf::from(guidance_dir);
    let workspace_path = std::env::current_dir().unwrap_or_default();
    let cfg = load_project_config(&workspace_path);
    let src_dirs = src_dirs_from_config_with(&workspace_path, &cfg);
    let mut total_files = 0usize;
    let mut stale_files = 0usize;
    let mut up_to_date = 0usize;
    for src_dir in &src_dirs {
        if !src_dir.is_dir() {
            continue;
        }
        let engine = SyncEngine::new(gdir.clone(), src_dir.clone());
        match engine.status() {
            Ok(status) => {
                total_files += status.total_files;
                stale_files += status.stale_files;
                up_to_date += status.up_to_date;
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }
    println!("Sync Status:");
    println!("  Total files: {total_files}");
    println!("  Stale files: {stale_files}");
    println!("  Up to date:  {up_to_date}");
    println!("  Clean:       {}", stale_files == 0);
}

fn cmd_clean(json_dir: &str, db_path: &str) {
    println!("Cleaning generated files...");
    let db = Path::new(db_path);
    if db.exists() {
        std::fs::remove_file(db)
            .unwrap_or_else(|e| eprintln!("Warning: could not remove {db_path}: {e}"));
        println!("  Removed {db_path}");
    }
    let guidance_src = Path::new(json_dir).join("src");
    if guidance_src.exists() {
        walk::walk_files(&guidance_src, &["json"], |path| {
            std::fs::remove_file(path)
                .unwrap_or_else(|e| eprintln!("Warning: could not remove {:?}: {e}", path));
        });
        println!("  Removed generated JSON files");
    }
    println!("Clean complete.");
}

async fn cmd_commit(dry_run: bool, debug: bool, force: bool) {
    // All work runs inside spawn_blocking because reqwest::blocking::Client
    // cannot be used from a tokio worker thread (it creates its own internal
    // runtime and calls block_on during construction and request dispatch).
    let result = tokio::task::spawn_blocking(move || cmd_commit_inner(dry_run, debug, force)).await;
    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("commit task failed: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_commit_inner(dry_run: bool, debug: bool, force: bool) {
    let workspace = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Failed to get current directory: {e}");
        std::process::exit(1);
    });
    let cfg = load_project_config(&workspace);

    // 1. Get staged diff.
    let diff = common_core::git::diff_staged(&workspace).unwrap_or_else(|e| {
        eprintln!("git diff failed: {e}");
        std::process::exit(1);
    });
    if diff.is_empty() {
        println!("No staged changes to commit. Use 'git add' to stage files first.");
        return;
    }

    // 2. Load guidance context for code files.
    let guidance_dir = workspace.join(&cfg.guidance_dir);
    let context = commit::load_guidance_context(&diff, &guidance_dir);

    // 3. Resolve commit model from config.
    let (api_url, model) = commit::resolve_commit_model(&cfg);

    // 4. Generate initial commit message via LLM.
    let initial_msg = commit::generate_commit_message(&diff, &context, &api_url, &model, debug)
        .unwrap_or_else(|e| {
            if debug {
                eprintln!("[commit] LLM error: {e}");
            }
            "* Update codebase".to_string()
        });

    if debug {
        eprintln!("[commit] generated message:\n{initial_msg}");
    }

    // 5. Write to temp file.
    let tmp_path = editor::write_temp_file(&initial_msg, "guidance_commit_").unwrap_or_else(|e| {
        eprintln!("Failed to create temp file: {e}");
        std::process::exit(1);
    });

    // 6. Record mtime before opening editor.
    let mtime_before = editor::file_mtime(&tmp_path);

    // 7. Open editor.
    if let Err(e) = editor::open_editor(&tmp_path) {
        eprintln!("Failed to open editor: {e}");
        editor::cleanup_temp(&tmp_path);
        std::process::exit(1);
    }

    // 8. Compare mtime — if unchanged, user didn't save.
    let mtime_after = editor::file_mtime(&tmp_path);
    if mtime_before == mtime_after {
        editor::cleanup_temp(&tmp_path);
        println!("Commit message not saved. Aborting.");
        return;
    }

    // 9. Read cleaned message (strip # comments, trim whitespace).
    let final_msg = match editor::read_cleaned(&tmp_path) {
        Ok(msg) if !msg.is_empty() => msg,
        _ => {
            editor::cleanup_temp(&tmp_path);
            println!("Commit message is empty. Aborting.");
            return;
        }
    };
    editor::cleanup_temp(&tmp_path);

    // 10. Dry run — print and exit.
    if dry_run {
        println!("--- Commit message ---\n{final_msg}\n---");
        return;
    }

    // 11. Commit.
    match common_core::git::commit(&workspace, &final_msg) {
        Ok(true) => println!("Committed successfully."),
        Ok(false) => {
            eprintln!("git commit failed (nothing to commit or hook rejected)");
            if !force {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("git commit failed: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_check(workspace: &str) {
    let workspace_path = PathBuf::from(workspace);
    let cfg = load_project_config(&workspace_path);
    let guidance_dir = workspace_path.join(".guidance");
    let mut all_passed = true;

    let src_dirs = src_dirs_from_config_with(&workspace_path, &cfg);
    let present_exts = walk::collect_extensions(&src_dirs);

    type StageFn = Box<dyn Fn() -> Result<(), String>>;
    let mut stages: Vec<(&str, StageFn)> = Vec::new();

    for (ext, argv) in &cfg.test_commands {
        if !present_exts.contains(ext.as_str()) {
            continue;
        }
        let argv: Vec<String> = argv.clone();
        let ext = ext.clone();
        stages.push((
            "test",
            Box::new(move || {
                let argv_refs: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
                if run_command(&argv_refs) {
                    Ok(())
                } else {
                    Err(format!("{ext}: test command failed"))
                }
            }),
        ));
    }

    for (ext, argv) in &cfg.lint_commands {
        if !present_exts.contains(ext.as_str()) {
            continue;
        }
        let argv: Vec<String> = argv.clone();
        let ext = ext.clone();
        stages.push((
            "lint",
            Box::new(move || {
                let argv_refs: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
                if run_command(&argv_refs) {
                    Ok(())
                } else {
                    Err(format!("{ext}: lint command failed"))
                }
            }),
        ));
    }

    for (ext, argv) in &cfg.fmt_commands {
        if !present_exts.contains(ext.as_str()) {
            continue;
        }
        let argv: Vec<String> = argv.clone();
        let ext = ext.clone();
        stages.push((
            "fmt",
            Box::new(move || {
                let argv_refs: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
                if run_command(&argv_refs) {
                    Ok(())
                } else {
                    Err(format!("{ext}: fmt command failed"))
                }
            }),
        ));
    }

    let cfg_gen = cfg.clone();
    let ws_gen = workspace_path.clone();
    let gd_gen = guidance_dir.clone();
    stages.push((
        "gen",
        Box::new(move || {
            let src_dirs = src_dirs_from_config_with(&ws_gen, &cfg_gen);
            let mut total_stale = 0usize;
            for src_dir in &src_dirs {
                if !src_dir.is_dir() {
                    continue;
                }
                let engine = SyncEngine::new(gd_gen.clone(), src_dir.clone());
                let status = engine.status().map_err(|e| format!("status: {e}"))?;
                total_stale += status.stale_files;
            }
            if total_stale > 0 {
                Err(format!("{total_stale} stale files need regeneration"))
            } else {
                Ok(())
            }
        }),
    ));

    let structure_path = workspace_path.join("STRUCTURE.md");
    stages.push((
        "structure",
        Box::new(move || {
            if structure_path.exists() {
                Ok(())
            } else {
                Err("STRUCTURE.md not found".into())
            }
        }),
    ));

    let db_path = workspace_path.join(".guidance.db");
    stages.push((
        "db",
        Box::new(move || {
            if db_path.exists() {
                Ok(())
            } else {
                Err(".guidance.db not found".into())
            }
        }),
    ));

    for (name, stage_fn) in &stages {
        print!("{name}... ");
        match stage_fn() {
            Ok(()) => println!("OK"),
            Err(e) => {
                println!("FAILED: {e}");
                all_passed = false;
                break;
            }
        }
    }

    if all_passed {
        println!("\nAll checks passed");
    } else {
        std::process::exit(1);
    }
}

fn cmd_todo() {
    let todo_path = Path::new(".guidance/doc/TODO.md");
    if todo_path.exists() {
        match common_core::io::read_to_string_err(todo_path) {
            Ok(content) => {
                println!("TODO items:\n");
                for line in content.lines() {
                    if line.trim().starts_with("- [") {
                        println!("  {line}");
                    }
                }
            }
            Err(e) => println!("Could not read TODO.md: {e}"),
        }
    } else {
        println!("TODO items:\n  No TODO.md found — create .guidance/doc/TODO.md");
    }
}

fn cmd_diary(text_or_path: &str) {
    let diary_dir = Path::new(".guidance/doc");
    ensure_dir_or_panic(diary_dir);
    let diary_path = diary_dir.join("DIARY.md");
    let timestamp = OffsetDateTime::now_utc()
        .format(
            &time::format_description::parse("[year]-[month]-[day] [hour]:[minute] UTC")
                .unwrap_or(time::format_description::parse("[year]-[month]-[day]").unwrap()),
        )
        .unwrap_or_else(|_| "unknown date".to_string());

    // If the argument is a readable file path, read its contents.
    // Otherwise, treat it as inline text.
    let content = if !text_or_path.is_empty() {
        let p = Path::new(text_or_path);
        if p.is_file() {
            match common_core::io::read_to_string_err(p) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Could not read {text_or_path}: {e}");
                    return;
                }
            }
        } else {
            text_or_path.to_string()
        }
    } else {
        text_or_path.to_string()
    };

    let entry = if content.is_empty() {
        format!("\n## {timestamp}\n\n(empty entry)\n")
    } else {
        format!("\n## {timestamp}\n\n{content}\n")
    };
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&diary_path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, entry.as_bytes()))
        .unwrap_or_else(|e| eprintln!("Could not write diary: {e}"));
    println!("Diary entry appended to {:?}", diary_path);
}

fn cmd_structure(json_dir: &str) {
    let gdir = PathBuf::from(json_dir);
    match structure::generate(&gdir) {
        Ok(output) => {
            common_core::io::write_atomic(std::path::Path::new("STRUCTURE.md"), output.as_bytes())
                .expect("write STRUCTURE.md");
            let line_count = output.lines().count();
            println!("STRUCTURE.md generated ({} lines)", line_count);
        }
        Err(e) => {
            eprintln!("error generating structure: {e}");
        }
    }
}

fn cmd_health(workspace: &str, _min_age: u32, format: &str, _db: &str) {
    let ws = PathBuf::from(workspace);
    let gdir = PathBuf::from(".guidance");
    let src_dir = gdir.join("src");
    let mut total_members = 0usize;
    let mut without_comments = 0usize;
    let mut files: Vec<String> = Vec::new();

    if src_dir.is_dir() {
        collect_health_stats(
            &src_dir,
            &mut total_members,
            &mut without_comments,
            &mut files,
        );
    }

    match format {
        "json" => {
            let report = serde_json::json!({
                "files_analyzed": files.len(),
                "total_members": total_members,
                "without_comments": without_comments,
                "comment_coverage_pct": if total_members > 0 {
                    100.0 - (without_comments as f64 / total_members as f64) * 100.0
                } else { 100.0 },
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_default()
            );
        }
        "human" => {
            println!("Health Report ({})", ws.display());
            println!("  Files analyzed:    {}", files.len());
            println!("  Total members:     {total_members}");
            println!("  Without comments:  {without_comments}");
            if total_members > 0 {
                let pct = (without_comments as f64 / total_members as f64) * 100.0;
                println!("  Comment coverage:  {:.1}%", 100.0 - pct);
            }
        }
        _ => {
            println!("## Health Report\n");
            println!("| Metric | Value |");
            println!("|--------|-------|");
            println!("| Files analyzed | {} |", files.len());
            println!("| Total members | {total_members} |");
            println!("| Without comments | {without_comments} |");
            if total_members > 0 {
                let pct = (without_comments as f64 / total_members as f64) * 100.0;
                println!("| Comment coverage | {:.1}% |", 100.0 - pct);
            }
        }
    }
}

fn collect_health_stats(
    dir: &Path,
    total: &mut usize,
    no_comments: &mut usize,
    files: &mut Vec<String>,
) {
    for (_path, doc) in walk_guidance_docs(dir) {
        files.push(doc.meta.source.as_str().to_string());
        for member in &doc.members {
            *total += 1;
            if member.comment.is_none() {
                *no_comments += 1;
            }
        }
    }
}

fn cmd_mcp(db_path: &str) {
    let db = PathBuf::from(db_path);
    if let Err(e) = mcp::serve_stdio_from_path(&db) {
        eprintln!("MCP server error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn test_cli_help() {
        let mut cmd = super::Cli::command();
        let help = cmd.render_help();
        assert!(!help.to_string().is_empty());
    }
}
