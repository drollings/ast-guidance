use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, Subcommand};
use common_core::ensure_dir;
use job_copilot::config::DaemonConfig;

#[derive(Parser)]
#[command(
    name = "job-copilot-daemon",
    about = "Local-only human-in-the-loop job application copilot"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable debug logging.
    #[arg(long, global = true)]
    debug: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the daemon (Native Messaging + HTTP loopback).
    Serve {
        /// Path to the user profile TOML file (required).
        #[arg(long)]
        profile: PathBuf,

        /// HTTP loopback port (default: 7182).
        #[arg(long, default_value_t = 7182)]
        rest_port: u16,

        /// Disable the HTTP loopback endpoint.
        #[arg(long)]
        no_rest: bool,

        /// Local LLM base URL (default: http://127.0.0.1:11434/v1).
        #[arg(long, default_value = "http://127.0.0.1:11434/v1")]
        llm_url: String,

        /// LLM model name (default: llama3).
        #[arg(long, default_value = "llama3")]
        llm_model: String,

        /// Path for the append-only JSONL audit log.
        #[arg(long)]
        audit_log: Option<PathBuf>,
    },

    /// Validate a profile TOML file.
    ValidateProfile {
        /// Path to the profile TOML file.
        profile_path: PathBuf,
    },

    /// Register the native messaging host with Chromium/Firefox.
    InstallNativeMessaging {
        /// Use Chrome-style manifest (allowed_origins).
        #[arg(long, conflicts_with = "firefox")]
        chrome: bool,

        /// Use Firefox-style manifest (allowed_extensions).
        #[arg(long, conflicts_with = "chrome")]
        firefox: bool,

        /// Directory to write the manifest file to.
        manifest_dir: PathBuf,

        /// Extension ID (required for Chrome).
        #[arg(long)]
        extension_id: Option<String>,
    },

    /// Run diagnostic checks on the daemon setup.
    Doctor {
        /// Optional profile path to validate (overrides default).
        #[arg(long)]
        profile: Option<PathBuf>,
    },

    /// Stream the audit log to stdout.
    AuditTail {
        /// Path to the audit log JSONL file.
        path: PathBuf,

        /// Follow the file (tail -f style).
        #[arg(long)]
        follow: bool,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let cli = Cli::parse();

    if cli.debug {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt().init();
    }

    let result = match cli.command {
        Commands::Serve {
            profile,
            rest_port,
            no_rest,
            llm_url,
            llm_model,
            audit_log,
        } => {
            let config = DaemonConfig::new()
                .profile_path(profile)
                .rest_port(rest_port)
                .enable_rest(!no_rest)
                .llm_url(llm_url)
                .llm_model(llm_model)
                .audit_log_path(audit_log.unwrap_or_default())
                .build();
            if let Err(e) = config.validate() {
                eprintln!("Configuration error: {e}");
                std::process::exit(1);
            }
            job_copilot::server::serve_native_messaging(config).await
        }
        Commands::ValidateProfile { profile_path } => {
            match job_copilot::profile::Profile::load_from_path(&profile_path) {
                Ok(profile) => match profile.validate() {
                    Ok(()) => {
                        println!("Profile is valid.");
                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("Profile validation error: {e}");
                        Err(e)
                    }
                },
                Err(e) => {
                    eprintln!("Profile load error: {e}");
                    Err(e)
                }
            }
        }
        Commands::InstallNativeMessaging {
            chrome,
            firefox,
            manifest_dir,
            extension_id,
        } => {
            run_install_native_messaging(chrome, firefox, &manifest_dir, extension_id);
            Ok(())
        }
        Commands::Doctor { profile } => {
            run_doctor(profile.as_ref());
            Ok(())
        }
        Commands::AuditTail { path, follow } => {
            run_audit_tail(&path, follow);
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Fatal error: {e}");
        std::process::exit(1);
    }
}

fn run_audit_tail(path: &std::path::Path, follow: bool) {
    use std::fs::File;
    use std::thread;
    use std::time::Duration;

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open audit log {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let mut reader = BufReader::new(file);

    // Print all existing lines.
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                print!("{line}");
            }
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }
    }

    if !follow {
        return;
    }

    // Follow mode: poll for new lines.
    loop {
        thread::sleep(Duration::from_millis(200));
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {} // No new data yet.
            Ok(_) => {
                print!("{line}");
            }
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }
    }
}

fn run_install_native_messaging(
    chrome: bool,
    firefox: bool,
    manifest_dir: &std::path::Path,
    extension_id: Option<String>,
) {
    if !chrome && !firefox {
        eprintln!("Error: specify --chrome or --firefox");
        std::process::exit(1);
    }

    let Some(ext_id) = extension_id else {
        eprintln!("Error: --extension-id is required (find it at chrome://extensions after loading unpacked)");
        std::process::exit(1);
    };

    let binary_path = match std::env::current_exe() {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(e) => {
            eprintln!("Error: cannot determine binary path: {e}");
            std::process::exit(1);
        }
    };

    let manifest = if chrome {
        serde_json::json!({
            "name": "io.github.anomalyco.job_copilot",
            "description": "Job Copilot daemon",
            "path": binary_path,
            "type": "stdio",
            "allowed_origins": [format!("chrome-extension://{ext_id}/")]
        })
    } else {
        serde_json::json!({
            "name": "io.github.anomalyco.job_copilot",
            "description": "Job Copilot daemon",
            "path": binary_path,
            "type": "stdio",
            "allowed_extensions": [ext_id]
        })
    };

    if let Err(e) = ensure_dir(manifest_dir) {
        eprintln!(
            "Error: cannot create manifest directory {}: {e}",
            manifest_dir.display()
        );
        std::process::exit(1);
    }

    let manifest_path = manifest_dir.join("io.github.anomalyco.job_copilot.json");
    let json = serde_json::to_string_pretty(&manifest).expect("valid JSON");

    if let Err(e) = common_core::io::write_atomic(&manifest_path, json.as_bytes()) {
        eprintln!(
            "Error: cannot write manifest to {}: {e}",
            manifest_path.display()
        );
        std::process::exit(1);
    }

    println!(
        "Native messaging manifest written to: {}",
        manifest_path.display()
    );
    let browser = if chrome { "Chrome" } else { "Firefox" };
    println!("Reload the {browser} extension to pick up the native host.");
}

fn run_doctor(profile_path: Option<&std::path::PathBuf>) {
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;

    // Load config: from the given path, or from CWD default, or use built-in defaults.
    let config = if let Some(p) = profile_path {
        match job_copilot::config::DaemonConfig::load(p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: could not load config from {}: {e}", p.display());
                job_copilot::config::DaemonConfig::default()
            }
        }
    } else {
        let default_path = std::path::Path::new(job_copilot::config::CONFIG_FILE_DEFAULT);
        if default_path.exists() {
            match job_copilot::config::DaemonConfig::load(default_path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!(
                        "Warning: could not load config from {}: {e}",
                        default_path.display()
                    );
                    job_copilot::config::DaemonConfig::default()
                }
            }
        } else {
            job_copilot::config::DaemonConfig::default()
        }
    };

    let mut passed = 0u32;
    let mut failed = 0u32;

    println!("Running daemon doctor checks...\n");

    // 1. Config validation
    match config.validate() {
        Ok(()) => {
            println!("  ✓ Config validation");
            passed += 1;
        }
        Err(e) => {
            println!("  ✗ Config validation: {e}");
            failed += 1;
        }
    }

    // 2. Profile file
    if config.profile_path.as_os_str().is_empty() {
        println!("  ✗ Profile file: no profile path configured (use --profile or set profile_path in config)");
        failed += 1;
    } else if !config.profile_path.exists() {
        println!(
            "  ✗ Profile file: file not found: {}",
            config.profile_path.display()
        );
        failed += 1;
    } else {
        println!("  ✓ Profile file");
        passed += 1;
    }

    // 3. LLM reachability
    {
        let models_url = format!("{}/models", config.llm_url.trim_end_matches('/'));
        let host_port = models_url
            .split("://")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .unwrap_or("127.0.0.1:11434");
        if let Some(Ok(stream)) = std::net::SocketAddr::from_str(host_port)
            .ok()
            .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_secs(2)))
        {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(50)));
            drop(stream);
            println!("  ✓ LLM reachability");
            passed += 1;
        } else {
            println!("  ✗ LLM reachability: cannot connect to {host_port}");
            failed += 1;
        }
    }

    // 4. Loopback port bindability
    {
        let addr = format!("{}:{}", config.rest_bind_addr, config.rest_port);
        match TcpListener::bind(&addr) {
            Ok(l) => {
                drop(l);
                println!("  ✓ Loopback port bindability");
                passed += 1;
            }
            Err(e) => {
                println!("  ✗ Loopback port bindability: cannot bind to {addr}: {e}");
                failed += 1;
            }
        }
    }

    // 5. Audit log path writability
    if let Some(path) = &config.audit_log_path {
        if path.as_os_str().is_empty() {
            println!("  ✗ Audit log writability: empty path");
            failed += 1;
        } else {
            let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
            match parent {
                Some(p) if p.exists() => {
                    let probe = p.join(".doctor-probe");
                    match common_core::io::write_atomic(&probe, b"") {
                        Ok(()) => {
                            let _ = std::fs::remove_file(probe);
                            println!("  ✓ Audit log writability");
                            passed += 1;
                        }
                        Err(e) => {
                            println!("  ✗ Audit log writability: directory not writable: {e}");
                            failed += 1;
                        }
                    }
                }
                Some(p) => {
                    println!(
                        "  ✗ Audit log writability: parent directory does not exist: {}",
                        p.display()
                    );
                    failed += 1;
                }
                None => {
                    println!(
                        "  ✗ Audit log writability: cannot determine parent directory for {}",
                        path.display()
                    );
                    failed += 1;
                }
            }
        }
    } else {
        println!("  ✓ Audit log writability (not configured)");
        passed += 1;
    }

    println!("\n{passed} passed, {failed} failed");
    if failed == 0 {
        println!("\nAll checks passed.");
    } else {
        println!("\nSome checks failed. Fix the issues above and re-run `doctor`.");
        std::process::exit(1);
    }
}
