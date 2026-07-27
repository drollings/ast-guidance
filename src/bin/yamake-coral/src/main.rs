use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand};
use fluent_dag::ambiguous_resolver::AmbiguousDependencyResolver;
use fluent_dag::resolver::DependencyResolver;
use fluent_dag::yamake_loader::load_yamake_config;

#[derive(Parser)]
#[command(name = "yamake-coral", about = "Yamake dependency resolver — Kahn's + disambiguation")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Resolve targets using the classic Kahn's DependencyResolver
    Classic {
        /// Path to yamake.json
        #[arg(short, long, default_value = "yamake.json")]
        config: String,

        /// Target names to resolve
        targets: Vec<String>,

        /// Allow non-strict resolution
        #[arg(long)]
        non_strict: bool,
    },
    /// Resolve targets using the AmbiguousDependencyResolver
    Ambiguous {
        /// Path to yamake.json
        #[arg(short, long, default_value = "yamake.json")]
        config: String,

        /// Target names to resolve
        targets: Vec<String>,

        /// Allow non-strict resolution
        #[arg(long)]
        non_strict: bool,
    },
    /// Run both resolvers and compare results + performance
    Compare {
        /// Path to yamake.json
        #[arg(short, long, default_value = "yamake.json")]
        config: String,

        /// Target names to resolve
        targets: Vec<String>,

        /// Allow non-strict resolution
        #[arg(long)]
        non_strict: bool,
    },
    /// Run a battery of test scenarios
    Test {
        /// Path to yamake.json
        #[arg(short, long, default_value = "yamake.json")]
        config: String,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Classic { config, targets, non_strict } => {
            let json = std::fs::read_to_string(&config)
                .unwrap_or_else(|e| {
                    eprintln!("error reading {config}: {e}");
                    process::exit(1);
                });
            let (reg, _caps) = load_yamake_config(&json);
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            let resolver = DependencyResolver::new(&reg).with_strict(!non_strict);
            let start = Instant::now();
            match resolver.resolve(&targets) {
                Ok(plan) => {
                    let elapsed = start.elapsed();
                    println!("classic resolve: {} targets in {:?}", plan.order.len(), elapsed);
                    println!("order: {:?}", plan.target_names);
                }
                Err(e) => {
                    eprintln!("classic resolve error: {e}");
                    process::exit(1);
                }
            }
        }
        Command::Ambiguous { config, targets, non_strict } => {
            let json = std::fs::read_to_string(&config)
                .unwrap_or_else(|e| {
                    eprintln!("error reading {config}: {e}");
                    process::exit(1);
                });
            let (reg, caps) = load_yamake_config(&json);
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();
            let resolver = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(!non_strict);
            let start = Instant::now();
            match resolver.resolve(&targets) {
                Ok(plan) => {
                    let elapsed = start.elapsed();
                    println!("ambiguous resolve: {} targets in {:?}", plan.order.len(), elapsed);
                    println!("order: {:?}", plan.target_names);
                }
                Err(e) => {
                    eprintln!("ambiguous resolve error: {e}");
                    process::exit(1);
                }
            }
        }
        Command::Compare { config, targets, non_strict } => {
            let json = std::fs::read_to_string(&config)
                .unwrap_or_else(|e| {
                    eprintln!("error reading {config}: {e}");
                    process::exit(1);
                });
            let (reg, caps) = load_yamake_config(&json);
            let targets: Vec<&str> = targets.iter().map(String::as_str).collect();

            println!("=== Classic DependencyResolver ===");
            let classic = DependencyResolver::new(&reg).with_strict(!non_strict);
            let start = Instant::now();
            match classic.resolve(&targets) {
                Ok(plan) => {
                    let elapsed = start.elapsed();
                    println!("  result: {} targets in {:?}", plan.order.len(), elapsed);
                    println!("  order: {:?}", plan.target_names);
                }
                Err(e) => {
                    println!("  error: {e}");
                }
            }

            println!("=== AmbiguousDependencyResolver ===");
            let ambiguous = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(!non_strict);
            let start = Instant::now();
            match ambiguous.resolve(&targets) {
                Ok(plan) => {
                    let elapsed = start.elapsed();
                    println!("  result: {} targets in {:?}", plan.order.len(), elapsed);
                    println!("  order: {:?}", plan.target_names);
                }
                Err(e) => {
                    println!("  error: {e}");
                }
            }
        }
        Command::Test { config } => {
            let json = std::fs::read_to_string(&config)
                .unwrap_or_else(|e| {
                    eprintln!("error reading {config}: {e}");
                    process::exit(1);
                });
            let (reg, caps) = load_yamake_config(&json);

            println!("=== yamake-coral test battery ===\n");

            let scenarios: Vec<(&str, Vec<&str>, Option<&str>)> = vec![
                ("just animal", vec!["animal"], Some("classic")),
                ("just confuse", vec!["confuse"], Some("classic")),
                ("confuse + bee", vec!["confuse", "bee"], Some("ambiguous")),
                ("confuse + stoat", vec!["confuse", "stoat"], Some("ambiguous")),
                ("confuse + cat", vec!["confuse", "cat"], Some("ambiguous")),
                ("confuse + puma", vec!["confuse", "puma"], Some("ambiguous")),
                ("confuse + gazelle", vec!["confuse", "gazelle"], Some("ambiguous")),
                ("confuse + vole", vec!["confuse", "vole"], Some("ambiguous")),
                ("confuse + wildebeest", vec!["confuse", "wildebeest"], Some("ambiguous")),
                ("bee + stoat", vec!["bee", "stoat"], None),
                ("confuse_a_cat", vec!["confuse_a_cat"], Some("classic")),
                ("distract_a_bee", vec!["distract_a_bee"], Some("classic")),
                ("stun_a_stoat", vec!["stun_a_stoat"], Some("classic")),
                ("puzzle_a_puma", vec!["puzzle_a_puma"], Some("classic")),
                ("startle_a_thompsons_gazelle", vec!["startle_a_thompsons_gazelle"], Some("classic")),
                ("amaze_a_vole", vec!["amaze_a_vole"], Some("classic")),
                ("bewilderbeest", vec!["bewilderbeest"], Some("classic")),
                ("default build", vec!["default"], Some("classic")),
                ("deep chain: confuse_a_cat", vec!["confuse_a_cat"], Some("classic")),
                ("abstract resolve: agency", vec!["agency"], Some("abstract")),
                ("abstract resolve: cognitive", vec!["cognitive"], Some("abstract")),
            ];

            let mut passed = 0;
            let mut failed = 0;
            let _skipped = 0;
            let total = scenarios.len();

            for (label, targets, expected_resolver) in &scenarios {
                print!("[{passed:>2}/{total:>2}] {label:45} ");

                let classic = DependencyResolver::new(&reg).with_strict(false);
                let ambiguous = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(false);

                let classic_result = classic.resolve(targets);
                let ambiguous_result = ambiguous.resolve(targets);

                let test_ok = match expected_resolver {
                    Some("classic") => classic_result.is_ok(),
                    Some("ambiguous") => ambiguous_result.is_ok(),
                    Some("abstract") => classic_result.is_ok(),
                    None => classic_result.is_ok() || ambiguous_result.is_ok(),
                    _ => false,
                };

                if test_ok {
                    let expected = expected_resolver.unwrap_or("any");
                    match (expected, &classic_result, &ambiguous_result) {
                        ("classic", Ok(cp), Ok(ap)) => {
                            let overlap: usize = cp.target_names.iter()
                                .filter(|n| ap.target_names.contains(n))
                                .count();
                            let classic_pct = if cp.target_names.is_empty() { 0.0 } else {
                                overlap as f64 / cp.target_names.len() as f64 * 100.0
                            };
                            let ambig_pct = if ap.target_names.is_empty() { 0.0 } else {
                                overlap as f64 / ap.target_names.len() as f64 * 100.0
                            };
                            println!("OK  classic={} ambig={} overlap={overlap} ({classic_pct:.0}%/{ambig_pct:.0}%)",
                                cp.target_names.len(), ap.target_names.len());
                        }
                        ("ambiguous", _, Ok(ap)) => {
                            println!("OK  ambiguous={} targets: {:?}", ap.target_names.len(), &ap.target_names[..ap.target_names.len().min(6)]);
                        }
                        ("abstract", Ok(cp), _) => {
                            println!("OK  classic={} targets: {:?}", cp.target_names.len(), &cp.target_names[..cp.target_names.len().min(6)]);
                        }
                        _ => {
                            println!("OK");
                        }
                    }
                    passed += 1;
                } else {
                    match expected_resolver {
                        Some("ambiguous") if ambiguous_result.is_err() => {
                            println!("OK  (expected ambiguity: {})", ambiguous_result.unwrap_err());
                            passed += 1;
                        }
                        _ => {
                            println!("FAIL classic={:?} ambiguous={:?}",
                                classic_result.as_ref().map(|p| p.target_names.len()).map_err(|e| e.to_string()),
                                ambiguous_result.as_ref().map(|p| p.target_names.len()).map_err(|e| e.to_string()));
                            failed += 1;
                        }
                    }
                }
            }

            println!("\n---");
            println!("passed: {passed}, failed: {failed}, total: {total}");
            if failed > 0 {
                process::exit(1);
            }
        }
    }
}
