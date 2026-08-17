use bitvec::vec::BitVec;
use common_core::interner::CapabilityRegistry;
use fluent_types::{ExecutorKind, TargetType};
use internment::ArcIntern;
use serde::Deserialize;

use crate::target::{Target, TargetRegistry};

#[derive(Debug, Deserialize)]
struct TargetDef {
    id: i64,
    name: String,
    target_type: String,
    #[serde(default)]
    depends: Vec<String>,
    #[serde(default)]
    provides: Vec<String>,
    #[serde(default)]
    essential: bool,
}

#[derive(Debug, Deserialize)]
struct YamakeConfig {
    targets: Vec<TargetDef>,
}

pub fn load_yamake_config(json: &str) -> (TargetRegistry, CapabilityRegistry) {
    let config: YamakeConfig = serde_json::from_str(json).expect("invalid yamake.json");
    let caps = CapabilityRegistry::new();
    let mut reg = TargetRegistry::new();

    for td in &config.targets {
        for name in td
            .depends
            .iter()
            .chain(td.provides.iter())
            .chain(std::iter::once(&td.name))
        {
            caps.intern(name);
        }
    }

    for td in &config.targets {
        let depends: BitVec =
            caps.to_bitvec(&td.depends.iter().map(String::as_str).collect::<Vec<_>>());
        let provides: BitVec =
            caps.to_bitvec(&td.provides.iter().map(String::as_str).collect::<Vec<_>>());

        // Concrete targets (File/Phony) implicitly provide their own name,
        // matching yamake-old.py semantics where depends reference Target
        // objects. Abstract targets do NOT self-provide — they are only
        // satisfied by other targets' provides lists. Self-provision of
        // abstracts would create cycles (e.g. staff → stage_hands → stage →
        // confuse_a_cat → staff).
        let mut combined_provides = provides.clone();
        let is_file = td.target_type.as_str() == "file" || td.target_type.as_str() == "phony";
        if is_file {
            let self_bit = caps.to_bitvec(&[td.name.as_str()]);
            for i in self_bit.iter_ones() {
                if i >= combined_provides.len() {
                    combined_provides.resize(i + 1, false);
                }
                combined_provides.set(i, true);
            }
        }

        let target_type = match td.target_type.as_str() {
            "file" => TargetType::File,
            "abstract" => TargetType::Abstract,
            "phony" => TargetType::Phony,
            other => panic!("unknown target_type: {other}"),
        };
        let name: ArcIntern<str> = ArcIntern::from(td.name.as_str());
        let target = Target::new()
            .id(td.id)
            .name(name)
            .target_type(target_type)
            .executor(ExecutorKind::Native)
            .depends(depends)
            .provides(combined_provides)
            .essential(td.essential)
            .build();
        reg.register(target)
            .expect("duplicate target in yamake.json");
    }

    (reg, caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_yamake_load() {
        let json = include_str!("../../../data/yamake.json");
        let (reg, _caps) = load_yamake_config(json);
        assert!(reg.len() > 50);
    }

    #[test]
    fn empty_graph_loads_empty_registry() {
        let (reg, caps) = load_yamake_config(r#"{"targets": []}"#);
        assert_eq!(reg.len(), 0);
        assert_eq!(caps.count(), 0);
    }

    #[test]
    #[should_panic(expected = "invalid yamake.json")]
    fn malformed_json_panics() {
        load_yamake_config("not json");
    }

    #[test]
    #[should_panic(expected = "unknown target_type")]
    fn unknown_target_type_panics() {
        load_yamake_config(
            r#"{"targets": [{"id": 1, "name": "a", "target_type": "bogus"}]}"#,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate target")]
    fn duplicate_target_panics() {
        load_yamake_config(
            r#"{"targets": [
                {"id": 1, "name": "a", "target_type": "file"},
                {"id": 2, "name": "a", "target_type": "file"}
            ]}"#,
        );
    }

    #[test]
    fn concrete_targets_self_provide_their_name() {
        let (reg, caps) =
            load_yamake_config(r#"{"targets": [{"id": 1, "name": "build", "target_type": "file", "provides": ["artifact"]}]}"#);
        let target = reg.get("build").expect("target registered");
        let provides = caps.bitvec_to_names(&target.provides);
        assert!(provides.iter().any(|n| &**n == "build"), "file self-provides");
        assert!(provides.iter().any(|n| &**n == "artifact"), "declared provide kept");
    }

    #[test]
    fn abstract_targets_do_not_self_provide() {
        let (reg, caps) =
            load_yamake_config(r#"{"targets": [{"id": 1, "name": "staff", "target_type": "abstract"}]}"#);
        let target = reg.get("staff").expect("target registered");
        assert!(
            caps.bitvec_to_names(&target.provides).is_empty(),
            "abstract targets must not self-provide (cycle guard)"
        );
    }

    #[test]
    fn phony_targets_self_provide_like_files() {
        let (reg, caps) =
            load_yamake_config(r#"{"targets": [{"id": 1, "name": "clean", "target_type": "phony"}]}"#);
        let target = reg.get("clean").expect("target registered");
        assert!(caps.bitvec_to_names(&target.provides).iter().any(|n| &**n == "clean"));
    }

    #[test]
    fn depends_and_essential_are_wired() {
        let (reg, _caps) = load_yamake_config(
            r#"{"targets": [
                {"id": 1, "name": "src", "target_type": "file"},
                {"id": 2, "name": "bin", "target_type": "file", "depends": ["src"], "essential": true}
            ]}"#,
        );
        let bin = reg.get("bin").expect("bin registered");
        assert!(bin.essential);
        let _ = reg.get("src").expect("src registered");
    }
}
