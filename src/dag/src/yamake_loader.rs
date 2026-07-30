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
}
