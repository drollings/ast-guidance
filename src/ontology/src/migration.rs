#[derive(Debug, Clone)]
pub struct OntologyVersion {
    pub version: String,
    pub loaded_at: f64,
    pub source_url: String,
    pub triple_count: u64,
}

pub type MigrateFn = fn();

pub struct OntologyMigration {
    pub from_version: &'static str,
    pub to_version: &'static str,
    pub migrate_fn: MigrateFn,
}

fn noop_migration() {}

pub const MIGRATIONS: &[OntologyMigration] = &[OntologyMigration {
    from_version: "4.5",
    to_version: "4.6",
    migrate_fn: noop_migration,
}];

pub struct VersionRegistry {
    versions: Vec<OntologyVersion>,
}

impl VersionRegistry {
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    pub fn record(&mut self, ver: OntologyVersion) {
        self.versions.push(ver);
    }

    pub fn latest(&self) -> Option<&OntologyVersion> {
        self.versions.last()
    }

    pub fn count(&self) -> usize {
        self.versions.len()
    }
}

impl Default for VersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../tests/migration.rs"]
mod tests;
