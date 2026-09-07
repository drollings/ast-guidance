//! Route resolution and routing configuration types.
//! `RoutingConfig` is the resolved routing table used by the classifier stage.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::ModelEntry;
use crate::config::split_model_key;
use crate::config::ModelGroup;
use crate::pipeline::RoutingTarget;
use crate::score_matrix::ScoreMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRef {
    pub group: String,
    #[serde(default = "default_pipelines")]
    pub pipelines: Vec<String>,
    #[serde(default)]
    pub description: String,
    /// Never let the classifier answer requests on this route directly: force
    /// dispatch to the route's group. For domains where the classifier model is
    /// overconfident (creative prose, code, translation, and specialized
    /// knowledge such as science/legal/medical), this guarantees the request
    /// reaches the route's model regardless of the classifier's own complexity
    /// judgment. `local`-style routes keep `always_route: false` so simple
    /// prompts are still answered directly.
    #[serde(default)]
    pub always_route: bool,
}

fn default_pipelines() -> Vec<String> {
    vec!["default".into()]
}

/// One entry of the routing-vocabulary table: the candidate model keys a role
/// resolves to (in config order) plus the named inference point each
/// candidate serves. Roles name *what the request needs*; `models` stays the
/// fleet inventory of *what exists*. Resolution to a concrete
/// `base:qualifier` target happens per request at dispatch time, never at
/// boot, so lazy models load correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleEntry {
    /// Candidate model keys, in config order.
    #[serde(default)]
    pub models: Vec<String>,
    /// Named inference point each candidate serves. `None` leaves the
    /// qualifier to the entry default / bare key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub routes: HashMap<String, RouteRef>,
    pub models: HashMap<String, ModelEntry>,
    pub model_groups: HashMap<String, ModelGroup>,
    pub system_prompt: String,
    pub safety_threshold: f64,
    pub default_route: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_matrix: Option<ScoreMatrix>,
    /// Registry keys of the configured in-process onnx roles (e.g. `onnx/llm`).
    /// A `model_groups` member that names one of these is a valid dispatch
    /// target served by the onnx `ChatBackend` — it is not a `models` entry, so
    /// the route resolver treats it specially. Populated at config-build time
    /// from `RouterConfig.onnx`; empty for a config with no onnx fleet.
    #[serde(default)]
    pub onnx_keys: BTreeSet<String>,
    /// Routing-vocabulary table (mirrors `RouterConfig.roles`): role name →
    /// candidate model keys + the inference point each candidate serves.
    /// Absent (the default) leaves group members untouched.
    #[serde(default)]
    pub roles: HashMap<String, RoleEntry>,
}

impl RoutingConfig {
    /// Expand a group's raw members through the roles table: a bare member
    /// naming a role fans out to the role's candidate keys (config order);
    /// sentinel members (`last`/`any`), qualified members (`base:point`), and
    /// unknown literals pass through untouched for the downstream stage
    /// (sentinel expansion / explicit-point / fail-closed lookup) to own.
    /// Without roles the expansion is identity. A bare role name shadows a
    /// same-named model key — address the model directly with a qualified
    /// member when both exist.
    pub fn role_expanded_members(&self, group: &str) -> Vec<String> {
        let Some(group_cfg) = self.model_groups.get(group) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(group_cfg.models().len());
        for member in group_cfg.models() {
            if member.contains(':') || member == "last" || member == "any" {
                out.push(member.clone());
            } else if let Some(role) = self.roles.get(member.as_str()) {
                out.extend(role.models.iter().cloned());
            } else {
                out.push(member.clone());
            }
        }
        out
    }

    /// The `model_group` a route resolves through — the route's own `RouteRef`
    /// group, or the default route's group when the route is unknown (mirrors
    /// the `resolve_route` lookup at the top of [`Self::resolve_route`]).
    ///
    /// The target-matching ladder uses this to find the ordered candidate
    /// list for a resolved route; `None` when neither the route nor the
    /// default route has a group entry.
    pub fn route_group(&self, route: &str) -> Option<&str> {
        self.routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))
            .map(|r| r.group.as_str())
    }

    /// Resolve a possibly-qualified model key (`base:qualifier`) to its
    /// `ModelEntry`. A bare key resolves directly; a qualified key strips the
    /// qualifier to find the owning model's entry. `None` when the base key has
    /// no `models` entry.
    pub fn entry_for_key(&self, key: &str) -> Option<&ModelEntry> {
        let (base, _) = split_model_key(key);
        self.models.get(base)
    }

    /// Build the dispatch `RoutingTarget` for a possibly-qualified model key
    /// (`base:qualifier`). A qualifier targets the named instance/group of the
    /// base model (`from_model_entry_instance`); `latest` and bare keys resolve
    /// to the entry's default dispatch point (`from_model_entry`). This is the
    /// canonical builder for `model_groups` members, so a group can pin a
    /// specific instance (e.g. `lfm2.5-2.6b:default`).
    pub fn target_for_key(&self, key: &str) -> Option<RoutingTarget> {
        let entry = self.entry_for_key(key)?;
        let (base, _) = split_model_key(key);
        Some(match split_model_key(key).1 {
            Some("latest") | None => RoutingTarget::from_model_entry(base, entry),
            Some(qualifier) => RoutingTarget::from_model_entry_instance(base, entry, qualifier),
        })
    }

    /// Resolve a route name to a fully-populated typed `RoutingTarget`.
    ///
    /// Reuses `resolve_route` (cheapest model in the route's group whose
    /// `intelligence` meets `min_complexity`, falling back to the cheapest in
    /// the group) and attaches the resolved group, the `target_name`, and the
    /// ordered `fallbacks`. This is the canonical target builder shared by the
    /// flat classifier stage and the classification-tree engine — a
    /// terminal node dispatches through it unchanged
    pub fn routing_target(&self, route: &str, min_complexity: Option<u8>) -> Option<RoutingTarget> {
        // A route whose group resolves to an in-process onnx role (e.g. the
        // generative `onnx/llm` routing model) dispatches to that role — it is
        // not a `models` entry, so it bypasses the ModelEntry ladder below.
        if let Some(rt) = self.resolve_onnx_route_target(route) {
            return Some(rt);
        }
        let member = self.resolve_route_member(route, min_complexity)?;
        let mut rt = self.target_for_key(&member)?;
        let group = self
            .routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))
            .map_or(String::new(), |r| r.group.clone());
        rt.group = Some(group);
        rt.target_name = Some(route.to_string());
        rt.fallbacks = self
            .all_dispatch_targets(route, min_complexity)
            .into_iter()
            .skip(1) // skip the primary (already included)
            .filter_map(|(name, _)| self.target_for_key(&name))
            .collect();
        Some(rt)
    }

    /// Whether the route's group resolves to a configured in-process onnx role,
    /// and if so build the onnx `RoutingTarget` for it. `None` when the route
    /// (or its group) does not name an onnx role key. The route's `group` and
    /// `target_name` are attached so downstream validation (intent→model_group,
    /// route name) sees the same shape as a ModelEntry-resolved target.
    fn resolve_onnx_route_target(&self, route: &str) -> Option<RoutingTarget> {
        let route_ref = self
            .routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))?;
        let key = self
            .role_expanded_members(&route_ref.group)
            .into_iter()
            .find(|k| {
                let (base, _) = split_model_key(k);
                self.onnx_keys.contains(base)
            })?;
        let (base, qualifier) = split_model_key(&key);
        let mut rt = RoutingTarget::from_onnx_role(base);
        if let Some(q) = qualifier {
            rt.instance = Some(q.to_string());
        }
        rt.group = Some(route_ref.group.clone());
        rt.target_name = Some(route.to_string());
        Some(rt)
    }

    pub fn resolve_route(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<(&ModelEntry, String)> {
        // A route whose group resolves to an in-process onnx role has no
        // `models` entry; it is served by the onnx backend, so there is no
        // ModelEntry to return here.
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let route_ref =
            match route_ref {
                Some(r) => Some(r),
                None => {
                    return self.models.get(route_name).map(|entry| {
                    let name = entry.name.clone().unwrap_or_else(|| route_name.to_string());
                    tracing::info!(target: "router.config", route = %route_name, model = %name,
                        "route resolved as direct model"
                    );
                    (entry, name)
                }).or_else(|| {
                    tracing::warn!(target: "router.config", route = %route_name,
                        default = %self.default_route,
                        "no route or model found for target"
                    );
                    None
                });
                }
            };

        let route_ref = route_ref?;

        if self.resolve_onnx_route_target(route_name).is_some() {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "route resolved to an onnx role (no ModelEntry)");
            return None;
        }

        let member = self.resolve_route_member(route_name, min_complexity)?;
        let entry = self.entry_for_key(&member)?;
        let name = entry.name.clone().unwrap_or_else(|| split_model_key(&member).0.to_string());
        tracing::info!(target: "router.config", route = %route_name, model = %name, member = %member, "route resolved");
        Some((entry, name))
    }

    /// Choose the `model_groups` member a route dispatches to: the cheapest
    /// member whose base `models` entry's `intelligence` meets `min_complexity`,
    /// else the cheapest member in the group. Returns the full member key
    /// (possibly qualified, e.g. `lfm2.5-2.6b:default`) so the caller can
    /// preserve any instance qualifier. Role members fan out to their
    /// candidate keys first; availability sentinels (`last`/`any`) have no
    /// meaning on this static path (no recency/liveness) and are skipped —
    /// the dispatch climb owns them.
    fn resolve_route_member(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<String> {
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route))?;
        if !self.model_groups.contains_key(route_ref.group.as_str()) {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "model group not found for route");
            return None;
        }
        let expanded = self.role_expanded_members(&route_ref.group);
        let model_keys: Vec<&str> = expanded
            .iter()
            .map(String::as_str)
            .filter(|m| *m != "last" && *m != "any")
            .collect();

        tracing::debug!(target: "router.config",
            route = %route_name,
            group = %route_ref.group,
            model_count = model_keys.len(),
            min_complexity = ?min_complexity,
            "resolving route"
        );

        let passing: Vec<&&str> = model_keys
            .iter()
            .filter(|n| {
                self.entry_for_key(n)
                    .is_some_and(|m| m.intelligence >= min_complexity.unwrap_or(0))
            })
            .collect();

        let cheapest = |a: &&&str, b: &&&str| {
            let (ca, cb) = (
                self.entry_for_key(a).map_or(f64::MAX, |m| m.cost_input + m.cost_output),
                self.entry_for_key(b).map_or(f64::MAX, |m| m.cost_input + m.cost_output),
            );
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        };

        if passing.is_empty() {
            tracing::debug!(target: "router.config", route = %route_name, "no candidates passed complexity filter, falling back to cheapest in group");
            model_keys.iter().min_by(cheapest).map(|s| (*s).to_string())
        } else {
            let entry_key = passing.into_iter().min_by(cheapest)?;
            tracing::info!(target: "router.config", route = %route_name, model = %entry_key, "route resolved (complexity match)");
            Some((*entry_key).to_string())
        }
    }

    /// Return ALL available dispatch targets across all model groups ordered by
    /// dispatch preference:
    ///
    /// 1. Models from the resolved route's own group (cheapest first)
    /// 2. Models from other groups sorted by intelligence proximity to the
    ///    target complexity (closest first, cheapest tie-break)
    ///
    /// When the primary target fails (rate-limited, timeout, etc.) the caller
    /// can iterate this list to find a working model.
    pub fn all_dispatch_targets(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Vec<(String, ModelEntry)> {
        // Resolve the route to find its group
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let primary_group = route_ref.map(|r| r.group.as_str());

        // Collect all (model_key, model_entry) with resolved names
        let mut seen = std::collections::HashSet::new();
        let mut entries: Vec<(String, ModelEntry, f64)> = Vec::new();

        let target_intelligence = f64::from(min_complexity.unwrap_or(0));

        for group_key in self.model_groups.keys() {
            let is_primary = primary_group == Some(group_key.as_str());
            // Role members fan out to candidate keys; sentinels and unknown
            // literals fall out below through the entry lookup, as today.
            for model_key in self.role_expanded_members(group_key) {
                if !seen.insert(model_key.clone()) {
                    continue;
                }
                if let Some(entry) = self.entry_for_key(&model_key) {
                    // Keep the full (possibly-qualified) member key so the
                    // caller can preserve the instance qualifier when building
                    // the fallback target.
                    let name = model_key.clone();
                    // Compute distance from target intelligence for cross-group sorting
                    let dist = if is_primary {
                        -f64::from(entry.intelligence) // primary group: negative so they sort first
                    } else {
                        (f64::from(entry.intelligence) - target_intelligence).abs()
                    };
                    entries.push((name, entry.clone(), dist));
                }
            }
        }

        // Sort: primary group first (dist < 0), then by intelligence proximity,
        // then by cost for tie-breaking
        entries.sort_by(|a, b| {
            let a_primary = a.2 < 0.0;
            let b_primary = b.2 < 0.0;
            match (a_primary, b_primary) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => {
                    let d = a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal);
                    if d == Ordering::Equal {
                        let cost_a = a.1.cost_input + a.1.cost_output;
                        let cost_b = b.1.cost_input + b.1.cost_output;
                        cost_a.partial_cmp(&cost_b).unwrap_or(Ordering::Equal)
                    } else {
                        d
                    }
                }
            }
        });

        entries
            .into_iter()
            .map(|(name, entry, _)| (name, entry))
            .collect()
    }
}
#[cfg(test)]
#[path = "../../tests/config_routing.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/routing_role_golden.rs"]
mod routing_role_golden_tests;
