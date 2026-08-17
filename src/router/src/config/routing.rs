//! Route resolution and routing configuration types.
//! `RoutingConfig` is the resolved routing table used by the classifier stage.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ModelEntry;
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
}

impl RoutingConfig {
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

    /// Resolve a route name to a fully-populated typed `RoutingTarget`.
    ///
    /// Reuses `resolve_route` (cheapest model in the route's group whose
    /// `intelligence` meets `min_complexity`, falling back to the cheapest in
    /// the group) and attaches the resolved group, the `target_name`, and the
    /// ordered `fallbacks`. This is the canonical target builder shared by the
    /// flat classifier stage and the classification-tree engine — a
    /// terminal node dispatches through it unchanged
    pub fn routing_target(&self, route: &str, min_complexity: Option<u8>) -> Option<RoutingTarget> {
        let (entry, model_name) = self.resolve_route(route, min_complexity)?;
        let group = self
            .routes
            .get(route)
            .or_else(|| self.routes.get(&self.default_route))
            .map_or(String::new(), |r| r.group.clone());

        let mut rt = RoutingTarget::from_model_entry(&model_name, entry);
        rt.group = Some(group);
        rt.target_name = Some(route.to_string());
        rt.fallbacks = self
            .all_dispatch_targets(route, min_complexity)
            .into_iter()
            .skip(1) // skip the primary (already included)
            .map(|(name, entry)| RoutingTarget::from_model_entry(&name, &entry))
            .collect();
        Some(rt)
    }

    pub fn resolve_route(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<(&ModelEntry, String)> {
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

        let model_names = self.model_groups.get(route_ref.group.as_str());
        let Some(model_names) = model_names else {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "model group not found for route");
            return None;
        };
        let model_keys = model_names.models();

        tracing::debug!(target: "router.config",
            route = %route_name,
            group = %route_ref.group,
            model_count = model_keys.len(),
            min_complexity = ?min_complexity,
            "resolving route"
        );

        let candidates: Vec<(&String, &ModelEntry)> = model_keys
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .filter(|(_, m)| m.intelligence >= min_complexity.unwrap_or(0))
            .collect();

        if candidates.is_empty() {
            tracing::debug!(target: "router.config", route = %route_name, "no candidates passed complexity filter, falling back to cheapest in group");
            model_keys
                .iter()
                .filter_map(|n| self.models.get(n).map(|m| (n, m)))
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(entry_key, entry)| {
                    let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
                    tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved (cheapest fallback)");
                    (entry, name)
                })
        } else {
            let (entry_key, entry) = candidates.into_iter().min_by(|(_, a), (_, b)| {
                (a.cost_input + a.cost_output)
                    .partial_cmp(&(b.cost_input + b.cost_output))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
            tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved");
            Some((entry, name))
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

        for (group_key, group) in &self.model_groups {
            let is_primary = primary_group == Some(group_key.as_str());
            for model_key in group.models() {
                if !seen.insert(model_key.clone()) {
                    continue;
                }
                if let Some(entry) = self.models.get(model_key) {
                    let name = entry.name.clone().unwrap_or_else(|| model_key.clone());
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
mod tests {
    use super::*;

    fn sample_config() -> RoutingConfig {
        serde_json::from_value(serde_json::json!({
            "routes": {
                "fast": {"group": "fast", "pipelines": ["default"], "description": "fast route"},
                "smart": {"group": "smart", "pipelines": ["default"], "always_route": true},
            },
            "models": {
                "tiny": {"endpoint": "http://a", "name": "tiny", "intelligence": 1,
                         "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.0, "speed": 10},
                "big": {"endpoint": "http://b", "name": "big", "intelligence": 5,
                        "cost_input": 9.0, "cost_output": 9.0, "cost_cached_read": 0.0, "speed": 10},
                "huge": {"endpoint": "http://c", "name": "huge", "intelligence": 8,
                         "cost_input": 20.0, "cost_output": 20.0, "cost_cached_read": 0.0, "speed": 10},
            },
            "model_groups": {"fast": ["tiny"], "smart": ["big", "huge"]},
            "system_prompt": "sys",
            "safety_threshold": 0.5,
            "default_route": "fast",
        }))
        .expect("valid routing config")
    }

    #[test]
    fn route_ref_serde_round_trip() {
        let r: RouteRef = serde_json::from_value(serde_json::json!({
            "group": "g", "pipelines": ["a"], "description": "d", "always_route": true
        }))
        .expect("deserialize");
        assert_eq!(r.group, "g");
        assert_eq!(r.pipelines, vec!["a"]);
        assert!(r.always_route);
        let back: RouteRef =
            serde_json::from_str(&serde_json::to_string(&r).expect("serialize")).expect("round trip");
        assert_eq!(back.group, "g");
    }

    #[test]
    fn route_ref_pipelines_defaults_to_default() {
        let r: RouteRef = serde_json::from_value(serde_json::json!({"group": "g"})).expect("deserialize");
        assert_eq!(r.pipelines, vec!["default"]);
        assert!(!r.always_route);
    }

    #[test]
    fn route_group_resolves_route_or_default() {
        let c = sample_config();
        assert_eq!(c.route_group("fast"), Some("fast"));
        assert_eq!(c.route_group("smart"), Some("smart"));
        // Unknown route falls back to the default route's group.
        assert_eq!(c.route_group("nope"), Some("fast"));
    }

    #[test]
    fn route_group_none_when_no_groups() {
        let c: RoutingConfig = serde_json::from_value(serde_json::json!({
            "routes": {"r": {"group": "missing"}},
            "models": {},
            "model_groups": {},
            "system_prompt": "s",
            "safety_threshold": 0.5,
            "default_route": "r",
        }))
        .expect("config");
        assert_eq!(c.route_group("r"), Some("missing"));
        // Unknown route and default route both groupless -> None.
        let c2: RoutingConfig = serde_json::from_value(serde_json::json!({
            "routes": {},
            "models": {},
            "model_groups": {},
            "system_prompt": "s",
            "safety_threshold": 0.5,
            "default_route": "missing",
        }))
        .expect("config");
        assert_eq!(c2.route_group("r"), None);
    }

    #[test]
    fn resolve_route_picks_cheapest_passing_complexity() {
        let c = sample_config();
        // smart group has big(int 5, cost 18) and huge(int 8, cost 40).
        let (entry, name) = c.resolve_route("smart", Some(6)).expect("resolve");
        // big is int 5 < 6 so filtered out; huge (8 >= 6) is the only candidate.
        assert_eq!(name, "huge");
        assert_eq!(entry.intelligence, 8);
        // No min_complexity -> cheapest eligible (big, cost 18).
        let (_, name) = c.resolve_route("smart", None).expect("resolve");
        assert_eq!(name, "big");
    }

    #[test]
    fn resolve_route_falls_back_to_cheapest_when_none_pass() {
        let c = sample_config();
        // min_complexity 9 filters everything; cheapest in group wins.
        let (entry, _) = c.resolve_route("smart", Some(9)).expect("resolve");
        assert_eq!(entry.intelligence, 5, "big is cheapest in the smart group");
    }

    #[test]
    fn resolve_route_unknown_route_uses_default_group() {
        let c = sample_config();
        let (_, name) = c.resolve_route("nope", None).expect("resolve via default");
        assert_eq!(name, "tiny");
    }

    #[test]
    fn resolve_route_direct_model_by_name() {
        // A model name that is not a route resolves as a direct model only
        // when neither the route nor the default route exists.
        let c: RoutingConfig = serde_json::from_value(serde_json::json!({
            "routes": {},
            "models": {
                "big": {"endpoint": "http://b", "name": "big", "intelligence": 5,
                        "cost_input": 9.0, "cost_output": 9.0, "cost_cached_read": 0.0, "speed": 10},
            },
            "model_groups": {},
            "system_prompt": "s",
            "safety_threshold": 0.5,
            "default_route": "missing",
        }))
        .expect("config");
        let (entry, name) = c.resolve_route("big", None).expect("resolve direct");
        assert_eq!(name, "big");
        assert_eq!(entry.intelligence, 5);
        // A completely unknown name with no default route -> None.
        assert!(c.resolve_route("nope", None).is_none());
    }

    #[test]
    fn resolve_route_missing_group_returns_none() {
        let c: RoutingConfig = serde_json::from_value(serde_json::json!({
            "routes": {"r": {"group": "missing"}},
            "models": {"m": {"endpoint": "http://a", "name": "m", "intelligence": 1,
                             "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.0, "speed": 10}},
            "model_groups": {},
            "system_prompt": "s",
            "safety_threshold": 0.5,
            "default_route": "r",
        }))
        .expect("config");
        assert!(c.resolve_route("r", None).is_none());
    }

    #[test]
    fn routing_target_attaches_group_target_and_fallbacks() {
        let c = sample_config();
        let rt = c.routing_target("smart", None).expect("routing target");
        assert_eq!(rt.model, "big");
        assert_eq!(rt.group.as_deref(), Some("smart"));
        assert_eq!(rt.target_name.as_deref(), Some("smart"));
        // Primary (big) is skipped; fallbacks are the remaining dispatch
        // targets in preference order.
        let fallback_models: Vec<&str> = rt.fallbacks.iter().map(|f| f.model.as_str()).collect();
        assert!(fallback_models.contains(&"big"));
        assert!(fallback_models.contains(&"tiny"));
    }

    #[test]
    fn routing_target_unknown_route_uses_default() {
        let c = sample_config();
        let rt = c.routing_target("nope", None).expect("routing target");
        assert_eq!(rt.model, "tiny");
        assert_eq!(rt.group.as_deref(), Some("fast"));
    }

    #[test]
    fn all_dispatch_targets_orders_primary_group_first() {
        let c = sample_config();
        let targets = c.all_dispatch_targets("smart", None);
        let names: Vec<&str> = targets.iter().map(|(n, _)| n.as_str()).collect();
        // Primary group (smart) first, ordered by intelligence descending
        // (huge int 8 before big int 5), then other groups (tiny).
        assert_eq!(&names[..2], &["huge", "big"]);
        assert!(names.contains(&"tiny"));
    }
}
