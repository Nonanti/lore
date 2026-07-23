//! `Persona`: the identity/character that makes an agent "that agent".
//!
//! Name, role, character, traits, and identity injection for the model. Persona
//! is versioned (`version` increments on change), so identity evolution is trackable.

use serde::{Deserialize, Serialize};

/// Identity and character of an agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persona {
    /// Name (e.g. "Aria").
    pub name: String,
    /// Role (e.g. "researcher").
    pub role: String,
    /// Free-text character description.
    pub description: String,
    /// Character traits (e.g. ["curious", "cautious"]).
    pub traits: Vec<String>,
    /// Extra/custom system instruction (optional).
    pub system_prompt: String,
    /// Additional identity lines appended to `identity_prompt()` (additive serde).
    /// Populated from role presets' `identity_extra` or manual additions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra: Vec<String>,
    /// Persona version (increments when identity changes).
    pub version: u32,
}

impl Persona {
    /// Creates a new persona with name and role (version = 1).
    pub fn new(name: impl Into<String>, role: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            role: role.into(),
            description: String::new(),
            traits: Vec::new(),
            system_prompt: String::new(),
            extra: Vec::new(),
            version: 1,
        }
    }

    /// Sets the character description (builder).
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Adds a single trait (builder).
    pub fn with_trait(mut self, t: impl Into<String>) -> Self {
        self.traits.push(t.into());
        self
    }

    /// Sets the trait list in bulk (builder).
    pub fn with_traits(mut self, ts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.traits = ts.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the extra system instruction (builder).
    pub fn with_system_prompt(mut self, s: impl Into<String>) -> Self {
        self.system_prompt = s.into();
        self
    }

    /// Adds identity extra lines (from role presets or manual additions).
    /// These are appended after traits and system_prompt in `identity_prompt()`.
    pub fn with_extra(mut self, lines: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extra.extend(lines.into_iter().map(Into::into));
        self
    }

    /// Marks the persona as revised, increments the version.
    pub fn revised(mut self) -> Self {
        self.version += 1;
        self
    }

    /// Produces the identity injection for the model.
    pub fn identity_prompt(&self) -> String {
        let mut p = format!("You are {}, a {}.", self.name, self.role);
        if !self.description.is_empty() {
            p.push(' ');
            p.push_str(&self.description);
        }
        if !self.traits.is_empty() {
            p.push_str(&format!(" Your traits: {}.", self.traits.join(", ")));
        }
        if !self.system_prompt.is_empty() {
            p.push(' ');
            p.push_str(&self.system_prompt);
        }
        if !self.extra.is_empty() {
            p.push(' ');
            p.push_str(&self.extra.join(" "));
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_prompt_contains_identity_fields() {
        let p = Persona::new("Aria", "researcher")
            .with_description("You seek knowledge.")
            .with_trait("curious")
            .with_trait("cautious");
        let ip = p.identity_prompt();
        assert!(ip.contains("Aria"));
        assert!(ip.contains("researcher"));
        assert!(ip.contains("You seek knowledge."));
        assert!(ip.contains("curious"));
        assert!(ip.contains("cautious"));
    }

    #[test]
    fn version_starts_at_one_and_bumps_on_revise() {
        let p = Persona::new("Aria", "researcher");
        assert_eq!(p.version, 1);
        assert_eq!(p.revised().version, 2);
    }

    #[test]
    fn with_traits_replaces_list() {
        let p = Persona::new("X", "y").with_traits(["a", "b", "c"]);
        assert_eq!(p.traits, vec!["a", "b", "c"]);
    }
}
