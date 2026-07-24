//! `Persona`: the identity/character that makes an agent "that agent".
//!
//! Name, role, character, traits, and identity injection for the model. Persona
//! is versioned (`version` increments on change), so identity evolution is trackable.
//!
//! # Input sanitization
//!
//! [`Persona::validate()`] rejects control characters (\x00–\x1F except TAB) and
//! newlines in `name`, `role`, and `traits`. This prevents log-forging and basic
//! prompt-injection via persona fields. It does NOT protect against semantic
//! prompt injection through `description` or `system_prompt` — those fields are
//! free-form model instructions by design. The residual prompt-injection boundary
//! is documented in this module's source and in the crate-level security notes.
//!
//! ## Residual prompt-injection boundary
//!
//! `description` and `system_prompt` are intentionally not sanitized beyond
//! trimming — they are model-facing free text where any character restriction
//! would be semantically wrong (markdown, code examples, etc.). A model that
//! interprets "ignore previous instructions" in a description as an override
//! is a model-level concern, not an input-validation one. The boundary is:
//!
//! - **Structural fields** (`name`, `role`, `traits`): validated — no control
//!   chars, no newlines. These are identifiers, not free text.
//! - **Semantic fields** (`description`, `system_prompt`): NOT validated for
//!   content — they are model instructions. Users who set these accept the
//!   responsibility for what the model does with them.
//! - **`extra`**: same as semantic fields — identity injection lines from
//!   role presets, assumed trusted.

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

    /// Validates structural persona fields, rejecting control characters and
    /// newlines in `name`, `role`, and `traits`.
    ///
    /// Control characters (\x00–\x1F except TAB \x09) and newlines (\n, \r)
    /// are rejected because they enable log-forging and can disrupt persona-based
    /// identity injection. TAB is permitted (names like "Tab\tSmith" are odd but
    /// not harmful). Empty/whitespace-only values are also rejected.
    ///
    /// Returns a list of field-specific error descriptions (empty = valid).
    pub fn validate(&self) -> Vec<&'static str> {
        let mut errors = Vec::new();
        Self::validate_field("name", &self.name, &mut errors);
        Self::validate_field("role", &self.role, &mut errors);
        for t in &self.traits {
            Self::validate_field("trait", t, &mut errors);
        }
        errors
    }

    /// Validates a single persona field value: rejects control chars (except TAB)
    /// and newlines, and rejects empty/whitespace-only strings.
    fn validate_field(label: &'static str, value: &str, errors: &mut Vec<&'static str>) {
        if value.trim().is_empty() {
            errors.push(label);
            return;
        }
        for ch in value.chars() {
            if ch == '\n' || ch == '\r' {
                errors.push(label);
                return;
            }
            // Control chars 0x00–0x1F except TAB (0x09).
            if ch as u32 <= 0x1F && ch != '\t' {
                errors.push(label);
                return;
            }
        }
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

    // ── Persona::validate() sanitization tests ────────────────────────────

    #[test]
    fn validate_rejects_newline_in_name() {
        let p = Persona::new("Aria\nEvil", "role");
        let bad = p.validate();
        assert!(
            bad.contains(&"name"),
            "newline in name should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_rejects_carriage_return_in_role() {
        let p = Persona::new("name", "role\rEvil");
        let bad = p.validate();
        assert!(
            bad.contains(&"role"),
            "CR in role should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_rejects_control_char_in_trait() {
        let p = Persona::new("name", "role").with_trait("curious\x08");
        let bad = p.validate();
        assert!(
            bad.contains(&"trait"),
            "backspace in trait should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_rejects_null_byte_in_name() {
        let p = Persona::new("Aria\x00Evil", "role");
        let bad = p.validate();
        assert!(
            bad.contains(&"name"),
            "null byte in name should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_allows_tab_in_name() {
        let p = Persona::new("Aria\tSmith", "role");
        let bad = p.validate();
        assert!(bad.is_empty(), "TAB should be allowed in name: {bad:?}");
    }

    #[test]
    fn validate_rejects_empty_name() {
        let p = Persona::new("", "role");
        let bad = p.validate();
        assert!(
            bad.contains(&"name"),
            "empty name should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_rejects_whitespace_only_role() {
        let p = Persona::new("name", "   ");
        let bad = p.validate();
        assert!(
            bad.contains(&"role"),
            "whitespace-only role should be rejected: {bad:?}"
        );
    }

    #[test]
    fn validate_rejects_multiple_bad_fields() {
        let p = Persona::new("bad\n", "bad\x01").with_trait("t\rex");
        let bad = p.validate();
        assert!(
            bad.contains(&"name") && bad.contains(&"role") && bad.contains(&"trait"),
            "all three fields should be flagged: {bad:?}"
        );
    }

    #[test]
    fn validate_passes_for_clean_persona() {
        let p = Persona::new("Aria", "researcher").with_trait("curious");
        let bad = p.validate();
        assert!(bad.is_empty(), "clean persona should pass: {bad:?}");
    }
}
