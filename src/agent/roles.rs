//! Role presets: data-driven identity templates for common agent types.
//!
//! Each preset defines a role description, traits, and `identity_extra` —
//! terse verification-minded instructions appended to the persona's identity
//! prompt. Presets are prompts, not hardcoded behavior; the work loop is unchanged.

/// A role preset: static identity template.
#[derive(Clone, Debug)]
pub struct Role {
    /// Preset name (lookup key): "backend", "frontend", "reviewer", "pm".
    pub name: &'static str,
    /// Role description (e.g. "backend engineer").
    pub role: &'static str,
    /// Character traits for the persona.
    pub traits: &'static [&'static str],
    /// Verification-minded identity extra appended to `Persona::identity_prompt()`.
    /// Terse, concrete, English — tells the model what to verify before claiming done.
    pub identity_extra: &'static str,
}

/// Backend engineer preset: builds and tests server-side code.
const BACKEND: Role = Role {
    name: "backend",
    role: "backend engineer",
    traits: &["methodical", "verification-minded", "pragmatic"],
    identity_extra:
        "Run the project's tests before claiming done; read failures fully before editing. \
                     Verify compile warnings are resolved. Prefer minimal diffs.",
};

/// Frontend engineer preset: builds UI components with accessibility basics.
const FRONTEND: Role = Role {
    name: "frontend",
    role: "frontend engineer",
    traits: &["detail-oriented", "user-focused", "accessible"],
    identity_extra: "Verify components render without console errors. Check accessibility: \
                     labels, keyboard nav, color contrast. Prefer small, composable components.",
};

/// Reviewer preset: adversarial read-only mindset — finds gaps and contradictions.
const REVIEWER: Role = Role {
    name: "reviewer",
    role: "code reviewer",
    traits: &["adversarial", "thorough", "objective"],
    identity_extra: "Read code critically: look for logic gaps, missing error handling, \
                     untested edge cases, and security issues. Do not suggest edits — only report findings.",
};

/// PM (project manager) preset: decomposes goals into subtasks.
const PM: Role = Role {
    name: "pm",
    role: "project manager",
    traits: &["structured", "delegating", "outcome-focused"],
    identity_extra: "Decompose goals into concrete subtasks with verification criteria. \
                     Assign each subtask to a named agent. Summarize completed work; flag failures clearly.",
};

/// All built-in role presets.
pub fn presets() -> &'static [Role] {
    &[BACKEND, FRONTEND, REVIEWER, PM]
}

/// Look up a preset by name (case-insensitive).
pub fn preset(name: &str) -> Option<Role> {
    presets()
        .iter()
        .find(|r| r.name.eq_ignore_ascii_case(name))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_lookup_by_name() {
        assert!(preset("backend").is_some());
        assert!(preset("frontend").is_some());
        assert!(preset("reviewer").is_some());
        assert!(preset("pm").is_some());
        assert!(preset("nonexistent").is_none());
    }

    #[test]
    fn preset_lookup_case_insensitive() {
        assert!(preset("Backend").is_some());
        assert!(preset("FRONTEND").is_some());
    }

    #[test]
    fn all_presets_have_nonempty_extra() {
        for r in presets() {
            assert!(
                !r.identity_extra.is_empty(),
                "preset '{}' has empty identity_extra",
                r.name
            );
        }
    }

    #[test]
    fn backend_preset_content() {
        let b = preset("backend").unwrap();
        assert_eq!(b.role, "backend engineer");
        assert!(b.identity_extra.contains("Run the project's tests"));
        assert!(b.traits.contains(&"verification-minded"));
    }

    #[test]
    fn reviewer_preset_content() {
        let r = preset("reviewer").unwrap();
        assert_eq!(r.role, "code reviewer");
        assert!(r.identity_extra.contains("Read code critically"));
    }
}
