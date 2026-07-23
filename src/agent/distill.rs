//! Work-task distillation: extract durable facts from a completed task.
//!
//! [`Agent::distill_work`] makes ONE model call (cheap prompt: final answer +
//! verify-log tail, not the whole transcript) and stores up to 3 semantic
//! memories via the normal `remember` path. Model/parse errors return `Ok(0)`
//! with a `tracing::warn` — learning is best-effort, never fails the task.

use crate::agent::Agent;
use crate::error::Result;
use crate::memory::{Memory, SemanticCat};
use crate::model::Prompt;

use super::work::{WorkReport, WorkSpec};

/// Importance of distilled semantic facts (explicit, deliberate knowledge).
const DISTILL_IMPORTANCE: f32 = 0.5;

/// Maximum items distilled from a single task (cost limit).
const DISTILL_MAX_ITEMS: usize = 3;

/// Cap on verify-log tail included in the distillation prompt (8 KiB).
const DISTILL_VERIFY_TAIL_CAP: usize = 8 * 1024;

/// Truncation marker for distillation prompt.
const DISTILL_TAIL_MARKER: &str = "\n[... truncated]";

/// Keep only the last `cap` bytes of a string, respecting char boundaries.
fn tail_cap(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let start = s.len() - cap;
    let mut i = start;
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}{}", DISTILL_TAIL_MARKER, &s[i..])
}

/// A single distilled item parsed from the model's JSON response.
#[derive(Clone, Debug, serde::Deserialize)]
struct DistillItem {
    /// Kind: convention, constraint, or fact.
    kind: String,
    /// Short title.
    title: String,
    /// Detail body.
    body: String,
}

/// Lenient first-complete-JSON parse: accepts `[...]` directly or
/// `{"items":[...]}` wrapper. Returns parsed items or None on failure.
fn parse_distill_json(raw: &str) -> Option<Vec<DistillItem>> {
    let trimmed = raw.trim();

    // Try direct array parse: `[...]`
    if let Ok(items) = serde_json::from_str::<Vec<DistillItem>>(trimmed) {
        return Some(items);
    }

    // Try wrapper: `{"items":[...]}` — lenient alternative format.
    if let Ok(wrapper) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(arr) = wrapper.get("items") {
            if let Ok(items) = serde_json::from_value::<Vec<DistillItem>>(arr.clone()) {
                return Some(items);
            }
        }
    }

    None
}

/// Map a distill kind string to the corresponding SemanticCat.
fn kind_to_cat(kind: &str) -> SemanticCat {
    match kind.to_lowercase() {
        k if k.starts_with("convention") => SemanticCat::Convention,
        k if k.starts_with("constraint") => SemanticCat::Constraint,
        _ => SemanticCat::Fact,
    }
}

impl Agent {
    /// Distills durable knowledge from a completed work task.
    ///
    /// Makes ONE model call with a cheap prompt (final answer + verify-log
    /// tail), asks for at most 3 items as JSON. Parsed items are stored as
    /// `MemoryKind::Semantic` via [`Agent::remember`] (existing dedup/merge
    /// machinery applies). Returns the count of stored items.
    ///
    /// Model errors or unparseable JSON → `Ok(0)` with a tracing::warn;
    /// **never** fails the task — learning is best-effort.
    pub async fn distill_work(&self, spec: &WorkSpec, report: &WorkReport) -> Result<usize> {
        let verify_tail = tail_cap(&report.verify_log, DISTILL_VERIFY_TAIL_CAP);

        let prompt = Prompt {
            system: format!(
                "{}\n\n\
                From this completed task, extract durable facts worth remembering for \
                future work in this project: conventions, gotchas, commands that worked. \
                Return JSON: \
                [{{\"kind\":\"convention\"|\"constraint\"|\"fact\",\"title\":\"...\",\"body\":\"...\"}}]. \
                At most {DISTILL_MAX_ITEMS} items; return an empty list [] if nothing durable.",
                self.persona.identity_prompt()
            ),
            context: vec![],
            history: vec![],
            user: format!(
                "Goal: {}\nFinal answer: {}\nVerify log (tail):\n{}\nIterations: {}",
                spec.goal, report.answer, verify_tail, report.iterations
            ),
        };

        let raw = match self.model.complete(&prompt).await {
            Ok(c) => c.text,
            Err(e) => {
                tracing::warn!(error = %e, "distill_work: model call failed — skipping");
                return Ok(0);
            }
        };

        let items = match parse_distill_json(&raw) {
            Some(i) => i,
            None => {
                tracing::warn!(
                    raw = %raw.chars().take(200).collect::<String>(),
                    "distill_work: unparseable JSON — skipping"
                );
                return Ok(0);
            }
        };

        if items.is_empty() {
            return Ok(0);
        }

        let mut stored = 0usize;
        for item in items.iter().take(DISTILL_MAX_ITEMS) {
            let cat = kind_to_cat(&item.kind);
            let statement = if item.body.is_empty() {
                item.title.clone()
            } else {
                format!("{} — {}", item.title, item.body)
            };
            let mem = Memory::semantic(self.scope(), statement, cat)
                .with_importance(DISTILL_IMPORTANCE)
                .with_key(format!(
                    "distilled:task:{}",
                    &spec.goal[..spec.goal.len().min(80)]
                ));
            if let Err(e) = self.remember(mem).await {
                tracing::warn!(error = %e, "distill_work: semantic record could not be written");
                continue;
            }
            stored += 1;
        }

        if stored > 0 {
            tracing::info!(stored, "distill_work: semantic distillation complete");
        }
        Ok(stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, Persona, WorkSpec};
    use crate::memory::{InMemoryStore, MemoryKind, MemoryStore, Query, Tier};
    use crate::model::{Completion, Model, Prompt};
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn make_temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lore-distill-test-{label}-{pid}",
            pid = std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &PathBuf) {
        std::fs::remove_dir_all(dir).ok();
    }

    struct ScriptedModel {
        replies: Mutex<VecDeque<String>>,
    }

    impl ScriptedModel {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for ScriptedModel {
        async fn complete(&self, _p: &Prompt) -> crate::error::Result<Completion> {
            let text = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| "no reply left".into());
            Ok(Completion::new(text))
        }
    }

    fn agent_with_model(model: Arc<dyn Model>) -> Agent {
        let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
        let persona = Persona::new("DistillAgent", "worker");
        Agent::new(persona, store, model)
    }

    fn spec_and_report() -> (WorkSpec, WorkReport) {
        let ws = make_temp_dir("distill-spec");
        let spec = WorkSpec::new(
            "implement feature X",
            ws.clone(),
            vec!["exit 0".to_string()],
        )
        .unwrap();
        let report = WorkReport {
            success: true,
            iterations: 2,
            answer: "feature X implemented".to_string(),
            verify_log: "all tests passed".to_string(),
        };
        (spec, report)
    }

    // ── 2-item JSON → 2 semantic memories of correct kinds ──────────

    #[tokio::test]
    async fn distill_work_two_items_stores_two_semantic_memories() {
        let model = Arc::new(ScriptedModel::new(&[
            r#"[{"kind":"convention","title":"use conventional commits","body":"project uses feat/fix/refactor prefixes"},{"kind":"fact","title":"test framework is cargo test","body":"Rust project verified via cargo test"}]"#,
        ]));
        let agent = agent_with_model(model);
        let (spec, report) = spec_and_report();

        let count = agent.distill_work(&spec, &report).await.unwrap();
        assert_eq!(count, 2, "two items should be stored");

        let sem = agent
            .recall(&Query::new("").tier(Tier::Semantic).limit(10))
            .await
            .unwrap();
        assert_eq!(sem.len(), 2, "two semantic records");

        // Check kinds.
        let kinds: Vec<String> = sem
            .iter()
            .map(|s| {
                if let MemoryKind::Semantic { category, .. } = &s.item.kind {
                    format!("{category:?}")
                } else {
                    "wrong".to_string()
                }
            })
            .collect();
        assert!(
            kinds.contains(&"Convention".to_string()),
            "should contain a Convention: {kinds:?}"
        );
        assert!(
            kinds.contains(&"Fact".to_string()),
            "should contain a Fact: {kinds:?}"
        );

        let (_, report) = spec_and_report();
        cleanup(&report.verify_log.into());
    }

    // ── garbage JSON → Ok(0), no error ──────────────────────────────

    #[tokio::test]
    async fn distill_work_garbage_json_returns_ok_zero() {
        let model = Arc::new(ScriptedModel::new(&["this is not JSON at all!!!"]));
        let agent = agent_with_model(model);
        let (spec, report) = spec_and_report();

        let count = agent.distill_work(&spec, &report).await.unwrap();
        assert_eq!(count, 0, "garbage JSON → Ok(0)");

        // No semantic memories were added.
        let sem = agent
            .recall(&Query::new("").tier(Tier::Semantic).limit(10))
            .await
            .unwrap();
        assert_eq!(sem.len(), 0, "no memories from garbage");
    }

    // ── empty list → Ok(0) ──────────────────────────────────────────

    #[tokio::test]
    async fn distill_work_empty_list_returns_ok_zero() {
        let model = Arc::new(ScriptedModel::new(&["[]"]));
        let agent = agent_with_model(model);
        let (spec, report) = spec_and_report();

        let count = agent.distill_work(&spec, &report).await.unwrap();
        assert_eq!(count, 0, "empty list → Ok(0)");
    }

    // ── wrapper format {"items":[...]} accepted ──────────────────────

    #[tokio::test]
    async fn distill_work_wrapper_format_accepted() {
        let model = Arc::new(ScriptedModel::new(&[
            r#"{"items":[{"kind":"constraint","title":"no unwrap outside tests","body":"use ? operator in production code"}]}"#,
        ]));
        let agent = agent_with_model(model);
        let (spec, report) = spec_and_report();

        let count = agent.distill_work(&spec, &report).await.unwrap();
        assert_eq!(count, 1, "wrapper format → 1 item");

        let sem = agent
            .recall(&Query::new("").tier(Tier::Semantic).limit(10))
            .await
            .unwrap();
        assert_eq!(sem.len(), 1);
        if let MemoryKind::Semantic { category, .. } = &sem[0].item.kind {
            assert_eq!(*category, crate::memory::SemanticCat::Constraint);
        } else {
            panic!("expected semantic");
        }
    }

    // ── model error → Ok(0) ─────────────────────────────────────────

    #[tokio::test]
    async fn distill_work_model_error_returns_ok_zero() {
        struct FailModel;
        #[async_trait::async_trait]
        impl Model for FailModel {
            async fn complete(
                &self,
                _p: &Prompt,
            ) -> crate::error::Result<crate::model::Completion> {
                Err(crate::error::LoreError::Model("connection refused".into()))
            }
        }
        let model: Arc<dyn Model> = Arc::new(FailModel);
        let agent = agent_with_model(model);
        let (spec, report) = spec_and_report();

        let count = agent.distill_work(&spec, &report).await.unwrap();
        assert_eq!(count, 0, "model error → Ok(0)");
    }

    // ── tail_cap helper ──────────────────────────────────────────────

    #[test]
    fn tail_cap_truncates_and_preserves_char_boundary() {
        let short = "hello";
        assert_eq!(tail_cap(short, 1024), short);

        let long = "A".repeat(10000);
        let t = tail_cap(&long, 8192);
        assert!(t.contains(DISTILL_TAIL_MARKER));
        assert!(std::str::from_utf8(t.as_bytes()).is_ok());
    }

    // ── kind_to_cat mapping ──────────────────────────────────────────

    #[test]
    fn kind_to_cat_maps_correctly() {
        assert_eq!(kind_to_cat("convention"), SemanticCat::Convention);
        assert_eq!(kind_to_cat("Convention"), SemanticCat::Convention);
        assert_eq!(kind_to_cat("constraint"), SemanticCat::Constraint);
        assert_eq!(kind_to_cat("fact"), SemanticCat::Fact);
        assert_eq!(kind_to_cat("unknown"), SemanticCat::Fact);
        assert_eq!(kind_to_cat("ConventionFoo"), SemanticCat::Convention);
    }
}
