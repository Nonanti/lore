//! PM agent: decompose goals into subtasks and synthesize child reports.
//!
//! `decompose` prompts the PM model with the goal + roster and extracts
//! subtask JSON leniently (first-complete-JSON-value approach). Unknown
//! agent names → clear error; invalid JSON → one corrective retry → fail.
//!
//! `synthesis_prompt` builds a combined summary from children reports.

use crate::error::{LoreError, Result};
use crate::model::{Model, Prompt};
use crate::task::TaskStore;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// A subtask extracted from PM decomposition.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubtaskSpec {
    /// Agent name to assign the subtask to.
    pub agent: String,
    /// Goal description for the subtask.
    pub goal: String,
    /// Verification criteria for the subtask.
    #[serde(default)]
    pub verify: Vec<String>,
}

/// Decompose a goal into subtask specs by prompting the PM agent model.
///
/// The model receives the goal and the roster of available agents.
/// Returns the list of [`SubtaskSpec`] on success, or an error if:
/// - JSON is invalid after one corrective retry
/// - Agent names are not in the roster
pub async fn decompose(
    model: &Arc<dyn Model>,
    goal: &str,
    roster: &[AgentEntry],
) -> Result<Vec<SubtaskSpec>> {
    let roster_text = roster
        .iter()
        .map(|a| format!("- {}: {}", a.name, a.role))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = Prompt {
        system: "You are a project manager. Decompose the given goal into concrete subtasks assigned to named agents.\n\
                  Return ONLY a JSON array: [{\"agent\": \"<name from roster>\", \"goal\": \"<description>\", \"verify\": [\"<criteria>\"]}]\n\
                  Each agent field must exactly match a name from the roster below. Do not invent agent names.".to_string(),
        context: vec![format!("Available agents (roster):\n{roster_text}")],
        user: goal.to_string(),
        ..Default::default()
    };

    let completion = model.complete(&prompt).await?;
    let specs = parse_subtasks(&completion.text)?;

    // Validate agent names against roster.
    let roster_names: Vec<&str> = roster.iter().map(|a| a.name.as_str()).collect();
    let unknown = specs
        .iter()
        .find(|s| !roster_names.contains(&s.agent.as_str()));
    if let Some(u) = unknown {
        return Err(LoreError::InvalidInput(format!(
            "unknown agent '{}' in decomposition (roster: {})",
            u.agent,
            roster_names.join(", ")
        )));
    }

    Ok(specs)
}

/// Decompose with one corrective retry on invalid JSON.
pub async fn decompose_with_retry(
    model: &Arc<dyn Model>,
    goal: &str,
    roster: &[AgentEntry],
) -> Result<Vec<SubtaskSpec>> {
    let first = decompose(model, goal, roster).await;
    match first {
        Ok(specs) => Ok(specs),
        Err(LoreError::InvalidInput(msg)) if msg.contains("unknown agent") => {
            // Agent validation error is NOT retryable — clear error.
            Err(LoreError::InvalidInput(msg))
        }
        Err(_) => {
            // JSON parse failure → one corrective retry.
            let retry_prompt = Prompt {
                system: "Your previous response was not valid JSON. Return ONLY a valid JSON array: [{\"agent\": \"<name>\", \"goal\": \"<description>\", \"verify\": [\"<criteria>\"]}]\n\
                          No prose, no code fences, no extra text. Just the JSON array.".to_string(),
                user: format!("Original goal: {goal}\n\nReturn the subtask decomposition as pure JSON now."),
                ..Default::default()
            };
            let completion = model.complete(&retry_prompt).await?;
            let specs = parse_subtasks(&completion.text)?;

            // Validate agent names again.
            let roster_names: Vec<&str> = roster.iter().map(|a| a.name.as_str()).collect();
            let unknown = specs
                .iter()
                .find(|s| !roster_names.contains(&s.agent.as_str()));
            if let Some(u) = unknown {
                return Err(LoreError::InvalidInput(format!(
                    "unknown agent '{}' in decomposition (roster: {})",
                    u.agent,
                    roster_names.join(", ")
                )));
            }
            Ok(specs)
        }
    }
}

/// Lenient JSON extraction: first-complete-JSON-value approach.
/// Accepts both a plain JSON array and a wrapped {"subtasks": [...]} object.
/// Also tolerates prose-wrapped content (text before/after the JSON).
fn parse_subtasks(text: &str) -> Result<Vec<SubtaskSpec>> {
    // Try extracting the first complete JSON value from the text.
    for (start, _) in text.match_indices('{') {
        let mut stream =
            serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>();
        let Some(Ok(v)) = stream.next() else {
            continue;
        };

        // Accept {"subtasks": [...]} wrapper.
        if let Some(arr) = v.get("subtasks").and_then(|a| a.as_array()) {
            let specs: Vec<SubtaskSpec> = arr
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect();
            if !specs.is_empty() {
                return Ok(specs);
            }
        }

        // Accept a plain array if the JSON value itself is an array.
        if v.is_array() {
            let specs: Vec<SubtaskSpec> = v
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect();
            if !specs.is_empty() {
                return Ok(specs);
            }
        }
    }

    // Try starting from '[' for a plain array not inside an object.
    for (start, _) in text.match_indices('[') {
        let mut stream =
            serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>();
        let Some(Ok(v)) = stream.next() else {
            continue;
        };
        if v.is_array() {
            let specs: Vec<SubtaskSpec> = v
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect();
            if !specs.is_empty() {
                return Ok(specs);
            }
        }
    }

    Err(LoreError::InvalidInput(
        "PM decomposition returned no valid JSON array of subtasks".to_string(),
    ))
}

/// An agent entry in the roster (name + role) used for decomposition.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentEntry {
    /// Agent name (must match persona file stem).
    pub name: String,
    /// Agent role description.
    pub role: String,
}

/// Build a roster from the agent JSON files in `<data>/agents/*.json`.
pub fn build_roster(data_dir: &Path) -> Result<Vec<AgentEntry>> {
    let agents_dir = data_dir.join("agents");
    if !agents_dir.exists() {
        return Ok(Vec::new());
    }
    let mut roster = Vec::new();
    for entry in std::fs::read_dir(&agents_dir).map_err(|e| LoreError::Storage(e.to_string()))? {
        let path = entry.map_err(|e| LoreError::Storage(e.to_string()))?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let json = std::fs::read_to_string(&path).map_err(|e| LoreError::Storage(e.to_string()))?;
        let rec: serde_json::Value = serde_json::from_str(&json)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let role = rec
            .get("persona")
            .and_then(|p| p.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("unknown")
            .to_string();
        roster.push(AgentEntry { name, role });
    }
    roster.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(roster)
}

/// Check whether a 'reviewer' persona exists in the roster.
pub fn has_reviewer(roster: &[AgentEntry]) -> bool {
    roster.iter().any(|a| a.name == "reviewer")
}

/// Build a synthesis prompt from children task reports.
pub fn synthesis_prompt(children_reports: &[ChildReport]) -> String {
    let reports_text = children_reports
        .iter()
        .map(|c| {
            format!(
                "## Agent: {} (status: {})\nGoal: {}\nReport:\n{}\n",
                c.agent, c.status, c.goal, c.report
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Synthesize the following subtask reports into a combined summary.\n\
         Highlight successes, flag any failures or gaps, and provide an overall assessment.\n\n\
         {reports_text}"
    )
}

/// A child task's report for synthesis.
#[derive(Clone, Debug)]
pub struct ChildReport {
    /// Agent name.
    pub agent: String,
    /// Task status string.
    pub status: String,
    /// Goal description.
    pub goal: String,
    /// Report text (WorkReport JSON or error message).
    pub report: String,
}

/// Collect child reports from the task store.
pub fn collect_child_reports(store: &TaskStore, parent_id: &str) -> Result<Vec<ChildReport>> {
    let children = store.children_of(parent_id)?;
    let mut reports = Vec::new();
    for child in children {
        let report_text = child
            .report
            .clone()
            .unwrap_or_else(|| "(no report)".to_string());
        reports.push(ChildReport {
            agent: child.agent.clone(),
            status: child.status.as_str().to_string(),
            goal: child.goal.clone(),
            report: report_text,
        });
    }
    Ok(reports)
}

/// Check whether a review child has already been enqueued for this parent.
pub fn has_review_child(store: &TaskStore, parent_id: &str) -> Result<bool> {
    let children = store.children_of(parent_id)?;
    Ok(children.iter().any(|c| c.agent == "reviewer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Completion, Model};
    use crate::task::{NewTask, TaskStore};
    use std::path::PathBuf;

    /// Scripted model for PM decomposition tests.
    struct ScriptedPm {
        replies: std::sync::Mutex<std::collections::VecDeque<String>>,
    }

    impl ScriptedPm {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: std::sync::Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for ScriptedPm {
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

    fn make_roster() -> Vec<AgentEntry> {
        vec![
            AgentEntry {
                name: "backend".to_string(),
                role: "backend engineer".to_string(),
            },
            AgentEntry {
                name: "frontend".to_string(),
                role: "frontend engineer".to_string(),
            },
            AgentEntry {
                name: "reviewer".to_string(),
                role: "code reviewer".to_string(),
            },
        ]
    }

    fn json_array_plain() -> String {
        serde_json::to_string(&serde_json::json!([
            {"agent": "backend", "goal": "implement API", "verify": ["cargo test"]},
            {"agent": "frontend", "goal": "build UI", "verify": ["npm test"]}
        ]))
        .unwrap()
    }

    fn json_wrapped_subtasks() -> String {
        serde_json::to_string(&serde_json::json!({
            "subtasks": [{"agent": "backend", "goal": "implement API", "verify": ["cargo test"]}]
        }))
        .unwrap()
    }

    fn json_single_backend() -> String {
        serde_json::to_string(&serde_json::json!([
            {"agent": "backend", "goal": "implement", "verify": ["cargo test"]}
        ]))
        .unwrap()
    }

    fn json_unknown_agent() -> String {
        serde_json::to_string(&serde_json::json!([
            {"agent": "nonexistent", "goal": "do stuff", "verify": []}
        ]))
        .unwrap()
    }

    fn json_verify_default() -> String {
        serde_json::to_string(&serde_json::json!([
            {"agent": "a", "goal": "g"}
        ]))
        .unwrap()
    }

    // ── Plain array JSON ────────────────────────────────────────────

    #[tokio::test]
    async fn decompose_plain_array() {
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[&json_array_plain()]));
        let specs = decompose(&model, "build the app", &make_roster())
            .await
            .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].agent, "backend");
        assert_eq!(specs[0].goal, "implement API");
        assert_eq!(specs[0].verify, vec!["cargo test".to_string()]);
        assert_eq!(specs[1].agent, "frontend");
    }

    // ── Wrapped {"subtasks": [...]} ─────────────────────────────────

    #[tokio::test]
    async fn decompose_wrapped_subtasks() {
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[&json_wrapped_subtasks()]));
        let specs = decompose(&model, "build the app", &make_roster())
            .await
            .unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, "backend");
    }

    // ── Prose-wrapped JSON ──────────────────────────────────────────

    #[tokio::test]
    async fn decompose_prose_wrapped() {
        let prose_wrapped = format!(
            "Here is the decomposition:\n{}\nLet me know if you need changes.",
            json_single_backend()
        );
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[&prose_wrapped]));
        let specs = decompose(&model, "build", &make_roster()).await.unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, "backend");
    }

    // ── Unknown agent → clear error ─────────────────────────────────

    #[tokio::test]
    async fn decompose_unknown_agent_error() {
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[&json_unknown_agent()]));
        let result = decompose(&model, "goal", &make_roster()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown agent"), "error: {err}");
        assert!(err.contains("nonexistent"), "error: {err}");
    }

    // ── Garbage → error after one retry ─────────────────────────────

    #[tokio::test]
    async fn decompose_garbage_error_after_retry() {
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[
            "I think we should break this into steps...",
            "Still no JSON here, just words.",
        ]));
        let result = decompose_with_retry(&model, "goal", &make_roster()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no valid JSON"), "error: {err}");
    }

    // ── Garbage → retry succeeds ────────────────────────────────────

    #[tokio::test]
    async fn decompose_retry_succeeds() {
        let model: Arc<dyn Model> = Arc::new(ScriptedPm::new(&[
            "Blah blah prose",      // First attempt: invalid
            &json_single_backend(), // Retry: valid
        ]));
        let specs = decompose_with_retry(&model, "build", &make_roster())
            .await
            .unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, "backend");
    }

    // ── Parse helpers ───────────────────────────────────────────────

    #[test]
    fn parse_subtasks_empty_text_errors() {
        let result = parse_subtasks("");
        assert!(result.is_err());
    }

    #[test]
    fn parse_subtasks_invalid_json_errors() {
        let result = parse_subtasks("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn parse_subtasks_plain_array() {
        let specs = parse_subtasks(&json_array_plain()).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].agent, "backend");
    }

    #[test]
    fn parse_subtasks_single_element_array() {
        let specs = parse_subtasks(
            &serde_json::to_string(&serde_json::json!([
                {"agent": "a", "goal": "g", "verify": ["v"]}
            ]))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].agent, "a");
    }

    #[test]
    fn parse_subtasks_wrapped_object() {
        let specs = parse_subtasks(&json_wrapped_subtasks()).unwrap();
        assert_eq!(specs.len(), 1);
    }

    #[test]
    fn parse_subtasks_verify_default_empty() {
        let specs = parse_subtasks(&json_verify_default()).unwrap();
        assert_eq!(specs[0].verify, Vec::<String>::new());
    }

    // ── Synthesis prompt ────────────────────────────────────────────

    #[test]
    fn synthesis_prompt_includes_all_reports() {
        let reports = vec![
            ChildReport {
                agent: "backend".to_string(),
                status: "Completed".to_string(),
                goal: "implement API".to_string(),
                report: "API done, tests pass".to_string(),
            },
            ChildReport {
                agent: "frontend".to_string(),
                status: "Completed".to_string(),
                goal: "build UI".to_string(),
                report: "UI done, a11y checked".to_string(),
            },
        ];
        let prompt = synthesis_prompt(&reports);
        assert!(prompt.contains("backend"));
        assert!(prompt.contains("frontend"));
        assert!(prompt.contains("API done"));
        assert!(prompt.contains("UI done"));
    }

    // ── has_reviewer ────────────────────────────────────────────────

    #[test]
    fn has_reviewer_in_roster() {
        assert!(has_reviewer(&make_roster()));
        let no_reviewer = vec![AgentEntry {
            name: "backend".to_string(),
            role: "backend engineer".to_string(),
        }];
        assert!(!has_reviewer(&no_reviewer));
    }

    // ── collect_child_reports ───────────────────────────────────────

    #[test]
    fn collect_child_reports_from_store() {
        let store = TaskStore::in_memory().unwrap();
        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec!["echo ok".to_string()],
                parent_id: None,
            })
            .unwrap();

        let c1 = store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "backend".to_string(),
                    goal: "implement API".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec!["cargo test".to_string()],
                    parent_id: None,
                },
            )
            .unwrap();
        store
            .complete(
                &c1.id,
                &serde_json::to_string(&serde_json::json!({"success":true,"answer":"API done"}))
                    .unwrap(),
            )
            .unwrap();

        let reports = collect_child_reports(&store, &parent.id).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].agent, "backend");
        assert_eq!(reports[0].status, "Completed");
        assert!(reports[0].report.contains("API done"));
    }

    // ── has_review_child ────────────────────────────────────────────

    #[test]
    fn has_review_child_detects_reviewer() {
        let store = TaskStore::in_memory().unwrap();
        let parent = store
            .enqueue(NewTask {
                agent: "pm".to_string(),
                goal: "build app".to_string(),
                workspace: PathBuf::from("/tmp"),
                verify: vec!["echo ok".to_string()],
                parent_id: None,
            })
            .unwrap();

        assert!(!has_review_child(&store, &parent.id).unwrap());

        store
            .enqueue_child(
                &parent.id,
                NewTask {
                    agent: "reviewer".to_string(),
                    goal: "review work".to_string(),
                    workspace: PathBuf::from("/tmp"),
                    verify: vec!["echo ok".to_string()],
                    parent_id: None,
                },
            )
            .unwrap();

        assert!(has_review_child(&store, &parent.id).unwrap());
    }
}
